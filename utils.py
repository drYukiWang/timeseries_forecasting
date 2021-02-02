import pandas as pd
import numpy as np
from scipy.stats import skew
from pathlib import Path
import pickle
import time
from datetime import datetime as dt
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler
from sklearn.metrics import mean_absolute_error

from fbprophet.make_holidays import make_holidays_df
from fbprophet.diagnostics import cross_validation, performance_metrics
import holidays

import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot_plotly

divider = '2020-09-01'

def df_cleaning(df, attr='VALUE'):
    """
    clean dataframe
    
    Parameters
    ----------
    df: pandas.DataFrame
    attr: attribute value of interest (the column)

    Returns
    -----------
    df: datetime indexed dataframe
    """

    df[df[attr] < 0] = 0
    df[attr].replace(0, np.nan, inplace = True)
    df[attr].fillna(df[attr].median(), inplace = True)
    df['datetime'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    df = df[[attr,'datetime']].set_index('datetime')
    df = df.sort_index()
    df = df[df.index.year >= 2018]
    return df


def df_resampling(df, predictor='y'):
    """
    resample data to 30 mins, 60 mins and daily

    Parameters 
    -----------
    df : pandas.DataFrame
        The cleaned fbprophet dataframe in the original interval 15 mins, with index and 
        columns: `ds` and `y`
    predictor: column to be resampled

    Returns
    -------
    df_hh : resampled at half hourly interval
    df_h : resampled at hourly interval 
    df_d : resampled at daily interval
    """
        
    # step 1: use datetimeIndex as index for resampling, etc.
    index_name = df.columns[0]
    if not isinstance(df.index, pd.DatetimeIndex):
        df[index_name] = df[index_name].apply(pd.to_datetime, errors='coerce')
        df = df.set_index(index_name).sort_index()

    hh = df[predictor].copy().resample('30T').mean() # half hourly
    h = df[predictor].copy().resample('H').mean() # hourly
    d = df[predictor].copy().resample('D').mean() # daily

    def fill_missing_values(df):
        if df.isnull().values.any():
            df.fillna(df.median(), inplace=True)
        return df

    hh = fill_missing_values(hh)
    h = fill_missing_values(h)
    d = fill_missing_values(d)

    # step 2: reset index to numerical and make datetimeIndex as a column
    df_hh = hh.reset_index()
    df_h = h.reset_index()
    df_d = d.reset_index()
    
    return df_hh, df_h, df_d


def df_train_test_split(df, divider):
    """
    split data into train and split

    Parameters
    ----------
    df : pandas.DataFrame 
        The cleaned fbprophet dataframe in either 15 mins, half hourly, hourly or daily interval, with index and 
        columns: 'ds' and 'y'
    divider: datetime index 
        The datetime separating the training set and the test set

    Returns
    -------
    df_train : pandas.DataFrame
        The training set, formatted for fbprophet.
    df_test :  pandas.Dataframe
        The test set, formatted for fbprophet.
    """

    indx = df[df['ds'] == divider].index.tolist()
    df_train = df.loc[:indx[0]-1, :]
    df_test = df.loc[indx[0]:, :]

    return df_train, df_test    


def df_combine(df, forecast):
    """
    To combine original data with the forecast into one dataframe

    Parameters
    ----------
    df : original dataframe
    """
    combine_df = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(df.set_index('ds'))
    combine_df = combine_df.reset_index()

    return combine_df

def df_verification(forecast, df_train, df_test):
    """
    Put together the forecast (coming from fbprophet) 
    and the observed data, and set the index to be a proper datetime index, 
    for plotting

    Parameters
    ----------
    forecast : pandas.DataFrame 
        The pandas.DataFrame coming from the `forecast` method of a fbprophet 
        model. 
    
    df_train : pandas.DataFrame
        The training set, pandas.DataFrame

    df_test : pandas.DataFrame
        The training set, pandas.DataFrame
    
    Returns
    -------
    forecast : 
        The forecast DataFrane including the original observed data as `y`.
    """

    forecastc = forecast.copy()
    df_trainc = df_train.copy()
    df_testc = df_test.copy()

    forecastc.index = pd.to_datetime(forecastc.ds)
    
    df_trainc.index = pd.to_datetime(df_trainc.ds)
    
    df_testc.index = pd.to_datetime(df_testc.ds)
    
    data = pd.concat([df_trainc, df_testc], axis=0)
    
    forecastc.loc[:,'y'] = data.loc[:,'y']

    return forecastc


def add_regressor(data, regressor, varname=None):
    """
    adds a regressor to a `pandas.DataFrame` of target values 
    for use in fbprophet 

    Parameters
    ----------
    data : pandas.DataFrame 
        The pandas.DataFrame in the fbprophet format, train set;
        or `future` dataframe
    regressor : pandas.DataFrame 
        A pandas.DataFrame containing the extra-regressor with the same index as the train set, 
        eg. ` temp.loc[temp['ds'] < divider, :]`;
        
        or for `future` dataframe this is within the same range and frequency,
        eg. `temp`
    varname : string 
        The name of the column in the `regressor` DataFrame to add to the `data` DataFrame

    Returns
    -------
    data_with_regressors : pandas.DataFrame
        The original `data` DataFrame with the column containing the 
        extra regressor `varname`

    """

    data_with_regressors = data.copy()
    
    data_with_regressors.loc[:,varname] = regressor.loc[:,varname]

    return data_with_regressors


def add_lockdown_as_holidays_df():
    """
    To add lockdown periods as custom holidays dataframe
    """
    lockdown = pd.DataFrame({'holiday': 'lockdown',
                            'ds': pd.to_datetime(['2020-03-25']),
                            'lower_window': 0,
                            'upper_window': 40}) # till 4 May

    lockdown_with_relaxation = pd.DataFrame({'holiday': 'lockdown with relaxation',
                            'ds': pd.to_datetime(['2020-05-04']),
                            'lower_window': 0,
                            'upper_window': 46}) # till 19 June
    
    lockdown_reginal = pd.DataFrame({'holiday': 'lockdown reginal',
                            'ds': pd.to_datetime(['2020-06-19']),
                            'lower_window': 0,
                            'upper_window': 12}) # till 30 June

    lockdown_with_relaxation_2 = pd.DataFrame({'holiday': 'lockdown with relaxation 2',
                            'ds': pd.to_datetime(['2020-07-01']),
                            'lower_window': 0,
                            'upper_window': 62}) # till 31 August

    lockdown_holiday_df = pd.concat([lockdown, lockdown_with_relaxation, lockdown_reginal, lockdown_with_relaxation_2])

    return lockdown_holiday_df


def add_lockdown_as_regressors(df):
    """
    To add during and after lockdown periods as regressors

    Parameters
    ----------
    df : pandas.DataFrame
        A Prophet input DataFrame to make prediction or
        A `future` DataFrame created by the fbprophet `make_future` method  
        
    Returns
    -------
    df : pandas.DataFrame
        The Prophet input DataFrame with the regressors added or
        The `future` DataFrame with the regressors added
    """
    dfc = df.copy()

    dfc['during_lockdown'] = 0
    dfc.loc[(dfc['ds'] >= pd.to_datetime('2020-03-25')) & (dfc['ds'] <= pd.to_datetime('2020-05-03')), 'during_lockdown'] = 1

    dfc['during_regional_lockdown'] = 0
    dfc.loc[(dfc['ds'] >= pd.to_datetime('2020-06-19')) & (dfc['ds'] <= pd.to_datetime('2020-06-30')), 'during_regional_lockdown'] = 1
    
    dfc['during_lockdown_with_relaxation'] = 0
    dfc.loc[(dfc['ds'] >= pd.to_datetime('2020-05-04')) & (dfc['ds'] <= pd.to_datetime('2020-06-18')), 'during_lockdown_with_relaxation'] = 1
    dfc.loc[(dfc['ds'] >= pd.to_datetime('2020-07-01')) & (dfc['ds'] <= pd.to_datetime('2020-08-31')), 'during_lockdown_with_relaxation'] = 1

    dfc['after_lockdown'] = 0
    dfc.loc[(dfc['ds'] >= pd.to_datetime('2020-09-01')), 'after_lockdown'] = 1

    lockdown_regressors = dfc.copy()


    return lockdown_regressors


def add_custom_holidays(year_list = [2018, 2019, 2020, 2021]):
    """
    To add custom holidays in India for 2018, 2019, 2020, 2021
    """
    # https://github.com/facebook/prophet/issues/909#issuecomment-479262114
    # https://www.calendarlabs.com/holidays/india/2018
    # https://github.com/dr-prodigy/python-holidays/blob/a5eb566cab72c360e7413391a67997ab87cfccd6/holidays/countries/india.py#L76

    hol_df =make_holidays_df(year_list=year_list, country='IN', province='TN')
    
    add_hol_dict = holidays.India(years=year_list, prov='TN')
    ldates, lnames = [], []
    for date, name in sorted(add_hol_dict.items()):
        ldates.append(date)
        lnames.append(name)
    ldates = np.array(ldates)
    lnames = np.array(lnames)
    add_hol_df = pd.DataFrame({'ds': ldates, 'holiday': lnames})

    # convert datetime format then combine two dataframes by time
    df1 = hol_df.set_index('ds').sort_index()
    df2 = add_hol_df.set_index('ds').sort_index()
    df1.index = pd.to_datetime(df1.index, format='%Y-%m-%d').strftime('%Y-%m-%d')
    df2.index = pd.to_datetime(df2.index, format='%Y-%m-%d').strftime('%Y-%m-%d')
    df1 = df1.reset_index()
    df2 = df2.reset_index()

    custom_holiday_df = pd.concat([df1, df2]).drop_duplicates(subset='ds', ignore_index=True)

    return custom_holiday_df


def grid_search_hyperparameters(df):
    """
    grid search the optimal hyperparameters

    """
    # https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation
    pass


def cv_results(model, cv_flag, initial='365 days', period='90 days', horizon='180 days'):
    """
    To calculate fbprophet cross_validation results

    Parameters:
    -----------
    model : the fitted fbprophet model
    cv_flag : 1 to calculate the cv results; 0 to ignore the cv results

    Returns:
    --------
    cv_results: cross validation results
    """

    if cv_flag:
        cv_results = cross_validation(model,
                            initial=initial,
                            period=period,
                            horizon=horizon,
                            parallel='processes')
    # remove where y=0
        cv_results = cv_results[cv_results.y != 0]
    else: 
        cv_results =  None

    return cv_results


def cv_mape(y_true, y_pred):
    """
    The Mean Absolute Percentage Error based on the cross validation results

    Parameters
    -----------
    y_true : from cv_results.y
    y_pred : from cv_results.yhat

    Returns
    --------
    mape: average mape
    """ 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cv_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return cv_mape

def sklearn_mae(verification):
    """
    The Mean Absolute Error based on sklearn metrics using `verification` dataframe
    """
    y_true = verification.y
    y_pred = verification.yhat
    mae = mean_absolute_error(np.array(y_true), np.array(y_pred))
    
    return mae

# ----------------Plotting Utilities-------------------------------------
def plot_changepoints(m, forecast):
    """
    To plot with changepoints detected
    """

    fig = m.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), m, forecast)
    plt.show()

    return fig

def plot_components(m, forecast):
    """
    To plot the forecasting components from fbprophet
    """
    fig = m.plot_components(forecast)
    plt.show()

    return fig

def plot_verification(verification, divider):
    """
    plots the forecasts and observed data, the `divider` argument is used to visualise 
    the division between the training and test sets. 

    Parameters
    ----------
    verification : pandas.DataFrame
        The `df_verification` DataFrame coming from the `df_verfication` function in this package

    divider : string
        The date used to separate the training and test set. Default 2020-05-01

    Returns
    -------
    fig : matplotlib Figure object

    """

    fig, ax = plt.subplots(figsize=(14, 8))

    train = verification[verification.index < divider]
    ax.plot(train.index, train.y, 'ko', markersize=3)
    ax.plot(train.index, train.yhat, color='steelblue', lw=0.5)
    ax.fill_between(train.index, train.yhat_lower, train.yhat_upper, color='steelblue', alpha=0.3)
    
    test = verification[verification.index >= divider]
    ax.plot(test.index, test.y, 'ro', markersize=3)
    ax.plot(test.index, test.yhat, color='coral', lw=0.5)
    ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, color='coral', alpha=0.3)
    
    ax.axvline(pd.to_datetime(divider), color='0.8', lw=3, alpha=0.7)
    # ax.grid(ls=':', lw=0.5)
    
    return fig

def plot_weekdays_weekends(data):
    """
    Plot the `hourly` or other intervals weekdays versus weekends
    """
    pass

def plot_joint_plot(data, x='yhat', y='y', title=None):
    """
    Scatter plot to show marginal distribution and correlation between observations and 
    modelled/predicted values

    Parameters
    ----------
    data : `train = verification[verification.index < divider]`; 
            `test = verification[verification.index >= divider]`

    Returns
    --------
    g : matplotlib Figure object
    """
    
    g = sns.jointplot(x='yhat', y='y', data = data, kind="reg", color="0.4")
    g.fig.set_figwidth(8)
    g.fig.set_figheight(8)

    ax = g.fig.axes[1]
    if title is not None: 
        ax.set_title(title, fontsize=16)
    ax = g.fig.axes[0]
    ax.set_xlim([7000, 15000])
    ax.set_ylim([7000, 18000])
    ax.text(7500, 16000, "R = {:+4.2f}\nMAE = {:4.1f}".format(data.loc[:,['y','yhat']].corr().iloc[0,1], mean_absolute_error(data.loc[:,'y'].values, data.loc[:,'yhat'].values)), fontsize=16)

    ax.set_xlabel("model's estimates", fontsize=15)
    ax.set_ylabel("observations", fontsize=15)
    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()];

    return g 

def plot_residuals_dist(data, title=None):
    """
    Plot residual distribution (train set) or (test set)

    Parameters
    ----------
    data : `train = verification[verification.index < divider]`; 
            `test = verification[verification.index >= divider]`

    Returns
    -------
    fig :  matplotlib Figure object
    """

    fig, ax = plt.subplots(figsize=(8,8))
    sns.distplot((data.yhat - data.y), ax=ax, color='0.4')

    ax.set_xlabel('residuals', fontsize=15)
    ax.set_ylabel("normalised frequency", fontsize=15)

    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()];

    ax.text(0.05, 0.9, "Skewness = {:+4.2f}\nMedian = {:+4.2f}".\
        format(skew(data.yhat - data.y), (data.yhat - data.y).median()), \
        fontsize=14, transform=ax.transAxes)

    ax.axvline(0, color='0.4')

    if title is not None:
        ax.set_title('Residuals distribution ({})'.format(title), fontsize=17)

    return fig

    def plot_forecast_blocks(verification, start_date, end_date, ax=None):
        pass

