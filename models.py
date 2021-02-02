import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
import utils

from fbprophet.plot import plot_plotly
from fbprophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')

def baseline(df, freq='H', periods=24*7, cv_flag=0):
    """
    The out of the box Prophet model as the baseline model
    
    Parameters
    ----------
    df : pandas.DataFrame
        in Prophet required format that has a minimum of two columns: ds and y
    freq : any valid frequency for pd.date_range, such as 'D' or 'M'
    periods : int number of periods to forecast forward
    cv_flag: 1 to calculate the cv results; 0 to ignore the cv results

    Returns:
    --------
    forecast : pandas.DataFrame
        the forecast produced by prophet model
    m : the fitted prophet model
    cv_results : cross validation results
    """
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(freq=freq, periods=periods)
    forecast = m.predict(future)

    # -----------validation------------------------------------------------------
    cv_results = utils.cv_results(m, cv_flag)
    # plot
    fig_changepoints = utils.plot_changepoints(m, forecast)
    fig_components = utils.plot_components(m, forecast)

    return m, forecast, cv_results

def with_seasonality_tuning(df, freq='H', periods=24*7, cv_flag=0):
    """
    Add the custom hyperparameters including custom seasonality, trend to the Prophet model
    
    Parameters
    ----------
    df : pandas.DataFrame
        in Prophet required format that has a minimum of two columns: ds and y
    freq : any valid frequency for pd.date_range, such as 'D' or 'M'
    periods : int number of periods to forecast forward
    cv_flag: 1 to calculate the cv results; 0 to ignore the cv results

    Returns:
    --------
    forecast : pandas.DataFrame
        the forecast produced by prophet model
    m : the fitted prophet model
    cv_results : cross validation results
    """
    m = Prophet(
        growth='linear', # default
        seasonality_mode='multiplicative', # default 'additive'
        changepoint_prior_scale=0.1, # default 0.05
        seasonality_prior_scale=10, # default 10
        holidays_prior_scale=10, # default 10
        daily_seasonality=False, # default 'auto'
        weekly_seasonality=True, # default 'auto'
        yearly_seasonality=False # default 'auto'
        )

    # ------------another example----------------
    # m = Prophet(mcmc_samples=300, holidays=None, holidays_prior_scale=0.25, changepoint_prior_scale=0.01, seasonality_mode='multiplicative', \
    #         yearly_seasonality=10, \
    #         weekly_seasonality=True, \
    #         daily_seasonality=False)
    # -------------------------------------------

    # m.add_seasonality(name='monthly',period=30.5, fourier_order=5)
    # m.add_seasonality(name='daily', period=1, fourier_order=5)
    # m.add_seasonality(name='weekly', period=7, fourier_order=10)
    m.add_seasonality(name='yearly', period=365.25, fourier_order=3, prior_scale=10, mode='additive')
    # m.add_seasonality(name='quarterly', period=365.25/4, fourier_order=5, prior_scale=15)
    
    # m.add_country_holidays(country_name='IN')

    m.fit(df)
    future = m.make_future_dataframe(freq=freq, periods=periods)
    forecast = m.predict(future)

    # ------------validation----------------------------------------------------
    cv_results = utils.cv_results(m, cv_flag)


    # plot
    fig_changepoints = utils.plot_changepoints(m, forecast)
    fig_components = utils.plot_components(m, forecast)

    return m, forecast, cv_results


def with_lockdown_regressors(df, freq='H', periods=24*7, cv_flag=0):
    """
    To add lockdown periods as extra regressors
    """

    lockdown_holiday_df = utils.add_lockdown_as_holidays_df()
    custom_holiday_df = utils.add_custom_holidays()

    m = Prophet(yearly_seasonality=False,
                # weekly_seasonality=False,
                # daily_seasonality=False,
                holidays=pd.concat([custom_holiday_df, lockdown_holiday_df]))
    m.add_seasonality(name='yearly', period=365.25, fourier_order=3, 
                  prior_scale=10, 
                  mode='additive')

    df_with_lockdown_regressors = utils.add_lockdown_as_regressors(df)

    m.add_regressor('during_lockdown', mode='additive')
    m.add_regressor('during_regional_lockdown', mode='additive')
    m.add_regressor('during_lockdown_with_relaxation', mode='additive')
    m.add_regressor('after_lockdown', mode='additive')

    m.fit(df_with_lockdown_regressors)
    future = m.make_future_dataframe(freq=freq, periods=periods)

    future_with_lockdown_regressors = utils.add_lockdown_as_regressors(future)

    forecast = m.predict(future_with_lockdown_regressors)

    # ------------validation----------------------------------------------------
    cv_results = utils.cv_results(m, cv_flag)
    # cv_avg_mape = np.mean(performance_metrics(cv_results).mape) * 100
    # cv_avg_mae = np.mean(performance_metrics(cv_results).mae)
    # print('fbprophet cross validation average mean absolute percentage error (MAPE) is {}'.format(cv_avg_mape), '/n',
    # 'fbprophet cross validation average mean absolute error (MAE) is {}'.format(cv_avg_mae))

    # plot
    fig_changepoints = utils.plot_changepoints(m, forecast)
    fig_components = utils.plot_components(m, forecast)

    return m, forecast, cv_results


def with_weather_lockdown_regressors(df_train, temp, divider, freq='H', periods=24*7, cv_flag=0):
    """
    Add the extra regressors to the Prophet model, including 
    weahter information, custom holidays, and lockdown regressors

    Parameters
    ----------
    df_train : training set
    df_test : test set
    """
    # https://towardsdatascience.com/forecasting-in-python-with-facebook-prophet-29810eb57e66
    # https://nbviewer.jupyter.org/github/nicolasfauchereau/Auckland_Cycling/blob/master/notebooks/Auckland_cycling_and_weather.ipynb#Instantiate,-then-fit-the-model-to-the-training-data

    lockdown_holiday_df = utils.add_lockdown_as_holidays_df()

    custom_holiday_df = utils.add_custom_holidays()

    m = Prophet(growth='linear',
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                holidays=pd.concat([custom_holiday_df, lockdown_holiday_df]),
                seasonality_mode='multiplicative',
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                changepoint_prior_scale=.05,
                mcmc_samples=0
                )
    # to reduce variability of yearly seasonality          
    m.add_seasonality(name='yearly', period=365.25, fourier_order=3, prior_scale=10, mode='additive') 

    # to add lockdown regressor
    df_with_lockdown_regressors = utils.add_lockdown_as_regressors(df_train)

    m.add_regressor('during_lockdown', prior_scale=10, mode='additive')
    m.add_regressor('during_regional_lockdown', prior_scale=10, mode='additive')
    m.add_regressor('during_lockdown_with_relaxation', prior_scale=10, mode='additive')
    m.add_regressor('after_lockdown', prior_scale=10, mode='additive')

    # add weather regressor
    df_with_weather_regressors = utils.add_regressor(df_train, temp.loc[temp['ds'] < divider, :], varname='temp')
    m.add_regressor('temp', prior_scale=0.5, mode='multiplicative')

    # combine regressors
    df_with_regressors = utils.add_regressor(df_with_lockdown_regressors, df_with_weather_regressors, varname='temp')

    m.fit(df_with_regressors) 

    future = m.make_future_dataframe(freq=freq, periods=periods)

    # add regressor to future 
    future_with_lockdown_regressors = utils.add_lockdown_as_regressors(future)

    future_with_regressors = utils.add_regressor(future_with_lockdown_regressors, temp, varname='temp')

    forecast = m.predict(future_with_regressors)

    # ------------validation----------------------------------------------------
    cv_results = utils.cv_results(m, cv_flag)
    # plot
    fig_changepoints = utils.plot_changepoints(m, forecast)
    fig_components = utils.plot_components(m, forecast)

    # mape = utils.cv_mape(cv_results.y, cv_results.yhat)
    # print('Model - mean absolute percentage error (MAPE): {}'.format(mape))

    # verification = utils.df_verification(forecast, df_train, df_test)
    # fig = utils.plot_verification(verification, divider)

    # # sklearn metric mae
    # mae = utils.sklearn_mae(verification)
    # print('Model - mean absolute error (MAE): {}'.format(mae))

    return m, forecast, cv_results



 

