# Copyright 2021 The ProLoaF Authors. All Rights Reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# ==============================================================================

import os
import plf_util.eval_metrics as metrics
import torch
import copy
import numpy as np
import pandas as pd
import csv
import statsmodels.tsa.statespace.sarimax as sarimax
import pickle
import matplotlib.pyplot as plt
import multiprocessing

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from arch import arch_model
from joblib import Parallel, delayed
from pmdarima.arima import auto_arima
# ======================================
# =======================================
# save SARIMAX model
# =============================================================================

def save_SARIMAX(path, fitted, predictions, save_predictions = True):
    print('Saving fitted sarimax model and predictions')
    if 'ResultsWrapper' in str(type(fitted)):
        fitted.save(path+"\\sarimax.pkl")
    else:
        with open(path+"\\sarimax.pkl", 'wb') as pkl:
            pickle.dump(fitted, pkl)
    if save_predictions:
        with open('sarimax_predictions.csv', 'w') as prediction_logger:
            wr = csv.writer(prediction_logger, quoting=csv.QUOTE_ALL)
            wr.writerow(predictions)
    print('Plot fitted (S)ARIMA(X) model diagnostics')
    fitted.plot_diagnostics(figsize=(15, 12))
    plt.show()
    return

# =============================================================================
# load SARIMAX model
# =============================================================================

def load_SARIMAX(path):
    full_path_to_file = path + "\\sarimax.pkl"
    if os.path.isfile(full_path_to_file):
        with open(full_path_to_file, "rb") as f:
            fitted = pickle.load(f)
        return fitted
    else:
        return None
# =============================================================================
# train SARIMAX model
# =============================================================================

def train_SARIMAX(endog, exog, order, seasonal_order=None, number_set=0, trend='n'):

    """
    Trains a SARIMAX model

    Parameters
    ----------
    endog                   : train values of target variable of type pd.df or np.array

    exog                    : values of exogenous features of type pd.df or np.array

    order                   : order of model

    seasonal_order          : seasonal order of model

    trend                   : trend method used for training, here we always use trend = n as default

    Returns
    -------
    1)fitted                : trained model instance
    2)model                 : model instance

    """
    model = sarimax.SARIMAX(endog, exog,
                            order=order, seasonal_order=seasonal_order, trend=trend,
                            time_varying_regression = True, mle_regression=False)
    fitted = model.fit(disp=-1)
    # use AICc - Take corrected Akaike Information Criterion here as default for performance check
    return fitted, model, fitted.aicc

def auto_sarimax_wrapper(endog, exog=None, order=None, seasonal_order=None, trend='n', grid_search = False):
    if grid_search:
        # Apply parameter tuning script for (seasonal) ARIMA using [auto-arima]
        # (https://github.com/alkaline-ml/pmdarima)
        print('Training SARIMA(X) with parameter grid search by auto-arima...')
        auto_model = auto_arima(endog, exog,
                                start_p=1, start_q=1, max_p=3, max_q=3, m=24, start_P=0,
                                seasonal=True, d=1, D=1, trend=trend, stepwise=True, trace=True,
                                suppress_warnings=True, error_action='ignore')
        # TODO: Not sure if m is set correctly per default. Check if there are better options
        auto_model.summary()
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
    sarimax_model, untrained_model, val_loss = train_SARIMAX(endog, exog, order, seasonal_order, trend=trend)
    return sarimax_model, untrained_model, val_loss
# =============================================================================
# evaluate forecasts
# =============================================================================

def eval_forecast(forecasts, output_matrix, target, upper_limits, lower_limits, total=True):

    """
    Calculates evaluation metrics

    Parameters
    ----------
    forecasts               : List of calculated forecasts

    output_matrix           : list of DataFrames for output

    target                  : str target column

    upper_limits            : List of upper interval for given forecasts

    lower_limits            : List of lower interval for given forecasts

    total                   : true if metrics shall be calculated over horizon


    Returns
    -------
    1)mse_horizon
    2)rmse_horizon
    4)sharpness_horizon
    5)coverage_horizon
    6)mis_horizon

    """
    forecasts = torch.tensor(forecasts)
    true_values = torch.tensor([i[target].T.values for i in output_matrix]).reshape(forecasts.shape).type(torch.FloatTensor)
    upper_limits = torch.tensor([i.values for i in upper_limits]).reshape(forecasts.shape).type(torch.FloatTensor)
    lower_limits = torch.tensor([i.values for i in lower_limits]).reshape(forecasts.shape).type(torch.FloatTensor)

    if total:
        mse = metrics.mse(true_values, [forecasts])
        rmse = metrics.rmse(true_values, [forecasts])
        mase=metrics.mase(true_values, [forecasts], 24, true_values) #MASE always needs a reference vale, here we assume a 24 timestep seasonality.
        rae= metrics.rae(true_values, [forecasts])
        mae=metrics.nmae(true_values, [forecasts])
        sharpness = metrics.sharpness(None,[upper_limits, lower_limits])
        coverage = metrics.picp(true_values, [upper_limits, lower_limits])
        mis = metrics.mis(true_values, [upper_limits, lower_limits], alpha=0.05)
        return mse, rmse, mase, rae, mae, sharpness, coverage, mis
    else:
        rmse_horizon = metrics.rmse(true_values, [forecasts], total)
        sharpness_horizon = metrics.sharpness(None,[upper_limits, lower_limits], total)
        coverage_horizon = metrics.picp(true_values, [upper_limits, lower_limits], total)
        mis_horizon = metrics.mis(true_values, [upper_limits, lower_limits], alpha=0.05, total=total)
        return rmse_horizon, sharpness_horizon, coverage_horizon, mis_horizon

# =============================================================================
# Create multiple forecasts with one model
# =============================================================================

def make_forecasts(endog_train, endog_val, exog, exog_forecast, fitted, forecast_horizon=1,
                   limit_steps=False, pi_alpha=1.96, update = False):

    """
    Calculates a number of forecasts

    Parameters
    ----------
    endog_train             : the new values of the target variable

    output_matrix           : the observed values of the target variable

    exog                    : the new exogenous features for new training

    exog_forecast           : the exogenous variables to match the prediction horizon

    fitted                  : trained model

    forecast_horizon        : number of future forecast time_steps, default = 1

    limit_steps             : limits the number of simulation/predictions into the future,
                              if None, steps is equal to length of validation set

    pi_alpha                : measure to adjust confidence interval, default is set to 1.96, which is a 95% PI

    update                  : bool value, if True the new observations are used to fit the model again
                                default = False
    Returns
    -------
    1)forecasts             : calculated forecasts

    """

    num_cores = multiprocessing.cpu_count()
    test_period = range(endog_val.shape[0])
    if limit_steps:
        test_period = range(limit_steps)

    def sarimax_predict(i, fitted=None):
        endog_train_i = pd.concat([endog_train, endog_val.iloc[0:i]])
        exog_train_i = pd.concat([exog, exog_forecast.iloc[0:i]])
        if i + forecast_horizon <= endog_val.shape[0]:
            if 'ResultsWrapper' in str(type(fitted)):
                if update:
                    # if we do standard parameters we have a SARIMAX object which can only be extended with apply (endog, exog)
                    fitted = fitted.apply(endog=endog_train_i, exog=exog_train_i)
                forecast = fitted.forecast(forecast_horizon, exog=exog_forecast[i:i+forecast_horizon])
            else:
                if update:
                    # when using hyperparam optimization we do have an arima object, then we need to use Update (y, X)
                    # Of course, you’ll have to re-persist your ARIMA model after updating it!
                    fitted = fitted.update(y=endog_train_i, X=exog_train_i)
                forecast = fitted.predict(n_periods=forecast_horizon, X=exog_forecast[i:i+forecast_horizon],
                                          return_conf_int = False)
            # standard way to fetch confidence interval
            fc_u, fc_l = naive_predictioninterval(forecast, endog_val[i:i+forecast_horizon], timestep=i,
                                                forecast_horizon=forecast_horizon, alpha=pi_alpha)
            # fetch conf_int by forecasting the upper and lower limit, given the fitted_models uncertainty params
            # (higher computation effort than standard way)
            # upper_limits, lower_limits = baselines.apply_PI_params(untrained_model, model, val_in, val_observed, target,
            #                                                       exog,  PAR['forecast_horizon'])
        return forecast, fc_u, fc_l

    expected_value, fc_u, fc_l = \
        zip(*Parallel(n_jobs=min(num_cores,len(test_period)))(delayed(sarimax_predict)(i, fitted) for i in test_period))
    return np.asarray(expected_value), np.asarray(fc_u), np.asarray(fc_l)

# =============================================================================
# predictioninterval functions
# =============================================================================

def naive_predictioninterval(forecasts, endog_val, timestep, forecast_horizon, alpha = 1.96):

    """
    Calculates predictioninterval for given forecasts with Naive method https://otexts.com/fpp2/prediction-intervals.html

    Parameters
    ----------
    forecasts               : List of forecasts

    endog_val               : List of True Values for given forecasts

    alpha                   : Confidence: default 1.96 equals to 95% confidence

    forecast_horizon        : number of future forecast time_steps

    Returns
    -------
    1)fc_u                  : upper interval for given forecasts with length:= forecast_horizon
    2)fc_l                  : lower interval for given forecasts with length:= forecast_horizon

    """
    sigma = np.std(endog_val[timestep]-forecasts[timestep])
    sigma_h = []

    for r in range(forecast_horizon):
        sigma_h.append(sigma * np.sqrt(r+1))

    sigma_hn = pd.Series(sigma_h)
    sigma_hn.index = endog_val[timestep].index

    fc_u = forecasts[timestep] + alpha * sigma_hn
    fc_l = forecasts[timestep] - alpha * sigma_hn

    return fc_u, fc_l

def GARCH_predictioninterval(forecasts, output_matrix, target, forecast_horizon, alpha = 1.96):

    """
    Calculates predictioninterval for given forecasts with GARCH method https://github.com/bashtage/arch

    Parameters
    ----------
    forecasts               : List of forecasts

    output_matrix           : List of True Values for given forecasts

    target                  : str target column

    alpha                   : Confidence: default 1.96 equals to 95% confidence

    forecast_horizon        : number of future forecast time_steps

    Returns
    -------
    1)upper_limits          : List of upper intervall for given forecasts
    2)lower_limits          : List of lower intervall for given forecasts

    """

    upper_limits = []
    lower_limits = []
    model_garch = arch_model(output_matrix[1:], mean=forecasts[1:], vol = 'GARCH', p=1, q=1)
    res = model_garch.fit(update_freq=5)
    sim_mod = arch_model(None, p=1, o=1, q=1, dist="skewt")
    sim_data = sim_mod.simulate(res.params, 1000)
    for i in range(len(forecasts)):
        sigma = np.std(output_matrix[i][target]-forecasts[i])
        sigma_h = []

        for r in range(forecast_horizon):
            sigma_h.append(sigma * np.sqrt(r+1))

        sigma_hn = pd.Series(sigma_h)
        sigma_hn.index = output_matrix[i].index

        fc_u = forecasts[i] + alpha * sigma_hn
        fc_l = forecasts[i] - alpha * sigma_hn

        upper_limits.append(fc_u)
        lower_limits.append(fc_l)

    return upper_limits, lower_limits

def apply_PI_params(model, fitted, input_matrix, output_matrix, target, exog, forecast_horizon):

    """
    Calculates custom predictioninterval --> Confidence intervals are used as model parameter. intervals added to forecast

    Parameters
    ----------
    model                   : model instance

    fitted                 : fitted model instance

    input_matrix           : list of DataFrames for input

    output_matrix          : list of DataFrames for output

    target                  : str target column

    exog                    : str features

    forecast_horizon        : number of future forecast time_steps

    Returns
    -------
    1)upper_limits          : List of upper intervall for given forecasts
    2)lower_limits          : List of lower intervall for given forecasts

    """

    lower_limits = []
    upper_limits = []

    params_conf = fitted.conf_int()

    lower_params = params_conf[0]
    upper_params = params_conf[1]

    m_l = copy.deepcopy(fitted)
    m_u = copy.deepcopy(fitted)

    m_l.initialize(model=model, params=lower_params)
    m_u.initialize(model=model, params=upper_params)

    for i in range(len(input_matrix)):
        m_l = m_l.apply(input_matrix[i][target], exog = input_matrix[i][exog])
        fc_l = m_l.forecast(forecast_horizon, exog= output_matrix[i][exog])
        lower_limits.append(fc_l)

        m_u = m_u.apply(input_matrix[i][target], exog = input_matrix[i][exog])
        fc_u = m_u.forecast(forecast_horizon, exog= output_matrix[i][exog])
        upper_limits.append(fc_u)

    return upper_limits, lower_limits

def simple_naive(series, horizon):
    return np.array([series[-1]] * horizon)

def persist_forecast(x_train, x_test, y_train, forecast_horizon, decomposed=False, alpha = 1.96):
    '''
    fits persistence forecast over the train data,
    calcuates the std deviation of residuals for each horizon over the horizon and
    finally outputs persistence forecast over the test data and std_deviation for each hour


    :param x_train,x_test: input train and test datasets
    :param y_train: target train dataset
    :param forecast_horizon: no. of future steps to be forecasted
    :param alpha: Confidence: default 1.96 equals to 95% confidence

    returns
    :expected_value : expected forecast values for each test sample over the forecast horizon
                        (shape:len(y_train),forecast_horizon)
    :upper_limits   : upper interval for given forecasts (shape: (1,forecast_horizon))
    :lower_limits   : lower interval for given forecasts (shape: (1,forecast_horizon))

    '''

    point_forecast_train = np.zeros(shape=(len(x_train), forecast_horizon))

    # difference between the observations and the corresponding fitted values
    residuals = np.zeros(shape=(len(x_train), forecast_horizon))

    if decomposed:
        for j in range(forecast_horizon):
            x_train[:, j] = seasonal_decomposed(np.reshape(x_train[:, j], (len(x_train), 1)))
            x_test[:, j] = seasonal_decomposed(np.reshape(x_test[:, j], (len(x_test), 1)))

    for i in range(len(x_train)):
        point_forecast_train[i, :] = simple_naive(x_train[i, :], forecast_horizon)
        residuals[i, :] = y_train[i, :] - point_forecast_train[i, :]

    # std deviation of residual for each hour(column wise)
    residual_std = np.std(residuals, axis=0)
    std_dev = np.zeros(shape=(1, forecast_horizon))

    for step in range(forecast_horizon):
        # std_dev[0,step] = residual_std[step]*np.sqrt(step+1)
        std_dev[0, step] = residual_std[step]

    expected_value = np.zeros(shape=(len(x_test), forecast_horizon))
    for i in range(len(x_test)):
        expected_value[i, :] = simple_naive(x_test[i, :], forecast_horizon)

    fc_u = expected_value + alpha * std_dev
    fc_l = expected_value - alpha * std_dev

    return expected_value, fc_u, fc_l

def seasonal_naive(series, forecast_horizon, periodicity):
    forecast = np.empty([forecast_horizon])
    for i in range(len(forecast)):
        if i + 1 > periodicity:
            forecast[i] = series[i - 2 * periodicity]
        else:
            forecast[i] = series[i - periodicity]
    return forecast

def seasonal_decomposed(series, periodicity=24):
    # seasonally adjusted time series Y(t)-S(t).
    stl = STL(series.squeeze(), period=periodicity)
    res = stl.fit()
    #plt = res.plot()
    #plt.show()
    return series.squeeze()-res.seasonal

def seasonal_forecast(x_train, x_test, y_train, forecast_horizon, seasonality=24, decomposed=False, alpha = 1.96):
    '''
    fits periodic forecast over the train data,
    calcuates the std deviation of residuals for each horizon over the horizon and
    finally outputs persistence forecast over the test data and std_deviation for each hour


    :param x_train, x_test: input train and test datasets
    :param y_train: target train dataset
    :param forecast_horizon: no. of future steps to be forecasted
    :param alpha: Confidence: default 1.96 equals to 95% confidence

    returns
    :expected_value : expected forecast values for each test sample over the forecast horizon
                        (shape:len(y_train),forecast_horizon)
    :upper_limits   : upper interval for given forecasts (shape: (1,forecast_horizon))
    :lower_limits   : lower interval for given forecasts (shape: (1,forecast_horizon))

    '''

    naive_forecast = np.zeros(shape=(len(x_train), forecast_horizon))
    # difference between the observations and the corresponding fitted values
    residuals = np.zeros(shape=(len(x_train), forecast_horizon))

    if decomposed:
        for j in range(forecast_horizon):
            x_train[:, j] = seasonal_decomposed(np.reshape(x_train[:, j], (len(x_train), 1)))
            x_test[:, j] = seasonal_decomposed(np.reshape(x_test[:, j], (len(x_test), 1)))

    for i in range(len(x_train)):
        naive_forecast[i, :] = seasonal_naive(x_train[i, :], forecast_horizon, periodicity=seasonality)
        residuals[i, :] = y_train[i, :] - naive_forecast[i, :]

    # std deviation of residual for each hour(column wise)
    residual_std = np.std(residuals, axis=0)
    std_dev = np.zeros(shape=(1, forecast_horizon))

    for step in range(forecast_horizon):
        # forecast_std[0,step] = residual_std[step]*np.sqrt(step+1)
        std_dev[0, step] = residual_std[step]

    expected_value = np.zeros(shape=(len(x_test), forecast_horizon))
    for i in range(len(x_test)):
        expected_value[i, :] = seasonal_naive(x_test[i, :], forecast_horizon, periodicity=seasonality)

    fc_u = expected_value + alpha * std_dev
    fc_l = expected_value - alpha * std_dev

    return expected_value, fc_u, fc_l

def exp_smoothing(y_train, y_test, forecast_horizon=1, limit_steps=False, pi_alpha=1.96, update = True):
    """
    Trains a SARIMAX model

    Parameters
    ----------
    y_train                 : train values of target variable of type pd.df or np.array

    y_test                  : values of exogenous features of type pd.df or np.array

    forecast_horizon        : number of future forecast time_steps, default = 1

    limit_steps             : limits the number of simulation/predictions into the future,
                              default False, meaning that steps is equal to length of validation set

    pi_alpha                : measure to adjust confidence interval, default is set to 1.96, which is a 95% PI

    update                  : bool value, if True the new observations are used to fit the model again
                                default = True

    Returns
    -------
    :expected_value : expected forecast values for each test sample over the forecast horizon
                        (shape:len(y_train),forecast_horizon)
    :upper_limits   : upper interval for given forecasts (shape: (1,forecast_horizon))
    :lower_limits   : lower interval for given forecasts (shape: (1,forecast_horizon))

    '''
    """

    num_cores = multiprocessing.cpu_count()

    model = ETSModel(y_train, error="add", trend="add", damped_trend=True, seasonal="add",
                     dates=y_train.index)
    fit = model.fit()

    def ets_predict(i):
        if update:
            # extend the train-series with observed values as we move forward in the prediction horizon
            # to achieve a receding window prediction
            y_train_i = pd.concat([y_train, y_test.iloc[0:i]])
            model = ETSModel(y_train_i, error="add", trend="add", damped_trend=True, seasonal="add",
                             dates=y_train_i.index)
            fit = model.fit()
        # There are several different ETS methods available:
        #  - forecast: makes out of sample predictions
        #  - predict: in sample and out of sample predictions
        #  - simulate: runs simulations of the statespace model
        #  - get_prediction: in sample and out of sample predictions, as well as prediction intervals
        if i + forecast_horizon <= len(y_test):
            pred = fit.get_prediction(start=y_test.index[i], end=y_test.index[
                i + forecast_horizon - 1]).summary_frame()  # with: method = 'simulated', simulate_repetitions=100 we can simulate the PI's
            ## --plotting current prediction--
            # plt.rcParams['figure.figsize'] = (12, 8)
            # pred["mean"].plot(label='mean prediction')
            # pred["pi_lower"].plot(linestyle='--', color='tab:blue', label='95% interval')
            # pred["pi_upper"].plot(linestyle='--', color='tab:blue', label='_')
            # y_test[i:i-1 + forecast_horizon].plot(label='true_values')
            # plt.legend()
            # plt.show()
        return pred["mean"], pred["pi_upper"], pred["pi_lower"]

    test_period = range(len(y_test))
    if limit_steps:
        test_period = range(len(limit_steps))

    expected_value, fc_u, fc_l = \
        zip(*Parallel(n_jobs=min(num_cores,len(test_period)))(delayed(ets_predict)(i) for i in test_period))

    return np.asarray(expected_value), np.asarray(fc_u), np.asarray(fc_l)