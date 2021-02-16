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

import plf_util.eval_metrics as metrics
import torch
import copy
import numpy as np
import pandas as pd
import csv
import statsmodels.tsa.statespace.sarimax as sarimax
import pickle

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
# =============================================================================
# save SARIMAX model
# =============================================================================

def save_SARIMAX(path, fitted, predictions, save_predictions = True):
    if 'ResultsWrapper' in str(type(fitted)):
        fitted.save(path+"\\sarimax.pickle")
    else:
        with open(path+"\\sarimax.pkl", 'wb') as pkl:
            pickle.dump(fitted, pkl)
    if save_predictions:
        with open('sarimax_predictions.csv', 'w') as prediction_logger:
            wr = csv.writer(prediction_logger, quoting=csv.QUOTE_ALL)
            wr.writerow(predictions)
    return

# =============================================================================
# load SARIMAX model
# =============================================================================

def load_SARIMAX(path):
    fitted = sarimax.SARIMAXResults.load(path)
    return fitted
# =============================================================================
# train SARIMAX model
# =============================================================================

def train_SARIMAX(input_matrix, target, exog, order, seasonal_order=None, number_set=0, trend='n'):

    """
    Trains a SARIMAX model

    Parameters
    ----------
    input_matrix            : List of input values

    output_matrix           : List of true output values Values

    target                  : str target column

    exog                    : str features

    order                   : order of model

    seasonal_order          : seasonal order of model

    trend                   : trend method used for training, here we always use trend = n as default

    forecast_horizon        : number of future forecast time_steps

    number_set              : choose index of input_matrix and output_matrix for training

    Returns
    -------
    1)fitted                : trained model instance
    2)model                 : model instance

    """
    model = sarimax.SARIMAX(input_matrix[number_set][target], exog = input_matrix[number_set][exog],
                            order=order, seasonal_order=seasonal_order, trend=trend,
                            freq=pd.to_datetime(input_matrix[number_set][target].index).inferred_freq, mle_regression=False)
    fitted = model.fit(disp=-1)
    # use AICc - Take corrected Akaike Information Criterion here as default for performance check
    return fitted, model, fitted.aicc
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

def make_forecasts(input_matrix, output_matrix, target, exog, fitted, forecast_horizon):

    """
    Calculates a number of forecasts

    Parameters
    ----------
    input_matrix            : List of input values

    output_matrix           : List of True output Values

    target                  : str target column

    exog                    : str features

    fitted                  : trained model

    forecast_horizon        : number of future forecast time_steps

    Returns
    -------
    1)forecasts             : List of calculated forecasts

    """
    forecasts = []
    for i in range(len(input_matrix)):
        if 'ResultsWrapper' in str(type(fitted)):
            # if we do standard parameters we have a SARIMAX object which can only be extended with apply (endog, exog)
            new_fitted = fitted.apply(endog=input_matrix[i][target], exog=input_matrix[i][exog])
            forecast = new_fitted.forecast(forecast_horizon, exog=output_matrix[i][exog])
        else:
            # when using hyperparam optimization we do have an arima object, then we need to use Update (y, X)
            # Or just add new observations?
            ## arima.add_new_observations(test)  # pretend these are the new ones
            ## Your model will now produce forecasts from the new latest observations.
            # Of course, you’ll have to re-persist your ARIMA model after updating it!
            new_fitted = fitted.update(y=input_matrix[i][target], X=input_matrix[i][exog])
            forecast = new_fitted.predict(n_periods=forecast_horizon, X=output_matrix[i][exog],return_conf_int = False)
        forecasts.append(forecast)
    return forecasts

# =============================================================================
# predictioninterval functions
# =============================================================================

def naive_predictioninterval(forecasts, output_matrix, target, forecast_horizon, alpha = 1.96):

    """
    Calculates predictioninterval for given forecasts with Naive method https://otexts.com/fpp2/prediction-intervals.html

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

def persist_forecast(x_train, x_test, y_train, forecast_horizon, alpha = 1.96):
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

def exp_smoothing(y_train, y_test, forecast_horizon):
    expected_value = np.zeros(shape=(len(y_test), forecast_horizon))
    fc_u = np.zeros(shape=(len(y_test), forecast_horizon))
    fc_l = np.zeros(shape=(len(y_test), forecast_horizon))
    time_delta = pd.to_datetime(y_train.index[1]) - pd.to_datetime(y_train.index[0])
    dates = pd.date_range(start=y_train.index[0], end=y_train.index[-1], freq=str(time_delta.seconds/3600)+'H')
    model = ETSModel(y_train, error="add", trend="add", seasonal="mul",
                     damped_trend=True, seasonal_periods=4)
    model = ETSModel(y_train, error="add", trend="add", seasonal="mul",
                     damped_trend = True, seasonal_periods = 4, dates = pd.to_datetime(y_train.index))
    fit = model.fit()
    # There are several different ETS methods available:
    #  - forecast: makes out of sample predictions
    #  - predict: in sample and out of sample predictions
    #  - simulate: runs simulations of the statespace model
    #  - get_prediction: in sample and out of sample predictions, as well as prediction intervals

    for i in range(len(y_test)):
        if i+forecast_horizon < len(y_test):
            pred = fit.get_prediction(start=y_test.index[i],
                                      end=y_test.index[i+forecast_horizon],
                                      simulate_repetitions=100).summary_frame()
            pred = fit.get_prediction(start=i,
                                      end=i + forecast_horizon,
                                      simulate_repetitions = 100).summary_frame()
            expected_value[i,:], fc_u[i,:], fc_l[i,:] = pred["mean"], pred["pi_upper"], pred["pi_lower"]

    return expected_value, fc_u, fc_l