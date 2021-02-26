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
import math

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from arch import arch_model
from joblib import Parallel, delayed
from pmdarima.arima import auto_arima
# ======================================
# =======================================
# save baseline model
# =============================================================================

def save_baseline(path, fitted,name='sarimax', predictions=None, save_predictions = True):
    print('Saving fitted sarimax model')
    if 'ResultsWrapper' in str(type(fitted)):
        fitted.save(path+name+".pkl")
    else:
        with open(path+name+".pkl", 'wb') as pkl:
            pickle.dump(fitted, pkl)
    if save_predictions:
        print('Saving predictions')
        with open('sarimax_predictions.csv', 'w') as prediction_logger:
            wr = csv.writer(prediction_logger, quoting=csv.QUOTE_ALL)
            wr.writerow(predictions)
    #print('Plot fitted (S)ARIMA(X) model diagnostics')
    #fitted.plot_diagnostics(figsize=(15, 12))
    #plt.show()
    return

# =============================================================================
# load baseline model
# =============================================================================

def load_baseline(path, name = 'sarimax'):
    full_path_to_file = path + name+".pkl"
    if os.path.isfile(full_path_to_file):
        with open(full_path_to_file, "rb") as f:
            fitted = pickle.load(f)
        return fitted
    else:
        return None
# =============================================================================
# train SARIMAX model
# =============================================================================

def train_SARIMAX(endog, exog, order, seasonal_order=None, trend='n'):

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
                            order=order, seasonal_order=seasonal_order, trend=trend) #time_varying_regression = True,
    fitted = model.fit(disp=-1)
    # use AICc - Take corrected Akaike Information Criterion here as default for performance check
    return fitted, model, fitted.aicc

def auto_sarimax_wrapper(endog, exog=None, order=None, seasonal_order=None, seasonal=True, trend='n',
                         grid_search = False, train_limit= 300):
    print('Train a (S)ARIMA(X) timeseries model...')
    if len(endog)>train_limit:
        print('Training set too long for ARIMA model. '
              'ARIMA does not improve with very long training sets. Set training set to: ',train_limit)
    else:
        train_limit=len(endog)
    if exog is not None: exog = exog.iloc[-train_limit:]

    if grid_search:
        # Apply parameter tuning script for (seasonal) ARIMA using [auto-arima]
        # (https://github.com/alkaline-ml/pmdarima)
        print('Training SARIMA(X) with parameter grid search by auto-arima...')

        auto_model = auto_arima(endog[-train_limit:], exog,
                                start_p=1, start_q=1, max_p=3, max_q=3, m=24, start_P=0,
                                start_D=0, max_D=2, seasonal=seasonal, trend=trend,
                                stepwise=True, trace=True, suppress_warnings=True, error_action='ignore')
        # TODO: Not sure if m is set correctly per default. Check if there are better options
        auto_model.summary()
        order = auto_model.order
        if seasonal: seasonal_order = auto_model.seasonal_order
    if seasonal is False and exog is not None:
        seasonal_order = None
        print('Train ARIMAX with order:', order)
    elif seasonal is False and exog is None:
        seasonal_order = None
        print('Train ARIMA with order:', order)
    elif seasonal is True and exog is not None:
        print('Train SARIMAX with order:',order,' and seasonal order:',seasonal_order)
    else:
        print('Train SARIMA with order:',order,' and seasonal order:',seasonal_order)
    sarimax_model, untrained_model, val_loss = train_SARIMAX(endog[-train_limit:], exog, order,
                                                             seasonal_order, trend=trend)
    return sarimax_model, untrained_model, val_loss
# =============================================================================
# evaluate forecasts
# =============================================================================

def eval_forecast(forecasts, endog_val, upper_limits, lower_limits, seasonality=24, alpha=0.05, total=True):

    """
    Calculates evaluation metrics

    Parameters
    ----------
    forecasts               : calculated forecasts

    endog_val               : reference, measured target variable, if available

    upper_limits            : upper interval for given forecasts

    lower_limits            : lower interval for given forecasts

    total                   : true if metrics shall be calculated over horizon

    num_pred:               : number of independent predictions with length=forecast_horizon over validation period


    Returns
    -------
    1)mse_horizon
    2)rmse_horizon
    4)sharpness_horizon
    5)coverage_horizon
    6)mis_horizon

    """
    forecasts = torch.tensor(forecasts)
    true_values = torch.tensor(endog_val)
    upper_limits = torch.tensor(upper_limits)
    lower_limits = torch.tensor(lower_limits)
    #torch.tensor([i[target].T.values for i in output_matrix]).reshape(forecasts.shape).type(torch.FloatTensor)
    #true_values = torch.tensor([i.T.values for i in endog_val).reshape(forecasts.shape).type(torch.FloatTensor)
    #upper_limits = torch.tensor([i.values for i in upper_limits]).reshape(forecasts.shape).type(torch.FloatTensor)
    #lower_limits = torch.tensor([i.values for i in lower_limits]).reshape(forecasts.shape).type(torch.FloatTensor)

    if total:
        mse = metrics.mse(true_values, [forecasts])
        rmse = metrics.rmse(true_values, [forecasts])
        mase=metrics.mase(true_values, [forecasts], seasonality, true_values) #MASE always needs a reference vale, here we assume a 24 timestep seasonality.
        rae= metrics.rae(true_values, [forecasts])
        mae=metrics.nmae(true_values, [forecasts])
        sharpness = metrics.sharpness(None,[upper_limits, lower_limits])
        coverage = metrics.picp(true_values, [upper_limits, lower_limits])
        mis = metrics.mis(true_values, [upper_limits, lower_limits], alpha=alpha)
        return mse, rmse, mase, rae, mae, sharpness, coverage, mis
    else:
        rmse_horizon = metrics.rmse(true_values, [forecasts], total)
        sharpness_horizon = metrics.sharpness(None,[upper_limits, lower_limits], total)
        coverage_horizon = metrics.picp(true_values, [upper_limits, lower_limits], total)
        mis_horizon = metrics.mis(true_values, [upper_limits, lower_limits], alpha=alpha, total=total)
        return rmse_horizon, sharpness_horizon, coverage_horizon, mis_horizon

# =============================================================================
# Create multiple forecasts with one model
# =============================================================================

def make_forecasts(endog_train, endog_val, exog=None, exog_forecast=None, fitted=None, forecast_horizon=1,
                   limit_steps=False, pi_alpha=1.96, online = False, train_limit=300):

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

    online                  : bool value, if True the new observations are used to fit the model again
                                default = False
    Returns
    -------
    1)forecasts             : calculated forecasts

    """

    num_cores = multiprocessing.cpu_count()
    test_period = range(endog_val.shape[0])

    if limit_steps:
        test_period = range(limit_steps)
    if online and len(endog_val-1) > train_limit:
        print('Added re-training set too long for SARIMAX model. '
              'SARIMAX does not improve with very long training sets. Set re-training set to latest: ', train_limit, ' entries')
    else:
        train_limit = len(endog_val)

    if exog is not None and exog_forecast is not None: print('Predicting with exogenous variables')
    else: print('Predictions made without exogenous variables')

    def sarimax_predict(i, fitted=None):
        exog_train_i = exog
        exog_forecast_i = exog_forecast
        #forecast = pd.Series(None, index=endog_val.index[i:i + forecast_horizon])
        #fc_u = pd.Series(None, index=endog_val.index[i:i + forecast_horizon])
        #fc_l = pd.Series(None, index=endog_val.index[i:i + forecast_horizon])
        endog_train_i = pd.concat([endog_train, endog_val.iloc[0:i]])
        if exog_train_i is not None and exog_forecast_i is not None:
            exog_forecast_i = exog_forecast[i:i + forecast_horizon]
            exog_train_i = pd.concat([exog, exog_forecast.iloc[0:i]]).iloc[-train_limit:]

        if 'ResultsWrapper' in str(type(fitted)):
            if online:
                # if we do standard parameters we have a SARIMAX object which can only be extended with apply (endog, exog)
                fitted = fitted.apply(endog=endog_train_i[-train_limit:], exog=exog_train_i)
            res = fitted.get_forecast(forecast_horizon, exog=exog_forecast_i)
        else:
            if online:
                # when using hyperparam optimization we do have an arima object, then we need to use Update (y, X)
                # Of course, you’ll have to re-persist your ARIMA model after updating it!
                fitted = fitted.update(y=endog_train_i[-train_limit:], X=exog_train_i)
            res = fitted.predict(n_periods=forecast_horizon, X=exog_forecast_i,
                                      return_conf_int = False)
        fc_u = res.predicted_mean + pi_alpha * res.se_mean
        fc_l = res.predicted_mean - pi_alpha * res.se_mean

        #conf_int

        # fetch conf_int by forecasting the upper and lower limit, given the fitted_models uncertainty params
        # (higher computation effort than standard way)
        # upper_limits, lower_limits = baselines.apply_PI_params(untrained_model, model, val_in, val_observed, target,
        #                                                       exog,  PAR['forecast_horizon'])
        return res.predicted_mean, fc_u, fc_l

    expected_value, fc_u, fc_l = \
        zip(*Parallel(n_jobs=min(num_cores,len(test_period)), mmap_mode = 'c',
                      temp_folder='/tmp')(delayed(sarimax_predict)(i, fitted)
                                          for i in test_period if i+forecast_horizon<=endog_val.shape[0]))
    print('Training and validating (S)ARIMA(X) completed.')
    return np.asarray(expected_value), np.asarray(fc_u), np.asarray(fc_l)

# =============================================================================
# predictioninterval functions
# =============================================================================

def naive_predictioninterval(forecasts, endog_val, forecast_horizon, alpha = 1.96):

    """
    Calculates predictioninterval for given forecasts with Naive method https://otexts.com/fpp2/prediction-intervals.html
    So it's rather an ex-post uncertainty measure than one which one can use for prob. forecasting
    as it assumes to have the future target values.

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
    residuals = endog_val-forecasts
    sigma = np.std(residuals)

    #for r in range(forecast_horizon): sigma_h.append(sigma * np.sqrt(r+1))
    sigma_h = [sigma * np.sqrt(x+1) for x in range(forecast_horizon)]

    sigma_hn = pd.Series(sigma_h)
    sigma_hn.index = endog_val.index

    fc_u = forecasts + alpha * sigma_hn
    fc_l = forecasts - alpha * sigma_hn

    return fc_u, fc_l

def GARCH_predictioninterval(endog_train, endog_val, forecast_horizon, mean_forecast=None, p=1, q=1, alpha = 1.96,
                             limit_steps=False):

    """
    Calculates predictioninterval for given forecasts with GARCH method https://github.com/bashtage/arch

    Parameters
    ----------
    endog_train             : train-set of target variable

    endog_val               : validation-set of target variable

    forecast_horizon        : number of future forecast time_steps

    alpha                   : Confidence: default 1.96 equals to 95% confidence


    Returns
    -------
    1)upper_limits          : List of upper intervall for given forecasts
    2)lower_limits          : List of lower intervall for given forecasts

    """
    print('Train a General autoregressive conditional heteroeskedasticity (GARCH) model...')
    test_period = range(endog_val.shape[0])
    num_cores = multiprocessing.cpu_count()

    if limit_steps:
        test_period = range(limit_steps)

    model_garch = arch_model(endog_train, vol = 'GARCH', p=p, q=q)#(endog_val[1:]
    res = model_garch.fit(update_freq=5)
    # do stepwise iterion and prolongation of the train data again, as a forecast can only work in-sample
    def garch_PI_predict(i):
        # extend the train-series with observed values as we move forward in the prediction horizon
        # to achieve a receding window prediction
        y_train_i = pd.concat([endog_train, endog_val.iloc[0:i]])
        #need to refit, since forecast cannot do out of sample forecasts
        model_garch = arch_model(y_train_i, vol='GARCH', p=p, q=q)
        res = model_garch.fit(update_freq=5)
        forecast = model_garch.forecast(res.params, horizon=forecast_horizon)# , start=endog_train.index[-1]
        #TODO: checkout that mean[0] might be NAN because it starts at start -2
        # TODO: could try to use the previous calculates sarimax mean forecast instead...
        if isinstance(mean_forecast,np.ndarray):
            expected_value=pd.Series(mean_forecast[i])
        else:
            expected_value = pd.Series(forecast.mean.iloc[-1])
        sigma = pd.Series([math.sqrt(number) for number in forecast.residual_variance.iloc[-1]])
        expected_value.index = endog_val.index[i:i + forecast_horizon]#this can be an issue if we reach the end of endog val with i
        sigma.index = endog_val.index[i:i + forecast_horizon]
        sigma_hn = sum(sigma)/len(sigma)

        fc_u = expected_value + alpha * sigma
        fc_l = expected_value - alpha * sigma

        print('Training and validating GARCH model completed.')
        return fc_u, fc_l

    fc_u, fc_l = zip(*Parallel(n_jobs=min(num_cores,len(test_period)), mmap_mode='c',
                               temp_folder='/tmp')(delayed(garch_PI_predict)(i)
                                                   for i in test_period if i+forecast_horizon<=endog_val.shape[0]))
    return np.asarray(fc_u), np.asarray(fc_l)

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
    if decomposed: print('Train a naive timeseries model cleared with seasonal decomposition (STL)...')
    else: print('Train a naive (persistence) timeseries model...')

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

    if decomposed: print('Training and validating naive (STL) completed.')
    else: print('Training and validating naive (persistence) model completed.')

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
    print('Train a seasonal naive timeseries model with seasonality=', seasonality, '...')

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
    print('Training and validating seasonal naive model completed.')

    return expected_value, fc_u, fc_l

def exp_smoothing(y_train, y_test, forecast_horizon=1, limit_steps=False, pi_alpha=1.96, online = True):
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
    print('Train an exponential smoothing timeseries model (ETS)...')
    num_cores = multiprocessing.cpu_count()

    model = ETSModel(y_train, error="add", trend="add", damped_trend=True, seasonal="add",
                     dates=y_train.index)
    fit = model.fit()

    def ets_predict(i):
        if online:
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
        test_period = range(limit_steps)

    expected_value, fc_u, fc_l = \
        zip(*Parallel(n_jobs=min(num_cores,len(test_period)), mmap_mode = 'c',
                      temp_folder='/tmp')(delayed(ets_predict)(i)
                                          for i in test_period
                                          if i+forecast_horizon<=len(y_test)))

    print('Training and validating ETS model completed.')
    return np.asarray(expected_value), np.asarray(fc_u), np.asarray(fc_l)