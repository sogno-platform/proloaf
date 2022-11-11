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
"""
Provides functions to load and save baselines, as well as functions to train (S)ARIMA(X) and GARCH models
and generate various baseline forecasts.
"""

import os
from typing import Any, Dict, List
from proloaf import metrics
import torch
import copy
import numpy as np
import pandas as pd
import csv
import statsmodels.tsa.statespace.sarimax as sarimax
import pickle
import multiprocessing
import math

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from joblib import Parallel, delayed
from pmdarima.arima import auto_arima
from proloaf.event_logging import create_event_logger

# ======================================
# =======================================
# save baseline model
# =============================================================================
from proloaf import plot

logger = create_event_logger(__name__)


def test_stationarity(timeseries, maxlag):
    # source: https://towardsdatascience.com/end-to-end-time-series-analysis-and-forecasting-a-trio-of-sarimax-lstm-and-prophet-part-1-306367e57db8
    # Perform Dickey-Fuller test:
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, maxlag=maxlag, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(round(dfoutput, 3))


def save_baseline(
    path, fitted, name="sarimax", predictions=None, save_predictions=True
):
    """
    Save the given fitted sarimax model

    Parameters
    ----------
    path : string
        The destination path
    fitted : statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper
        A class that holds a trained model instance
    name : string, default = 'sarimax'
        The name to give the saved model
    predictions : ndarray, default = None
        The predictions to be saved
    save_predictions : bool, default = True
        If True, save the given predictions to a file called 'sarimax_predictions.csv'

    Returns
    -------
    No return value
    """

    print("Saving fitted sarimax model")
    if not os.path.exists(os.path.join(path)):  # make output folder if it doesn't exist
        os.makedirs(os.path.join(path))
    if "ResultsWrapper" in str(type(fitted)):
        fitted.save(path + name + ".pkl")
    else:
        with open(path + name + ".pkl", "wb") as pkl:
            pickle.dump(fitted, pkl)
    if save_predictions:
        print("Saving predictions")
        with open("sarimax_predictions.csv", "w") as prediction_logger:
            wr = csv.writer(prediction_logger, quoting=csv.QUOTE_ALL)
            wr.writerow(predictions)
    # print('Plot fitted (S)ARIMA(X) model diagnostics')
    # fitted.plot_diagnostics(figsize=(15, 12))
    # plt.show()
    return


# =============================================================================
# load baseline model
# =============================================================================


def load_baseline(path, name="sarimax"):
    """
    Load a fitted sarimax model with the given name from the specified path

    .. warning::
        Loading pickled models can be insecure if the data are erroneous or maliciously constructed.
        Only unpickle data from a trusted, authenticated source.

    Parameters
    ----------
    path : string
        The path to the file which should be loaded
    name : string, default = 'sarimax'
        The name of the file to load

    Returns
    -------
    statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper or None
        Return a trained model instance, or None if loading was unsuccessful
    """

    full_path_to_file = path + name + ".pkl"
    if os.path.isfile(full_path_to_file):
        with open(full_path_to_file, "rb") as f:
            fitted = pickle.load(f)
        return fitted
    else:
        return None


# =============================================================================
# train SARIMAX model
# =============================================================================


def train_SARIMAX(endog, exog, order, seasonal_order=None, trend="c"):
    """
    Train a SARIMAX model

    For more information on the parameters, see
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    Parameters
    ----------
    endog : array_like
        Train values of target variable of type pandas.DataFrame or np.array
    exog : array_like
        Array of exogenous features of type pandas.DataFrame or np.array
    order : iterable
        The (p,d,q) order of the model
    seasonal_order : iterable
        The (P,D,Q,s) order of the seasonal component of the model
    trend : string, default = 'n'
        Indicates type of trend polynomial. Valid options are 'n', 'c', 't', or 'ct'

    Returns
    -------
    statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper
        A class that holds the trained model instance. Methods from statsmodels.tsa.statespace.mlemodel.MLEResults can
        be called on it
    statsmodels.tsa.statespace.sarimax.SARIMAX
        The untrained model instance
    float
        Akaike Information Criterion with small sample correction
    """

    model = sarimax.SARIMAX(
        endog=endog, ecog=exog, order=order, seasonal_order=seasonal_order, trend=trend
    )  # time_varying_regression = True,
    fitted = model.fit(disp=-1)
    # use AICc - Take corrected Akaike Information Criterion here as default for performance check
    return fitted, model, fitted.aicc


def auto_sarimax_wrapper(
    endog,
    exog=None,
    order=None,
    seasonal_order=None,
    seasonal=True,
    m=1,
    lag=1,
    trend="c",
    grid_search=False,
    train_limit=300,
):
    """
    Train a (S)ARIMA(X) timeseries model

    Parameters
    ----------
    endog : array_like
        Train values of target variable of type pandas.DataFrame or np.array
    exog : array_like
        Array of exogenous features of type pandas.DataFrame or np.array
    order : iterable
        The (p,d,q) order of the model
    seasonal_order : iterable
        The (P,D,Q,s) order of the seasonal component of the model
    seasonal : bool, default = True
        Specifies whether to fit a seasonal ARIMA or not
    m : int, default = 1
        The number of periods in each season (4 is quarterly, 12 is monthly, etc.)
    lag : int, default = 1
        Maximum seasonal moving average order
    trend : string, default = 'n'
        Indicates type of trend polynomial
    grid_search : bool, default = False
        Specifies whether to apply parameter tuning script for (seasonal) ARIMA using [auto-arima]
    train_limit : int, default = 300
        Used to set the maximum length of the training set. This way, the training set contains
        up to 'train_limit' of the latest entries.

    Returns
    -------
    statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper
        A class that holds the trained model instance. Methods from statsmodels.tsa.statespace.mlemodel.MLEResults can
        be called on it
    statsmodels.tsa.statespace.sarimax.SARIMAX
        The untrained model instance
    float
        Akaike Information Criterion with small sample correction
    """

    print("Train a (S)ARIMA(X) timeseries model...")
    if len(endog) > train_limit:
        print(
            "Training set too long for ARIMA model. "
            "ARIMA does not improve with very long training sets. Set training set to: ",
            train_limit,
        )
    else:
        train_limit = len(endog)

    if exog is not None:
        exog = exog.iloc[-train_limit:]
        trend = None  # TODO: calrify why. If exogeneous variables have a constant trend e.g. trend = 'c' will fail.

    if grid_search:
        # Apply parameter tuning script for (seasonal) ARIMA using [auto-arima]
        # (https://github.com/alkaline-ml/pmdarima)
        # TODO: add warning that auto arima does not support exogeneous features
        print("Training SARIMA(X) with parameter grid search by auto-arima...")
        if not seasonal:
            m = 1
        auto_model = auto_arima(
            endog[-train_limit:],
            exog,
            start_p=1,
            start_q=1,
            max_p=lag,
            max_q=lag,
            m=m,
            start_P=0,
            start_D=0,
            max_D=2,
            seasonal=seasonal,
            trend=trend,
            stepwise=True,
            trace=True,
            suppress_warnings=True,
            error_action="ignore",
        )

        auto_model.summary()
        order = auto_model.order
        if seasonal:
            seasonal_order = auto_model.seasonal_order
    if seasonal is False and exog is not None:
        seasonal_order = None
        print("Train ARIMAX with order:", order)
    elif seasonal is False and exog is None:
        seasonal_order = None
        print("Train ARIMA with order:", order)
    elif seasonal is True and exog is not None:
        print(
            "Train SARIMAX with order:", order, " and seasonal order:", seasonal_order
        )
    else:
        print("Train SARIMA with order:", order, " and seasonal order:", seasonal_order)
    sarimax_model, untrained_model, val_loss = train_SARIMAX(
        endog[-train_limit:], exog, order, seasonal_order, trend=trend
    )
    return sarimax_model, untrained_model, val_loss, order, seasonal_order


# =============================================================================
# evaluate forecasts
# =============================================================================


def eval_forecast(
    forecasts,
    endog_val,
    upper_limits,
    lower_limits,
    model_name:str,
    path:str,
    config: Dict[str,Any],
    analyzed_metrics_avg: List[metrics.Metric],
    analyzed_metrics_sample: List[metrics.Metric],
    analyzed_metrics_timesteps: List[metrics.Metric],
):
    """
    Calculate evaluation metrics

    Parameters
    ----------
    forecasts : array_like
        The calculated forecasts
    endog_val : array_like
        The reference or measured target variable, if available
    upper_limits : array_like
        The upper interval for the given forecasts
    lower_limits : array_like
        The lower interval for the given forecasts
    model_name:str
        Name used in the save path for the plots
    path:str,
        Prepend for the save path
    config: Dict[str,Any]
        The Project config for the deeplearning models the baselines are to be compared with
    analyzed_metrics_avg: List[metrics.Metric]
        List of metrics to be evaluated as totals over sample and timestep
    analyzed_metrics_sample: List[metrics.Metric]
        List of metrics to be evaluated per sample.
    analyzed_metrics_timesteps: List[metrics.Metric]
        List of metrics to be evaluated per timestep.

    Returns
    -------
    List[torch.Tensor]
        List of 0D tensors with the results of the metrics `analyzed_metric_avg`.
    pandas.DataFrame
        Dataframe containing the metrics per timestep. Index is integer, each column is a different metric.
    torch.Tensor
        2D torch.Tensor with dimension (samples,timesteps) with true (not predicted) values. Only for convienience.
    torch.Tensor
        2D torch.Tensor with dimension (samples,timesteps) with predited values. Only for convienience.
    torch.Tensor
        2D torch.Tensor with dimension (samples,timesteps) with predicted upper limits for the values. Only for convienience.
    torch.Tensor
        2D torch.Tensor with dimension (samples,timesteps) with predicted lower limits for the values. Only for convienience.
    
    

    """
    forecasts = torch.tensor(forecasts)
    true_values = torch.tensor(endog_val)
    upper_limits = torch.tensor(upper_limits)
    lower_limits = torch.tensor(lower_limits)
    #print(f"{upper_limits.size() = }")
    #print(f"{true_values.size() = }")

    quantile_predictions = metrics.QuantilePrediction(
        torch.stack([forecasts, upper_limits, lower_limits], dim=2),
        quantiles=(0.5, 0.975, 0.025),
    )

    # targets_rescaled, output_rescaled = dh.rescale_manually(..)
    results_avg = {}
    for metric in analyzed_metrics_avg:
        results_avg[metric.id] = metric.from_quantiles(
            true_values.unsqueeze(dim=2), quantile_predictions, avg_over="all"
        )
    results_samp = {}
    for metric in analyzed_metrics_sample:
        results_samp[metric.id] = metric.from_quantiles(
            true_values.unsqueeze(dim=2), quantile_predictions, avg_over="time"
        )
    results_ts = {}
    for metric in analyzed_metrics_timesteps:
        results_ts[metric.id] = metric.from_quantiles(
            true_values.unsqueeze(dim=2), quantile_predictions, avg_over="sample"
        )
        ts_length = len(results_ts[metric.id])

    df_results = pd.DataFrame(results_ts, index=range(ts_length))
    df_sample = pd.DataFrame(results_samp)

    # plot forecast for sample time steps
    # the first and 24th timestep relative to the start of the Test-Dataset
    testhours = [0, 1]
    for i in testhours:
        plot.plot_timestep(
            true_values[i].detach().numpy(),
            forecasts[i].detach().numpy(),
            upper_limits[i].detach().numpy(),
            lower_limits[i].detach().numpy(),
            i,
            path + model_name,
            config["cap_limit"],
        )
    # BOXPLOTS
    plot.plot_boxplot(
        df_sample.filter(['Mse', 'Rmse']),
        sample_frequency=24,
        save_to=path + model_name,
    )
    return (
        results_avg.values(),
        df_results,
        pd.DataFrame(results_samp),
        true_values,
        forecasts,
        upper_limits,
        lower_limits,
    )


# =============================================================================
# Create multiple forecasts with one model
# =============================================================================


def make_forecasts(
    endog_train,
    endog_val,
    exog=None,
    exog_forecast=None,
    fitted=None,
    forecast_horizon=1,
    limit_steps=False,
    pi_alpha=1.96,
    online=False,
    train_limit=300,
):
    """
    Calculate a number of forecasts

    Parameters
    ----------
    endog_train : pandas.DataFrame
        the new values of the target variable
    endog_val : pandas.DataFrame
        The retraining set
    exog : pandas.DataFrame
        The new exogenous features for new training
    exog_forecast : pandas.DataFrame
        The exogenous variables to match the prediction horizon
    fitted : statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper
        The trained model
    forecast_horizon : int, default = 1
        Number of future steps to be forecasted
    limit_steps : int, default = False
        limits the number of simulation/predictions into the future. If False, steps is equal to length of validation set
    pi_alpha : float, default = 1.96
        Measure to adjust confidence interval, default is set to 1.96, which is a 95% PI
    online : bool, default = False
         If True, the new observations are used to fit the model again
    train_limit : int, default = 300
        Used to set the maximum length of the re-training set. This way, the re-training set contains
        up to 'train_limit' of the latest entries.

    Returns
    -------
    numpy.ndarray
        Expected values of the prediction
    numpy.ndarray
        Upper limit of prediction
    numpy.ndarray
        Lower limit of prediction
    """
    num_cores = max(multiprocessing.cpu_count() - 2, 1)
    test_period = range(endog_val.shape[0])

    if limit_steps:
        test_period = range(limit_steps)
    if online and len(endog_val - 1) > train_limit:
        print(
            "Added re-training set too long for SARIMAX model. "
            "SARIMAX does not improve with very long training sets. Set re-training set to latest: ",
            train_limit,
            " entries",
        )
    else:
        train_limit = len(endog_val)

    if exog is not None and exog_forecast is not None:
        print("Predicting with exogenous variables")
    else:
        print("Predictions made without exogenous variables")

    def sarimax_predict(i, fitted=None):
        exog_train_i = exog
        exog_forecast_i = exog_forecast
        # forecast = pd.Series(None, index=endog_val.index[i:i + forecast_horizon])
        # fc_u = pd.Series(None, index=endog_val.index[i:i + forecast_horizon])
        # fc_l = pd.Series(None, index=endog_val.index[i:i + forecast_horizon])
        endog_train_i = pd.concat([endog_train, endog_val.iloc[0:i]])
        if exog_train_i is not None and exog_forecast_i is not None:
            exog_forecast_i = exog_forecast[i : i + forecast_horizon]
            exog_train_i = pd.concat([exog, exog_forecast.iloc[0:i]]).iloc[
                -train_limit:
            ]

        if "ResultsWrapper" in str(type(fitted)):
            if online and exog is None:
                # if we do standard parameters we have a SARIMAX object which can only be extended with apply (endog, exog)
                # statsmodels v0.12.2: This [issue](https://github.com/statsmodels/statsmodels/issues/7019) occurs when
                # we only append small horizons here on which the trend is constant.
                # So if the model was trained with a trend, it would through a ValueError.
                # As in most tests we append one time-step we set the trend to none here to bypass the issue.
                # First tests show that the model is not suffering much in terms of forecast quality.
                fitted = fitted.apply(endog=endog_train_i[-train_limit:])
            res = fitted.get_forecast(forecast_horizon, exog=exog_forecast_i)
        else:
            if online and exog is None:
                # when using hyperparam optimization we do have an arima object, then we need to use Update (y, X)
                # Of course, you’ll have to re-persist your ARIMA model after updating it!
                # Same issue with constant trend warning
                fitted = fitted.update(y=endog_train_i[-train_limit:])
            res = fitted.predict(
                n_periods=forecast_horizon, X=exog_forecast_i, return_conf_int=False
            )
        fc_u = res.predicted_mean + pi_alpha * res.se_mean
        fc_l = res.predicted_mean - pi_alpha * res.se_mean

        # conf_int

        # fetch conf_int by forecasting the upper and lower limit, given the fitted_models uncertainty params
        # (higher computation effort than standard way)
        # upper_limits, lower_limits = baselines.apply_PI_params(untrained_model, model, val_in, val_observed, target,
        #                                                       exog,  PAR['forecast_horizon'])
        return res.predicted_mean, fc_u, fc_l

    expected_value, fc_u, fc_l = zip(
        *Parallel(
            n_jobs=min(num_cores, len(test_period)), mmap_mode="c", temp_folder="/tmp"
        )(
            delayed(sarimax_predict)(i, fitted)
            for i in test_period
            if i + forecast_horizon <= endog_val.shape[0]
        )
    )
    print("Training and validating (S)ARIMA(X) completed.")
    return np.asarray(expected_value), np.asarray(fc_u), np.asarray(fc_l)


# =============================================================================
# predictioninterval functions
# =============================================================================

def GARCH_predictioninterval(
    endog_train,
    endog_val,
    forecast_horizon,
    periodicity=1,
    mean_forecast=None,
    p=1,
    q=1,
    alpha=1.96,
    limit_steps=False,
):
    """
    Calculate the prediction interval for the given forecasts using the GARCH method https://github.com/bashtage/arch

    Parameters
    ----------
    endog_train : pandas.DataFrame
        Training set of target variable
    endog_val : pandas.DataFrame
        Validation set of target variable
    forecast_horizon : int
        Number of future steps to be forecasted
    periodicity : int or int list
            Either a scalar integer value indicating lag length or a list of integers specifying lag locations.
    mean_forecast : numpy.ndarray, default = None
            Previously forecasted expected values (e.g. the sarimax mean forecast). If set to None,
        the mean values of the GARCH forecast are used instead.
    p : int
            Lag order of the symmetric innovation
    q : int
            Lag order of lagged volatility or equivalent
    alpha : float, default = 1.96
        Measure to adjust confidence interval. Default is set to 1.96, which equals to 95% confidence
    limit_steps : int, default = False
        Limits the number of simulation/predictions into the future. If False, steps is equal to length of validation set

    Returns
    -------
    pandas.Series
        The forecasted expected values
    pandas.Series
        The upper interval for the given forecasts
    pandas.Series
        The lower interval for the given forecasts
    """

    print(
        "Train a General autoregressive conditional heteroeskedasticity (GARCH) model..."
    )
    test_period = range(endog_val.shape[0])
    num_cores = max(multiprocessing.cpu_count() - 2, 1)

    if limit_steps:
        test_period = range(limit_steps)

    model_garch = arch_model(
        endog_train, vol="GARCH", mean="LS", lags=periodicity, p=p, q=q
    )  # (endog_val[1:]
    res = model_garch.fit(update_freq=20)
    # do stepwise iteration and prolongation of the train data again, as a forecast can only work in-sample
    def garch_PI_predict(i):
        # extend the train-series with observed values as we move forward in the prediction horizon
        # to achieve a receding window prediction
        y_train_i = pd.concat([endog_train, endog_val.iloc[0:i]])
        # need to refit, since forecast cannot do out of sample forecasts
        model_garch = arch_model(
            y_train_i, vol="GARCH", mean="LS", lags=periodicity, p=p, q=q
        )
        res = model_garch.fit(update_freq=20)
        forecast = model_garch.forecast(
            res.params, horizon=forecast_horizon, reindex=True
        )  # , start=endog_train.index[-1]
        # TODO: checkout that mean[0] might be NAN because it starts at start -2
        # TODO: could try to use the previously calculated sarimax mean forecast instead...
        if isinstance(mean_forecast, np.ndarray):
            expected_value = pd.Series(mean_forecast[i])
        else:
            expected_value = pd.Series(forecast.mean.iloc[-1])
        sigma = pd.Series(
            [math.sqrt(number) for number in forecast.residual_variance.iloc[-1]]
        )
        expected_value.index = endog_val.index[
            i : i + forecast_horizon
        ]  # this can be an issue if we reach the end of endog val with i
        sigma.index = endog_val.index[i : i + forecast_horizon]
        sigma_hn = sum(sigma) / len(sigma)

        fc_u = expected_value + alpha * sigma
        fc_l = expected_value - alpha * sigma

        print("Training and validating GARCH model completed.")
        return expected_value, fc_u, fc_l

    expected_value, fc_u, fc_l = zip(
        *Parallel(
            n_jobs=min(num_cores, len(test_period)), mmap_mode="c", temp_folder="/tmp"
        )(
            delayed(garch_PI_predict)(i)
            for i in test_period
            if i + forecast_horizon <= endog_val.shape[0]
        )
    )
    print("Training and validating GARCH model completed.")
    return np.asarray(expected_value), np.asarray(fc_u), np.asarray(fc_l)


def simple_naive(series, horizon):
    """
    A simple naive forecast

    Return a forecast of length = 'horizon', where all elements are equal to the last element of 'series'

    Parameters
    ----------
    series : ndarray
        The data to use for the naive forecast
    horizon : int
        The length of the forecast

    Returns
    -------
    ndarray
        A simple naive forecast (Shape: ('horizon', ))
    """

    return np.array([series[-1]] * horizon)


def persist_forecast(
    x_train,
    x_test,
    y_train,
    forecast_horizon,
    periodicity=None,
    seasonality=1,
    decomposed=False,
    alpha=1.96,
):
    """
    Train and validate a naive (persistence) timeseries model or, (if decomposed = True), a naive
    timeseries model cleared with seasonal decomposition (STL)

    Fit persistence forecast over the train data, and calculate the standard deviation of residuals for each horizon
    over the horizon. Output the persistence forecast over the test data, and residual_std (upper and lower intervals)
    for each hour.

    Parameters
    ----------
    x_train : ndarray
        the input training dataset. (Shape: (length of training data,forecast_horizon))
    x_test : ndarray
        the input test dataset. (Shape: (length of test data,forecast_horizon))
    y_train : ndarray
        The targets (actual values) of the training dataset. (Shape: (length of training data,forecast_horizon))
    forecast_horizon : int
        Number of future steps to be forecasted
    periodicity : int, default = None
        The periodicity of the sequence. If endog (first arg of seasonal_decomposed) is ndarray,
        periodicity must be provided.
    seasonality : int, default = 1
        Length of the seasonal smoother. Must be an odd integer, and should normally be >= 7
    decomposed : bool, default = False
        If True, use seasonal decomposition
    alpha : float, default = 1.96
        Measure to adjust confidence interval. Default is set to 1.96, which equals to 95% confidence

    Returns
    -------
    ndarray
        Expected forecast values for each test sample over the forecast horizon.
        (Shape: (len(x_test),forecast_horizon))
    ndarray
        The upper interval for the given forecasts. (Shape: (len(x_test),forecast_horizon))
    ndarray
        The lower interval for the  forecasts. (Shape: (len(x_test),forecast_horizon))
    """

    if decomposed:
        print(
            "Train a naive timeseries model cleared with seasonal decomposition (STL)..."
        )
    else:
        print("Train a naive (persistence) timeseries model...")

    point_forecast_train = np.zeros(shape=(len(x_train), forecast_horizon))

    # difference between the observations and the corresponding fitted values
    residuals = np.zeros(shape=(len(x_train), forecast_horizon))

    if decomposed:
        for j in range(x_train.shape[1]):
            x_train[:, j] = seasonal_decomposed(
                np.reshape(x_train[:, j], (len(x_train), 1)),
                periodicity=periodicity,
                seasonality=seasonality,
            )
            x_test[:, j] = seasonal_decomposed(
                np.reshape(x_test[:, j], (len(x_test), 1)),
                periodicity=periodicity,
                seasonality=seasonality,
            )

    for i in range(len(x_train)):
        point_forecast_train[i, :] = simple_naive(x_train[i, :], forecast_horizon)
        residuals[i, :] = y_train[i, :] - point_forecast_train[i, :]

    # std deviation of residual for each hour(column wise)
    residual_std = np.nanstd(residuals, axis=0)
    expected_value = np.zeros(shape=(len(x_test), forecast_horizon))

    for i in range(len(x_test)):
        expected_value[i, :] = simple_naive(x_test[i, :], forecast_horizon)

    fc_u = expected_value + alpha * residual_std
    fc_l = expected_value - alpha * residual_std

    if decomposed:
        print("Training and validating naive (STL) completed.")
    else:
        print("Training and validating naive (persistence) model completed.")

    return expected_value, fc_u, fc_l


def seasonal_naive(series, forecast_horizon, periodicity, seasonality=1):
    """
    A seasonal naive forecast

    Parameters
    ----------
    series : ndarray
        The data to use for the seasonal naive forecast
    forecast_horizon : int
        Number of future steps to be forecasted
    periodicity : int
        The periodicity of the sequence.
    seasonality : int, default = 1
        Length of the seasonal smoother. Must be an odd integer, and should normally be >= 7

    Returns
    -------
    ndarray
        A seasonal naive forecast
    """
    forecast = np.array(
        [
            series[i % (periodicity * seasonality) - periodicity * seasonality]
            for i in range(forecast_horizon)
        ]
    )
    return forecast


def seasonal_decomposed(series, periodicity=None, seasonality=1):
    """
    Return the given time series after seasonal adjustment ( Y(t) - S(t) )

    Parameters
    ----------
    series : ndarray
        The series to be seasonally decomposed
    periodicity : int
        The periodicity of the sequence.
    seasonality : int, default = 1
        Length of the seasonal smoother. Must be an odd integer, and should normally be >= 7

    Returns
    -------
    ndarray
        The seasonally decomposed series
    """
    # seasonally adjusted time series Y(t)-S(t).
    if periodicity:
        if (periodicity % 2) == 0:
            if (seasonality % 2) == 0:
                stl = STL(
                    series.squeeze(), period=periodicity + 1, seasonal=seasonality + 1
                )
            else:
                stl = STL(
                    series.squeeze(), period=periodicity + 1, seasonal=seasonality
                )
        else:
            if (seasonality % 2) == 0:
                stl = STL(
                    series.squeeze(), period=periodicity, seasonal=seasonality + 1
                )
            else:
                stl = STL(series.squeeze(), period=periodicity, seasonal=seasonality)
    else:
        if (seasonality % 2) == 0:
            stl = STL(series.squeeze(), seasonal=seasonality + 1)
        else:
            stl = STL(series.squeeze(), seasonal=seasonality)

    res = stl.fit()
    # plt = res.plot()
    # plt.show()
    return series.squeeze() - res.seasonal


def seasonal_forecast(
    x_train,
    x_test,
    y_train,
    forecast_horizon,
    periodicity=1,
    seasonality=1,
    decomposed=False,
    alpha=1.96,
):
    """
    Train and validate a seasonal naive timeseries model with the given seasonality

    Fit periodic forecast over the train data, calculate the standard deviation of residuals for
    each horizon over the horizon. Output the periodic forecast generated using the test
    data and the residual_std (upper and lower intervals) for each hour.

    Parameters
    ----------
    x_train : ndarray
        the input training dataset (Shape: (length of training data,forecast_horizon))
    x_test : ndarray
        the input test dataset (Shape: (length of test data,forecast_horizon))
    y_train : ndarray
        The targets (actual values) of the training dataset. (Shape: (length of training data,forecast_horizon))
    forecast_horizon : int
        Number of future steps to be forecasted
    periodicity : int, default = 1
        The periodicity of the sequence.
    seasonality : int, default = 1
        Length of the seasonal smoother. Must be an odd integer, and should normally be >= 7
    decomposed : bool, default = False
        If True, use seasonal decomposition
    alpha : float, default = 1.96
        Measure to adjust confidence interval. Default is set to 1.96, which equals to 95% confidence

    Returns
    -------
    ndarray
        Expected forecast values for each test sample over the forecast horizon.
        (Shape: (len(x_test),forecast_horizon))
    ndarray
        The upper interval for the given forecasts. (Shape: (len(x_test),forecast_horizon))
    ndarray
        The lower interval for the  forecasts. (Shape: (len(x_test),forecast_horizon))
    """

    print(
        "Train a seasonal naive timeseries model with seasonality=", seasonality, "..."
    )

    naive_forecast = np.zeros(shape=(len(x_train), forecast_horizon))
    # difference between the observations and the corresponding fitted values
    residuals = np.zeros(shape=(len(x_train), forecast_horizon))

    if decomposed:
        for j in range(forecast_horizon):
            x_train[:, j] = seasonal_decomposed(
                np.reshape(x_train[:, j], (len(x_train), 1)), periodicity, seasonality
            )
            x_test[:, j] = seasonal_decomposed(
                np.reshape(x_test[:, j], (len(x_test), 1)), periodicity, seasonality
            )

    for i in range(len(x_train)):
        naive_forecast[i, :] = seasonal_naive(
            x_train[i, :], forecast_horizon, periodicity, seasonality
        )
        residuals[i, :] = y_train[i, :] - naive_forecast[i, :]

    # std deviation of residual for each hour(column wise)
    residual_std = np.nanstd(residuals, axis=0)
    expected_value = np.zeros(shape=(len(x_test), forecast_horizon))

    for i in range(len(x_test)):
        expected_value[i, :] = seasonal_naive(
            x_test[i, :], forecast_horizon, periodicity, seasonality
        )

    fc_u = expected_value + alpha * residual_std
    fc_l = expected_value - alpha * residual_std
    print("Training and validating seasonal naive model completed.")

    return expected_value, fc_u, fc_l


def exp_smoothing(
    y_train, y_test, forecast_horizon=1, limit_steps=False, pi_alpha=1.96, online=True
):
    """
    Train an exponential smoothing timeseries model (ETS)

    Parameters
    ----------
    y_train : pandas.DataFrame
        The train values of the target variable
    y_test : pandas.DataFrame
        Values of exogenous features
    forecast_horizon : int, default = 1
        Number of future steps to be forecasted
    limit_steps : int, default = False
        limits the number of simulation/predictions into the future. If False, steps is equal to length of validation set
    pi_alpha : float, default = 1.96
        Measure to adjust confidence interval, default is set to 1.96, which is a 95% PI
    online : bool, default = True
        if True the new observations are used to fit the model again

    Returns
    -------
    ndarray
        Expected forecast values for each test sample over the forecast horizon.
        (Shape: (len(y_train),forecast_horizon))
    ndarray
        The upper interval for the given forecasts. (Shape: (1,forecast_horizon))
    ndarray
        The lower interval for the  forecasts. (Shape: (1,forecast_horizon))
    """

    print("Train an exponential smoothing timeseries model (ETS)...")
    num_cores = max(multiprocessing.cpu_count() - 2, 1)

    model = ETSModel(
        y_train,
        error="add",
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=7*24,
        dates=y_train.index,
    )
    fit = model.fit()

    def ets_predict(i, fit):
        if online:
            # extend the train-series with observed values as we move forward in the prediction horizon
            # to achieve a receding window prediction
            y_train_i = pd.concat([y_train, y_test.iloc[0:i]])
            model = ETSModel(
                y_train_i,
                error="add",
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=7 * 24,
                dates=y_train_i.index,
            )
            fit = model.fit()
        # There are several different ETS methods available:
        #  - forecast: makes out of sample predictions
        #  - predict: in sample and out of sample predictions
        #  - simulate: runs simulations of the statespace model
        #  - get_prediction: in sample and out of sample predictions, as well as prediction intervals
        pred = fit.get_prediction(
            start=y_test.index[i], end=y_test.index[i + forecast_horizon - 1]
        ).summary_frame()  # with: method = 'simulated', simulate_repetitions=100 we can simulate the PI's
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

    expected_value, fc_u, fc_l = zip(
        *Parallel(
            n_jobs=min(num_cores, len(test_period)), mmap_mode="c", temp_folder="/tmp"
        )(
            delayed(ets_predict)(i, fit)
            for i in test_period
            if i + forecast_horizon <= len(y_test)
        )
    )

    print("Training and validating ETS model completed.")
    return np.asarray(expected_value), np.asarray(fc_u), np.asarray(fc_l)
