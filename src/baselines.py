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

# -*- coding: utf-8 -*-
"""
Create baseline forecasts and evaluate them. 

First, load and prepare training and validation data. Then optionally create any/all of the
following baseline forecasts:

- A forecast using a created or loaded fitted (S)ARIMA(X) model
- A simple naive forecast
- A seasonal naive forecast
- A forecast using a naive STL model
- A forecast using an ETS model
- A forecast using a GARCH model
Finally evaluate all baseline forecasts using several different metrics and plot the results
"""

import os
import sys
import argparse

import pandas as pd
import torch
import proloaf.plot as plot

torch.set_printoptions(linewidth=120) # Display option for output
torch.set_grad_enabled(True)

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)

import proloaf.datahandler as dh
import proloaf.metrics as metrics
import proloaf.baselinehandler as baselines

from proloaf.confighandler import read_config
from proloaf.cli import parse_with_loss
from proloaf.cli import create_event_logger

logger = create_event_logger(__name__)

class flag_and_store(argparse._StoreAction):
    def __init__(self, option_strings, dest, dest_const, const,
                 nargs = 0, **kwargs):
        self.dest_const = dest_const
        if isinstance(const, list):
            self.val = const[1:]
            self.flag = const[0]
        else:
            self.val = []
            self.flag = const
        if isinstance(nargs, int):
            nargs_store = nargs+len(self.val)
        else:
            nargs_store = nargs
        super(flag_and_store, self).__init__(option_strings, dest, const = None,
                                             nargs = nargs_store, **kwargs)
        self.nargs = nargs

    def __call__(self, parser, namespace, values, option_strings = None):
        setattr(namespace, self.dest_const,self.flag)
        if isinstance(values,list):
            self.val.extend(values)
        elif values is not None:
            self.val.append(values)

        if len(self.val) == 1:
            self.val = self.val[0]
        super().__call__(parser, namespace, self.val, option_strings)

def eval_baseline(config, mean_forecast, df_target, upper_PI, lower_PI, load_limit, hours, path, baseline_method='baseline', analyzed_metrics=["mse"],
                  forecast_horizon=1, anchor_adjustment =0):
    """
    Evaluate a baseline forecast using several different metrics and plot the results.

    A non-zero value for anchor_adjustment can additionally adjust benchmark forecasts to the
    forecast start time of the ProLoaF RNN method, to facilitate the comparison of sample days

    Parameters
    ----------
    mean_forecast : array-like
        Contains forecasted expected values
    df_target : array-like
        The reference or measured target variable
    upper_PI : array-like
        Upper prediction interval
    lower_PI : array-like
        Lower prediction interval
    load_limit : float
        The cap limit
    hours : array-like
        The actual time from the data set
    baseline_method : string, default = 'baseline'
        The text for the title of the plot
    forecast_horizon : int, default = 1
        Number of future steps to be forecasted
    anchor_adjustment : int, default = 0
        A positive value which is subtracted from the standard evaluation hours (0, 12, 24,
        48, 100, and 112), enabling hours at a fixed offset from the standard ones to be
        evaluated. I.e.:
        - For anchor_adjustment = 3, evaluate forecasts for hours 9, 21, 45, 97, and 109
    """
    #forecasts,
    # endog_val,
    # upper_limits,
    # lower_limits,
    # model_name,
    # path,
    # config,
    # analyzed_metrics
    return baselines.eval_forecast(
        forecasts=mean_forecast,
        endog_val=df_target,
        upper_limits=upper_PI,
        lower_limits=lower_PI,
        model_name=baseline_method,
        path=path,
        config=config,
        analyzed_metrics=analyzed_metrics)

def main(infile, target_id):
    sarimax_model = None
    # Read load data
    df = pd.read_csv(infile, sep=';',index_col=0)
    dh.fill_if_missing(df, periodicity=SEASONALITY)
    #df['Time'] = pd.to_datetime(df['Time'])
    #df = df.set_index('Time')
    df = df.asfreq(freq='H')
    time_delta = pd.to_datetime(df.index[1]) - pd.to_datetime(df.index[0])
    timestamps = pd.date_range(start=pd.to_datetime(df.index, dayfirst=DAY_FIRST)[0],
                          end=pd.to_datetime(df.index, dayfirst=DAY_FIRST)[-1],
                          freq=str(time_delta.seconds / 3600) + 'H')
    df.index = timestamps
    if ('col_list' in PAR):
        if(PAR['col_list'] != None):
            df[target_id] = df[PAR['col_list']].sum(axis=1)

    try:
        target = PAR['target_id']
        enc_features = PAR['encoder_features']
        dec_features = PAR['decoder_features']
        #now just make sure that the exogenous features do not include the target itself as endog already does.
        if target in enc_features: enc_features.remove(target)
        if target in dec_features: dec_features.remove(target)

        if SCALE_DATA:
            df, _ = dh.scale_all(df, **PAR)

        df_train = df.iloc[:int(PAR['validation_split'] * len(df))]
        df_val = df.iloc[int(PAR['validation_split'] * len(df)):]

        x_train = dh.extract(df_train.iloc[:-PAR['forecast_horizon']], PAR['forecast_horizon'])
        x_val = dh.extract(df_val.iloc[:-PAR['forecast_horizon']], PAR['forecast_horizon'])
        y_train = dh.extract(df_train.iloc[PAR['forecast_horizon']:], PAR['forecast_horizon'])
        y_val = dh.extract(df_val.iloc[PAR['forecast_horizon']:], PAR['forecast_horizon'])

        target_column = df.columns.get_loc(target[0])

        x_train_1D = x_train[:,:,target_column]
        x_val_1D = x_val[:,:,target_column]
        y_train_1D = y_train[:,:,target_column]

        mean_forecast = []
        upper_PI = []
        lower_PI = []
        baseline_method = []

        if EXOG:
            df_exog_train = df_train[enc_features+dec_features]
            df_exog_val = df_val[enc_features+dec_features]
        else:# empty exogenous features, to ignore them in following baseline models, can be removed actually...
            df_exog_train = None
            df_exog_val = None

        if 'simple-naive' in CALC_BASELINES:
            # Naïve
            naive_expected_values, naive_y_pred_upper, naive_y_pred_lower = \
                baselines.persist_forecast(x_train_1D, x_val_1D, y_train_1D, PAR['forecast_horizon'], alpha=ALPHA)
            mean_forecast.append(naive_expected_values)
            upper_PI.append(naive_y_pred_upper)
            lower_PI.append(naive_y_pred_lower)
            baseline_method.append('simple-naive')

        if 'seasonal-naive' in CALC_BASELINES:
            # SNaïve
            s_naive_expected_values, s_naive_y_pred_upper, s_naive_y_pred_lower = \
                baselines.seasonal_forecast(x_train_1D, x_val_1D, y_train_1D, PAR['forecast_horizon'], PERIODICITY,
                                            alpha=ALPHA)
            mean_forecast.append(s_naive_expected_values)
            upper_PI.append(s_naive_y_pred_upper)
            lower_PI.append(s_naive_y_pred_lower)
            baseline_method.append('seasonal-naive')

        if 'naive-stl' in CALC_BASELINES:
            # decomposition(+ any#model)
            sd_naive_expected_values, sd_naive_y_pred_upper, sd_naive_y_pred_lower = \
                baselines.persist_forecast(x_train_1D, x_val_1D, y_train_1D, PAR['forecast_horizon'],
                                           periodicity=PERIODICITY, seasonality=SEASONALITY,
                                           decomposed=True, alpha=ALPHA)
            mean_forecast.append(sd_naive_expected_values)
            upper_PI.append(sd_naive_y_pred_upper)
            lower_PI.append(sd_naive_y_pred_lower)
            baseline_method.append('naive-stl')

        #baselines.test_stationarity(df_train[target],maxlag=PERIODICITY*SEASONALITY)
        arima_order = ORDER
        sarima_order = ORDER
        sarima_sorder = sORDER
        if 'arima' in CALC_BASELINES:
            ###############################ARIMA####################################
            arima_model=None
            if(APPLY_EXISTING_MODEL): arima_model = baselines.load_baseline(OUTDIR,name='ARIMA')
            if arima_model == None:
                arima_model,_,_,arima_order,_ = baselines.auto_sarimax_wrapper(
                    endog=df_train[target],
                    order=ORDER,
                    seasonal_order=None,
                    seasonal = False,
                    lag=PERIODICITY*SEASONALITY,
                    grid_search = PAR['exploration'],
                    train_limit=LIMIT_HISTORY
                )
            else: logger.info('Loaded existing fitted ARIMA model from {!s}'.format(OUTDIR))
            ARIMA_expected_values, ARIMA_y_pred_upper, ARIMA_y_pred_lower = \
                baselines.make_forecasts(
                    endog_train = df_train[target],
                    endog_val = df_val[target],
                    fitted = arima_model,
                    forecast_horizon = PAR['forecast_horizon'],
                    train_limit = LIMIT_HISTORY,
                    limit_steps = NUM_PRED,
                    pi_alpha = ALPHA,
                    online = True)
            mean_forecast.append(ARIMA_expected_values)
            upper_PI.append(ARIMA_y_pred_upper)
            lower_PI.append(ARIMA_y_pred_lower)
            baseline_method.append('ARIMA')
            baselines.save_baseline(OUTDIR, arima_model, name='ARIMA',  save_predictions=False)

        if 'sarima' in CALC_BASELINES:
            ##############################SARIMA####################################
            sarima_model=None
            if(APPLY_EXISTING_MODEL): sarima_model = baselines.load_baseline(OUTDIR,name='SARIMA')
            if sarima_model == None:
                sarima_model,_,_,sarima_order,sarima_sorder = baselines.auto_sarimax_wrapper(
                    endog=df_train[target],
                    order=ORDER,
                    seasonal_order=sORDER,
                    seasonal = True,
                    lag=PERIODICITY*SEASONALITY,
                    m=SEASONALITY,
                    train_limit=LIMIT_HISTORY,
                    grid_search = PAR['exploration']
                )
            else: logger.info('Loaded existing fitted SARIMA model from {!s}'.format(OUTDIR))
            SARIMA_expected_values, SARIMA_y_pred_upper, SARIMA_y_pred_lower = \
                baselines.make_forecasts(
                    endog_train = df_train[target],
                    endog_val = df_val[target],
                    fitted = sarima_model,
                    forecast_horizon = PAR['forecast_horizon'],
                    train_limit = LIMIT_HISTORY,
                    limit_steps = NUM_PRED,
                    pi_alpha = ALPHA,
                    online = True
                )
            mean_forecast.append(SARIMA_expected_values)
            upper_PI.append(SARIMA_y_pred_upper)
            lower_PI.append(SARIMA_y_pred_lower)

            baseline_method.append('SARIMA')
            sarima_model.summary()
            baselines.save_baseline(OUTDIR, sarima_model, name='SARIMA', save_predictions=False)

        if 'arimax' in CALC_BASELINES:
            ###############################ARIMAX####################################
            arimax_model = None
            if(APPLY_EXISTING_MODEL): arimax_model = baselines.load_baseline(OUTDIR, name='ARIMAX')
            if arimax_model == None:
                arimax_model,_,_,_,_ = baselines.auto_sarimax_wrapper(
                    endog=df_train[target],
                    exog=df_exog_train,
                    order=arima_order,
                    seasonal_order=None,
                    seasonal=False,
                    train_limit=LIMIT_HISTORY,
                    lag=PERIODICITY*SEASONALITY,
                    grid_search=False
                )
            else: logger.info('Loaded existing fitted ARIMAX model from {!s}'.format(OUTDIR))
            ARIMAX_expected_values, ARIMAX_y_pred_upper, ARIMAX_y_pred_lower = \
                baselines.make_forecasts(
                    endog_train=df_train[target],
                    endog_val=df_val[target],
                    exog = df_exog_train,
                    exog_forecast = df_exog_val,
                    fitted=arimax_model,
                    forecast_horizon=PAR['forecast_horizon'],
                    train_limit=LIMIT_HISTORY,
                    limit_steps=NUM_PRED,
                    pi_alpha=ALPHA,
                    online=True)
            mean_forecast.append(ARIMAX_expected_values)
            upper_PI.append(ARIMAX_y_pred_upper)
            lower_PI.append(ARIMAX_y_pred_lower)
            baseline_method.append('ARIMAX')
            baselines.save_baseline(OUTDIR, arimax_model, name='ARIMAX', save_predictions=False)
        if 'sarimax' in CALC_BASELINES:
            ###############################SARIMAX###################################
            sarimax_model = None
            if (APPLY_EXISTING_MODEL): sarimax_model = baselines.load_baseline(OUTDIR, name='SARIMAX')
            if sarimax_model == None:
                sarimax_model,_,_,_,_ = baselines.auto_sarimax_wrapper(
                    endog=df_train[target],
                    exog=df_exog_train,
                    order=sarima_order,
                    seasonal_order=sarima_sorder,
                    seasonal = True,
                    train_limit=LIMIT_HISTORY,
                    lag=PERIODICITY*SEASONALITY,
                    m=SEASONALITY,
                    grid_search = False
                )
            else: logger.info('Loaded existing fitted SARIMAX model from {!s}'.format(OUTDIR))
            SARIMAX_expected_values, SARIMAX_y_pred_upper, SARIMAX_y_pred_lower = \
                baselines.make_forecasts(
                    endog_train=df_train[target],
                    endog_val=df_val[target],
                    exog = df_exog_train,
                    exog_forecast = df_exog_val,
                    fitted=sarimax_model,
                    forecast_horizon=PAR['forecast_horizon'],
                    train_limit=LIMIT_HISTORY,
                    limit_steps=NUM_PRED,
                    pi_alpha=ALPHA,
                    online=True
                )
            mean_forecast.append(SARIMAX_expected_values)
            upper_PI.append(SARIMAX_y_pred_upper)
            lower_PI.append(SARIMAX_y_pred_lower)
            baseline_method.append('SARIMAX')
            baselines.save_baseline(OUTDIR, sarimax_model, name='SARIMAX', save_predictions=False)

        if 'ets' in CALC_BASELINES:
        # Exponential smoothing
            #with contextlib.redirect_stdout(None):
            train=pd.Series(df_train[target].values.squeeze(), index=df_train[target].index)
            test=pd.Series(df_val[target].values.squeeze(), index=df_val[target].index)
            ets_expected_values, ets_y_pred_upper, ets_y_pred_lower = \
                baselines.exp_smoothing(train, test, PAR['forecast_horizon'],
                                        limit_steps=NUM_PRED, online = True)
            mean_forecast.append(ets_expected_values)
            upper_PI.append(ets_y_pred_upper)
            lower_PI.append(ets_y_pred_lower)
            baseline_method.append('ets')

        #GARCH
        if 'garch' in CALC_BASELINES:
            mean = None
            p = 1
            q = 1
            if 'arima' in CALC_BASELINES or 'sarima' in CALC_BASELINES:
                # e.g. Garch (p'=1,q'=1) has the Arch order 1(past residuals squared) and the Garch order 1(past variance:=sigma^2)
                # in comparison, ARMA (p=1,p=1) has AR order 1(past observations) and MA order 1(past residuals)
                # --> this is why we just use the computed mean by sarimax (called SARIMAX_expected_values here)

                # why not more than one Garch component? when forecasting market returns e.g.,
                # all the effects of conditional variance of t-2 are contained in the conditional variance t-1
                # p: is arch order and q ist garch order
                # arch component is nearly equivalent to the MA component of SARIMAX_expected_values here --> p' = q
                # garch component is nearly equivalent to the AR component of SARIMAX -->q'= p
                mean = ARIMA_expected_values
                #p= arimax_model.model_orders['ma']-->lagges variances
                #q =arimax_model.model_orders['ar']--> lagged residuals
            GARCH_expected_values, GARCH_y_pred_upper, GARCH_y_pred_lower = \
                baselines.GARCH_predictioninterval(df_train[target], df_val[target], PAR['forecast_horizon'],
                                                   mean_forecast=mean, p=p, q=q, alpha=ALPHA, limit_steps=NUM_PRED,
                                                   periodicity=PERIODICITY)
            mean_forecast.append(mean)
            upper_PI.append(GARCH_y_pred_upper)
            lower_PI.append(GARCH_y_pred_lower)
            baseline_method.append('garch')

        #further options possible, checkout this blog article: https://towardsdatascience.com/an-overview-of-time-series-forecasting-models-a2fa7a358fcb
        #Dynamic linear models
        #TBATS
        #Prophet
        #----------
        #randomforest --already existing in notebook
        #gradboosting --already existing in old project

        #==========================Evaluate ALL==========================
        #for a proper benchmark with the rnn we need to consider that the rnn starts at anchor=t0(first data-entry)+history horizon
        #our baseline forecasts start at anchor=t0+forecast_horizon
        #so we have a shift in the start of the forecast simulation of about shift=Par[forecast-horizon]-Par[hist-horizon]
        #shift=PAR['forecast_horizon']-PAR['history_horizon']
        analyzed_metrics = [
            "mse",
            "rmse",
            "sharpness",
            "picp",
            "rae",
            "mae",
            "mis",
            "mase",
            "pinball_loss",
            "residuals"
        ]
        results=pd.DataFrame(index=analyzed_metrics)
        results_per_timestep = {}
        true_values= torch.zeros([len(mean_forecast), NUM_PRED, PAR['forecast_horizon']])
        forecasts=torch.zeros([len(mean_forecast), NUM_PRED, PAR['forecast_horizon']])
        upper_limits=torch.zeros([len(mean_forecast), NUM_PRED, PAR['forecast_horizon']])
        lower_limits=torch.zeros([len(mean_forecast), NUM_PRED, PAR['forecast_horizon']])
        for i, mean_forecast in enumerate(mean_forecast):
            if len(mean_forecast) - PAR['forecast_horizon'] == len(y_val):
                results[baseline_method[i]], results_per_timestep[baseline_method[i]], true_values[i],forecasts[i],upper_limits[i],lower_limits[i] = \
                    eval_baseline(
                        config=PAR,
                        mean_forecast=mean_forecast[PAR['forecast_horizon']:PAR['forecast_horizon']+NUM_PRED],
                        df_target=y_val[:NUM_PRED,:,target_column],
                        upper_PI=upper_PI[i][PAR['forecast_horizon']:PAR['forecast_horizon']+NUM_PRED,:],
                        lower_PI=lower_PI[i][PAR['forecast_horizon']:PAR['forecast_horizon']+NUM_PRED,:],
                        load_limit=PAR['cap_limit'],
                        hours = pd.to_datetime(df_val[PAR['forecast_horizon']:].index).to_series(),
                        path=OUTDIR,
                        baseline_method='test'+baseline_method[i],
                        forecast_horizon= PAR['forecast_horizon'],
                        analyzed_metrics=analyzed_metrics,
                    )
            else:
                results[baseline_method[i]], results_per_timestep[baseline_method[i]], true_values[i],forecasts[i],upper_limits[i],lower_limits[i] = \
                    eval_baseline(
                        config=PAR,
                        mean_forecast=mean_forecast[:NUM_PRED],
                        df_target=y_val[:NUM_PRED, :, target_column],
                        upper_PI=upper_PI[i][:NUM_PRED],
                        lower_PI=lower_PI[i][:NUM_PRED],
                        load_limit=PAR['cap_limit'],
                        hours=pd.to_datetime(df_val.index).to_series(),
                        path=OUTDIR,
                        baseline_method='test'+baseline_method[i],
                        forecast_horizon=PAR['forecast_horizon'],
                        analyzed_metrics=analyzed_metrics,
                    )
        results_per_timestep_per_baseline = pd.concat(
            results_per_timestep.values(),
            keys=results_per_timestep.keys(),
            axis=1
        )

        # plot metrics
        # rmse_horizon, sharpness_horizon, coverage_horizon, mis_horizon, OUTPATH, title
        plot.plot_metrics(
            results_per_timestep_per_baseline.xs('rmse', axis=1, level=1, drop_level=False),
            results_per_timestep_per_baseline.xs('sharpness', axis=1, level=1, drop_level=False),
            results_per_timestep_per_baseline.xs('picp', axis=1, level=1, drop_level=False),
            results_per_timestep_per_baseline.xs('mis', axis=1, level=1, drop_level=False),
            OUTDIR + "baselines",
        )
        plot.plot_hist(
            targets=true_values,
            expected_values=forecasts,
            y_pred_upper=upper_limits,
            y_pred_lower=lower_limits,
            analyzed_metrics=["residuals"],
            sample_frequency=24,
            save_to_disc=OUTDIR + "baselines",
            bins=80,
        )

        logger.info(
            'results: \n {!s}'.format(
                metrics.results_table(
                    "baselines",
                    results,
                    save_to_disc=OUTDIR,
                )
            )
        )


    except KeyboardInterrupt:
        logger.info('\nmanual interrupt')
    finally:
        logger.info(','.join(baseline_method) +' fitted and evaluated')

if __name__ == '__main__':
    ARGS, LOSS_OPTIONS = parse_with_loss()
    PAR = read_config(model_name=ARGS.station, config_path=ARGS.config, main_path=MAIN_PATH)
    OUTDIR = os.path.join(MAIN_PATH, PAR['evaluation_path'])
    SCALE_DATA = True
    LIMIT_HISTORY = 300#PAR['history_horizon']
    NUM_PRED = 10
    SLIDING_WINDOW = 1
    CALC_BASELINES =['simple-naive', 'seasonal-naive','ets','garch','naive-stl','arima','arimax','sarima','sarimax']
    DAY_FIRST = True
    ORDER = (3, 1, 0)
    sORDER = (2, 0, 0, 24)
    SEASONALITY=24
    PERIODICITY=7
    ALPHA = 1.96
    EXOG = True
    APPLY_EXISTING_MODEL = False
    main(infile=os.path.join(MAIN_PATH, PAR['data_path']), target_id=PAR['target_id'])
