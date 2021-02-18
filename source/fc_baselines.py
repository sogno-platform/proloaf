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
Baseline Functions
"""
import os
import sys
import argparse

import pandas as pd
import torch
import matplotlib.pyplot as plt
import contextlib

torch.set_printoptions(linewidth=120) # Display option for output
torch.set_grad_enabled(True)

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)

import plf_util.datatuner as dt
import plf_util.eval_metrics as metrics
import plf_util.baseline_forecasts as baselines

from plf_util.config_util import read_config, parse_with_loss

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

def main(infile, target_id):
    sarimax_model = None
    # Read load data
    df = pd.read_csv(infile, sep=';',index_col=0)
    dt.fill_if_missing(df)
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

        if SCALE_DATA:
            df, _ = dt.scale_all(df, **PAR)

        df_train = df.iloc[:int(PAR['train_split'] * len(df))]
        df_val = df.iloc[int(PAR['train_split'] * len(df)):int(PAR['validation_split'] * len(df))]

        x_train = dt.extract(df_train.iloc[:-PAR['forecast_horizon']], PAR['forecast_horizon'])
        x_val = dt.extract(df_val.iloc[:-PAR['forecast_horizon']], PAR['forecast_horizon'])
        y_train = dt.extract(df_train.iloc[PAR['forecast_horizon']:], PAR['forecast_horizon'])
        y_val = dt.extract(df_val.iloc[PAR['forecast_horizon']:], PAR['forecast_horizon'])

        target_column = df.columns.get_loc(target)
        x_train_1D = x_train[:,:,target_column]
        x_val_1D = x_val[:,:,target_column]
        y_train_1D = y_train[:,:,target_column]
        y_val_1D = y_val[:,:,target_column]

        if 'sarimax' in CALC_BASELINES:
            exog_train_1D = pd.concat([x_train[:, :, enc_features], y_train[:, :, dec_features]], axis=1)
            exog_val_1D = pd.concat([x_val[:, :, enc_features], y_val[:, :, dec_features]], axis=1)
            exog_train_1D = exog_train_1D.columns.drop(target)
            exog_val_1D = exog_val_1D.columns.drop(target)
        else:
            exog_train_1D = None
            exog_val_1D = None

        if 'sarima' or 'sarimax' in CALC_BASELINES:
            sarima_jobs = 1
            # TODO: count how often 'arima' is in the list and go through each option
            # (like seasonal_order = (0,0,0,0) for Arima and so on...
            sarimax_model = baselines.load_SARIMAX(OUTDIR)
            if sarimax_model == None: #no previous model found in output directory, could give a print-info here
                sarimax_model,_,_ = baselines.auto_sarimax_wrapper(endog=x_train_1D, exog=exog_train_1D,
                                                                   order=ORDER, seasonal_order=sORDER,
                                                                   trend='n', grid_search = PAR['exploration'])
            #forecast = model_fit.predict(start=start_index, end=end_index)
            SARIMAX_expected_values, SARIMAX_y_pred_upper, SARIMAX_y_pred_lower = \
                baselines.make_forecasts(x_train_1D, y_val_1D, exog_train_1D, exog_val_1D, sarimax_model,
                                         PAR['forecast_horizon'], limit_steps=NUM_PRED, pi_alpha = ALPHA)
            print('Training and validating (S)ARIMA(X) completed.')

        #Naïve
        print('Train a naive (persistence) timeseries model...')
        naive_expected_values, naive_y_pred_upper, naive_y_pred_lower  = \
            baselines.persist_forecast(x_train_1D, x_val_1D, y_train_1D, PAR['forecast_horizon'], alpha=ALPHA)
        print('Training and validating naive (persistence) model completed.')

        #SNaïve
        print('Train a seasonal naive timeseries model with seasonality%5d...' % (SEASONALITY))
        s_naive_expected_values, s_naive_y_pred_upper, s_naive_y_pred_lower = \
            baselines.seasonal_forecast(x_train_1D, x_val_1D, y_train_1D, PAR['forecast_horizon'],
                                        seasonality=SEASONALITY, alpha=ALPHA)
        print('Training and validating seasonal naive model completed./n')

        #decomposition(+ any#model)
        print('Train a naive timeseries model cleared with seasonal decomposition (STL)...')
        sd_naive_expected_values, sd_naive_y_pred_upper, sd_naive_y_pred_lower  = \
            baselines.persist_forecast(x_train_1D, x_val_1D, y_train_1D, PAR['forecast_horizon'],
                                        decomposed=True, alpha=ALPHA)
        print('Training and validating naive (STL) completed.')

        # Exponential smoothing
        if 'ets' in CALC_BASELINES:
            print('Train an exponential smoothing timeseries model (ETS)...')
            with contextlib.redirect_stdout(None):
                ets_expected_values, ets_y_pred_upper, ets_y_pred_lower = \
                    baselines.exp_smoothing(df_train[target], df_val[target], PAR['forecast_horizon'],
                                            limit_steps=NUM_PRED, update = True)
            print('Training and validating ETS model completed.')
        #GARCH
        if ('sarima' and 'garch') or ('sarimax' and 'garch') in CALC_BASELINES:
            print('Train a General autoregressive conditional heteroeskedasticity (GARCH) model...')
            GARCH_expected_values = SARIMAX_expected_values
            GARCH_y_pred_upper, GARCH_y_pred_lower = \
                baselines.GARCH_predictioninterval(SARIMAX_expected_values, y_val, target,
                                                   PAR['forecast_horizon'], alpha=ALPHA)
            print('Training and validating GARCH model completed.')

        #Dynamic linear models
        #TBATS
        #Prophet
        #NNETAR
        #==========================Evaluate ALL==========================
        mse, rmse, mase, rae, mae, sharpness, coverage, mis = \
            baselines.eval_forecast(SARIMAX_expected_values, y_val, PAR['target_id'],
                                    SARIMAX_y_pred_upper, SARIMAX_y_pred_lower, total=True)
        rmse_horizon, sharpness_horizon, coverage_horizon, mis_horizon = \
            baselines.eval_forecast(SARIMAX_expected_values, y_val, PAR['target_id'],
                                    SARIMAX_y_pred_upper, SARIMAX_y_pred_lower, total=False)
        print(metrics.results_table(['SARIMAX-'+ARGS.station],  mse.numpy(), rmse.numpy(), mase.numpy(),
                                    rae.numpy(), mae.numpy(), sharpness.numpy(), coverage.numpy(), mis.numpy()))

        #plot metrics
        metrics.plot_metrics(rmse_horizon.numpy(), sharpness_horizon.numpy(), coverage_horizon.numpy(),
                             mis_horizon.numpy(), OUTDIR+'baseline_', 'baseline-metrics')

        testhours = [0, 12, 24, 48, 100, 112]

        for i in testhours:
            if testhours[i]+PAR['forecast_horizon']<len(SARIMAX_expected_values):
                targets = y_val[i][PAR['target_id']].values
                expected_values = SARIMAX_expected_values[i].to_numpy()
                y_pred_upper = SARIMAX_y_pred_upper[i].to_numpy()
                y_pred_lower = SARIMAX_y_pred_lower[i].to_numpy()
                hours = pd.to_datetime(y_val[i].index).to_series()
                metrics.evaluate_hours(targets, expected_values, y_pred_upper, y_pred_lower, i, OUTDIR+'baseline_',
                                       PAR['cap_limit'], hours)

    except KeyboardInterrupt:
        print()
        print('manual interrupt')
    finally:
        if sarimax_model:
            baselines.save_SARIMAX(OUTDIR, sarimax_model, SARIMAX_expected_values, save_predictions = True)

if __name__ == '__main__':
    ARGS, LOSS_OPTIONS = parse_with_loss()
    PAR = read_config(model_name=ARGS.station, config_path=ARGS.config, main_path=MAIN_PATH)
    OUTDIR = os.path.join(MAIN_PATH, PAR['evaluation_path'])
    SCALE_DATA = False # True
    LIMIT_HISTORY = True
    NUM_PRED = 10
    SLIDING_WINDOW = 1
    CALC_BASELINES =['sarimax','ets','garch'] #
    DAY_FIRST = True
    ORDER = (2, 1, 3)
    sORDER = (2, 1, 1, 24)
    SEASONALITY=24
    ALPHA = 1.96
    main(infile=os.path.join(MAIN_PATH, PAR['data_path']), target_id=PAR['target_id'])