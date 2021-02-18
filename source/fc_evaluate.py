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

import plf_util.datatuner as dt
import plf_util.tensorloader as dl
import plf_util.eval_metrics as metrics
import plf_util.modelhandler as mh
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from plf_util.config_util import read_config, parse_basic

# TODO: find workaround for PICP numpy issue
import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
import sys
import os

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)

warnings.filterwarnings('ignore')

def shape_model_input(df, columns_p, horizon_p, horizon_f):
    # shape input data that is measured in the Past and can be fetched from UDW/LDW
    x_p = dt.extract(df[columns_p].iloc[:-horizon_f, :], horizon_p)
    # shape input data that is known for the Future, here take perfect hourly temp-forecast
    x_f = dt.extract(df.drop(columns_p, axis=1).iloc[horizon_p:, :], horizon_f)
    # shape y
    y = dt.extract(df[[target_id]].iloc[horizon_p:, :], horizon_f)
    return x_p, x_f, y

if __name__ == '__main__':
    ARGS = parse_basic()
    PAR = read_config(model_name=ARGS.station, config_path=ARGS.config, main_path=MAIN_PATH)

    # DEFINES
    torch.manual_seed(1)
    INFILE = os.path.join(MAIN_PATH, PAR['data_path'])  # input factsheet

    # after training
    INMODEL = os.path.join(MAIN_PATH, PAR['output_path'], PAR['model_name'])
    OUTDIR = os.path.join(MAIN_PATH, PAR['evaluation_path'])

    target_id = PAR['target_id']
    DEVICE = 'cpu'
    SPLIT_RATIO = PAR['validation_split']

    HISTORY_HORIZON = PAR['history_horizon']
    FORECAST_HORIZON = PAR['forecast_horizon']
    feature_groups = PAR['feature_groups']

    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    # Read load data
    df = pd.read_csv(INFILE, sep=';')
    df = dt.fill_if_missing(df)

    if ('target_list' in PAR):
        if(PAR['target_list'] is not None):
            df[target_id] = df[PAR['target_list']].sum(axis=1)

    # reload trained NN
    with torch.no_grad():
        net = torch.load(INMODEL, map_location=torch.device(DEVICE))  # mapping to CPU

        df_new, _ = dt.scale_all(df, **PAR)

        target_index = df_new.columns.get_loc(target_id)

        split_index = int(len(df_new.index) * SPLIT_RATIO)
        train_df = df_new.iloc[0:split_index]
        test_df = df_new.iloc[split_index:]

        train_data_loader = dl.make_dataloader(train_df, target_id, PAR['encoder_features'], PAR['decoder_features'],
                                              history_horizon=PAR['history_horizon'], forecast_horizon=PAR['forecast_horizon'],
                                              shuffle=False).to(DEVICE)
        test_data_loader = dl.make_dataloader(test_df, target_id, PAR['encoder_features'], PAR['decoder_features'],
                                              history_horizon=PAR['history_horizon'], forecast_horizon=PAR['forecast_horizon'],
                                              shuffle=False).to(DEVICE)

        # check performance
        horizon = test_data_loader.dataset.targets.shape[1]
        number_of_targets = test_data_loader.dataset.targets.shape[2]

        record_targets, record_output = mh.get_prediction(net, test_data_loader, horizon, number_of_targets)

        net.eval()
        # TODO: check model type (e.g gnll)
        criterion = net.criterion
        # get metrics parameters
        y_pred_upper, y_pred_lower, record_expected_values = mh.get_pred_interval(record_output, criterion, record_targets)

        # rescale(test_output, test_targets)
        # dt.rescale_manually(..)

        # calculate the metrics
        mse_horizon = metrics.mse(record_targets, [record_expected_values], total=False)
        rmse_horizon = metrics.rmse(record_targets, [record_expected_values], total=False)
        sharpness_horizon = metrics.sharpness(None, [y_pred_upper, y_pred_lower], total=False)
        coverage_horizon = metrics.picp(record_targets, [y_pred_upper,
                                                        y_pred_lower], total=False)
        mis_horizon = metrics.mis(record_targets, [y_pred_upper, y_pred_lower], alpha=0.05, total=False)

        # collect metrics by disregarding the development over the horizon
        mse = metrics.mse(record_targets, [record_expected_values])
        rmse = metrics.rmse(record_targets, [record_expected_values])
        mase = metrics.mase(record_targets, [record_expected_values], 7 * 24, train_data_loader.dataset.targets)
        rae = metrics.rae(record_targets, [record_expected_values])
        mae = metrics.nmae(record_targets, [record_expected_values])

        sharpness = metrics.sharpness(None, [y_pred_upper, y_pred_lower])
        coverage = metrics.picp(record_targets, [y_pred_upper, y_pred_lower])
        mis = metrics.mis(record_targets, [y_pred_upper, y_pred_lower], alpha=0.05)

        # plot metrics
        metrics.plot_metrics(rmse_horizon.detach().numpy(), sharpness_horizon.detach().numpy(), coverage_horizon.detach().numpy(), mis_horizon.detach().numpy(), OUTDIR, 'metrics-evaluation')

        # plot forecast for sample days
        if 'ci_tests' in PAR['data_path']:
            testhours = [0, 12]
        else:
            testhours = [0, 12, 24, 48, 100, 112]

        actual_time = pd.to_datetime(df.loc[split_index:, 'Time'])
        for i in testhours:
            hours = actual_time.iloc[i:i + FORECAST_HORIZON]
            metrics.evaluate_hours(record_targets[i].detach().numpy(), record_expected_values[i].detach().numpy(), y_pred_upper[i].detach().numpy(), y_pred_lower[i].detach().numpy(), i, OUTDIR, PAR['cap_limit'], hours)

        target_stations = [PAR['model_name']]
        print(metrics.results_table(target_stations, mse.cpu().numpy(), rmse.cpu().numpy(), mase.cpu().numpy(), rae.cpu().numpy(),
                          mae.cpu().numpy(), sharpness.cpu().numpy(), coverage.cpu().numpy(), mis.cpu().numpy(), save_to_disc=OUTDIR))

    print('Done!!')
