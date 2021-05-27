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
Evaluate a previously trained model.

Run the trained net on test data and evaluate the result. 
Provide several graphs as outputs that show the performance of the trained model.

Notes
-----
- The output consists of the actual load prediction graph split up into multiple svg images and 
a metrics plot containing information about the model's performance.
- The output path for the graphs can be changed under the "evaluation_path": option 
in the corresponding config file.

"""

import plf_util.datatuner as dt
import plf_util.tensorloader as dl
import plf_util.eval_metrics as metrics
import plf_util.modelhandler as mh
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler


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

from plf_util.config_util import read_config, parse_basic

if __name__ == "__main__":
    ARGS = parse_basic()
    PAR = read_config(
        model_name=ARGS.station, config_path=ARGS.config, main_path=MAIN_PATH
    )

    # DEFINES
    torch.manual_seed(1)
    INFILE = os.path.join(MAIN_PATH, PAR["data_path"])  # input factsheet

    # after training
    INMODEL = os.path.join(MAIN_PATH, PAR["output_path"], PAR["model_name"])
    OUTDIR = os.path.join(MAIN_PATH, PAR["evaluation_path"])

    target_id = PAR["target_id"]
    DEVICE = "cpu"
    SPLIT_RATIO = PAR["validation_split"]

    HISTORY_HORIZON = PAR["history_horizon"]
    FORECAST_HORIZON = PAR["forecast_horizon"]
    feature_groups = PAR["feature_groups"]

    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    # Read load data
    df = pd.read_csv(INFILE, sep=";")
    df = dt.fill_if_missing(df)

    if "target_list" in PAR:
        if PAR["target_list"] is not None:
            df[target_id] = df[PAR["target_list"]].sum(axis=1)

    # reload trained NN
    with torch.no_grad():
        net = torch.load(INMODEL, map_location=torch.device(DEVICE))  # mapping to CPU

        df_new, _ = dt.scale_all(df, scalers=net.scalers, **PAR)
        df_new.index = df['Time']
        target_index = df_new.columns.get_loc(target_id)

        split_index = int(len(df_new.index) * SPLIT_RATIO)
        train_df = df_new.iloc[0:split_index]
        test_df = df_new.iloc[split_index:]

        test_data_loader = dl.make_dataloader(
            test_df,
            target_id,
            PAR["encoder_features"],
            PAR["decoder_features"],
            history_horizon=PAR["history_horizon"],
            forecast_horizon=PAR["forecast_horizon"],
            shuffle=False,
        ).to(DEVICE)

        # check performance
        horizon = test_data_loader.dataset.targets.shape[1]
        number_of_targets = test_data_loader.dataset.targets.shape[2]

        record_targets, record_output = mh.get_prediction(
            net, test_data_loader, horizon, number_of_targets
        )

        net.eval()
        # TODO: check model type (e.g gnll)
        criterion = net.criterion
        # get metrics parameters
        y_pred_upper, y_pred_lower, record_expected_values = mh.get_pred_interval(
            record_output, criterion, df[target_id]
        )

        # rescale(test_output, test_targets)
        # dt.rescale_manually(..)

        # calculate the metrics
        mse_horizon = metrics.mse(record_targets, [record_expected_values], total=False)
        rmse_horizon = metrics.rmse(
            record_targets, [record_expected_values], total=False
        )
        sharpness_horizon = metrics.sharpness(
            None, [y_pred_upper, y_pred_lower], total=False
        )
        coverage_horizon = metrics.picp(
            record_targets, [y_pred_upper, y_pred_lower], total=False
        )
        mis_horizon = metrics.mis(
            record_targets, [y_pred_upper, y_pred_lower], alpha=0.05, total=False
        )
        # collect metrics by disregarding the development over the horizon
        mse = metrics.mse(record_targets, [record_expected_values])
        rmse = metrics.rmse(record_targets, [record_expected_values])
        mase = metrics.mase(record_targets, [record_expected_values], 7*24)
        rae = metrics.rae(record_targets, [record_expected_values])
        mae = metrics.mae(record_targets, [record_expected_values])
        qs = metrics.pinball_loss(record_targets, [y_pred_upper, y_pred_lower], [0.025, 0.975])

        sharpness = metrics.sharpness(None, [y_pred_upper, y_pred_lower])
        coverage = metrics.picp(record_targets, [y_pred_upper, y_pred_lower])
        mis = metrics.mis(record_targets, [y_pred_upper, y_pred_lower], alpha=0.05)

        # plot metrics
        metrics.plot_metrics(
            rmse_horizon.detach().numpy(),
            sharpness_horizon.detach().numpy(),
            coverage_horizon.detach().numpy(),
            mis_horizon.detach().numpy(),
            OUTDIR,
            "metrics-evaluation",
        )

        # plot forecast for sample days
        if "ci_tests" in PAR["data_path"]:
            testhours = [0, 12]
        else:
            testhours = [0, 12, 24, 48, 100, 112]

        actual_time = pd.to_datetime(df.loc[PAR['history_horizon']+split_index:, "Time"])
        for i in testhours:
            hours = actual_time.iloc[i : i + FORECAST_HORIZON]
            metrics.evaluate_hours(
                record_targets[i].detach().numpy(),
                record_expected_values[i].detach().numpy(),
                y_pred_upper[i].detach().numpy(),
                y_pred_lower[i].detach().numpy(),
                i,
                OUTDIR,
                PAR["cap_limit"],
                hours,
            )

        #TODO: wrap up all following sample tests
        target_stations = [PAR["model_name"]]
        print(
            metrics.results_table(
                target_stations,
                mse.detach().numpy(),
                rmse.detach().numpy(),
                mase.detach().numpy(),
                rae.detach().numpy(),
                mae.detach().numpy(),
                sharpness.detach().numpy(),
                coverage.detach().numpy(),
                mis.detach().numpy(),
                qs.cpu().numpy(), 
                save_to_disc=OUTDIR,
            )
        )
       # BOXPLOTS
        mse_per_sample = [metrics.mse(record_targets[i], [record_expected_values[i]]) for i, value in
                          enumerate(record_targets)]
        rmse_per_sample = [metrics.rmse(record_targets[i], [record_expected_values[i]]) for i, value in
                          enumerate(record_targets)]
        sharpness_per_sample = [metrics.sharpness(None, [y_pred_upper[i],y_pred_lower[i]]) for i, value in
                          enumerate(record_targets)]
        coverage_per_sample = [metrics.picp(record_targets[i], [y_pred_upper[i],y_pred_lower[i]]) for i, value in
                          enumerate(record_targets)]
        rae_per_sample = [metrics.rae(record_targets[i], [record_expected_values[i]]) for i, value in
                          enumerate(record_targets)]
        mae_per_sample = [metrics.mae(record_targets[i], [record_expected_values[i]]) for i, value in
                          enumerate(record_targets)]
        mis_per_sample = [metrics.mis(record_targets[i], [y_pred_upper[i], y_pred_lower[i]], alpha=0.05) for i, value in
                          enumerate(record_targets)]
        mase_per_sample = [metrics.mase(record_targets[i], [record_expected_values[i]], 0,
                                        insample_target=record_targets.roll(7*24,0)[i]) for i, value in
                          enumerate(record_targets)]

        residuals_per_sample = [metrics.residuals(record_targets[i],
                                            [record_expected_values[i]]) for i, value in enumerate(record_targets)]
        quantile_score_per_sample = [metrics.pinball_loss(record_targets[i], [y_pred_upper[i], y_pred_lower[i]], [0.025,0.975])
                                  for i, value in enumerate(record_targets)]

        metrics_per_sample = pd.DataFrame(mse_per_sample).astype('float')
        metrics_per_sample['MSE'] = pd.DataFrame(mse_per_sample).astype('float')
        metrics_per_sample['RMSE'] = pd.DataFrame(rmse_per_sample).astype('float')
        metrics_per_sample['Sharpness'] = pd.DataFrame(sharpness_per_sample).astype('float')
        metrics_per_sample['PICP'] = pd.DataFrame(coverage_per_sample).astype('float')
        metrics_per_sample['RAE'] = pd.DataFrame(rae_per_sample).astype('float')
        metrics_per_sample['MAE'] = pd.DataFrame(mae_per_sample).astype('float')
        metrics_per_sample['MASE'] = pd.DataFrame(mase_per_sample).astype('float')
        metrics_per_sample['MIS'] = pd.DataFrame(mis_per_sample).astype('float')
        metrics_per_sample['Quantile Score'] = pd.DataFrame(quantile_score_per_sample).astype('float')
        metrics_per_sample['Residuals'] = pd.DataFrame(residuals_per_sample).astype('float')

        ax1 = metrics_per_sample.iloc[::24].boxplot(column=['RMSE', 'MASE', 'Quantile Score'],
                                                    color=dict(boxes='k', whiskers='k', medians='k', caps='k'),
                                                    figsize=(8.5, 10), fontsize=24)
        ax1.set_xlabel('Mean error per sample on 40h prediction horizon', fontsize=24)
        ax1.set_ylabel('Error measure on scaled data', fontsize=24)
        ymin, ymax = -0.01, 1.5
        ax1.set_ylim([ymin, ymax])
        plt.show()
    print('Done!!')
