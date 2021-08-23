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

import utils.datahandler as dh
import utils.tensorloader as dl
import utils.metrics as metrics
import utils.plot as plot
import utils.modelhandler as mh

# TODO: find workaround for PICP numpy issue
import pandas as pd
import torch
import warnings
import sys
import os

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)
warnings.filterwarnings("ignore")

from utils.confighandler import read_config
from utils.cli import parse_basic

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

    target_stations = PAR["model_name"]

    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    # Read load data
    df = pd.read_csv(INFILE, sep=";")
    df = dh.fill_if_missing(df)

    if "target_list" in PAR:
        if PAR["target_list"] is not None:
            df[target_id] = df[PAR["target_list"]].sum(axis=1)

    # reload trained NN
    with torch.no_grad():
        net = torch.load(INMODEL, map_location=torch.device(DEVICE))  # mapping to CPU

        df_new, _ = dh.scale_all(df, scalers=net.scalers, **PAR)
        df_new.index = df["Time"]

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
            net,
            test_data_loader,
            horizon,
            number_of_targets
        )

        net.eval()
        # TODO: check model type (e.g gnll)
        criterion = net.criterion
        # get metrics parameters
        y_pred_upper, y_pred_lower, record_expected_values = mh.get_pred_interval(
            record_output,
            criterion,
            record_targets
        )

        # targets_rescaled, output_rescaled = dh.rescale_manually(..)
        analyzed_metrics=[
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
        results = metrics.fetch_metrics(
            targets=record_targets,
            expected_values=record_expected_values,
            y_pred_upper=y_pred_upper,
            y_pred_lower=y_pred_lower,
            analyzed_metrics=analyzed_metrics #all above listes metrics are fetched
        )
        results_per_timestep = metrics.fetch_metrics(
            targets=record_targets,
            expected_values=record_expected_values,
            y_pred_upper=y_pred_upper,
            y_pred_lower=y_pred_lower,
            analyzed_metrics=["rmse",
                              "sharpness",
                              "picp",
                              "mis"],
            total=False,
        )
        # plot metrics
        plot.plot_metrics(
            results_per_timestep["rmse"],
            results_per_timestep["sharpness"],
            results_per_timestep["picp"],
            results_per_timestep["mis"],
            OUTDIR,
            "metrics-evaluation",
        )

        # plot forecast for sample time steps
        # the first and 24th timestep relative to the start of the Test-Dataset
        testhours = [0,24]

        actual_time = pd.to_datetime(
            df.loc[PAR["history_horizon"] + split_index :, "Time"]
        )
        for i in testhours:
            actuals = actual_time.iloc[i : i + FORECAST_HORIZON]
            plot.plot_timestep(
                record_targets[i].detach().numpy(),
                record_expected_values[i].detach().numpy(),
                y_pred_upper[i].detach().numpy(),
                y_pred_lower[i].detach().numpy(),
                i,
                OUTDIR,
                PAR["cap_limit"],
                actuals,
            )

        print(
            metrics.results_table(
                target_stations,
                results,
                save_to_disc=OUTDIR,
            )
        )

        # BOXPLOTS
        plot.plot_boxplot(
            targets=record_targets,
            expected_values=record_expected_values,
            y_pred_upper=y_pred_upper,
            y_pred_lower=y_pred_lower,
            analyzed_metrics=['mse', 'rmse'],
            sample_frequency=24,
            save_to_disc=OUTDIR,
        )
    print("Done!!")
