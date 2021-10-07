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
import utils.datahandler as dh
import utils.tensorloader as tl
import utils.metrics as metrics
import utils.plot as plot
import utils.modelhandler as mh


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
    df = dh.fill_if_missing(df, periodicity=24)

    if "target_list" in PAR:
        if PAR["target_list"] is not None:
            df[target_id] = df[PAR["target_list"]].sum(axis=1)

    # reload trained NN
    with torch.no_grad():
        net = mh.ModelHandler.load_model(
            f"{INMODEL}.pkl"
        )  # torch.load(INMODEL, map_location=torch.device(DEVICE))  # mapping to CPU
        net.to(DEVICE)
        df_new, _ = dh.scale_all(df, scalers=net.scalers, **PAR)
        df_new.index = df["Time"]

        split_index = int(len(df_new.index) * SPLIT_RATIO)
        train_df = df_new.iloc[0:split_index]
        test_df = df_new.iloc[split_index:]

        test_data_loader = tl.make_dataloader(
            test_df,
            target_id,
            PAR["encoder_features"],
            PAR["decoder_features"],
            history_horizon=PAR["history_horizon"],
            forecast_horizon=PAR["forecast_horizon"],
            shuffle=False,
        ).to(DEVICE)

        # bench = mh.ModelHandler.benchmark(
        #     test_data_loader,
        #     [net],
        #     [metrics.NllGauss(), metrics.Rmse()],  # metrics.Rmse(), metrics.Picp()],
        #     total=True,
        # )

        test_metrics_timesteps = [
            metrics.NllGauss(),
            metrics.Rmse(),
            metrics.Sharpness(),
            metrics.Picp(),
            metrics.Mis(),
        ]
        test_metrics_total = [
            metrics.NllGauss(),
            metrics.Mse(),
            metrics.Rmse(),
            metrics.Sharpness(),
            metrics.Picp(),
            metrics.Rae(),
            metrics.Mae(),
            metrics.Mis(),
            metrics.Mase(),
            metrics.PinnballLoss(),
            metrics.Residuals(),
        ]
        results_total_per_forecast = (
            mh.ModelHandler.benchmark(
                test_data_loader, [net], test_metrics=test_metrics_total, total=True
            )
            .iloc[0]
            .unstack()
        )
        results_per_timestep_per_forecast = mh.ModelHandler.benchmark(
            test_data_loader,
            [net],
            test_metrics=test_metrics_timesteps,
            total=False,
        )
        results_per_timestep_per_forecast.head()

        # XXX why is this different from the rest?
        rmse_values = pd.DataFrame(
            data=results_per_timestep_per_forecast.xs(
                "Rmse", axis=1, level=1, drop_level=False
            ).values,
            columns=[net.name],
        )
        sharpness_values = pd.DataFrame(
            data=results_per_timestep_per_forecast.xs(
                "Sharpness", axis=1, level=1, drop_level=True
            ),
            columns=[net.name],
        )
        picp_values = pd.DataFrame(
            data=results_per_timestep_per_forecast.xs(
                "Picp", axis=1, level=1, drop_level=True
            ),
            columns=[net.name],
        )
        mis_values = pd.DataFrame(
            data=results_per_timestep_per_forecast.xs(
                "Mis", axis=1, level=1, drop_level=True
            ),
            columns=[net.name],
        )
        # plot metrics
        plot.plot_metrics(
            rmse_values,
            sharpness_values,
            picp_values,
            mis_values,
            OUTDIR,
            "metrics-evaluation",
        )
        # plot forecast for sample time steps
        # the first and 24th timestep relative to the start of the Test-Dataset
        testhours = [0, 24]

        actual_time = pd.to_datetime(
            df.loc[PAR["history_horizon"] + split_index :, "Time"]
        )
        for i in testhours:
            inputs_enc, inputs_dec, targets = test_data_loader[i]

            prediction = net.predict(inputs_enc, inputs_dec)
            (
                y_pred_upper,
                y_pred_lower,
                expected_values,
            ) = net.loss_metric.get_prediction_interval(prediction)
            actuals = actual_time.iloc[i : i + FORECAST_HORIZON]
            plot.plot_timestep(
                targets.detach().squeeze().numpy(),
                expected_values.detach().squeeze().numpy(),
                y_pred_upper.detach().squeeze().numpy(),
                y_pred_lower.detach().squeeze().numpy(),
                i,
                OUTDIR,
                PAR["cap_limit"],
                actuals,
            )

        results_total_per_forecast.to_csv(
            os.path.join(OUTDIR, f"{net.name}.csv"), sep=";", index=True
        )
        print(results_total_per_forecast)
        exit()
        # BOXPLOTS
        plot.plot_boxplot(
            targets=record_targets,
            expected_values=record_expected_values,
            y_pred_upper=y_pred_upper,
            y_pred_lower=y_pred_lower,
            analyzed_metrics=["mse", "rmse"],
            sample_frequency=24,
            save_to_disc=OUTDIR,
        )
        torch.save(record_expected_values, "RNN_best_forecasts.pt")
        torch.save(y_pred_upper, "RNN_best_upper_limits.pt")
        torch.save(y_pred_lower, "RNN_best_lower_limits.pt")
        torch.save(record_targets, "RNN_best_true_values.pt")
    print("Done!!")
