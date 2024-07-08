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


import pandas as pd
import torch
import warnings
import sys
import os
from functools import partial

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)
warnings.filterwarnings("ignore")

from proloaf.confighandler import read_config
from proloaf.cli import parse_basic
import proloaf.datahandler as dh
import proloaf.tensorloader as tl
import proloaf.metrics as metrics
import proloaf.plot as plot
import proloaf.modelhandler as mh
import proloaf.models as models
from proloaf.event_logging import create_event_logger

logger = create_event_logger("evaluate")

if __name__ == "__main__":
    ARGS = parse_basic()
    PAR = read_config(model_name=ARGS.station, config_path=ARGS.config, main_path=MAIN_PATH)

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
    df = pd.read_csv(INFILE, sep=";", index_col=0, parse_dates=True)

    if "target_list" in PAR:
        if PAR["target_list"] is not None:
            df[target_id] = df[PAR["target_list"]].sum(axis=1)

    # reload trained NN
    with torch.no_grad():
        net = mh.ModelHandler.load_model(f"{INMODEL}.pkl", locate="cpu")
        net.to(DEVICE)

        train_df, test_df = dh.split(df, [SPLIT_RATIO])

        test_data = tl.TimeSeriesData(
            test_df,
            device=DEVICE,
            preparation_steps=[
                partial(dh.set_to_hours, freq="1H"),
                partial(dh.fill_if_missing, periodicity=PAR.get("periodicity", 24)),
                dh.add_cyclical_features,
                dh.add_onehot_features,
                net.scalers.transform,
                dh.check_continuity,
            ],
            **PAR,
        )
        test_metrics_timesteps = [
            metrics.NllGauss(),
            metrics.Rmse(),
            metrics.Sharpness(),
            metrics.Picp(),
            metrics.Mis(),
        ]
        test_metrics_sample = [metrics.Mse(), metrics.Rmse()]
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
                test_data,
                [net],
                test_metrics=test_metrics_total,
                avg_over="all",
            )
            .iloc[0]
            .unstack()
        )
        results_per_sample_per_forecast = mh.ModelHandler.benchmark(
            test_data,
            [net],
            test_metrics=test_metrics_sample,
            avg_over=("time", "feature"),
        )
        results_per_timestep_per_forecast = mh.ModelHandler.benchmark(
            test_data,
            [net],
            test_metrics=test_metrics_timesteps,
            avg_over=("sample", "feature"),
        )
        results_per_timestep_per_forecast.head()
        rmse_values = pd.DataFrame(
            data=results_per_timestep_per_forecast.xs("Rmse", axis=1, level=1, drop_level=True),
            columns=[net.name],
        )
        sharpness_values = pd.DataFrame(
            data=results_per_timestep_per_forecast.xs("Sharpness", axis=1, level=1, drop_level=True),
            columns=[net.name],
        )
        picp_values = pd.DataFrame(
            data=results_per_timestep_per_forecast.xs("Picp", axis=1, level=1, drop_level=True),
            columns=[net.name],
        )
        mis_values = pd.DataFrame(
            data=results_per_timestep_per_forecast.xs("Mis", axis=1, level=1, drop_level=True),
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
        df["Time"] = df.index
        # actual_time = pd.to_datetime(
        #    df.iloc[PAR["history_horizon"] + split_index:].index
        # )
        actual_time = pd.Series(pd.to_datetime(df.index), index=df.index, dtype=object)
        for i in testhours:
            (
                inputs_enc,
                inputs_enc_aux,
                inputs_dec,
                inputs_dec_aux,
                last_value,
                targets,
            ) = test_data[
                i
            ]  # No Error over time for autoencoder
            if isinstance(net.model, models.AutoEncoder):
                logger.warning(
                    "Autoencoders do not create a prediction for the future, no plot will be generated comparing the prediciton to the target."
                )
                break
            prediction = net.predict(
                inputs_enc=inputs_enc.unsqueeze(dim=0),
                inputs_enc_aux=inputs_enc_aux.unsqueeze(dim=0),
                inputs_dec=inputs_dec.unsqueeze(dim=0),
                inputs_dec_aux=inputs_dec_aux.unsqueeze(dim=0),
                last_value=last_value.unsqueeze(dim=0),
            )
            quantile_prediction = net.loss_metric.get_quantile_prediction(
                predictions=prediction,
                target=targets.unsqueeze(0),
                inputs_enc=inputs_enc.unsqueeze(dim=0),
                inputs_enc_aux=inputs_enc_aux.unsqueeze(dim=0),
            )
            if isinstance(quantile_prediction, tuple):
                quantile_prediction = quantile_prediction[0]
            expected_values = quantile_prediction.get_quantile(0.5)
            y_pred_upper = quantile_prediction.select_upper_bound().values.squeeze(dim=2)
            y_pred_lower = quantile_prediction.select_lower_bound().values.squeeze(dim=2)
            actuals = actual_time[i : i + FORECAST_HORIZON]
            for j in range(targets.shape[1]):
                try:
                    os.mkdir(f"{OUTDIR}/{test_data.target_id[j]}")
                except FileExistsError:
                    pass
                plot.plot_timestep(
                    targets.detach().numpy()[:, j],
                    expected_values.detach().numpy()[0, :, j],
                    y_pred_upper.detach().numpy()[0, :, j],
                    y_pred_lower.detach().numpy()[0, :, j],
                    i,
                    f"{OUTDIR}/{test_data.target_id[j]}/",
                    PAR["cap_limit"],
                    actuals,
                )

        results_total_per_forecast.to_csv(os.path.join(OUTDIR, f"{net.name}.csv"), sep=";", index=True)
        logger.info("Results total per forecast:\n{!s}".format(results_total_per_forecast))
        (
            inputs_enc,
            inputs_enc_aux,
            inputs_dec,
            inputs_dec_aux,
            last_value,
            targets,
        ) = test_data[0]
        prediction = net.predict(
            inputs_enc=inputs_enc.unsqueeze(dim=0),
            inputs_enc_aux=inputs_enc_aux.unsqueeze(dim=0),
            inputs_dec=inputs_dec.unsqueeze(dim=0),
            inputs_dec_aux=inputs_dec_aux.unsqueeze(dim=0),
            last_value=last_value.unsqueeze(dim=0),
        )
        quantile_prediction = net.loss_metric.get_quantile_prediction(
            predictions=prediction,
            target=targets.unsqueeze(dim=0),
            inputs_enc=inputs_enc.unsqueeze(dim=0),
            inputs_enc_aux=inputs_enc_aux.unsqueeze(dim=0),
        )
        if isinstance(quantile_prediction, tuple):
            quantile_prediction = quantile_prediction[0]
        expected_values = quantile_prediction.get_quantile(0.5)
        y_pred_upper = quantile_prediction.select_upper_bound().values.squeeze(dim=-1)
        y_pred_lower = quantile_prediction.select_lower_bound().values.squeeze(dim=-1)
        # BOXPLOTS
        plot.plot_boxplot(
            metrics_per_sample=results_per_sample_per_forecast[net.name],
            sample_frequency=24,
            save_to=OUTDIR,
        )
        torch.save(expected_values, "RNN_best_forecasts.pt")
        torch.save(y_pred_upper, "RNN_best_upper_limits.pt")
        torch.save(y_pred_lower, "RNN_best_lower_limits.pt")
        torch.save(targets, "RNN_best_true_values.pt")
    logger.info("Done!!")
