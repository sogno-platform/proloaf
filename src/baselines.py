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

import numpy as np
import pandas as pd
import torch

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)
import proloaf.plot as plot
import proloaf.datahandler as dh
import proloaf.baselinehandler as baselines

from proloaf import metrics
from proloaf.tensorloader import TimeSeriesData
from proloaf.confighandler import read_config
from proloaf.cli import parse_with_loss
from proloaf.event_logging import create_event_logger

torch.set_printoptions(linewidth=120)  # Display option for output
torch.set_grad_enabled(True)

logger = create_event_logger("baselines")


def main(infile, target_id):
    sarimax_model = None
    # Read load data
    df = pd.read_csv(infile, sep=";", index_col=0)
    df = dh.fill_if_missing(df, periodicity=SEASONALITY)

    df = dh.set_to_hours(df, freq=RESOLUTION)
    logger.info(f"{df.head() = }")
    if "col_list" in PAR:
        if PAR["col_list"] is not None:
            df[target_id] = df[PAR["col_list"]].sum(axis=1)

    try:
        target = PAR["target_id"]
        enc_features = PAR["encoder_features"]
        dec_features = PAR["decoder_features"]
        # now just make sure that the exogenous features do not include the target itself as endog already does.
        if target in enc_features:
            enc_features.remove(target)
        if target in dec_features:
            dec_features.remove(target)

        if SCALE_DATA:
            scaler = dh.MultiScaler(PAR["feature_groups"])
            df = scaler.fit_transform(df)

        df_train, df_val = dh.split(df, splits=[PAR["validation_split"]])

        dataset_train = TimeSeriesData(
            df_train,
            history_horizon=PAR["history_horizon"],
            forecast_horizon=PAR["forecast_horizon"],
            encoder_features=PAR["target_id"],
            decoder_features=None,
            target_id=PAR["target_id"],
        )
        dl_train = dataset_train.make_data_loader(batch_size=1, shuffle=False)
        x_train_1d = np.array(
            [input.squeeze().numpy() for input, _, _, _, _, _ in dl_train]
        )
        y_train_1d = np.array(
            [target.squeeze().numpy() for _, _, _, _, _, target in dl_train]
        )

        dataset_val = TimeSeriesData(
            df_val,
            history_horizon=PAR["history_horizon"],
            forecast_horizon=PAR["forecast_horizon"],
            encoder_features=PAR["target_id"],
            decoder_features=None,
            target_id=PAR["target_id"],
        )
        dl_val = dataset_val.make_data_loader(batch_size=1, shuffle=False)
        x_val_1d = np.array([input.squeeze().numpy() for input, _, _, _, _, _ in dl_val])
        y_val_1d = np.array([target.squeeze().numpy() for _, _, _, _, _, target in dl_val])

        mean_forecast = []
        upper_pi = []
        lower_pi = []
        baseline_method = []

        if EXOG:
            df_exog_train = df_train[enc_features + dec_features]
            df_exog_val = df_val[enc_features + dec_features]
        else:  # empty exogenous features, to ignore them in following baseline models, can be removed actually...
            df_exog_train = None
            df_exog_val = None

        if "simple-naive" in CALC_BASELINES:
            # Naïve
            (
                naive_expected_values,
                naive_y_pred_upper,
                naive_y_pred_lower,
            ) = baselines.persist_forecast(
                x_train_1d, x_val_1d, y_train_1d, PAR["forecast_horizon"], alpha=ALPHA
            )
            mean_forecast.append(naive_expected_values)
            upper_pi.append(naive_y_pred_upper)
            lower_pi.append(naive_y_pred_lower)
            baseline_method.append("simple-naive")

        if "seasonal-naive" in CALC_BASELINES:
            # SNaïve
            (
                s_naive_expected_values,
                s_naive_y_pred_upper,
                s_naive_y_pred_lower,
            ) = baselines.seasonal_forecast(
                x_train_1d,
                x_val_1d,
                y_train_1d,
                PAR["forecast_horizon"],
                PERIODICITY,
                alpha=ALPHA,
            )
            mean_forecast.append(s_naive_expected_values)
            upper_pi.append(s_naive_y_pred_upper)
            lower_pi.append(s_naive_y_pred_lower)
            baseline_method.append("seasonal-naive")

        if "naive-stl" in CALC_BASELINES:
            # decomposition(+ any#model)
            (
                sd_naive_expected_values,
                sd_naive_y_pred_upper,
                sd_naive_y_pred_lower,
            ) = baselines.persist_forecast(
                x_train_1d,
                x_val_1d,
                y_train_1d,
                PAR["forecast_horizon"],
                periodicity=PERIODICITY,
                seasonality=SEASONALITY,
                decomposed=True,
                alpha=ALPHA,
            )
            mean_forecast.append(sd_naive_expected_values)
            upper_pi.append(sd_naive_y_pred_upper)
            lower_pi.append(sd_naive_y_pred_lower)
            baseline_method.append("naive-stl")

        # baselines.test_stationarity(df_train[target],maxlag=PERIODICITY*SEASONALITY)
        arima_order = ORDER
        sarima_order = ORDER
        sarima_sorder = sORDER
        if "arima" in CALC_BASELINES:
            # ##############################ARIMA####################################
            arima_model = None
            if APPLY_EXISTING_MODEL:
                arima_model = baselines.load_baseline(OUTDIR, name="ARIMA")
            if arima_model is None:
                arima_model, _, _, arima_order, _ = baselines.auto_sarimax_wrapper(
                    endog=df_train[target],
                    order=ORDER,
                    seasonal_order=None,
                    seasonal=False,
                    lag=PERIODICITY * SEASONALITY,
                    grid_search=PAR["exploration"],
                    train_limit=LIMIT_HISTORY,
                )
            else:
                logger.info(
                    "Loaded existing fitted ARIMA model from {!s}".format(OUTDIR)
                )
            (
                ARIMA_expected_values,
                ARIMA_y_pred_upper,
                ARIMA_y_pred_lower,
            ) = baselines.make_forecasts(
                endog_train=df_train[target],
                endog_val=df_val[target],
                fitted=arima_model,
                forecast_horizon=PAR["forecast_horizon"],
                train_limit=LIMIT_HISTORY,
                limit_steps=NUM_PRED,
                pi_alpha=ALPHA,
                online=True,
            )
            mean_forecast.append(ARIMA_expected_values)
            upper_pi.append(ARIMA_y_pred_upper)
            lower_pi.append(ARIMA_y_pred_lower)
            baseline_method.append("ARIMA")
            baselines.save_baseline(
                OUTDIR, arima_model, name="ARIMA", save_predictions=False
            )

        if "sarima" in CALC_BASELINES:
            # #############################SARIMA####################################
            sarima_model = None
            if APPLY_EXISTING_MODEL:
                sarima_model = baselines.load_baseline(OUTDIR, name="SARIMA")
            if sarima_model is None:
                (
                    sarima_model,
                    _,
                    _,
                    sarima_order,
                    sarima_sorder,
                ) = baselines.auto_sarimax_wrapper(
                    endog=df_train[target],
                    order=ORDER,
                    seasonal_order=sORDER,
                    seasonal=True,
                    lag=PERIODICITY * SEASONALITY,
                    m=SEASONALITY,
                    train_limit=LIMIT_HISTORY,
                    grid_search=PAR["exploration"],
                )
            else:
                logger.info(
                    "Loaded existing fitted SARIMA model from {!s}".format(OUTDIR)
                )
            (
                SARIMA_expected_values,
                SARIMA_y_pred_upper,
                SARIMA_y_pred_lower,
            ) = baselines.make_forecasts(
                endog_train=df_train[target],
                endog_val=df_val[target],
                fitted=sarima_model,
                forecast_horizon=PAR["forecast_horizon"],
                train_limit=LIMIT_HISTORY,
                limit_steps=NUM_PRED,
                pi_alpha=ALPHA,
                online=True,
            )
            mean_forecast.append(SARIMA_expected_values)
            upper_pi.append(SARIMA_y_pred_upper)
            lower_pi.append(SARIMA_y_pred_lower)

            baseline_method.append("SARIMA")
            sarima_model.summary()
            baselines.save_baseline(
                OUTDIR, sarima_model, name="SARIMA", save_predictions=False
            )

        if "arimax" in CALC_BASELINES:
            # ##############################ARIMAX####################################
            arimax_model = None
            if APPLY_EXISTING_MODEL:
                arimax_model = baselines.load_baseline(OUTDIR, name="ARIMAX")
            if arimax_model is None:
                arimax_model, _, _, _, _ = baselines.auto_sarimax_wrapper(
                    endog=df_train[target],
                    exog=df_exog_train,
                    order=arima_order,
                    seasonal_order=None,
                    seasonal=False,
                    train_limit=LIMIT_HISTORY,
                    lag=PERIODICITY * SEASONALITY,
                    grid_search=False,
                )
            else:
                logger.info(
                    "Loaded existing fitted ARIMAX model from {!s}".format(OUTDIR)
                )
            (
                ARIMAX_expected_values,
                ARIMAX_y_pred_upper,
                ARIMAX_y_pred_lower,
            ) = baselines.make_forecasts(
                endog_train=df_train[target],
                endog_val=df_val[target],
                exog=df_exog_train,
                exog_forecast=df_exog_val,
                fitted=arimax_model,
                forecast_horizon=PAR["forecast_horizon"],
                train_limit=LIMIT_HISTORY,
                limit_steps=NUM_PRED,
                pi_alpha=ALPHA,
                online=True,
            )
            mean_forecast.append(ARIMAX_expected_values)
            upper_pi.append(ARIMAX_y_pred_upper)
            lower_pi.append(ARIMAX_y_pred_lower)
            baseline_method.append("ARIMAX")
            baselines.save_baseline(
                OUTDIR, arimax_model, name="ARIMAX", save_predictions=False
            )
        if "sarimax" in CALC_BASELINES:
            # ##############################SARIMAX###################################
            sarimax_model = None
            if APPLY_EXISTING_MODEL:
                sarimax_model = baselines.load_baseline(OUTDIR, name="SARIMAX")
            if sarimax_model is None:
                sarimax_model, _, _, _, _ = baselines.auto_sarimax_wrapper(
                    endog=df_train[target],
                    exog=df_exog_train,
                    order=sarima_order,
                    seasonal_order=sarima_sorder,
                    seasonal=True,
                    train_limit=LIMIT_HISTORY,
                    lag=PERIODICITY * SEASONALITY,
                    m=SEASONALITY,
                    grid_search=False,
                )
            else:
                logger.info(
                    "Loaded existing fitted SARIMAX model from {!s}".format(OUTDIR)
                )
            (
                SARIMAX_expected_values,
                SARIMAX_y_pred_upper,
                SARIMAX_y_pred_lower,
            ) = baselines.make_forecasts(
                endog_train=df_train[target],
                endog_val=df_val[target],
                exog=df_exog_train,
                exog_forecast=df_exog_val,
                fitted=sarimax_model,
                forecast_horizon=PAR["forecast_horizon"],
                train_limit=LIMIT_HISTORY,
                limit_steps=NUM_PRED,
                pi_alpha=ALPHA,
                online=True,
            )
            mean_forecast.append(SARIMAX_expected_values)
            upper_pi.append(SARIMAX_y_pred_upper)
            lower_pi.append(SARIMAX_y_pred_lower)
            baseline_method.append("SARIMAX")
            baselines.save_baseline(
                OUTDIR, sarimax_model, name="SARIMAX", save_predictions=False
            )

        if "ets" in CALC_BASELINES:
            # Exponential smoothing
            # with contextlib.redirect_stdout(None):
            train = pd.Series(
                df_train[target].values.squeeze(), index=df_train[target].index
            )
            test = pd.Series(
                df_val[target].values.squeeze(), index=df_val[target].index
            )
            (
                ets_expected_values,
                ets_y_pred_upper,
                ets_y_pred_lower,
            ) = baselines.exp_smoothing(
                train,
                test,
                PAR["forecast_horizon"],
                limit_steps=PAR["history_horizon"] + NUM_PRED,
                online=True,
            )
            mean_forecast.append(ets_expected_values[PAR["history_horizon"] :])
            upper_pi.append(ets_y_pred_upper[PAR["history_horizon"] :])
            lower_pi.append(ets_y_pred_lower[PAR["history_horizon"] :])
            baseline_method.append("ets")

        # GARCH
        if "garch" in CALC_BASELINES:
            mean = None
            p = 1
            q = 1
            if "arima" in CALC_BASELINES or "sarima" in CALC_BASELINES:
                # e.g. Garch (p'=1,q'=1) has the Arch order 1(past residuals squared) and the Garch order 1
                # (past variance:=sigma^2) in comparison, ARMA (p=1,p=1) has AR order 1(past observations) and
                # MA order 1(past residuals) --> this is why we just use the computed mean by sarimax

                # why not more than one Garch component? when forecasting market returns e.g.,
                # all the effects of conditional variance of t-2 are contained in the conditional variance t-1
                # p: is arch order and q ist garch order
                # arch component is nearly equivalent to the MA component of SARIMAX_expected_values here --> p' = q
                # garch component is nearly equivalent to the AR component of SARIMAX -->q'= p
                mean = ARIMA_expected_values
                # p= arimax_model.model_orders['ma']-->lagges variances
                # q =arimax_model.model_orders['ar']--> lagged residuals
            (
                GARCH_expected_values,
                GARCH_y_pred_upper,
                GARCH_y_pred_lower,
            ) = baselines.GARCH_predictioninterval(
                df_train[target],
                df_val[target],
                PAR["forecast_horizon"],
                mean_forecast=mean,
                p=p,
                q=q,
                alpha=ALPHA,
                limit_steps=NUM_PRED,
                periodicity=PERIODICITY,
            )
            # TODO this can add a None to mean_forecast which will fail later, currently fixed by skipping if None
            mean_forecast.append(mean)
            upper_pi.append(GARCH_y_pred_upper)
            lower_pi.append(GARCH_y_pred_lower)
            baseline_method.append("garch")

        # further options possible, checkout this blog article: https://towardsdatascience.com/an-overview-of-time-series-forecasting-models-a2fa7a358fcb
        # Dynamic linear models
        # TBATS
        # Prophet
        # ----------
        # randomforest --already existing in notebook
        # gradboosting --already existing in old project

        # ==========================Evaluate ALL==========================
        # for a proper benchmark with the rnn we need to consider that the rnn starts at anchor=t0
        # (first data-entry)+history horizon our baseline forecasts start at anchor=t0+forecast_horizon
        # so we have a shift in the start of the forecast simulation of about
        # shift=PAR['forecast_horizon']-PAR['history_horizon']

        analyzed_metrics_avg = [
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

        analyzed_metrics_sample = [
            metrics.Mse(),
            metrics.Rmse(),
            metrics.Sharpness(),
            metrics.Picp(),
            metrics.Rae(),
            metrics.Mis(),
            metrics.PinnballLoss(),
            metrics.Residuals(),
        ]

        analyzed_metrics_ts = [
            metrics.Rmse(),
            metrics.Sharpness(),
            metrics.Picp(),
            metrics.Mis(),
        ]

        results = pd.DataFrame(index=[metric.id for metric in analyzed_metrics_avg])
        results_per_sample = {}  # pd.DataFrame(index=[metric.id for metric in analyzed_metrics_sample])
        results_per_timestep = {}
        true_values = torch.zeros(
            [len(mean_forecast), NUM_PRED, PAR["forecast_horizon"]]
        )
        forecasts = torch.zeros([len(mean_forecast), NUM_PRED, PAR["forecast_horizon"]])
        upper_limits = torch.zeros(
            [len(mean_forecast), NUM_PRED, PAR["forecast_horizon"]]
        )
        lower_limits = torch.zeros(
            [len(mean_forecast), NUM_PRED, PAR["forecast_horizon"]]
        )
        for i, mean in enumerate(mean_forecast):
            if mean is None:
                # skip if no mean prediciton
                continue
            print(f"{mean = }")
            print(f"{y_val_1d = }")
            (
                results[baseline_method[i]],
                results_per_timestep[baseline_method[i]],
                results_per_sample[i],
                true_values[i],
                forecasts[i],
                upper_limits[i],
                lower_limits[i],
            ) = baselines.eval_forecast(
                config=PAR,
                forecasts=mean[:NUM_PRED],
                endog_val=y_val_1d[:NUM_PRED, :],
                upper_limits=upper_pi[i][:NUM_PRED],
                lower_limits=lower_pi[i][:NUM_PRED],
                path=OUTDIR,
                model_name="test" + baseline_method[i],
                analyzed_metrics_avg=analyzed_metrics_avg,
                analyzed_metrics_sample=analyzed_metrics_sample,
                analyzed_metrics_timesteps=analyzed_metrics_ts,
            )
        results_per_timestep_per_baseline = pd.concat(
            results_per_timestep.values(), keys=results_per_timestep.keys(), axis=1
        )
        results_per_sample = pd.concat(
            results_per_sample.values(), keys=results_per_timestep.keys(), axis=1
        )

        # plot metrics
        # rmse_horizon, sharpness_horizon, coverage_horizon, mis_horizon, OUTPATH, title
        plot.plot_metrics(
            results_per_timestep_per_baseline.xs(
                "Rmse", axis=1, level=1, drop_level=False
            ),
            results_per_timestep_per_baseline.xs(
                "Sharpness", axis=1, level=1, drop_level=False
            ),
            results_per_timestep_per_baseline.xs(
                "Picp", axis=1, level=1, drop_level=False
            ),
            results_per_timestep_per_baseline.xs(
                "Mis", axis=1, level=1, drop_level=False
            ),
            OUTDIR + "baselines",
        )
        print(f"{results = }")

        plot.plot_hist(
            data=results_per_sample.xs("Residuals", axis=1, level=1, drop_level=True),
            save_to=OUTDIR + "baselines",
            bins=80,
        )

        logger.info("results: \n {!s}".format(results))
        results.to_csv(f"{OUTDIR}/baselines.csv", sep=";", index=True)

    except KeyboardInterrupt:
        logger.info("\nmanual interrupt")
    finally:
        logger.info(",".join(baseline_method) + " fitted and evaluated")


if __name__ == "__main__":
    ARGS, LOSS_OPTIONS = parse_with_loss()
    PAR = read_config(
        model_name=ARGS.station, config_path=ARGS.config, main_path=MAIN_PATH
    )
    OUTDIR = os.path.join(MAIN_PATH, PAR["evaluation_path"])
    SCALE_DATA = True
    LIMIT_HISTORY = 300
    NUM_PRED = 2
    SLIDING_WINDOW = 24
    CALC_BASELINES = [
        "simple-naive",
        "seasonal-naive",
        "ets",
        "garch",
        "naive-stl",
        "arima",
        "arimax",
        "sarima",
        "sarimax",
    ]
    DAY_FIRST = True
    ORDER = (3, 1, 0)
    sORDER = (2, 0, 0, 24)
    SEASONALITY = 24
    PERIODICITY = 7
    ALPHA = 1.96
    EXOG = True
    APPLY_EXISTING_MODEL = False
    RESOLUTION = "H"
    main(infile=os.path.join(MAIN_PATH, PAR["data_path"]), target_id=PAR["target_id"])
