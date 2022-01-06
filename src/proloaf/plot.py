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
Provides handy plot functions to visualize predictions and performancec over selected timeseries
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import proloaf.metrics as metrics


def plot_timestep(
    target,
    pred,
    y_pred_upper,
    y_pred_lower,
    timestep,
    OUTPATH,
    limit,
    actual_time=None,
    draw_limit=False,
):
    """
    Create a matplotlib.pyplot.subplot to compare true and predicted values.
    Save the resulting plot at (OUTPATH + 'eval_hour{}'.format(timestep))

    Parameters
    ----------
    target : ndarray
        Numpy array containing true values
    pred : ndarray
        Numpy array containing predicted values
    y_pred_upper : ndarray
        Numpy array containing upper limit of prediction confidence interval
    y_pred_lower : ndarray
        Numpy array containing lower limit of prediction confidence interval
    timestep : int
        The timestep of the prediction, relative to the beginning of the dataset
    OUTPATH : string
        Path to where the plot should be saved
    limit : float
        The cap limit. Used to draw a horizontal line with height = limit.
    actual_time : pandas.Series, default = None
        The actual time from the data set

    Returns
    -------
    No return value
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(target, ".-k", label="Truth")  # true values
    ax.plot(pred, "b", label="Predicted")
    # insert actual time, assuming the underlying data to be in hourly resolution
    try:
        ax.set_title(actual_time.iloc[0].strftime("%a, %Y-%m-%d"), fontsize=20)
        positions = range(0, len(pred), 2)
        labels = (
            pd.to_datetime(actual_time).dt.to_period().to_numpy()
        )  # assuming hourly resolution in most of the evaluations
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=90)
        mdates = matplotlib.dates.date2num(actual_time)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=20))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        new_labels = labels[positions]
        plt.xticks(positions, new_labels)
    except:
        ax.set_title("Forecast along horizon", fontsize=22)
    ax.fill_between(
        np.arange(pred.shape[0]),
        pred.squeeze(),
        y_pred_upper.squeeze(),
        alpha=0.1,
        color="g",
    )
    ax.fill_between(
        np.arange(pred.shape[0]),
        y_pred_lower.squeeze(),
        pred.squeeze(),
        alpha=0.1,
        color="g",
    )

    ax.set_xlabel("Hour", fontsize=18)
    ax.set_ylabel("Scaled Residual Load (-1,1)", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(b=True, linestyle="-")
    if limit and draw_limit:
        plt.axhline(linewidth=2, color="r", y=limit)
    ax.grid()
    ax.set_xlabel(
        "Time of Day", fontsize=20
    )  # assuming hourly resolution in most of the evaluations
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.savefig(OUTPATH + "eval_hour{}".format(timestep))
    # plt.show(fig)
    plt.close(fig)


def plot_metrics(
    rmse_horizon, sharpness_horizon, coverage_horizon, mis_horizon, OUTPATH, title=""
):
    """
    Create a matplotlib.pyplot.figure with plots for the given metrics
    Save the resulting figure at (OUTPATH + 'metrics_plot')

    Parameters
    ----------
    rmse_horizon : ndarray
        The values for the root mean square error over the horizon
    sharpness_horizon : ndarray
        The values for the sharpness over the horizon
    coverage_horizon : ndarray
        The values for the PICP (prediction interval coverage probability or % of true
        values in the predicted intervals) over the horizon
    mis_horizon : ndarray
        The values for the mean interval score over the horizon
    OUTPATH : string
        The path to where the figure should be saved
    title : string
        The text for a centered title for the figure

    Returns
    -------
    No return value
    """

    with plt.style.context("seaborn"):
        fig = plt.figure(figsize=(16, 12))
        st = fig.suptitle(title, fontsize=25)
        plt.rc("xtick", labelsize=15)
        plt.rc("ytick", labelsize=15)

        ax_rmse = plt.subplot(2, 2, 1)
        ax_sharpness = plt.subplot(2, 2, 3)
        ax_PICP = plt.subplot(2, 2, 2)
        ax_MIS = plt.subplot(2, 2, 4)

        for element in rmse_horizon:
            ax_rmse.plot(rmse_horizon[element], label=element)
        ax_rmse.set_title("RMSE along horizon", fontsize=22)
        ax_rmse.set_xlabel("Hour", fontsize=18)
        ax_rmse.set_ylabel("RMSE", fontsize=20)
        ax_rmse.legend(fontsize=20)
        ax_rmse.grid(b=True, linestyle="-")

        for element in sharpness_horizon:
            ax_sharpness.plot(sharpness_horizon[element], label=element)
        ax_sharpness.set_title("sharpness along horizon", fontsize=22)
        ax_sharpness.set_xlabel("Hour", fontsize=18)
        ax_sharpness.set_ylabel("sharpness", fontsize=20)
        ax_sharpness.legend(fontsize=20)
        ax_sharpness.grid(b=True, linestyle="-")

        for element in coverage_horizon:
            ax_PICP.plot(coverage_horizon[element], label=element)
        ax_PICP.set_title("coverage along horizon", fontsize=22)
        ax_PICP.set_xlabel("Hour", fontsize=18)
        ax_PICP.set_ylabel("coverage in %", fontsize=20)
        ax_PICP.legend(fontsize=20)
        ax_PICP.grid(b=True, linestyle="-")

        for element in mis_horizon:
            ax_MIS.plot(mis_horizon[element], label=element)
        ax_MIS.set_title("Mean Interval score", fontsize=22)
        ax_MIS.set_xlabel("Hour", fontsize=18)
        ax_MIS.set_ylabel("MIS", fontsize=20)
        ax_MIS.legend(fontsize=20)
        ax_MIS.grid(b=True, linestyle="-")

        st.set_y(1.08)
        fig.subplots_adjust(top=0.95)
        plt.tight_layout()
        plt.savefig(OUTPATH + title + "_metrics_plot")
        plt.show()


def plot_boxplot(
    metrics_per_sample: pd.DataFrame,
    sample_frequency=1,
    save_to_disc=False,
    fig_title="boxplot",
):
    """
    Create a matplotlib.pyplot.figure with boxplots for the given metrics
    Save the resulting figure at (save_to_disc + 'boxplot')
    To visualize more clearly a sampling frequency can be defined,
    i.e. the time-delta of the moving windows over the test set.
    Per default over the test set, every timestep initiates a new out-of-sample prediction.
    E.g. if the evaluations hall happen on daily basis with hourly time steps sample_frequency can be set to 24.

    Parameters
    ----------
    TODO

    Returns
    -------
    No return value
    """
    fig = plt.figure()  # plt.figure(figsize=(16, 12))
    ax1 = metrics_per_sample.iloc[::sample_frequency].boxplot(
        # column=analyzed_metrics,  # list(metrics_per_sample.keys()),
        color=dict(boxes="k", whiskers="k", medians="k", caps="k"),
        # figsize=(8.5, 10),fontsize=24,
    )
    ax1.set_xlabel(
        "Mean error per sample with predefined sampling steps: "
        + str(sample_frequency)
        + " on prediction horizon",
    )  # fontsize=24
    ax1.set_ylabel("Error measure")  # , fontsize=24
    # ymin, ymax = -0.01, 1.5
    # ax1.set_ylim([ymin, ymax])
    if save_to_disc:
        plt.savefig(save_to_disc + fig_title + ".png")
    plt.show()


def plot_hist(
    targets,
    expected_values,
    y_pred_upper,
    y_pred_lower,
    analyzed_metrics=["mse"],
    sample_frequency=1,
    save_to_disc=False,
    fig_title="Error Probability Distribution",
    method="method",
    bins=10,
):
    """
    Create a matplotlib.pyplot.figure with kde plots for the given metrics
    Save the resulting figure at (save_to_disc + 'Error Probability Distribution')
    To visualize more clearly a sampling frequency can be defined,
    i.e. the time-delta of the moving windows over the test set.
    Per default over the test set, every timestep initiates a new out-of-sample prediction.
    E.g. if the evaluations hall happen on daily basis with hourly time steps sample_frequency can be set to 24.

    Parameters
    ----------
    TODO

    Returns
    -------
    No return value
    """
    metrics_per_sample = {}
    for results in range(targets.shape[0]):
        mse_per_sample = [
            metrics.mse(targets[results][i], [expected_values[results][i]])
            for i, value in enumerate(targets[results])
        ]
        metrics_per_sample[results] = pd.DataFrame(
            data=mse_per_sample, columns=["mse"]
        ).astype("float")
        if "rmse" in analyzed_metrics:
            rmse_per_sample = [
                metrics.rmse(targets[results][i], [expected_values[results][i]])
                for i, value in enumerate(targets[results])
            ]
            metrics_per_sample[results]["rmse"] = pd.DataFrame(rmse_per_sample).astype(
                "float"
            )
        if "sharpness" in analyzed_metrics:
            sharpness_per_sample = [
                metrics.sharpness([y_pred_upper[results][i], y_pred_lower[results][i]])
                for i, value in enumerate(targets[results])
            ]
            metrics_per_sample[results]["sharpness"] = pd.DataFrame(
                sharpness_per_sample
            ).astype("float")
        if "picp" in analyzed_metrics:
            coverage_per_sample = [
                metrics.picp(
                    targets[i], [y_pred_upper[results][i], y_pred_lower[results][i]]
                )
                for i, value in enumerate(targets[results])
            ]
            metrics_per_sample[results]["picp"] = pd.DataFrame(
                coverage_per_sample
            ).astype("float")
        if "rae" in analyzed_metrics:
            rae_per_sample = [
                metrics.rae(targets[i], [expected_values[results][i]])
                for i, value in enumerate(targets[results])
            ]
            metrics_per_sample[results]["rae"] = pd.DataFrame(rae_per_sample).astype(
                "float"
            )
        if "mae" in analyzed_metrics:
            mae_per_sample = [
                metrics.mae(targets[i], [expected_values[results][i]])
                for i, value in enumerate(targets[results])
            ]
            metrics_per_sample[results]["mae"] = pd.DataFrame(mae_per_sample).astype(
                "float"
            )
        if "mis" in analyzed_metrics:
            mis_per_sample = [
                metrics.mis(
                    targets[results][i],
                    [y_pred_upper[results][i], y_pred_lower[results][i]],
                    alpha=0.05,
                )
                for i, value in enumerate(targets[results])
            ]
            metrics_per_sample[results]["mis"] = pd.DataFrame(mis_per_sample).astype(
                "float"
            )
        if "mase" in analyzed_metrics:
            mase_per_sample = [
                metrics.mase(
                    targets[results][i],
                    [expected_values[results][i]],
                    0,
                    insample_target=targets[results].roll(7 * 24, 0)[i],
                )
                for i, value in enumerate(targets[results])
            ]
            metrics_per_sample[results]["mase"] = pd.DataFrame(mase_per_sample).astype(
                "float"
            )
        if "residuals" in analyzed_metrics:
            residuals_per_sample = [
                metrics.residuals(targets[results][i], [expected_values[results][i]])
                for i, value in enumerate(targets[results])
            ]
            metrics_per_sample[results]["residuals"] = pd.DataFrame(
                residuals_per_sample
            ).astype("float")
        if "qs" in analyzed_metrics:
            quantile_score_per_sample = [
                metrics.pinball_loss(
                    targets[i],
                    [y_pred_upper[results][i], y_pred_lower[results][i]],
                    [0.025, 0.975],
                )
                for i, value in enumerate(targets[results])
            ]
            metrics_per_sample[results]["qs"] = pd.DataFrame(
                quantile_score_per_sample
            ).astype("float")
    all_results = pd.concat(
        metrics_per_sample.values(), keys=metrics_per_sample.keys(), axis=1
    )
    with plt.style.context("seaborn"):
        fig = plt.figure(figsize=(16, 12))  # plt.figure()
        results = pd.DataFrame(
            data=all_results.xs("residuals", axis=1, level=1, drop_level=False).values,
            columns=method,
        )
        # print("Residuals DataFrame Head:",results.head())
        for element in results:
            ax1 = results[element].plot(
                kind="hist", bins=bins, alpha=0.5, label=element
            )
            ax1.set_title("Histogram of Residuals", fontsize=22)
            ax1.set_xlabel("Residuals", fontsize=18)
            ax1.set_ylabel("Frequency", fontsize=20)
            ax1.legend(fontsize=20, loc="upper center", fancybox=True)
        if save_to_disc:
            plt.savefig(save_to_disc + fig_title + ".png")
        plt.show()
