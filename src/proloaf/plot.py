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
Provides handy plot functions to visualize predictions and performance over selected timeseries
"""
import os
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
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        mdates = matplotlib.dates.date2num(actual_time)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=20))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
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
    ax.grid(visible=True, linestyle="-")
    if limit and draw_limit:
        plt.axhline(linewidth=2, color="r", y=limit)
    ax.grid()
    ax.set_xlabel(
        "Time of Day", fontsize=20
    )  # assuming hourly resolution in most of the evaluations
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.savefig(OUTPATH + "eval_hour{}".format(timestep))
    plt.show()
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

    with plt.style.context("seaborn-v0_8"):
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
        ax_rmse.grid(visible=True, linestyle="-")

        for element in sharpness_horizon:
            ax_sharpness.plot(sharpness_horizon[element], label=element)
        ax_sharpness.set_title("sharpness along horizon", fontsize=22)
        ax_sharpness.set_xlabel("Hour", fontsize=18)
        ax_sharpness.set_ylabel("sharpness", fontsize=20)
        ax_sharpness.legend(fontsize=20)
        ax_sharpness.grid(visible=True, linestyle="-")

        for element in coverage_horizon:
            ax_PICP.plot(coverage_horizon[element], label=element)
        ax_PICP.set_title("coverage along horizon", fontsize=22)
        ax_PICP.set_xlabel("Hour", fontsize=18)
        ax_PICP.set_ylabel("coverage in %", fontsize=20)
        ax_PICP.legend(fontsize=20)
        ax_PICP.grid(visible=True, linestyle="-")

        for element in mis_horizon:
            ax_MIS.plot(mis_horizon[element], label=element)
        ax_MIS.set_title("Mean Interval score", fontsize=22)
        ax_MIS.set_xlabel("Hour", fontsize=18)
        ax_MIS.set_ylabel("MIS", fontsize=20)
        ax_MIS.legend(fontsize=20)
        ax_MIS.grid(visible=True, linestyle="-")

        st.set_y(1.08)
        fig.subplots_adjust(top=0.95)
        plt.tight_layout()
        plt.savefig(OUTPATH + title + "_metrics_plot")
        plt.show()


def plot_boxplot(
    metrics_per_sample: pd.DataFrame,
    sample_frequency: int = 1,
    save_to: str = None,
    fig_title: str = "boxplot",
):
    """
    Create a matplotlib.pyplot.figure with boxplots for the given metric values.
    Save the resulting figure as <fig_title>.png
    To visualize more clearly a sampling frequency can be defined,
    i.e. the time-delta of the moving windows over the test set.
    If the evaluations shall happen on daily basis with hourly time steps sample_frequency can be set to 24.

    Parameters
    ----------
    metrics_per_sample: pd.DataFrame
        Dataframe with values for different metrics as columns and samples for rows.
    sample_frequency: int, default = 1
        Frequency of samples considered in timesteps. 1 means every sample will be shown.
    save_to: str, default = None
        Path to the folder where plot should be saved. If `None` the plot will not be saved.
    fig_title: str, default = "boxplot"
        Title of the plot.
    Returns
    -------
    No return value
    """
    plt.clf()
    ax1 = metrics_per_sample.iloc[::sample_frequency].boxplot(
        color=dict(boxes="k", whiskers="k", medians="k", caps="k"),
    )
    ax1.set_xlabel(
        "Mean error per sample with predefined sampling steps: "
        + str(sample_frequency)
        + " on prediction horizon",
    )  
    ax1.set_ylabel("Error measure")
    if save_to:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        plt.savefig(os.path.join(save_to, f"{fig_title}.png"))
    plt.show()

# TODO Only used in baselines
def plot_hist(
    data: pd.DataFrame,
    save_to: str=None,
    fig_title: str = "Error Probability Distribution",
    bins: int = 10,
):
    """
    Create a matplotlib.pyplot.figure with kde plots for the given metrics
    Save the resulting figure at "<save_to>/<fig_title>.png".

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe with values for different metrics as columns and prediction names for rows.
    save_to: str, default = None
        Path to the folder where plot should be saved. If `None` the plot will not be saved.
    fig_title: str, default = "Error Probability Distribution"
        Title of the plot.

    Returns
    -------
    No return value
    """
    with plt.style.context("seaborn"):
        fig = plt.figure(figsize=(16, 12))  # plt.figure()
        # print("Residuals DataFrame Head:",results.head())
        for element in data.columns:
            ax1 = data[element].plot(
                kind="hist", bins=bins, alpha=0.5, label=element
            )
            ax1.set_title("Histogram of Residuals", fontsize=22)
            ax1.set_xlabel("Residuals", fontsize=18)
            ax1.set_ylabel("Frequency", fontsize=20)
            ax1.legend(fontsize=20, loc="upper center", fancybox=True)
        if save_to:
            plt.savefig(save_to + fig_title + ".png")
        plt.show()
