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
Provides functions for handling and manipulating dataframes

Includes functions for scaling, rescaling, filling missing values etc.
Some functions are no longer used directly in the project, but may nonetheless be
useful for testing or future applications.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# import torch
# import shutil
# import matplotlib.pyplot as plt
# import sklearn as sk
# from sklearn import preprocessing
# import torch
# import sklearn


def load_dataframe(data_path):
    """
    Load the excel file at the given path into a pandas.DataFrame

    .. deprecated::
            This function is no longer used. Instead, use fc_prep.load_raw_data_xlsx, which
            does the same thing but allows multiple files to be read at once, and ensures the
            date column is correctly formatted.

    Parameters
    ----------
    data_path : string
        The path to the excel file that is to be loaded

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the loaded data
    """
    def parser(x):
        return pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

    print("> Load data from '{}'... ".format(data_path), end="")
    df = pd.read_excel(data_path, parse_dates=[0], date_parser=parser)
    print("done!")

    return df


def ranges(nums):
    """
    Take a list of numbers (sorted or unsorted) and return all contiguous ranges within the list

    Ranges should be returned as tuples, where the first value is the start of the range and
    the last value is the end (inclusive). Single numbers are returned as tuples where both
    values equal the number. e.g. [0, 2, 3, 4, 6] -> [(0,0), (2,4), (6,6)]

    Parameters
    ----------
    nums : ndarray
        A ndarray containing a list of numbers

    Returns
    -------
    List
        A list containing tuples, where each tuple represents a contiguous range from the
        input ndarray
    """
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def custom_interpolate(df):
    """
    Interpolate the features with missing values in a time series data frame

    For each feature/columns:

    - finds the range of intervals of missing values
    - for each of the missing value in these intervals
        + collect the the previous day's value and the next day's value (t-24 & t+24) at that time instant
        + if any of them are missing, go for the next day(t-48 , t+48 and so on)
        + take their average
        + to account for the trend, shift the values by the slope of the interval extremes

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with missing values

    Returns
    -------
    pandas.DataFrame
        DataFrame with interpolated values
    """
    rows, columns = np.where(pd.isnull(df))
    miss_rows_ranges = ranges(rows)

    for i in range(len(miss_rows_ranges)):

        start, end = miss_rows_ranges[i]
        # dur = end - start
        p = 24  # periodicity

        for col in range(len(df.columns)):
            seas = np.zeros(len(df))
            if (
                start == end and end + 1 <= df.shape[0] and start - 1 >= 0
            ):  # if single point, take average of the nearby ones
                t = start
                df.iloc[t, col] = (df.iloc[t - 1, col] + df.iloc[t + 1, col]) / 2
            else:
                if (start - p) <= 0 or (end + p) > (df.shape[0]):
                    df = df.interpolate(method="pchip")  # check this if ok
                else:
                    for t in range(start, end + 1):
                        p1 = p
                        p2 = p
                        while np.isnan(df.iloc[t - p1, col]):
                            p1 += p
                        while np.isnan(df.iloc[t + p2, col]):
                            p2 += p

                        seas[t] = (df.iloc[t - p1, col] + df.iloc[t + p2, col]) / 2
                    trend1 = np.poly1d(
                        np.polyfit([start, end], [seas[start], seas[end]], 1)
                    )
                    trend2 = np.poly1d(
                        np.polyfit(
                            [start - 1, end + 1],
                            [df.iloc[start - 1, col], df.iloc[end + 1, col]],
                            1,
                        )
                    )

                    for t in range(start, end + 1):
                        df.iloc[t, col] = seas[t] - trend1(t) + trend2(t)
    return df


def fill_if_missing(df):
    """
    If the given pandas.DataFrame has any NaN values, they are replaced with interpolated values

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas.DataFrame for which NaN values need to be replaced by interpolated values

    Returns
    -------
    pandas.DataFrame
        A pandas.DataFrame with no NaN values
    """
    if df.isnull().values.any():
        print("Some values are NaN. They are being filled...")
        custom_interpolate(df)
        print("...interpolation finished! No missing data left.")
    else:
        print("No missing data \n")
    return df


def extract(df, horizon, anchor_key=0, filter_df=False):
    """
    Extract data from the input DataFrame and reshape it into a suitable input form for a LSTM cell

    The input DataFrame is reshaped into an ndarray with a number of entries (samples), such that each entry
    contains n = 'horizon' rows (including all features) from the input DataFrame. There are i = 1,...,m entries in the
    output ndarray, with m such that row (m + n - 1) is the final row from the input DataFrame.
    e.g. The first entries in the ndarray would be:

    [row 1, row 2, ..., row n], [row 2, row 3, ..., row n+1], etc.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame
    horizon : int
        The horizon/forecast length

    Returns
    -------
    ndarray
        The reshaped data in a numpy array (Shape: (number of samples, horizon, number of features))
    """

    number_of_samples = df.shape[0] - horizon + 1
    idx_remove = []
    if number_of_samples <= 0:
        number_of_samples = 1
    if df.ndim > 1:
        number_of_features = df.shape[1]
        reshaped_data = np.empty([number_of_samples, horizon, number_of_features])
        for i in range(number_of_samples):
            reshaped_data[i, :, :] = df.iloc[i : i + horizon, :]
            if isinstance(anchor_key, str) and anchor_key not in df.iloc[i].name:
                idx_remove.append(i)
    else:
        reshaped_data = np.empty([number_of_samples, horizon])
        for i in range(number_of_samples):
            reshaped_data[i, :] = df.iloc[i : i + horizon]
            if isinstance(anchor_key, str) and anchor_key not in df.iloc[i].name:
                idx_remove.append(i)
    if isinstance(anchor_key, str):
        reshaped_data = np.delete(reshaped_data, idx_remove, axis=0)
    if filter_df:
        filtered_df = df
        filtered_df = filtered_df.drop(filtered_df.index[idx_remove], inplace=True)
        return reshaped_data, filtered_df
    return reshaped_data


def scale(df, scaler):
    """
    Scales the given pandas.DataFrame using the specified scikit-learn scaler

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame
    scaler : sklearn.preprocessing scaler
        The scikit-learn scaler used by the model while training

    Returns
    -------
    pandas.DataFrame
        The scaled DataFrame
    """
    df_new = scaler.transform(df)
    df_new = pd.DataFrame(df_new, columns=df.columns)
    return df_new


def rescale(values, scaler):
    """
    Scale the given data back to its original representation

    Parameters
    ----------
    values : array-like, sparse matric of shape (n samples, n features)
        The data to be rescaled
    scaler : sklearn.preprocessing scaler
        The scaler that was used to scale the data originally

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the rescaled data
    """
    df_rescaled = pd.DataFrame(scaler.inverse_transform(values))
    return df_rescaled


def rescale_manually(net, output, targets, target_position=0, **PAR):
    """
    Manually rescales data that was previously scaled

    Parameters
    ----------
    net : plf_util.fc_network.EncoderDecoder
        The model that was used to generate the predictions.
    output : list
        A list containing predicted values. Each entry in the list is a set of predictions
    targets : torch.Tensor
        The actual or true values
    target_position : int, default = 0
        Which column of the data to rescale
    **PAR : dict
        A dictionary containing config parameters, see fc_train.py for more.

    Returns
    -------
    torch.Tensor
        The expected values of the prediction, after rescaling
    torch.Tensor
        The targets (untransformed)
    """
    #TODO: isn't this also in a function of datatuner
    #TODO: finish documentation
    #get parameters 'scale' and 'center'
    for group in PAR["feature_groups"]:
        if group["features"] is not None and PAR["target_id"] in group["features"]:
            scaler_name = group["name"]
            scaler = net.scalers[scaler_name]
            scale = scaler.scale_.take(target_position)
            break  # assuming target column can only be scaled once

    if type(scaler) == RobustScaler:
        # customized inversion robust_scaler
        if scaler.center_.any() == False:  # no center shift applied
            center = 0
        else:
            center = scaler.center_.take(target_position)
        scale = scaler.scale_.take(
            target_position
        )  # The (scaled) interquartile range for each feature in the training set
    elif type(scaler) == StandardScaler:
        if scaler.mean_.take(target_position) == None:
            center = 0
        else:
            center = scaler.mean_.take(target_position)
        scale = scaler.scale_.take(target_position)
    elif type(scaler) == MinMaxScaler:
        range_min = scaler.feature_range[0]
        range_max = scaler.feature_range[1]
        data_max = scaler.data_max_.take(target_position)
        data_min = scaler.data_min_.take(target_position)
        scale = (data_max - data_min) / (range_max - range_min)
        center = data_min - range_min * scale
    #TODO: else options
    
    # rescale
    loss_type = net.criterion  # check the name here
    targets_rescaled = (targets * scale) + center
    if loss_type == "pinball":
        expected_values = (output[1] * scale) + center
        quantiles = (output[0] * scale) + center
    elif loss_type == "nll_gauss" or loss_type == "crps":
        expected_values = (output[0] * scale) + center
        variances = output[1] * (scale ** 2)
    # TODO: else options

    return expected_values, targets


def scale_all(df: pd.DataFrame, feature_groups, start_date=None, scalers={}, **_):
    """
    Scale and return the specified feature groups of the given DataFrame, each with their own
    scaler, beginning at the index 'start_date'

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame with the data to be scaled
    feature_groups : array
        An array of dicts. Each dict has entries with the following keywords:

        - "name", stores the name of the feature group
        - "scaler", stores a list in which the first entry is the name of the feature
        group's scaler. Valid names are 'standard', 'robust' or 'minmax'. Additional
        entries in the list are for scaler parameters.
        - "features", stores a list with the names of the features belonging to the feature group
    start_date : int, default = None
        The index of the date from which to begin scaling

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the scaled features/targets
    dict
        A dict of sklearn.preprocessing scalers with their corresponding feature group
        names (e.g."main", "add") as keywords

    Raises
    ------
    RuntimeError
        Raised when no scaler could be generated - invalid scaler name in config file.
    """
    # grouping should be an array of dicts.
    # each dict defines a scaler and the features to be scaled
    # returns a list of dataframes with the scaled features/targets, and the according scalers in a dict defined by the "name" keyword.
    # TODO should these be named for potential double scaling (name can be used as suffix in join)
    # TODO check if it is critical that we do not use fitted scalers in evaluate script
    scaled_features = pd.DataFrame(index=df.index)[start_date:]
    # scalers = {}
    for group in feature_groups:
        df_to_scale = df.filter(group["features"])[start_date:]

        if group["name"] in scalers:
            scaler = scalers[group["name"]]
        else:
            scaler = None
            if group["scaler"] is None or group["scaler"][0] is None:
                if group["name"] != "aux":
                    print(
                        group["name"]
                        + " features were not scaled, if this was unintentional check the config file."
                    )
            elif group["scaler"][0] == "standard":
                scaler = StandardScaler()
            elif group["scaler"][0] == "robust":
                scaler = RobustScaler(
                    quantile_range=(group["scaler"][1], group["scaler"][2])
                )
            elif group["scaler"][0] == "minmax":
                scaler = MinMaxScaler(
                    feature_range=(group["scaler"][1], group["scaler"][2])
                )
            else:
                raise RuntimeError("scaler could not be generated")
            if scaler is not None:
                scaler.fit(df_to_scale)

        if group["features"] is not None:
            df_to_scale = df.filter(group["features"])[start_date:]
            if scaler is not None:
                add_scaled_features = pd.DataFrame(
                    scaler.transform(df_to_scale),
                    columns=df_to_scale.columns,
                    index=df_to_scale.index,
                )
                scaled_features = scaled_features.join(
                    add_scaled_features
                )  # merge differently scaled dataframes
            else:
                scaled_features = scaled_features.join(df_to_scale)

            scalers[group["name"]] = scaler
    return scaled_features, scalers


def constructDf(
    data,
    columns,
    train_split,
    forecast_horizon,
    history_horizon,
    interval=1,
    number_forecasts=0,
    limit_memory=False,
):
    """
    Construct and reorder data for training

    Parameters
    ----------
    data : pandas.DataFrame with all data
        The DataFrame containing all data to be reordered
    columns : list
        List of columns used in model target+exog
    train_split : float
        Fraction of data to use for training
    forecast_horizon : int
        The number of forecast steps into the future
    history_horizon : int
        The size of the history horizon
    interval : int, default = 1
        The number of time_steps between every forecast. By default, forecast with moving
        window of 1 timestep
    number_forecasts : int, default = 0
        The number of forecast.  By default, forecast over whole test-period
    limit_memory : bool, default = False
        Set to True when the history horizon should limit the recent memory

    Returns
    -------
    List
         A list of DataFrames for input
    List
        A list of DataFrames for output
    """
    train = []
    test = []

    train_test_split = int(train_split * len(data))
    df = data[columns]
    recent_memory = 0
    if number_forecasts == 0:
        number_forecasts = len(data)

    if history_horizon != 0 and limit_memory:
        recent_memory = max(0, train_test_split - history_horizon)

    for i in range(number_forecasts):
        forecast_start = train_test_split + 1 + interval * i
        if (forecast_start + forecast_horizon) > len(df):
            # break when end of test period is reached in next iteration.
            break
        train.append(
            pd.DataFrame(df.iloc[recent_memory:forecast_start], columns=df.columns)
        )
        test.append(
            pd.DataFrame(
                df.iloc[forecast_start : forecast_start + forecast_horizon],
                columns=df.columns,
            )
        )
    return train, test
