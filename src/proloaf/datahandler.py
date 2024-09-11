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

Includes functionality for scaling, rescaling, filling missing values etc.
Some functions are no longer used directly in the project, but may nonetheless be
useful for testing or future applications.
"""

from datetime import datetime
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.validation import check_is_fitted
import proloaf.tensorloader as tl
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from proloaf.event_logging import create_event_logger

logger = create_event_logger(__name__)


def load_raw_data_xlsx(files, path):
    """
    Load data from an xlsx file

    After loading, the date column in the raw data is converted to a UTC datetime

    Parameters
    ----------
    files : list
        A list of files to read. See the Notes section for more information
    path : string
        The path specification which holds the input files in .XLSX format

    Returns
    -------
    list
        A list containing a DataFrame for each file that was read

    Notes
    -----
    - Files is an array of maps containing the following data with the keyword (keyword)
        + ('file_name') the name of the xlsx file
        + ('date_column') the name of the date_column in the raw_data
        + ('time_zone') specifier for the timezone the raw data is recorded in
        + ('sheet_name') name or list of names of the sheets that are to be read
        + ('combine') boolean, all datasheets with true are combined into one, all others are read individually
        + ('start_column') Columns between this and ('end_column') are loaded
        + ('end_column')

    """
    logger.info("Importing XLSX Data...")

    combined_files = []
    individual_files = []

    for xlsx_file in files:
        logger.info("importing " + xlsx_file["file_name"])
        date_column = xlsx_file["date_column"]
        raw_data = pd.read_excel(
            path + xlsx_file["file_name"],
            xlsx_file["sheet_name"],
            parse_dates=[date_column],
        )

        # convert load data to UTC
        if xlsx_file["time_zone"] != "UTC":
            raw_data[date_column] = (
                pd.to_datetime(raw_data[date_column])
                .dt.tz_localize(xlsx_file["time_zone"], ambiguous="infer")
                .dt.tz_convert("UTC")
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            if xlsx_file["dayfirst"]:
                raw_data[date_column] = pd.to_datetime(
                    raw_data[date_column], format="%d-%m-%Y %H:%M:%S"
                ).dt.tz_localize(None)
            else:
                raw_data[date_column] = pd.to_datetime(
                    raw_data[date_column], format="%Y-%m-%d %H:%M:%S"
                ).dt.tz_localize(None)

        if xlsx_file["data_abs"]:
            raw_data.loc[:, xlsx_file["start_column"] : xlsx_file["end_column"]] = raw_data.loc[
                :, xlsx_file["start_column"] : xlsx_file["end_column"]
            ].abs()
        # rename column IDs, specifically Time, this will be used later as the df index
        raw_data.rename(columns={date_column: "Time"}, inplace=True)
        raw_data.head()  # now the data is positive and set to UTC
        raw_data.info()
        # interpolating for missing entries created by asfreq and original missing values if any
        raw_data.interpolate(method="time", inplace=True)

        if xlsx_file["combine"]:
            combined_files.append(raw_data)
        else:
            individual_files.append(raw_data)
    if len(combined_files) > 0:
        individual_files.append(pd.concat(combined_files))
    return individual_files


def load_raw_data_csv(files: List[Dict[str, Any]], path: str):
    """
    Load data from a csv file

    After loading, the date column in the raw data is converted to a UTC datetime

    Parameters
    ----------
    files : list
        A list of files to read. See the Notes section for more information
    path : str
        The path specification which holds the input files in .CSV format

    Returns
    -------
    list
        A list containing a DataFrame for each file that was read

    Notes
    -----
    - Files is an array of maps containing the following data with the keyword (keyword)
        + ('file_name') the name of the load_file
        + ('date_column') the name of the date_column in the raw_data
        + ('dayfirst') specifier for the formatting of the read time
        + ('sep') separator used in this file
        + ('combine') boolean, all datasheets with true are combined into one, all others are read individually
        + ('use_columns') list of columns that are loaded

    """

    logger.info("Importing CSV Data...")

    combined_files = []
    individual_files = []

    for csv_file in files:
        logger.info("Importing " + csv_file["file_name"] + " ...")
        date_column = csv_file["date_column"]
        raw_data = pd.read_csv(
            path + csv_file["file_name"],
            sep=csv_file["sep"],
            usecols=csv_file["use_columns"],
            parse_dates=[date_column],
            dayfirst=csv_file["dayfirst"],
        )
        # pd.read_csv(INPATH + name, sep=sep, usecols=cols, parse_dates=[date_column] , dayfirst=dayfirst)
        if csv_file["time_zone"] != "UTC":
            raw_data[date_column] = (
                pd.to_datetime(raw_data[date_column])
                .dt.tz_localize(csv_file["time_zone"], ambiguous="infer")
                .dt.tz_convert("UTC")
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            if csv_file["dayfirst"]:
                raw_data[date_column] = pd.to_datetime(
                    raw_data[date_column], format="%d-%m-%Y %H:%M:%S"
                ).dt.tz_localize(None)
            else:
                raw_data[date_column] = pd.to_datetime(
                    raw_data[date_column], format="%Y-%m-%d %H:%M:%S"
                ).dt.tz_localize(None)

        logger.info("...Importing finished. ")
        raw_data.rename(columns={date_column: "Time"}, inplace=True)

        if csv_file["combine"]:
            combined_files.append(raw_data)
        else:
            individual_files.append(raw_data)

    if len(combined_files) > 0:
        individual_files.append(pd.concat(combined_files, sort=False))
    return individual_files


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates and adds trionemetric values to the DataFrame in respect to the index 'Time'.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame that is complemented with cyclical time features

    Returns
    -------
    df
        The modified DataFrame

    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("data frame is not a timeseries, cylical time features can not be generated.")
    ## source http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    df["hour_sin"] = np.sin(df.index.hour * (2.0 * np.pi / 24))
    df["hour_cos"] = np.cos(df.index.hour * (2.0 * np.pi / 24))
    df["weekday_sin"] = np.sin((df.index.weekday) * (2.0 * np.pi / 7))
    df["weekday_cos"] = np.cos((df.index.weekday) * (2.0 * np.pi / 7))
    df["mnth_sin"] = np.sin((df.index.month - 1) * (2.0 * np.pi / 12))
    df["mnth_cos"] = np.cos((df.index.month - 1) * (2.0 * np.pi / 12))
    return df


def add_missing_features(df: pd.DataFrame, all_columns: List[str]):
    for col in all_columns:
        if col not in df.columns:
            df[col] = 0
    return df


def add_onehot_features(df):
    """
    Generates and adds one-hot encoded values to the DataFrame in respect to the index 'Time'.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame that is complemented with one-hot coded time features

    Returns
    -------
    df
        The modified DataFrame

    """
    # add one-hot encoding for Hour, Month & Weekdays
    hours = pd.get_dummies(df.index.hour, prefix="hour").set_index(df.index)  # one-hot encoding of hours
    month = pd.get_dummies(df.index.month, prefix="month").set_index(df.index)  # one-hot encoding of month
    weekday = pd.get_dummies(df.index.dayofweek, prefix="weekday").set_index(df.index)  # one-hot encoding of weekdays
    # df = pd.concat([df, hours, month, weekday], axis=1)
    df[hours.columns] = hours
    df[month.columns] = month
    df[weekday.columns] = weekday
    # logger.info(df.head())
    return df


def check_continuity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raises value error upon violation of continuity constraint of the timeseries data.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose index shall be checked against time continuity

    Returns
    -------
    ValueError
        If index of DataFrame is not a continous datetime index.

    """
    if not hasattr(df.index, "freq") or not df.index.equals(
        pd.date_range(min(df.index), max(df.index), freq=df.index.freq)
    ):
        raise ValueError("DateTime index is not continuous")
    return df


def check_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose values shall be checked against missing values

    Returns
    -------
    pandas.DataFrame
        Same DataFrame as input to allow pipeline structure

    Raises
    ------
    ValueError
        If DataFrame contains NaN's
    """
    if not df.isnull().values.any():
        logger.info("No missing data \n")
    else:
        logger.warning("Missing data detected \n")
        raise ValueError("DataFrame contains NaN's")
    return df


def set_to_hours(df: pd.DataFrame, timecolumn="Time", freq="h") -> pd.DataFrame:
    """
    Sets the index of the DataFrame to 'Time' and the frequency to hours.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose index and frequency are to be changed

    Returns
    -------
    df
        The modified DataFrame

    """
    if df.index.name != timecolumn:
        df.set_index(timecolumn, inplace=True)

    df.index = pd.to_datetime(df.index)
    df = df.asfreq(freq=freq)
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


def custom_interpolate(df: pd.DataFrame, periodicity=1) -> pd.DataFrame:
    """
    Interpolate the features with missing values in a time series data frame

    For each feature/columns:

    - finds the range of intervals of missing values
    - for each of the missing value in these intervals
        + collect the previous day's value and the next day's value (t-24 & t+24) at that time instant
        + if any of them are missing, go for the next day(t-48 , t+48 and so on)
        + take their average
        + to account for the trend, shift the values by the slope of the interval extremes

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with missing values

    periodicity : int, default = 1
        An int value that allows the customized intepolation method, which makes use of the timeseries periodicity

    Returns
    -------
    pandas.DataFrame
        DataFrame with interpolated values
    """
    rows, columns = np.where(pd.isnull(df))  # all rows and columns with missing values
    # the missing values in "rows" are organized by row number, from first to last row
    # the values with the same index in "columns" are the corresponding columns, in which the value is missing
    # goal: find the ranges of missing values within each column, organized by columns:

    miss_rows_ranges = []  # all ranges of missing rows (for all columns)
    num_columns = df.shape[1]
    miss_columns = []  # all columns with missing rows, but in order from first column to last column
    for c in range(num_columns):  # for all possible columns
        miss_rows = rows[columns == c]  # each missing row for that column
        if len(miss_rows) != 0:  # if this column has missing values
            # find ranges of rows for that column and append to end of list
            new_ranges = ranges(miss_rows)  # all ranges of missing values for column c
            miss_rows_ranges.extend(new_ranges)  # add new ranges to list of all ranges of all columns
            for n in range(len(new_ranges)):
                # append x times the number of the column, where x is the number of ranges for that column
                miss_columns.append(c)  # corresponding columns to "miss_rows_ranges"

    last_index = df.shape[0] - 1
    assert df.iloc[last_index, 0] == df.iloc[-1, 0]  # should be the last value
    # loop over all ranges
    for col, (start, end) in zip(miss_columns, miss_rows_ranges):
        assert type(col) == int

        # dur = end - start
        p = periodicity  # periodicity
        seas = np.zeros(len(df))
        if start == end and end + 1 <= last_index and start - 1 >= 0:
            # if single point and previous and next value exists, take average of the nearby ones
            t = start
            df.iloc[t, col] = (df.iloc[t - 1, col] + df.iloc[t + 1, col]) / 2

        elif start == end and (end + 1 > last_index or start - 1 <= 0):
            # if single point, but the single point is at the beginning or end of the series, take the nearby one
            t = start
            if start - 1 <= 0:  # point is first value of the series
                df.iloc[t, col] = df.iloc[t + 1, col]
            if end + 1 > last_index:  # point is last value of the series
                df.iloc[t, col] = df.iloc[t - 1, col]
        else:
            # now we are dealing with a range
            if (start - p) <= 0 or (end + p) > (df.shape[0]):  # if the range is close to the beginning or end

                if start == 0:  # special case: range begins with first value
                    # fill all beginning NaN values with the next existing value
                    df.iloc[start : end + 1, col] = df.iloc[end + 1, col]
                elif end == last_index:  # special case: range ends on last value
                    # fill all last NaN values with the last existing value
                    df.iloc[start : end + 1, col] = df.iloc[start - 1, col]
                else:
                    new_column = df.iloc[:, col].interpolate(method="pchip")
                    df.iloc[start : end + 1, col] = new_column[start : end + 1]
            else:
                for t in range(start, end + 1):
                    p1 = p
                    p2 = p
                    while np.isnan(df.iloc[t - p1, col]):
                        p1 += p
                    while np.isnan(df.iloc[t + p2, col]):
                        p2 += p

                    seas[t] = (df.iloc[t - p1, col] + df.iloc[t + p2, col]) / 2
                trend1 = np.poly1d(np.polyfit([start, end], [seas[start], seas[end]], 1))
                trend2 = np.poly1d(
                    np.polyfit(
                        [start - 1, end + 1],
                        [df.iloc[start - 1, col], df.iloc[end + 1, col]],
                        1,
                    )
                )
                for t in range(start, end + 1):
                    df.iloc[t, col] = seas[t] - trend1(t) + trend2(t)
    test = np.where(pd.isnull(df))  # test if all values are filled
    assert len(test[0]) == 0
    assert len(test[1]) == 0
    return df


def fill_if_missing(df: pd.DataFrame, periodicity: int = 24) -> pd.DataFrame:
    """
    If the given pandas.DataFrame has any NaN values, they are replaced with interpolated values

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas.DataFrame for which NaN values need to be replaced by interpolated values

    periodicity : int, default = 24
        An int value that allows the customized intepolation method, which makes use of the timeseries periodicity

    Returns
    -------
    pandas.DataFrame
        A pandas.DataFrame with no NaN values
    """
    if df.isnull().values.any():
        logger.info("Some values are NaN. They are being filled...")
        df = custom_interpolate(df, periodicity)
        logger.info("...interpolation finished! No missing data left.")
    else:
        logger.info("No missing data")
    return df


class MultiScaler(sklearn.base.TransformerMixin):
    """
    SciKit-learn style scaler, that combines multiple scalers
    into one, then just applies them to their corresponding data columns.
    The MultiScaler is callable and will try to transform the data without fitting.
    If that false fit_tranform is called.

    Note: While this should work with numpy.ndarrays as input (dim 0 being the "columns")
    this is not tested and should only be used with caution.


    Parameters
    ----------
    feature_groups : List[Dict[str,Any]]
        An array of dicts. Each dict has entries with the following keywords:

        - "name", stores the name of the feature group
        - "scaler", stores a list in which the first entry is the name of the feature
        group's scaler. Valid names are 'standard', 'robust' or 'minmax'. Additional
        entries in the list are for scaler parameters.
        - "features", stores a list with the names of the features belonging to the feature group

    """

    def __init__(
        self,
        feature_groups: List[Dict] = None,
        scalers: Dict[str, sklearn.base.TransformerMixin] = None,
    ):
        if feature_groups is None:
            feature_groups = []
        self.feature_groups = feature_groups
        self._init_scalers()
        if scalers is not None:
            logger.warning("This is an experimental feature and might lead to unexpected behaviour without error")
            # TODO make sure input scalers are same type and parameters as definend in feature groups
            self.scalers.update(scalers)
            self._mark_if_fitted()

    @staticmethod
    def _extract_scaler_from_config(scaler_dict: Dict) -> sklearn.base.TransformerMixin:
        if scaler_dict[0] == "standard":
            return StandardScaler()
        elif scaler_dict[0] == "robust":
            return RobustScaler(quantile_range=(scaler_dict[1], scaler_dict[2]))
        elif scaler_dict[0] == "minmax":
            return MinMaxScaler(feature_range=(scaler_dict[1], scaler_dict[2]))
        else:
            raise RuntimeError(
                f"scaler could not be generated. {scaler_dict[0]} is not a known Identifier of a scaler."
            )

    def _init_scalers(self):
        self.scalers = {}
        for group in self.feature_groups:
            if "features" not in group or not group["features"]:
                logger.info(f"Feature group {group.get('name', 'UNNAMED')} has no features and will be skipped.")
                continue
            if group["scaler"] is None or group["scaler"][0] is None:
                if group.get("name") != "aux":
                    logger.info(
                        f"{group.get('name','UNNAMED')} features were not scaled, if this was unintentional check the config file."
                    )
                continue

            self.scalers.update(
                {feature: self._extract_scaler_from_config(group["scaler"]) for feature in group["features"]}
            )
        return self

    def fit(self, X: pd.DataFrame):
        """
        Fits the scaler to the data.

        Parameters
        ----------
        X: pandas.DataFrame
            Reference data to fit the scaler on.
        """
        for feature in X.columns:
            if feature in self.scalers:
                self.scalers[feature].fit(X[[feature]])
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, inplace: bool = False):
        """
        Transforms the data using fitted scalers.

        Parameters
        ----------
        X: pandas.DataFrame
            Data to be transformed.
        inplace: bool, default = False
            If True the input DataFrame X will be used to store the scaled data, if False the DataFrame will be copied.
            Careful, when inplace is True and transform raises an exception the input DataFrame might already be partially moddified.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            if not all scalers are fitted before transforming.
        """
        if not inplace:
            X = X.copy()
        for feature in X.columns:
            if feature in self.scalers:
                X[feature] = self.scalers[feature].transform(X[[feature]])
        return X

    def inverse_transform(self, X: pd.DataFrame, inplace=False):
        """
        Reverses the transformation with this scaler.

        Parameters
        ----------
        X: pandas.DataFrame
            Data to be transformed.
        inplace: bool, default = False
            If True the input DataFrame X will be used to store the scaled data, if False the DataFrame will be copied.
            Careful, when inplace is True and transform raises an exception the input DataFrame might already be partially moddified.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            if not all scalers are fitted before transforming.
        """
        if not inplace:
            X = X.copy()
        for feature in X.columns:
            if feature in self.scalers:
                X[feature] = self.scalers[feature].inverse_transform(X[[feature]])
        return X

    def manual_transform(
        self, X: pd.DataFrame, scale_as: str, shift: bool = True, inplace: bool = False
    ) -> pd.DataFrame:
        """
        Transforms the data using a specific scaler that was fitted before.

        Parameters
        ----------
        X: pandas.DataFrame
            Data to be transformed.
        scale_as: str
            Name of feature as which the input data X should be treated.
        shift: bool, default = True
            True coresponds to the normal scaler behaviour. False does not shift the values, this can be used to scale std. deviation or differences.
        inplace: bool, default = False
            If True the input DataFrame X will be used to store the scaled data, if False the DataFrame will be copied.
            Careful, when inplace is True and transform raises an exception the input DataFrame might already be partially moddified.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            if not all scalers are fitted before transforming.
        """
        if not inplace:
            X = X.copy()
        scaler = self.scalers[scale_as]
        offset = 0
        if not shift:
            # if we don't want to shift we reverse the shifting afterwards
            # all scalers are linear in sklearn (but not all transformers are which is where this breaks down)
            offset = scaler.transform([[0.0]])[0][0]
        X[:] = scaler.transform(X) - offset
        return X

    def manual_inverse_transform(
        self, X: pd.DataFrame, scale_as: str, shift: bool = True, inplace: bool = False
    ) -> pd.DataFrame:
        """
        Transforms the data using a specific scaler that was fitted before.

        Parameters
        ----------
        X: pandas.DataFrame
            Data to be transformed.
        scale_as: str
            Name of feature as which the input data X should be treated.
        shift: bool, default = True
            True coresponds to the normal scaler behaviour. False does not shift the values, this can be used to scale std. deviation or differences.
        inplace: bool, default = False
            If True the input DataFrame X will be used to store the scaled data, if False the DataFrame will be copied.
            Careful, when inplace is True and transform raises an exception the input DataFrame might already be partially moddified.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            if not all scalers are fitted before transforming.
        """
        if not inplace:
            X = X.copy()

        scaler = self.scalers[scale_as]
        offset = 0
        if not shift:
            # if we don't want to shift we reverse the shifting afterwards
            offset = scaler.inverse_transform([[0.0]])[0][0]
        X[:] = scaler.inverse_transform(X) - offset
        return X

    def __call__(self, X: pd.DataFrame, inplace: bool = False):
        """
        Transforms the data using fitted scalers if they are fitted or fits them otherwise.

        Parameters
        ----------
        X: pandas.DataFrame
            Data to be transformed.
        inplace: bool, default = False
            If True the input DataFrame X will be used to store the scaled data, if False the DataFrame will be copied.
            DO NOT use inplace=True if there is a chance that the scalers are partially fitted.
            This will not happen unless fitting fails or scalers are provided externanly.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            if not all scalers are fitted before transforming.
        """
        try:
            return self.transform(X, inplace=inplace)
        except sklearn.exceptions.NotFittedError:
            # XXX if inplace=True this can cause undefined behaviour
            # if some scalers are already fitted and others are not.
            return self.fit_transform(X, inplace=inplace)

    def _mark_if_fitted(self):
        for sc in self.scalers.values():
            try:
                check_is_fitted(sc)
            except sklearn.exceptions.NotFittedError:
                return
        self.is_fitted_ = True


def split(df: pd.DataFrame, splits: List[Union[float, datetime]]):
    """Splits a dataframe at the specified points.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe to be split
    splits: List[float || datetime]
        Relative points at which the DataFrame should be split (e.g. [0.8,0.9] or ["2019-06-01 01:00:00", "2020-01-31 16:00:00"]). Should be in ascending order.

    Returns
    -------
    List[pandas.DataFrame]
        List of DataFrames into which the input has been split.
    """
    split_index = [int(len(df) * split) if isinstance(split, float) else split for split in splits]
    intervals = zip([None, *split_index], [*split_index, None])
    return [df[a:b] for a, b in intervals]
