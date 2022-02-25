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

from typing import Dict, List
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.validation import check_is_fitted
import proloaf.tensorloader as tl
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from proloaf.cli import create_event_logger

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
        # if isinstance(file_name, str):
        #     file_name = [file_name,'UTC']
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
            raw_data.loc[
                :, xlsx_file["start_column"] : xlsx_file["end_column"]
            ] = raw_data.loc[
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


def load_raw_data_csv(files, path):
    """
    Load data from a csv file

    After loading, the date column in the raw data is converted to a UTC datetime

    Parameters
    ----------
    files : list
        A list of files to read. See the Notes section for more information
    path : string
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
    # for frame in individual_files:
    #    frame.rename(columns={date_column: 'Time'}, inplace=True)
    return individual_files


def add_cyclical_features(df):
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
    ## source http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    df["hour_sin"] = np.sin(df.index.hour * (2.0 * np.pi / 24))
    df["hour_cos"] = np.cos(df.index.hour * (2.0 * np.pi / 24))
    df["weekday_sin"] = np.sin((df.index.weekday) * (2.0 * np.pi / 7))
    df["weekday_cos"] = np.cos((df.index.weekday) * (2.0 * np.pi / 7))
    df["mnth_sin"] = np.sin((df.index.month - 1) * (2.0 * np.pi / 12))
    df["mnth_cos"] = np.cos((df.index.month - 1) * (2.0 * np.pi / 12))
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
    hours = pd.get_dummies(df.index.hour, prefix="hour").set_index(
        df.index
    )  # one-hot encoding of hours
    month = pd.get_dummies(df.index.month, prefix="month").set_index(
        df.index
    )  # one-hot encoding of month
    weekday = pd.get_dummies(df.index.dayofweek, prefix="weekday").set_index(
        df.index
    )  # one-hot encoding of weekdays
    # df = pd.concat([df, hours, month, weekday], axis=1)
    df[hours.columns] = hours
    df[month.columns] = month
    df[weekday.columns] = weekday
    # logger.info(df.head())
    return df


def check_continuity(df):
    """
    Raises value error upon violation of continuity constraint of the timeseries data.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose index shall be checked against time continuity

    Returns
    -------
    ValueError
        The error message

    """
    if not df.index.equals(
        pd.date_range(min(df.index), max(df.index), freq=df.index.freq)
    ):
        raise ValueError("DateTime index is not continuous")
    return df


def check_nans(df):
    """
    Print information upon missing values in the timeseries data.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose values shall be checked against missing values

    Returns
    -------
    Print
        Print message whether or not data is missing in the given DataFrame

    """
    if not df.isnull().values.any():
        logger.info("No missing data \n")
    else:
        logger.info("Missing data detected \n")

    return


def set_to_hours(df: pd.DataFrame, timecolumn="Time", freq="H"):
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

    logger.info("> Load data from '{}'... ".format(data_path), end="")
    df = pd.read_excel(data_path, parse_dates=[0], date_parser=parser)
    logger.info("done!")

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


def custom_interpolate(df, periodicity=1):
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

    periodicity : int, default = 1
        An int value that allows the customized intepolation method, which makes use of the timeseries periodicity

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
        p = periodicity  # periodicity

        for col in range(len(df.columns)):
            seas = np.zeros(len(df))
            if (
                start == end and end + 1 <= df.shape[0] and start - 1 >= 0
            ):  # if single point, take average of the nearby ones
                t = start
                try:
                    df.iloc[t, col] = (df.iloc[t - 1, col] + df.iloc[t + 1, col]) / 2
                except TypeError as err:
                    if df.iloc[t - 1, col] == df.iloc[t + 1, col]:
                        df.iloc[t, col] = df.iloc[t - 1, col]
                    else:
                        raise err
            elif start == end and (
                end + 1 > df.shape[0] or start - 1 <= 0
            ):  # if single point, but the single point is at the beginning or end of the series, take the nearby one
                t = start
                if start - 1 <= 0:
                    df.iloc[t, col] = df.iloc[t + 1, col]
                if end + 1 > df.shape[0]:
                    df.iloc[t, col] = df.iloc[t - 1, col]
            else:
                # now we are dealing with a range
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


def fill_if_missing(df, periodicity=1):
    """
    If the given pandas.DataFrame has any NaN values, they are replaced with interpolated values

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas.DataFrame for which NaN values need to be replaced by interpolated values

    periodicity : int, default = 1
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
        logger.info("No missing data \n")
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


def scale(df: pd.DataFrame, scaler: sklearn.base.TransformerMixin):
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
    try:
        df_new = scaler.transform(df)
    except sklearn.exceptions.NotFittedError:
        df_new = scaler.fit_transform(df)
    df_new = pd.DataFrame(df_new, columns=df.columns, index=df.index)
    return df_new


def rescale(values, scaler):
    """
    Scale the given data back to its original representation

    Parameters
    ----------
    values : array-like, sparse matrix of shape (n samples, n features)
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
    net : proloaf.models.EncoderDecoder
        The model that was used to generate the predictions.
    output : list
        A list containing predicted values. Each entry in the list is a set of predictions
    targets : torch.Tensor
        The actual or true values
    target_position : int, default = 0
        Which column of the data to rescale
    **PAR : dict
        A dictionary containing config parameters, see train.py for more.

    Returns
    -------
    torch.Tensor
        The targets (rescaled)
    list
        The expected values (for prob.: prediction intervals of the forecast, after rescaling, in form of output list)
    """
    # TODO: isn't this also in a function of datatuner
    # TODO: finish documentation
    # get parameters 'scale' and 'center'
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
    # TODO: else options

    # rescale
    loss_type = net.criterion  # check the name here
    targets_rescaled = (targets * scale) + center
    output_rescaled = output
    if loss_type == "pinball":
        output_rescaled[1] = (output[1] * scale) + center  # expected value
        output_rescaled[0] = (output[0] * scale) + center  # quantile
    elif loss_type == "nll_gauss" or loss_type == "crps":
        output_rescaled[1] = (output[0] * scale) + center  # expected value
        output_rescaled[0] = output[1] * (scale ** 2)  # variance
    else:  # case: rmse, case, mse, case rae etc.
        output_rescaled = (output * scale) + center
    # TODO: else options

    return targets_rescaled, output_rescaled


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
    feature_groups : List[Dict[str,any]]
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
            logger.warning(
                "This is an experimental feature and might lead to unexpected behaviour without error"
            )
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
                logger.info(
                    f"Feature group {group.get('name', 'UNNAMED')} has no features and will be skipped."
                )
                continue
            if group["scaler"] is None or group["scaler"][0] is None:
                if group.get("name") != "aux":
                    logger.info(
                        f"{group.get('name','UNNAMED')} features were not scaled, if this was unintentional check the config file."
                    )
                continue

            self.scalers.update(
                {
                    feature: self._extract_scaler_from_config(group["scaler"])
                    for feature in group["features"]
                }
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
        self, X: pd.DataFrame, scale_as: str, shift: bool = True, inplace=False
    ) -> pd.DataFrame:
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
        self, X: pd.DataFrame, scale_as: str, shift: bool = True, inplace=False
    ) -> pd.DataFrame:
        if not inplace:
            X = X.copy()

        scaler = self.scalers[scale_as]
        offset = 0
        if not shift:
            # if we don't want to shift we reverse the shifting afterwards
            offset = scaler.inverse_transform([[0.0]])[0][0]
        X[:] = scaler.inverse_transform(X) - offset
        return X

    def __call__(self, X: pd.DataFrame, inplace=False):
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


def scale_all(df: pd.DataFrame, feature_groups, start_date=None, scalers=None, **_):
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
    if scalers is None:
        scalers = {}
    for group in feature_groups:
        df_to_scale = df.filter(group["features"])[start_date:]

        if group["name"] in scalers:
            scaler = scalers[group["name"]]
        else:
            scaler = None
            if group["scaler"] is None or group["scaler"][0] is None:
                if group["name"] != "aux":
                    logger.warning(
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
                raise RuntimeError(
                    f"scaler could not be generated. {group['scaler']} is not a known Identifier of a scaler."
                )
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


# TODO not in use, deprecated?
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


def split(df: pd.DataFrame, splits: List[float]):
    split_index = [int(len(df) * split) for split in splits]
    intervals = zip([None, *split_index], [*split_index, None])
    logger.info(intervals)
    # intervals = zip(split_index, split_index)
    # print(list())
    return [df[a:b] for a, b in intervals]


def transform(
    df: pd.DataFrame,
    encoder_features,
    decoder_features,
    batch_size,
    history_horizon,
    forecast_horizon,
    target_id,
    train_split=0.7,
    validation_split=0.85,
    device="cpu",
    **_,
):
    """
    Construct tensor-data-loader transformed for encoderdecoder model input

    Parameters
    ----------
    TODO: check if consistent with new structure
    df : pandas.DataFrame
        The data frame containing the model features, to be split into sets for training
    encoder_features : string list
        A list containing desired encoder feature names as strings
    decoder_features : string list
        A list containing desired decoder feature names as strings
    batch_size : int scalar
        The size of a batch for the tensor data loader
    history_horizon : int scalar
        The length of the history horizon in hours
    forecast_horizon : int scalar
        The length of the forecast horizon in hours
    train_split : float scalar
        Where to split the data frame for the training set, given as a fraction of data frame length
    validation_split : float scalar
        Where to split the data frame for the validation set, given as a fraction of data frame length
    device : string
        defines whether to handle data with cpu or cuda

    Returns
    -------
    proloaf.tensorloader.CustomTensorDataLoader
        The training data loader
    proloaf.tensorloader.CustomTensorDataLoader
        The validation data loader
    proloaf.tensorloader.CustomTensorDataLoader
        The test data loader
    """

    split_index = int(len(df.index) * train_split)
    subsplit_index = int(len(df.index) * validation_split)

    df_train = df.iloc[0:split_index]
    df_val = df.iloc[split_index:subsplit_index]
    df_test = df.iloc[subsplit_index:]

    logger.info("Size training set: \t{}".format(df_train.shape[0]))
    logger.info("Size validation set: \t{}".format(df_val.shape[0]))

    # shape input data that is measured in the Past and can be fetched from UDW/LDW
    if 0 in df_train.shape:
        train_data_loader = None
    else:
        # train_data_loader = tl.TimeSeriesData(
        #     df_train,
        #     history_horizon=history_horizon,
        #     forecast_horizon=forecast_horizon,
        #     history_features=encoder_features,
        #     future_features=decoder_features,
        #     target_features=[target_id],
        #     preparation_steps=[],
        #     device=device,
        # )
        train_data_loader = tl.make_dataloader_wip(
            df_train,
            target_id,
            encoder_features,
            decoder_features,
            history_horizon=history_horizon,
            forecast_horizon=forecast_horizon,
            batch_size=batch_size,
        ).to(device)
    if 0 in df_val.shape:
        validation_data_loader = None
    else:
        validation_data_loader = tl.make_dataloader_wip(
            df_val,
            target_id,
            encoder_features,
            decoder_features,
            history_horizon=history_horizon,
            forecast_horizon=forecast_horizon,
            batch_size=batch_size,
        ).to(device)
    if 0 in df_test.shape:
        test_data_loader = None
    else:
        test_data_loader = tl.make_dataloader_wip(
            df_test,
            target_id,
            encoder_features,
            decoder_features,
            history_horizon=history_horizon,
            forecast_horizon=forecast_horizon,
            batch_size=1,
        ).to(device)

    return train_data_loader, validation_data_loader, test_data_loader
