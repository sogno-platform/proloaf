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

"""   Handles dataframes and holds functions for scaling, rescaling, fill_missing..."""

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
    def parser(x):
        return pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    print('> Load data from \'{}\'... '.format(data_path), end='')
    df = pd.read_excel(data_path, parse_dates=[0], date_parser=parser)
    print('done!')

    return df

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def custom_interpolate(df):
    """
    Interpolates the features with missing values in a time series data frame

    for each feature/columns,
    ---> finds the range of intervals of missing values
    ---> for each of the missing value in these intervals
        ---> collect the the previous day's value and the next day's value (t-24 & t+24) at that time instant
        ---> if any of them is missing, go for the next day(t-48 , t+48 as such)
        ---> take average of them
        ---> to account for the trend, shift the values by the slope of the interval extremes

    :param: data frame with missing values
    :return: data frame with interpolated values
    """
    rows, columns = np.where(pd.isnull(df))
    miss_rows_ranges = ranges(rows)

    for i in range(len(miss_rows_ranges)):

        start, end = miss_rows_ranges[i]
        #dur = end - start
        p = 24  # periodicity

        for col in range(len(df.columns)):
            seas = np.zeros(len(df))
            if start == end and end+1<=df.shape[0] and start-1>=0:  # if single point, take average of the nearby ones
                t = start
                df.iloc[t, col] = (df.iloc[t - 1, col] + df.iloc[t + 1, col]) / 2
            else:
                if (start-p) <=0 or (end+p)>(df.shape[0]):
                    df = df.interpolate(method='pchip') # check this if ok
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
                        np.polyfit([start - 1, end + 1], [df.iloc[start - 1, col], df.iloc[end + 1, col]], 1))

                    for t in range(start, end + 1):
                        df.iloc[t, col] = seas[t] - trend1(t) + trend2(t)
    return df

def fill_if_missing(df):
    if df.isnull().values.any():
        print('Some values are NaN. They are being filled...')
        custom_interpolate(df)
        print('...interpolation finished! No missing data left.')
    else:
        print('No missing data \n')
    return df

def extract(df, horizon, anchor_key=0, filter_df=False):

    """
    Extracts data from the input data frame and reshapes into suitable input form for LSTM cell

    Parameters
    ----------
    df      : pandas.dataframe
              input dataframe

    horizon : int
              horizon/forecast length

    Returns
    -------
    reshaped_data: nd-array
                   numpy input array
    """

    number_of_samples = df.shape[0] - horizon + 1
    idx_remove=[]
    if number_of_samples <= 0:
        number_of_samples = 1
    if df.ndim > 1:
        number_of_features = df.shape[1]
        reshaped_data = np.empty([number_of_samples, horizon, number_of_features])
        for i in range(number_of_samples):
            reshaped_data[i, :, :] = df.iloc[i:i + horizon, :]
            if isinstance(anchor_key, str) and anchor_key not in df.iloc[i].name:
                idx_remove.append(i)
    else:
        reshaped_data = np.empty([number_of_samples, horizon])
        for i in range(number_of_samples):
            reshaped_data[i, :] = df.iloc[i:i + horizon]
            if isinstance(anchor_key, str) and anchor_key not in df.iloc[i].name:
                idx_remove.append(i)
    if isinstance(anchor_key, str):
        reshaped_data = np.delete(reshaped_data, idx_remove, axis=0)
    if filter_df:
        filtered_df=df
        filtered_df=filtered_df.drop(filtered_df.index[idx_remove], inplace=True)
        return reshaped_data, filtered_df
    return reshaped_data

def scale(df, scaler):
    """

    :param df: input dataframe
    :param scaler: scikit-learn scaler used by the model while training
    :return: scaled dataframe
    """
    df_new = scaler.transform(df)
    df_new = pd.DataFrame(df_new, columns=df.columns)
    return df_new

def rescale(values, scaler):
    # rescaling
    df_rescaled = pd.DataFrame(scaler.inverse_transform(values))
    return df_rescaled

def rescale_manually(net, output, targets, target_position=0, **PAR):
    #TODO: isn't this also in a function of datatuner
    #get parameters 'scale' and 'center'
    for group in PAR['feature_groups']:
        if group['features'] is not None and PAR['target_id'] in group['features']:
            scaler_name = group['name']
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
            target_position)  # The (scaled) interquartile range for each feature in the training set
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

    #rescale
    loss_type = net.criterion # check the name here
    targets_rescaled = (targets * scale) + center
    if loss_type == 'pinball':
        expected_values = (output[1]* scale)+center
        quantiles = (output[0]* scale)+center
    elif loss_type == 'nll_gauss' or loss_type == 'crps':
        expected_values = (output[0]* scale)+center
        variances = output[1]* (scale**2)
    #TODO: else options

    return expected_values, targets

def scale_all(df:pd.DataFrame, feature_groups, start_date = None, **_):
    # grouping should be an array of dicts.
    # each dict defines a scaler and the featrues to be scaled
    # returns a list of dataframes with the scaled features/targets, and the according scalers in a dict defined by the "name" keyword.
    # TODO should these be named for potential double scaling (name can be used as suffix in join)
    # TODO check if it is critical that we do not use fitted scalers in evaluate script
    scaled_features = pd.DataFrame(index=df.index)[start_date:]
    scalers = {}
    for group in feature_groups:
        df_to_scale = df.filter(group['features'])[start_date:]
        scaler = None

        if (group['scaler'] is None or group['scaler'][0] is None):
            if (group['name'] != 'aux'):
                print(group['name'] + ' features were not scaled, if this was unintentional check the config file.')
        elif(group['scaler'][0] == 'standard'):
            scaler = StandardScaler()
        elif(group['scaler'][0] == 'robust'):
            scaler = RobustScaler(quantile_range = (group['scaler'][1],group['scaler'][2]))
        elif(group['scaler'][0] == 'minmax'):
            scaler = MinMaxScaler(feature_range=(group['scaler'][1],group['scaler'][2]))
        else:
            raise RuntimeError("scaler could not be generated") 

        if group['features'] is not None:
            df_to_scale = df.filter(group['features'])[start_date:]
            if scaler is not None: 
                add_scaled_features = pd.DataFrame(scaler.fit_transform(df_to_scale), columns=df_to_scale.columns, index=df_to_scale.index)
                scaled_features = scaled_features.join(add_scaled_features)  #merge differently scaled dataframes
                scalers[group['name']] = scaler
            else:
                scaled_features = scaled_features.join(df_to_scale)
                
    return scaled_features, scalers

def constructDf(data, columns, train_split, forecast_horizon, history_horizon,
                interval=1, number_forecasts=0, limit_memory=False):
    """
    Constructs and re-orders data for training

    Parameters
    ----------
    data                     : DataFrame with all data

    columns                  : columns used in model target+exog

    forecast_time            : str Time where forecasts should start

    forecast_horizon         : number of forecast steps into the future

    number_forecasts         : number of forecasts er default 0 -->forecast over whole test-period

    interval                 : number of time_steps between every forecast, default 1 -->simulate forecast with moving
                                window of 1 timestep

    recent_memory            : number of past time_steps if None all past data is used

    limit_memory             : true when history horizon shall limit the recent memory


    Returns
    -------
    1)input_matrix          : list of DataFrames for input
    2)output_matrix         : list of DataFrames for output

    """
    train = []
    test = []

    train_test_split = int(train_split * len(data))
    df = data[columns]
    recent_memory = 0
    if number_forecasts == 0:
        number_forecasts = len(data)

    if history_horizon!=0 and limit_memory:
        recent_memory = max(0,train_test_split - history_horizon)

    for i in range(number_forecasts):
        forecast_start = train_test_split + 1 + interval * i
        if (forecast_start+forecast_horizon)>len(df):
            #break when end of test period is reached in next iteration.
            break
        train.append(pd.DataFrame(df.iloc[recent_memory:forecast_start], columns=df.columns))
        test.append(pd.DataFrame(df.iloc[forecast_start:forecast_start+forecast_horizon], columns=df.columns))
    return train, test