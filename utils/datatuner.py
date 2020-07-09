"""
   Handles dataframes and holds functions for scaling, rescaling, fill_missing...
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

def extract(df, horizon):

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
    number_of_features = df.shape[1]
    number_of_samples = df.shape[0] - horizon + 1
    if number_of_samples <= 0:
        number_of_samples = 1

    reshaped_data = np.empty([number_of_samples, horizon, number_of_features])
    for i in range(number_of_samples):
        reshaped_data[i, :, :] = df.iloc[i:i + horizon, :]
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