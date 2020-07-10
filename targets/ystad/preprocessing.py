import sys
import pandas as pd
import numpy as np

def load_raw_data(type, date_column, start_column=None, end_column=None, file_name=None, sheet_name=None):
    # type, string like 'load data' or 'meteorological data'
    # file_name, string like 'Historical data Lund 2019-06-28 one hour avg 4 years.xlsx'
    # date_column, string like 'Tid'
    # start_column, string like 'Last2'
    # end_column string like 'Last 7'
    # sheet_name, string like 'sheet1'
    if type == 'load data':
        '''Read load data. The source file is assumed to be a xlsx file and to have an hourly resolution'''
        print('Importing Load Data...')
        raw_data = pd.read_excel(INPATH + file_name, sheet_name,
                                 parse_dates=[date_column])
        # convert load data to UTC
        
        raw_data[date_column] = pd.to_datetime(raw_data[date_column], format='%Y-%m-%d %H:%M:%S')#.dt.tz_localize('CET', ambiguous="infer").\
            #dt.tz_convert('UTC').dt.strftime('%Y-%m-%d %H:%M:%S')
        print('...Importing finished. ')
        raw_data_values = raw_data.loc[:, start_column:end_column]
        raw_data.loc[:, start_column:end_column] = raw_data_values.loc[:, start_column:end_column]
        # rename column IDs, specifically Time, this will be used later as the df index
        raw_data.rename(columns={date_column: 'Time'}, inplace=True)
        raw_data.head()  # now the data is positive and set to UTC
        raw_data.info()
        # interpolating for missing entries created by asfreq and original missing values if any
        raw_data.interpolate(method='time', inplace=True)
    elif type == 'load data CET':
        '''Read load data. The source file is assumed to be a xlsx file and to have an hourly resolution'''
        print('Importing Load Data...')
        raw_data = pd.read_excel(INPATH + file_name, sheet_name,
                                 parse_dates=[date_column])
        # convert load data to UTC
        raw_data[date_column] = pd.to_datetime(raw_data[date_column]).dt.tz_localize('CET', ambiguous="infer").\
            dt.tz_convert('UTC').dt.strftime('%Y-%m-%d %H:%M:%S')
        print('...Importing finished. ')
        raw_data_values = raw_data.loc[:, start_column:end_column]
        raw_data.loc[:, start_column:end_column] = raw_data_values.loc[:, start_column:end_column]
        # rename column IDs, specifically Time, this will be used later as the df index
        raw_data.rename(columns={date_column: 'Time'}, inplace=True)
        raw_data.head()  # now the data is positive and set to UTC
        raw_data.info()
        # interpolating for missing entries created by asfreq and original missing values if any
        raw_data.interpolate(method='time', inplace=True)
    elif type == 'meteorological data':
        '''Read temperature data. The source files are assumed to be csv, in hourly resolution. Windspeed and temperature'''
        print('Importing Meteorological Data...')
        # Using for loop
        frames = []
        for i in range(len(file_name)):
            weather = pd.read_csv(INPATH + file_name[i], sep=',', usecols=[0, 1, 2])
            frames.append(weather)
        print('...Importing finished. ')

        raw_data = pd.concat(frames)
        raw_data.rename(columns={date_column: 'Time'}, inplace=True)
        raw_data.head()
        raw_data.info()
    return raw_data


def set_to_hours(df):
    '''setting the frequency as required'''
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.set_index('Time')
    df = df.asfreq(freq='H')
    return df

def check_missing(df):
    '''Check for missing entries in the weather data'''
    if df.isnull().values.any():
        rows, columns = np.where(pd.isnull(df))
        missing_rows = sorted(set(rows))
        print('There are {}  missing values in your dataset'.format(len(missing_rows) * len(df.columns)))
        df.head()
        return True
    else:
        return False

def ranges(nums):
    '''get ranges of indices in which values are missing'''
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def cust_interpolate(df):
    '''custom interpolation of consecutively missing values like more than 24 hrs'''
    print('Filling missing values...')
    rows, columns = np.where(pd.isnull(df))
    miss_rows_ranges = ranges(rows)

    for i in range(len(miss_rows_ranges)):

        start, end = miss_rows_ranges[i]
        dur = end - start
        p = 24  # periodicity

        for col in range(len(df.columns)):
            seas = np.zeros(len(df))
            if start == end:  # if single point, take average of the nearby ones
                t = start
                df.iloc[t, col] = (df.iloc[t - 1, col] + df.iloc[t + 1, col]) / 2

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

    print('...interpolated.')
    return df

if __name__ == '__main__':
    # DEFINES
    INPATH = './eon-data/ystad-energi/'
    LOAD_FILE1 = '20190805_Ystad.xlsx'
    LOAD_FILE2 = 'Trafik.xlsx'
    WEATHER_FILES = ['weather_forecast_2017.csv', 'weather_forecast_2018.csv', 'weather_forecast_2019.csv']
    OUTPATH = './eon-data/'
    OUTFILE = OUTPATH + 'last_ystad.csv'
    # Prepare Load Data
    load_data = load_raw_data(type='load data', file_name=LOAD_FILE1,
                              date_column='Date', sheet_name='Ystad Energi')
    train_data = load_raw_data(type='load data CET', file_name=LOAD_FILE2,
                              date_column='MEVA_DATETO', sheet_name='Trafikverket Ystad')
    hourly_load = set_to_hours(df=load_data)
    hourly_train_load = set_to_hours(df=train_data)
    if check_missing(hourly_load):
        cust_interpolate(hourly_load)

    # Prepare Weather Data
    weather_data = load_raw_data(type='meteorological data', file_name=WEATHER_FILES, date_column='time')
    hourly_weather = set_to_hours(df=weather_data)
    if check_missing(hourly_weather):
        cust_interpolate(hourly_weather)

    # Merge load and weather data to one df
    ## Since traffic load data is recorded in the least interval, lets join all dfs based on its time index
    hourly_train_load = hourly_train_load[hourly_train_load.index[0]:hourly_load.index[-1]]
    hourly_load = hourly_load[hourly_train_load.index[0]:hourly_train_load.index[-1]]
    hourly_weather = hourly_weather[hourly_train_load.index[0]:hourly_train_load.index[-1]]
    df_list = [hourly_weather, hourly_load, hourly_train_load]
    df = pd.concat(df_list, axis=1)
    if not df.isnull().values.any():
        print('No missing data \n')
    df.head()

    # Add columns for Hour & Month - Cyclical Feature Engineering like in
    ## http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    df['hour_sin'] = np.sin(df.index.hour * (2. * np.pi / 24))
    df['hour_cos'] = np.cos(df.index.hour * (2. * np.pi / 24))
    df['mnth_sin'] = np.sin((df.index.month - 1) * (2. * np.pi / 12))
    df['mnth_cos'] = np.cos((df.index.month - 1) * (2. * np.pi / 12))
    # fetch back the datetime again
    hourly_load['T'] = df.index

    # add one-hot encoding for Hour & Month
    hours = pd.get_dummies(hourly_load['T'].dt.hour, prefix='hour')  # one-hot encoding of hours
    df = pd.concat([df, hours], axis=1)
    month = pd.get_dummies(hourly_load['T'].dt.month, prefix='month')  # one-hot encoding of month
    df = pd.concat([df, month], axis=1)
    weekday = pd.get_dummies(hourly_load['T'].dt.dayofweek, prefix='weekday')  # one-hot encoding of month
    df = pd.concat([df, weekday], axis=1)

    # store new df as csv
    df.head()
    df.to_csv(OUTFILE, sep=';', index=True)
