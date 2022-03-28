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
Preprocesses your input data for use with ProLoaF

Transforms the data to a common format (pandas.DataFrame as csv) for all stations.

Notes
-----
- This script can load xlsx or csv files.
- If your data does not match the criteria, you can use a custom script that saves your
data as a pandas.DataFrame with datetimeindex to a csv file with a “;” as separator to
accomplish the same thing.

"""

import pandas as pd
import sys
import os
import proloaf.datahandler as dh

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)

from proloaf.confighandler import read_config
from proloaf.cli import parse_basic
from proloaf.event_logging import create_event_logger

logger = create_event_logger('preprocess')

if __name__ == "__main__":

    ARGS = parse_basic()
    config_file = os.path.join(MAIN_PATH, "targets", ARGS.station, "preprocessing.json")
    PAR = read_config(config_path=config_file)

    # DEFINES
    if PAR["local"]:
        INPATH = os.path.join(MAIN_PATH, PAR["raw_path"])
    else:
        INPATH = PAR["raw_path"]
    OUTFILE = os.path.join(MAIN_PATH, PAR["data_path"])

    # Prepare Load Data
    df_list = []
    if "xlsx_files" in PAR:
        xlsx_data = dh.load_raw_data_xlsx(files=PAR["xlsx_files"], path=INPATH)
        for data in xlsx_data:
            hourly_data = dh.set_to_hours(df=data)
            dh.fill_if_missing(hourly_data, periodicity=24)
            df_list.append(hourly_data)

    if "csv_files" in PAR:
        csv_data = dh.load_raw_data_csv(files=PAR["csv_files"], path=INPATH)
        for data in csv_data:
            hourly_data = dh.set_to_hours(df=data)
            dh.fill_if_missing(hourly_data, periodicity=24)
            print(hourly_data)
            df_list.append(hourly_data)

    logger.debug(df_list)
    # When concatenating, the arrays are filled with NaNs if the index is not available.
    # Since the DataFrames were already interpolated there are non "natural" NaNs left so
    # dropping all rows with NaNs finds the maximum overlap in indices
    # # Merge load and weather data to one df
    df = pd.concat(df_list, axis=1)

    df.dropna(inplace=True)

    dh.check_continuity(df)
    dh.check_nans(df)
    df.head()

    df = dh.add_cyclical_features(df)
    df = dh.add_onehot_features(df)

    # store new df as csv
    df.head()
    df.to_csv(OUTFILE, sep=";", index=True)
