---
title: "fc_prep.py"
linkTitle: "fc_prep.py"
date: 2020-03-09
weight: 1
description: >
  Basic information on the script for data preparation.
---

fc_prep.py transforms the data to a common format(pandas dataframe as csv) for all stations.
This script for now only allows weather data in csv format and has not timezone support for it while it only allows xlsx files for load. If your data does not match the criteria using a custom script that saves your data as pandas dataframe with datetimeindex to a csv file with ";" as separator will accomplish the same.


### config
The prep config defines 4 parameters:
1. "data_path": (str) path to the outputfile, relative to the main directory w.r.t. the project folder.
2. "raw_path": (str) directory were all the raw data files are located w.r.t. the project folder.
3. "weather_files": (list) of dicts for each weather file. Each dict should define:
    1. "file_name": (str) full name of the file.
    2. "date_column": (str) name of the column which contains the date.
    3. "dayfirst": (boolean) wether the date format writes day first or not.
    4. "sep": (str) separator used in the raw_data csv.
    5. "combine": (boolean) all files that have true are appended to each other, w.r.t. time.
    6. "use_columns": (list or null) list of columns to use from the file, uses all columns if null.
4. "load_files": (list) of dicts for each load file. Each dict should define:
    1. "file_name": (str) full name of the file
    2. "date_column": (str) name of the column which contains the date.
    3. "time_zone": (str) short for the timezone the data is recorded in.
    4. "sheet_name": (int, str, list, null) number starting at 0 or name or list of those of the sheet names that should be loaded, null corresponds to all sheets.
    5. "combine": (boolean) all files that have true are appended to each other, w.r.t. time.
    6. "start_column": (str) name of the first column affected by data_abs
    7. "end_column": (str) first column not affected by data_abs anymore
    8. "data_abs": (boolean) column between start_column and end_column.

### inputs
* historical data provided by the user

### outputs
* data path as defined in the used config.json
