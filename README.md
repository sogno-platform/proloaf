# CoordiNet Load Forecasting

Development of a t+40 hourly load prediction in the context of the H2020 EU CoordiNet project. This Repository includes preprocessed Gefcom2017 Data on which
a Model can be trained.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Requirements

To install all required packages run:
```
pip install -r requirements.txt
```

On low RAM machines the option
```
pip install -r requirements.txt --no-cache-dir
```
might be necessary.

## Running the code

The repo contains following modules:
* All scripts are located in the source folder.
* To start one of the scripts use 'python3 fc_script.py -s argument', where the argument is the path of the correspondig config file located in the model_ folders.
* To prepare input data (e.g. load and weather) from selected stations run ./source/fc_prep.py
* To train a recurrent model specifically with the current training setup optionally including hyperparameter exploration for the selected station run ./scripts/fc_train.py
* To predict the next 40 hours in reference to the current time and print the result in a csv file run ./source/fc_run.py
* To analyze the performance of the forecast: ./model_station/evaluation.py


## Scripts
This project contains 4 scripts that can be used in conjunction with one another or separatly.

### fc_prep.py
fc_prep.py transforms the data to a common format(pandas dataframe to csv) for all stations. It is no pre-requisite for training and only support to sort the input data, if desired.
This script for now only allows weather data in csv format and has not timezone support for it while only allow xlsx files for load. If your data does not match the criteria using a custom script that saves yourdata as pandas dataframe with datetimeindex to a csv file with ";" as separator will acomplish the same.

#### config
The prep config defines 4 parameters:
1. "data_path": (str) path to the outputfile, relative to the main directory w.r.t. the project folder.
2. "raw_path": (str) directory were all the raw data files are located w.r.t. the project folder.
3. "xlsx_files": (list) of dicts for each weather file. Each dict should define:
    1. "file_name": (str) full name of the file.
    2. "date_column": (str) name of the column which contains the date.
    3. "dayfirst": (boolean) wether the date format writes day first or not.
    4. "sep": (str) separator used in the raw_data csv.
    5. "combine": (boolean) all files that have true are appended to each other, w.r.t. time.
    6. "use_columns": (list or null) list of columns to use from the file, uses all columns if null.
4. "csv_files": (list) of dicts for each load file. Each dict should define:
    1. "file_name": (str) full name of the file
    2. "date_column": (str) name of the column which contains the date.
    3. "time_zone": (str) short for the timezone the data is recorded in.
    4. "sheet_name": (int, str, list, null) number starting at 0 or name or list of those of the sheet names that should be loaded, null corresponds to all sheets.
    5. "combine": (boolean) all files that have true are appended to each other, w.r.t. time.
    6. "start_column": (str) name of the first column affected by data_abs
    7. "end_column": (str) first column not affected by data_abs anymore
    8. "data_abs": (boolean) column between start_column and end_column.

### fc_train.py
Trains a neural net on prepared data loaded as pandas dataframe from a csv and/or xlsx file.
Holds hyperparameter exploration&tuning options if desired.
The trained net is saved afterwards and can be loaded via torch.load().
This script scales the data, then loads a custom datastructure and then generates and trains a neural net.

#### Hyper parameter exploration
Any training parameter is considered a hyper parameter as long as it is specified in either *config.json* or *tuning.json*.
The latter is the standard file where the so far best found configuartion is saved and should usually not be manually adapted unless new tests are for some reason not compareable to older ones (e.g. after changing the loss function).

#### config
*config.json*

The config should define an parameter that is not considered a hyperparameter. Anything considered a hyperparameter (see above) may be defined but will be overwritten in training. These are many parameters not listed here, the structure is evident from the config however. The not so obvious parameters are:
1. "feature_groups" (list) of dicts each specifying:
    1. "name": (str) name of the group for idefication
    2. "scaler" (list or null) [0] name of a scaler, [1]and [2] the arguments for its creation. null for unscaled features
    3. "features" (list) feature names that should be scaled by the corresponding scaler, no double listing supported.
    A feature that is not listed in any feature group is not considered by the net.
2. "encoder/decoder_features": (list) features that are visible to the encoder/decoder.

*fc_station_parameterhandler_config.json*

Here all settings considering hyperparameter exploration can be adjusted. Any hyper parameter that is to be explored should be defined here in the "settings" dict (example below). In addition to settings dict there are two settings:
1. "rel_base_path": (str or null) relative path to the base config w.r.t. the project folder. If the file does not exist it will be created in the end to safe the best performing setup. If the file exists the base setup will be trained first to be compared with the others (or to train the net with the most performant parameters).
2. "number_of_tests": (int optional) number of nets that are trained and compared, defaults to 1 if not specified. This number includes the base setup if one is defined.
3. "settings" (dict of dicts): defines parameters as keys of this dict e.g. "learning_rate" could be a key of the settings dict and its value is either:
    1. A value, the parameter is set to this value for all iterations. If the value itself is supposed to be a list it has to be a list in a list otherwise it will be interpreted as list of values.
    2. A list of values, the list is cycled over for each test.
    3. A dict defining:
        1. "function": (str) a string defining the import to a function e.g "random.gauss" will import the fucntion gauss from the random module. This function will be called multiple times if necessary to generate values. If the function generates a list each value will be used exactly once before generating new values, making random.choices() a candidate to prevent to much repitition. This should work with custom functions but this is not tested.
        2. "kwargs": (dict) a dict conatining all necassary arguments to call the function.

Example
```
{
    "rel_base_path": "model_targets/sege/tuning.json",
    "number_of_tests": 20,
    "settings": {
        "learning_rate": {
            "function": "random.gauss",
            "kwargs":{
                "mu": 0.001,
                "sigma": 0.0001
            }
        },
        "batch_size": {
            "function": "random.randint",
            "kwargs":{
                "a": 5,
                "b": 300
            }
        },
        "core_net" : ["gru","lstm"],
        "drop_out_core": 0.2
    }
}
```
Possible hyper parameters are:

target_id, learning_rate, batch_size,
relu_leak, dropout_fc, dropout_core,
rel_linear_hidden_size, rel_core_hidden_size

#### inputs
* ./data/Gefcom2017/Stations/station_data_clean.csv

#### outputs
* ./oracles/model_station
* ./oracles/load_forecast_station.csv

### fc_run.py (not supported since 06/2020)
This script loads a trained net and input data and generates a prediction based on the data

### fc_evaluate.py
Runs the the trained net on test data and evaluates the result.


#### inputs
* ./oracles/model_station
* ./oracles/load_forecast_station.csv

#### outputs
./oracles/eval_sege/


### Configs
* The script use per default config.json. There is an additional one concerning hyperparameter exploration/tuning.

* To adapt the behavior of the scipts for a certain station, change options in the config files located in the targets folders.
* Parameters can be added to the config files by adding or removing them by hand (standard json format) or using the config_maker.py. Add a line
```
par['parameter_name'] = value
```
in the config_maker.py file and it will be added under that name.
Then use
```
python3 config_maker.py --mod path_to_config
```
to apply the changes. The argument again is a station name or the already existing config file.
* Using
```
python3 config_maker.py --new path_to_config
```
Will clear the config file before applying changes so be careful with that.

## Authors

* **Gonca Gürses-Tran** - *developer* - [RWTH AACHEN - ACS](https://www.acs.eonerc.rwth-aachen.de/cms/E-ON-ERC-ACS/Das-Institut/Mitarbeiter/Energy-Flexibility-Management-Optimiza/~rhdv/Gonca-Guerses-Tran/?allou=1)

* **Florian Oppermann** - *developer* - [Software Development](https://git-ce.rwth-aachen.de/florian.oppermann)

* **Vinay Kanth Reddy Bheemreddy** - *developer* - [student researcher](https://git.rwth-aachen.de/vinay94)

* **Hendrik Flamme** - *developer* - [RWTH AACHEN - ACS](https://www.acs.eonerc.rwth-aachen.de/cms/E-ON-ERC-ACS/Das-Institut/Mitarbeiter/Energy-Management-Systems-Data-Analyti/~rydj/Flamme-Hendrik/?allou=1)

## License



## Acknowledgments

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement n° 824414.

[The CoordiNet Project](https://coordinet-project.eu/)

"TSO – DSO – Consumer: Large-scale demostrations of innovative grid services through demand response, storage and small-scale (RES) generation"
