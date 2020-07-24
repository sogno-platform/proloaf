---
title: "fc_train.py"
linkTitle: "fc_train"
date: 2020-03-09
description: >
  Basic information on the training script.
---


## fc_train.py
This script trains a neural net on prepared data loaded as pandas dataframe from a csv file. Also does some hyperparameter exploration if desired. The trained net is safed afterwards and can be loaded via torch.load() or run by using the fc_run.yp script. This script scales the data, loads a custom datastructur and then generates and trains a neural net.

### Hyper parameter exploration
Any training parameter is considered a hyper parameter as long as it is specified in either *fc_station_parameterhandler_config.json* or *fc_station_tuned_hyper.json*. The latter is the standard file where the (so far) best found configuartion is saved. This file should usually not be manually adapted. If new tests are for some reason not compareable to older ones (e.g. after changing the loss function), it is sufficent to change *fc_station_parameterhandler_config.json* to use a different base_configuartion.

### config
*fc_station_train_config.json*

The config should define an parameter that is not considered a hyperparameter. Anything considered a hyperparameter (see above) may be defined but will be overwritten in training. These are many parameters not listed here, the structure is evident from the config however. The not so obvious parameters are:
1. "feature_groups" (list) of dicts each specifying:
    1. "name": (str) name of the group for identification
    2. "scaler" (list or null) [0] name of a scaler, [1] and [2] the arguments for its creation. null for unscaled features
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
```json
{
    "rel_base_path": "model_targets/sege/fc_sege_tuned_hyper_config.json",
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

target_column, encoder_features, decoder_features, max_epochs, learning_rate, batch_size,
shuffle, history_horizon, forecast_horizon,
train_split, validation_split, core_net,
relu_leak, dropout_fc, dropout_core,
rel_linear_hidden_size, rel_core_hidden_size,
optimizer_name, cuda_id

### inputs
* ./eon-data/last_station.csv

### outputs
* ./output/model_station
* ./output/load_forecast_station.csv
