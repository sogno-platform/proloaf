---
title: "train.py"
linkTitle: "train.py"
date: 2020-03-09
weight: 2
description: >
  Basic information on the training script.
---
### Overview
This script trains an RNN model on previously prepared data, which is loaded into a pandas Dataframe from a csv file. 
Hyperparameter exploration using optuna is also possible if desired. The trained model is saved at the location 
specified under ` "output_path": ` in the corresponding config.json and can be loaded via torch.load() or evaluated by 
using the evaluate.py script. This script scales the data, loads a custom data structure and then generates and 
trains a neural net.

### Hyperparameter Exploration
Any training parameter is considered a hyper parameter as long as it is specified in either *config.json* or 
*tuning.json*. The latter is the standard file where the (so far) best found configuration is saved and should usually 
not be manually adapted unless new tests are for some reason not comparable to older ones (e.g. after changing the loss 
function).

### .json Config Files
#### *config.json*

The config should define parameters that are not considered hyperparameters. Anything considered a hyperparameter (see above) may be defined but will be overwritten in training. These are many parameters not listed here, the structure is evident from the config however. The not so obvious parameters are:
1. "feature_groups" (list) of dicts each specifying:
    1. "name": (str) name of the group for identification
    2. "scaler": (list or null) with [0] the name of a scaler, [1] and [2] the arguments for its creation, or null for unscaled features
    3. "features": (list) feature names that should be scaled by the corresponding scaler, no double listing supported.
    A feature that is not listed in any feature group is not considered by the net.
2. "encoder/decoder_features": (list) features that are visible to the encoder/decoder.

#### *tuning.json*

Here all settings considering hyperparameter exploration can be adjusted. Any hyper parameter that is to be explored should be defined here in the "settings" dict (example below). In addition to settings dict there are two settings:
1. "rel_base_path": (str or null) relative path to the base config w.r.t. the project folder. If the file does not exist, it will be created at the end to save the best performing setup. If the file exists, the base setup will be trained first to be compared with the others (or to train the net with the most performant parameters).
2. "number_of_tests": (int optional) number of nets that are trained and compared, defaults to 1 if not specified. This number includes the base setup if one is defined.
3. "settings" (dict of dicts): defines parameters as keys of this dict e.g. "learning_rate" could be a key of the settings dict and its value is either:
    1. A value, the parameter is set to this value for all iterations. If the value itself is supposed to be a list, it has to be a list inside a list otherwise it will be interpreted as list of values.
    2. A list of values, the list is cycled over for each test.
    3. A dict defining:
        1. "function": (str) a string defining the import to a function, e.g 'random.gauss' will import the function 'gauss' from the 'random' module. This function will be called multiple times if necessary to generate values. If the function generates a list, each value will be used exactly once before generating new values, making random.choices() a candidate to prevent too much repetition. This should work with custom functions, but this has not been tested.
        2. "kwargs": (dict) a dict containing all necessary arguments to call the function.

*Example:*
```json
{
  "number_of_tests": 100,
  "settings": {
      "learning_rate": {
          "function": "suggest_loguniform",
          "kwargs": {
              "name": "learning_rate",
              "low": 0.000001,
              "high": 0.0001
          }
      },
      "history_horizon": {
          "function": "suggest_int",
          "kwargs": {
              "name": "history_horizon",
              "low": 1,
              "high": 168
          }
      },
      "batch_size": {
          "function": "suggest_int",
          "kwargs": {
              "name": "batch_size",
              "low": 32,
              "high": 120
          }
    }
  }
}
```
Possible hyperparameters are:

target_column, encoder_features, decoder_features, max_epochs, learning_rate, batch_size,
shuffle, history_horizon, forecast_horizon,
train_split, validation_split, core_net,
relu_leak, dropout_fc, dropout_core,
rel_linear_hidden_size, rel_core_hidden_size,
optimizer_name, cuda_id

### Inputs
One of the following:
* ./data/opsd.csv
* ./data/<<user_provided_data>>.csv
* ./data/gefcom2017/<<station>>_data.csv (data not available as of March 2022, but present in earlier versions

### Outputs
* ./oracles/model_station

### Reference Documentation
If you need more details, please take a look at the [docs]({{< resource url="reference/proloaf/proloaf/src/train.html" >}}) for 
this script.
