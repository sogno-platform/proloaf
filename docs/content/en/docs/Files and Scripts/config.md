---
title: "config.json"
linkTitle: "config.json"
date: 2020-19-08
weight: 4
description: >
  Basic information about the config.json file
---

### Configuration
This page describes the 

#### Main Config
ProLoaF's configuration file is written using JSON. As such, whitespace is allowed and ignored in your syntax.
We mostly use strings, boolean, and ``null``, to specify the paths, and parameters of our targeted forecasting project. 
Numbers and booleans should be  unquoted.

For better readability, we assume in the examples below, 
that your working directory is set to the main project path of the cloned 
[repository](https://github.com/sogno-platform/proloaf). There is also template config file availble in the [targets folder](# TODO link).

We will split this config into multiple thematical parts and discuss all available options. 
<!-- TODO Config quick start? -->


#### Path Settings

Through the config file the user specifies the data source location, and the directories for logging, 
exporting performance analyses and most importantly, the trained RNN model binary.

**Path Specs:**
```json
    "data_path": "./data/<FILE-NAME>.csv",
    "evaluation_path": "./oracles/eval_<MODEL-NAME>/",
    "output_path": "./oracles/",
    "exploration_path": "./targets/opsd/tuning.json", 
    "log_path": "./logs/"
```
The output-, exploration- and log- paths may stay unchanged, but the data path and evaluation path must be specified.

* data_path: Specifies the path to the input data as csv.
* evaluation_path: The path where plots and data from the evaluation is stored when `evaluation.py` is run.
* output_path: Path to the folder where the model should be stored.
* exploration_path: Path to the tuning config. If no tuning is to be done (`exploration = false` see below.) this can be ommited (or `null`) <!-- TODO link to description -->
* log_path: Output to the logs from hyperparameter tuning.



**Data Specs**
```json
    "start_date": null,
    "history_horizon": 147,
    "forecast_horizon": 24,
    "frequency": "1h",
    "train_split": 0.6,
    "validation_split": 0.8,
    "periodicity": 24,
    "feature_groups": [
        {
            "name": "main",
            "scaler": [
                "minmax",
                0.0,
                1.0
            ],
            "features": [
                "AT_load_actual_entsoe_transparency",
                "DE_load_actual_entsoe_transparency",
                "DE_temperature",
                "DE_radiation_direct_horizontal",
                "DE_radiation_diffuse_horizontal"
            ]
        },
        {
            "name": "aux",
            "scaler": null,
            "features": [
                "hour_sin",
                "weekday_sin",
                "mnth_sin"
            ]
        }
    ],
```
* start_date: All value before this date will be cut off.
* history_horizon: Number of timesteps taken into account in the past to create a forecast. It is possible to change this value between training and running the model if an RNN model is used, this might however negatively impact performance.
* forecast_horizon: Number of timesteps that are being forecast as well as how many other forecasted values (from external sources) are considered for this. It is possible to change this value between training and running the model if an RNN model is used, this might however negatively impact performance.
* frequency: Frequency of the data, number and unit. For hourly data use `"1h"` for quarter-hourly `"15min"`. 
* train_split: Float between 0.0 and 1.0, should be smaller than `validation_split`. Gives the ratio of the whole dataset is to be used for training as opposed to validation and evaluataion. It is also possible to specify the split by time in the iso 8601 (`YYYY-MM-DD HH:MM:SS`) format.
* validation_split: Float between 0.0 and 1.0. should be larger than `train_split`. Defines the second split between validation data and evaluation data. It is also possible to specify the split by time in the iso 8601 (`YYYY-MM-DD HH:MM:SS`) format.
* periodicity: The expected period of the data in number of timesteps. For data following a daily cycle and 15 min frequency this should be 95 (= 24 * 4). This is used to extrapolate gaps in data.
* feature_groups: A list of json object that define groups of features and how they should be scaled. Features are usually scaled so the relative magnitude of their values (and chosen units) does not impact the forecasting model. Features are only grouped to reduce the number parameters that need to be set.
These groups consist of 
    * name: Some name, only for identifaction. The name "aux" implies that there should not be a warning for not scaling these features.
    * scaler: Name of the scaler, available are ["minmax"](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html), ["standard"](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and ["robust"](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html). The numbers following are the arguments given to its constructor. For more information refer to the provided links.
    * features: List of feature names that are to be scaled.

#### Model Settings
These settings describe the model itself, including inputs, outputs and model structure.

**Model Specs** 
```json
    "model_name": "opsd_recurrent",
    "target_id": [
        "DE_load_actual_entsoe_transparency",
        "AT_load_actual_entsoe_transparency"
    ],
    "encoder_features": [
        "AT_load_actual_entsoe_transparency",
        "DE_load_actual_entsoe_transparency",
        "DE_temperature",
        "DE_radiation_direct_horizontal",
        "DE_radiation_diffuse_horizontal"
    ],
    "decoder_features": [
        "DE_temperature"
    ],
    "aux_features": [
        "hour_sin",
        "weekday_sin",
        "mnth_sin"
    ],
    "model_class": "recurrent",
    "model_parameters": {
        "recurrent": {
            "core_net": "torch.nn.LSTM",
            "core_layers": 1,
            "dropout_fc": 0.4,
            "dropout_core": 0.3,
            "rel_linear_hidden_size": 1.0,
            "rel_core_hidden_size": 1.0,
            "relu_leak": 0.1
        },
        "simple_transformer": {
            "num_layers": 3,
            "dropout": 0.4,
            "n_heads": 6
        }
    }
```
* model_name: Any string that will be used to identify the model. Will be used for the filename of the model.
* target_id: List of column names that are to be predicted.
* encoder_features: Features that are known to the encoder. These are the features whose values from the past are available, i. e. the past values of the predicted feature.
* decoder_features: Features that are known to the decoder. These are features whose value lies in the future, this could be from a weather forecast or a previously decided schedule.
* aux_features: These features are known to both the decoder and the encoder, if you are using one-hot or geometrically encoded time features these can be included here. In the future these features might be added automatically when they are produced in the preprocessing.
* model_class: Type of model to be used, available are `"recurrent"`, `"simple_transformer"`, `"autoencoder"`, `"dualmodel"`. This also selects which parameters are used from `model_parameters` (see below) For more information see <!-- TODO link -->
* model_parameters: Parameters that describe the specific model. The keys correspond directly to the `model_class` setting. Multiple keys are possible to easily switch between different setups but only the one select using `model_class` is ever active and required. For a description of the available model classes see <!-- TODO -->

#### Training Settings
These settings, configure how the training is conducted.

```json
    "optimizer_name": "adam",
    "exploration": false,
    "cuda_id": null,
    "max_epochs": 50,
    "batch_size": 28,
    "learning_rate": 9.9027931032814e-05,
    "early_stopping_patience": 7,
    "early_stopping_margin": 0.0,
```
* optimizer_name: Name of the optimizer used for training. Available are `"adam"` (default), `"sgd"` (stochastic gradient decent), `"adagrad"`, `"adamax"`, `"rmsprop"`. For more information consult the [pytorch documentation](https://pytorch.org/docs/stable/optim.html#algorithms)
* exploration: Boolean value that determines whether hyper parameter exploration is performed. If this is `true` a an `exploration_path` needs to be provided in the [path settings](#path-settings)
* cuda_id: ID of a CUDA capabale GPU incase multiple are available and a specific one should be used. If this is null or ommited a GPU will be selected automatically.
* max_epochs: Maximum number of epochs during training. In each epoch the training dataset is completely run through once. It is generally recommend to use a rather high setting as there is an early stopping mechanism that stops on convergence.
* batch_size: Number samples per batch. Each batch is evaluated at once before doing an optimization step. 
* learning_rate: Factor for the stepsiez in optimization step. Very dependent on the specific use-case. Generally a higher learning rate leads to faster convergence but less accurate results.
* early_stopping_patience: Number of epoch without improvement in validation error after which training is stopped.
* early_stopping_margin: Minimum improvement in validation error to considered an improvement in terms of early stopping. This is an absolute change.


### Tuning Config
The tuning config manages the exploration hyper-parameter. To use exploration, activate it in the [main config](#model-settings) and set the [exploration path](#path-settings).
In this config specify the tumber of test runs that should be conducted using the attribute
```json
"number_of_tests": 5
```
In each test a model will be trained and evaluate against each other. 
The remainder of the config is under the key `"settings"`.
Its value is an object that represents the structure of the [main config](#main-config) only the values work differently and none of the parameters is required, the main config will be used for absent parameters.
```json
    "batch_size": {
            "function": "suggest_int",
            "kwargs": {
                "name": "batch_size",
                "low": 12,
                "high": 120
            }
        },
```
The value of the parameter specifies a function and map ("kwargs") that specifies the inputs to that function. Eligible functions are the methods of [optuna trials](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html). Obviously only settings can be used whose output corresponds to valid values in main config.

### Saliency Config
The saliency config dictates how the generation of saliency maps <!-- TODO ref --> is conducted.
```json
{
    "rel_interpretation_path": "oracles/interpretation",
    "date": "27.07.2019 00:00:00",
    "ref_batch_size": 40,
    "max_epochs": 3000,
    "n_trials": 5,
    "lr_low": 1e-3,
    "lr_high": 0.1,
    "relative_errors": true,
    "lambda": 1,
    "cuda_id" : null
}
```
* rel_interpretation_path: Path relative to the project folder where the results are supposed to be stored.
* date: date and time for which the saliency map is constructed.
* ref_batch_size: How many noisy samples are used to analyze the impact of said noise on that result. Higher numbers slow down the algorithm, lower ones make the result dependent on stochastic fluctuation.
* max_epoch: Number of optimization steps to find maximum noise with minium result deterioration.
* n_trials: How many noise optimizations should be conducted and compared.
* lr_low: Lower bound for the learning rate during noise optimization.
* lr_high: Upper bound for the learning rate during noise optimization.
* relative_errors: Boolean that switchen between relative and absolute weighting of errors. If true (recommended) the terms for "low amount of noise applied" and "quality deterioration" are normalized according to their values from the initial naive guess for a noise mask. For more detail see <!-- TODO ref to saliency description -->
* lambda: Weighting of the two errors against eachother. If `realatve_erros: true` this is a relative weighting Value >1 increases the value on applying as much noise as possible, <1 increases the focus on not negatively impacting the prediction quality. If `realatve_erros: false` this is absolute and the value might vary greatly (e. g. `lambda: 1e-4 might be reasonable) and is highly dependent on the use-case. 
* cuda_id: ID of a CUDA capabale GPU incase multiple are available and a specific one should be used. If this is null or ommited a GPU will be selected automatically.


### Specifying a config file

The default location of the main configuration file is `./targets/` or better `./targets/<STATION>`. 
The best practice is to generate sub-directories for each forecasting exercise, i.e. a new *station*. 
As the project originated from electrical load forecasting on substation-level, 
the term *station* or *target-station* is used to refer to the location or substation identifier
from which the measurement data is originating.   
Most of the example scripts in ProLoaF use the config file for training and evaluation, 
as it serves as a central place for parametrization.

At this stage you should have the main config file for your forecasting project: `./targets/<STATION>/config.json`.
[ProLoaF](https://github.com/sogno-platform/proloaf/tree/master/src/proloaf) comes with basic functions to parse, 
edit and store the config file. We make use of this when calling e.g. our example training script: 

```sh
$ python src/train.py -s opsd
```

The flag ``-s`` allows us to specify the station name (=target directory) through the string that follows, 
i.e. *opsd*. The 'train' script will expect and parse the config.json given in the target directory.
<!---
```sh
from proloaf.cli import read_config, parse_with_loss
MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)

ARGS, LOSS_OPTIONS = parse_with_loss()
    PAR = read_config(model_name=ARGS.station, config_path=ARGS.config, main_path=MAIN_PATH)
```
--->
You can also manually specify the path to the config file by adding ``-c <CONFIG_PATH>`` to the above mentioned
statement. The equivalent statement would be:
```sh
$ python src/train.py -c ./targets/opsd/config.json
```


<!-- TODO this should not be in the config description but in the train.py description > **_Note:_** If not otherwise specified, during training, per default the neural network maximizes the 
[Maximum Likelihood Estimation of Gaussian Parameters](http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html), 
for a 95% prediction interval. 
This so-called loss criterion can be changed to any metric that quantifies the (probabilistic) performance 
of the forecast. A common non-parametric option is the quantile loss.
You can apply quantile loss criterion as follows:

```sh
    $ python src\train.py -s opsd --quantiles 0.025 0.975
```
> Here we have specified the 95% prediction interval, by setting ``q1=0.025`` and ``q2=0.975``.

See more detailed descriptions and further loss options in the full [list of parameters](#parameter-list). -->




