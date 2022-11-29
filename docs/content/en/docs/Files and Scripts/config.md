---
title: "config.json"
linkTitle: "config.json"
date: 2020-19-08
weight: 4
description: >
  Basic information about the config.json file
---

### Configuration

#### Main Config
ProLoaF's configuration file is written using JSON. As such, whitespace is allowed and ignored in your syntax.
We mostly use strings, boolean, and ``null``, to specify the paths, and parameters of our targeted forecasting project. 
Numbers and booleans should be  unquoted.

For better readability, we assume in the examples below, 
that your working directory is set to the main project path of the cloned 
[repository](https://github.com/sogno-platform/proloaf).

#### Generating a Configuration File

A *new* config file can be generated automatically by using:
```sh
python src\configmaker.py --new targets/<STATION-NAME>/config.json
```
or manually.

To *modify* the file, you can change parameters with our helper or apply modifications manually. 
When using the helper, your modifications must be set in `configmaker.py`. You can then run:

```sh
python src\configmaker.py --mod targets/<STATION-NAME>/config.json
```

#### Configuration Loading

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
$ python src\train.py -s opsd
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
statement.

> **_Note:_** If not otherwise specified, during training, per default the neural network maximizes the 
[Maximum Likelihood Estimation of Gaussian Parameters](http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html), 
for a 95% prediction interval. 
This so-called loss criterion can be changed to any metric that quantifies the (probabilistic) performance 
of the forecast. A common non-parametric option is the quantile loss.
You can apply quantile loss criterion as follows:

```sh
    $ python src\train.py -s opsd --quantiles 0.025 0.975
```
> Here we have specified the 95% prediction interval, by setting ``q1=0.025`` and ``q2=0.975``.

See more detailed descriptions and further loss options in the full [list of parameters](#parameter-list).

#### Path Settings

Through the config file the user specifies the data source location, and the directories for logging, 
exporting performance analyses and most importantly, the trained RNN model binary.

**Path Specs:**
```json
{
    "data_path": "./data/<FILE-NAME>.csv",
    "evaluation_path": "./oracles/eval_<MODEL-NAME>/",
    "output_path": "./oracles/",
    "exploration_path": "./targets/sege/tuning.json", 
    "log_path": "./logs/"
}
```
The output-, exploration- and log- paths may stay unchanged, but the data path and evaluation path must be specified.

> **_Note:_** The data path should contain a csv file that includes all input data column-wise in any time-resolution.
In our example train-& evaluation scripts, the first column is treated as datetime information and declared as 
pandas datetime index. *oracles* is the default naming of the output directory, 
>in which the prediction model and predictive performance are stored.  

#### Timeseries Settings

ProLoaF is a machine-learning based timeseries forecasting project. The supervised learning requires data with 
a (pandas) datetime index. Typical time resolutions are:`ms`, `s`, `m`, `h`, `d`.
Endogenous (lagged) inputs and exogenous (=explanatory) variables that affect the future explained variable, 
are split into multiple windows with the sequence length of ``history_horizon`` and fed to the encoder.
For better understanding, we recommend 
[the illustrative guide](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb) 
on a similar sequence-to-sequence architecture authored by Ben Trevett.

The time step size is equal to the underlying timeseries data resolution. 
The example files apply day-ahead forecasting in hourly resolution. 
> **_Note:_** A forecast that produces a predicted sequence starting from the forecasting execution time and including the next day, 
is >=24h, depending on the forecast execution time t, e.g. a day-ahead forecast executed at 9 am, shall produce a 40 hour horizon.

Following parameters configure the input- and output- sequence length:
```json
{
  "history_horizon": 42,
  "forecast_horizon": 24
}
```
#### Data Partitioning

In machine learning, we typically split available data to train the model and test its performance. 
With the training set, the model is parameterized. By checking against validation data, we track the fitting process.
In a final step, the test set serves to assess and confirm the predictive power of the trained model. 
To configure the size of each mentioned set, specify:

- **train_split**:
Given an input dataframe df, all timestamps before the specified split are used for training: 
It is also possible to specify the split by time in the iso 8601 (`YYYY-MM-DD HH:MM:SS`) format.
``df[:train_split*df.shape[0]]``.

- **validation_split**:
For validation during training, we use all data between train and validation limiters in the dataframe df:
``df[train_split*df.shape[0]:validation_split*df.shape[0]]``.

> **_Note:_** The *test_split* is set per default, through the remaining input data from df: 
>``df[validation_split*df.shape[0]:]``. 
  
#### Data Pre-processing through Scaling
  
#### Feature Selection

```json
    "feature_groups": [
        {
            "name": "main",
            "scaler": [
                "robust",
                15,
                85
            ],
            "features": [
                "<COLUMN-IDENTIFIER-1>"
            ]
        }
```

```json
        {
            "name": "add",
            "scaler": [
                "minmax",
                -1.0,
                1.0
            ],
            "features": [
                "<COLUMN-IDENTIFIER-2>"
            ]
        }
```

```json
        {
            "name": "aux",
            "scaler": null,
            "features": [
                "<COLUMN-IDENTIFIER-3>",
                "<COLUMN-IDENTIFIER-4>"
            ]
        }
    ]
```
``` "encoder_features": [
        "<COLUMN-IDENTIFIER-1>",
        "<COLUMN-IDENTIFIER-2>"
    ],
    "decoder_features": [
        "<COLUMN-IDENTIFIER-3>",
        "<COLUMN-IDENTIFIER-4>"
    ],
    "aux_features": [
        "<COLUMN-IDENTIFIER-3>",
        "<COLUMN-IDENTIFIER-4>"
    ],

### RNN Cell Type
- GRU: trains typically faster (per epoch) with similar results compared to LSTM cells.
- LSTM
```json
{
    "history_horizon": 42,
    "forecast_horizon": 24
}
```

#### Hyperparameters and Tuning

```json
{
  "max_epochs": 1,
  "batch_size": 2,
  "learning_rate": 0.0001,
  "core_layers": 1,
  "rel_linear_hidden_size": 1.0,
  "rel_core_hidden_size": 1.0,
  "dropout_fc": 0.4,
  "dropout_core": 0.3
}
```

- **max_epochs**:....

- **batch_size**: ...

- **learning_rate**: ...

- **core_layers**: ...

- **rel_linear_hidden_size**: ...

- **rel_core_hidden_size**: ...

- **dropout_fc**: ...

- **dropout_core**: ...
  
Configure which hyperparameters are optimized and specify each parameters search space through a separate 
[tuning config](#tuning-config).
> **_Note:_** Best practice is to save the tuning.config in the same directory in which the main config is given. 
>However, by setting a [specific exploration_path](#path-settings), the user can direct to a 
>different location on the machine.

#### GPU Specs
Some text on cuda id

#### Selecting the best model
- **best_loss**: 
- **best_score**: 

#### Parameter List

The following table summarizes the default parameters of the main config file:

**Config Params**

| Parameter   |      Data Type      |  Value Range |
|:-------:|:-------------:|:-------:|
| history_horizon | int |  > 0 |
| forecast_horizon | int |  > 0 |

**Shell Params Upon Script Execution**

Example:
```sh
    $ python src\train.py -s opsd --rmse
```
or
```sh
    $ python src\train.py -s opsd --smoothed_quantiles 0.025 0.975
```

| Parameter | Data Type  |  Value Range | Short Description |
|:---------:|:-------------:|:-------:|:-------------------:|
| --< loss > |  string in shell | {mse, mape, rmse, mis, nll_gauss, quantiles, smoothed_quantiles, crps} | Set the loss function for training |
| --ci |  boolean | True or False | Enables execution mode optimized for GitLab's CI |
| --logname |  str | " " | Name of the run, displayed in Tensorboard |

### Tuning Config

> Tensorboard utilization

This project uses Tensorboard to display in-depth information about each train run. Information about the run itself like training time and validation
loss are visible in the `Scalars` tab. The `HParams` tab allows sorting of runs by hyper parameters as well as parallel and scatter plots. To define which
data should be logged, a `log.json` of following structure is used in the targets/ folder of each station:

```
{
  "features": [
    {
      "name": "time_stamp"
    },
    {
      "name": "train_loss"
    },
    {
      "name": "val_loss"
    },
    {
      "name": "total_time"
    }
  ]
}
```

Install Tensorboard and run the Tensorboard command-line interface as described [here](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html). The default directory that runs get saved in is the `runs/` directory.
To display runs in Tensorboard, use `tensorboard --logdir=runs`, then open a browser and navigate to [127.0.0.1:6006](127.0.0.1:6006)

### Preprocessing Config


