---
title: "train.py"
linkTitle: "train.py"
date: 2020-03-09
weight: 2
description: >
  Basic information on the training script.
---
### Overview
This script trains a forecasting model on a dataset, which is loaded into a pandas Dataframe from a csv file. 
Hyperparameter exploration using [optuna](https://optuna.org/) is also possible if desired. The trained model is saved at the location specified under `"output_path"` in the corresponding [config.json](config.md#main-config) and can be loaded via `torch.load()` or evaluated by using the [evaluate.py](./evaluate.md) script. This script performs desired preprocessing, like adding one-hot encoded time features and scaling, loads a custom data structure and then generates and trains a neural net.

### Hyperparameter Exploration
Any training parameter is considered a hyper parameter as long as it is specified in either [config.json](config.md#main-config). The latter is the standard file where the (so far) best found configuration is saved and should usually not be manually adapted unless new tests are for some reason not comparable to older ones (e.g. after changing the loss function).

### Usage
To run training you need to create at least a the main [config.json](config.md#main-config) first. 
Then you can run
```sh
$ python src/train -s <station_name>
```
where station name is a folder in the folder `src/targets` that contains a config names `conifg.json`.
Alternatively you can run training using
```sh
$ python src/train -c <path_to_config>
```
to specify a path relative to the project folder directly to the config file. 


If not otherwise specified, during training, per default the neural network maximizes the 
[Maximum Likelihood Estimation of Gaussian Parameters](http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html), 
for a 95% prediction interval. This so-called loss criterion can be changed to any metric that quantifies the (probabilistic) performance of the forecast. A common non-parametric option is the quantile loss. You can apply quantile loss criterion as follows:

```sh
$ python src\train.py -s opsd --quantiles 0.025 0.975
```
> Here we have specified the 95% prediction interval, by setting ``q1=0.025`` and ``q2=0.975``.

See more detailed descriptions and further loss options in the full [list of available training metrics](#metrics).

### Structure
To be able to adjust the training script if necessary it might be helpful to understand the overall structure of the script. If you feel like the script is missing general features, feel free to [pose a feature request](../Contribution%20Guidelines/_index.md)
>[!Note]
>There might be slight differences from the code presented here and the up-to-date `train.py`.
>The file itself should be the correct version, what is presented here should however always be recognizable. 
**imports**
```python
import os
import warnings
from copy import deepcopy
from functools import partial

import pandas as pd
import torch

import proloaf.confighandler as ch
import proloaf.datahandler as dh
import proloaf.modelhandler as mh
import proloaf.tensorloader as tl
from proloaf.cli import parse_with_loss

from proloaf.confighandler import read_config
from proloaf.event_logging import create_event_logger
```
Nothing out of the ordinary, note however the direct dependency on pandas and torch.
**globals** 
```python
MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

torch.set_printoptions(linewidth=120)  # Display option for output
torch.set_grad_enabled(True)

warnings.filterwarnings("ignore")
logger = create_event_logger("train")
# torch.manual_seed(1)
```
`MAIN_PATH` is the path all other path settings are relative to.
Enable uncomment `torch.manual_seed(1)` to have deterministic training.

#### Main
This function trains a model and comparse it to an existing model with the same name, if it exists. It then discards the model performing worse. It does everything but loading the config and parsing cli arguments.

**Load & Prepare Data**
```python
logger.info("Current working directory is {:s}".format(work_dir))

    # Read load data
    config = deepcopy(config)
    # log_df = log.init_logging(model_name=station_name, work_dir=work_dir, config=config)
    try:
        df = pd.read_csv(infile, sep=";", index_col=0)

        train_df, val_df = dh.split(df, [config.get("train_split", 0.7)])

        scaler = dh.MultiScaler(config["feature_groups"])
        train_dataset = tl.TimeSeriesData(
            train_df,
            device=device,
            preparation_steps=[
                partial(dh.set_to_hours, freq=PAR.get("frequency", "1h")),
                partial(dh.fill_if_missing, periodicity=config.get("periodicity", 24)),
                dh.add_cyclical_features,
                dh.add_onehot_features,
                scaler.fit_transform,
                dh.check_continuity,
            ],
            **config,
        )
        val_dataset = tl.TimeSeriesData(
            val_df,
            device=device,
            preparation_steps=[
                partial(dh.set_to_hours, freq=PAR.get("frequency", "1H")),
                # TODO check if periodicity is correct
                partial(dh.fill_if_missing, periodicity=config.get("periodicity", 24)),
                dh.add_cyclical_features,
                dh.add_onehot_features,
                scaler.transform,
                dh.check_continuity,
            ],
            **config,
        )
```
Data is first loaded into a Pandas dataframe, then split and loaded into the `TimeSeriesData` objects.
These data structures fulfill the purpose of exposing the data as the original pandas dataframe (via the `.data` attribute) on one side and createing data samples for training on the other side.
The `preparation_steps` are callables that take a dataframe and return a (modified) dataframe. These steps are chained to prepare the data, like interpolating gaps.
Note that the scaler is is fit on the training data and then only applied on the validation data.
The `partial` function can be used to provide additional arguments. All steps in the preparation are applied lazily to allow dynamic adjustments.

**Model Creation & Training**
```python
if config.get("exploration_path") is None:
    tuning_config = None
else:
    tuning_config = read_config(
        config_path=config["exploration_path"],
        main_path=work_dir,
    )

modelhandler = mh.ModelHandler(
    work_dir=work_dir,
    config=config,
    tuning_config=tuning_config,
    scalers=scaler,
    loss=loss,
    loss_kwargs=loss_kwargs,
    device=device,
)

modelhandler.fit(
    train_dataset,
    val_dataset,
)
train_dataset = None # unbind data to save some memory
```
First load the tuning config, then create the modelhandler. The modelhandler provides an api with a `fit` and a `predict` method as it is common in [scikit-learns predictors](https://scikit-learn.org/1.5/developers/develop.html) though with slightly different arguments. 

**Model compariosn**
```python
try:
    ref_model_1 = modelhandler.load_model(
        os.path.join(
            work_dir,
            config.get("output_path", ""),
            f"{config['model_name']}.pkl",
        )
    )
except FileNotFoundError:
    ref_model_1 = None
    logger.info("No old version of the trained model was found for the new one to compare to")
except Exception:
    ref_model_1 = None
    logger.warning(
        "An older version of this model was found but could not be loaded, this is likely due to diverignig ProLoaF versions."
    )
try:
    if ref_model_1 is not None:
        modelhandler.select_model(
            val_dataset,
            [ref_model_1, modelhandler.model_wrap],
            modelhandler._model_wrap.loss_metric,
        )
except:
    logger.warning(
        "An older version of this model was found but could not be compared to the new one, this might be due to diverging model structures."
    )
```
If there is already a model that would be overwritten by the new one, it will be loaded and compared to the new one. This is done on the basis of the validation dataset and the training loss used for the current training.
The worse performing model, is then discarded. This allows to compared the trials from hyper-parameter tuning with the original configuration.

**Saving**
```python
modelhandler.save_current_model(
    os.path.join(work_dir, config.get("output_path", ""), f"{config['model_name']}.pkl")
)
config.update(modelhandler.get_config())
ch.write_config(
    config,
    model_name=ARGS.station,
    config_path=ARGS.config,
    main_path=work_dir,
)
```
Both the model and current config are saved. The config is saved in case hyper-parameter tuning was performed. 


### Reference Documentation
If you need more details, please take a look at the [reference]({{< resource url="reference/proloaf/proloaf/src/train.html" >}}) for this script.
