---
title: "Getting Started"
linkTitle: "Getting Started"
weight: 2
description: >
  An introduction to ProLoaF.
---

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

To run this project you need to have python 3.6 or higher installed on your machine.
You will need to install all packages listed in the *requirements.txt* file.
To install all required packages using pip run:
```
pip install -r requirements.txt
```

On low RAM machines the option
```
pip install -r requirements.txt --no-cache-dir
```
might be necessary. Depending on your machine it might be necessary to use pip3 instead of pip.


## Running the code
This project contains 3 scripts that can be used in conjunction with one another or separatly. Configuration for these scripts is given in a config.json file in the targets/ folder.

* All scripts are located in the source folder.
* To start one of the scripts use 'Python3 fc_script.py argument', where the argument is either the name of a station (-s) (e.g. 'gefcom2017/ct_data') or the path (-c) of the correspondig config file located in the model_ folders.
* To prepare load and weather data from selected stations run ./source/fc_prep.py
* To train a recurrent neural network model specifically parametrized for the selected station and prepared data run ./source/fc_train.py
* To analyze the performance of the forecast: ./source/fc_evaluate.py

## Config
* The scripts use a config.json file located in the targets/ folder. This file is used to give further information and settings needed to train a model.
* In addition, a tuning.json file is used to define settings regarding hyperparameter tuning. This file is only required when using hyperparameter tuning.


* To adapt the behavior of the scipts for a certain station change options in the correspondig config file.
* Parameters can be added to the config file by adding or removing them by hand (standard json format) or using the config_maker.py.

Add a line
```
par['parameter_name'] = value
```
in the config_maker.py file and it will be added under that name.
Then use
```
python3 config_maker.py --mod path_to_config
```
to apply the changes. The argument again is a station name or the already existing config file.

Using
```
python3 config_maker.py --new path_to_config
```
Will clear the config file before applying changes so be careful with that.
