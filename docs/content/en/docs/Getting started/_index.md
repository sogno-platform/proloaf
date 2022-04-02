---
title: "Getting Started"
linkTitle: "Getting Started"
weight: 2
description: >
  An introduction to ProLoaF.
---

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

To run this project you need to have python 3.8 or higher installed on your machine.

First, clone this Repository and initialize the submodules:
```bash
git clone --recurse-submodule https://git.rwth-aachen.de/acs/public/automation/plf/proloaf.git
```
or if you have already cloned the project and you are missing e.g. the open data directory for execution run:
```bash
git submodule update --init --recursive
```

Now you will need to install all packages listed in the *requirements.txt* file.
To install all required packages using pip, run:
```bash
pip install -r requirements.txt
```

On low RAM machines the option
```bash
pip install -r requirements.txt --no-cache-dir
```
might be necessary. Depending on your machine you might need to use pip3 instead of pip.

## Running the code
This project contains 3 scripts that can be used in conjunction with one another or separately. Configuration for these scripts is given in a config.json file in the targets/ folder.

* All scripts are located in the source folder.
* To start one of the scripts use 'Python3 script.py argument', where the argument is either the name of a station (-s) (e.g. 'opsd') or the path (-c) of the corresponding config file located in the model_ folders.
* To prepare load and weather data from selected stations run ./src/preprocess.py
* To train a recurrent neural network model specifically parametrized for the selected station and prepared data run ./src/train.py
* To analyze the performance of the forecast: ./src/evaluate.py

## Config
* The scripts use a config.json file located in the targets/ folder. This file is used to give further information and settings needed to train a model.
* In addition, a tuning.json file is used to define settings regarding hyperparameter tuning. This file is only required when using hyperparameter tuning.


* To adapt the behavior of the scripts for a certain station change options in the corresponding config file.
* Parameters can be added to the config file by adding or removing them by hand (standard json format) or using the config_maker.py.

Add a line
```
par['parameter_name'] = value
```
in the configmaker.py file and it will be added under that name.
Then use
```
python3 configmaker.py --mod path_to_config
```
to apply the changes. The argument again is a station name or the already existing config file.

Using
```
python3 configmaker.py --new path_to_config
```
Will clear the config file before applying changes so be careful with that.
