---
title: "Getting Started"
linkTitle: "Getting Started"
weight: 2
description: >
  An introduction to the load forecasting subproject of the CoordiNet project.
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
This project contains 4 scripts that can be used in conjunction with one another or separatly. For each one there is a corresponding config file in the JSON format.

* All scripts are located in the scripts folder.
* To start one of the scripts use 'Python3 fc_script.py argument', where the argument is either the name of a station (e.g. 'sege', 'kraftringen') or the path of the correspondig config file located in the model_ folders.
* To prepare load and weather data from selected stations run ./scripts/fc_prep.py
* To train a recurrent neural network model specifically parametrized for the selected station and prepared data run ./scripts/fc_train.py
* To predict the next 40 hours in reference to the current time and print the result in a csv file run ./scripts/fc_run.py
* *27.10.2019* To analyze the performance of the forecast: ./model_station/evaluation.py

For further information use the links in the navigation bar on the left.

## Configs
* Each script has its own config file and there are two additional ones concerning hyperparameter exploration.
* Each config only defines variables needed in the corresponding script changes in a different context. Some settings (like model_path) can occur in more than one config, this is necessary to keep the scripts independent. Make sure to apply changes to all relevant configs.

* To adapt the behavior of the scipts for a certain station change options in the config files located in the model_ folders.
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


