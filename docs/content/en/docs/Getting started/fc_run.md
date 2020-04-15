---
title: "fc_run.py"
linkTitle: "fc_run"
date: 2020-03-09
description: >
  Basic information on the script for running the trained neural net.
---

This script loads a trained net and input data and generates a prediction based on the data
### config
The config looks almost identical to *fc_station_train_config.json* with some parameters removed or added. These are still separate to allow independant use of the scripts. The two most important additions are:
1. "model_path": (str) path from which the trained net can be loaded, w.r.t. the project folder
2. "forecast_path" (str) path where the forecast is saved, w.r.t. the project folder.

### inputs
* ./output/model_station
* ./output/load_forecast_station.csv

### outputs
* ./output/load_forecast_station.csv

