---
title: "evaluate.py"
linkTitle: "evaluate.py"
date: 2020-03-09
weight: 3
description: >
  Basic information on the evaluation script.
---
### Overview
This script runs the trained net on test data and evaluates the result. It provides several graphs as outputs that show the performance of the trained model.
The output path can be changed under the the `"evaluation_path": ` option in the corresponding config file.

### Inputs
* ./data/<path_to_data_csv>.csv
* ./oracles/model_station

### Outputs
* ./oracles/<name_of_evaluation_folder>/

### Example Output Files
The output consists of the actual load prediction graph split up into multiple .png images and a metrics plot containing information about the model's performance.
![Predicted load at different hours]({{< resource url="img/eval_hour_plots.png" >}} "Predicted residual load at different hours")*Plots of the predicted load at different hours*

![Metrics plot with performance information]({{< resource url="img/metrics_plot.png" >}} "Metrics plot with performance information")*Metrics plot with results of evaluation*

### Summarized Description of Performance Metrics
If you need more details on which deterministic and probabilistic metrics we take to quantify if the produced forecast is considered "good" or "bad", please take a look at the [Tutorial Notebook]({{< resource url="https://github.com/sogno-platform/proloaf/blob/master/notebooks/A%20User%20Guide%20on%20the%20ProLoaF%20Forecast%20Performance%20Metrics.ipynb" >}}).

### Reference Documentation
If you need more details, please take a look at the [docs]({{< resource url="reference/proloaf/proloaf/src/evaluate.html" >}}) for 
this script.