---
title: "fc_evaluate.py"
linkTitle: "fc_evaluate.py"
date: 2020-03-09
weight: 3
description: >
  Basic information on the evaluation script.
---

This script runs the trained net on test data and evaluates the result. It provides several graphs as outputs that show the performance of the trained model.
The output path can be changed under the the `"evaluation_path": ` option in the corresponding config file.

### inputs
* ./oracles/model_station

### outputs
* ./oracles/<name_of_evaluation_folder>/

### Example output files
The ouput consists of the actual load prediction graph split up in multiple svg images and a metrics plot containing information about the model's performance.
