---
title: "metrics"
linkTitle: "metrics"
date: 2020-19-08
weight: 4
description: >
  Introduction to forecasting metrics.
---

### Metrics
When training a model its exact behaviour depends on the metric. For example a model using mean-squared-error is more sensitive to outliers than mean-absolute-error, which is equivalent with predicting the median. In addition these metrics can have implicit assumptions, e. g. the negative-log-likelihood-Gaussian (nllg) loss assumes that the value at each timestep is a normal distributed random variable. These losses are also parameterized in different ways. ProLoaF predicts these additional parameters directly (e. g. the log variance for nllg).
Using ProLoaF will make sure that the model is constructed with the correct number of output parameters every time. 

#### Usage
Every metric in ProLoaF inherits from an abstract base class. All the metric objects are callable with predicitons and targets and will return a single value by default. They all have an argument named `avg_over`. Possible inputs are `"sample"`, `"time"`, `"feature"`, `"all"` Or a combination of them in a tuple. For training `"all"` is used these can be used to analyse the performance of the model in more detail. Due to their definition not all metrics support averageing over each dimension separately. 
Since the model output depends on the metric used for training different metrics can not be applied directly. To circumvent this every loss metric has a method `get_quantile_prediction` that changes to prediction into multiple predicitons, one for each quantile specified and each feature. The metrics also have a `from_quantiles` method that allows to calculate the metric from quantile predictions. Since the quantiles do not necessary support the assumptions in loss (e. g. being normally distributed) these calculations may be approximations.

For a general overview of the metric there is a Jupyter notebook (TODO link) going more into detail.
For a full list of all available metrics please refer the to reference (TODO link).