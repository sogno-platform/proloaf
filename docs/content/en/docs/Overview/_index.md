---
title: "Overview"
linkTitle: "Overview"
weight: 1
description: >
  ProLoaF is a probabilistic load forecasting project.
---
ProLoaF is package that allows training and evaluation of deep-neural-networks for forecasting of timeseries data. Note that ProLoaF does not provided pretrained model that can used out of the box. Also please be aware that while ProLoaF is fully usable, code structure and API's are subject to change. The same is true for the model structure, trained models might not be compatible between different versions of ProLoaF.

## Key Capabilities

ProLoaF comes with the following scripts as entry points

- [preprocess.py]({{< resource url="docs/files-and-scripts/prepprocess/" >}}): Preprocess your input data for use with ProLoaF. This is somewhat specific and it might be easier to write a custom script that specifcally fits your original data fromat.

- [train.py]({{< resource url="docs/files-and-scripts//train/" >}}): Train an RNN model for load forecasting based on provided data.

- [evaluate.py]({{< resource url="docs/files-and-scripts/evaluate/" >}}): Evaluate a previously trained model.

- [evaluate_saliency.py]({{< resource url="docs/files-and-scripts/evaluate_saliency/" >}}): Evaluate a previously trained model on how much impact each of the features has on the forecast.

- [baselines.py]({{< resource url="docs/files-and-scripts/baselines/" >}}): Train and compare performance with typical statsmodels (Currently not working due to version conflict).

![Dependency diagrams of basic entry points to ProLoaF]({{< resource url="docs/static/img/modules_all_reduced_forweb.svg" >}} "Dependency diagrams of basic entry points to ProLoaF")*Dependency Diagrams*

## Dependencies
Proloaf is a Python package based on [PyTorch](https://pytorch.org/) and uses [Pandas](https://pandas.pydata.org/), [SciKit-learn](https://scikit-learn.org/stable/) and [TensorBoard](https://www.tensorflow.org/tensorboard) to supplement the workflow.
The current version requires Python version >=3.9.
A complete list of dependencies including version restrictions can be found in the `pyproject.toml` of the project.

## Project structure
The codebase splits into two parts, the library of ProLoaF is found in `src/proloaf` and contains all the functions and classes to prepare train and evalute a forecasting models. 
The remaining scripts in the `src/` folder use what is provided in the library to perform the work.
Data should be stored in a subfolder of the `data/` folder, outputs can be found in the `oracles/` folder. Most of these pathes can be adjusted in the [config(s)](../Files%20and%20Scripts/config.md).
For a normal user, using these scripts after adapting the these should be sufficient to train and evaluate a model.


## Test Data

This repository contains the following submodules:
* [proloaf-data](https://git.rwth-aachen.de/acs/public/automation/plf/plf-data): Example gefcom2017 and open power system data for testing purposes.

## Where should I go next?

* [Getting Started]({{< resource url="docs/getting-started/" >}}): Get started with your own project
* [Source Code Documentation]({{< resource url="reference/proloaf/proloaf.html" >}}): Dive in here if you're looking for a detailed reference guide.
