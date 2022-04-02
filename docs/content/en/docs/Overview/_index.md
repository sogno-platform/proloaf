---
title: "Overview"
linkTitle: "Overview"
weight: 1
description: >
  ProLoaF is a probabilistic load forecasting project.
---

## Key Capabilities

ProLoaF comes with the following scripts as entry points

- [preprocess.py]({{< resource url="docs/files-and-scripts/fc_prep/" >}}): Preprocess your input data for use with ProLoaF.

- [train.py]({{< resource url="docs/files-and-scripts//fc_train/" >}}): Train an RNN model for load forecasting based on provided data.

- [evaluate.py]({{< resource url="docs/files-and-scripts/fc_evaluate/" >}}): Evaluate a previously trained model.

- [baselines.py]({{< resource url="docs/files-and-scripts/baselines/" >}}): Train and compare performance with typical statsmodels.

![Dependency diagrams of basic entry points to ProLoaF]({{< resource url="img/modules_all_reduced_forweb.svg" >}} "Dependency diagrams of basic entry points to ProLoaF")*Dependency Diagrams*

## Example Workflow

* Add the data a model should be trained on to the data/ folder

* Preprocess the data using [prep.py]({{< resource url="docs/files-and-scripts/fc_prep/" >}}) or custom scripts if needed.

* Specify all required and desired settings and values in [config.json]({{< resource url="docs/files-and-scripts/config/" >}}) in the targets/ folder

* Train a model using `python3 ./src/train.py -s <path_to_config_in_targets_folder>`

* Evaluate the model using evaluate.py in the same fashion.

## Test Data

This repository contains the following submodules:
* [proloaf-data](https://git.rwth-aachen.de/acs/public/automation/plf/plf-data): Example gefcom2017 and open power system data for testing purposes.

## Where should I go next?

* [Getting Started]({{< resource url="docs/getting-started/" >}}): Get started with your own project
* [Source Code Documentation]({{< resource url="reference/proloaf/proloaf.html" >}}): Dive in here if you're looking for a detailed reference guide.
