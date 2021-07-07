---
title: "Overview"
linkTitle: "Overview"
weight: 1
description: >
  ProLoaF is a probabilistic load forecasting project.
---

## Key Capabilities

ProLoaF comes with the following scripts

- [fc_prep.py]({{< resource url="docs/files-and-scripts/fc_prep/" >}}): Preprocess your input data for use with ProLoaF.

- [fc_train.py]({{< resource url="docs/files-and-scripts//fc_train/" >}}): Train an RNN model for load forecasting based on provided data.

- [fc_evaluate.py]({{< resource url="docs/files-and-scripts/fc_evaluate/" >}}): Evaluate a previously trained model.

## Example Workflow

* Add the data a model should be trained on to the data/ folder

* Preprocess the data using [fc_prep.py]({{< resource url="docs/files-and-scripts/fc_prep/" >}}) or custom scripts if needed.

* Specify all required and desired settings and values in [config.json]({{< resource url="docs/files-and-scripts/config/" >}}) in the targets/ folder

* Train a model using `python3 ./source/fc_train.py -s <path_to_config_in_targets_folder>`

* Evaluate the model using fc_evaluate.py in the same fashion.

## Tensorboard utilization

This project uses Tensorboard to display in-depth information about each train run. Information about the run itself like training time and validation
loss are visible in the `Scalars` tab. The `HParams` tab allows sorting of runs by hyper parameters as well as parallel and scatter plots. To define which
data should be logged, a `log.json` of following structure is used in the targets/ folder of each station:

```
{
  "features": [
    {
      "name": "time_stamp"
    },
    {
      "name": "train_loss"
    },
    {
      "name": "val_loss"
    },
    {
      "name": "total_time"
    }
  ]
}
```

Install Tensorboard and run the Tensorboard command-line interface as described [here](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html). The default directory that runs get saved in is the `runs/` directory.
To display runs in Tensorboard, use `tensorboard --logdir=runs`, then open a browser and navigate to [127.0.0.1:6006](127.0.0.1:6006)




## Submodules

This Repository contains the following submodules:
* [plf_data](https://git.rwth-aachen.de/acs/public/automation/plf/plf-data): Example gefcom2017 data for testing purposes.

## Where should I go next?

* [Getting Started]({{< resource url="docs/getting-started/" >}}): Get started with your own project
* [Source Code Documentation]({{< resource url="reference/proloaf/proloaf.html" >}}): Dive in here if you're looking for a detailed reference guide.
