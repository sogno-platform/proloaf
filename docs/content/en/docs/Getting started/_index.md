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
(Optional) Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate
```

To install all required packages first change into the directory then install the package:
```bash
cd proloaf
pip install .
```
There are two variants with extra dependencies. To install jupyter for being able to use the tutorial notebooks use 
```bash
pip install .[tutorial]
```
or use
```bash
cd proloaf
pip install -e .[dev]
```
to install in editable mode and include optional development dependencies.

On low RAM machines the option
```bash
pip install . --no-cache-dir
```
might be necessary.

ProLoaF supports Python 3.8 and higher.

## Running the code
This project contains 4 scripts that can be used in conjunction with one another or separately. Configuration for these scripts is given in a [config.json]({{< resource url="docs/files-and-scripts/config/" >}}) file in the subfolders of the `/targets/` folder, the config in `targets/opsd` should be a good starting point.

## Example Workflow

1. Add the data a model should be trained on to the `/data/` folder

2. Preprocess the data using [prep.py]({{< resource url="docs/files-and-scripts/fc_prep/" >}}) or custom scripts if needed. Data including targets should be all in one `.csv` file separated with `;`.

3. Create a new folder in the `targets/` folder corresponding to your forecast. Copy the config files from `targets/opsd` and adjust it to your needs. In the very least everything to do with your data needs to be adjusted (like column names etc). For more information have a look in the descripiton of the [config]({{< resource url="docs/files-and-scripts/config/" >}}).

4. Train a model using `python3 ./src/train.py -s <name_of_folder_containing_configs>`. You can also give a path to the config relative to the project folder using `-c` instead of `-s`

5. Evaluate the model using ``python3 ./src/evaluate.py.py -s <name_of_folder_containing_configs>` in the same fashion.

6. To understand how to generate a single prediction have a look at the examples.


