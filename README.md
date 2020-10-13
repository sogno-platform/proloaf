# ProLoaF: A Probabilistic Load Forecasting System

[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org)

[**Website**](https://git.rwth-aachen.de/acs/public/automation/plf/plf-training)
| [**Docs**](https://acs.pages.rwth-aachen.de/public/automation/plf/plf-docs/)
| [**Install Guide**](https://acs.pages.rwth-aachen.de/public/automation/plf/plf-docs/docs/getting-started/)
| [**Tutorial**](https://acs.pages.rwth-aachen.de/public/automation/plf/plf-docs/docs/tutorials/)

ProLoaF is a probabilistic load forecasting system in the context of the H2020 EU CoordiNet project.


## Key Capabilities

ProLoaF comes with the following scripts

- [fc_prep.py](https://acs.pages.rwth-aachen.de/public/automation/plf/plf-docs/docs/files-and-scripts/fc_prep/): Preprocess your input data for use with ProLoaF.
- [fc_train.py](https://acs.pages.rwth-aachen.de/public/automation/plf/plf-docs/docs/files-and-scripts/fc_train/): Train an RNN model for load forecasting based on provided data.
- [fc_evaluate.py](https://acs.pages.rwth-aachen.de/public/automation/plf/plf-docs/docs/files-and-scripts/fc_evaluate/): Evaluate a previously trained model.


## Example Workflow

* Add the data a model should be trained on to the data/ folder
* Preprocess the data using [fc_prep.py](https://acs.pages.rwth-aachen.de/public/automation/plf/plf-docs/docs/files-and-scripts/fc_prep/) or custom scripts if needed.
* Specify all required and desired settings and values in [config.json](https://acs.pages.rwth-aachen.de/public/automation/plf/plf-docs/docs/files-and-scripts/config/) in the targets/ folder
* Train a model using `python3 ./source/fc_train.py -s <path_to_config_in_targets_folder>`
* Evaluate the model using fc_evaluate.py in the same fashion.

## Submodules

This Repository contains the following submodules:
* [plf_util](https://git.rwth-aachen.de/acs/public/automation/plf/plf-util): A Utilities-Package for ProLoaF.
* [plf_data](https://git.rwth-aachen.de/acs/public/automation/plf/plf-data): Example gefcom2017 data for testing purposes.

## Installation

First, clone this Repository and initialize the submodules:
```bash
git clone --recurse-submodule https://git.rwth-aachen.de/acs/public/automation/plf/plf-training.git
```
To install all required packages first run:
```bash
pip install -r requirements.txt
```

On low RAM machines the option
```bash
pip install -r requirements.txt --no-cache-dir
```
might be necessary.

ProLoaF supports Python 3.6 and higher. For further information see [Getting Started](https://acs.pages.rwth-aachen.de/public/automation/plf/plf-docs/docs/getting-started/)

## Related Publications
G. Gürses-Tran, H. Flamme, and A. Monti, "Probabilistic Load Forecasting for Day-Ahead Congestion Mitigation," The 16th International Conference on Probabilistic Methods Applied to Power Systems, PMAPS 2020, 2020-08-18 - 2020-08-21, Liège, Belgium, ISBN 978-1-7281-2822-1, [Access Online](http://aimontefiore.org/PMAPS2020/openconf/modules/request.php?module=oc_proceedings&action=view.php&id=92&type=3&a=Accept)

## License
