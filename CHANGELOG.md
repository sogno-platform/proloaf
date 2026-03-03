# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- ## [Unreleased]
### Added
### Changed
### Fixed -->

## [0.4.0] - 2026-03-03
### Added
- Preparation step for adding onehot-encoding for holidays
    - 3 categories: weekday, saturday, sunday. Holidays are sundays
    - `country_code` is an attribute to add to the config, specifying the holiday lookup. [More details](https://holidays.readthedocs.io/en/latest/#available-countries)
- It is now possible to stack data columns treating them as a single feature during training.
- The `Timeseries` class now has a method, `get_tensor_input(...)` that can be used when the `Timeseries` is used at inference time to prepare input data. 
- Precommit hook for formating

### Changed
- Use `timestep` instead of `hour` encoding
    - 1 `timestep` feature per timesteps i. e. 15 min time delta leads to 96 timestep columns
- Cyclical time of day now continues instead of "full-hour" based.
- Training multiple models at the same time is now possible (checkpoints do not collide anymore)
- Interrupting training saves the best model trained so far under a unique name.
- Change location of docker files etc.

### Fixed
- Multiple bugs related to calculating the length of the dataloader, resulting in incomplete batches as the iteration continued further than it should have.
- Spelling on Pinball loss (was "Pinnball")

## [0.3.0] - 2024-10-23
### Fixed
- Corner case in data interpolation
- Fix faulty indexing of dataloader when measuremnt frequency increases in the dataset

### Changed
- Unified internal data structure
- Plotting for multiple features
- Documentation updates

### Added
- Additional preparation steps for extending the dataframe when no decoder input is in the original data and adding 0 columns for non-present onehot features
- Ability to do multi feature forecasts
- Saliency map evaluation (explainability method)

### Known Issues
The comparison with baseline methods like Arima is currently not working due to versioning conflicts. It will be readded if `pmdarima` is compatible with `pandas>=2.0` or a different solution is found.

## [0.2.2] - 2024-07-05
### Fixed
- Update numpy typing
- Typo in cli flag
- Fix plotting style
- Pin versions of dependencies

### Changed
- Datatloader returns aux features separately
- Dataloader returns final datapoint separately
- replace `setup.py` and `requirements.txt` with `pyproject.toml`
- Internal tensores are now 4D, [batch, time, feature, subfeature (e.g. mean, variance)]
- Adjust all performance metrics to operate on the last dimension
- Evaluate creates one plot per feature

### Added
- Possibility to forecast multiple features at a time (quantitative testing required)
- Allow spliting datasets by datetime
- Hybrid Autoencoder model

## [0.2.1] - 2022-01-14

### Fixed

-   Ignore reference model in training if it can not be loaded (e.g. because of diverging proloaf versions)
-   No longer safes the data with the model after training (model go from 1gb -> 30mb)

### Changed

-   Config file splits model configuration into generic (like older version) and modelspecific, which is a dict with model name and the acording parameters.
-   Added model selection to config
-   Some additional parameters in datahandler functions
-   Usage of dataobject changed everywhere to accompany the new Dataset

### Added

-   Complete new implementation of dataloader
    -   preprocessing can now be done more flexible
    -   storage should be more efficient than the old dataloader
    -   more available hyperparameters
-   MultiScaler added that encompases the functionallities of scale_all, rescale_manually and some more.
-   First version of a transformer model

## [0.2.0] - in prep

### Fixed

-   Fixed ...

### Changed

-   metrics as class...

### Added

-   Created `Changelog.md`
-   Modelwrapper...

## [0.1.1] - 2021-10-01

### Fixed

-   Fixed smaller issues, such as links and consistent naming
-   Fixed rescaling bug

### Changed

-   Updated User Guide
-   Updated Benchmark Models
-   Updated plotting and rvaluation to compare multiple models with each other
-   Updates to Code Documentation now available at: https://sogno-platform.github.io/proloaf/
-   Simplified many script names e.g. `eval_metrics` to `metrics` or `fc_train.py` to `train.py`
-   Restructured many functions, extracted from script files `evaluation.py` and `train.py` and moved to `urils` module
-   tidied up preprocess endpoint

### Added

-   Added Functionalities to explain importance of features in training (Draft version)
-   New plot functionalities, such as error distribution plots
-   New example data
-   Initial tutorial notebook added

## [0.1.0] - 2021-06-05

### Added

-   First tag and release for the ProLoaF project
-   First notebooks are added
-   Firt auto-documentation added
-   Added DCO sign-offs
-   Added CI/CD jobs for lintering, test runs and the docu page

### Changed

-   remove seperate license files from former repos and
-   renamed `plf-utils` package to `utils`
-   added some files to gitignore

### Fixed

-   fix `utils` package
-   issue in NMAE (normalization was applied twice before)
-   typos in `eval_metrics`
-   fixed figure size when plotting (before it was cropped)
