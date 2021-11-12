# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [0.2.0] - in prep

### Fixed

 - Fixed ...

### Changed

 - metrics as class...

### Added

 - Created `Changelog.md`
 - Modelwrapper...

## [0.1.1] - 2021-10-01

### Fixed

 - Fixed smaller issues, such as links and consistent naming
 - Fixed rescaling bug

### Changed

 - Updated User Guide
 - Updated Benchmark Models
 - Updated plotting and rvaluation to compare multiple models with each other
 - Updates to Code Documentation now available at: https://sogno-platform.github.io/proloaf/ 
 - Simplified many script names e.g. `eval_metrics` to `metrics` or `fc_train.py` to `train.py`
 - Restructured many functions, extracted from script files `evaluation.py` and `train.py` and moved to `urils` module
 - tidied up preprocess endpoint

### Added

 - Added Functionalities to explain importance of features in training (Draft version)
 - New plot functionalities, such as error distribution plots
 - New example data
 - Initial tutorial notebook added

## [0.1.0] - 2021-06-05

### Added

- First tag and release for the ProLoaF project
- First notebooks are added
- Firt auto-documentation added
- Added DCO sign-offs
- Added CI/CD jobs for lintering, test runs and the docu page

### Changed

- remove seperate license files from former repos and 
- renamed `plf-utils` package to `utils`
- added some files to gitignore

### Fixed

- fix `utils` package
- issue in NMAE (normalization was applied twice before)
- typos in `eval_metrics`
- fixed figure size when plotting (before it was cropped)