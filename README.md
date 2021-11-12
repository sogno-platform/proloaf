<img src="./logo.png" width="500">

# 

[![pipeline status](https://git.rwth-aachen.de/acs/public/automation/plf/proloaf/badges/master/pipeline.svg)](https://git.rwth-aachen.de/acs/public/automation/plf/proloaf/-/commits/master)
[![Python](https://img.shields.io/badge/python->=3.6-blue)](https://www.python.org)

[**Website**](https://git.rwth-aachen.de/acs/public/automation/plf/proloaf)
| [**Docs**](https://acs.pages.rwth-aachen.de/public/automation/plf/proloaf/)
| [**Install Guide**](https://acs.pages.rwth-aachen.de/public/automation/plf/proloaf/docs/getting-started/)
| [**Tutorial**](https://acs.pages.rwth-aachen.de/public/automation/plf/proloaf/docs/tutorials/)

ProLoaF is a probabilistic load forecasting project.

## Installation

First, clone this Repository and initialize the submodules:
```bash
git clone --recurse-submodule https://git.rwth-aachen.de/acs/public/automation/plf/proloaf.git
```
or if you have already cloned the project and you are missing e.g. the open data directory for execution run:
```bash
git submodule update --init --recursive
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

ProLoaF supports Python 3.8 and higher. For further information see [Getting Started](https://acs.pages.rwth-aachen.de/public/automation/plf/proloaf/docs/getting-started/)


## Gettings started, Key Capabilities, Example Workflows & References 
To keep all infos on what you can do with ProLoaF and how to get there, our
[documentation overview](https://acs.pages.rwth-aachen.de/public/automation/plf/proloaf/docs/overview/), is the place where you'll find all answers.

## Related Publications
G. Gürses-Tran, H. Flamme, and A. Monti, "Probabilistic Load Forecasting for Day-Ahead Congestion Mitigation," The 16th International Conference on Probabilistic Methods Applied to Power Systems, PMAPS 2020, 2020-08-18 - 2020-08-21, Liège, Belgium, ISBN 978-1-7281-2822-1, [Access Online](http://aimontefiore.org/PMAPS2020/openconf/modules/request.php?module=oc_proceedings&action=view.php&id=92&type=3&a=Accept)


## License
This project is licensed to the [Apache Software Foundation (ASF)](http://www.apache.org/licenses/LICENSE-2.0).
