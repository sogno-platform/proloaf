<img src="./logo.png" width="500">

#

[![pipeline status](https://git.rwth-aachen.de/acs/public/automation/plf/proloaf/badges/master/pipeline.svg)](https://git.rwth-aachen.de/acs/public/automation/plf/proloaf/-/commits/master)
[![Python](https://img.shields.io/badge/python->=3.8-blue)](https://www.python.org)

[**Website**](https://git.rwth-aachen.de/acs/public/automation/plf/proloaf)
| [**Docs**](https://acs.pages.rwth-aachen.de/public/automation/plf/proloaf/)
| [**Install Guide**](https://acs.pages.rwth-aachen.de/public/automation/plf/proloaf/docs/getting-started/)
| [**Tutorial**](https://acs.pages.rwth-aachen.de/public/automation/plf/proloaf/docs/tutorials/)

ProLoaF is a probabilistic load forecasting project.

## Description

ProLoaF makes use of the big data paradigm that allows machine learning algorithms, trained with data from the power system field. The core comprises a recurrent neural network (encoder-decoder architecture) to predict the target variable.

The targets can vary from timerseries of PV, Wind or other generators to most commonly the total energy consumption.
Both, the relevant input data history and prediction horizon are arbitrarily long and customizable for any specific need.

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
[user documentation](https://sogno-platform.github.io/proloaf/docs/overview/), is the place where you'll find all answers.

## Related Publications

We kindly ask all academic publications employing components of ProLoaF to cite one of the following papers:

- G. Gürses-Tran, H. Flamme, and A. Monti, "Probabilistic Load Forecasting for Day-Ahead Congestion Mitigation," The 16th International Conference on Probabilistic Methods Applied to Power Systems, PMAPS 2020, 2020-08-18 - 2020-08-21, Liège, Belgium, ISBN 978-1-7281-2822-1, [Access Online](https://ieeexplore.ieee.org/document/9183670)

## Copyright

2019-2021, Institute for Automation of Complex Power Systems, EONERC

## License

This project is licensed to the [Apache Software Foundation (ASF)](http://www.apache.org/licenses/LICENSE-2.0).

```
This program is free software: you can redistribute it and/or modify
it under the terms of the Apache License version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
Apache License for more details.

You should have received a copy of the Apache License
along with this program.  If not, see <http://www.apache.org/licenses/LICENSE-2.>.
```

For other licensing options please consult [Prof. Antonello Monti](mailto:amonti@eonerc.rwth-aachen.de).

## Funding

<a rel="funding" href="https://cordis.europa.eu/project/id/824414"><img alt="CoordiNet" style="border-width:0" src="docs/static/img/coordinet_logo.png" height="63"/><img alt="PLATONE" style="border-width:0" src="docs/static/img/platone_logo.png" height="63"/></a><a rel="funding" href="https://cordis.europa.eu/project/id/864300"><img alt="H2020" style="border-width:0" src="https://hyperride.eu/wp-content/uploads/2020/10/europa_flag_low.jpg" height="63"/></a><br />
This work was supported by <a rel="CoordiNet" href="https://coordinet-project.eu/">CoordiNet Project</a> and <a rel="Platone" href="https://platone-h2020.eu/">PLATform for Operation of distribution NEtworks </a> (Platone), funded by the European Union's Horizon 2020 research and innovation programme under respectively <a rel="H2020" href="https://cordis.europa.eu/project/id/824414"> grant agreement No. 824414</a> and <a rel="H2020" href="https://cordis.europa.eu/project/id/864300"> grant agreement No. 864300</a>.
<!-- This work was supported by <a rel="Platone" href="https://platone-h2020.eu/">PLATform for Operation of distribution NEtworks </a> (Platone), funded by the European Union's Horizon 2020 research and innovation programme under <a rel="H2020" href="https://cordis.europa.eu/project/id/864300"> grant agreement No. 864300</a>. -->

## Contact

[![EONERC ACS Logo](docs/static/img/eonerc_logo.png)](http://www.acs.eonerc.rwth-aachen.de)

- Gonca Gürses-Tran <gguerses@eonerc.rwth-aachen.de>
- Florian Oppermann <florian.oppermann@eonerc.rwth-aachen.de>

[Institute for Automation of Complex Power Systems (ACS)](http://www.acs.eonerc.rwth-aachen.de)
[EON Energy Research Center (EONERC)](http://www.eonerc.rwth-aachen.de)
[RWTH University Aachen, Germany](http://www.rwth-aachen.de)
