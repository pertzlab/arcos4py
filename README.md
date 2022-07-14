# arcos4py


[![pypi](https://img.shields.io/pypi/v/arcos4py.svg)](https://pypi.org/project/arcos4py/)
[![python](https://img.shields.io/pypi/pyversions/arcos4py.svg)](https://pypi.org/project/arcos4py/)
[![Build Status](https://github.com/bgraedel/arcos4py/actions/workflows/dev.yml/badge.svg)](https://github.com/bgraedel/arcos4py/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/bgraedel/arcos4py/branch/main/graphs/badge.svg)](https://codecov.io/github/bgraedel/arcos4py)



Arcos4py is a python package to detect collective Spatio-temporal phenomena.

* Documentation: <https://bgraedel.github.io/arcos4py>
* GitHub: <https://github.com/bgraedel/arcos4py>
* PyPI: <https://pypi.org/project/arcos4py/>
* Free software: MIT


## Features

Automated Recognition of Collective Signalling for python (arcos4py) is a python port of the R package ARCOS (https://github.com/dmattek/ARCOS) to identify collective spatial events in time-series data.
The software identifies collective protein activation in 2- and 3D cell cultures and can track events over time. Such collective waves have been recently identified in various biological systems and have been demonstrated to play a crucial role in the maintenance of epithelial homeostasis (Gagliardi et al., 2020, Takeuchi et al., 2020, Aikin et al., 2020),
in the acinar morphogenesis (Ender et al., 2020), osteoblast regeneration (De Simone et al., 2021), and the coordination of collective cell migration (Aoki et al., 2017, Hino et al., 2020).

Despite its focus on cell signaling, the framework can also be applied to other spatiotemporally correlated phenomena.

### Todo's
- Add additional tests for binarization and de-biasing modules.
- Add local indicators of spatial autocorrelation (LISA) as a binarization method option.

Data Format
-----------
The time series should be arranged in a long table format where each row defines the object's location, time, and optionally the measurement value.

ARCOS defines an ARCOS object on which several class methods can be used to prepare the data and calculate collective events.
Optionally the objects used in the ARCOS class can be used individually by importing them from arcos.tools

Installation
------------
Arcos4py can be installed from PyPI with:

        pip install arcos4py

Napari Plugin
-------------
Arcos4py is also available as a Napari Plugin [arcos-gui](https://github.com/bgraedel/arcos-gui). 
[arcos-gui](https://github.com/bgraedel/arcos-gui) can simplify parameter finding and visualization.


[![arcos_demo](https://img.youtube.com/vi/hG_z_BFcAiQ/0.jpg)](https://www.youtube.com/watch?v=hG_z_BFcAiQ)

## Credits

[Maciej Dobrzynski](https://github.com/dmattek) created the original ARCOS algorithm.

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
