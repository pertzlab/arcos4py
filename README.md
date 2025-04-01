# arcos4py

[![pypi](https://img.shields.io/pypi/v/arcos4py.svg)](https://pypi.org/project/arcos4py/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/arcos4py)](https://anaconda.org/conda-forge/arcos4py)
[![python](https://img.shields.io/pypi/pyversions/arcos4py.svg)](https://pypi.org/project/arcos4py/)
[![Build Status](https://github.com/pertzlab/arcos4py/actions/workflows/dev.yml/badge.svg)](https://github.com/bgraedel/arcos4py/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/pertzlab/arcos4py/branch/main/graphs/badge.svg)](https://codecov.io/github/bgraedel/arcos4py)

**arcos4py** is a Python package designed to detect and analyze collective spatiotemporal phenomena in biological imaging data.

- **Documentation:** [https://pertzlab.github.io/arcos4py](https://pertzlab.github.io/arcos4py)
- **GitHub Repository:** [https://github.com/pertzlab/arcos4py](https://github.com/pertzlab/arcos4py)
- **PyPI Package:** [https://pypi.org/project/arcos4py/](https://pypi.org/project/arcos4py/)
- **Free Software License:** MIT

---

## Features

**Automated Recognition of Collective Signalling for Python (arcos4py)** identifies collective spatial events in time-series data or microscopy images. The software tracks waves of protein activity in 2D and 3D cell cultures and follows them over time.

Such collective dynamics have been observed in:
- Epithelial homeostasis (Gagliardi et al., 2020; Takeuchi et al., 2020; Aikin et al., 2020)
- Acinar morphogenesis (Ender et al., 2020)
- Osteoblast regeneration (De Simone et al., 2021)
- Coordination of collective cell migration (Aoki et al., 2017; Hino et al., 2020)

The R package ARCOS ([https://github.com/dmattek/ARCOS](https://github.com/dmattek/ARCOS)) provides a similar R implementation. The `arcos4py` version includes more recent upgrades and added functionality:
- Event tracking directly on image data
- Split/merge detection
- Motion prediction for robust temporal linking

**Data format:** Long-table format with object coordinates, time, and optionally measurements; or binary image sequences for pixel-level analysis.

**Modular API:** Use the full ARCOS class or individual tools via `arcos.tools`. Process binary images directly using `track_events_images` in `arcos4py.tools`.

---

## New in ARCOS with ARCOS.px 🎉

We recently released a major update, **ARCOS.px**, extending `arcos4py` to track *subcellular dynamic structures* like actin waves, podosomes, and focal adhesions directly from binarized time-lapse images.

**Publication:**  
Tracking Coordinated Cellular Dynamics in Time-Lapse Microscopy with ARCOS.px. [*bioRxiv*](https://doi.org/10.1101/2025.03.14.643386)

**What’s new:**
- **Pixel-based tracking** of discontinuous, irregular structures
- **Lineage tracking** across merges and splits
- Optional **Motion prediction** and frame-to-frame linking with **optimal transport**
- **Support for DBSCAN and HDBSCAN** clustering and custom clustering methods
- **Improved memory usage and lazy evaluation** for long time series
- Integrated into **Napari** via the [`arcosPx-napari plugin`](https://github.com/pertzlab/arcospx-napari) plugin
---

### Notebooks and Reproducible Analysis

To facilitate reproducibility and provide practical examples, we have made available a collection of Jupyter notebooks that demonstrate the use of ARCOS.px in various scenarios. These notebooks cover:

**Wave Simulation**: Scripts to simulate circular & directional waves, and target & chaotic patterns using cellular automaton.

**Synthetic RhoA** Activity Wave: Analysis of optogenetically induced synthetic RhoA activity waves.

**Podosome Dynamics**: Tracking and analysis of podosome-like structures under different conditions.

**Actin Wave Tracking**: Tracking and analysis of actin waves in 2D and extractin temporal order.

You can access these notebooks in the [ARCOSpx-publication](https://github.com/pertzlab/ARCOSpx-publication) repository under the scripts directory.


## Installation

Install from PyPI:
```bash
pip install arcos4py
```

Napari Plugin
-------------
Arcos4py is also available as a Napari Plugin [arcos-gui](https://github.com/pertzlab/arcos-gui).
[arcos-gui](https://github.com/pertzlab/arcos-gui) can simplify parameter finding and visualization.

or images directly: [arcosPx-napari](https://github.com/pertzlab/arcospx-napari)

[![arcos_demo](https://img.youtube.com/vi/hG_z_BFcAiQ/0.jpg)](https://www.youtube.com/watch?v=hG_z_BFcAiQ)

## Credits

[Maciej Dobrzynski](https://github.com/dmattek) created the first version of ARCOS.

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
