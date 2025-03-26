# Changelog

## [0.3.0] - 2025-03-26
### Added
- Added support for split/merge detection

## [0.2.5] - 2024-09-10
### Fixed
- scikit-image dependency version

### Changed
- Updated transportation linking to use the POT library

## [0.2.4] - 2024-03-19
### Fixed
- Hard-coded column names in stats functions
- False example for a plot in the documentation

### Changed
- Updated plotOriginalDetrended to include separate methods for plotting detrended and original data
- Updated plotOriginalDetrended to include markers for binarized regions
- changes to parameter names to unify naming conventions across packages
old Parameter names are still supported but will be deprecated in the future.
- updated documentation to reflect changes in parameter names
- Noodleplot supports kwargs for plot customization
- Changes to the way p value is represented on the validation plots
- Validation plots now dont include original metrics

### Added
- Support for python 3.12
- Drop support for python 3.8

## [0.2.3] - 2023-10-10
### Fixed
- Noodle plot would produce an axis error if no collective events were detected
- Noodle plot would produce false results if object id was not an integer

### Changed
- More input data validation for stats functions

## [0.2.2] - 2023-09-22
### Fixed
- Bug in eps estimation for DBSCAN clustering

### Changed
- Updated input data validation for remove_background function
- ImageTracker, DataFrameTracker, remove_background can be imported from arcos4py.tools
- Updated api documentation

### Added
- Added new function to calculate more statistics of collective events
- Added new function to calculate statistics per frame of collective events
- Accont for downsampling in in track_events_image fuction for parameters

## [0.2.1] - 2023-08-09
### Fixed
- patch for dependencies in pyproject.toml file


## [0.2.0] - 2023-08-09
### Added
- Funcionallity to directly apply ARCOS to images
- Simple movement predictor to improve tracking
- HDBSCAN as an alternative clustering method
- Transportation linking as an alternative linking method
- Preprocessing function for detrending of images
- Unit tests for image tracking
- Added optional Progressbar

### Changed
- Refactorization of event detection to improve memory usage and simplify algorithm
- ARCOS main class now also supports event detection without specifying a tracking column
- Package is now tested on python 3.8 to 3.11 (dropped 3.7 and added 3.11)
- Event detection can now ingest data lazily
- Old detectCollev class is now deprecated in favor of track_events_image and track_events_dataframe

## [0.1.6] - 2022-05-02
### Fixed
- Bug where trackCollev would overwrite the inputdata in the ARCOS object, preventing repeat tracking of collective events.
- Spelling mistake in \__init__.py

### Changed
- None detrending now rescales measurements to 0,1 range on a global scale and not on a track-by-track basis.
- Added the parameter epsPrev by separating it from eps.
epsPrev is the maximum distance cells within collective events can be separated from each other when linking them from one frame to the next.
If set to 'None', as default, the same value as for eps is used.

### Added
- New function estimate_eps (import from tools) to estimate the eps paramter used for DBSCAN clustering based on the nearest neighbour distribution. Three methods are supported, either mean of NN, median of NN or kneepoint of the sorted NN distribution.
- Functions to perform resampling and bootstrapping to perform validation of arcos output.
- Unittests for added functionallity.

## [0.1.5] - 2022-08-23
### Changed
- Changed running median for global smoothing of trajectories from scipy running_median
to pandas running.median, since this allows a different endrule mode
- Changed running median endrule for local smoothing of trajectories from constant to nearest

## [0.1.4] - 2022-06-24
### Fixed
- Fix for lm detrending
- Fix for none detrending
- Fix grouping failure when object id was a string in rescale data method in binarization class

### Changed
- Binarization Thresholding value now sets everything to be active for >=, as opposed to > like it was before

## [0.1.3] - 2022-05-23
### Fixed
- Bug where if object id was a string, splitting arrays into groups would fail.
- Hardcoded collective id name in stats module
- Wrong example in main module

### Added
- More examples for plots in plotting module
- mkdocstrings-python-legacy extension (required for building docs)

## [0.1.2] - 2022-05-03
### Added
- NoodlePlot for collective events

### Changed
- binarize_detrend: converted pandas operations to numpy for performance improvements
- detect_events: converted pandas operations to numpy for performance imporovements
- stats: converted pandas operations to numpy for performance improvements
- various small changes
- updated docstrings to match changes

### Fixed
- numpy warning caused by stats module

## [0.1.1] - 2022-04-04
### Added
- More plotting functionallity to the plotting module.
    - Measurment density plot
    - Tracklength histogram
    - Position/T plot
    - Collective event statistcs plot

### Changed
- Interpolation class in tools now uses pandas.interpolate to interpolate missing values.
- Interpolation now interpolates all values in all columns of the dataframe.
- Improved usage section in the documentation.

### Fixed
- Bug in trackCollev class that would lead to an error message in some cases.
- Spelling in docstrings.

## [0.1.0] - 2022-03-26
### Added
- First release on PyPI.
