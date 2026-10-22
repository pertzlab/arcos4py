"""Tools for detecting collective events."""

__author__ = """Benjamin Graedel"""
__email__ = "benjamin.graedel@unibe.ch"
__version__ = '0.3.3'

from ._binarize_detrend import binData
from ._cleandata import clipMeas, interpolation, remove_image_background
from ._coactivation import (
    CoactivationCalibration,
    CoactivationReport,
    calibrate_coactivation,
    detect_coactivation_events,
)
from ._coactivation_image import detect_coactivation_events_image
from ._detect_events import (
    DataFrameTracker,
    ImageTracker,
    Linker,
    detectCollev,
    estimate_eps,
    track_events_dataframe,
    track_events_image,
)
from ._filter_events import filterCollev
from ._stats import calcCollevStats, calculate_statistics, calculate_statistics_per_frame

__all__ = [
    "binData",
    "clipMeas",
    "interpolation",
    "detectCollev",
    "filterCollev",
    "calcCollevStats",
    "estimate_eps",
    "track_events_dataframe",
    "track_events_image",
    "detect_coactivation_events",
    "detect_coactivation_events_image",
    "calibrate_coactivation",
    "CoactivationCalibration",
    "CoactivationReport",
    "calculate_statistics",
    "calculate_statistics_per_frame",
    "Linker",
    "ImageTracker",
    "DataFrameTracker",
    "remove_image_background",
]
