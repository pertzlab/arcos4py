"""Arcos4py top level module.

This package is a python package for the detection
and tracking of collective events intime-series data and raster images.
"""

__author__ = """Benjamin Graedel"""
__email__ = "benjamin.graedel@unibe.ch"
__version__ = '0.3.2'

from ._arcos4py import ARCOS
from .tools._detect_events import track_events_dataframe, track_events_image

__all__ = ["ARCOS", "track_events_dataframe", "track_events_image"]
