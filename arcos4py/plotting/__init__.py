"""Tools for plotting collective events."""

__author__ = """Benjamin Graedel"""
__email__ = "benjamin.graedel@unibe.ch"
__version__ = '0.2.5'

from ._plotting import LineagePlot, NoodlePlot, dataPlots, plotOriginalDetrended, statsPlots, save_animation_frames

__all__ = ["plotOriginalDetrended", "dataPlots", "statsPlots", "NoodlePlot", "LineagePlot", "save_animation_frames"]
