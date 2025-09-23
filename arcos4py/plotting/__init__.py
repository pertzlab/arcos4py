"""Tools for plotting collective events."""

__author__ = """Benjamin Graedel"""
__email__ = "benjamin.graedel@unibe.ch"
__version__ = '0.3.2'

from ._plotting import LineagePlot, NoodlePlot, dataPlots, plotOriginalDetrended, save_animation_frames, statsPlots

__all__ = ["plotOriginalDetrended", "dataPlots", "statsPlots", "NoodlePlot", "LineagePlot", "save_animation_frames"]
