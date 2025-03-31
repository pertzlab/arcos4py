"""Tools for plotting collective events."""

__author__ = """Benjamin Graedel"""
__email__ = "benjamin.graedel@unibe.ch"
__version__ = '0.3.0'

from ._plotting import LineagePlot, NoodlePlot, dataPlots, plotOriginalDetrended, statsPlots

__all__ = ["plotOriginalDetrended", "dataPlots", "statsPlots", "NoodlePlot", "LineagePlot"]
