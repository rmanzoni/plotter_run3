"""cmsplot - a fast uproot/numpy/mplhep stack-plotter for the R(J/psi) ntuples.

Reads flat ntuples once into memory, auto-guesses a sensible binning for every
branch, and produces CMS-styled stacked MC vs data plots with a ratio panel.
"""
from .core import Sample, Process, Histogrammer, StackPlotter, run
from . import style, binning

__all__ = [
    "Sample", "Process", "Histogrammer", "StackPlotter", "run",
    "style", "binning",
]
