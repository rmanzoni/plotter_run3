"""cmsplot - a fast uproot/numpy/mplhep stack-plotter for the R(J/psi) ntuples.

Reads flat ntuples once into memory, auto-guesses a sensible binning for every
branch, and produces CMS-styled stacked MC vs data plots with a ratio panel.
"""
from .core import Sample, Process, Histogrammer, StackPlotter, run
from . import style, binning, derived
from .derived import (Derived, p4_ptetaphim, p4_ptetaphie, invariant_mass,
                      MASS_E, MASS_MU, MASS_PI, MASS_K, MASS_PROTON, MASS_JPSI,
                      MASS_BPLUS)

__all__ = [
    "Sample", "Process", "Histogrammer", "StackPlotter", "run",
    "style", "binning", "derived",
    "Derived", "p4_ptetaphim", "p4_ptetaphie", "invariant_mass",
    "MASS_E", "MASS_MU", "MASS_PI", "MASS_K", "MASS_PROTON", "MASS_JPSI",
    "MASS_BPLUS",
]
