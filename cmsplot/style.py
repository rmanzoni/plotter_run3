"""CMS plotting style and accessible (Petroff) colour palettes.

The CMS plotting guidelines adopt M. Petroff's colour-vision-deficiency-friendly
colour sequences (arXiv:2107.02270). The 6/8/10-colour hex values below are the
CMS-endorsed sets (identical to ROOT's kP6*/kP10* and mplhep's petroff_* cycles).
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# --- CMS-recommended, CVD-safe categorical palettes ----------------------------
PETROFF_6 = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
PETROFF_8 = ["#1845fb", "#ff5e02", "#c91f16", "#c849a9",
             "#adad7d", "#86c8dd", "#578dff", "#656364"]
PETROFF_10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
              "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]


def palette(n: int) -> list[str]:
    """Return ``n`` distinct CMS-accessible colours (cycles the 10-set if n>10)."""
    if n <= 6:
        base = PETROFF_6
    elif n <= 8:
        base = PETROFF_8
    elif n <= 10:
        base = PETROFF_10
    else:
        base = PETROFF_10 * (n // 10 + 1)
    return list(base[:n])


_MPLHEP = None


def mplhep():
    """Lazily import mplhep; return the module or ``False`` if unavailable."""
    global _MPLHEP
    if _MPLHEP is None:
        try:
            import mplhep as _hep
            _MPLHEP = _hep
        except ImportError:
            _MPLHEP = False
    return _MPLHEP


def set_cms_style():
    """Apply the CMS matplotlib style (mplhep if present, else a close fallback)."""
    hep = mplhep()
    if hep:
        plt.style.use(hep.style.CMS)
    else:  # minimal approximation: inward ticks on all four sides, no legend frame
        plt.rcParams.update({
            "font.size": 16, "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.top": True, "ytick.right": True,
            "xtick.major.size": 10, "ytick.major.size": 10,
            "xtick.minor.size": 5, "ytick.minor.size": 5,
            "xtick.minor.visible": True, "ytick.minor.visible": True,
            "axes.linewidth": 1.2, "legend.frameon": False,
        })


def cms_label(ax, *, lumi=None, com=13.6, data=True, extra="Preliminary"):
    """Draw the standard CMS top-left label and top-right lumi/sqrt(s) tag.

    ``data=False`` automatically prepends 'Simulation'.
    """
    hep = mplhep()
    if hep:
        hep.cms.label(extra, ax=ax, data=data, lumi=lumi, com=com, loc=0)
        return
    ax.text(0.0, 1.005, "CMS", transform=ax.transAxes,
            fontweight="bold", fontsize="large", va="bottom")
    tag = ("" if data else "Simulation ") + extra
    ax.text(0.16, 1.005, tag, transform=ax.transAxes, style="italic", va="bottom")
    right = (f"{lumi:.1f} fb$^{{-1}}$ " if lumi else "") + f"({com:g} TeV)"
    ax.text(1.0, 1.005, right, transform=ax.transAxes, ha="right", va="bottom")
