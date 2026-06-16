"""Automatic binning and axis-label heuristics.

``guess_binning`` produces robust edges for an arbitrary branch:
 - integer-valued branches  -> unit-width integer bins
 - bounded variables (cos*, *prob, phi) -> physical bounds
 - everything else          -> robust percentile range (tames fat tails such as
   the IP-significance branches that can reach +/-2000)
"""
from __future__ import annotations

import re
import numpy as np

# Pretty LaTeX axis labels for common branches; falls back to the raw name.
_LABELS = {
    "mass": r"$m_{3\mu}$ [GeV]",
    "mcorr": r"$m_{\mathrm{corr}}$ [GeV]",
    "pt": r"$p_{\mathrm{T}}$ [GeV]",
    "eta": r"$\eta$",
    "phi": r"$\phi$",
    "q2": r"$q^{2}$ [GeV$^{2}$]",
    "q_sq": r"$q^{2}$ [GeV$^{2}$]",
    "m_miss_sq": r"$m^{2}_{\mathrm{miss}}$ [GeV$^{2}$]",
    "e_mu_star": r"$E^{*}_{\mu}$ [GeV]",
    "lxy": r"$L_{xy}$ [cm]",
    "lxyz": r"$L_{xyz}$ [cm]",
    "cos2d": r"$\cos\theta_{2D}$",
    "cos3d": r"$\cos\theta_{3D}$",
    "prob": "vertex prob.",
}

_UNIT_HINT = re.compile(r"(pt|mass|mcorr|lxy|lxyz|energy|_e$|_p$|dxy|dz)", re.I)


def axis_label(name: str) -> str:
    key = name.lower()
    if key in _LABELS:
        return _LABELS[key]
    for stem, lab in _LABELS.items():
        if key.endswith("_" + stem) or key == stem:
            # prepend the prefix (e.g. mu1_pt -> "mu1 pt [GeV]")
            prefix = name[: -len(stem)].rstrip("_").replace("_", " ")
            return (prefix + " " + lab).strip()
    return name.replace("_", " ")


def guess_binning(values: np.ndarray, name: str = "", nbins: int = 40,
                  pmin: float = 0.5, pmax: float = 99.5) -> np.ndarray:
    """Return bin edges for ``values`` (NaN/inf-safe)."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.linspace(0.0, 1.0, nbins + 1)

    nl = name.lower()
    uniq = np.unique(v)

    # integer-like (charge, multiplicities, gen codes, flags)
    if uniq.size <= 200 and np.allclose(v, np.round(v)):
        lo, hi = float(uniq.min()), float(uniq.max())
        if hi - lo <= 200:
            return np.arange(lo - 0.5, hi + 1.5, 1.0)

    # robust range: Tukey (IQR) rule, which tames heavy tails (e.g. the
    # IP-significance branches) far better than raw percentiles. Fall back to
    # percentiles only for near-degenerate distributions.
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        lo = max(lo, float(v.min()))     # never extend beyond the data support
        hi = min(hi, float(v.max()))
    else:
        lo, hi = np.percentile(v, [pmin, pmax])

    # physical bounds for known bounded quantities
    if "phi" in nl and "dphi" not in nl:
        lo, hi = -np.pi, np.pi
    elif nl.endswith("prob") or "_prob" in nl:
        lo, hi = 0.0, 1.0
    elif "cos" in nl:
        lo, hi = max(lo, -1.0), min(hi, 1.0)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            hi = lo + 1.0

    span = hi - lo
    lo -= 0.02 * span
    hi += 0.02 * span
    return np.linspace(lo, hi, nbins + 1)
