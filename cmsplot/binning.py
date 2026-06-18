"""Automatic binning and axis-label heuristics.

``guess_binning`` produces robust edges for an arbitrary branch:
 - integer-valued branches  -> unit-width integer bins
 - bounded variables (cos*, *prob, phi) -> physical bounds
 - everything else          -> robust percentile range (tames fat tails such as
   the IP-significance branches that can reach +/-2000)
"""
from __future__ import annotations

import numpy as np

# =============================================================================
# Branch -> x-axis title (issue 4)
#
# Three layers, tried in order:
#   1. AXIS_TITLES  -- exact full-branch overrides (highest priority, also the
#      place where the config's AXIS_TITLES gets merged in at runtime);
#   2. a parser that peels recognised trailing tokens off the branch name
#      (error/significance, then hypothesis/vertex qualifiers), maps the leading
#      physics quantity via _STEMS, decorates an object prefix, and re-attaches
#      the qualifiers as a short parenthetical;
#   3. fall back to the raw name with underscores turned into spaces.
# It is deliberately a "first good enough" pass: extend the three small dicts
# below (or set AXIS_TITLES in the sample config) to refine any label.
# =============================================================================

# 1. exact overrides ----------------------------------------------------------
AXIS_TITLES = {
    "mass":      r"$m_{3\mu}$ [GeV]",
    "jpsi_mass": r"$m_{\mu\mu}$ [GeV]",
    "mcorr":     r"$m_{\mathrm{corr}}$ [GeV]",
}

# 2a. leading physics quantity -> (LaTeX symbol, unit) ------------------------
#     longer (more specific) stems are matched before shorter ones.
_STEMS = {
    "m_miss2":          (r"m^{2}_{\mathrm{miss}}", r"GeV$^{2}$"),
    "dist_along_b_dir": (r"d_{\parallel}",         "cm"),
    "dist_to_b_dir":    (r"d_{\perp}",             "cm"),
    "reliso":           (r"\mathrm{rel.\,iso.}",   ""),
    "ip3d":             (r"\mathrm{IP}_{3D}",      "cm"),
    "cos2d":            (r"\cos\theta_{2D}",       ""),
    "cos3d":            (r"\cos\theta_{3D}",       ""),
    "lxyz":             (r"L_{xyz}",               "cm"),
    "lxy":              (r"L_{xy}",                "cm"),
    "mcorr":            (r"m_{\mathrm{corr}}",     "GeV"),
    "mass":             (r"m",                     "GeV"),
    "q2":               (r"q^{2}",                 r"GeV$^{2}$"),
    "pt":               (r"p_{\mathrm{T}}",        "GeV"),
    "eta":              (r"\eta",                  ""),
    "phi":              (r"\phi",                  ""),
    "prob":             (r"\mathrm{vtx.\,prob.}",  ""),
    "dxy":              (r"d_{xy}",                "cm"),
    "dz":               (r"d_{z}",                 "cm"),
}
_STEMS_BY_LEN = sorted(_STEMS, key=len, reverse=True)

# 2b. object prefixes -> symbol (e.g. mu1_pt -> p_T(mu_1)) --------------------
_PREFIX_SYM = {
    "mu":   r"\mu",   "mu1": r"\mu_{1}", "mu2": r"\mu_{2}", "mu3": r"\mu_{3}",
    "nu1":  r"\nu_{1}", "nu2": r"\nu_{2}",
    "jpsi": r"J/\psi", "b": r"B_{c}", "bc": r"B_{c}",
}

# 2c. trailing hypothesis / vertex qualifier tokens -> short words ------------
#     (these are flight-direction or anchor-vertex suffixes in the ntuple)
_QUAL_WORD = {
    "coll":  "coll.",     "ev":   "equal-vel.",
    "jpsi":  r"$J/\psi$ dir.",
    "fjpsi": r"$J/\psi$ SV", "f3m": r"3$\mu$ SV",
    "sv":    r"3$\mu$ SV", "pv":   "PV",
    "sol1":  "sol. 1",    "sol2": "sol. 2",
}
_QUAL_TOKENS = sorted(_QUAL_WORD, key=len, reverse=True)

_USER_TITLES = {}


def set_user_titles(mapping):
    """Merge user/config-supplied exact branch->title overrides (issue 4)."""
    if mapping:
        _USER_TITLES.update(mapping)


def _cone_word(d):
    """Isolation/cone suffix digits -> a Delta R qualifier ('04' -> dR<0.4)."""
    if len(d) == 2 and d[0] == "0":
        return r"$\Delta R<0.%s$" % d[1]
    if len(d) == 1:
        return r"$\Delta R<0.%s$" % d
    return r"$\Delta R<%s$" % d


def _compose(stem, prefix, suffix, qual_words):
    latex, unit = _STEMS[stem]
    sym = latex
    if prefix:
        psym = _PREFIX_SYM.get(prefix, prefix.replace("_", r"\,"))
        sym = r"%s(%s)" % (latex, psym)
    if suffix == "sig":                       # significance: value / sigma, unitless
        core, unit = r"$%s/\sigma$" % sym, ""
    elif suffix == "err":                     # uncertainty: sigma(value), keep unit
        core = r"$\sigma(%s)$" % sym
    else:
        core = r"$%s$" % sym
    out = core + (" [%s]" % unit if unit else "")
    # drop consecutive duplicate qualifiers (e.g. sv_sv -> one "3mu SV")
    dedup = [w for i, w in enumerate(qual_words) if i == 0 or w != qual_words[i - 1]]
    if dedup:
        out += " (%s)" % ", ".join(dedup)
    return out


def axis_label(name: str) -> str:
    if name in _USER_TITLES:
        return _USER_TITLES[name]
    if name in AXIS_TITLES:
        return AXIS_TITLES[name]

    base = name.lower()
    # peel a single error/significance suffix
    suffix = None
    if base.endswith("_sig"):
        suffix, base = "sig", base[:-4]
    elif base.endswith("_err"):
        suffix, base = "err", base[:-4]
    # peel a trailing numeric cone token (e.g. mu_reliso_04 -> dR<0.4)
    cone = None
    head, _, tail = base.rpartition("_")
    if head and tail.isdigit():
        cone, base = _cone_word(tail), head
    # peel trailing hypothesis/vertex qualifier tokens (right to left)
    quals = []
    changed = True
    while changed:
        changed = False
        for tok in _QUAL_TOKENS:
            if base.endswith("_" + tok):
                quals.insert(0, tok)
                base = base[: -(len(tok) + 1)]
                changed = True
                break
    # match the remaining core against a known stem, allowing an object prefix
    stem, prefix = None, ""
    if base in _STEMS:
        stem = base
    else:
        for s in _STEMS_BY_LEN:
            if base.endswith("_" + s):
                stem, prefix = s, base[: -(len(s) + 1)]
                break
    if stem is None:
        return name.replace("_", " ")
    qual_words = [_QUAL_WORD[q] for q in quals] + ([cone] if cone else [])
    return _compose(stem, prefix, suffix, qual_words)


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
