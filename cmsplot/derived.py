"""Derived variables: new columns computed on the fly from existing branches.

A derived variable is declared in the config as a ``Derived(func, inputs)`` where
``func(arrays) -> 1D np.ndarray`` maps the per-sample dict of branch arrays to a
new per-event column, and ``inputs`` lists the branches ``func`` reads (declared
explicitly so the reader can still prune the ntuple to just those columns and
keep memory bounded). Example (config side)::

    DERIVED = {
        "jpsi_k_mass": Derived(
            func=lambda a: invariant_mass(
                p4_ptetaphim(a["jpsi_rf_pt"], a["jpsi_rf_eta"],
                             a["jpsi_rf_phi"], a["jpsi_rf_mass"]),
                p4_ptetaphim(a["mu3_pt"], a["mu3_eta"], a["mu3_phi"], MASS_K)),
            inputs=("jpsi_rf_pt", "jpsi_rf_eta", "jpsi_rf_phi", "jpsi_rf_mass",
                    "mu3_pt", "mu3_eta", "mu3_phi")),
    }

Once computed the new column behaves exactly like a real branch: it can be
plotted (``--branches jpsi_k_mass``), gets ``BINNING``/``AXIS_TITLES`` overrides
by name, and can feed a datacard (``--datacard-branches jpsi_k_mass``).

The engine computes derived columns once per sample, right after the selection-
filtered arrays are read and BEFORE the cocktail split, so every split component
inherits the column for free. A derived variable whose inputs are not all present
in a given sample (e.g. a gen-only input on the data sample) is silently skipped
for that sample.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# --- particle masses [GeV] (PDG) ---------------------------------------------
MASS_E      = 0.0005109989
MASS_MU     = 0.1056583755
MASS_PI     = 0.13957039     # charged pion
MASS_K      = 0.493677       # charged kaon
MASS_PROTON = 0.93827208
MASS_JPSI   = 3.0969
MASS_BPLUS  = 5.27934
MASS_BC     = 6.27447       # PDG Bc+ mass [GeV] (matches RJPsiGenHistory.M_BC)

# proper-time conversion: t[ps] = (m/p) * L[cm] * PS_PER_CM, with
# PS_PER_CM = 1e12 / c[cm/s] = 1e12 / 2.99792458e10 ~ 33.3564 ps/cm.
PS_PER_CM   = 1.0e12 / 2.99792458e10


@dataclass
class Derived:
    """A new column computed from existing branches.

    func   : callable mapping the per-sample ``arrays`` dict -> 1D numpy array
             (one value per surviving event).
    inputs : the branch names ``func`` reads. Declared so the reader prunes the
             ntuple to just these columns; if any is absent in a given sample the
             variable is skipped there rather than raising.
    """
    func: Callable
    inputs: tuple = field(default_factory=tuple)


# --- four-vector helpers ------------------------------------------------------
# A p4 is a (4, N) float64 array of [px, py, pz, E]; helpers broadcast a scalar
# mass/energy over the N events. Add p4s with ``+`` and feed the sum to
# ``invariant_mass`` (which also accepts the addends directly, variadically).
def p4_ptetaphim(pt, eta, phi, mass):
    """Build a (4, N) [px, py, pz, E] p4 from (pt, eta, phi) and a mass
    hypothesis (scalar, e.g. MASS_K, or a per-event array)."""
    pt = np.asarray(pt, "float64")
    eta = np.asarray(eta, "float64")
    phi = np.asarray(phi, "float64")
    mass = np.asarray(mass, "float64")
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return np.stack([px, py, pz, e])


def p4_ptetaphie(pt, eta, phi, energy):
    """Build a (4, N) [px, py, pz, E] p4 from (pt, eta, phi, energy)."""
    pt = np.asarray(pt, "float64")
    eta = np.asarray(eta, "float64")
    phi = np.asarray(phi, "float64")
    energy = np.asarray(energy, "float64")
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return np.stack([px, py, pz, energy])


def invariant_mass(*p4s):
    """Invariant mass of a sum of p4s. Pass either one (4, N) array (already a
    sum) or several to be summed. Negative m^2 (numerical) is floored to 0."""
    s = p4s[0] if len(p4s) == 1 else sum(p4s)
    s = np.asarray(s, "float64")
    px, py, pz, e = s[0], s[1], s[2], s[3]
    m2 = e * e - (px * px + py * py + pz * pz)
    return np.sqrt(np.clip(m2, 0.0, None))


# --- binning / stitching helpers ---------------------------------------------
# Used to build LHCb-style multi-dimensional *stitched* templates: each event is
# placed into a 0-based bin index along each axis, the per-axis indices are then
# folded into a single global bin index, and that global index is histogrammed
# with unit-width integer edges -> one long 1D super-template whose blocks are
# the 2D (decay-time x Z) cells, each holding an m^2_miss distribution.
def bin_index(values, edges):
    """Map ``values`` to a 0-based bin index for the given monotonic ``edges``.

    Returns a float array: bin index in ``[0, len(edges)-2]`` for in-range
    values, ``NaN`` for under/overflow (and for non-finite inputs). The top edge
    is treated as belonging to the last bin (np.histogram convention), so this
    matches how the engine later histograms the stitched index. NaN entries are
    dropped downstream by ``_hist`` (i.e. NO over/underflow folding).
    """
    v = np.asarray(values, "float64")
    edges = np.asarray(edges, "float64")
    n = len(edges) - 1
    out = np.digitize(v, edges, right=False).astype("float64") - 1.0
    out[v == edges[-1]] = n - 1            # close the last bin on the right
    bad = ~np.isfinite(v) | (out < 0) | (out >= n)
    out[bad] = np.nan
    return out


def equal_velocity_momentum(visible_p4, m_parent):
    """Equal-velocity ("rest-frame") parent momentum magnitude |p_parent|.

    Given the visible-system four-vector ``visible_p4`` ((4, N) [px, py, pz, E])
    and the parent mass ``m_parent``, returns ``|p_parent| = (m_parent / m_vis) *
    |p_vis|`` -- the same construction the candidate uses for ``bc_full_p4_*``
    (the flight direction only orients it; the magnitude is direction-free).
    """
    s = np.asarray(visible_p4, "float64")
    px, py, pz = s[0], s[1], s[2]
    p_vis = np.sqrt(px * px + py * py + pz * pz)
    m_vis = invariant_mass(s)
    good = m_vis > 0.0
    return np.where(good, (m_parent / np.where(good, m_vis, 1.0)) * p_vis, np.nan)


def proper_time_ps(length_cm, mass, momentum):
    """Proper decay time [ps] = mass * L / (c * |p|) for L in cm, mass/|p| in GeV.

    ``length_cm`` is the lab-frame flight length (cm), ``mass`` the parent mass
    and ``momentum`` the parent momentum magnitude (both GeV). Non-positive
    momenta yield NaN.
    """
    L = np.asarray(length_cm, "float64")
    p = np.asarray(momentum, "float64")
    return mass * L / np.where(p > 0.0, p, np.nan) * PS_PER_CM


def stitch_index(indices, sizes):
    """Fold per-axis 0-based bin indices into one global (row-major) bin index.

    ``indices`` is an outer-to-inner list of 1D index arrays (e.g. [it, iZ, im]),
    ``sizes`` the corresponding axis sizes ([N_t, N_Z, N_m]). Returns
    ``(((i0)*N1 + i1)*N2 + i2 ...)`` as a float array; NaN in any axis (an
    out-of-range event) propagates to NaN, so the event is dropped from the
    histogram. The global template therefore has ``prod(sizes)`` bins.
    """
    g = np.asarray(indices[0], "float64")
    for k in range(1, len(indices)):
        g = g * sizes[k] + np.asarray(indices[k], "float64")
    return g


# --- engine-side helpers ------------------------------------------------------
def expand_inputs(names, derived):
    """Resolve a set of requested names (which may include derived-variable
    names, possibly chained) down to the set of REAL branches that must be read.
    A name that is not a derived key is assumed to be a real branch."""
    if not names:
        return set()
    out, seen, stack = set(), set(), list(names)
    while stack:
        nm = stack.pop()
        if nm in seen:
            continue
        seen.add(nm)
        if derived and nm in derived:
            stack.extend(derived[nm].inputs)
        else:
            out.add(nm)
    return out


def compute_derived(arrays, derived, to_float32=True):
    """Add each derived column to ``arrays`` in place (and return it).

    Skips a variable if it would clobber a real branch of the same name, if any
    of its declared inputs is missing from this sample, or if its func raises /
    returns a wrongly shaped result.
    """
    if not derived or not arrays:
        return arrays
    nrows = len(next(iter(arrays.values())))
    for name, spec in derived.items():
        if name in arrays:                       # never shadow a real branch
            continue
        inputs = getattr(spec, "inputs", ()) or ()
        if inputs and not all(b in arrays for b in inputs):
            continue                             # input absent in this sample
        try:
            val = np.asarray(spec.func(arrays))
        except Exception as e:                   # one bad var must not abort load
            print("  ! derived '%s' failed (%s) -- skipped for this sample"
                  % (name, e))
            continue
        if val.ndim != 1 or val.shape[0] != nrows:
            print("  ! derived '%s' produced shape %s (expected (%d,)) -- skipped"
                  % (name, val.shape, nrows))
            continue
        arrays[name] = (val.astype("float32")
                        if to_float32 and val.dtype == np.float64 else val)
    return arrays
