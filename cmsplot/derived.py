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
