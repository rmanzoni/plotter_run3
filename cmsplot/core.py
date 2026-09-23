"""Core engine: sample model, fast uproot reader, histogramming and plotting.

Design for speed:
 - each sample's files are read **once** with uproot into flat numpy arrays;
 - samples are read concurrently (threads: uproot releases the GIL while
   decompressing, so this scales without pickling overhead);
 - binning is decided from the in-memory arrays (no second read);
 - histograms are filled with two ``np.histogram`` calls per process/branch
   (sumw and sumw2), with over/underflow folded into the edge bins.
"""
from __future__ import annotations

import os
import re
import glob as _glob
import multiprocessing as _mp
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional

import numpy as np

# Force a non-interactive backend BEFORE `style` triggers the first pyplot
# import. This makes fork-based plotting workers safe everywhere (a GUI backend
# imported pre-fork can deadlock/crash children) and avoids backend probing on
# headless worker nodes.
import matplotlib
matplotlib.use("Agg")

from . import style, binning
from .derived import Derived, expand_inputs, compute_derived


# Per-branch horizontal canvas scale for the 1D plots. Default width is 8 in;
# a branch listed here gets its width multiplied by this factor (height kept).
# e.g. the 10-bin Z variable reads better stretched out.
BRANCH_WIDTH_SCALE = {
    "lhcb_3d_index": 8.0,
}

BRANCH_HEIGHT_SCALE = {
    "lhcb_3d_index": 2.0,
}


def _unit_from_label(lbl: str) -> str:
    """Pull the trailing ``[unit]`` out of an axis label.

    e.g. ``"$t$ [ps]" -> "ps"``, ``"$q^{2}$ [GeV$^{2}$]" -> "GeV$^{2}$"``;
    returns ``""`` when the label carries no bracketed unit.
    """
    m = re.findall(r"\[([^\[\]]+)\]", lbl or "")
    return m[-1].strip() if m else ""


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------
@dataclass
class WeightFactor:
    """One multiplicative per-event weight factor, with optional variations.

    inputs     : real branches the callables read (added to the column pruning).
    nominal    : callable(arrays) -> per-event factor (np.ndarray). Enters the
                 nominal weight, i.e. every plot, yield and datacard template.
    variations : {nuisance: (up, down)}, each a callable(arrays) -> per-event
                 factor that REPLACES ``nominal`` for that nuisance (all other
                 factors stay nominal). The nuisance name is the Combine name,
                 so two samples declaring the same nuisance are 100% correlated
                 in the datacard (e.g. bc and the misID Bc subtraction).

    The callables are the place to fail loud: they see the selected arrays and
    must return finite factors or raise -- ``_hist`` silently drops events with
    a non-finite weight, so a NaN leaking out of here would be an invisible
    event loss. ``_make_processes`` re-checks finiteness as a backstop.
    """
    inputs: tuple
    nominal: object
    variations: dict = field(default_factory=dict)


@dataclass
class Sample:
    """One physical sample. May fan out into several plotted processes.

    Parameters
    ----------
    name        : identifier used in filenames / yields.
    files       : list of paths or globs (local or ``root://`` xrootd).
    label       : legend label (defaults to ``name``).
    is_data     : drawn as points, never stacked, weight forced to 1.
    is_signal   : placed on top of the stack and hatched.
    color       : hex colour; auto-assigned from the Petroff palette if None.
    scale       : global normalisation (e.g. lumi * xsec / N_gen).
    weight_branches : per-event weight branches multiplied together.
    weight_factors  : {name: WeightFactor} computed per-event factors (e.g.
                  Hammer FF, Bc lifetime), multiplied into the nominal weight;
                  their ``variations`` become shape nuisances in the datacards.
    selection   : numpy-evaluable string on the branches (trusted input).
    split_by    : branch whose integer code splits the sample into processes
                  (e.g. ``gen_bc_decay`` for the Bc cocktail).
    split_map   : {code: (label, color, is_signal)}. Several codes that share the
                  same label are MERGED into one stacked process (this is how the
                  cocktail is grouped into physics components).
    split_default : (label, color, is_signal) for codes not in split_map (and for
                  NaN / non-finite codes). If None, each unmapped code becomes its
                  own auto-coloured process.
    group       : non-empty -> processes from different samples sharing this tag
                  are summed into a single stacked entry (e.g. hb1+hb2 -> "Hb").
    tree        : tree name inside the files.
    """
    name: str
    files: list
    label: str = ""
    is_data: bool = False
    is_signal: bool = False
    color: Optional[str] = None
    scale: float = 1.0
    weight_branches: list = field(default_factory=list)
    weight_factors: dict = field(default_factory=dict)
    selection: str = ""
    split_by: Optional[str] = None
    split_map: dict = field(default_factory=dict)
    split_default: Optional[tuple] = None
    group: str = ""
    tree: str = "tree"
    datacard: str = ""    # Combine datacard process tag: "Bc"/"Hb"/"misID" feed
                          # templates (reads sharing a tag are summed), "data_obs"
                          # is the observation, "" excludes (e.g. combinatorial).
    fakerate: Optional[tuple] = None
    # (pt_branch, edges, values): per-event fake rate FR(pt) looked up from a
    # binned table and folded into the weight. ``edges`` has len(values)+1 entries
    # (use np.inf for the open top bin). Used by the data-driven misID estimate:
    # a +scale data read and -scale MC reads share a `group` so the plotter sums
    # them into FR x (data_fail - MC_fail). FR procs are exempt from scale-to-data.

    def __post_init__(self):
        if not self.label:
            self.label = self.name


@dataclass
class Process:
    """A single histogram-able process (one entry in the stack/legend)."""
    key: str
    label: str
    color: Optional[str]
    is_data: bool
    is_signal: bool
    arrays: dict          # branch -> 1D numpy array
    weight: np.ndarray    # per-event weight, same length as arrays
    group: str = ""       # stack-merge tag across samples (e.g. "hb")
    is_fakerate: bool = False  # built from a fake-rate read (exempt scale-to-data)
    datacard: str = ""    # Combine datacard process tag (inherited from Sample)
    sample_name: str = ""      # originating Sample.name (datacard-composition key)
    split_codes: tuple = ()    # gen codes folded into this component (split samples)
    syst_weights: dict = field(default_factory=dict)
    # {nuisance: (w_up, w_down)} full per-event weights (same length as
    # ``weight``), built from Sample.weight_factors variations. Datacard-only:
    # plots always use the nominal ``weight``.


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _expand(files):
    out = []
    for f in files:
        if "://" in f:           # remote (xrootd) - cannot glob, pass through
            out.append(f)
            continue
        hits = _glob.glob(f)
        out.extend(hits if hits else [f])
    return out


def _to_dict(arr):
    """uproot may hand back a dict or a numpy structured array; normalise."""
    if isinstance(arr, dict):
        return arr
    if hasattr(arr, "dtype") and arr.dtype.names:
        return {n: arr[n] for n in arr.dtype.names}
    raise TypeError("unexpected uproot return type: %r" % type(arr))


def _names_in_expr(expr):
    """Branch names referenced in a selection string (for column pruning)."""
    import ast
    if not expr:
        return set()
    return {n.id for n in ast.walk(ast.parse(expr, mode="eval"))
            if isinstance(n, ast.Name)} - {"np"}


def _needed_columns(sample, needed_plot, derived=None):
    """Columns we must read for this sample, or None to read all numeric ones.

    ``needed_plot`` may contain derived-variable names; those are resolved to the
    real branches their funcs consume (so the ntuple is still pruned tightly).
    """
    if needed_plot is None:
        return None  # plotting every branch -> need them all
    # selection may itself reference derived variables -> read their real inputs
    want = expand_inputs(needed_plot, derived) \
        | expand_inputs(_names_in_expr(sample.selection), derived) \
        | set(sample.weight_branches)
    for wf in sample.weight_factors.values():
        want |= set(wf.inputs)
    if sample.split_by:
        want.add(sample.split_by)
    if sample.fakerate:
        want.add(sample.fakerate[0])   # pt branch for the FR(pt) lookup
    return want


def _numeric_branches(path, tree):
    import uproot
    with uproot.open(path) as fh:
        return [k for k, t in fh[tree].typenames().items()
                if any(s in t for s in
                       ("int", "float", "double", "bool", "short", "long"))]


def _read_file(path, tree, members, max_events=-1, step_size="150 MB",
               to_float32=True, derived=None):
    """ONE pass over one file, shared by every Sample that lists it.

    ``members`` is ``[(sample, want)]`` with ``want`` the sample's column set
    (None = all numeric). The file is decompressed once with the union of the
    members' columns; each chunk is then masked with each member's selection
    and every member keeps only its own surviving rows and its own columns.
    This is what makes e.g. ``bc`` (pass-iso) and ``misid_bc_sub`` (fail-iso),
    or ``data`` and ``misid_data``, cost one read of the ntuple instead of two.

    Returns ``{sample.name: (list_of_survivor_chunks, rows_read)}``. With
    ``max_events`` > 0 the file stops after that many rows (the per-sample cap
    across files is applied when the files are stitched, see ``_assemble``).
    """
    import uproot

    avail = _numeric_branches(path, tree)
    wants = [w for _, w in members]
    union = None if any(w is None for w in wants) else set().union(*wants)
    read_list = avail if union is None else [b for b in avail if b in union]
    out = {smp.name: ([], 0) for smp, _ in members}
    if not read_list:
        return out

    # derived variables used by ANY member's selection must exist before the
    # masks are evaluated: compute them once per chunk, for all members
    sel_names = {smp.name: _names_in_expr(smp.selection) for smp, _ in members}
    sel_derived = ({k: v for k, v in derived.items()
                    if any(k in n for n in sel_names.values())}
                   if derived else {})
    keep = {smp.name: (None if w is None
                       else set(w) | (sel_names[smp.name] & set(sel_derived)))
            for smp, w in members}

    survivors = {smp.name: [] for smp, _ in members}
    read_total = 0
    for chunk in uproot.iterate(path + ":" + tree, expressions=read_list,
                                library="np", step_size=step_size):
        chunk = {k: v for k, v in _to_dict(chunk).items()
                 if getattr(v, "ndim", 0) == 1 and v.dtype.kind in "biufc"}
        if not chunk:
            continue
        read_total += len(next(iter(chunk.values())))
        if sel_derived:
            compute_derived(chunk, sel_derived, to_float32=False)
        for smp, _ in members:
            mask = _selection_mask(chunk, smp.selection)
            if not mask.any():
                continue
            kp = keep[smp.name]
            survivors[smp.name].append({
                k: (v[mask].astype("float32")
                    if to_float32 and v.dtype == np.float64 else v[mask])
                for k, v in chunk.items() if kp is None or k in kp})
        del chunk
        if 0 < max_events <= read_total:
            break
    return {name: (chunks, read_total) for name, chunks in survivors.items()}


def _assemble(sample, per_file, max_events=-1, to_float32=True, derived=None):
    """Stitch one sample's survivors across its files (in ``sample.files``
    order, honouring ``max_events`` as a cap on rows READ), then add the
    remaining derived columns. Chunk lists are consumed as they are merged so
    the per-chunk copies and the merged arrays never coexist in full."""
    chunks = []
    read_total = 0
    for f in _expand(sample.files):
        c, n = per_file.get(f, ([], 0))
        chunks.extend(c)
        read_total += n
        if 0 < max_events <= read_total:
            break
    if not chunks:
        return {}
    out = {}
    for k in list(chunks[0]):
        out[k] = np.concatenate([c.pop(k) for c in chunks])
    del chunks
    return compute_derived(out, derived, to_float32=to_float32)


def _read_sample(sample, needed_plot=None, max_events=-1,
                 step_size="150 MB", to_float32=True, derived=None):
    """Single-sample reader (kept for API compatibility; Histogrammer.load
    uses the shared per-file pass directly)."""
    want = _needed_columns(sample, needed_plot, derived)
    per_file = {f: _read_file(f, sample.tree, [(sample, want)], max_events,
                              step_size, to_float32, derived)[sample.name]
                for f in _expand(sample.files)}
    return _assemble(sample, per_file, max_events, to_float32, derived)


def _selection_mask(arrays, expr):
    if not expr:
        n = len(next(iter(arrays.values())))
        return np.ones(n, dtype=bool)
    ns = dict(arrays)
    ns["np"] = np
    try:
        return np.asarray(eval(expr, {"__builtins__": {}}, ns), dtype=bool)  # trusted
    except NameError as e:
        # pinpoint which referenced names are not branches in this sample
        import ast
        used = {n.id for n in ast.walk(ast.parse(expr, mode="eval"))
                if isinstance(n, ast.Name)}
        missing = sorted(u for u in used if u != "np" and u not in arrays)
        raise NameError(
            "selection references name(s) not found as branches: %s\n"
            "  (use np.<func> for functions, e.g. np.abs; available branches "
            "start with: %s ...)"
            % (missing, ", ".join(sorted(arrays)[:12]))) from e


def _fakerate_lookup(pt, edges, values):
    """Per-event FR(pt) from a binned table. ``edges`` has len(values)+1 entries
    (np.inf allowed for the open top bin). Non-finite pt -> FR 0 (event dropped)."""
    if pt is None:
        raise KeyError("fakerate pt branch not present in the read arrays")
    pt = np.asarray(pt, "float64")
    edges = np.asarray(edges, "float64")
    values = np.asarray(values, "float64")
    if edges.size != values.size + 1:
        raise ValueError("fakerate edges must have len(values)+1 entries "
                         "(%d edges vs %d values)" % (edges.size, values.size))
    idx = np.clip(np.digitize(pt, edges) - 1, 0, values.size - 1)
    fr = values[idx]
    fr[~np.isfinite(pt)] = 0.0
    return fr


def _apply_weight_factors(sample, arrays, base, n):
    """Fold Sample.weight_factors into ``base``; build the variation weights.

    Returns ``(nominal, {nuisance: (w_up, w_down)})``. A variation weight is
    ``base`` times every factor at its nominal value except the varied one,
    multiplied in the SAME order as the nominal, so a variation that equals the
    nominal factor on some events reproduces the nominal weight bit for bit
    there (the datacard writer relies on this to skip no-effect templates).
    """
    if not sample.weight_factors:
        return base, {}
    if sample.is_data:
        raise ValueError("sample %r: weight_factors on a data sample"
                         % sample.name)
    missing = sorted({b for wf in sample.weight_factors.values()
                      for b in wf.inputs if b not in arrays})
    if missing:
        raise KeyError("sample %r: weight_factors need branch(es) %s that are "
                       "not in the ntuple %s" % (sample.name, missing,
                                                 sample.files))

    def _eval(fn, what):
        v = np.asarray(fn(arrays), dtype="float64")
        if v.shape != (n,):
            raise ValueError("sample %r: %s returned shape %s, expected (%d,)"
                             % (sample.name, what, v.shape, n))
        bad = ~np.isfinite(v)
        if bad.any():
            raise ValueError("sample %r: %s is non-finite for %d/%d events"
                             % (sample.name, what, int(bad.sum()), n))
        return v

    names = list(sample.weight_factors)
    nom = {k: _eval(sample.weight_factors[k].nominal, "factor %r" % k)
           for k in names}

    def _product(swap_key=None, swap_val=None):
        w = np.asarray(base, dtype="float64")
        for k in names:
            w = w * (swap_val if k == swap_key else nom[k])
        return w

    weight = _product()
    syst = {}
    for k in names:
        for nuis, (fup, fdn) in sample.weight_factors[k].variations.items():
            if nuis in syst:
                raise ValueError("sample %r: nuisance %r declared by two "
                                 "weight factors" % (sample.name, nuis))
            syst[nuis] = (
                _product(k, _eval(fup, "factor %r, %sUp" % (k, nuis))),
                _product(k, _eval(fdn, "factor %r, %sDown" % (k, nuis))))
    return weight, syst


def _make_processes(sample: Sample, arrays: dict, fallback_color=None):
    """Turn (already selection-filtered) arrays into Process objects (weight/split)."""
    if not arrays:
        return []

    n = len(next(iter(arrays.values())))  # selection was applied during reading

    if sample.is_data:
        weight = np.ones(n)
    else:
        weight = np.full(n, float(sample.scale))
        for wb in sample.weight_branches:
            if wb in arrays:
                weight = weight * arrays[wb]

    # fold a pt-binned fake rate into the weight (applies before any split, so
    # split slices inherit it). data-driven misID uses this on data and MC reads.
    if sample.fakerate is not None:
        ptb, fr_edges, fr_vals = sample.fakerate
        weight = weight * _fakerate_lookup(arrays.get(ptb), fr_edges, fr_vals)

    weight, syst = _apply_weight_factors(sample, arrays, weight, n)

    if not sample.split_by or sample.split_by not in arrays:
        return [Process(sample.name, sample.label, sample.color or fallback_color,
                        sample.is_data, sample.is_signal, arrays, weight,
                        group=sample.group, is_fakerate=bool(sample.fakerate),
                        datacard=sample.datacard, sample_name=sample.name,
                        syst_weights=syst)]

    # integer codes, with non-finite (NaN) routed to the default component
    raw = arrays[sample.split_by].astype("float64")
    finite = np.isfinite(raw)
    codes = np.full(raw.shape, -999, dtype="int64")
    codes[finite] = np.round(raw[finite]).astype("int64")

    from collections import OrderedDict
    comps = OrderedDict()  # component label -> [color, is_signal, mask, code list]
    for code in np.unique(codes):
        entry = sample.split_map.get(int(code), sample.split_default)
        if entry is None:                       # no default -> one process per code
            entry = ("%s_%d" % (sample.name, int(code)), None, False)
        lab, col, is_sig = entry
        m = codes == code
        if lab in comps:
            comps[lab][2] |= m
            comps[lab][3].append(int(code))
        else:
            comps[lab] = [col, bool(is_sig), m, [int(code)]]

    procs = []
    for lab, (col, is_sig, m, codelist) in comps.items():
        procs.append(Process(
            "%s::%s" % (sample.name, lab), lab, col,
            sample.is_data, is_sig,
            {k: v[m] for k, v in arrays.items()}, weight[m],
            group=sample.group, is_fakerate=bool(sample.fakerate),
            datacard=sample.datacard, sample_name=sample.name,
            split_codes=tuple(sorted(codelist)),
            syst_weights={k: (u[m], d[m]) for k, (u, d) in syst.items()}))
    return procs


# ---------------------------------------------------------------------------
# Histogrammer
# ---------------------------------------------------------------------------
def _hist(x, w, edges, overflow=False):
    x = np.asarray(x, "float64")
    w = np.asarray(w, "float64")
    m = np.isfinite(x) & np.isfinite(w)
    x, w = x[m], w[m]
    if overflow and x.size:
        x = np.clip(x, edges[0], edges[-1] - 1e-12 * (edges[-1] - edges[0]))
    # when overflow is False, np.histogram simply ignores out-of-range entries
    sw, _ = np.histogram(x, bins=edges, weights=w)
    sw2, _ = np.histogram(x, bins=edges, weights=w * w)
    return sw, sw2


class Histogrammer:
    """Loads samples (concurrently) and yields the list of Process objects."""

    def __init__(self, samples, jobs=4, max_events=-1, max_range_events=300_000,
                 binning_overrides=None, needed_plot=None, step_size="150 MB",
                 to_float32=True, derived=None):
        self.samples = list(samples)
        self.jobs = jobs
        self.max_events = max_events
        self.max_range_events = max_range_events
        self.binning = dict(binning_overrides or {})
        self.needed_plot = needed_plot          # set of branches to plot, or None
        self.step_size = step_size
        self.to_float32 = to_float32
        self.derived = derived or {}
        self.processes: list[Process] = []

    def _needed_set(self):
        if self.needed_plot is None:
            return None
        # branches to plot + those used by any sample's binning override keys
        return set(self.needed_plot)

    def load(self):
        need = self._needed_set()
        names = [s.name for s in self.samples]
        if len(set(names)) != len(names):
            raise ValueError("duplicate Sample names: %s"
                             % sorted({n for n in names if names.count(n) > 1}))

        # Invert sample -> files into (file, tree) -> samples, so every ntuple
        # is decompressed ONCE however many Samples read it (bc + misid_bc_sub,
        # hb + misid_hb_sub, data + misid_data, ...). Threads run over files.
        from collections import OrderedDict
        by_file = OrderedDict()
        for s in self.samples:
            want = _needed_columns(s, need, self.derived)
            files = _expand(s.files)
            dup = sorted({f for f in files if files.count(f) > 1})
            if dup:
                raise ValueError("sample %r lists file(s) more than once: %s"
                                 % (s.name, dup))
            for f in files:
                by_file.setdefault((f, s.tree), []).append((s, want))

        per_sample = {s.name: {} for s in self.samples}
        with ThreadPoolExecutor(max_workers=self.jobs) as ex:
            futs = {ex.submit(_read_file, f, tree, members, self.max_events,
                              self.step_size, self.to_float32, self.derived):
                    (f, tree) for (f, tree), members in by_file.items()}
            for fut, (f, tree) in futs.items():
                for name, res in fut.result().items():
                    per_sample[name][f] = res

        # assign fallback colours to non-data, non-split samples lacking one
        procs = []
        for s in self.samples:
            # assemble one sample at a time and drop its raw arrays as soon as
            # its Processes exist (split samples copy per component), so at
            # most one sample is ever held twice
            arrays = _assemble(s, per_sample.pop(s.name), self.max_events,
                               self.to_float32, self.derived)
            procs += _make_processes(s, arrays)
            del arrays
        # colour any process still missing a colour (one colour per group/key)
        need, seen = [], set()
        for p in procs:
            gkey = p.group or p.key
            if p.color is None and not p.is_data and gkey not in seen:
                seen.add(gkey)
                need.append(gkey)
        cmap = dict(zip(need, style.palette(max(len(need), 1))))
        for p in procs:
            if p.color is None and not p.is_data:
                p.color = cmap[p.group or p.key]
        self.processes = procs
        return procs

    def branches(self):
        """Union of branch names across all (non-data informs gen branches too)."""
        names = set()
        for p in self.processes:
            names |= set(p.arrays)
        return sorted(names)

    def edges_for(self, branch, nbins=40):
        # manual override: tuple (nbins, lo, hi) -> uniform; list/array -> edges
        spec = self.binning.get(branch)
        if spec is not None:
            if isinstance(spec, tuple):
                n, lo, hi = spec
                return np.linspace(float(lo), float(hi), int(n) + 1)
            return np.asarray(spec, dtype="float64")
        # Derive the bin range from the SIGNAL-region processes only. Data-driven
        # control-region estimates (fake-rate misID from the fail-iso region, and
        # any `data_driven` background such as the J/psi-sideband combinatorial)
        # have different distributions; letting them drive the auto-binning makes
        # the bins depend on whether misID is enabled -- which silently desyncs
        # plots from datacards. Exclude them here (they are still histogrammed
        # with the chosen edges).
        def _drives_binning(p):
            return not (p.is_fakerate or getattr(p, "data_driven", False))
        vals = []
        for p in self.processes:
            if branch in p.arrays and _drives_binning(p):
                a = p.arrays[branch]
                if a.size > self.max_range_events:
                    a = a[:: max(1, a.size // self.max_range_events)]
                vals.append(a)
        if not vals:  # fall back to all processes if nothing else has the branch
            for p in self.processes:
                if branch in p.arrays:
                    a = p.arrays[branch]
                    if a.size > self.max_range_events:
                        a = a[:: max(1, a.size // self.max_range_events)]
                    vals.append(a)
        if not vals:
            return np.linspace(0, 1, nbins + 1)
        return binning.guess_binning(np.concatenate(vals), branch, nbins=nbins)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
class StackPlotter:
    def __init__(self, lumi=None, com=13.6, extra="Preliminary", normalize=False,
                 overflow=False):
        self.lumi = lumi
        self.com = com
        self.extra = extra
        self.normalize = normalize
        self.overflow = overflow
        style.set_cms_style()

    @staticmethod
    def _band(ax, edges, lo, hi, **kw):
        ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]],
                        step="post", **kw)

    def draw(self, branch, edges, processes, outdir, label, nbins=40):
        import matplotlib.pyplot as plt

        present = [p for p in processes if branch in p.arrays]
        mc = [p for p in present if not p.is_data]
        data = [p for p in present if p.is_data]
        if not mc:
            return

        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        # variable-width binning -> plot a *density* (bin content / bin width) so
        # the shape is not distorted by the wider bins; uniform binning unchanged.
        variable_bins = bool(widths.size) and not np.allclose(widths, widths[0])
        dens = (1.0 / widths) if variable_bins else np.ones_like(widths)

        sw, sw2 = {}, {}
        for p in present:
            sw[p.key], sw2[p.key] = _hist(p.arrays[branch], p.weight, edges,
                                          overflow=self.overflow)

        # merge MC processes that share a `group` tag into single stack entries
        from collections import OrderedDict
        entries = OrderedDict()
        for p in mc:
            gkey = p.group or p.key
            if gkey in entries:
                entries[gkey]["sw"] += sw[p.key]
                entries[gkey]["sw2"] += sw2[p.key]
            else:
                entries[gkey] = dict(sw=sw[p.key].copy(), sw2=sw2[p.key].copy(),
                                     label=p.label, color=p.color,
                                     is_signal=p.is_signal)
        stack = list(entries.values())
        stack.sort(key=lambda e: e["is_signal"])   # backgrounds first, signal on top

        tot = np.sum([e["sw"] for e in stack], axis=0)
        tot_e = np.sqrt(np.sum([e["sw2"] for e in stack], axis=0))
        data_sw = np.sum([sw[p.key] for p in data], axis=0) if data else None

        norm = 1.0
        if self.normalize and tot.sum() > 0:
            norm = 1.0 / tot.sum()

        have_data = data and data_sw.sum() > 0
        wscale = BRANCH_WIDTH_SCALE.get(branch, 1.0)
        hscale = BRANCH_HEIGHT_SCALE.get(branch, 1.0)
        if have_data:
            fig, (ax, rax) = plt.subplots(
                2, 1, figsize=(8 * wscale, 8 * hscale), sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07})
        else:
            fig, ax = plt.subplots(figsize=(8 * wscale, 7) * hscale)
            rax = None

        # --- stacked MC ---
        bottom = np.zeros_like(tot)
        for e in stack:
            y = e["sw"] * norm * dens
            ax.bar(centers, y, width=widths, bottom=bottom, color=e["color"],
#                    label=e["label"], align="center", linewidth=0.4,
#                    edgecolor="black",
                   label=e["label"], align="center", linewidth=0.4,
                   edgecolor=e["color"],
                   hatch="///" if e["is_signal"] else None)
            bottom = bottom + y

        # --- MC stat band ---
        lo = (tot - tot_e) * norm * dens
        hi = (tot + tot_e) * norm * dens
        self._band(ax, edges, lo, hi, facecolor="none", edgecolor="gray",
                   hatch="xxxxx", linewidth=0.0, label="MC stat. unc.")

        # --- data ---
        if have_data:
            yd = data_sw * norm * dens
            yderr = np.sqrt(data_sw) * norm * dens
            m = data_sw > 0
            ax.errorbar(centers[m], yd[m], yerr=yderr[m], fmt="o", color="black",
                        markersize=4, label="Data", zorder=5)

        if self.normalize:
            ax.set_ylabel("a.u.")
        elif variable_bins:
            _u = _unit_from_label(binning.axis_label(branch))
            ax.set_ylabel(("Events / %s" % _u) if _u else "Events / bin width")
        else:
            ax.set_ylabel("Events")
        ax.set_xlim(edges[0], edges[-1])
        ncol = 2 if len(stack) <= 8 else 3
        leg = ax.legend(ncol=ncol, fontsize="x-small", loc="upper right")
        style.cms_label(ax, lumi=self.lumi, com=self.com,
                        data=have_data, extra=self.extra)

        # --- ratio panel ---
        if rax is not None:
            m = (tot > 0) & (data_sw > 0)
            ratio = np.full_like(tot, np.nan)
            rerr = np.full_like(tot, np.nan)
            ratio[m] = data_sw[m] / tot[m]
            rerr[m] = np.sqrt(data_sw[m]) / tot[m]
            band_lo = np.where(tot > 0, 1 - tot_e / np.where(tot > 0, tot, 1), 1)
            band_hi = np.where(tot > 0, 1 + tot_e / np.where(tot > 0, tot, 1), 1)
            self._band(rax, edges, band_lo, band_hi, facecolor="none",
                       edgecolor="gray", hatch="xxxxx", linewidth=0.0)
            rax.errorbar(centers[m], ratio[m], yerr=rerr[m], fmt="o",
                         color="black", markersize=4)
            rax.axhline(1.0, color="black", lw=1)
            rax.set_ylim(0.8, 1.2)
            rax.set_ylabel("Data / MC", fontsize="small")
            rax.set_xlabel(binning.axis_label(branch))
        else:
            ax.set_xlabel(binning.axis_label(branch))

        # Headroom: size the y-axis so the tallest stack/data point clears the
        # legend instead of being hidden behind it. We measure the legend's
        # lower edge (in axes fraction) after a draw and put the peak just below
        # it; if the measurement is unavailable we fall back to an estimate from
        # the number of legend rows. Both linear and log use the same target so
        # the gap looks consistent.
        ymax = max(bottom.max(),
                   ((data_sw * norm * dens).max()) if have_data else 0.0)
        n_handles = len(stack) + 1 + (1 if have_data else 0)   # +1 MC-stat band
        nrows = int(np.ceil(n_handles / max(1, ncol)))
        target = self._headroom_target(fig, ax, leg, nrows)

        ax.set_ylim(0.0, (ymax / target) if ymax > 0 else 1.0)
        self._save(fig, ax, outdir, label, branch, logy=False)
        # log version: keep the same fractional clearance below the legend
        ax.set_yscale("log")
        bot = (0.3 * norm) if self.normalize else 0.3
        peak = ymax if ymax > 0 else 1.0
        top = (bot * (peak / bot) ** (1.0 / target)) if peak > bot else bot * 50
        ax.set_ylim(bot, top)
        self._save(fig, ax, outdir, label, branch, logy=True)
        plt.close(fig)

    @staticmethod
    def _headroom_target(fig, ax, leg, nrows):
        """Fraction of the axes height the data should be allowed to fill.

        Measures the legend's lower edge in axes-fraction coordinates (so the
        peak sits a small margin below it). The legend is anchored to the axes,
        not the data, so this fraction does not depend on the y-limits we are
        about to set. Falls back to a row-count estimate if the draw/extent is
        unavailable, and is clamped so there is always some headroom and we
        never demand an absurdly tall axis.
        """
        frac = None
        try:
            fig.canvas.draw()
            bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
            if np.isfinite(bb.y0) and 0.0 < bb.y0 < 1.0:
                frac = float(bb.y0)
        except Exception:
            frac = None
        if frac is None:                       # estimate from legend rows
            frac = 0.96 - 0.07 * nrows
        return min(0.92, max(0.45, frac - 0.06))

    @staticmethod
    def _save(fig, ax, outdir, label, branch, logy):
        sub = "log" if logy else "lin"
        for ext in ("png", "pdf"):
            d = os.path.join(outdir, label, ext, sub)
            os.makedirs(d, exist_ok=True)
            fig.savefig(os.path.join(d, "%s.%s" % (branch, ext)),
                        bbox_inches="tight", dpi=150 if ext == "png" else None)


# ---------------------------------------------------------------------------
# Parallel plotting (fork + copy-on-write)
# ---------------------------------------------------------------------------
# matplotlib's pyplot state is process-global and not thread-safe, and the
# artist/layout code holds the GIL, so the per-branch draw loop does not scale
# with threads -- it needs separate processes. We avoid pickling the (large)
# in-memory arrays to every worker by using a *fork* pool: the parent stashes
# the already-loaded Process list and a StackPlotter in a module global before
# forking, and each worker inherits them copy-on-write (plotting only reads
# them, so the pages are never actually copied). Tasks then carry only the
# branch name + its precomputed bin edges, which are tiny.
_PLOT_STATE: dict = {}


def _draw_one(task):
    branch, edges, outdir, label, nbins = task
    procs = _PLOT_STATE["procs"]
    plotter = _PLOT_STATE["plotter"]
    try:
        plotter.draw(branch, edges, procs, outdir, label, nbins=nbins)
        return (branch, None)
    except Exception as e:                 # never let one bad branch kill the run
        return (branch, "%s" % e)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(samples, outdir="plots", label=None, branches=None, exclude=(),
        nbins=40, jobs=4, lumi=None, com=13.6, extra="Preliminary",
        normalize=False, overflow=False, scale_to_data=False,
        binning_overrides=None, max_events=-1, step_size="150 MB",
        to_float32=True, verbose=True,
        datacard_branches=None, datacard_signal="Bc", datacard_def=None,
        axis_titles=None, derived=None):
    """Load samples and produce one stacked plot per branch.

    ``derived`` is an optional ``{name: Derived(func, inputs)}`` mapping of new
    columns to compute from existing branches (see cmsplot.derived). Derived
    names may be used in ``branches`` and ``datacard_branches`` like real ones.
    """
    from datetime import datetime

    # merge any config-supplied exact branch->axis-title overrides (issue 4)
    binning.set_user_titles(axis_titles)

    if label is None:
        label = datetime.now().strftime("%d%b%Y_%Hh%Mm%Ss")

    # If a branch list is given, read ONLY those columns (plus selection/weight/
    # split columns, and any datacard-only branches) -- this is what keeps memory
    # bounded on large samples. Derived names here are resolved to their real
    # input branches by the reader.
    needed_plot = set(branches) if branches else None
    if needed_plot is not None and datacard_branches:
        needed_plot |= set(datacard_branches)

    hg = Histogrammer(samples, jobs=jobs, max_events=max_events,
                      binning_overrides=binning_overrides,
                      needed_plot=needed_plot, step_size=step_size,
                      to_float32=to_float32, derived=derived)
    if verbose:
        print("[cmsplot] reading %d samples (%s) ..."
              % (len(list(samples)),
                 "%d columns" % len(needed_plot) if needed_plot else "all columns"))
    procs = hg.load()

    # datacards must use the NOMINAL normalisation (so the fit can determine the
    # process yields), so build them BEFORE scale-to-data mutates the weights.
    if datacard_branches:
        from . import datacard as _dc
        dc_todo = [b for b in datacard_branches if b not in set(exclude)]
        if verbose:
            print("[cmsplot] writing %d datacard(s) -> %s/%s/datacards"
                  % (len(dc_todo), outdir, label))
        try:
            _dc.write_datacards(procs, hg, dc_todo, outdir, label, nbins=nbins,
                                signal=datacard_signal, overflow=overflow,
                                verbose=verbose, datacard_def=datacard_def)
        except Exception as e:
            print("  ! datacard generation failed: %s" % e)

    # fix the total MC normalisation to the data yield (keeps the Bc:Hb ratio,
    # which is set by the per-sample `scale`s, untouched)
    if scale_to_data:
        # genuine (simulated) MC only; data-driven fake-rate procs are a
        # data-derived background and must NOT be rescaled -- so the genuine MC
        # is scaled to (data - fakes), keeping the fake estimate fixed.
        genuine = [p for p in procs if not p.is_data and not p.is_fakerate]
        mc_tot = sum(float(np.sum(p.weight)) for p in genuine)
        fake_tot = sum(float(np.sum(p.weight))
                       for p in procs if not p.is_data and p.is_fakerate)
        dat_tot = sum(float(np.sum(p.weight)) for p in procs if p.is_data)
        target = dat_tot - fake_tot
        if mc_tot > 0 and target > 0:
            f = target / mc_tot
            for p in genuine:
                p.weight = p.weight * f
            if verbose:
                print("[cmsplot] scale-to-data: genuine MC x %.4g "
                      "(MC %.1f -> data %.0f - fakes %.1f)"
                      % (f, mc_tot, dat_tot, fake_tot))
        elif verbose:
            print("[cmsplot] scale-to-data requested but MC or "
                  "(data - fakes) yield is <= 0")

    if verbose:
        for p in procs:
            print("  %-22s %10d events  w.sum=%.3g"
                  % (p.label, p.weight.size, float(np.sum(p.weight))))

    todo = branches if branches else hg.branches()
    todo = [b for b in todo if b not in set(exclude)]
    if verbose:
        print("[cmsplot] plotting %d branches -> %s/%s"
              % (len(todo), outdir, label))

    plotter = StackPlotter(lumi=lumi, com=com, extra=extra, normalize=normalize,
                           overflow=overflow)

    # Bin edges depend on the loaded arrays (held in the parent), so compute them
    # here once; workers then only need (branch, edges).
    edges_map = {b: hg.edges_for(b, nbins=nbins) for b in todo}

    if jobs > 1 and len(todo) > 1 and "fork" in _mp.get_all_start_methods():
        # fork pool: workers inherit `procs`/`plotter` copy-on-write (no array
        # pickling). matplotlib state is per-process, so draws never collide.
        _PLOT_STATE["procs"] = procs
        _PLOT_STATE["plotter"] = plotter
        tasks = [(b, edges_map[b], outdir, label, nbins) for b in todo]
        try:
            ctx = _mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
                for i, (b, err) in enumerate(ex.map(_draw_one, tasks), 1):
                    if err:
                        print("  ! skipping %s (%s)" % (b, err))
                    if verbose and i % 20 == 0:
                        print("    ... %d/%d" % (i, len(todo)))
        finally:
            _PLOT_STATE.clear()
    else:
        for i, b in enumerate(todo, 1):
            try:
                plotter.draw(b, edges_map[b], procs, outdir, label, nbins=nbins)
            except Exception as e:  # never let one bad branch kill the run
                print("  ! skipping %s (%s)" % (b, e))
            if verbose and i % 20 == 0:
                print("    ... %d/%d" % (i, len(todo)))

    _write_yields(procs, outdir, label)
    _write_selection(samples, outdir, label,
                     scale_to_data=scale_to_data, overflow=overflow)
    if verbose:
        print("[cmsplot] done.")
    return os.path.join(outdir, label)


def _write_yields(procs, outdir, label):
    from collections import OrderedDict
    os.makedirs(os.path.join(outdir, label), exist_ok=True)
    path = os.path.join(outdir, label, "yields.txt")
    mc, data = OrderedDict(), OrderedDict()
    for p in procs:
        bucket = data if p.is_data else mc
        key = p.group or p.label
        bucket[key] = bucket.get(key, [p.label, 0.0])
        bucket[key][1] += float(np.sum(p.weight))
    with open(path, "w") as f:
        tot = 0.0
        for lab, y in mc.values():
            tot += y
            print("%-30s %12.2f" % (lab, y), file=f)
        print("%-30s %12.2f" % ("total expected", tot), file=f)
        for lab, y in data.values():
            print("%-30s %12.0f" % (lab, y), file=f)


def _write_selection(samples, outdir, label, scale_to_data=False, overflow=False):
    os.makedirs(os.path.join(outdir, label), exist_ok=True)
    path = os.path.join(outdir, label, "selection.txt")
    with open(path, "w") as f:
        print("# cmsplot run options", file=f)
        print("scale_to_data = %s" % scale_to_data, file=f)
        print("overflow      = %s" % overflow, file=f)
        print("", file=f)
        print("# per-sample selection and normalisation", file=f)
        for s in samples:
            print("[%s]%s" % (s.name, "  (data)" if s.is_data else ""), file=f)
            print("  files     = %s" % (s.files,), file=f)
            if not s.is_data:
                print("  scale     = %g" % s.scale, file=f)
                if s.weight_branches:
                    print("  weights   = %s" % " * ".join(s.weight_branches), file=f)
                if s.weight_factors:
                    print("  factors   = %s" % " * ".join(s.weight_factors), file=f)
                    nuis = [nm for wf in s.weight_factors.values()
                            for nm in wf.variations]
                    if nuis:
                        print("  shape nuisances = %s" % " ".join(nuis), file=f)
                if s.group:
                    print("  group     = %s" % s.group, file=f)
            print("  selection = %s" % (s.selection if s.selection else "(none)"),
                  file=f)
            print("", file=f)
