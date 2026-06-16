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
import glob as _glob
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from . import style, binning


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------
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
    selection: str = ""
    split_by: Optional[str] = None
    split_map: dict = field(default_factory=dict)
    split_default: Optional[tuple] = None
    group: str = ""
    tree: str = "tree"

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


def _needed_columns(sample, needed_plot):
    """Columns we must read for this sample, or None to read all numeric ones."""
    if needed_plot is None:
        return None  # plotting every branch -> need them all
    want = set(needed_plot) | _names_in_expr(sample.selection) \
        | set(sample.weight_branches)
    if sample.split_by:
        want.add(sample.split_by)
    return want


def _read_sample(sample, needed_plot=None, max_events=-1,
                 step_size="150 MB", to_float32=True):
    """Stream a sample's files, keeping only needed columns and selected rows.

    Reads in chunks (uproot.iterate), applies the per-sample selection to each
    chunk and retains only surviving rows, so peak memory is ~one chunk plus the
    (usually small) selected subset -- not the whole input.
    """
    import uproot

    files = _expand(sample.files)
    if not files:
        return {}

    # available numeric branches (from the first file), then prune to what we need
    with uproot.open(files[0]) as fh:
        tree = fh[sample.tree]
        avail = [k for k, t in tree.typenames().items()
                 if any(s in t for s in
                        ("int", "float", "double", "bool", "short", "long"))]
    want = _needed_columns(sample, needed_plot)
    read_list = avail if want is None else [b for b in avail if b in want]
    if not read_list:
        return {}

    paths = [f + ":" + sample.tree for f in files]
    survivors, read_total = [], 0
    for chunk in uproot.iterate(paths, expressions=read_list, library="np",
                                step_size=step_size):
        chunk = {k: v for k, v in _to_dict(chunk).items()
                 if getattr(v, "ndim", 0) == 1 and v.dtype.kind in "biufc"}
        if not chunk:
            continue
        read_total += len(next(iter(chunk.values())))
        mask = _selection_mask(chunk, sample.selection)
        if mask.any():
            survivors.append({
                k: (v[mask].astype("float32")
                    if to_float32 and v.dtype == np.float64 else v[mask])
                for k, v in chunk.items()})
        if 0 < max_events <= read_total:
            break

    if not survivors:
        return {}
    keys = set(survivors[0])
    return {k: np.concatenate([s[k] for s in survivors]) for k in keys}


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

    if not sample.split_by or sample.split_by not in arrays:
        return [Process(sample.name, sample.label, sample.color or fallback_color,
                        sample.is_data, sample.is_signal, arrays, weight,
                        group=sample.group)]

    # integer codes, with non-finite (NaN) routed to the default component
    raw = arrays[sample.split_by].astype("float64")
    finite = np.isfinite(raw)
    codes = np.full(raw.shape, -999, dtype="int64")
    codes[finite] = np.round(raw[finite]).astype("int64")

    from collections import OrderedDict
    comps = OrderedDict()  # component label -> [color, is_signal, combined mask]
    for code in np.unique(codes):
        entry = sample.split_map.get(int(code), sample.split_default)
        if entry is None:                       # no default -> one process per code
            entry = ("%s_%d" % (sample.name, int(code)), None, False)
        lab, col, is_sig = entry
        m = codes == code
        if lab in comps:
            comps[lab][2] |= m
        else:
            comps[lab] = [col, bool(is_sig), m]

    procs = []
    for lab, (col, is_sig, m) in comps.items():
        procs.append(Process(
            "%s::%s" % (sample.name, lab), lab, col,
            sample.is_data, is_sig,
            {k: v[m] for k, v in arrays.items()}, weight[m],
            group=sample.group))
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
                 to_float32=True):
        self.samples = list(samples)
        self.jobs = jobs
        self.max_events = max_events
        self.max_range_events = max_range_events
        self.binning = dict(binning_overrides or {})
        self.needed_plot = needed_plot          # set of branches to plot, or None
        self.step_size = step_size
        self.to_float32 = to_float32
        self.processes: list[Process] = []

    def _needed_set(self):
        if self.needed_plot is None:
            return None
        # branches to plot + those used by any sample's binning override keys
        return set(self.needed_plot)

    def load(self):
        raw = {}
        need = self._needed_set()
        with ThreadPoolExecutor(max_workers=self.jobs) as ex:
            futs = {ex.submit(_read_sample, s, need, self.max_events,
                              self.step_size, self.to_float32): s
                    for s in self.samples}
            for fut, s in futs.items():
                raw[s.name] = fut.result()

        # assign fallback colours to non-data, non-split samples lacking one
        procs = []
        for s in self.samples:
            procs += _make_processes(s, raw.get(s.name, {}))
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
        vals = []
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
        if have_data:
            fig, (ax, rax) = plt.subplots(
                2, 1, figsize=(8, 8), sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07})
        else:
            fig, ax = plt.subplots(figsize=(8, 7))
            rax = None

        # --- stacked MC ---
        bottom = np.zeros_like(tot)
        for e in stack:
            y = e["sw"] * norm
            ax.bar(centers, y, width=widths, bottom=bottom, color=e["color"],
#                    label=e["label"], align="center", linewidth=0.4,
#                    edgecolor="black",
                   label=e["label"], align="center", linewidth=0.4,
                   edgecolor=e["color"],
                   hatch="///" if e["is_signal"] else None)
            bottom = bottom + y

        # --- MC stat band ---
        lo = (tot - tot_e) * norm
        hi = (tot + tot_e) * norm
        self._band(ax, edges, lo, hi, facecolor="none", edgecolor="gray",
                   hatch="xxxxx", linewidth=0.0, label="MC stat. unc.")

        # --- data ---
        if have_data:
            yd = data_sw * norm
            yderr = np.sqrt(data_sw) * norm
            m = data_sw > 0
            ax.errorbar(centers[m], yd[m], yerr=yderr[m], fmt="o", color="black",
                        markersize=4, label="Data", zorder=5)

        ax.set_ylabel("a.u." if self.normalize else "Events")
        ax.set_xlim(edges[0], edges[-1])
        ax.legend(ncol=2 if len(stack) <= 8 else 3, fontsize="x-small",
                  loc="upper right")
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
            rax.set_ylim(0.0, 2.0)
            rax.set_ylabel("Data / MC", fontsize="small")
            rax.set_xlabel(binning.axis_label(branch))
        else:
            ax.set_xlabel(binning.axis_label(branch))

        ymax = max(bottom.max(), (data_sw.max() * norm) if have_data else 0.0)
        ax.set_ylim(0.0, 1.55 * ymax if ymax > 0 else 1.0)  # headroom for legend
        self._save(fig, ax, outdir, label, branch, logy=False)
        # log version
        ax.set_yscale("log")
        ax.set_ylim(0.3 * norm if self.normalize else 0.3, ymax * 50 + 1)
        self._save(fig, ax, outdir, label, branch, logy=True)
        plt.close(fig)

    @staticmethod
    def _save(fig, ax, outdir, label, branch, logy):
        sub = "log" if logy else "lin"
        for ext in ("png", "pdf"):
            d = os.path.join(outdir, label, ext, sub)
            os.makedirs(d, exist_ok=True)
            fig.savefig(os.path.join(d, "%s.%s" % (branch, ext)),
                        bbox_inches="tight", dpi=150 if ext == "png" else None)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(samples, outdir="plots", label=None, branches=None, exclude=(),
        nbins=40, jobs=4, lumi=None, com=13.6, extra="Preliminary",
        normalize=False, overflow=False, scale_to_data=False,
        binning_overrides=None, max_events=-1, step_size="150 MB",
        to_float32=True, verbose=True):
    """Load samples and produce one stacked plot per branch."""
    from datetime import datetime

    if label is None:
        label = datetime.now().strftime("%d%b%Y_%Hh%Mm%Ss")

    # If a branch list is given, read ONLY those columns (plus selection/weight/
    # split columns) -- this is what keeps memory bounded on large samples.
    needed_plot = set(branches) if branches else None

    hg = Histogrammer(samples, jobs=jobs, max_events=max_events,
                      binning_overrides=binning_overrides,
                      needed_plot=needed_plot, step_size=step_size,
                      to_float32=to_float32)
    if verbose:
        print("[cmsplot] reading %d samples (%s) ..."
              % (len(list(samples)),
                 "%d columns" % len(needed_plot) if needed_plot else "all columns"))
    procs = hg.load()

    # fix the total MC normalisation to the data yield (keeps the Bc:Hb ratio,
    # which is set by the per-sample `scale`s, untouched)
    if scale_to_data:
        mc_tot = sum(float(np.sum(p.weight)) for p in procs if not p.is_data)
        dat_tot = sum(float(np.sum(p.weight)) for p in procs if p.is_data)
        if mc_tot > 0 and dat_tot > 0:
            f = dat_tot / mc_tot
            for p in procs:
                if not p.is_data:
                    p.weight = p.weight * f
            if verbose:
                print("[cmsplot] scale-to-data: MC x %.4g (MC %.1f -> data %.0f)"
                      % (f, mc_tot, dat_tot))
        elif verbose:
            print("[cmsplot] scale-to-data requested but MC or data yield is 0")

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
    for i, b in enumerate(todo, 1):
        edges = hg.edges_for(b, nbins=nbins)
        try:
            plotter.draw(b, edges, procs, outdir, label, nbins=nbins)
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
                if s.group:
                    print("  group     = %s" % s.group, file=f)
            print("  selection = %s" % (s.selection if s.selection else "(none)"),
                  file=f)
            print("", file=f)
