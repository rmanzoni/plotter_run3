#!/usr/bin/env python3
"""Draw the LHCb-style stitched 3D template (decay time x Z x m^2_miss).

The R(J/psi) 3D fit is run on ONE long 1D "stitched" channel whose global bin is
``g = (i_t * N_Z + i_Z) * N_m + i_m`` -- i.e. for every (decay-time, Z) cell the
m^2_miss distribution is laid end-to-end. This script reads the shape file that
``cmsplot`` writes for that channel (one TH1 per Combine process + ``data_obs``,
all with N_T*N_Z*N_M bins), unrolls it back to (N_T, N_Z, N_m), and draws the
familiar stitched stack: m^2_miss runs inside each panel, light dotted lines
separate the Z cells, heavier lines separate the decay-time blocks, and a
data/model ratio sits underneath.

Usage:
    python3 postfit_tools/stitch_plot.py \
        plots/<label>/datacards/lhcb_3d_index.root \
        --config samples_rjpsi.py \
        --out plots/<label>/lhcb_stitched

``--config`` is optional: it is only read to pick up the exact bin edges/sizes
(LHCB_T_EDGES, LHCB_Q2_EDGES, LHCB_ESTAR_EDGES, N_T/N_Z/N_M) for the axis labels.
Without it the script falls back to the defaults baked in below, and the
structure can still be overridden with --nt/--nz/--nm.

The input is the *pre-fit* template file produced by ``plot.py
--datacard-branches lhcb_3d_index``. (Post-fit shapes from a FitDiagnostics run
have a different, per-channel directory layout; use postfit_plots.py for those.)
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

# headless-safe before any pyplot import (mirrors cmsplot.core)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# make the sibling cmsplot package importable (this file lives in postfit_tools/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from cmsplot import style
from cmsplot.style import PETROFF_10 as P

# --- defaults (overridden by --config / --nt/--nz/--nm) ----------------------
DEF_T_EDGES     = [0.0, 0.4, 0.7, 1.1, 1.6, 3.0]
DEF_Q2_EDGES    = [0.0, 8.0, 10.12]
DEF_ESTAR_EDGES = [0.0, 0.5, 0.9, 1.3, 3.5, 5.0]
DEF_MMISS2_RANGE = (-10.0, 10.0)

# datacard process -> (colour, legend label). Mirrors the COMPONENTS/DATACARD
# split in samples_rjpsi.py so the stitched plot matches the standard plots.
PROC_STYLE = {
    "jpsi_mu":  (P[0], r"$B_c\!\to\! J/\psi\,\mu\nu$"),
    "feeddown": (P[1], r"$B_c\!\to\!(\psi',\chi_c,h_c)\,\ell\nu$"),
    "jpsi_D":   (P[4], r"$B_c\!\to\! J/\psi + D_{(s)}$"),
    "bc_other": (P[3], r"$B_c\!\to\! J/\psi + \mathrm{had.}$"),
    "Bc":       (P[6], r"$B_c$ (merged)"),
    "Hb":       (P[5], r"$H_b\!\to\! J/\psi + X$"),
    "misID":    (P[9], r"misID (data-driven)"),
    "jpsi_tau": (P[2], r"$B_c\!\to\! J/\psi\,\tau\nu$"),
}
# stack order: backgrounds first (bottom), signal last (top of stack)
STACK_ORDER = ["jpsi_mu", "feeddown", "jpsi_D", "bc_other", "Bc", "Hb",
               "misID", "jpsi_tau"]


def load_structure(config_path, nt, nz, nm):
    """Return (t_edges, q2_edges, estar_edges, mmiss2_range, N_T, N_Z, N_M).

    Reads the edges/sizes from the config module if given; CLI --nt/--nz/--nm
    take final precedence (and are also the only way to set the structure when
    no config is supplied).
    """
    t_edges, q2_edges, e_edges = DEF_T_EDGES, DEF_Q2_EDGES, DEF_ESTAR_EDGES
    mrange = DEF_MMISS2_RANGE
    N_T = N_Z = N_M = None
    if config_path:
        spec = importlib.util.spec_from_file_location("stitch_cfg", config_path)
        cfg = importlib.util.module_from_spec(spec)
        sys.modules["stitch_cfg"] = cfg
        spec.loader.exec_module(cfg)
        t_edges = list(getattr(cfg, "LHCB_T_EDGES", t_edges))
        q2_edges = list(getattr(cfg, "LHCB_Q2_EDGES", q2_edges))
        e_edges = list(getattr(cfg, "LHCB_ESTAR_EDGES", e_edges))
        mrange = tuple(getattr(cfg, "LHCB_MMISS2_RANGE", mrange))
        N_T = getattr(cfg, "N_T", None)
        N_Z = getattr(cfg, "N_Z", None)
        N_M = getattr(cfg, "N_M", None)
    N_T = nt or N_T or (len(t_edges) - 1)
    N_Z = nz or N_Z or ((len(q2_edges) - 1) * (len(e_edges) - 1))
    N_M = nm or N_M or getattr(load_structure, "_default_nm", 20)
    return t_edges, q2_edges, e_edges, mrange, N_T, N_Z, N_M


def read_templates(path):
    """Read every TH1 in the shape file -> {name: (values, variances)} + data."""
    import uproot
    f = uproot.open(path)
    procs, data = {}, None
    for key in set(k.split(";")[0] for k in f.keys()):
        obj = f[key]
        if not hasattr(obj, "values"):
            continue
        vals = np.asarray(obj.values(flow=False), "float64")
        try:
            var = np.asarray(obj.variances(flow=False), "float64")
        except Exception:
            var = vals.copy()
        if key == "data_obs":
            data = (vals, var)
        else:
            procs[key] = (vals, var)
    return procs, data


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shapefile", help="cmsplot datacard shape file "
                    "(<...>/datacards/lhcb_3d_index.root)")
    ap.add_argument("--config", default=None,
                    help="sample config to read the exact edges/sizes from")
    ap.add_argument("--out", default=None,
                    help="output path stem (default: alongside the shape file)")
    ap.add_argument("--signal", default="jpsi_tau",
                    help="signal process name (drawn on top of the stack)")
    ap.add_argument("--nt", type=int, default=None, help="override N decay-time bins")
    ap.add_argument("--nz", type=int, default=None, help="override N Z bins")
    ap.add_argument("--nm", type=int, default=None, help="override N m^2_miss bins")
    ap.add_argument("--lumi", type=float, default=None)
    ap.add_argument("--com", type=float, default=13.6)
    ap.add_argument("--extra", default="Preliminary")
    ap.add_argument("--logy", action="store_true")
    args = ap.parse_args()

    (t_edges, q2_edges, e_edges, mrange,
     N_T, N_Z, N_M) = load_structure(args.config, args.nt, args.nz, args.nm)
    N_TOT = N_T * N_Z * N_M

    procs, data = read_templates(args.shapefile)
    if not procs:
        raise SystemExit("no process templates found in %s" % args.shapefile)
    nbins = len(next(iter(procs.values()))[0])
    if nbins != N_TOT:
        raise SystemExit(
            "shape file has %d bins but structure says N_T*N_Z*N_M = %d*%d*%d "
            "= %d.\nPass --config or --nt/--nz/--nm to match (e.g. the binning "
            "used when the datacard was written)." % (nbins, N_T, N_Z, N_M, N_TOT))

    style.set_cms_style()
    x = np.arange(N_TOT) + 0.5                       # bin centres 0.5 .. N_TOT-0.5

    fig = plt.figure(figsize=(min(2.2 + 0.022 * N_TOT, 34), 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)

    # --- stacked backgrounds + signal on top --------------------------------
    order = [p for p in STACK_ORDER if p in procs]
    order += [p for p in procs if p not in order]    # any unforeseen process
    if args.signal in order:                          # ensure signal is last
        order = [p for p in order if p != args.signal] + [args.signal]

    bottom = np.zeros(N_TOT)
    total = np.zeros(N_TOT)
    for name in order:
        vals = procs[name][0]
        colour, lab = PROC_STYLE.get(name, (None, name))
        ax.bar(x, vals, width=1.0, bottom=bottom, align="center",
               color=colour, linewidth=0.0, label=lab, zorder=2)
        bottom = bottom + vals
        total = total + vals

    # --- data points ---------------------------------------------------------
    if data is not None:
        d, dvar = data
        derr = np.sqrt(np.where(dvar > 0, dvar, d))
        ax.errorbar(x, d, yerr=derr, fmt="o", ms=2.0, color="k",
                    elinewidth=0.7, capsize=0, label="Data", zorder=4)

    # --- divider lines + decay-time block labels -----------------------------
    ymax = max(total.max(), (data[0].max() if data is not None else 0.0))
    ax.set_ylim(0 if not args.logy else max(0.5, total[total > 0].min() * 0.5),
                ymax * (40 if args.logy else 1.50))
    if args.logy:
        ax.set_yscale("log")

    for k in range(1, N_T * N_Z):                     # Z-cell separators (light)
        if k % N_Z:
            ax.axvline(k * N_M, color="0.80", lw=0.4, ls=":", zorder=1)
    for it in range(1, N_T):                          # decay-time separators
        ax.axvline(it * N_Z * N_M, color="0.25", lw=1.1, zorder=3)

    ax.set_xlim(0, N_TOT)
    ax.set_ylabel("Events")
    style.cms_label(ax, lumi=args.lumi, com=args.com,
                    data=(data is not None), extra=args.extra)

    # decay-time labels at the LEFT of each block (clears the upper-right legend)
    ytxt = ax.get_ylim()[1] * (8 if args.logy else 0.97)
    for it in range(N_T):
        x0 = it * N_Z * N_M + 0.008 * N_TOT
        ax.text(x0, ytxt, r"$t\in[%g,%g)$ ps" % (t_edges[it], t_edges[it + 1]),
                ha="left", va="top", fontsize=8.5, color="0.15")

    # --- legend (reverse so top-of-stack is first) ---------------------------
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], ncol=2, fontsize=9, loc="upper right",
              frameon=False)

    # --- ratio panel ---------------------------------------------------------
    if data is not None:
        d, dvar = data
        good = total > 0
        ratio = np.where(good, d / np.where(good, total, 1.0), np.nan)
        rerr = np.where(good, np.sqrt(np.where(dvar > 0, dvar, d))
                        / np.where(good, total, 1.0), np.nan)
        rax.errorbar(x, ratio, yerr=rerr, fmt="o", ms=2.0, color="k",
                     elinewidth=0.7, capsize=0)
    rax.axhline(1.0, color="0.4", ls="--", lw=0.8)
    for it in range(1, N_T):
        rax.axvline(it * N_Z * N_M, color="0.25", lw=1.1)
    rax.set_ylim(0.0, 2.0)
    rax.set_ylabel("Data / pred.", fontsize=10)
    rax.set_xlabel(r"stitched bin: $m^{2}_{\mathrm{miss}}\in[%g,%g)$ GeV$^{2}$ "
                   r"within each $(t, Z)$ cell" % mrange)
    plt.setp(ax.get_xticklabels(), visible=False)

    out = args.out or os.path.splitext(args.shapefile)[0] + "_stitched"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (out, ext), bbox_inches="tight", dpi=140)
        print("wrote %s.%s" % (out, ext))
    plt.close(fig)


if __name__ == "__main__":
    main()
