#!/usr/bin/env python3
"""Likelihood scan of the R(J/psi) signal-strength POI with CMS Combine.

The datacards written by cmsplot declare a single POI ``r`` that multiplies the
signal template (process id 0). With ``SPLIT_BC_IN_DATACARD = True`` that signal
is the ``jpsi_tau`` template (B_c -> J/psi tau nu), and the other templates float
as ``rateParam``s, so a plain ``MultiDimFit --algo grid`` over ``r`` profiles
everything else and gives the 1-D profiled likelihood as a function of the
jpsi_tau strength.

What it does
------------
  1. ``text2workspace.py card.txt -o card.root``   (skipped if given a .root)
  2. ``combine -M MultiDimFit ... --algo grid``     (the scan)
  3. read the ``limit`` tree with uproot and plot 2*dNLL vs r, marking the
     68% (2dNLL=1) and 95% (2dNLL=3.84) crossings and the best fit.

Steps 1-2 need ``combine``/``text2workspace.py`` on PATH (your CMSSW + Combine
area). Step 3 needs uproot/numpy/matplotlib. If those live in different
environments, run the fit once, then re-plot with::

    python3 likelihood_scan.py card.txt --skip-fit --input higgsCombine....root

Examples
--------
    # observed scan, r in [0, 5], 60 points
    python3 likelihood_scan.py datacards/m_miss2_jpsi.txt

    # expected (Asimov) sensitivity with signal injected at r = 1
    python3 likelihood_scan.py datacards/m_miss2_jpsi.txt --expected 1

    # just print the commands
    python3 likelihood_scan.py datacards/m_miss2_jpsi.txt --dry-run
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np


# --------------------------------------------------------------------------
# running combine
# --------------------------------------------------------------------------
def _run(cmd, dry_run):
    print("  $ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def make_workspace(card, dry_run):
    """text2workspace on a .txt card; pass a .root workspace straight through.

    The workspace is written next to the card as ``<stem>.workspace.root`` --
    NOT ``<stem>.root``, which is the shape file the datacard's ``shapes`` line
    points at. Reusing that name makes text2workspace open the shape file with
    RECREATE and destroy it before it can read the templates back ("Failed to
    find data_obs ..."). text2workspace is run in the card's own directory so
    the relative shape-file reference resolves; an absolute workspace path is
    returned so the later chdir into the output dir still finds it.
    """
    if card.endswith(".root"):
        return os.path.abspath(card)
    if not dry_run and not shutil.which("text2workspace.py"):
        sys.exit("text2workspace.py not found on PATH -- source your Combine area "
                 "(or pass an already-built .root workspace).")
    carddir = os.path.dirname(os.path.abspath(card)) or "."
    cardbase = os.path.basename(card)
    ws_base = os.path.splitext(cardbase)[0] + ".workspace.root"
    cwd = os.getcwd()
    try:
        os.chdir(carddir)
        _run(["text2workspace.py", cardbase, "-o", ws_base], dry_run)
    finally:
        os.chdir(cwd)
    return os.path.join(carddir, ws_base)


def run_scan(ws, poi, points, rmin, rmax, mass, name, outdir,
             expected, extra, dry_run):
    """Run the MultiDimFit grid scan; return the expected output ROOT path."""
    if not dry_run and not shutil.which("combine"):
        sys.exit("combine not found on PATH -- source your Combine area, or use "
                 "--skip-fit --input <higgsCombine...root> to only (re)plot.")
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        "combine", "-M", "MultiDimFit", ws,
        "-n", name, "-m", str(mass),
        "--algo", "grid", "--points", str(points),
        "-P", poi, "--floatOtherPOIs", "1",
        "--setParameterRanges", "%s=%g,%g" % (poi, rmin, rmax),
        "--saveNLL",
    ]
    if expected is not None:                 # Asimov toy: inject signal at `expected`
        cmd += ["-t", "-1", "--setParameters", "%s=%g" % (poi, expected)]
    if extra:
        cmd += extra.split()
    # combine drops higgsCombine<name>.MultiDimFit.mH<mass>.root in $PWD
    cwd = os.getcwd()
    try:
        os.chdir(outdir)
        _run(cmd, dry_run)
    finally:
        os.chdir(cwd)
    return find_output(outdir, name, mass)


def find_output(outdir, name, mass):
    exact = os.path.join(outdir, "higgsCombine%s.MultiDimFit.mH%s.root"
                         % (name, mass))
    if os.path.exists(exact):
        return exact
    hits = sorted(glob.glob(os.path.join(
        outdir, "higgsCombine%s.MultiDimFit.mH*.root" % name)))
    return hits[-1] if hits else exact


# --------------------------------------------------------------------------
# reading + interval extraction
# --------------------------------------------------------------------------
def load_scan(path, poi):
    """Return (r, two_dnll) sorted by r, with the scan's global minimum at 0."""
    import uproot
    with uproot.open(path) as f:
        t = f["limit"]
        arr = t.arrays([poi, "deltaNLL"], library="np")
    r = np.asarray(arr[poi], "float64")
    dnll = np.asarray(arr["deltaNLL"], "float64")
    m = np.isfinite(r) & np.isfinite(dnll)
    r, dnll = r[m], dnll[m]
    order = np.argsort(r)
    r, dnll = r[order], dnll[order]
    two = 2.0 * (dnll - dnll.min())          # re-anchor so the minimum is exactly 0
    return r, two


def _crossings(x, y, level):
    """All x where y(x) crosses `level`, by linear interpolation."""
    out = []
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        if (y0 - level) * (y1 - level) < 0.0:
            t = (level - y0) / (y1 - y0)
            out.append(x[i] + t * (x[i + 1] - x[i]))
    return out


def interval(r, two, level):
    """68%/95% interval edges bracketing the best fit (None where it runs to the
    scan boundary, i.e. unconstrained on that side)."""
    rhat = r[int(np.argmin(two))]
    xs = _crossings(r, two, level)
    lo = max([x for x in xs if x <= rhat], default=None)
    hi = min([x for x in xs if x >= rhat], default=None)
    return rhat, lo, hi


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def plot_scan(r, two, poi, outdir, name, expected, com=13.6, poi_label=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import mplhep as hep
        plt.style.use(hep.style.CMS)
    except Exception:
        hep = None

    if poi_label is None:
        poi_label = r"\mu_{B_c\to J/\psi\,\tau\nu}"
    rhat, lo68, hi68 = interval(r, two, 1.0)
    _,    lo95, hi95 = interval(r, two, 3.84)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(r, two, "-", color="#3f90da", lw=2, zorder=3)
    ax.plot(r, two, "o", color="#3f90da", ms=3, zorder=4)

    for lvl, lab in ((1.0, "68% CL"), (3.84, "95% CL")):
        ax.axhline(lvl, color="gray", ls="--", lw=1)
        ax.text(r.max(), lvl, " " + lab, va="bottom", ha="right",
                color="gray", fontsize="small")
    ax.axvline(rhat, color="#bd1f01", ls=":", lw=1.5)

    ax.set_xlim(r.min(), r.max())
    # always show a little past the 95% line; cap so a steep tail doesn't
    # squash the region around the minimum
    ax.set_ylim(0.0, max(4.5, min(two.max() * 1.05, 25.0)))
    ax.set_xlabel(r"$%s$  ($%s$)" % (poi_label, poi))
    ax.set_ylabel(r"$-2\,\Delta\ln L$")
    if hep:
        hep.cms.label("Preliminary", ax=ax, data=(expected is None),
                      com=com, loc=0)

    # best-fit annotation (asymmetric 68% errors where defined)
    def _fmt(rhat, lo, hi):
        if lo is None or hi is None:
            return "%s = %.2f" % (poi, rhat)
        return "%s = %.2f$^{+%.2f}_{-%.2f}$" % (poi, rhat, hi - rhat, rhat - lo)
    ax.text(0.04, 0.96, ("Expected\n" if expected is not None else "") + _fmt(rhat, lo68, hi68),
            transform=ax.transAxes, va="top", ha="left", fontsize="medium")

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, (name.lstrip(".") or "scan"))
    for ext in ("png", "pdf"):
        fig.savefig(stem + "." + ext, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
    plt.close(fig)

    print("\n[scan] best fit %s" % _fmt(rhat, lo68, hi68))
    if lo95 is not None or hi95 is not None:
        print("[scan] 95%% interval: [%s, %s]"
              % ("%.2f" % lo95 if lo95 is not None else "-inf",
                 "%.2f" % hi95 if hi95 is not None else "+inf"))
    print("[scan] wrote %s.{png,pdf}" % stem)
    return stem


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("card", help="datacard (.txt) or built workspace (.root)")
    ap.add_argument("--poi", default="r", help="POI name (default: r = jpsi_tau strength)")
    ap.add_argument("--points", type=int, default=60, help="grid points (default 60)")
    ap.add_argument("--r-min", type=float, default=0.0, dest="rmin")
    ap.add_argument("--r-max", type=float, default=5.0, dest="rmax")
    ap.add_argument("--mass", default="125", help="-m label (cosmetic; default 125)")
    ap.add_argument("--com", type=float, default=13.6, help="sqrt(s) for the CMS label")
    ap.add_argument("--poi-label", default=None, dest="poi_label",
                    help="LaTeX for the POI on the x-axis (default: jpsi_tau mu)")
    ap.add_argument("--name", default=None,
                    help="combine -n tag (default derived from the card name)")
    ap.add_argument("--outdir", default="scan", help="output directory")
    ap.add_argument("--expected", type=float, nargs="?", const=1.0, default=None,
                    help="run an Asimov (expected) scan injecting the POI at this "
                         "value (default 1 if the flag is given without a number)")
    ap.add_argument("--extra", default="", help="extra args passed verbatim to combine")
    ap.add_argument("--skip-fit", action="store_true",
                    help="do not run combine; only (re)plot an existing output")
    ap.add_argument("--input", default=None,
                    help="higgsCombine...root to plot (use with --skip-fit)")
    ap.add_argument("--dry-run", action="store_true", help="print commands and exit")
    args = ap.parse_args()

    name = args.name or (".scan_" + os.path.splitext(os.path.basename(args.card))[0])

    if args.skip_fit:
        out = args.input or find_output(args.outdir, name, args.mass)
        if not os.path.exists(out):
            sys.exit("no scan output to plot: %s (run without --skip-fit first, "
                     "or pass --input)" % out)
    else:
        ws = make_workspace(args.card, args.dry_run)
        out = run_scan(ws, args.poi, args.points, args.rmin, args.rmax,
                       args.mass, name, args.outdir, args.expected,
                       args.extra, args.dry_run)
        if args.dry_run:
            print("\n[dry-run] would then plot %s" % out)
            return

    r, two = load_scan(out, args.poi)
    plot_scan(r, two, args.poi, args.outdir, name, args.expected,
              com=args.com, poi_label=args.poi_label)


if __name__ == "__main__":
    main()
