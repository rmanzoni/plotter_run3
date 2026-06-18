#!/usr/bin/env python3
"""Command-line driver for the cmsplot stack-plotter.

    python3 plot.py --config samples_rjpsi.py
    python3 plot.py --config samples_rjpsi.py --branches mass mcorr q2 --jobs 8
    python3 plot.py --config samples_rjpsi.py --normalize --no-data   # shape check

The --config file must define a module-level ``samples`` list; it may also
define LUMI, COM and EXTRA (overridden by the matching command-line flags).
"""
import argparse
import importlib.util
import os
import sys


def load_config(path):
    spec = importlib.util.spec_from_file_location("rjpsi_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rjpsi_cfg"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="python config defining `samples`")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--label", default=None, help="output subdir (default: timestamp)")
    ap.add_argument("--branches", nargs="*", default=None, help="subset to plot")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=4, help="concurrent sample reads")
    ap.add_argument("--lumi", type=float, default=None)
    ap.add_argument("--com", type=float, default=None)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--normalize", action="store_true", help="area-normalise (shapes)")
    ap.add_argument("--scale-to-data", action="store_true",
                    help="scale total MC (Bc+Hb) to the data yield")
    ap.add_argument("--overflow", action="store_true",
                    help="fold over/underflow into the edge bins (default: drop)")
    ap.add_argument("--no-data", action="store_true", help="drop data samples")
    ap.add_argument("--maxevents", type=int, default=-1, help="cap events/sample")
    ap.add_argument("--step-size", default="150 MB",
                    help="uproot read chunk size, e.g. '100 MB' or an int #entries")
    ap.add_argument("--float64", action="store_true",
                    help="keep float64 (default downcasts to float32 to save RAM)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--datacard-branches", nargs="*", default=None,
                    help="branches to also emit Combine datacards for "
                         "(.root + .txt under <outdir>/<label>/datacards)")
    ap.add_argument("--datacard-signal", default="Bc",
                    help="datacard process treated as signal (process id 0); "
                         "the rest float with free rateParams")
    args = ap.parse_args()

    # cmsplot must be importable: add the dir holding this script to sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cmsplot import run

    cfg = load_config(args.config)
    samples = list(cfg.samples)
    if args.no_data:
        samples = [s for s in samples if not s.is_data]

    com = args.com if args.com is not None else getattr(cfg, "COM", 13.6)
    lumi = args.lumi if args.lumi is not None else getattr(cfg, "LUMI", None)
    extra = args.extra if args.extra is not None else getattr(cfg, "EXTRA", "Preliminary")
    binning_overrides = getattr(cfg, "BINNING", {})
    scale_to_data = args.scale_to_data or getattr(cfg, "SCALE_TO_DATA", False)
    datacard_def = getattr(cfg, "DATACARD", None)
    axis_titles = getattr(cfg, "AXIS_TITLES", None)
    derived = getattr(cfg, "DERIVED", None)

    run(samples, outdir=args.outdir, label=args.label,
        branches=args.branches, exclude=args.exclude, nbins=args.bins,
        jobs=args.jobs, lumi=lumi, com=com, extra=extra,
        normalize=args.normalize, overflow=args.overflow,
        scale_to_data=scale_to_data, binning_overrides=binning_overrides,
        max_events=args.maxevents, step_size=args.step_size,
        to_float32=not args.float64, verbose=not args.quiet,
        datacard_branches=args.datacard_branches,
        datacard_signal=args.datacard_signal, datacard_def=datacard_def,
        axis_titles=axis_titles, derived=derived)


if __name__ == "__main__":
    main()
