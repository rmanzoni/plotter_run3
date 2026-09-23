#!/usr/bin/env python3
"""Check merged ntuples for duplicated rows -- no input chunks needed.

    python3 check_duplicates.py FILE.root [FILE.root ...] [--keys run lumi event pt jpsi_mass]

A row is identified by the --keys branches (default: those of
count_duplicated_events.py -- run/lumi/event plus two candidate-level
quantities, so several candidates of one event are NOT duplicates of each
other). Floats are compared bitwise. The hadd_uproot.py cycle bug writes every
affected input file twice, so its signature is: extra copies ~= half of all
rows, and (almost) every duplicated key has multiplicity exactly 2.

Exit code 1 if any file has duplicates.
"""
import argparse
import sys
from collections import Counter

import numpy as np
import uproot


def check(path, keys, tree="tree"):
    t = uproot.open(path)[tree]
    missing = [k for k in keys if k not in t.keys()]
    if missing:
        raise SystemExit("%s: key branch(es) %s not in the tree" % (path, missing))
    a = t.arrays(keys, library="np")
    rec = np.empty(t.num_entries, dtype=[(k, a[k].dtype.newbyteorder("=")) for k in keys])
    for k in keys:
        rec[k] = a[k]
    del a
    raw = np.ascontiguousarray(rec).view(np.dtype((np.void, rec.dtype.itemsize)))
    _, counts = np.unique(raw, return_counts=True)
    n, n_unique = t.num_entries, counts.size
    mult = Counter(counts[counts > 1].tolist())
    return n, n_unique, mult


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--keys", nargs="+", default=["run", "lumi", "event", "pt", "jpsi_mass"])
    ap.add_argument("--tree", default="tree")
    a = ap.parse_args()
    bad = False
    for f in a.files:
        n, u, mult = check(f, a.keys, a.tree)
        extra = n - u
        print("%-60s rows %11d  unique %11d  extra copies %11d (%.2f%%)  multiplicities %s"
              % (f.split("/")[-1], n, u, extra, 100.0 * extra / max(n, 1),
                 dict(sorted(mult.items())) or "-"))
        if extra:
            bad = True
            if mult.get(2, 0) >= 0.99 * sum(mult.values()) and extra > 0.4 * n:
                print("    -> matches the hadd_uproot cycle bug (whole files written twice)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
