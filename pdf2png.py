#!/usr/bin/env python3
"""Render the PNGs of a cmsplot output tree from its PDFs.

    python3 pdf2png.py <outdir>/<label> [--dpi 150] [--jobs 4] [--overwrite]

For every ``pdf/{lin,log}/<branch>.pdf`` writes ``png/{lin,log}/<branch>.png``
at ``--dpi`` (150 = what cmsplot's own PNGs use). This lets the plotting jobs
save PDFs only (``plot.py --formats pdf``) and leaves the rasterisation to a
cheap step afterwards; submit_slurm.py --png-from-pdf runs it in the merge job.

Uses ``pdftoppm`` (poppler), else Ghostscript ``gs``; with neither on PATH it
stops before converting anything. Standard library only. The PNGs match the
matplotlib ones in size (same bounding box, same dpi) but are rasterised by a
different engine, so they are not pixel-identical to --formats png output.
Exit code: 0 all converted, 1 some failed (listed), 2 nothing done.
"""
import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def converter():
    """(name, fn(pdf, png, dpi) -> argv) for the first available tool."""
    if shutil.which("pdftoppm"):
        # -singlefile writes <prefix>.png; the prefix must not carry .png
        return "pdftoppm", lambda pdf, png, dpi: [
            "pdftoppm", "-png", "-r", str(dpi), "-singlefile", pdf, png[:-4]]
    if shutil.which("gs"):
        return "gs", lambda pdf, png, dpi: [
            "gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m",
            "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", "-r%d" % dpi,
            "-o", png, pdf]
    return None, None


def todo_list(root, branches=None):
    out = []
    for sub in ("lin", "log"):
        d = os.path.join(root, "pdf", sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".pdf"):
                continue
            b = f[:-4]
            if branches is not None and b not in branches:
                continue
            out.append((os.path.join(d, f),
                        os.path.join(root, "png", sub, b + ".png")))
    return out


def convert(root, dpi=150, jobs=4, overwrite=False, branches=None, quiet=False):
    """Convert; return (n_done, [failures]). Raises SystemExit(2) if no tool
    or if a target exists without ``overwrite`` (nothing converted then)."""
    name, argv = converter()
    if name is None:
        raise SystemExit("pdf2png: neither pdftoppm nor gs on PATH; nothing "
                         "converted")
    pairs = todo_list(root, branches)
    if not pairs:
        raise SystemExit("pdf2png: no PDFs under %s/pdf/{lin,log}" % root)
    clash = [png for _, png in pairs if os.path.exists(png)]
    if clash and not overwrite:
        raise SystemExit("pdf2png: %d PNG(s) already exist (e.g. %s); pass "
                         "--overwrite" % (len(clash), clash[0]))
    for _, png in pairs:
        os.makedirs(os.path.dirname(png), exist_ok=True)

    def one(pair):
        pdf, png = pair
        r = subprocess.run(argv(pdf, png, dpi), capture_output=True, text=True)
        if r.returncode != 0 or not os.path.isfile(png):
            return "%s: %s rc=%d %s" % (pdf, name, r.returncode,
                                        r.stderr.strip()[:200])
        return None

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        fails = [e for e in ex.map(one, pairs) if e]
    if not quiet:
        print("[pdf2png] %d/%d PDF(s) -> PNG at %d dpi with %s"
              % (len(pairs) - len(fails), len(pairs), dpi, name))
    return len(pairs) - len(fails), fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="<outdir>/<label> of a cmsplot run")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    _, fails = convert(os.path.abspath(a.root), a.dpi, a.jobs, a.overwrite)
    for f in fails:
        print("  ! " + f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
