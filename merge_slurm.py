#!/usr/bin/env python3
"""Merge the outputs of a submit_slurm.py array into <outdir>/<label>/.

    python3 merge_slurm.py <outdir>/<label>

Run by the merge job that submit_slurm.py queues with
``--dependency=afterany:<array>``, i.e. only after EVERY array task has
terminated. Standard library only.

Nothing is moved unless all of these hold (exit 2 otherwise, tree untouched):
  * every task has its DONE marker (a failed / timed-out / cancelled task has
    none) -- the missing ones are listed, fix and ``submit_slurm.py --resubmit``;
  * every task reports exactly the branches it was given (plot_status.json);
  * every "ok" branch has its files (formats x lin/log), every datacard
    branch its .root and .txt;
  * yields.txt and selection.txt are byte-identical across tasks: all tasks
    read the same events with the same weights, so any difference means the
    tasks did not see the same inputs, and the merged plots would not belong
    together;
  * no two tasks produced the same file and nothing is overwritten.
Then every file is moved (same filesystem: rename) into the final tree, the
task statuses are merged into plot_status.json, and MERGED is written.
Branches whose drawing raised are listed and make the exit code 1, but the
merge is kept: those plots are simply absent, as in an interactive run.
With --png-from-pdf the PNGs are then rendered from the merged PDFs
(pdf2png.py); a missing converter stops the merge before anything moves, a
failed conversion is listed and makes the exit code 1.
"""
import json
import os
import sys

BOOKKEEPING = {"DONE", "yields.txt", "selection.txt", "plot_status.json",
               "resources.txt"}


def plot_files(formats):
    return [os.path.join(ext, sub, "%s." + ext)
            for ext in formats for sub in ("lin", "log")]


def fail(msg, code=2):
    print("[merge] FAILED: " + msg)
    sys.exit(code)


def main():
    if len(sys.argv) != 2:
        fail("usage: merge_slurm.py <outdir>/<label>")
    final = os.path.abspath(sys.argv[1])
    slurmdir = os.path.join(final, "_slurm")
    man = json.load(open(os.path.join(slurmdir, "submission.json")))
    tasks = {int(k): v for k, v in man["tasks"].items()}
    tdir = {i: os.path.join(slurmdir, "tasks", "task%03d" % i) for i in tasks}
    PLOT_FILES = plot_files(man.get("formats", ["png", "pdf"]))
    png_from_pdf = man.get("png_from_pdf", False)
    if png_from_pdf:
        import pdf2png
        if pdf2png.converter()[0] is None:
            fail("--png-from-pdf but neither pdftoppm nor gs on %s"
                 % os.uname().nodename)

    if os.path.exists(os.path.join(final, "MERGED")):
        fail("%s is already merged" % final)

    # 1. completeness
    missing = [i for i in sorted(tasks)
               if not os.path.isfile(os.path.join(tdir[i], "DONE"))]
    if missing:
        fail("%d/%d task(s) have no DONE marker: %s\n  logs: %s/logs/\n"
             "  then: python3 submit_slurm.py --resubmit %s"
             % (len(missing), len(tasks), missing, slurmdir, final))

    # 2. per-task content
    problems, status, moves = [], {}, []
    for i in sorted(tasks):
        t, d = tasks[i], tdir[i]
        st = json.load(open(os.path.join(d, "plot_status.json")))
        want = set(t.get("branches", []))
        if set(st) != want:
            problems.append("task %d: status covers %s, expected %s"
                            % (i, sorted(set(st) ^ want), "its chunk"))
        for b, s in st.items():
            if s == "ok":
                absent = [f % b for f in PLOT_FILES
                          if not os.path.isfile(os.path.join(d, f % b))]
                if absent:
                    problems.append("task %d: %s is 'ok' but lacks %s"
                                    % (i, b, absent))
        for b in t.get("datacard_branches", []):
            for ext in ("root", "txt"):
                if not os.path.isfile(os.path.join(d, "datacards",
                                                   "%s.%s" % (b, ext))):
                    problems.append("task %d: datacard %s.%s not written"
                                    % (i, b, ext))
        status.update(st)
        for root, _, files in os.walk(d):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), d)
                if rel not in BOOKKEEPING:
                    moves.append((os.path.join(d, rel), rel, i))

    # 3. cross-task consistency
    for name in ("yields.txt", "selection.txt"):
        ref_i = min(tasks)
        ref = open(os.path.join(tdir[ref_i], name), "rb").read()
        for i in sorted(tasks):
            if open(os.path.join(tdir[i], name), "rb").read() != ref:
                problems.append("%s of task %d differs from task %d"
                                % (name, i, ref_i))

    # 4. collisions
    seen = {}
    for _, rel, i in moves:
        if rel in seen:
            problems.append("%s produced by tasks %d and %d" % (rel, seen[rel], i))
        seen[rel] = i
        if os.path.exists(os.path.join(final, rel)):
            problems.append("%s already exists in %s" % (rel, final))
    for name in ("yields.txt", "selection.txt", "plot_status.json"):
        if os.path.exists(os.path.join(final, name)):
            problems.append("%s already exists in %s" % (name, final))

    if problems:
        fail("%d problem(s), nothing merged:\n  " % len(problems)
             + "\n  ".join(problems))

    # --- merge ---
    for src, rel, _ in moves:
        dst = os.path.join(final, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
    ref = tdir[min(tasks)]
    for name in ("yields.txt", "selection.txt"):
        with open(os.path.join(ref, name), "rb") as fi, \
                open(os.path.join(final, name), "wb") as fo:
            fo.write(fi.read())
    with open(os.path.join(final, "plot_status.json"), "w") as f:
        json.dump(status, f, indent=1, sort_keys=True)

    res = {}
    for i in sorted(tasks):
        p = os.path.join(tdir[i], "resources.txt")
        if os.path.isfile(p):
            for line in open(p):
                k, _, v = line.strip().partition(" ")
                res.setdefault(k, []).append((v, i))
    conv_fails = []
    if png_from_pdf:
        ok = {b for b, s in status.items() if s == "ok"}
        n, conv_fails = pdf2png.convert(
            final, dpi=150, branches=ok,
            jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    errors = sorted(b for b, s in status.items() if s.startswith("error"))
    n_ok = sum(1 for s in status.values() if s == "ok")
    n_nomc = sum(1 for s in status.values() if s == "no_mc")
    summary = {"tasks": len(tasks), "ok": n_ok, "no_mc": n_nomc,
               "errors": errors, "png_conversion_failures": conv_fails}
    with open(os.path.join(final, "MERGED"), "w") as f:
        json.dump(summary, f, indent=1)

    print("[merge] %d task(s) merged into %s" % (len(tasks), final))
    print("[merge] %d plotted, %d without MC, %d failed" % (n_ok, n_nomc,
                                                            len(errors)))
    if "elapsed_s" in res:
        v, i = max(res["elapsed_s"], key=lambda x: int(x[0]))
        print("[merge] slowest task: %s s (task %d) -- limit it against the "
              "partition's wall time" % (v, i))
    if "peak" in res:
        kb = [(int(v.split()[1]), i) for v, i in res["peak"]
              if len(v.split()) > 1 and v.split()[1].isdigit()]
        if kb:
            v, i = max(kb)
            print("[merge] largest peak RSS: %.0f MB (task %d; fork workers "
                  "share pages, so Slurm's view may differ)" % (v / 1024.0, i))
    if errors:
        print("[merge] branches whose drawing raised (see plot_status.json):")
        for b in errors:
            print("  %s: %s" % (b, status[b]))
    if conv_fails:
        print("[merge] PNG conversions failed (rerun: python3 pdf2png.py %s "
              "--overwrite):" % final)
        for f in conv_fails:
            print("  " + f)
    if errors or conv_fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
