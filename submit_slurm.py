#!/usr/bin/env python3
"""Split a cmsplot run into Slurm array jobs on the PSI T3, then merge.

    python3 submit_slurm.py [slurm options] -- <plot.py options>
    python3 submit_slurm.py --resubmit <outdir>/<label>

Everything after ``--`` is parsed by plot.py's own argument parser, so any
plot.py option is accepted and means the same thing it means interactively.
The result lands where ``plot.py`` would put it: ``<outdir>/<label>/{png,pdf}/
{lin,log}/``, ``yields.txt``, ``selection.txt``, ``datacards/``, plus
``plot_status.json``. Bookkeeping lives in ``<outdir>/<label>/_slurm/``.

Layout of a submission
----------------------
* On the UI (here): the config is loaded and the plottable branches are listed
  from the ntuple SCHEMAS on /pnfs (no data read); they are cut into chunks of
  ``--branches-per-job``. plot.py, cmsplot/ and the config are SNAPSHOT into
  ``_slurm/code/`` so editing the working copy while jobs queue cannot make
  two tasks run different code.
* One Slurm array: task 0 writes the datacards (``--datacards-only``) when
  ``--datacard-branches`` is given; every other task plots one branch chunk.
  Each task reads the /pnfs inputs through the dCache xrootd door (/pnfs is
  not mounted on the worker nodes), works in /scratch, and on success moves
  its output to ``_slurm/tasks/taskNNN/`` and drops a DONE marker in it.
* One merge job with ``--dependency=afterany:<array>``: Slurm starts it only
  once EVERY array task has terminated (success or not). It merges nothing
  unless every task has its DONE marker and all tasks agree on yields.txt and
  selection.txt; otherwise it lists what is missing and exits non-zero. Then
  ``--resubmit`` resubmits only the missing tasks plus a fresh merge job.
"""
import argparse
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot as _plot                                     # noqa: E402

XROOTD_T3 = "root://t3dcachedb03.psi.ch:1094/"
SNAPSHOT_KEEP = ("plot.py", "cmsplot", "merge_slurm.py", "pdf2png.py")


# ---------------------------------------------------------------------------
# argv <-> namespace
# ---------------------------------------------------------------------------
def ns_to_argv(parser, ns, skip=()):
    """Rebuild a plot.py argv from a parsed namespace (non-default values only).

    Generic over plot.py's parser, so a new plot.py option is forwarded to the
    jobs without touching this file -- unless it uses an action type this
    serialiser does not know, which is a hard error rather than a silently
    dropped option.
    """
    argv = []
    for act in parser._actions:
        if not act.option_strings or act.dest in skip or act.dest == "help":
            continue
        val = getattr(ns, act.dest)
        if val == act.default:
            continue
        opt = max(act.option_strings, key=len)             # the --long form
        if isinstance(act, argparse._StoreTrueAction):
            argv.append(opt)
        elif isinstance(act, argparse._StoreAction):
            if act.nargs in ("*", "+"):
                argv += [opt] + [str(v) for v in val]
            else:
                argv.append("%s=%s" % (opt, val))
        else:
            raise SystemExit("submit_slurm: plot.py option %s uses %s, which "
                             "ns_to_argv cannot forward -- extend it"
                             % (opt, type(act).__name__))
    return argv


# ---------------------------------------------------------------------------
# environment / preflight
# ---------------------------------------------------------------------------
def env_setup_lines(override=None):
    """Shell lines that recreate THIS python environment on a worker node.

    Mirrors the ntuplizer submitters for CMSSW (cmsset_default + scram runtime
    from $CMSSW_BASE); a conda env is re-activated from its prefix; a venv is
    sourced. ``--env-setup`` replaces all of this verbatim.
    """
    if override:
        return [override]
    if os.environ.get("CMSSW_BASE"):
        return ["export SCRAM_ARCH=%s" % os.environ.get("SCRAM_ARCH", ""),
                "source /cvmfs/cms.cern.ch/cmsset_default.sh",
                "pushd %s/src > /dev/null" % os.environ["CMSSW_BASE"],
                "eval `scramv1 runtime -sh`",
                "popd > /dev/null"]
    if os.environ.get("CONDA_PREFIX"):
        conda_exe = os.environ.get("CONDA_EXE")
        if not conda_exe:
            raise SystemExit("submit_slurm: CONDA_PREFIX is set but CONDA_EXE "
                             "is not; pass --env-setup explicitly")
        base = os.path.dirname(os.path.dirname(conda_exe))
        return ["source %s/etc/profile.d/conda.sh" % base,
                "conda activate %s" % os.environ["CONDA_PREFIX"]]
    if os.environ.get("VIRTUAL_ENV"):
        return ["source %s/bin/activate" % os.environ["VIRTUAL_ENV"]]
    return ["export PATH=%s:$PATH" % os.path.dirname(sys.executable)]


def preflight_imports():
    mods = ["numpy", "uproot", "matplotlib", "mplhep"]
    missing = []
    for m in mods:
        try:
            __import__(m)
        except Exception as e:
            missing.append("%s (%s)" % (m, e))
    if missing:
        raise SystemExit("submit_slurm: this environment (which the jobs will "
                         "reproduce) cannot import: %s" % ", ".join(missing))


def check_converter_in_env():
    """--png-from-pdf runs pdftoppm/gs in the merge job on a worker node.
    The T3 WNs have neither system-wide (merge 472988 on t3wn81), and a tool
    the UI finds in /usr/bin says nothing about the WNs -- so it must come
    from the environment the jobs recreate (e.g. conda-forge poppler)."""
    import pdf2png
    name, _ = pdf2png.converter()
    path = shutil.which(name) if name else None
    prefixes = [p for p in (sys.prefix, os.environ.get("CONDA_PREFIX"),
                            os.environ.get("CMSSW_BASE")) if p]
    if path and any(os.path.realpath(path).startswith(os.path.realpath(p) + os.sep)
                    for p in prefixes):
        print("[submit_slurm] PNG converter from the job environment: %s" % path)
        return
    raise SystemExit(
        "submit_slurm: --png-from-pdf needs pdftoppm or gs INSIDE the job "
        "environment (the worker nodes have neither); found: %s\n"
        "  -> conda install -c conda-forge poppler   (in this env), or drop "
        "--png-from-pdf" % (path or "nothing"))


def check_xrootd_read(samples, prefix):
    """Read one input exactly as the jobs will: through the door, with
    cmsplot's own open options, and down to basket data (the metadata alone
    -- e.g. num_entries -- does not exercise the byte-offset reads that the
    jobs make). Importing the XRootD bindings is not enough either: uproot may
    route root:// through fsspec, which needs fsspec-xrootd."""
    import uproot
    from cmsplot.core import _open_options, _numeric_branches
    for s in samples:
        for f in s.files:
            if f.startswith("/pnfs/") and prefix:
                url = prefix + f
            elif "://" in f:
                url = f
            else:
                continue
            print("[submit_slurm] test read through xrootd: %s" % url)
            try:
                flat = _numeric_branches(url, s.tree)
                chunk = next(uproot.iterate(url + ":" + s.tree,
                                            expressions=flat[:3],
                                            step_size="1 MB", library="np",
                                            **_open_options(url)))
                n = len(next(iter(chunk.values())))
            except Exception as e:
                hint = ""
                if "fsspec-xrootd" in str(e):
                    hint = ("\n  -> pip install fsspec-xrootd   (in this env; "
                            "the jobs reuse it)")
                raise SystemExit("submit_slurm: cannot read %s: %s: %s%s"
                                 % (url, type(e).__name__, e, hint))
            print("[submit_slurm]   ok, read %d entries of %s"
                  % (n, flat[:3]))
            return


def check_proxy(time_min):
    px = os.environ.get("X509_USER_PROXY")
    if not px:
        raise SystemExit("submit_slurm: X509_USER_PROXY is not set. Point it "
                         "to a proxy on a filesystem the worker nodes see "
                         "(not /tmp), or pass --no-proxy.")
    px = os.path.abspath(px)
    if not os.path.isfile(px):
        raise SystemExit("submit_slurm: X509_USER_PROXY=%s does not exist" % px)
    if px.startswith("/tmp/"):
        raise SystemExit("submit_slurm: X509_USER_PROXY=%s is on the UI's "
                         "local /tmp, invisible to the worker nodes; copy it "
                         "to e.g. ~/.x509up and re-export" % px)
    try:
        left = int(subprocess.check_output(
            ["voms-proxy-info", "--file", px, "--timeleft"],
            stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        print("  ! could not run voms-proxy-info; proxy lifetime unchecked")
        return px
    if left < 60 * (time_min + 120):
        raise SystemExit("submit_slurm: proxy has %.1f h left; renew it "
                         "(jobs may queue, then run up to %d min)"
                         % (left / 3600.0, time_min))
    return px


# ---------------------------------------------------------------------------
# scripts
# ---------------------------------------------------------------------------
TASK_SH = r"""#!/bin/bash
# cmsplot array task (generated by submit_slurm.py -- do not edit)
set -uo pipefail
TASK=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
SLURMDIR={slurmdir}
LABEL={label}
DEST=$SLURMDIR/tasks/task$TASK
SCRATCH={scratch_base}/cmsplot_${{LABEL}}_${{SLURM_ARRAY_JOB_ID}}_$TASK
NCPU=${{SLURM_CPUS_PER_TASK:-1}}

echo ">>>> task $TASK on $(hostname) at $(date)"
mkdir -p "$SCRATCH" || {{ echo ">>>> FATAL: cannot create $SCRATCH"; exit 1; }}
trap 'rm -rf "$SCRATCH"' EXIT

# --- environment (captured at submission) ---
{env_setup}
echo ">>>> python: $(command -v python3)"
python3 -c "import numpy, uproot, matplotlib, mplhep{xrootd_import}" || {{
    echo ">>>> FATAL: plotting environment incomplete on $(hostname)"; exit 1; }}

# --- grid proxy: private copy that survives the job ---
{proxy}

# --- per-task arguments: one per line ---
mapfile -t ARGS < "$SLURMDIR/tasks/task$TASK.args"
[ ${{#ARGS[@]}} -gt 0 ] || {{ echo ">>>> FATAL: no args for task $TASK"; exit 1; }}

export MPLBACKEND=Agg
cd "$SLURMDIR/code"
START=$SECONDS
TIMER=()
[ -x /usr/bin/time ] && TIMER=(/usr/bin/time -f ">>>> peak RSS %M kB" -o "$SCRATCH/rss.txt")
"${{TIMER[@]}}" python3 plot.py "${{ARGS[@]}}" --outdir "$SCRATCH/out" --label "$LABEL" --jobs "$NCPU"
RC=$?
ELAPSED=$((SECONDS - START))
[ -f "$SCRATCH/rss.txt" ] && cat "$SCRATCH/rss.txt"
echo ">>>> plot.py exit code $RC after $ELAPSED s"
[ $RC -eq 0 ] || exit $RC

# --- publish: copy to the shared FS, then an atomic rename + DONE marker ---
TMP=$DEST.tmp.$SLURM_JOB_ID
rm -rf "$TMP"
cp -r "$SCRATCH/out/$LABEL" "$TMP" || {{ echo ">>>> FATAL: copy to $TMP failed"; rm -rf "$TMP"; exit 1; }}
{{ echo "elapsed_s $ELAPSED"; [ -f "$SCRATCH/rss.txt" ] && sed 's/>>>> //' "$SCRATCH/rss.txt"; echo "host $(hostname)"; echo "slurm_job $SLURM_JOB_ID"; }} > "$TMP/resources.txt"
touch "$TMP/DONE"
rm -rf "$DEST" && mv "$TMP" "$DEST" || {{ echo ">>>> FATAL: publishing $DEST failed"; exit 1; }}
echo ">>>> task $TASK done at $(date)"
"""

MERGE_SH = r"""#!/bin/bash
# cmsplot merge job (generated by submit_slurm.py -- do not edit)
set -uo pipefail
{env_setup}
cd {slurmdir}/code
python3 merge_slurm.py {final}
"""


def write_scripts(slurmdir, final, label, a, need_xrootd):
    env = "\n".join(env_setup_lines(a.env_setup))
    if a.no_proxy:
        proxy = 'echo ">>>> no grid proxy (--no-proxy)"'
    else:
        proxy = "\n".join([
            'cp "$X509_USER_PROXY" "$SCRATCH/x509proxy" || '
            '{ echo ">>>> FATAL: cannot copy proxy $X509_USER_PROXY"; exit 1; }',
            'chmod 600 "$SCRATCH/x509proxy"',
            'export X509_USER_PROXY="$SCRATCH/x509proxy"'])
    with open(os.path.join(slurmdir, "task.sh"), "w") as f:
        f.write(TASK_SH.format(
            slurmdir=shlex.quote(slurmdir), label=shlex.quote(label),
            scratch_base=a.scratch_base, env_setup=env, proxy=proxy,
            xrootd_import=", fsspec_xrootd" if need_xrootd else ""))
    with open(os.path.join(slurmdir, "merge.sh"), "w") as f:
        f.write(MERGE_SH.format(env_setup=env, slurmdir=shlex.quote(slurmdir),
                                final=shlex.quote(final)))
    for n in ("task.sh", "merge.sh"):
        os.chmod(os.path.join(slurmdir, n), 0o755)


def snapshot_code(slurmdir, config, extra):
    code = os.path.join(slurmdir, "code")
    os.makedirs(code)
    for n in SNAPSHOT_KEEP:
        src = os.path.join(HERE, n)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(code, n),
                            ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, code)
    shutil.copy2(config, code)
    for e in extra:
        shutil.copy2(e, code)
    return os.path.join(code, os.path.basename(config))


def git_provenance():
    def _git(*args):
        try:
            return subprocess.check_output(["git", "-C", HERE] + list(args),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {"commit": _git("rev-parse", "HEAD"),
            "dirty_files": (_git("status", "--porcelain") or "").splitlines()}


# ---------------------------------------------------------------------------
# sbatch
# ---------------------------------------------------------------------------
def sbatch(cmd, dry_run, history):
    line = " ".join(shlex.quote(c) for c in cmd)
    with open(history, "a") as f:
        f.write("%s  %s\n" % (datetime.datetime.now().isoformat(timespec="seconds"),
                              line))
    print("  " + line)
    if dry_run:
        return "DRYRUN"
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("submit_slurm: sbatch failed (%d): %s"
                         % (out.returncode, out.stderr.strip()))
    jobid = out.stdout.strip().split(";")[0]
    with open(history, "a") as f:
        f.write("    -> job %s\n" % jobid)
    return jobid


def compress_ranges(indices):
    """[0,1,2,3,7,9,10] -> '0-3,7,9-10' (sbatch --array syntax)."""
    idx, out, i = sorted(indices), [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append(str(idx[i]) if i == j else "%d-%d" % (idx[i], idx[j]))
        i = j + 1
    return ",".join(out)


def submit(slurmdir, indices, sl, dry_run):
    """One array over ``indices`` + one merge job depending on all of it."""
    logs = os.path.join(slurmdir, "logs")
    history = os.path.join(slurmdir, "sbatch_history.txt")
    common = ["-p", sl["partition"], "--account=%s" % sl["account"]]
    if sl["nodelist"]:
        common.append("--nodelist=%s" % sl["nodelist"])
    arr = compress_ranges(indices)
    if sl["max_concurrent"]:
        arr += "%%%d" % sl["max_concurrent"]
    array_id = sbatch(["sbatch", "--parsable"] + common + [
        "--job-name=cmsplot_%s" % sl["label"],
        "--array=%s" % arr,
        "--time=%d" % sl["time"], "--mem=%d" % sl["mem"],
        "--nodes=1", "--ntasks=1", "--cpus-per-task=%d" % sl["cpus"],
        "-o", os.path.join(logs, "task_%A_%a.log"),
        "-e", os.path.join(logs, "task_%A_%a.err"),
        os.path.join(slurmdir, "task.sh")], dry_run, history)
    merge_id = sbatch(["sbatch", "--parsable"] + common + [
        "--job-name=cmsplot_%s_merge" % sl["label"],
        "--dependency=afterany:%s" % array_id,
        "--time=%d" % sl["merge_time"], "--mem=%d" % sl["merge_mem"],
        "--nodes=1", "--ntasks=1", "--cpus-per-task=%d" % sl["merge_cpus"],
        "-o", os.path.join(logs, "merge_%j.log"),
        "-e", os.path.join(logs, "merge_%j.err"),
        os.path.join(slurmdir, "merge.sh")], dry_run, history)
    return array_id, merge_id


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [slurm options] -- <plot.py options>\n"
              "       %(prog)s --resubmit <outdir>/<label>")
    ap.add_argument("--resubmit", metavar="RUNDIR", default=None,
                    help="resubmit the tasks of RUNDIR without a DONE marker, "
                         "plus a new merge job (all other options ignored)")
    ap.add_argument("--branches-per-job", type=int, default=10,
                    help="every task re-reads the selection/weight columns "
                         "(~22 of ~24 columns read for a 2-branch run), so "
                         "this sets how much plotting amortises that read; "
                         "calibrate on the [read] lines of a first run")
    ap.add_argument("--partition", default="short")
    ap.add_argument("--account", default="t3")
    ap.add_argument("--time", type=int, default=60, help="minutes per task")
    ap.add_argument("--mem", type=int, default=6000, help="MB per task")
    ap.add_argument("--cpus", type=int, default=4,
                    help="cpus per task (= plot.py --jobs)")
    ap.add_argument("--nodelist", default="t3wn[80-91]",
                    help="as in the ntuplizer submitters; '' to drop")
    ap.add_argument("--max-concurrent", type=int, default=0,
                    help="array throttle (%%N); 0 = none")
    ap.add_argument("--merge-time", type=int, default=30)
    ap.add_argument("--merge-mem", type=int, default=2000)
    ap.add_argument("--merge-cpus", type=int, default=2,
                    help="also the pdf2png parallelism with --png-from-pdf")
    ap.add_argument("--png-from-pdf", action="store_true",
                    help="tasks save PDFs only (plot.py --formats pdf); the "
                         "merge job renders the PNGs with pdf2png.py")
    ap.add_argument("--xrootd-prefix", default=XROOTD_T3,
                    help="door used by the jobs for /pnfs inputs")
    ap.add_argument("--scratch-base", default="/scratch/$USER",
                    help="expanded on the worker node")
    ap.add_argument("--env-setup", default=None,
                    help="shell line(s) that set up the plotting env on the "
                         "WN (default: recreate the current env)")
    ap.add_argument("--snapshot-extra", nargs="*", default=[],
                    help="extra files the config imports, copied next to it")
    ap.add_argument("--no-proxy", action="store_true",
                    help="do not require/stage X509_USER_PROXY")
    ap.add_argument("--dry-run", action="store_true",
                    help="write everything, print the sbatch lines, submit nothing")
    return ap


def resubmit(rundir, dry_run):
    final = os.path.abspath(rundir)
    slurmdir = os.path.join(final, "_slurm")
    man_path = os.path.join(slurmdir, "submission.json")
    if not os.path.isfile(man_path):
        raise SystemExit("submit_slurm: %s is not a submission (no %s)"
                         % (rundir, man_path))
    if os.path.exists(os.path.join(final, "MERGED")):
        raise SystemExit("submit_slurm: %s is already merged" % rundir)
    man = json.load(open(man_path))
    sl = man["slurm"]
    # never resubmit while a task of this run may still run or be queued: it
    # would be duplicated, and two copies publishing the same DONE race
    try:
        live = subprocess.check_output(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%j %i",
             "-n", "cmsplot_%s,cmsplot_%s_merge" % (sl["label"], sl["label"])],
            stderr=subprocess.DEVNULL).decode().split()
    except Exception:
        live = None
        print("  ! squeue unavailable: make sure no job of this run is queued")
    if live and not dry_run:
        raise SystemExit("submit_slurm: jobs of %s are still queued/running "
                         "(squeue: %s); wait for them" % (sl["label"], live))
    missing = [int(i) for i in sorted(man["tasks"], key=int)
               if not os.path.isfile(os.path.join(slurmdir, "tasks",
                                                  "task%03d" % int(i), "DONE"))]
    if not missing:
        print("[submit_slurm] every task is DONE; resubmitting only the merge")
    else:
        print("[submit_slurm] resubmitting %d task(s): %s"
              % (len(missing), missing))
    if not missing:
        history = os.path.join(slurmdir, "sbatch_history.txt")
        common = ["-p", sl["partition"], "--account=%s" % sl["account"]]
        if sl["nodelist"]:
            common.append("--nodelist=%s" % sl["nodelist"])
        sbatch(["sbatch", "--parsable"] + common + [
            "--job-name=cmsplot_%s_merge" % sl["label"],
            "--time=%d" % sl["merge_time"], "--mem=%d" % sl["merge_mem"],
            "--nodes=1", "--ntasks=1", "--cpus-per-task=%d" % sl["merge_cpus"],
            "-o", os.path.join(slurmdir, "logs", "merge_%j.log"),
            "-e", os.path.join(slurmdir, "logs", "merge_%j.err"),
            os.path.join(slurmdir, "merge.sh")], dry_run, history)
        return
    submit(slurmdir, missing, sl, dry_run)


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        own, plot_argv = argv[:i], argv[i + 1:]
    else:
        own, plot_argv = argv, []
    a = build_parser().parse_args(own)
    if a.resubmit:
        return resubmit(a.resubmit, a.dry_run)
    if not plot_argv:
        raise SystemExit("submit_slurm: give the plot.py options after `--`")

    pparser = _plot.build_parser()
    p = pparser.parse_args(plot_argv)
    if a.png_from_pdf and "--formats" in plot_argv:
        raise SystemExit("submit_slurm: --png-from-pdf sets the task formats "
                         "itself; drop plot.py's --formats")
    if a.png_from_pdf:
        check_converter_in_env()
    formats = ["pdf"] if a.png_from_pdf else list(p.formats)
    if p.xrootd_prefix:
        raise SystemExit("submit_slurm: set the door with submit_slurm's own "
                         "--xrootd-prefix, not plot.py's")
    if p.maxevents > 0:
        try:
            int(p.step_size)
        except ValueError:
            raise SystemExit(
                "submit_slurm: --maxevents with a byte-sized --step-size (%r) "
                "stops each task after a different number of events (chunks "
                "are sized from the columns each task reads), so tasks would "
                "disagree on yields and binning. Pass an entry count, e.g. "
                "--step-size 100000." % p.step_size)

    label = p.label or datetime.datetime.now().strftime("%d%b%Y_%Hh%Mm%Ss")
    final = os.path.abspath(os.path.join(p.outdir, label))
    if final.startswith("/pnfs/"):
        raise SystemExit("submit_slurm: --outdir on /pnfs is not visible from "
                         "the worker nodes; use /work (or another shared FS)")
    if os.path.exists(final) and os.listdir(final):
        raise SystemExit("submit_slurm: %s exists and is not empty; pick a new "
                         "--label (or --resubmit it)" % final)

    # --- the config, as the UI sees it (/pnfs mounted) ---
    cfg = _plot.load_config(p.config)
    samples = list(cfg.samples)
    if p.no_data:
        samples = [s for s in samples if not s.is_data]
    all_files = [f for s in samples for f in s.files]
    need_xrootd = any(f.startswith("/pnfs/") or "://" in f for f in all_files)
    local = sorted({f for f in all_files
                    if not f.startswith("/pnfs/") and "://" not in f})
    if local:
        print("  ! these inputs are read from a local path on the worker "
              "nodes; make sure it is a shared FS:\n    "
              + "\n    ".join(local))
    preflight_imports()
    if need_xrootd:
        check_xrootd_read(samples, a.xrootd_prefix)
    px = None if a.no_proxy else check_proxy(a.time)

    from cmsplot.core import plottable_branches
    if p.datacards_only:
        todo = []
    elif p.branches:
        todo = [b for b in p.branches if b not in set(p.exclude)]
    else:
        print("[submit_slurm] listing plottable branches from the ntuple "
              "schemas ...")
        todo = plottable_branches(samples, getattr(cfg, "DERIVED", None),
                                  exclude=p.exclude)
    if not todo and not p.datacard_branches:
        raise SystemExit("submit_slurm: nothing to do (no branches, no "
                         "datacards)")
    n = a.branches_per_job
    chunks = [todo[i:i + n] for i in range(0, len(todo), n)]

    # --- per-task argv: everything the user asked, minus what we set per task
    per_task_skip = {"config", "outdir", "label", "jobs", "branches",
                     "exclude", "datacard_branches", "datacards_only",
                     "xrootd_prefix", "formats"}
    slurmdir = os.path.join(final, "_slurm")
    os.makedirs(os.path.join(slurmdir, "tasks"))
    os.makedirs(os.path.join(slurmdir, "logs"))
    cfg_snap = snapshot_code(slurmdir, os.path.abspath(p.config),
                             [os.path.abspath(e) for e in a.snapshot_extra])
    base = (["--config", cfg_snap]
            + ns_to_argv(pparser, p, skip=per_task_skip)
            + ["--formats"] + formats
            + (["--xrootd-prefix", a.xrootd_prefix] if a.xrootd_prefix else []))
    tasks = {}
    if p.datacard_branches:
        tasks[len(tasks)] = {"kind": "datacards",
                             "datacard_branches": list(p.datacard_branches),
                             "argv": base + ["--datacards-only",
                                             "--datacard-branches"]
                                          + list(p.datacard_branches)}
    for c in chunks:
        tasks[len(tasks)] = {"kind": "plots", "branches": c,
                             "argv": base + ["--branches"] + c}
    for i, t in tasks.items():
        with open(os.path.join(slurmdir, "tasks", "task%03d.args" % i), "w") as f:
            f.write("\n".join(t["argv"]) + "\n")

    write_scripts(slurmdir, final, label, a, need_xrootd)
    sl = {"label": label, "partition": a.partition, "account": a.account,
          "time": a.time, "mem": a.mem, "cpus": a.cpus, "nodelist": a.nodelist,
          "max_concurrent": a.max_concurrent, "merge_time": a.merge_time,
          "merge_mem": a.merge_mem, "merge_cpus": a.merge_cpus}
    manifest = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
                "label": label, "final": final, "plot_argv": plot_argv,
                "git": git_provenance(), "x509_user_proxy": px,
                "formats": formats, "png_from_pdf": a.png_from_pdf,
                "slurm": sl, "tasks": tasks}
    with open(os.path.join(slurmdir, "submission.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print("[submit_slurm] %d branches in %d plot task(s)%s -> %s"
          % (len(todo), len(chunks),
             " + 1 datacard task" if p.datacard_branches else "", final))
    submit(slurmdir, sorted(tasks), sl, a.dry_run)
    if a.dry_run:
        print("[submit_slurm] dry run: nothing submitted. Try one task on the "
              "UI with\n  SLURM_ARRAY_TASK_ID=1 SLURM_ARRAY_JOB_ID=local "
              "SLURM_JOB_ID=local SLURM_CPUS_PER_TASK=%d bash %s/task.sh"
              % (a.cpus, slurmdir))


if __name__ == "__main__":
    main()
