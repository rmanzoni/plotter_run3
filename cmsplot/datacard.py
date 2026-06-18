"""Combine datacard + shape-file writer for template binned ML fits.

For each requested branch this writes, into ``<outdir>/<label>/datacards/``:
  * ``<branch>.root`` -- one TH1 per datacard process (with proper Sumw2) plus
    ``data_obs``;
  * ``<branch>.txt``  -- a CMS Combine datacard (one channel = the branch).

Processes are grouped by the ``datacard`` tag carried on each Sample/Process:
  "Bc", "Hb", "misID"  -> stacked templates (all reads sharing a tag are summed,
                          so e.g. every gen_bc_decay component of bc.root lands
                          in one "Bc" template);
  "data_obs"           -> the observation;
  ""                   -> excluded from the datacard (e.g. `combinatorial`).

Templates are built from the NOMINAL weights (before any scale-to-data), so the
fit is free to determine the normalisations:

  * all Bc contributions (every template originating from the ``bc`` sample,
    the signal jpsi_tau included) are tied to ONE shared, unconstrained overall
    normalisation ``bc_norm`` -- so the whole Bc cocktail moves together rather
    than each component floating on its own;
  * the signal (jpsi_tau) additionally carries the default POI ``r`` from
    text2workspace, so its yield scales as ``r * bc_norm``. ``bc_norm`` is fixed
    by the mu channel + cocktail, ``r`` is the tau-only excess on top of it, i.e.
    the tau/mu ratio R(J/psi) (its absolute calibration follows the template
    normalisation: r = R(J/psi)/R_SM if the tau template carries the SM BF, r =
    R(J/psi) directly if it is normalised to the mu BF);
  * each non-signal Bc contribution gets an independent ``lnN`` (default 10%);
  * the non-Bc backgrounds (Hb, misID) keep their own free ``rateParam``;
  * ``autoMCStats`` covers bin-by-bin MC statistics.

In the collapsed/merged card (a single "Bc" template = signal) there is no
second Bc process to tie, so ``bc_norm`` is omitted and the signal falls back to
``r`` alone (legacy behaviour).

The Bc set, the lnN size, and the bc_norm range can be overridden from the
``datacard_def`` config via the optional keys ``bc_sample`` (default "bc"),
``bc_lnn`` (default 0.10), ``bc_lnn_skip`` (tuple of Bc tags to leave without a
lnN, e.g. ("jpsi_mu",) to keep the mu channel an uncertainty-free reference),
``bc_norm_name`` and ``bc_norm_range``.
"""
import os
import numpy as np


def _to_th1(name, sw, sw2, edges):
    """Build an uproot-writable TH1 carrying explicit bin contents and Sumw2."""
    from uproot.writing import identify
    sw = np.asarray(sw, "float64")
    sw2 = np.asarray(sw2, "float64")
    edges = np.asarray(edges, "float64")
    # ROOT TH1 storage includes under/overflow -> pad both ends with zeros
    cont = np.concatenate([[0.0], sw, [0.0]])
    sumw2 = np.concatenate([[0.0], sw2, [0.0]])
    return identify.to_TH1x(
        fName=name, fTitle=name, data=cont,
        fEntries=float(np.sum(sw != 0.0)),
        fTsumw=float(sw.sum()), fTsumw2=float(sw2.sum()),
        fTsumwx=0.0, fTsumwx2=0.0, fSumw2=sumw2,
        fXaxis=identify.to_TAxis(fName="xaxis", fTitle="", fNbins=sw.size,
                                 fXmin=float(edges[0]), fXmax=float(edges[-1]),
                                 fXbins=edges))


def _resolve_data_obs(procs):
    """Pick the observation processes. Explicit datacard='data_obs' wins; else a
    single is_data process is taken by convention; ambiguity is an error."""
    tagged = [p for p in procs if p.is_data and p.datacard == "data_obs"]
    if tagged:
        return tagged
    untagged = [p for p in procs if p.is_data and not p.datacard]
    if len(untagged) == 1:
        return untagged
    if len(untagged) == 0:
        return []                                   # Asimov fallback upstream
    raise ValueError(
        "multiple data samples present and none tagged datacard='data_obs': %s. "
        "Tag exactly one (others get datacard='' to exclude, e.g. combinatorial)."
        % [p.key for p in untagged])


def _mc_tags(procs, signal, etag=None):
    """Ordered list of MC datacard process names, signal first.

    ``etag(p)`` returns the effective datacard tag for a process (defaults to
    ``p.datacard``), so a config-driven composition is honoured here too.
    """
    if etag is None:
        etag = lambda p: p.datacard
    tags = []
    for p in procs:
        t = etag(p)
        if not p.is_data and t and t != "data_obs":
            if t not in tags:
                tags.append(t)
    if signal not in tags:
        raise ValueError("datacard signal %r not among MC processes %s "
                         "(check the `datacard=` tags / DATACARD config / "
                         "--datacard-signal)" % (signal, tags))
    tags.remove(signal)
    return [signal] + tags                          # signal -> process id 0


def _norm_sel(sel):
    """Normalise a contribution selector into a dict.

    Accepted shorthands (all equivalent to the dict on the right):
      "bc"              -> {"sample": "bc"}                (whole sample)
      ("bc", [1, 7])    -> {"sample": "bc", "codes": [1, 7]}
      {"sample": "bc", "codes": [1]}                       (canonical)
      {"group": "misid"}                                   (match by stack group)
    """
    if isinstance(sel, str):
        return {"sample": sel}
    if isinstance(sel, (tuple, list)):
        d = {"sample": sel[0]}
        if len(sel) > 1 and sel[1] is not None:
            d["codes"] = list(sel[1])
        return d
    if isinstance(sel, dict):
        return sel
    raise TypeError("unrecognised datacard selector: %r" % (sel,))


def _selector_matches(p, sel):
    """Does Process ``p`` belong to this selector?

    ``group`` matches the stack-merge tag. ``sample`` matches the originating
    Sample.name; with ``codes`` it additionally requires the process' gen codes
    to be a subset of the listed codes (so datacard granularity must be at least
    as coarse as the plotting split -- you cannot subdivide an already-merged
    component). Without ``codes`` every process from that sample matches.
    """
    if sel.get("group"):
        return p.group == sel["group"]
    smp = sel.get("sample")
    if smp is None or getattr(p, "sample_name", "") != smp:
        return False
    codes = sel.get("codes")
    if codes is not None:
        sc = set(getattr(p, "split_codes", ()) or ())
        if not sc or not sc.issubset(set(int(c) for c in codes)):
            return False
    return True


def resolve_datacard_tags(procs, datacard_def, verbose=True):
    """Map each MC Process to a datacard process name per ``datacard_def``.

    ``datacard_def`` is a dict::

        {"signal": "<process name>",
         "processes": OrderedDict(name -> [selector, selector, ...])}

    The first datacard process whose selector list matches a Process wins; a
    Process matching nothing is tagged "" (excluded from the datacard). Returns
    ``(tag_of, signal)`` where ``tag_of`` is ``{id(p): tag}`` for non-data
    processes (data keeps its own ``datacard`` tag for the observation).
    """
    spec = datacard_def.get("processes", datacard_def)
    order = list(spec.items())
    signal = datacard_def.get("signal")
    tag_of, unmatched = {}, []
    for p in procs:
        if p.is_data:
            continue
        chosen = ""
        for name, selectors in order:
            sels = selectors if isinstance(selectors, (list, tuple)) else [selectors]
            if any(_selector_matches(p, _norm_sel(s)) for s in sels):
                chosen = name
                break
        tag_of[id(p)] = chosen
        if not chosen:
            unmatched.append(p.label)
    if verbose and unmatched:
        print("  ! datacard composition: %d process(es) matched no datacard "
              "process and are EXCLUDED: %s" % (len(unmatched), unmatched))
    return tag_of, signal


def write_datacards(procs, hg, branches, outdir, label, nbins=40,
                    signal="Bc", overflow=False, floor=1e-6,
                    rate_param_range=(0.0, 10.0), verbose=True,
                    datacard_def=None, bc_sample="bc",
                    bc_norm_name="bc_norm", bc_norm_range=(0.0, 10.0),
                    bc_lnn=0.10, bc_lnn_skip=()):
    """Write a shape file + datacard per branch. Returns the datacards dir.

    If ``datacard_def`` is given it re-composes the datacard processes from the
    loaded Processes (e.g. splitting the Bc cocktail into separate templates)
    and may override ``signal``; otherwise the per-Sample ``datacard=`` tags are
    used unchanged.
    """
    import uproot
    from .core import _hist                         # lazy: avoids import cycle

    dcdir = os.path.join(outdir, label, "datacards")
    os.makedirs(dcdir, exist_ok=True)

    # effective datacard tag per process (config-driven override, or the
    # per-sample tag). data_obs is resolved separately and never overridden.
    if datacard_def:
        tag_of, sig = resolve_datacard_tags(procs, datacard_def, verbose=verbose)
        if sig:
            signal = sig
        # optional Bc-normalisation overrides carried on the same config dict
        bc_sample = datacard_def.get("bc_sample", bc_sample)
        bc_norm_name = datacard_def.get("bc_norm_name", bc_norm_name)
        bc_norm_range = tuple(datacard_def.get("bc_norm_range", bc_norm_range))
        bc_lnn = datacard_def.get("bc_lnn", bc_lnn)
        bc_lnn_skip = tuple(datacard_def.get("bc_lnn_skip", bc_lnn_skip))
    else:
        tag_of = {}

    def etag(p):
        return tag_of.get(id(p), p.datacard)

    obs_procs = _resolve_data_obs(procs)
    tags = _mc_tags(procs, signal, etag)
    lo, hi = rate_param_range

    # Which datacard tags are Bc contributions? A tag is "Bc" if any process
    # feeding it originates from the `bc` sample. These are tied together by the
    # single shared `bc_norm` parameter in the card.
    bc_tags = set()
    for p in procs:
        if p.is_data:
            continue
        t = etag(p)
        if t in tags and getattr(p, "sample_name", "") == bc_sample:
            bc_tags.add(t)
    if signal not in bc_tags and verbose:
        print("  ! datacard: signal %r is not a Bc-sample process; it will not "
              "be tied to %s" % (signal, bc_norm_name))

    n_ok, n_skip, n_fail = 0, 0, 0
    for branch in branches:
      try:
        present = [p for p in procs if branch in p.arrays]
        if not any((not p.is_data and etag(p) in tags) for p in present):
            if verbose:
                print("  ! datacard %s: no MC templates for this branch, skip"
                      % branch)
            n_skip += 1
            continue
        edges = hg.edges_for(branch, nbins=nbins)

        # --- accumulate templates by datacard tag --------------------------
        sw = {t: np.zeros(len(edges) - 1) for t in tags}
        sw2 = {t: np.zeros(len(edges) - 1) for t in tags}
        for p in present:
            t = etag(p)
            if p.is_data or t not in tags:
                continue
            s, s2 = _hist(p.arrays[branch], p.weight, edges, overflow=overflow)
            sw[t] += s
            sw2[t] += s2

        # floor non-positive expected yields (Combine requires >= 0)
        n_floored = {}
        for t in tags:
            bad = sw[t] <= 0.0
            n_floored[t] = int(bad.sum())
            sw[t][bad] = floor

        # --- observation ---------------------------------------------------
        if obs_procs:
            d_sw = np.zeros(len(edges) - 1)
            for p in obs_procs:
                if branch in p.arrays:
                    s, _ = _hist(p.arrays[branch], p.weight, edges,
                                 overflow=overflow)
                    d_sw += s
            obs_is_asimov = False
        else:                                        # no data -> Asimov dataset
            d_sw = np.sum([sw[t] for t in tags], axis=0)
            obs_is_asimov = True

        # --- write shape file ---------------------------------------------
        root_path = os.path.join(dcdir, "%s.root" % branch)
        with uproot.recreate(root_path) as f:
            for t in tags:
                f[t] = _to_th1(t, sw[t], sw2[t], edges)
            # data_obs: integer-ish counts; variance = counts
            f["data_obs"] = _to_th1("data_obs", d_sw, d_sw, edges)

        # --- write datacard text ------------------------------------------
        _write_card(os.path.join(dcdir, "%s.txt" % branch), branch, tags, sw,
                    d_sw, signal, lo, hi, obs_is_asimov,
                    bc_tags=bc_tags, bc_norm_name=bc_norm_name,
                    bc_norm_range=bc_norm_range, bc_lnn=bc_lnn,
                    bc_lnn_skip=bc_lnn_skip)
        n_ok += 1

        if verbose:
            fl = ", ".join("%s:%d" % (t, n_floored[t]) for t in tags
                           if n_floored[t])
            print("  datacard %-22s -> %s.{root,txt}  obs=%d%s%s"
                  % (branch, branch, int(round(d_sw.sum())),
                     " (Asimov)" if obs_is_asimov else "",
                     ("  floored[" + fl + "]") if fl else ""))
      except Exception as e:
        # never let one bad branch abort the loop and leave LATER branches'
        # datacards stale on disk (a silent plot/datacard mismatch). Remove any
        # half-written file for this branch so a stale copy can't be mistaken
        # for a fresh one, and carry on.
        n_fail += 1
        for ext in ("root", "txt"):
            stale = os.path.join(dcdir, "%s.%s" % (branch, ext))
            try:
                os.remove(stale)
            except OSError:
                pass
        print("  ! datacard %s FAILED (%s) -- removed any stale file, continuing"
              % (branch, e))

    if verbose:
        print("[cmsplot] datacards: %d written, %d skipped, %d failed"
              % (n_ok, n_skip, n_fail))
    return dcdir


def _write_card(path, channel, tags, sw, d_sw, signal, lo, hi, asimov,
                bc_tags=(), bc_norm_name="bc_norm", bc_norm_range=(0.0, 10.0),
                bc_lnn=0.10, bc_lnn_skip=()):
    nbkg = len(tags) - 1
    procid = {t: (0 if t == signal else i)
              for i, t in enumerate([signal] + [t for t in tags if t != signal])}

    bc_set = set(bc_tags)
    share_bc = len(bc_set) > 1            # a shared Bc norm needs >1 Bc process
    skip = set(bc_lnn_skip)
    lnn_tags = ([t for t in tags
                 if t in bc_set and t != signal and t not in skip]
                if (share_bc and bc_lnn) else [])
    bnlo, bnhi = bc_norm_range

    rate_strs = {t: "%.4f" % sw[t].sum() for t in tags}

    # Left label field: wide enough for the widest row label, *including* the
    # systematic rows (which carry "<name> lnN" in that field). Value columns
    # must in turn exceed the widest data cell, otherwise %*s right-justifies
    # with zero leading space and adjacent cells run together -- e.g. three
    # "m_miss2_jpsi" bin cells printing as one unparseable blob.
    syst_names = [t + "_norm_unc" for t in lnn_tags]
    namew = max([len(x) for x in
                 (["observation", "process", "rate", "bin"]
                  + list(tags) + syst_names)])
    lead = max(14, namew + 5)             # room for "<name> lnN"
    widest = max([len(channel)]
                 + [len(t) for t in tags]
                 + [len(str(procid[t])) for t in tags]
                 + [len(rate_strs[t]) for t in tags])
    col = max(12, widest + 2)             # +2 guarantees >=2 spaces between cells

    def row(cells):
        return ("%-*s" % (lead, cells[0])) + "".join("%*s" % (col, c)
                                                      for c in cells[1:])

    def syst_row(name, typ, cells):
        return row(["%-*s %s" % (namew, name, typ)] + cells)

    lines = []
    lines.append("# Combine datacard generated by cmsplot for branch '%s'"
                 % channel)
    if asimov:
        lines.append("# NOTE: no data sample -> data_obs is the Asimov "
                     "(sum of MC templates)")
    lines.append("imax 1  number of channels")
    lines.append("jmax %d  number of backgrounds" % nbkg)
    lines.append("kmax *  number of nuisance parameters")
    lines.append("-" * 80)
    lines.append("shapes * %s %s.root $PROCESS $PROCESS_$SYSTEMATIC"
                 % (channel, channel))
    lines.append("-" * 80)
    lines.append(row(["bin", channel]))
    lines.append(row(["observation", "%d" % int(round(d_sw.sum()))]))
    lines.append("-" * 80)
    lines.append(row(["bin"] + [channel] * len(tags)))
    lines.append(row(["process"] + list(tags)))
    lines.append(row(["process"] + [str(procid[t]) for t in tags]))
    lines.append(row(["rate"] + [rate_strs[t] for t in tags]))
    lines.append("-" * 80)

    # (1) ONE shared, unconstrained overall Bc normalisation. Repeating the SAME
    #     parameter name on every Bc process ties them to a single factor; the
    #     signal (jpsi_tau) additionally carries the default POI 'r' from
    #     text2workspace, so its yield scales as r * bc_norm -- bc_norm is the
    #     common Bc rate, r is the tau/mu ratio R(J/psi).
    if share_bc:
        for t in tags:
            if t in bc_set:
                lines.append("%-13s rateParam %s %s 1 [%g,%g]"
                             % (bc_norm_name, channel, t, bnlo, bnhi))

    # (2) every non-Bc background keeps its own free rateParam. In the collapsed
    #     card (single Bc = signal) the Bc also falls through to r alone.
    for t in tags:
        if t == signal:
            continue
        if share_bc and t in bc_set:
            continue
        lines.append("%-13s rateParam %s %s 1 [%g,%g]"
                     % (t + "_norm", channel, t, lo, hi))

    # (3) each non-signal Bc contribution gets an independent lnN. The signal
    #     jpsi_tau is governed by r and bc_norm only (no extra BR nuisance).
    if lnn_tags:
        kappa = "%.3f" % (1.0 + bc_lnn)
        for t in lnn_tags:
            cells = [kappa if u == t else "-" for u in tags]
            lines.append(syst_row(t + "_norm_unc", "lnN", cells))

    lines.append("* autoMCStats 0")
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
