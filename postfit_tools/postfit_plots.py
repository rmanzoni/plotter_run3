#!/usr/bin/env python
"""
Pre-fit / post-fit plots from a combine FitDiagnostics output file.

Reads the shapes_* directories written by e.g.
    combine -M FitDiagnostics <datacard> --saveShapes --saveNormalizations --plots
and draws, per channel, a stacked histogram of the fitted processes with the
data overlaid and a data/model ratio panel.

Usage:
    python postfit_plots.py [fitDiagnosticsTest.root] [outdir]

Note on the uncertainty band: a meaningful post-fit total uncertainty requires
--saveWithUncertainties at the FitDiagnostics step. If it was not passed, the
bin errors on `total` are not the fit uncertainty, so the band is omitted
automatically (the script detects this).
"""

import os
import sys
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

# ---- config -----------------------------------------------------------------
INFILE = sys.argv[1] if len(sys.argv) > 1 else "fitDiagnosticsTest.root"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "postfit_plots"

# which fits to plot: directory name -> human label
FITS = {
    "shapes_prefit": "Pre-fit",
    "shapes_fit_s":  "Post-fit (S+B)",
    # "shapes_fit_b": "Post-fit (B-only)",   # uncomment to add
}

# optional: fix colours per process name; unlisted procs get auto colours.
PROC_STYLE = {
    # "signal":   ROOT.kRed + 1,
    # "bkg_comb": ROOT.kAzure + 1,
}
PALETTE = [ROOT.kAzure + 1, ROOT.kOrange + 1, ROOT.kGreen + 2, ROOT.kRed + 1,
           ROOT.kViolet + 1, ROOT.kCyan + 2, ROOT.kYellow + 1, ROOT.kGray + 1,
           ROOT.kMagenta + 1, ROOT.kSpring + 4]
# -----------------------------------------------------------------------------


def get_channels(d):
    """Channel subdir names, or [None] if shapes sit directly in d (single channel)."""
    chans = [k.GetName() for k in d.GetListOfKeys()
             if k.ReadObj().InheritsFrom("TDirectory")]
    return chans if chans else [None]


def get_subdir(f, fitdir, channel):
    path = fitdir if channel is None else f"{fitdir}/{channel}"
    return f.Get(path)


def collect_processes(d):
    """TH1 process templates in d (excluding total*), sorted descending by integral."""
    procs = {}
    for k in d.GetListOfKeys():
        obj = k.ReadObj()
        name = k.GetName()
        if obj.InheritsFrom("TH1") and not name.startswith("total"):
            procs[name] = obj
    return dict(sorted(procs.items(), key=lambda kv: kv[1].Integral(), reverse=True))


def has_band(htot):
    """True if `total` carries non-trivial bin errors (i.e. --saveWithUncertainties was used)."""
    return any(htot.GetBinError(i) > 0 for i in range(1, htot.GetNbinsX() + 1))


def make_ratio_graph(gdata, htot):
    """data/model ratio as a TGraphAsymmErrors (errors propagated from data only)."""
    gr = ROOT.TGraphAsymmErrors()
    j = 0
    for i in range(gdata.GetN()):
        x = gdata.GetPointX(i)
        y = gdata.GetPointY(i)
        m = htot.GetBinContent(htot.FindBin(x))
        if m <= 0:
            continue
        gr.SetPoint(j, x, y / m)
        gr.SetPointError(j, gdata.GetErrorXlow(i), gdata.GetErrorXhigh(i),
                         gdata.GetErrorYlow(i) / m, gdata.GetErrorYhigh(i) / m)
        j += 1
    return gr


def make_band_ratio(htot):
    """Relative uncertainty band (total/total) centred at 1 for the ratio panel."""
    h = htot.Clone(htot.GetName() + "_relband")
    for i in range(1, h.GetNbinsX() + 1):
        c = htot.GetBinContent(i)
        h.SetBinContent(i, 1.0)
        h.SetBinError(i, htot.GetBinError(i) / c if c > 0 else 0.0)
    return h


def plot_channel(f, fitdir, label, channel, outpath):
    d = get_subdir(f, fitdir, channel)
    if not d:
        print(f"  [skip] {fitdir}/{channel}: directory not found")
        return

    procs = collect_processes(d)
    htot = d.Get("total")
    gdata = d.Get("data")
    if not procs or not htot:
        print(f"  [skip] {fitdir}/{channel}: no processes or no total")
        return

    keep = []  # keep python refs alive past the loop

    # --- stack ---
    stack = ROOT.THStack("stack", "")
    leg = ROOT.TLegend(0.62, 0.58, 0.92, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    for i, (name, h) in enumerate(procs.items()):
        h.SetFillColor(PROC_STYLE.get(name, PALETTE[i % len(PALETTE)]))
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(1)
        stack.Add(h)
        keep.append(h)
    for name, h in reversed(list(procs.items())):  # legend top entry = top of stack
        leg.AddEntry(h, name, "f")

    drawband = has_band(htot)
    if drawband:
        htot.SetFillColor(ROOT.kBlack)
        htot.SetFillStyle(3354)
        htot.SetMarkerSize(0)
        leg.AddEntry(htot, "Uncertainty", "f")

    if gdata:
        gdata.SetMarkerStyle(20)
        gdata.SetMarkerSize(0.9)
        gdata.SetLineColor(ROOT.kBlack)
        leg.AddEntry(gdata, "Data", "pe")

    # --- canvas with ratio pad ---
    c = ROOT.TCanvas("c", "", 700, 750)
    p1 = ROOT.TPad("p1", "", 0, 0.30, 1, 1.0)
    p2 = ROOT.TPad("p2", "", 0, 0.0, 1, 0.30)
    p1.SetBottomMargin(0.02)
    p2.SetTopMargin(0.04)
    p2.SetBottomMargin(0.32)
    p1.Draw()
    p2.Draw()

    # top pad
    p1.cd()
    data_max = max((gdata.GetPointY(i) for i in range(gdata.GetN())), default=0.0) if gdata else 0.0
    stack.SetMaximum(1.5 * max(stack.GetMaximum(), data_max))
    stack.Draw("hist")
    stack.GetYaxis().SetTitle("Events")
    stack.GetYaxis().SetTitleSize(0.05)
    stack.GetXaxis().SetLabelSize(0)
    if drawband:
        htot.Draw("E2 same")
    if gdata:
        gdata.Draw("P same")
    leg.Draw()

    title = channel if channel else "combined"
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.SetTextSize(0.045)
    txt.DrawLatex(0.13, 0.92, f"{label}  -  {title}")

    # bottom pad (ratio)
    p2.cd()
    frame = htot.Clone("frame")
    frame.Reset()
    frame.GetYaxis().SetRangeUser(0.8, 1.2)
    frame.GetYaxis().SetTitle("Data / Fit")
    frame.GetYaxis().SetNdivisions(505)
    frame.GetYaxis().SetTitleSize(0.11)
    frame.GetYaxis().SetTitleOffset(0.45)
    frame.GetYaxis().SetLabelSize(0.09)
    frame.GetXaxis().SetTitle(htot.GetXaxis().GetTitle() or "Bin")
    frame.GetXaxis().SetTitleSize(0.12)
    frame.GetXaxis().SetLabelSize(0.10)
    frame.Draw("hist")

    if drawband:
        rb = make_band_ratio(htot)
        rb.SetFillColor(ROOT.kBlack)
        rb.SetFillStyle(3354)
        rb.SetMarkerSize(0)
        rb.Draw("E2 same")
        keep.append(rb)

    line = ROOT.TLine(frame.GetXaxis().GetXmin(), 1.0, frame.GetXaxis().GetXmax(), 1.0)
    line.SetLineStyle(2)
    line.Draw("same")

    if gdata:
        rg = make_ratio_graph(gdata, htot)
        rg.SetMarkerStyle(20)
        rg.SetMarkerSize(0.9)
        rg.Draw("P same")
        keep.append(rg)

    c.SaveAs(outpath)
    print(f"  wrote {outpath}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    f = ROOT.TFile.Open(INFILE)
    if not f or f.IsZombie():
        raise SystemExit(f"cannot open {INFILE}")

    for fitdir, label in FITS.items():
        d = f.Get(fitdir)
        if not d:
            print(f"[skip] {fitdir} not in file")
            continue
        print(f"[{fitdir}]  ({label})")
        for ch in get_channels(d):
            outpath = os.path.join(OUTDIR, f"{fitdir}_{ch or 'combined'}.pdf")
            plot_channel(f, fitdir, label, ch, outpath)

    f.Close()


if __name__ == "__main__":
    main()
