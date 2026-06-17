import ROOT
f = ROOT.TFile.Open("fitDiagnosticsTest.root")

for setname in ["norm_prefit", "norm_fit_b", "norm_fit_s"]:
    norms = f.Get(setname)
    if not norms:
        print(f"{setname}: not found")
        continue
    print(f"\n=== {setname} ===")
    for var in norms:                      # RooArgSet is directly iterable now
        print(f"{var.GetName():35s} {var.getVal():12.4f} +/- {var.getError():.4f}")



