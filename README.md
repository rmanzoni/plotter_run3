# cmsplot — fast stack plotter for the R(J/ψ) ntuples

uproot + numpy + mplhep. Reads each ntuple **once** into memory, auto-guesses a
binning for every branch, and writes CMS-styled stacked MC-vs-data plots (with a
ratio panel) in `png` + `pdf`, `lin` + `log`.

## Install (plotting env, not CMSSW/FWLite)
```bash
pip3 install uproot mplhep numpy matplotlib
```

## Run
```bash
python3 plot.py --config samples_rjpsi.py                 # all branches
python3 plot.py --config samples_rjpsi.py --branches mass mcorr q2 --jobs 8
python3 plot.py --config samples_rjpsi.py --no-data --normalize   # shape check
```
Output goes to `plots/<label>/{png,pdf}/{lin,log}/<branch>.{png,pdf}` plus
`yields.txt`. `<label>` defaults to a timestamp.

## Configure
Edit `samples_rjpsi.py`. A `Sample` is:
```python
Sample(name, files, label="", is_data=False, is_signal=False, color=None,
       scale=1.0, weight_branches=[], selection="", split_by=None,
       split_map={}, split_default=None, group="", tree="tree")
```
- `scale` is the global norm (lumi·σ/N_gen); `weight_branches` are per-event
  columns multiplied in (e.g. `["puWeight", "ctau_weight_central"]`).
- `selection` is a numpy-evaluable string on the branches, e.g.
  `"(mass>6.0)&(mass<6.6)"` (trusted input — it is `eval`'d, with `np` available).

### Cocktail components (Bc)
`split_by="gen_bc_decay"` + `split_map={code:(label,color,is_signal)}` fans the
sample by gen code. **Several codes that share the same `label` are merged into
one stacked process** — that is how the 22 `BcGenDecay` channels collapse into
physics components. `split_default` catches any unmapped code (e.g. the `-1`
"unknown Bc"). The shipped grouping:

| component | codes | |
|---|---|---|
| `J/ψ μν` (norm) | 1 | |
| `J/ψ τν` (signal) | 7 | hatched, on top |
| feed-down `(ψ′,χc,hc)ℓν` | 2,3,4,5,6,8 | |
| `J/ψ + D(s)` | 13–20 | |
| `J/ψ + hadrons` | 9,10,11,12,21,22 (+ −1) | |

Re-bucket by editing the code lists in `COMPONENTS`.

### Bc / Hb partition
`gen_bc_decay`: 1–22 real Bc channels, 0 = no Bc, −1 = Bc-but-unclassified.
The two MC sources tile the J/ψ-from-b phase space without overlap:
- Bc sample keeps `KEEP_BC = "(gen_bc_decay>=1)|(gen_bc_decay==-1)"`;
- Hb sample keeps `EXCLUDE_BC = "(gen_bc_decay==0)|(gen_bc_decay!=gen_bc_decay)"`
  (i.e. code 0 or NaN), so every Bc event is removed from Hb.

### Computed weights and shape systematics
`weight_factors={name: WeightFactor(inputs, nominal, variations)}` multiplies
computed per-event factors into the nominal weight (plots, yields, templates).
Each `variations` entry `{nuisance: (up, down)}` replaces that factor's nominal
and becomes a Combine `shape` nuisance with `<process>_<nuisance>{Up,Down}`
templates, attached only to the datacard processes it actually moves. Samples
declaring the same nuisance name are fully correlated (e.g. `bc` and the Bc
subtraction inside misID). The callables must return finite factors or raise.
In `samples_rjpsi.py`, `BC_WEIGHT_FACTORS` = Hammer FF (`ff_ev00..14`) x Bc
lifetime (`bc_ctau`).

### Cross-sample grouping
`group="hb"` on both `hb1` and `hb2` sums them into a single stacked "Hb" entry
(and one line in `yields.txt`), while each keeps its own `scale`.

## Auto-ranging
Per branch: integer-valued → unit bins; `cos*`/`*prob`/`phi` → physical bounds;
everything else → Tukey/IQR robust range (q1−3·IQR, q3+3·IQR clipped to the data
support). This keeps the heavy-tailed IP-significance branches readable without
manual limits. Over/underflow is folded into the first/last bin. NaN/inf are
dropped. Override `--bins`, or pass `--branches` to restrict, `--exclude` to skip.

## Notes
- `com` defaults from the config (`COM`); set `LUMI` (fb⁻¹) once unblinded — with
  no data / no lumi the label switches to *Simulation*.
- Colours follow the CMS guidelines (Petroff CVD-safe 6/8/10 sequences).
- Reading is threaded across samples (`--jobs`); uproot releases the GIL while
  decompressing, so this scales without pickling overhead.
- Branches present only in MC (gen-level) are drawn MC-only (no data/ratio).
