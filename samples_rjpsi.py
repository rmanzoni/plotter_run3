"""Sample configuration for the R(J/psi) analysis.

    python3 plot.py --config samples_rjpsi.py

Two MC sources plus data:
  * Bc cocktail  -> split into physics components via gen_bc_decay (1..22);
  * Hb (hb1/hb2) -> inclusive b-hadron -> J/psi, with the Bc fraction REMOVED
                    so it is not double-counted against the dedicated Bc sample.

gen_bc_decay convention (BcGenDecay / RJPsiGenHistory): 1..22 = real Bc channels,
0 = no Bc in the event, -1 = Bc found but channel unrecognised.
"""
import numpy as np
from collections import OrderedDict
from cmsplot import Sample
from cmsplot.style import PETROFF_10 as P

# --- run conditions -----------------------------------------------------------
COM = 13.6            # Run 2 MC = 13 TeV; switch to 13.6 for Run 3
LUMI = None           # fb^-1 once you compare to data; None -> Simulation label
EXTRA = "Preliminary"

# NTUPLE_DIR = "/pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_ntuples"  # EDIT
NTUPLE_DIR = "/Users/manzoni/Documents/rjpsi_run3/ntuples/15jun26"  # EDIT

# --- global MC normalisations -------------------------------------------------
# (2) Tune the absolute Bc and Hb yields here. lumi * sigma / N_gen, times any
#     k-factor / data-driven scale you want. These set the Bc:Hb *ratio*.
BC_SCALE = 1.0
HB_SCALE = 1.0        # applied to both hb1 and hb2 (each keeps its own below if needed)

# (4) Fix the TOTAL MC (Bc+Hb) to the data yield, preserving the Bc:Hb ratio
#     above. Equivalent to the --scale-to-data flag.
SCALE_TO_DATA = False

# --- manual binning overrides -------------------------------------------------
# (3) Per-branch binning. Anything not listed is auto-ranged.
#     tuple (nbins, lo, hi) -> uniform bins;  list [...] -> explicit edges.
BINNING = {
#     "mass":  (40, 6.0, 6.6),
#     "q2":    (30, 0.0, 11.0),

    "mass":  (40, 2.5, 8),

    "q2_jpsi"                    : (40, -10, 12),
    "q2_sv"                      : (40, -10, 12),
    "m_miss2_jpsi"               : (40, -10, 10),
    "m_miss2_sv"                 : (40, -10, 10),

    "nu1_q2_jpsi"                : (40,   0, 12),
    "nu2_q2_jpsi"                : (40,   0, 12),
    "nu1_q2_sv"                  : (40,   0, 12),
    "nu2_q2_sv"                  : (40,   0, 12),

    "mu_ip3d_jpsi_pv"            : (40, - 0.1, 0.1 ),
    "mu_ip3d_jpsi_pv_err"        : (40,   0  , 0.02),
    "mu_ip3d_jpsi_pv_sig"        : (50, - 5  , 10  ),

    "mu_ip3d_jpsi_sv"            : (40, - 0.1, 0.1 ),
    "mu_ip3d_jpsi_sv_err"        : (40,   0  , 0.02),
    "mu_ip3d_jpsi_sv_sig"        : (50, - 5  , 5   ),

    "mu_ip3d_sv_pv"              : (40, - 0.1, 0.1 ),
    "mu_ip3d_sv_pv_err"          : (40,   0  , 0.02),
    "mu_ip3d_sv_pv_sig"          : (50, - 5  , 10  ),

    "mu_ip3d_sv_sv"              : (40, - 0.1, 0.1 ),
    "mu_ip3d_sv_sv_err"          : (40,   0  , 0.02),
    "mu_ip3d_sv_sv_sig"          : (50, - 5  , 5   ),

    "mu_dist_to_b_dir_jpsi"      : (40,   0  , 0.03),
    "mu_dist_to_b_dir_jpsi_err"  : (40,   0  , 0.02),
    "mu_dist_to_b_dir_jpsi_sig"  : (50, - 5  , 5   ),

    "mu_dist_to_b_dir_sv"        : (40,   0  , 0.03),
    "mu_dist_to_b_dir_sv_err"    : (40,   0  , 0.02),
    "mu_dist_to_b_dir_sv_sig"    : (50, - 5  , 5   ),

    "mu_dist_along_b_dir_jpsi_pv": (40,  0, 0.6),
    "mu_dist_along_b_dir_jpsi_sv": (40,  0, 0.4),
    "mu_dist_along_b_dir_sv_pv"  : (40,  0, 0.6),
    "mu_dist_along_b_dir_sv_sv"  : (40,  0, 0.4),

    "lxy"     :  np.logspace(-4, np.log10(2), 40),   # variable-width example
    "jpsi_lxy":  np.logspace(-4, np.log10(2), 40),   # variable-width example
}

# =============================================================================
# Bc cocktail components.  Each component = (label, colour, is_signal, [codes]).
# Channel codes come straight from BC_CHANNELS in RJPsiGenHistory:
#   1 Jpsi_mu_nu   2 psi2S_mu_nu  3 chic0_mu_nu  4 chic1_mu_nu  5 chic2_mu_nu
#   6 hc_mu_nu     7 Jpsi_tau_nu  8 psi2S_tau_nu 9 Jpsi_pi     10 Jpsi_3pi
#  11 Jpsi_5pi    12 Jpsi_K      13 Jpsi_Ds     14 Jpsi_Dsstar 15 Jpsi_D0bar_K
#  16 Jpsi_D0starbar_K  17 Jpsi_Dstar_Kstar  18 Jpsi_D_Kstar
#  19 Jpsi_D      20 Jpsi_Dstar  21 Jpsi_p_pbar_pi  22 Jpsi_K_K_pi
# =============================================================================
COMPONENTS = OrderedDict([
    ("jpsi_mu",  (r"$B_c\!\to\! J/\psi\,\mu\nu$",            P[0], False, [1])),
    ("jpsi_tau", (r"$B_c\!\to\! J/\psi\,\tau\nu$",           P[2], True,  [7])),
    # feed-down: higher charmonia (psi(2S), chi_c0/1/2, h_c) semileptonic, mu+tau
    ("feeddown", (r"$B_c\!\to\!(\psi',\chi_c,h_c)\,\ell\nu$", P[1], False,
                  [2, 3, 4, 5, 6, 8])),
    # J/psi + open charm (D, Ds, D*, plus the D+K(*) associated modes)
    ("jpsi_D",   (r"$B_c\!\to\! J/\psi + D_{(s)}$",          P[4], False,
                  [13, 14, 15, 16, 17, 18, 19, 20])),
    # everything else: J/psi + light hadrons (pi, K, ppbar pi, KK pi)
    ("other",    (r"$B_c\!\to\! J/\psi + \mathrm{hadrons}$", P[3], False,
                  [9, 10, 11, 12, 21, 22])),
])
# Prefer a tighter "J/psi + single D"? Move 15-18 (J/psi D K(*)) into "other"
# by editing the two code lists above.

# expand to the {code: (label, colour, is_signal)} map the plotter consumes
BC_SPLIT = {}
for _name, (_lab, _col, _sig, _codes) in COMPONENTS.items():
    for _c in _codes:
        BC_SPLIT[_c] = (_lab, _col, _sig)
# unmapped codes (and -1 "unknown") fall into the "other" component
_other = COMPONENTS["other"]
BC_DEFAULT = (_other[0], _other[1], _other[2])

# keep every true Bc decay (codes 1..22 and the -1 "unknown"); the -1 rows are
# routed to the "other" component by BC_DEFAULT. This is the exact complement of
# EXCLUDE_BC below, so the Bc and Hb samples partition the J/psi-from-b phase space.
KEEP_BC = "(gen_bc_decay >= 1) | (gen_bc_decay == -1)"
# Hb: drop every event with a Bc (codes -1 and 1..22); keep 0 / NaN only
EXCLUDE_BC = "(gen_bc_decay == 0) | (gen_bc_decay != gen_bc_decay)"

JPSI_IN  = "(np.abs(jpsi_mass - 3.0969) < 0.1)"
JPSI_OUT = "(np.abs(jpsi_mass - 3.0969) > 0.15)"

ISO_PASS = "(mu_reliso_04 < 0.2)"
ISO_FAIL = "(mu_reliso_04 > 0.4)"

# common selection
COMMON_SELECTION = " & ".join([
#     "(np.abs(jpsi_mass - 3.0969) < 0.1)",
#     "(np.abs(jpsi_mass - 3.0969) > 0.15)",
    "(jpsi_good_vtx > 0.5)",
    "(jpsi_reliso_04 < 0.4)",
    "(mu1_pt > 4)",
    "(mu2_pt > 3)",
    "(jpsi_lxy_sig > 3)",
#     "(mu3_id_tight > 0.5)",
    "(mu3_id_soft_mva > 0.5)",
#     "(mu_ip3d_jpsi_sv_sig > -3)",
    "(mass < 6.275)",
#     "(mass > 6.275)",
#     "(jpsi_lxy<0.3)",
#     "(p4_par_jpsi>0)",
#     "(lxyz_sig<18)",
#     "(mu_ip3d_jpsi_pv_sig>0)"
])

# =============================================================================
samples = [
    # --- dedicated Bc signal+cocktail MC --------------------------------------
    Sample(
        name="bc",
        files=[f"{NTUPLE_DIR}/bc.root"],
        scale=BC_SCALE,                  # lumi * sigma(Bc) / N_gen  (see top)
        weight_branches=[],              # e.g. ["puWeight", "ctau_weight_central"]
        selection=f"({COMMON_SELECTION}) & ({KEEP_BC}) & ({JPSI_IN}) & ({ISO_PASS})" ,
        split_by="gen_bc_decay",
        split_map=BC_SPLIT,
        split_default=BC_DEFAULT,
    ),

    # --- inclusive Hb -> J/psi + X (hb1 & hb2), Bc removed --------------------
    # hb1/hb2 are two pT-filter scales: list both file sets and tag them with the
    # same `group` so they stack into a single "Hb" entry. Give each its own
    # `scale` (lumi * sigma / N_gen) since the two productions normalise apart.
    Sample(
        name="hb", files=[f"{NTUPLE_DIR}/hb.root"],
        label=r"$H_b\!\to\! J/\psi + X$", color=P[5], group="hb",
        scale=HB_SCALE, 
        selection=f"({COMMON_SELECTION}) & ({EXCLUDE_BC}) & ({JPSI_IN}) & ({ISO_PASS})" ,
        weight_branches=[],
    ),

    # --- data -----------------------------------------------------------------
    Sample(
        name="data", 
        files=[f"{NTUPLE_DIR}/data.root"], 
        selection=f"({COMMON_SELECTION}) & ({JPSI_IN}) & ({ISO_PASS})" ,
        is_data=True
    ),

    # --- misID -----------------------------------------------------------------
    Sample(
        name="misID",
        files=[f"{NTUPLE_DIR}/data.root"],
        label=r"misID",
        color=P[6],
        group="misID",
        scale=20.0,            # see note below — almost certainly not 1.0
        selection=f"({COMMON_SELECTION}) & ({JPSI_IN}) & ({ISO_FAIL})",
        # no is_data
    ),






#     Sample(
#         name="combinatorial",
#         files=[f"{NTUPLE_DIR}/data.root"],
#         label=r"combinatorial",
#         color=P[6],
#         group="combinatorial",
#         scale=20.0,            # see note below — almost certainly not 1.0
#         selection=f"({COMMON_SELECTION}) & ({JPSI_OUT})",
#         # no is_data
#     ),

]
