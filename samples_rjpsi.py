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
from cmsplot import (Sample, WeightFactor, Derived, p4_ptetaphim, invariant_mass,
                     MASS_K, MASS_PI, MASS_MU, MASS_BC,
                     bin_index, equal_velocity_momentum, proper_time_ps,
                     stitch_index)
from cmsplot.style import PETROFF_10 as P

# --- run conditions -----------------------------------------------------------
COM = 13.6            # Run 2 MC = 13 TeV; switch to 13.6 for Run 3
# LUMI = None           # fb^-1 once you compare to data; None -> Simulation label
LUMI = 308           # fb^-1 once you compare to data; None -> Simulation label
EXTRA = "Preliminary"

NTUPLE_DIR = "/pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_run3"  # EDIT

# MC ntuples: covflow-corrected (the nominal branch names carry the corrected
# quantities, the uncorrected ones live in *_raw twins), the Bc one with the
# Hammer FF weights and the Bc lifetime weights added. Used by the genuine MC
# samples AND by the MC subtraction inside the data-driven misID, so the two
# stay in lockstep. Data is untouched by covflow.
BC_FILE = f"{NTUPLE_DIR}/bc_hammer_ctau_covflow.root"
HB_FILE = f"{NTUPLE_DIR}/hb_covflow.root"
# NTUPLE_DIR = "/Users/manzoni/Documents/rjpsi_run3/ntuples/15jun26"  # EDIT

# --- global MC normalisations -------------------------------------------------
# (2) Tune the absolute Bc and Hb yields here. lumi * sigma / N_gen, times any
#     k-factor / data-driven scale you want. These set the Bc:Hb *ratio*.

# partial luminosities https://twiki.cern.ch/twiki/bin/view/CMSPublic/LumiPublicResults#Summary_proton_proton_collisions
lumi2022 = 38.01/308.
lumi2023 = 30.10/308.
lumi2024 = 112.70/308.
lumi2025 = 114.85/308.
lumi2026 = 30.36/308.


BC_SCALE =  1.5 * (2.81 * 2.45 * 1.373 * 1.185 * 1.51 * 1.54 * 1.3839001 * 1.44 * 1.19 * 1.18 * 1.2 * 0.015 * 0.4267616659357488 * 1.03605435648848   )
HB_SCALE =  1.2 * 1.6 * (2.81 * 2.45 * 1.373 * 1.185 * 1.51 * 1.54 * 1.3839001 * 1.44 * 1.19 * 1.18 * 0.95 * 0.04  * 0.8141294120498126 * 0.5831798345092318) # applied to both hb1 and hb2 (each keeps its own below if needed)
MISID_SCALE = 1.0        # DATA fail-region count enters UNSCALED; only FR(pt) weights it.
                         # (was 0.05: an arbitrary 20x suppression of the data term while the
                         #  MC-subtraction terms used the genuine BC/HB scales -> the fake-factor
                         #  bracket FR*(data_fail - MC_fail) was internally inconsistent and the
                         #  per-bin template flipped sign.)

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

    "q2_jpsi"                    : (20, -10, 12),
    "q2_sv"                      : (20, -10, 12),
#     "m_miss2_jpsi"               : (40, -10, 10),
#     "m_miss2_sv"                 : (40, -10, 10),
    "m_miss2_jpsi"               : (20, -10, 10),
    "m_miss2_sv"                 : (20, -10, 10),
#     "m_miss2_jpsi"               : (25, - 5, 10),
#     "m_miss2_sv"                 : (25, - 5, 10),
    "q2_coll"                    : (22,   0, 11),

    "nu1_q2_jpsi"                : (24,   0, 12),
    "nu2_q2_jpsi"                : (24,   0, 12),
    "nu1_q2_sv"                  : (24,   0, 12),
    "nu2_q2_sv"                  : (24,   0, 12),

    "mu_ip3d_jpsi_pv"            : (40,  - 0.04, 0.04),
    "mu_ip3d_jpsi_pv_err"        : (40,   0    , 0.02),
    "mu_ip3d_jpsi_pv_sig"        : (50, - 5    , 10  ),

    "mu_ip3d_jpsi_sv"            : (40,  - 0.04, 0.04),
    "mu_ip3d_jpsi_sv_err"        : (40,   0    , 0.02),
    "mu_ip3d_jpsi_sv_sig"        : (50, - 5    , 5   ),

    "mu_ip3d_sv_pv"              : (40, - 0.04, 0.04 ),
    "mu_ip3d_sv_pv_err"          : (40,   0   , 0.02 ),
    "mu_ip3d_sv_pv_sig"          : (50, - 5   , 10   ),

    "mu_ip3d_sv_sv"              : (40, - 0.04, 0.04 ),
    "mu_ip3d_sv_sv_err"          : (40,   0   , 0.02 ),
    "mu_ip3d_sv_sv_sig"          : (50, - 5   , 5    ),

    "mu_dist_to_b_dir_jpsi"      : (40,   0  , 0.03),
    "mu_dist_to_b_dir_jpsi_err"  : (40,   0  , 0.02),
    "mu_dist_to_b_dir_jpsi_sig"  : (50, - 5  , 5   ),

    "mu_dist_to_b_dir_sv"        : (40,   0  , 0.03),
    "mu_dist_to_b_dir_sv_err"    : (40,   0  , 0.02),
    "mu_dist_to_b_dir_sv_sig"    : (50, - 5  , 5   ),

    "mu_dist_along_b_dir_jpsi_pv": (40,  -1, 0.6),
    "mu_dist_along_b_dir_jpsi_sv": (40,  -1, 0.4),
    "mu_dist_along_b_dir_sv_pv"  : (40,  -1, 0.6),
    "mu_dist_along_b_dir_sv_sv"  : (40,  -1, 0.4),

    "lxy"     :  np.logspace(-4, np.log10(2), 40),   # variable-width example
    "jpsi_lxy":  np.logspace(-4, np.log10(2), 40),   # variable-width example

    # derived: m(J/psi K+) with the bachelor under the kaon hypothesis. True
    # B+ -> J/psi K+ piles up at the B+ mass (5.279); everything else smears.
    "jpsi_k_mass": (50, 4.5, 7.0),
    "jpsi_pi_mass": (50, 4.5, 7.0),
    # NB: the LHCb stitched-template branches (lhcb_decay_time / lhcb_Z /
    # lhcb_3d_index) get their binning via BINNING.update(...) further down,
    # once their edge arrays / sizes are defined (single source of truth).
}

# --- derived variables (computed on the fly from existing branches) ----------
# DERIVED = {name: Derived(func, inputs)} -- see cmsplot.derived. Each becomes a
# first-class column: plot it (--branches jpsi_k_mass), give it a BINNING /
# AXIS_TITLES entry by name, or feed a datacard (--datacard-branches jpsi_k_mass).
#
# Example: reconstruct the B+ -> J/psi h+ mass under an arbitrary bachelor mass
# hypothesis, to slice those backgrounds out. The J/psi is the refitted,
# mass-constrained four-vector (jpsi_rf_{pt,eta,phi,mass}); the bachelor muon
# (mu3) is re-interpreted with the chosen hadron mass instead of the muon mass,
# so genuine B+ -> J/psi K+ (or J/psi pi+) events peak at m(B+) while
# combinatorial and other Bc/Hb modes spread out.
#
# `Derived.func` is always called as func(arrays), so to parametrise the mass we
# BIND it with a closure (factory below). `inputs` must list only real branches.
_JPSI_HAD_INPUTS = ("jpsi_rf_pt", "jpsi_rf_eta", "jpsi_rf_phi", "jpsi_rf_mass",
                    "mu3_pt", "mu3_eta", "mu3_phi")

def jpsi_had_mass(had_mass):
    """Return a Derived computing m(J/psi + bachelor) for the given bachelor
    mass hypothesis (e.g. MASS_K, MASS_PI, MASS_PROTON, MASS_MU)."""
    def _f(a, _m=had_mass):
        jpsi   = p4_ptetaphim(a["jpsi_rf_pt"], a["jpsi_rf_eta"],
                              a["jpsi_rf_phi"], a["jpsi_rf_mass"])
        hadron = p4_ptetaphim(a["mu3_pt"], a["mu3_eta"], a["mu3_phi"], _m)
        return invariant_mass(jpsi + hadron)
    return Derived(func=_f, inputs=_JPSI_HAD_INPUTS)

# =============================================================================
# LHCb-style multidimensional (3D) stitched template  -------------------------
#
# Reproduces LHCb's R(J/psi) 3D fit in (m^2_miss, decay time, Z) -- where Z is a
# binned 2D function of (q^2, E*_mu) -- by STITCHING the per-(t, Z)-cell m^2_miss
# distributions into one long 1D super-template. Each event is assigned a global
# bin index   g = (i_t * N_Z + i_Z) * N_m + i_m   (decay-time outermost, then Z,
# then m^2_miss innermost); histogramming g with unit integer edges yields the
# N_t x N_Z x N_m = stitched template the binned fit consumes.
#
# Variable choices (CMS, J/psi-direction reco; differ slightly from LHCb):
#   * decay time t = M_Bc * jpsi_lxyz / (c * |p_Bc|), with the equal-velocity
#     |p_Bc| = (M_Bc / m_vis) * |p_vis|, visible_p4 = jpsi_rfp4 + bachelor mu;
#     this is bc_full_p4_jpsi.P() rebuilt from branches (no momentum branch is
#     stored), so no ntuple change is needed.
#   * Z = bin(q2_jpsi) x bin(mu_jpsi_e), mu_jpsi_e = E*_mu in the J/psi frame
#     (better tau/mu separation than the Bc-frame energy);
#   * m^2_miss = m_miss2_jpsi, kept on the existing (20, -10, 10) GeV^2 binning.
# Out-of-grid events get NaN at the offending axis and are DROPPED (no overflow).
# -----------------------------------------------------------------------------
LHCB_T_EDGES     = [0.0, 0.2, 0.4, 0.8, 2.5]      # decay time [ps]   -> 5 bins
LHCB_Q2_EDGES    = [0.0, 8.0, 10.12]                   # q2_jpsi  [GeV^2]  -> 2 bins
LHCB_ESTAR_EDGES = [0.0, 0.5, 0.9, 1.3, 3.5]      # mu_jpsi_e [GeV]   -> 5 bins
LHCB_MMISS2_NBINS = 20                                 # m_miss2_jpsi      -> 20 bins
LHCB_MMISS2_RANGE = (-10.0, 10.0)                      # GeV^2 (matches the BINNING below)

_LHCB_MMISS2_EDGES = np.linspace(LHCB_MMISS2_RANGE[0], LHCB_MMISS2_RANGE[1],
                                 LHCB_MMISS2_NBINS + 1)
N_T     = len(LHCB_T_EDGES) - 1                         # 5
N_Q2    = len(LHCB_Q2_EDGES) - 1                        # 2
N_ESTAR = len(LHCB_ESTAR_EDGES) - 1                     # 5
N_Z     = N_Q2 * N_ESTAR                                # 10  (q^2 outer, E* inner)
N_M     = LHCB_MMISS2_NBINS                             # 20
N_STITCH = N_T * N_Z * N_M                              # 5 * 10 * 20 = 1000

# binning for the stitched index (unit integer bins) and its 1D ingredients,
# registered here so the numbers live in exactly one place (the edges above)
BINNING.update({
    "lhcb_decay_time": LHCB_T_EDGES,                    # variable-width [ps]
    "lhcb_Z":          (N_Z, 0, N_Z),                   # integer Z bins 0..9
    "lhcb_3d_index":   (N_STITCH, 0, N_STITCH),         # unit-width global bins
})

# decay-time inputs: jpsi 3D flight length + the pieces of the visible 4-vector
_DT_INPUTS = ("jpsi_lxyz", "jpsi_rf_pt", "jpsi_rf_eta", "jpsi_rf_phi",
              "jpsi_rf_mass", "mu3_pt", "mu3_eta", "mu3_phi")

def _lhcb_decay_time(a):
    visible = (p4_ptetaphim(a["jpsi_rf_pt"], a["jpsi_rf_eta"],
                            a["jpsi_rf_phi"], a["jpsi_rf_mass"])
               + p4_ptetaphim(a["mu3_pt"], a["mu3_eta"], a["mu3_phi"], MASS_MU))
    p_bc = equal_velocity_momentum(visible, MASS_BC)    # = bc_full_p4_jpsi.P()
    return proper_time_ps(a["jpsi_lxyz"], MASS_BC, p_bc)

def _lhcb_Z(a):
    # row-major over (q^2, E*): low-q^2 row = Z 0..4, high-q^2 row = Z 5..9
    iq = bin_index(a["q2_jpsi"],   LHCB_Q2_EDGES)
    ie = bin_index(a["mu_jpsi_e"], LHCB_ESTAR_EDGES)
    return iq * N_ESTAR + ie

def _lhcb_3d_index(a):
    it = bin_index(a["lhcb_decay_time"], LHCB_T_EDGES)
    iz = a["lhcb_Z"]                                     # already a 0..N_Z-1 index
    im = bin_index(a["m_miss2_jpsi"], _LHCB_MMISS2_EDGES)
    return stitch_index([it, iz, im], [N_T, N_Z, N_M])


DERIVED = {
    "jpsi_k_mass":  jpsi_had_mass(MASS_K),
    "jpsi_pi_mass": jpsi_had_mass(MASS_PI),
    # add any hypothesis in one line, e.g.:
    # "jpsi_p_mass": jpsi_had_mass(MASS_PROTON),

    # LHCb 3D stitched template. Declared in dependency order: decay time and Z
    # are computed first, then the composite index reads them (compute_derived
    # is a single ordered pass, so the two pieces must precede the index). Build
    # the datacard with:  python3 plot.py --config samples_rjpsi.py \
    #     --branches lhcb_3d_index lhcb_decay_time lhcb_Z m_miss2_jpsi \
    #     --datacard-branches lhcb_3d_index
    # then draw the stitched plot with postfit_tools/stitch_plot.py (below).
    "lhcb_decay_time": Derived(func=_lhcb_decay_time, inputs=_DT_INPUTS),
    "lhcb_Z":          Derived(func=_lhcb_Z, inputs=("q2_jpsi", "mu_jpsi_e")),
    "lhcb_3d_index":   Derived(func=_lhcb_3d_index,
                               inputs=("lhcb_decay_time", "lhcb_Z",
                                       "m_miss2_jpsi")),
}


# --- axis-title overrides (issue 4) ------------------------------------------
# cmsplot.binning.axis_label already turns branch names into LaTeX x-axis titles
# (e.g. q2_coll -> "q^2 [GeV^2] (coll.)", mu_ip3d_jpsi_pv_sig -> "IP_3D(mu)/sigma
# (J/psi dir., PV)"). Add exact-branch overrides here to refine any of them;
# anything not listed uses the automatic resolver.
AXIS_TITLES = {
    "jpsi_k_mass": r"$m(J/\psi\,K^{+})$ [GeV]",
    "jpsi_pi_mass": r"$m(J/\psi\,\pi^{+})$ [GeV]",
    "lhcb_decay_time": r"$t$ [ps]",
    "lhcb_Z":          r"$Z(q^{2}_{J/\psi},\,E^{*}_{\mu})$ bin",
    "lhcb_3d_index":   r"stitched $(t,\,Z,\,m^{2}_{\mathrm{miss}})$ bin",
#     "q2_coll":     r"$q^{2}_{\mathrm{coll}}$ [GeV$^{2}$]",
#     "m_miss2_jpsi": r"$m^{2}_{\mathrm{miss}}$ (J/$\psi$ dir.) [GeV$^{2}$]",
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

# =============================================================================
# Datacard composition (issue 3).  This is INDEPENDENT of the plotting split
# above: it declares which contributions become which Combine template, so you
# can change the fit granularity without touching the `samples` list.
#
# Format:
#   DATACARD = {"signal": "<process name>",          # POI 'r' scales this one
#               "processes": OrderedDict(name -> [selector, selector, ...])}
# A selector picks Processes by origin; the first datacard process whose
# selector list matches a given Process claims it. Selector shorthands:
#   "bc"                      whole sample 'bc'
#   ("bc", [1, 7])            sample 'bc', only these gen_bc_decay codes
#   {"group": "misid"}        every read sharing stack-group 'misid'
# NOTE: datacard codes must be a *superset* of a plotting component's codes --
# you can merge plotting components but not subdivide one (it is already summed).
#
# Set DATACARD = None (or just don't define it) to fall back to the per-sample
# `datacard=` tags (single collapsed "Bc" template, signal "Bc").
#
# Flip this to break the Bc cocktail into separate templates (signal = tau):
SPLIT_BC_IN_DATACARD = True

_DATACARD_SPLIT = {
    "signal": "jpsi_tau",
    "processes": OrderedDict([
        ("jpsi_mu",  [("bc", COMPONENTS["jpsi_mu"][3])]),   # B_c -> J/psi mu nu
        ("jpsi_tau", [("bc", COMPONENTS["jpsi_tau"][3])]),  # B_c -> J/psi tau nu (POI)
        ("feeddown", [("bc", COMPONENTS["feeddown"][3])]),  # higher charmonia l nu
        ("jpsi_D",   [("bc", COMPONENTS["jpsi_D"][3])]),    # J/psi + open charm
        ("bc_other", [("bc",)]),                            # catch-all remaining Bc
        ("Hb",       [("hb",)]),                            # inclusive Hb -> J/psi X
        ("misID",    [{"group": "misid"}]),                 # data-driven fake bachelor
    ]),
}

# Collapsed equivalent of the legacy behaviour (one Bc template, signal "Bc"):
_DATACARD_MERGED = {
    "signal": "Bc",
    "processes": OrderedDict([
        ("Bc",    [("bc",)]),
        ("Hb",    [("hb",)]),
        ("misID", [{"group": "misid"}]),
    ]),
}

DATACARD = _DATACARD_SPLIT if SPLIT_BC_IN_DATACARD else _DATACARD_MERGED

# keep every true Bc decay (codes 1..22 and the -1 "unknown"); the -1 rows are
# routed to the "other" component by BC_DEFAULT. This is the exact complement of
# EXCLUDE_BC below, so the Bc and Hb samples partition the J/psi-from-b phase space.
KEEP_BC = "(gen_bc_decay >= 1) | (gen_bc_decay == -1)"
# Hb: drop every event with a Bc (codes -1 and 1..22); keep 0 / NaN only
EXCLUDE_BC = "(gen_bc_decay == 0) | (gen_bc_decay != gen_bc_decay)"

JPSI_IN  = "(np.abs(jpsi_mass - 3.0969) < 0.1)"
# JPSI_IN  = "1"
JPSI_OUT = "(np.abs(jpsi_mass - 3.0969) > 0.15)"


# Strict signal gen-match: the OS pair is the J/psi (role 1) AND the bachelor is
# the genuine signal muon (role 2). Applied to every (ccbar)+lep-nu mode so that
# events where the wrong muon was picked up are NOT counted as signal.
BC_GEN_MATCH = "(mu1_gen_role==1) & (mu2_gen_role==1) & (mu3_gen_role==2)"

# J/psi + D_(s) cocktail modes (codes 13-20): the bachelor muon comes from the D
# decay, so it is not the role-2 signal muon and the strict match above kills the
# whole component. STOPGAP (no ntuple/inspector change yet): release the
# bachelor-role requirement for these modes and match only the OS J/psi pair, so
# B_c -> J/psi + D_(s) is restored in the plots and datacards.
#   NOTE: this does NOT yet require the bachelor to originate from the D (that
#   needs a dedicated gen role assigned in the inspector). Until then a fraction
#   of the released jpsi_D events carry a fake/combinatorial bachelor; tighten
#   to "bachelor from D" once the gen role exists, then drop this exemption.
_JPSI_D_CODES = COMPONENTS["jpsi_D"][3]                      # [13, 14, ..., 20]
_IS_JPSI_D = "(" + " | ".join("(gen_bc_decay==%d)" % c for c in _JPSI_D_CODES) + ")"

# Per-mode cocktail gen-match used for the `bc` sample: always require the J/psi
# pair (roles 1,1); require the role-2 bachelor for EVERY mode except jpsi_D,
# where the bachelor-role requirement is released.
BC_GEN_MATCH_COCKTAIL = (
    "(mu1_gen_role==1) & (mu2_gen_role==1) & ((mu3_gen_role==2) | %s)" % _IS_JPSI_D
)

# common selection
COMMON_SELECTION = " & ".join([
#     "(np.abs(jpsi_mass - 3.0969) < 0.1)",
#     "(np.abs(jpsi_mass - 3.0969) > 0.15)",
    "(jpsi_good_vtx > 0.5)",
    "(jpsi_reliso_04 < 0.4)",
    "(mu_reliso_04 < 0.3)",
    "(mu1_pt > 4)",
    #"(mu2_pt > 3)",
    "(mu2_pt > 3.5)",
    "(mu3_pt > 2.0)",
    "(jpsi_lxy_sig > 3)",
#     "(mu3_id_tight > 0.5)",
    "(mu3_id_soft_mva > 0.5)",
    "(mu_ip3d_jpsi_sv_sig > -3)",
#     "(mass < 6.275)",
    "(mass < 6.1)",
#     "(mass > 6.275)",
    "(jpsi_lxy<0.3)",
    "(p4_par_jpsi>0)",
    "(lxyz_sig<18)",
#     "(mu_ip3d_jpsi_pv_sig>0)",
    "(np.abs(jpsi_k_mass-5.27)>0.1)",

#     "(run<=357482)", #2022C -- parentheses REQUIRED: `&` binds tighter than
                     # `<=`, so a bare `... & run<=357482` evaluates as
                     # `(... & run) <= 357482`, i.e. True for EVERY event, and
                     # the whole COMMON_SELECTION silently becomes a no-op.

    # extra handles to reduce bkg    
#     "(nu1_jpsi_pz>0.)",
#     "(mu_jpsi_e<2.5)",
#     "(mcorr_jpsi<8)",
#     "(cos_theta_l_nu2<0)",
#     
#     "(q2_coll>8)",

#     "(nu1_mu_b_e_jpsi < 1.8)",
#     "(nu2_mu_b_e_jpsi < 1.8)",

#     "(mu_ip3d_jpsi_sv_sig > 0)",

#     "(cos_theta_l_nu1<0)",
#     "(cos_theta_l_nu2<0)",
    
])

# =============================================================================
# Data-driven misID (fake bachelor muon) background  ---------------------------
#
# Zero-th order fake-factor estimate:
#   misID(signal region) = FR(pt) x [ data(fail-iso) - genuine MC(fail-iso) ]
# i.e. take data in the bachelor-muon FAIL-isolation region, subtract the
# genuine processes (Bc + Hb) predicted there by MC, and transfer the remainder
# into the PASS-isolation signal region with a pt-binned fake rate FR(pt).
#
# Implementation maps onto the engine's `fakerate` + `group` hooks:
#   * one +scale DATA read  in the fail region, FR-weighted  (is_data=False so it
#     STACKS as a background rather than being drawn as points);
#   * negative-scale MC reads (Bc, Hb) in the fail region, FR-weighted;
#   * all three share group="misid", so the plotter sums them bin-by-bin into a
#     single stacked entry = FR x (data_fail - MC_fail).
# All carry is_fakerate=True (set automatically because `fakerate` is given), so
# they are excluded from --scale-to-data.
#
# FAIL region = COMMON_SELECTION with the bachelor-muon isolation cut INVERTED.
# Only the bachelor iso (mu_reliso_04) is flipped; everything else (incl. the
# J/psi iso and JPSI_IN window) is held fixed, so fail and signal differ ONLY in
# the variable the fake rate is measured against.
_MU_ISO_PASS = "(mu_reliso_04 < 0.3)"
_MU_ISO_FAIL = "(mu_reliso_04 > 0.3)"
assert _MU_ISO_PASS in COMMON_SELECTION, \
    "bachelor-iso term changed in COMMON_SELECTION; update _MU_ISO_PASS"
COMMON_SELECTION_FAIL = COMMON_SELECTION.replace(_MU_ISO_PASS, _MU_ISO_FAIL)

# pt-binned fake-rate table.  (pt_branch, edges, values); edges has len(values)+1
# entries, np.inf closes the top bin.  Bachelor muon = mu3.
# >>> PLACEHOLDER: flat FR = 0.4 in every pt bin.  Replace VALUES once measured.
FR_PT_BRANCH = 'mu3_pt'
FR_PT_EDGES  = [3, 4, 5, 6, 8, 10, 13, 17, np.inf]
# FR_PT_VALUES = 0.15 * np.array([1.8397, 1.6658, 1.3692, 1.1056, 0.8944, 0.8107, 0.7505, 0.8603]) # with loose selection
FR_PT_VALUES =  0.4 * np.array([0.6799, 0.7029, 0.5096, 0.6518, 0.4602, 0.6667, 0.5758, 2.3333]) # with the same selection as here
# FR_PT_VALUES = np.array([0.6799, 0.7029, 0.5096, 0.6518, 0.4602, 0.6667, 0.5758, 2.3333]) # with the same selection as here
FR_TABLE = (FR_PT_BRANCH, FR_PT_EDGES, FR_PT_VALUES)
# =============================================================================
# Bc per-event weights: Hammer form factors (Kiselev -> Harrison-2024 BGLVar)
# and Bc lifetime (0.507 -> PDG 0.510 ps). Both are applied to every read of
# the Bc ntuple -- the genuine `bc` sample AND its misID subtraction -- and
# their variations become shape nuisances in the datacards, correlated between
# the two reads because they share the nuisance names.
#
# Branch schema: Bmmm/Analysis/python/HammerFF.py (hammer_*) and
# JpsiChargedBranches.py (gen_bc_ctau_weight*). Hb gets neither: the Bc events
# are removed from it by EXCLUDE_BC.
#
# The weights shift yields as well as shapes (FF rate change + acceptance). For
# now that is accepted as is; the plan is to factor the pure yield change out
# with per-channel <w> measured on the UNFILTERED sample.
# =============================================================================
HAMMER_N_EV = 15                               # BctoJpsiBGLVar eigen-directions
HAMMER_SIGNAL_CODES = (1, 7)                   # gen_bc_decay: J/psi mu nu, J/psi tau nu
# hammer_status codes, HammerFF.py: STATUS_OK, _NOT_SIGNAL, _NO_LEAVES,
# _DECLINED, _NONFINITE = range(5)
HAM_OK, HAM_NOT_SIGNAL, HAM_NO_LEAVES, HAM_DECLINED, HAM_NONFINITE = range(5)
# Signal events whose Hammer weight came out non-finite (status 4, ~per mille)
# get this flat weight, for the nominal AND every FF variation (i.e. they carry
# no FF shape uncertainty). >>> PLACEHOLDER: ~<w_nominal>; replace with the
# value measured on the unfiltered sample -- or drop once status 4 is fixed at
# the origin (see the open-items note).
HAMMER_NONFINITE_FALLBACK = 0.49

_HAMMER_VAR_BRANCHES = tuple("hammer_ff_ev%02d_%s" % (j, d)
                             for j in range(HAMMER_N_EV) for d in ("up", "dn"))


def _hammer_factor(branch, report=False):
    """Per-event Hammer factor from ``branch`` (nominal or one variation).

    status 0 (OK)          -> the weight, which must be finite;
    status 1 (NOT_SIGNAL)  -> 1  (every mode but J/psi mu nu / J/psi tau nu);
    status 4 (NONFINITE)   -> HAMMER_NONFINITE_FALLBACK;
    status 2/3, or a status inconsistent with gen_bc_decay -> hard error.
    """
    def _f(a):
        st = np.asarray(a["hammer_status"], "float64")
        code = np.asarray(a["gen_bc_decay"], "float64")
        w = np.asarray(a[branch], "float64")
        if not np.all(np.isfinite(st)):
            raise ValueError("hammer_status is NaN for %d events"
                             % int((~np.isfinite(st)).sum()))
        st = np.round(st).astype("int64")
        is_sig = np.isin(code, HAMMER_SIGNAL_CODES)          # NaN -> False
        incons = (is_sig & (st == HAM_NOT_SIGNAL)) | (~is_sig & (st != HAM_NOT_SIGNAL))
        if incons.any():
            raise ValueError("hammer_status inconsistent with gen_bc_decay for "
                             "%d events" % int(incons.sum()))
        known = (HAM_OK, HAM_NOT_SIGNAL, HAM_NONFINITE)
        unhandled = ~np.isin(st, known)
        if unhandled.any():
            vals, cnt = np.unique(st[unhandled], return_counts=True)
            raise ValueError("hammer_status without a weight policy: %s "
                             "(2 = missing gen leaves, 3 = declined by Hammer)"
                             % dict(zip(vals.tolist(), cnt.tolist())))
        ok = st == HAM_OK
        if not np.all(np.isfinite(w[ok])):
            raise ValueError("%s non-finite for %d status-OK events" % (
                branch, int((~np.isfinite(w[ok])).sum())))
        out = np.ones(w.shape, "float64")
        out[ok] = w[ok]
        nonfin = st == HAM_NONFINITE
        out[nonfin] = HAMMER_NONFINITE_FALLBACK
        if report:
            nsig = int(is_sig.sum())
            print("  [hammer] %d signal events, %d status-4 -> w = %.3f "
                  "(%.2f permille); <%s>_OK = %.4f"
                  % (nsig, int(nonfin.sum()), HAMMER_NONFINITE_FALLBACK,
                     1e3 * nonfin.sum() / max(nsig, 1), branch,
                     float(w[ok].mean()) if ok.any() else float("nan")))
        return out
    return _f


def _ctau_factor(branch):
    """Bc lifetime factor. NaN means a gen Bc with an unusable decay length
    (see bc_ctau_weight in JpsiChargedBranches.py): a broken gen record, never
    silently weighted 1."""
    def _f(a):
        w = np.asarray(a[branch], "float64")
        bad = ~np.isfinite(w)
        if bad.any():
            raise ValueError("%s non-finite for %d of %d selected Bc events"
                             % (branch, int(bad.sum()), w.size))
        return w
    return _f


BC_WEIGHT_FACTORS = OrderedDict([
    ("hammer_ff", WeightFactor(
        inputs=("hammer_weight", "hammer_status", "gen_bc_decay")
               + _HAMMER_VAR_BRANCHES,
        # plot-only runs read just these (no datacard -> no variations)
        nominal_inputs=("hammer_weight", "hammer_status", "gen_bc_decay"),
        nominal=_hammer_factor("hammer_weight", report=True),
        variations=OrderedDict(
            ("ff_ev%02d" % j, (_hammer_factor("hammer_ff_ev%02d_up" % j),
                               _hammer_factor("hammer_ff_ev%02d_dn" % j)))
            for j in range(HAMMER_N_EV)),
    )),
    ("bc_ctau", WeightFactor(
        inputs=("gen_bc_ctau_weight", "gen_bc_ctau_weight_up",
                "gen_bc_ctau_weight_down"),
        nominal_inputs=("gen_bc_ctau_weight",),
        nominal=_ctau_factor("gen_bc_ctau_weight"),
        variations={"bc_ctau": (_ctau_factor("gen_bc_ctau_weight_up"),
                                _ctau_factor("gen_bc_ctau_weight_down"))},
    )),
])

# =============================================================================
samples = [
    # --- dedicated Bc signal+cocktail MC --------------------------------------
    Sample(
        name="bc",
        files=[BC_FILE],
        datacard="Bc",                   # all gen_bc_decay components -> one Bc template
        scale=BC_SCALE,                  # lumi * sigma(Bc) / N_gen  (see top)
        weight_branches=[],              # e.g. ["puWeight"]
        weight_factors=BC_WEIGHT_FACTORS,  # Hammer FF x Bc ctau (+ variations)
        selection=f"({COMMON_SELECTION}) & ({KEEP_BC}) & ({JPSI_IN}) & ({BC_GEN_MATCH_COCKTAIL})" ,
        split_by="gen_bc_decay",
        split_map=BC_SPLIT,
        split_default=BC_DEFAULT,
    ),

    # --- inclusive Hb -> J/psi + X (hb1 & hb2), Bc removed --------------------
    # hb1/hb2 are two pT-filter scales: list both file sets and tag them with the
    # same `group` so they stack into a single "Hb" entry. Give each its own
    # `scale` (lumi * sigma / N_gen) since the two productions normalise apart.
    Sample(
        name="hb", files=[HB_FILE],
        label=r"$H_b\!\to\! J/\psi + X$", color=P[5], group="hb",
        datacard="Hb",
        scale=HB_SCALE, 
        selection=f"({COMMON_SELECTION}) & ({EXCLUDE_BC}) & ({JPSI_IN})" ,
        weight_branches=[],
    ),

    # --- data -----------------------------------------------------------------
    Sample(
        name="data", 
#         files=[f"{NTUPLE_DIR}/data_2024.root"], 
        files=[
            f"{NTUPLE_DIR}/data_2022.root",
            f"{NTUPLE_DIR}/data_2023.root",
            f"{NTUPLE_DIR}/data_2024.root",
            f"{NTUPLE_DIR}/data_2025.root",
            f"{NTUPLE_DIR}/data_2026.root",
        ], 
        selection=f"({COMMON_SELECTION}) & ({JPSI_IN}) & (in_golden_json>0.5)" ,
        is_data=True, datacard="data_obs"
    ),

#     Sample(
#         name="combinatorial",
#         files=[f"{NTUPLE_DIR}/data.root"],
#         label=r"combinatorial ($J/\psi$ sidebands)", color=P[8], group="comb",
#         datacard="combinatorial",        # free rateParam template in the fit (set to ""
#                                          # ONLY if you intend the misID template to absorb it)
#         selection=f"({COMMON_SELECTION}) & ({JPSI_OUT})" ,
#         is_data=False,                   # data-DERIVED BACKGROUND: must STACK, never be drawn
#                                          # as data points. (was is_data=True, which summed the
#                                          # J/psi-sideband counts into the black data points so
#                                          # the plotted "Data" no longer matched datacard data_obs.)
#     ),

    # --- data-driven misID (fake bachelor muon) ------------------------------
    # FR x (data_fail - Bc_fail - Hb_fail); the three reads merge via group="misid".
    # is_data=False on the data read so it STACKS (FR-weighted) instead of being
    # drawn as points. MC reads carry a NEGATIVE scale to subtract.
    Sample(
        name="misid_data",
        files=[
            f"{NTUPLE_DIR}/data_2022.root",
            f"{NTUPLE_DIR}/data_2023.root",
            f"{NTUPLE_DIR}/data_2024.root",
            f"{NTUPLE_DIR}/data_2025.root",
            f"{NTUPLE_DIR}/data_2026.root",
        ], 
        label=r"misID (data-driven)", color=P[9], group="misid",
        datacard="misID",
        scale=MISID_SCALE,
        selection=f"({COMMON_SELECTION_FAIL}) & ({JPSI_IN}) & (in_golden_json>0.5)",
        fakerate=FR_TABLE,
        # NOTE: is_data stays False on purpose (this is a stacked background,
        # built from data but not the measurement).
    ),
    Sample(
        name="misid_bc_sub",
        files=[BC_FILE],
        group="misid", datacard="misID",
        scale=-BC_SCALE,                 # subtract genuine Bc predicted in fail
        weight_branches=[],              # mirror the genuine `bc` sample's weights
        weight_factors=BC_WEIGHT_FACTORS,  # same factors + nuisances as `bc`
        selection=f"({COMMON_SELECTION_FAIL}) & ({KEEP_BC}) & ({JPSI_IN}) & ({BC_GEN_MATCH_COCKTAIL})",
        fakerate=FR_TABLE,
    ),
    Sample(
        name="misid_hb_sub",
        files=[HB_FILE],
        group="misid", datacard="misID",
        scale=-HB_SCALE,                 # subtract genuine Hb predicted in fail
        weight_branches=[],              # mirror the genuine `hb` sample's weights
        selection=f"({COMMON_SELECTION_FAIL}) & ({EXCLUDE_BC}) & ({JPSI_IN})",
        fakerate=FR_TABLE,
    ),

]
