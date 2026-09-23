import uproot
d = "/pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_run3/"
for f in ["hb_covflow.root", "bc_hammer_ctau_covflow.root", "data_2024.root", "data_2022.root", "data_2023.root", "data_2025.root", "data_2026.root"]:
    t = uproot.open(d + f)["tree"]; b = t["jpsi_mass"]
    print(f"{f:32s} {t.file.compression!s:12s} {b.compression!s:12s} "
          f"baskets={b.num_baskets:6d}  comp={b.compressed_bytes/1e6:8.1f} MB  "
          f"uncomp={b.uncompressed_bytes/1e6:8.1f} MB")