"""Finite-difference force constants (phonon proxy): fp16 vs fp32.

Perfect Si64 crystal; displace atom 0 along x by +-h, central difference
gives one column of the force-constant matrix Phi. Phonon codes (e.g.
phonopy) use h ~ 0.01 A. The fp16 error on Phi directly bounds fp16
usability for phonons/elastic constants.
"""
import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-mastreina-Desktop-mace-mlx/25918f5b-8f0e-48d3-a2d3-5900e397f165/scratchpad")
import numpy as np

from team_fp16_common import free_calc, load_calc, make_si, update_results

HS = [0.005, 0.01, 0.03]

atoms0 = make_si(2, rattle=0.0)  # perfect Si64
n = len(atoms0)

phi = {}  # (dtype, h) -> (N,3) column of force-constant matrix
for dtype in ("float32", "float16"):
    calc = load_calc(dtype)
    for h in HS:
        F = {}
        for sign in (+1, -1):
            a = atoms0.copy()
            p = a.get_positions()
            p[0, 0] += sign * h
            a.set_positions(p)
            a.calc = calc
            F[sign] = a.get_forces().astype(np.float64)
        phi[(dtype, h)] = -(F[+1] - F[-1]) / (2.0 * h)  # eV/A^2
        print(f"{dtype} h={h}: done", flush=True)
    free_calc(calc)

out = {"system": "Si64 perfect crystal", "model": "medium-mpa-0",
       "displaced": "atom 0, x direction", "h_A": HS}
for h in HS:
    p32, p16 = phi[("float32", h)], phi[("float16", h)]
    d = p16 - p32
    self_term = p32[0, 0]
    res = {
        "phi_self_xx_fp32_eV_A2": float(self_term),
        "phi_self_xx_err": float(d[0, 0]),
        "phi_self_xx_rel_err_pct": float(100.0 * d[0, 0] / self_term),
        "col_norm_fp32": float(np.linalg.norm(p32)),
        "col_err_norm": float(np.linalg.norm(d)),
        "col_rel_err_pct": float(100.0 * np.linalg.norm(d) / np.linalg.norm(p32)),
        "max_abs_err_eV_A2": float(np.abs(d).max()),
        "acoustic_sum_fp32": [float(x) for x in p32.sum(axis=0)],
        "acoustic_sum_fp16": [float(x) for x in p16.sum(axis=0)],
    }
    out[f"h_{h}"] = res
    print(f"h={h:0.3f}: Phi_xx(self)={self_term:.3f} eV/A^2, "
          f"rel col err={res['col_rel_err_pct']:.2f}%, "
          f"max abs err={res['max_abs_err_eV_A2']:.4f} eV/A^2")

# fp32 h-convergence as internal reference: difference between h choices
ref_spread = float(np.linalg.norm(phi[("float32", 0.005)] - phi[("float32", 0.03)]))
out["fp32_h_sensitivity_norm_0.005_vs_0.03"] = ref_spread
print(f"fp32 anharmonic h-sensitivity (norm diff h=0.005 vs 0.03): {ref_spread:.4f}")
update_results("fd_force_constants_Si64_medium", out)
print("done")
