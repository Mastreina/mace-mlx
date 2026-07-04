"""Geometry optimization convergence: rattled Si64, BFGS, fp32 vs fp16.

One BFGS run per dtype with fmax=0.01 target (cap 250 steps); per-step fmax
history gives steps-to-converge for every intermediate threshold
(0.05/0.03/0.02/0.01) plus the fp16 force-noise floor.
Final structures of both runs are cross-evaluated with the fp32 calculator.
"""
import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-mastreina-Desktop-mace-mlx/25918f5b-8f0e-48d3-a2d3-5900e397f165/scratchpad")
import numpy as np
from ase.optimize import BFGS

from team_fp16_common import free_calc, load_calc, make_si, update_results

THRESHOLDS = [0.05, 0.03, 0.02, 0.01]
MAX_STEPS = 250

atoms0 = make_si(2, rattle=0.05, seed=42)  # Si64
print(f"system: Si64 rattled(0.05, seed42), n={len(atoms0)}")

out = {"system": "Si64_rattled(0.05, seed42)", "model": "medium-mpa-0",
       "optimizer": "BFGS", "max_steps": MAX_STEPS}
finals = {}

for dtype in ("float32", "float16"):
    atoms = atoms0.copy()
    calc = load_calc(dtype)
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    fmax_hist, e_hist = [], []
    for converged in opt.irun(fmax=0.01, steps=MAX_STEPS):
        f = atoms.get_forces()
        fmax_hist.append(float(np.linalg.norm(f, axis=1).max()))
        e_hist.append(float(atoms.get_potential_energy()))
    fmax_arr = np.array(fmax_hist)

    res = {"converged_0.01": bool(converged), "n_steps_run": len(fmax_hist) - 1,
           "fmax_initial": fmax_hist[0], "fmax_final": fmax_hist[-1],
           "fmax_min": float(fmax_arr.min()),
           "fmax_last50_min": float(fmax_arr[-50:].min()),
           "fmax_last50_median": float(np.median(fmax_arr[-50:])),
           "E_final_own_eV": e_hist[-1],
           "fmax_history": [float(x) for x in fmax_hist],
           "E_history": [float(x) for x in e_hist]}
    for thr in THRESHOLDS:
        below = np.nonzero(fmax_arr <= thr)[0]
        if len(below):
            first = int(below[0])
            res[f"steps_to_{thr}"] = first
            res[f"bounces_above_{thr}_after_first_crossing"] = int(
                (fmax_arr[first:] > thr).sum())
        else:
            res[f"steps_to_{thr}"] = None
            res[f"bounces_above_{thr}_after_first_crossing"] = None
    out[dtype] = res
    finals[dtype] = atoms.copy()
    print(f"{dtype}: init fmax={fmax_hist[0]:.3f}  "
          f"steps->0.05/0.03/0.02/0.01 = "
          f"{[res[f'steps_to_{t}'] for t in THRESHOLDS]}  "
          f"min fmax={res['fmax_min']:.4f}  converged(0.01)={converged}",
          flush=True)
    free_calc(calc)

# ---- cross-evaluate both final structures with fp32 -------------------------
calc32 = load_calc("float32")
for dtype, a in finals.items():
    a.calc = calc32
    e = a.get_potential_energy()
    f = a.get_forces()
    out[dtype]["E_final_fp32calc_eV"] = float(e)
    out[dtype]["fmax_final_fp32calc"] = float(np.linalg.norm(f, axis=1).max())
n = len(atoms0)
dE = (out["float16"]["E_final_fp32calc_eV"] - out["float32"]["E_final_fp32calc_eV"]) * 1000.0 / n
d = finals["float16"].get_positions() - finals["float32"].get_positions()
d -= d.mean(axis=0)
out["final_dE_meV_per_atom_fp32calc"] = float(dE)
out["final_rmsd_A"] = float(np.sqrt((d**2).sum(axis=1).mean()))
print(f"final structures: dE(fp16-fp32, both eval fp32) = {dE:+.3f} meV/atom, "
      f"RMSD = {out['final_rmsd_A']:.4f} A")
print(f"fp16-run final structure true residual fmax (fp32 calc) = "
      f"{out['float16']['fmax_final_fp32calc']:.4f} eV/A")
print(f"fp32-run final structure residual fmax (fp32 calc)      = "
      f"{out['float32']['fmax_final_fp32calc']:.4f} eV/A")
free_calc(calc32)

update_results("bfgs_Si64_medium", out)
print("done")
