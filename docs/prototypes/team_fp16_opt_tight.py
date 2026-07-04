"""Find the tightest usable BFGS fmax per dtype (stall floor).

Stage 1: relax Si64 with fp32 BFGS to fmax=0.01 (as in team_fp16_opt.py).
Stage 2: from that structure, run BFGS with fmax=0.002 (cap 150 steps) in
fp32 and fp16; record the fmax floor each dtype can actually reach.
Also: single-point fp16-vs-fp32 force comparison AT the relaxed structure
(near-zero forces), to measure the fp16 force error in the small-force regime.
"""
import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-mastreina-Desktop-mace-mlx/25918f5b-8f0e-48d3-a2d3-5900e397f165/scratchpad")
import numpy as np
from ase.optimize import BFGS

from team_fp16_common import (force_metrics, free_calc, load_calc, make_si,
                              update_results)

out = {"system": "Si64 pre-relaxed (fp32 BFGS fmax=0.01)", "model": "medium-mpa-0"}

# ---- stage 1: fp32 pre-relaxation ------------------------------------------
atoms0 = make_si(2, rattle=0.05, seed=42)
calc32 = load_calc("float32")
atoms0.calc = calc32
BFGS(atoms0, logfile=None).run(fmax=0.01, steps=250)
F32_relaxed = atoms0.get_forces().astype(np.float64)
E32_relaxed = float(atoms0.get_potential_energy())
print(f"pre-relaxed: fmax={np.linalg.norm(F32_relaxed, axis=1).max():.4f} eV/A",
      flush=True)

# ---- force error at the relaxed structure (small-force regime) -------------
calc16 = load_calc("float16")
a16 = atoms0.copy()
a16.calc = calc16
F16_relaxed = a16.get_forces().astype(np.float64)
E16_relaxed = float(a16.get_potential_energy())
fm = force_metrics(F32_relaxed, F16_relaxed)
out["relaxed_structure_force_error"] = fm
out["relaxed_structure_dE_meV_per_atom"] = (E16_relaxed - E32_relaxed) * 1000.0 / len(atoms0)
print(f"at relaxed structure: F_rms_ref={fm['F_rms_ref']:.4f} eV/A, "
      f"rms err={fm['F_rms_err']:.5f}, max vec err={fm['max_vector_err']:.5f} eV/A",
      flush=True)
free_calc(calc16)

# ---- stage 2: tight BFGS from the same start, both dtypes -------------------
for dtype in ("float32", "float16"):
    atoms = atoms0.copy()
    calc = load_calc(dtype)
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    hist = []
    for converged in opt.irun(fmax=0.002, steps=150):
        f = atoms.get_forces()
        hist.append(float(np.linalg.norm(f, axis=1).max()))
    hist_a = np.array(hist)
    res = {"converged_0.002": bool(converged), "n_steps": len(hist) - 1,
           "fmax_min": float(hist_a.min()),
           "fmax_last30_min": float(hist_a[-30:].min()),
           "fmax_last30_median": float(np.median(hist_a[-30:])),
           "fmax_history": [float(x) for x in hist]}
    for thr in (0.005, 0.003, 0.002):
        below = np.nonzero(hist_a <= thr)[0]
        res[f"steps_to_{thr}"] = int(below[0]) if len(below) else None
    out[f"tight_{dtype}"] = res
    print(f"{dtype}: converged(0.002)={converged} in {len(hist)-1} steps, "
          f"min fmax={hist_a.min():.5f}, last30 median={np.median(hist_a[-30:]):.5f}",
          flush=True)
    free_calc(calc)

free_calc(calc32)
update_results("bfgs_tight_Si64_medium", out)
print("done")
