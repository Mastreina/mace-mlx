"""Quantify GPU run-to-run non-determinism (recompute noise) per dtype.

This sets the noise floor against which fp16-vs-fp32 differences must be read.
"""
import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-mastreina-Desktop-mace-mlx/25918f5b-8f0e-48d3-a2d3-5900e397f165/scratchpad")
import numpy as np

from team_fp16_common import (free_calc, load_calc, make_si, make_water_box,
                              update_results)

# ---- water box contact analysis -------------------------------------------
w = make_water_box()
from ase.neighborlist import neighbor_list

i, j, d = neighbor_list("ijd", w, cutoff=2.5)
mol = np.arange(len(w)) // 3  # builder appends molecules as O,H,H triples
inter = mol[i] != mol[j]
intra = ~inter
print(f"water box: min intra-molecular d = {d[intra].min():.3f} A, "
      f"min inter-molecular d = {d[inter].min():.3f} A")

# ---- recompute noise -------------------------------------------------------
si = make_si(3)  # Si216 rattled, the reference system
N = len(si)
out = {}
for dtype in ("float32", "float16"):
    calc = load_calc(dtype)
    a = si.copy()
    a.calc = calc
    E, F = [], []
    for rep in range(5):
        calc.results = {}
        E.append(a.get_potential_energy())
        F.append(a.get_forces().astype(np.float64))
    E = np.array(E)
    dE = (E - E[0]) * 1000.0 / N  # meV/atom vs first run
    dF = np.array([np.abs(F[k] - F[0]).max() for k in range(1, 5)])
    rmsF = np.array([np.sqrt(((F[k] - F[0]) ** 2).mean()) for k in range(1, 5)])
    out[dtype] = {
        "E_spread_meV_per_atom": float(np.ptp(E) * 1000.0 / N),
        "dE_runs_meV_per_atom": [float(x) for x in dE[1:]],
        "F_maxdiff_eV_A_max": float(dF.max()),
        "F_rmsdiff_eV_A_max": float(rmsF.max()),
    }
    print(f"{dtype}: E spread {out[dtype]['E_spread_meV_per_atom']:.5f} meV/atom, "
          f"max|dF| {dF.max():.2e} eV/A, rms dF {rmsF.max():.2e} eV/A over 5 recomputes")
    free_calc(calc)

update_results("run_to_run_noise_Si216_medium", out)
