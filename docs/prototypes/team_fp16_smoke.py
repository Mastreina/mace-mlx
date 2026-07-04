"""Smoke test: interfaces, dtypes, stress, water-box geometry sanity."""
import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-mastreina-Desktop-mace-mlx/25918f5b-8f0e-48d3-a2d3-5900e397f165/scratchpad")
import numpy as np

from team_fp16_common import (free_calc, load_calc, make_quartz, make_si,
                              make_water_box, min_distance)

# Water box geometry sanity
w = make_water_box()
print(f"water box: {len(w)} atoms, cell={w.cell.lengths()}, min_dist={min_distance(w):.3f} A")
q = make_quartz()
print(f"quartz: {len(q)} atoms, min_dist={min_distance(q):.3f} A")

si = make_si(1)  # 8 atoms
for dtype in ("float32", "float16"):
    calc = load_calc(dtype)
    a = si.copy()
    a.calc = calc
    s = a.get_stress()
    e = a.get_potential_energy()
    f = a.get_forces()
    print(f"{dtype}: E={e:.6f} eV  F.dtype={f.dtype}  F.shape={f.shape}  "
          f"S.dtype={s.dtype}  S[0]={s[0]:.6e} eV/A^3")
    # determinism: force a full recompute on the same positions
    calc.results = {}
    e2 = a.get_potential_energy()
    f2 = a.get_forces()
    print(f"{dtype}: recompute identical: E {e2 == e}, F {np.array_equal(f, f2)}")
    free_calc(calc)
print("smoke OK")
