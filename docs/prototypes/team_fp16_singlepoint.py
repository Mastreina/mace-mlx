"""fp16 vs fp32 single-point error matrix across systems (and model sizes).

Systems (fixed seeds): rattled Si216/Si1000, rattled fcc Cu256, water box 81,
rattled NaCl216, rattled alpha-quartz SiO2 162.
Model medium-mpa-0 everywhere; small/large additionally on Si216.
Metrics: dE (meV/atom), relative force RMS error, per-|F|-bin errors,
stress error (GPa).
"""
import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-mastreina-Desktop-mace-mlx/25918f5b-8f0e-48d3-a2d3-5900e397f165/scratchpad")
import os

import numpy as np

from team_fp16_common import (SCRATCH, force_metrics, free_calc, load_calc,
                              make_cu, make_nacl, make_quartz, make_si,
                              make_water_box, stress_metrics, update_results)


def build_systems():
    # Ordered small -> large so the memory-heavy job runs last per dtype pass.
    return {
        "Water81_H2O": make_water_box(),          # 81 atoms, H+O
        "SiO2_quartz162": make_quartz(),          # 162 atoms, Si+O
        "Si216_rattled": make_si(3),              # 216 atoms
        "NaCl216_rattled": make_nacl(3),          # 216 atoms, Na+Cl
        "Cu256_fcc_rattled": make_cu(4),          # 256 atoms, metal
        "Si1000_rattled": make_si(5),             # 1000 atoms (memory-heavy)
    }


JOBS = [
    # (model, list of system names)
    ("medium-mpa-0", ["Water81_H2O", "SiO2_quartz162", "Si216_rattled",
                      "NaCl216_rattled", "Cu256_fcc_rattled", "Si1000_rattled"]),
    ("small", ["Si216_rattled"]),
    ("large", ["Si216_rattled"]),
]


def main():
    systems = build_systems()
    raw = {}  # (model, name, dtype) -> dict with E, F, S
    for model, names in JOBS:
        for dtype in ("float32", "float16"):
            print(f"=== {model} / {dtype} ===", flush=True)
            calc = load_calc(dtype, model=model)
            for name in names:
                a = systems[name].copy()
                a.calc = calc
                try:
                    S = a.get_stress()  # one pass computes E+F+S
                except Exception as exc:  # degrade gracefully
                    print(f"  {name}: stress failed ({exc}); E+F only", flush=True)
                    S = None
                E = a.get_potential_energy()
                F = a.get_forces()
                raw[(model, name, dtype)] = {
                    "E": float(E),
                    "F": np.asarray(F, dtype=np.float64),
                    "F_dtype": str(F.dtype),
                    "S": None if S is None else np.asarray(S, dtype=np.float64),
                }
                print(f"  {name}: E={E:.4f} eV  n={len(a)}  F.dtype={F.dtype}",
                      flush=True)
            free_calc(calc)

    # ---- compare ----------------------------------------------------------
    matrix = {}
    npz = {}
    for model, names in JOBS:
        for name in names:
            r32 = raw[(model, name, "float32")]
            r16 = raw[(model, name, "float16")]
            n = len(systems[name])
            entry = {
                "natoms": n,
                "elements": sorted(set(systems[name].get_chemical_symbols())),
                "E_fp32_eV": r32["E"],
                "E_fp16_eV": r16["E"],
                "dE_meV_per_atom": (r16["E"] - r32["E"]) * 1000.0 / n,
                "forces": force_metrics(r32["F"], r16["F"]),
                "F_numpy_dtype_fp16": r16["F_dtype"],
            }
            if r32["S"] is not None and r16["S"] is not None:
                entry["stress"] = stress_metrics(r32["S"], r16["S"])
            matrix[f"{model}/{name}"] = entry
            npz[f"{model}__{name}__F32"] = r32["F"]
            npz[f"{model}__{name}__F16"] = r16["F"]
            fm = entry["forces"]
            print(f"{model}/{name}: dE={entry['dE_meV_per_atom']:+.3f} meV/atom  "
                  f"F_rel_rms={fm['rel_rms_pct']:.2f}%  "
                  f"maxdF={fm['max_vector_err']:.4f} eV/A", flush=True)

    update_results("singlepoint_matrix", matrix)
    np.savez_compressed(os.path.join(SCRATCH, "team_fp16_forces_raw.npz"), **npz)
    print("done")


if __name__ == "__main__":
    main()
