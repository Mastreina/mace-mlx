"""Replacement SiO2 matrix row: ideal beta-cristobalite Si64O128 (rattled).

The first attempt at alpha-quartz via ase.spacegroup produced a wrong
stoichiometry (Si:O = 1:1, all |F| ~ 9.6 eV/A). That row is kept in the
results as an extreme-strain stress test ('SiO2_strained216'); this script
adds a physically sensible SiO2 row. Also decomposes the fp16 energy error
per element via node energies for both SiO2 variants.
"""
import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-mastreina-Desktop-mace-mlx/25918f5b-8f0e-48d3-a2d3-5900e397f165/scratchpad")
import numpy as np

from team_fp16_common import (force_metrics, free_calc, load_calc,
                              make_cristobalite, make_quartz, min_distance,
                              stress_metrics, update_results)

crist = make_cristobalite(2)  # 192 atoms
strained = make_quartz()      # the wrong 'quartz' = strained SiO stress test
print(f"cristobalite: {len(crist)} atoms {crist.get_chemical_formula()}, "
      f"min_dist={min_distance(crist):.3f} A")

raw = {}
for dtype in ("float32", "float16"):
    calc = load_calc(dtype)
    for name, at in (("crist", crist), ("strained", strained)):
        a = at.copy()
        a.calc = calc
        S = a.get_stress()
        raw[(name, dtype)] = {
            "E": float(a.get_potential_energy()),
            "F": a.get_forces().astype(np.float64),
            "S": np.asarray(S, dtype=np.float64),
            "node_e": np.asarray(a.calc.results["node_energy"], dtype=np.float64),
        }
    free_calc(calc)

for name, at, label in (("crist", crist, "SiO2_cristobalite192"),
                        ("strained", strained, "SiO2_strained216")):
    r32, r16 = raw[(name, "float32")], raw[(name, "float16")]
    n = len(at)
    entry = {
        "natoms": n,
        "elements": sorted(set(at.get_chemical_symbols())),
        "E_fp32_eV": r32["E"],
        "E_fp16_eV": r16["E"],
        "dE_meV_per_atom": (r16["E"] - r32["E"]) * 1000.0 / n,
        "forces": force_metrics(r32["F"], r16["F"]),
        "stress": stress_metrics(r32["S"], r16["S"]),
    }
    # per-element node-energy decomposition of the fp16 energy error
    sym = np.array(at.get_chemical_symbols())
    dnode = (r16["node_e"] - r32["node_e"]) * 1000.0  # meV
    entry["node_dE_meV_by_element"] = {
        s: {"mean": float(dnode[sym == s].mean()),
            "std": float(dnode[sym == s].std()),
            "n": int((sym == s).sum())}
        for s in sorted(set(sym))
    }
    fm = entry["forces"]
    print(f"{label}: dE={entry['dE_meV_per_atom']:+.3f} meV/atom  "
          f"F_rms_ref={fm['F_rms_ref']:.3f}  rel_rms={fm['rel_rms_pct']:.2f}%  "
          f"maxdF={fm['max_vector_err']:.4f} eV/A")
    for s, v in entry["node_dE_meV_by_element"].items():
        print(f"   node dE [{s}]: {v['mean']:+.3f} +- {v['std']:.3f} meV/atom (n={v['n']})")
    update_results(f"singlepoint_extra/{label}", entry)
print("done")
