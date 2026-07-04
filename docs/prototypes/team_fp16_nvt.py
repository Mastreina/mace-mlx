"""NVT (Langevin) sanity: does the fp16 drift matter under a thermostat?

Si216 rattled, Langevin 300 K, friction 0.02/fs, 1 fs, 500 steps, identical
init and identical thermostat noise sequence (fixed rng seed) for both dtypes.
Compare temperature statistics and mean potential energy over the second half.
"""
import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-mastreina-Desktop-mace-mlx/25918f5b-8f0e-48d3-a2d3-5900e397f165/scratchpad")
import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary)

from team_fp16_common import free_calc, load_calc, make_si, update_results

STEPS = 500
atoms0 = make_si(3, rattle=0.05, seed=42)
rng0 = np.random.default_rng(2024)
MaxwellBoltzmannDistribution(atoms0, temperature_K=300, rng=rng0)
Stationary(atoms0)
n = len(atoms0)

out = {"system": "Si216_rattled", "model": "medium-mpa-0",
       "thermostat": "Langevin 300K friction=0.02/fs dt=1fs", "steps": STEPS}
for dtype in ("float32", "float16"):
    atoms = atoms0.copy()
    calc = load_calc(dtype)
    atoms.calc = calc
    dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300,
                   friction=0.02, rng=np.random.default_rng(777))
    temp, epot = [], []

    def log(atoms=atoms):
        temp.append(atoms.get_temperature())
        epot.append(atoms.get_potential_energy() / n)

    dyn.attach(log, interval=1)
    for chunk in range(5):
        dyn.run(STEPS // 5)
        print(f"  {dtype}: step {(chunk + 1) * STEPS // 5}/{STEPS} T={temp[-1]:.1f} K",
              flush=True)
    temp = np.array(temp)
    epot = np.array(epot)
    h = len(temp) // 2
    out[dtype] = {
        "T_2nd_half_mean_K": float(temp[h:].mean()),
        "T_2nd_half_std_K": float(temp[h:].std()),
        "Epot_2nd_half_mean_meV_per_atom": float(epot[h:].mean() * 1000.0),
        "series_T_K": [float(x) for x in temp],
    }
    print(f"{dtype}: T(2nd half) = {temp[h:].mean():.1f} +- {temp[h:].std():.1f} K, "
          f"<Epot> = {epot[h:].mean()*1000:.2f} meV/atom", flush=True)
    free_calc(calc)

dT = out["float16"]["T_2nd_half_mean_K"] - out["float32"]["T_2nd_half_mean_K"]
dU = (out["float16"]["Epot_2nd_half_mean_meV_per_atom"]
      - out["float32"]["Epot_2nd_half_mean_meV_per_atom"])
out["dT_K"] = float(dT)
out["dEpot_meV_per_atom"] = float(dU)
print(f"NVT fp16-fp32: dT = {dT:+.2f} K, d<Epot> = {dU:+.3f} meV/atom")
update_results("nvt_langevin_Si216_medium", out)
print("done")
