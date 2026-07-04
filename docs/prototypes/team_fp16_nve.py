"""NVE trajectories: medium-mpa-0 / Si216 (rattle 0.05), VelocityVerlet 1 fs,
300 K Maxwell-Boltzmann init (fixed seed), 500 steps, fp32 vs fp16.

Metrics: total-energy drift (linear fit, meV/atom/ps), energy fluctuation
around the trend, temperature statistics over the last 250 steps.
Identical initial positions AND momenta for both runs.
"""
import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-mastreina-Desktop-mace-mlx/25918f5b-8f0e-48d3-a2d3-5900e397f165/scratchpad")
import numpy as np
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary)
from ase.md.verlet import VelocityVerlet

from team_fp16_common import free_calc, load_calc, make_si, update_results

STEPS = 500
DT_FS = 1.0

# ---- shared initial state ---------------------------------------------------
atoms0 = make_si(3, rattle=0.05, seed=42)  # Si216
rng = np.random.default_rng(2024)
MaxwellBoltzmannDistribution(atoms0, temperature_K=300, rng=rng)
Stationary(atoms0)
print(f"initial T = {atoms0.get_temperature():.1f} K, n = {len(atoms0)}")

out = {"system": "Si216_rattled(0.05, seed42)", "model": "medium-mpa-0",
       "dt_fs": DT_FS, "steps": STEPS, "T_init_seed": 2024}
final_pos = {}

for dtype in ("float32", "float16"):
    atoms = atoms0.copy()  # copies positions and momenta
    calc = load_calc(dtype)
    atoms.calc = calc
    n = len(atoms)

    f0 = atoms.get_forces()
    force_dtype = str(f0.dtype)

    dyn = VelocityVerlet(atoms, timestep=DT_FS * units.fs)
    t_fs, epot, ekin, temp = [], [], [], []

    def log(dyn=dyn, atoms=atoms):
        t_fs.append(dyn.get_time() / units.fs)
        epot.append(atoms.get_potential_energy() / n)
        ekin.append(atoms.get_kinetic_energy() / n)
        temp.append(atoms.get_temperature())

    dyn.attach(log, interval=1)
    log()  # step 0
    for chunk in range(5):
        dyn.run(STEPS // 5)
        print(f"  {dtype}: step {(chunk + 1) * STEPS // 5}/{STEPS}  "
              f"T={temp[-1]:.1f} K  Etot={(epot[-1] + ekin[-1]):.6f} eV/atom",
              flush=True)

    t_ps = np.array(t_fs) / 1000.0
    etot = (np.array(epot) + np.array(ekin)) * 1000.0  # meV/atom
    # drift: linear fit over the whole run and over the second half
    slope_all = float(np.polyfit(t_ps, etot, 1)[0])
    half = len(t_ps) // 2
    slope_half = float(np.polyfit(t_ps[half:], etot[half:], 1)[0])
    resid = etot - np.polyval(np.polyfit(t_ps, etot, 1), t_ps)
    tempa = np.array(temp)
    out[dtype] = {
        "force_numpy_dtype": force_dtype,
        "drift_meV_per_atom_per_ps_fit_all": slope_all,
        "drift_meV_per_atom_per_ps_fit_2nd_half": slope_half,
        "etot_std_detrended_meV_per_atom": float(resid.std()),
        "etot_peak_to_peak_meV_per_atom": float(np.ptp(etot)),
        "etot_first_meV_per_atom": float(etot[0]),
        "etot_last_meV_per_atom": float(etot[-1]),
        "T_last250_mean_K": float(tempa[-250:].mean()),
        "T_last250_std_K": float(tempa[-250:].std()),
        "series_t_fs": [float(x) for x in t_fs],
        "series_etot_meV_per_atom": [float(x) for x in etot],
        "series_T_K": [float(x) for x in tempa],
    }
    final_pos[dtype] = atoms.get_positions().copy()
    print(f"{dtype}: drift(all)={slope_all:+.3f} meV/atom/ps  "
          f"drift(2nd half)={slope_half:+.3f}  "
          f"fluct(detrended std)={resid.std():.4f} meV/atom  "
          f"T={tempa[-250:].mean():.1f}+-{tempa[-250:].std():.1f} K  "
          f"F dtype={force_dtype}", flush=True)
    free_calc(calc)

# trajectory divergence (chaotic separation, for context only)
d = final_pos["float16"] - final_pos["float32"]
d -= d.mean(axis=0)
out["final_pos_rmsd_A"] = float(np.sqrt((d**2).sum(axis=1).mean()))
print(f"final-configuration RMSD fp16 vs fp32: {out['final_pos_rmsd_A']:.4f} A "
      f"(chaotic divergence, expected)")

update_results("nve_Si216_medium", out)
print("done")
