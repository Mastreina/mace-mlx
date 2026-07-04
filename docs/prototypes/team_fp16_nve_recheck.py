"""True energy drift of fp16-driven NVE dynamics.

The Etot logged during an fp16 NVE run contains ~0.1 meV/atom of fp16
energy-readout noise, which masks the real (symplectic) drift. Here we rerun
the same fp16 NVE (identical init), dump a frame every 5 steps, then
re-evaluate Epot of every frame with the fp32 calculator:
Etot_true = Epot_fp32(frame) + Ekin(frame). Its slope is the true drift of
the fp16 dynamics on the reference potential-energy surface.
"""
import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-mastreina-Desktop-mace-mlx/25918f5b-8f0e-48d3-a2d3-5900e397f165/scratchpad")
import numpy as np
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary)
from ase.md.verlet import VelocityVerlet

from team_fp16_common import free_calc, load_calc, make_si, update_results

STEPS, EVERY = 500, 5
VSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

atoms0 = make_si(3, rattle=0.05, seed=42)
rng = np.random.default_rng(VSEED)  # 2024 = same init as team_fp16_nve.py
MaxwellBoltzmannDistribution(atoms0, temperature_K=300, rng=rng)
Stationary(atoms0)
n = len(atoms0)

frames = []  # (t_fs, positions, ekin_per_atom)

atoms = atoms0.copy()
calc16 = load_calc("float16")
atoms.calc = calc16
dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)


def snap():
    frames.append((dyn.get_time() / units.fs,
                   atoms.get_positions().copy(),
                   atoms.get_kinetic_energy() / n))


dyn.attach(snap, interval=EVERY)
snap()
for chunk in range(5):
    dyn.run(STEPS // 5)
    print(f"fp16 NVE: step {(chunk + 1) * STEPS // 5}/{STEPS}", flush=True)
free_calc(calc16)

print(f"cross-evaluating {len(frames)} frames with fp32 ...", flush=True)
calc32 = load_calc("float32")
t_fs, etot = [], []
ref = atoms0.copy()
ref.calc = calc32
for k, (t, pos, ekin) in enumerate(frames):
    ref.set_positions(pos)
    epot = ref.get_potential_energy() / n
    t_fs.append(t)
    etot.append((epot + ekin) * 1000.0)  # meV/atom
    if (k + 1) % 25 == 0:
        print(f"  frame {k + 1}/{len(frames)}", flush=True)
free_calc(calc32)

t_ps = np.array(t_fs) / 1000.0
etot = np.array(etot)
slope = float(np.polyfit(t_ps, etot, 1)[0])
resid = etot - np.polyval(np.polyfit(t_ps, etot, 1), t_ps)
out = {
    "description": "fp16-driven NVE, energies re-evaluated with fp32",
    "velocity_seed": VSEED,
    "true_drift_meV_per_atom_per_ps": slope,
    "etot_std_detrended_meV_per_atom": float(resid.std()),
    "etot_peak_to_peak_meV_per_atom": float(np.ptp(etot)),
    "n_frames": len(frames),
    "series_t_fs": [float(x) for x in t_fs],
    "series_etot_true_meV_per_atom": [float(x) for x in etot],
}
print(f"fp16 dynamics, TRUE drift (fp32-evaluated): {slope:+.4f} meV/atom/ps, "
      f"fluct std {resid.std():.4f} meV/atom, "
      f"peak-to-peak {np.ptp(etot):.4f} meV/atom")
update_results(f"nve_fp16_true_drift_Si216_medium_seed{VSEED}", out)
print("done")
