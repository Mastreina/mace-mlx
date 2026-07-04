"""Surface-system trajectory correctness check for mace-mlx 0.5.0.

Per system (one process, run serially):
  1. initial-frame single point: mlx fp32 vs mace-torch fp64 CPU (gold)
  2. NVE 600 steps @ 1 fs (Maxwell 300 K, fixed seed): total-energy drift
     (linear fit after a 100-step equilibration window) and fluctuation
  3. NVT Langevin 500 steps: temperature statistics + structure health
     (min interatomic distance, max displacement, evaporated atoms)
  4. final-frame single point vs torch again (off-equilibrium geometry is
     the stricter comparison point)
Reports MD ms/step as a by-product (valid: GPU is otherwise idle).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.build import bulk, diamond100, fcc111, surface
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

OUT = Path(__file__).parent / "surf_check_results.json"


def mgo100():
    b = bulk("MgO", "rocksalt", a=4.212, cubic=True)
    s = surface(b, (1, 0, 0), layers=5, vacuum=12.0)
    return s.repeat((5, 5, 1))


def tio2_110():
    from ase.spacegroup import crystal

    rutile = crystal(
        ["Ti", "O"],
        basis=[(0, 0, 0), (0.3053, 0.3053, 0)],
        spacegroup=136,
        cellpar=[4.594, 4.594, 2.959, 90, 90, 90],
    )
    s = surface(rutile, (1, 1, 0), layers=4, vacuum=12.0)
    return s.repeat((4, 6, 1))


SYSTEMS = {
    "cu111": lambda: fcc111("Cu", size=(12, 12, 6), vacuum=12.0),
    "si100": lambda: diamond100("Si", size=(8, 8, 6), vacuum=12.0),
    "mgo100": mgo100,
    "tio2_110": tio2_110,
    "cu111_big": lambda: fcc111("Cu", size=(16, 16, 10), vacuum=12.0),
}


def health(atoms, ref_pos, zlim):
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    dmin = float(d.min())
    disp = float(np.abs(atoms.positions - ref_pos).max())
    evap = int((atoms.positions[:, 2] > zlim).sum())
    return dmin, disp, evap


def torch_singlepoint(atoms):
    from mace.calculators import mace_mp as mace_mp_torch

    calc = mace_mp_torch(model="medium-mpa-0", device="cpu",
                         default_dtype="float64")
    a = atoms.copy()
    a.calc = calc
    return a.get_potential_energy(), a.get_forces()


def compare(tag, e_m, f_m, e_t, f_t, n):
    de = (e_m - e_t) * 1000 / n
    df = np.abs(f_m - f_t)
    frms = np.sqrt((f_t ** 2).mean())
    print(f"  {tag}: dE={de:+.3f} meV/atom  dF_max={df.max():.2e} eV/A  "
          f"dF_rms={np.sqrt((df**2).mean()):.2e} (F_rms={frms:.3f})",
          flush=True)
    return dict(de_mev_atom=de, df_max=float(df.max()),
                df_rms=float(np.sqrt((df ** 2).mean())), f_rms=float(frms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=list(SYSTEMS), required=True)
    ap.add_argument("--temp", type=float, default=300.0)
    args = ap.parse_args()

    from mace_mlx.calculators import mace_mp

    atoms = SYSTEMS[args.system]()
    atoms.pbc = [True, True, True]  # vacuum 24 A total > 2x cutoff
    n = len(atoms)
    zlim = atoms.positions[:, 2].max() + 3.0
    ref_pos = atoms.positions.copy()
    syms = sorted(set(atoms.get_chemical_symbols()))
    print(f"== {args.system}: {n} atoms {syms}, T={args.temp:.0f} K ==",
          flush=True)

    res = {"natoms": n, "elements": syms, "temp_K": args.temp}

    calc = mace_mp(model="medium-mpa-0")
    atoms.calc = calc

    # 1. initial-frame cross check
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    et0, ft0 = torch_singlepoint(atoms)
    res["initial"] = compare("initial vs torch-fp64", e0, f0, et0, ft0, n)

    # 2. NVE
    rng = np.random.default_rng(42)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp, rng=rng)
    dyn = VelocityVerlet(atoms, 1.0 * units.fs)
    etot, times = [], []
    t0 = time.perf_counter()
    for i in range(600):
        dyn.run(1)
        if i % 5 == 4:
            etot.append(atoms.get_potential_energy()
                        + atoms.get_kinetic_energy())
    nve_ms = (time.perf_counter() - t0) / 600 * 1e3
    etot = np.array(etot)
    t_ps = (np.arange(len(etot)) * 5 + 5) / 1000.0
    win = t_ps > 0.1  # skip equilibration of the unrelaxed surface
    slope = np.polyfit(t_ps[win], etot[win], 1)[0] * 1000 / n
    fluct = float(np.std(etot[win] - np.polyval(
        np.polyfit(t_ps[win], etot[win], 1), t_ps[win])) * 1000 / n)
    dmin, disp, evap = health(atoms, ref_pos, zlim)
    print(f"  NVE 600x1fs: drift={slope:+.3f} meV/atom/ps  "
          f"fluct={fluct:.4f} meV/atom  T_end={atoms.get_temperature():.0f} K"
          f"  ({nve_ms:.0f} ms/step)", flush=True)
    print(f"    health: dmin={dmin:.3f} A  max_disp={disp:.2f} A  "
          f"evaporated={evap}", flush=True)
    res["nve"] = dict(drift=float(slope), fluct=fluct, ms_per_step=nve_ms,
                      dmin=dmin, max_disp=disp, evaporated=evap)

    # 3. NVT Langevin
    dyn2 = Langevin(atoms, 1.0 * units.fs, temperature_K=args.temp,
                    friction=0.2, rng=np.random.default_rng(7))  # 0.2/ase-time ~ 0.02/fs
    temps = []
    t0 = time.perf_counter()
    for i in range(500):
        dyn2.run(1)
        if i % 5 == 4:
            temps.append(atoms.get_temperature())
    nvt_ms = (time.perf_counter() - t0) / 500 * 1e3
    temps = np.array(temps)
    thalf = temps[len(temps) // 2:]
    dmin, disp, evap = health(atoms, ref_pos, zlim)
    print(f"  NVT 500x1fs: T(2nd half)={thalf.mean():.1f}+-{thalf.std():.1f} K"
          f"  ({nvt_ms:.0f} ms/step)", flush=True)
    print(f"    health: dmin={dmin:.3f} A  max_disp={disp:.2f} A  "
          f"evaporated={evap}", flush=True)
    res["nvt"] = dict(T_mean=float(thalf.mean()), T_std=float(thalf.std()),
                      ms_per_step=nvt_ms, dmin=dmin, max_disp=disp,
                      evaporated=evap)

    # 4. final-frame cross check (off-equilibrium, thermalized geometry)
    ef = atoms.get_potential_energy()
    ff = atoms.get_forces()
    etf, ftf = torch_singlepoint(atoms)
    res["final"] = compare("final   vs torch-fp64", ef, ff, etf, ftf, n)

    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[args.system] = res
    OUT.write_text(json.dumps(data, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
