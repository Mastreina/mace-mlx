"""Shared helpers for the fp16-vs-fp32 accuracy experiments (mace-mlx).

All systems use fixed seeds so every script is exactly reproducible.
No timing is measured anywhere (GPU is shared with other agents).
"""
import gc
import json
import os

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule

SCRATCH = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(SCRATCH, "team_fp16_results.json")

EV_A3_TO_GPA = 160.21766208


# --------------------------------------------------------------------------- #
# System builders (all deterministic)
# --------------------------------------------------------------------------- #
def make_si(n: int = 3, rattle: float = 0.05, seed: int = 42) -> Atoms:
    at = bulk("Si", "diamond", a=5.43, cubic=True) * (n, n, n)
    if rattle:
        at.rattle(stdev=rattle, seed=seed)
    return at


def make_cu(n: int = 4, rattle: float = 0.05, seed: int = 42) -> Atoms:
    at = bulk("Cu", "fcc", a=3.615, cubic=True) * (n, n, n)
    if rattle:
        at.rattle(stdev=rattle, seed=seed)
    return at


def make_nacl(n: int = 3, rattle: float = 0.05, seed: int = 42) -> Atoms:
    at = bulk("NaCl", "rocksalt", a=5.64, cubic=True) * (n, n, n)
    if rattle:
        at.rattle(stdev=rattle, seed=seed)
    return at


def make_quartz(rep=(3, 3, 2), rattle: float = 0.05, seed: int = 42) -> Atoms:
    from ase.spacegroup import crystal

    at = crystal(
        ("Si", "O"),
        basis=[(0.4697, 0.0, 0.0), (0.4135, 0.2669, 0.1191)],
        spacegroup=152,
        cellpar=[4.913, 4.913, 5.405, 90, 90, 120],
    ) * rep
    if rattle:
        at.rattle(stdev=rattle, seed=seed)
    return at


def make_cristobalite(n: int = 2, a: float = 7.43, rattle: float = 0.05,
                      seed: int = 42) -> Atoms:
    """Ideal beta-cristobalite SiO2: Si on a diamond lattice, O at every
    Si-Si bond midpoint. a=7.43 gives Si-O = a*sqrt(3)/8 = 1.609 A.
    Unambiguous construction (8 Si + 16 O per cubic cell)."""
    from ase.neighborlist import neighbor_list

    si = bulk("Si", "diamond", a=a, cubic=True)
    i, j, D = neighbor_list("ijD", si, cutoff=a * np.sqrt(3) / 4 + 0.1)
    pts, seen = [], set()
    for ii, DD in zip(i, D):
        mid = (si.positions[ii] + DD / 2.0) % a
        key = tuple(np.round(mid, 3) % a)
        if key in seen:
            continue
        seen.add(key)
        pts.append(mid)
    assert len(pts) == 16, f"expected 16 O, got {len(pts)}"
    at = si + Atoms("O" * 16, positions=pts, cell=si.cell, pbc=True)
    at = at * (n, n, n)
    if rattle:
        at.rattle(stdev=rattle, seed=seed)
    return at


def make_water_box(n_side: int = 3, spacing: float = 3.3, seed: int = 7,
                   rattle: float = 0.02) -> Atoms:
    """n_side^3 rigid H2O molecules on a grid with random orientations, PBC."""
    rng = np.random.default_rng(seed)
    h2o = molecule("H2O")
    h2o.positions -= h2o.get_center_of_mass()
    box = Atoms(cell=[n_side * spacing] * 3, pbc=True)
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                m = h2o.copy()
                phi, theta, psi = rng.uniform(0.0, 360.0, 3)
                m.euler_rotate(phi=phi, theta=theta, psi=psi, center="COM")
                m.translate((np.array([i, j, k], dtype=float) + 0.5) * spacing)
                box += m
    if rattle:
        box.rattle(stdev=rattle, seed=seed + 1)
    return box


def min_distance(atoms: Atoms) -> float:
    from ase.neighborlist import neighbor_list

    d = neighbor_list("d", atoms, cutoff=3.0)
    return float(d.min()) if len(d) else float("inf")


# --------------------------------------------------------------------------- #
# Calculator handling
# --------------------------------------------------------------------------- #
def load_calc(dtype: str, model: str = "medium-mpa-0"):
    from mace_mlx.calculators import mace_mp

    return mace_mp(model=model, default_dtype=dtype)


def free_calc(calc) -> None:
    import mlx.core as mx

    del calc
    gc.collect()
    mx.clear_cache()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
FORCE_BINS = [0.0, 0.05, 0.1, 0.3, 1.0, np.inf]


def force_metrics(F_ref: np.ndarray, F_test: np.ndarray) -> dict:
    """Relative force-error metrics: RMS ratio plus per-magnitude-bin stats."""
    F_ref = np.asarray(F_ref, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    d = F_test - F_ref
    rms_ref = float(np.sqrt((F_ref**2).mean()))
    rms_err = float(np.sqrt((d**2).mean()))
    fn = np.linalg.norm(F_ref, axis=1)   # per-atom |F| reference
    dn = np.linalg.norm(d, axis=1)       # per-atom |dF|
    bins = []
    for lo, hi in zip(FORCE_BINS[:-1], FORCE_BINS[1:]):
        m = (fn >= lo) & (fn < hi)
        if not m.any():
            continue
        rel = dn[m] / np.maximum(fn[m], 1e-12)
        bins.append({
            "range_eV_A": [lo, None if np.isinf(hi) else hi],
            "n_atoms": int(m.sum()),
            "F_median": float(np.median(fn[m])),
            "abs_err_median": float(np.median(dn[m])),
            "abs_err_max": float(dn[m].max()),
            "rel_err_median": float(np.median(rel)),
            "rel_err_max": float(rel.max()),
        })
    return {
        "F_rms_ref": rms_ref,
        "F_rms_err": rms_err,
        "rel_rms_pct": 100.0 * rms_err / rms_ref if rms_ref > 0 else float("nan"),
        "max_component_err": float(np.abs(d).max()),
        "max_vector_err": float(dn.max()),
        "bins": bins,
    }


def stress_metrics(S_ref: np.ndarray, S_test: np.ndarray) -> dict:
    """Stress error in GPa (inputs: ASE Voigt-6 in eV/A^3)."""
    S_ref = np.asarray(S_ref, dtype=np.float64) * EV_A3_TO_GPA
    S_test = np.asarray(S_test, dtype=np.float64) * EV_A3_TO_GPA
    d = S_test - S_ref
    p_ref = -S_ref[:3].mean()
    p_test = -S_test[:3].mean()
    return {
        "ref_max_abs_GPa": float(np.abs(S_ref).max()),
        "err_max_abs_GPa": float(np.abs(d).max()),
        "err_rms_GPa": float(np.sqrt((d**2).mean())),
        "pressure_ref_GPa": float(p_ref),
        "pressure_err_GPa": float(p_test - p_ref),
        "rel_max_pct": float(100.0 * np.abs(d).max() / max(np.abs(S_ref).max(), 1e-12)),
    }


# --------------------------------------------------------------------------- #
# Results file
# --------------------------------------------------------------------------- #
def update_results(key: str, value) -> None:
    data = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            data = json.load(f)
    data[key] = value
    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[results] wrote section '{key}' -> {RESULTS_JSON}")
