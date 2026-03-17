"""Comprehensive benchmark of all MACE Foundation Models on MLX vs PyTorch CPU.

Tests every available model across multiple system sizes, reporting:
- Inference time (energy + forces)
- Accuracy vs PyTorch reference
- Per-component breakdown (neighbor list, forward, backward)

Usage:
    python scripts/benchmark_all_models.py --quick --correctness --runs 5
    python scripts/benchmark_all_models.py --models small medium large --runs 5 --correctness
    python scripts/benchmark_all_models.py --no-torch --runs 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from ase import Atoms
from ase.build import bulk

ALL_MODELS = [
    # (model_name, head, description)
    ("small", None, "MACE-MP-0 Small (3.8M, 128x0e)"),
    ("medium", None, "MACE-MP-0 Medium (4.7M, 128x0e+128x1o)"),
    ("large", None, "MACE-MP-0 Large (15.8M, 256x0e+256x1o)"),
    ("small-0b", None, "MACE-MP-0b Small (8.2M, Agnesi)"),
    ("medium-0b", None, "MACE-MP-0b Medium (9.1M, Agnesi)"),
    ("small-0b2", None, "MACE-MP-0b2 Small (8.2M, Density)"),
    ("medium-0b2", None, "MACE-MP-0b2 Medium (9.1M, Density)"),
    ("large-0b2", None, "MACE-MP-0b2 Large (Density)"),
    ("medium-0b3", None, "MACE-MP-0b3 Medium (Density)"),
    ("medium-mpa-0", None, "MACE-MPA-0 Medium (new default)"),
    ("small-omat-0", None, "MACE-OMAT-0 Small"),
    ("medium-omat-0", None, "MACE-OMAT-0 Medium"),
    ("mace-matpes-pbe-0", None, "MACE-MatPES PBE"),
    ("mace-matpes-r2scan-0", None, "MACE-MatPES R2SCAN"),
    ("mh-1", "matpes_r2scan", "MACE-MH-1 (matpes_r2scan)"),
    ("mh-1", "mp_pbe_refit_add", "MACE-MH-1 (mp_pbe)"),
    ("mh-1", "omol", "MACE-MH-1 (omol)"),
]

SYSTEMS = [
    # (name, atoms_fn, description)
    ("H2O", lambda: Atoms('H2O', positions=[[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]]), "Water molecule"),
    ("Si-2", lambda: bulk('Si', 'diamond', a=5.43), "Si unit cell"),
    ("Si-16", lambda: bulk('Si', 'diamond', a=5.43) * (2, 2, 2), "Si 2x2x2"),
    ("Cu-27", lambda: bulk('Cu', 'fcc', a=3.6) * (3, 3, 3), "Cu 3x3x3"),
    ("Si-54", lambda: bulk('Si', 'diamond', a=5.43) * (3, 3, 3), "Si 3x3x3"),
    ("Si-128", lambda: bulk('Si', 'diamond', a=5.43) * (4, 4, 4), "Si 4x4x4"),
    ("Si-250", lambda: bulk('Si', 'diamond', a=5.43) * (5, 5, 5), "Si 5x5x5"),
    ("Si-432", lambda: bulk('Si', 'diamond', a=5.43) * (6, 6, 6), "Si 6x6x6"),
]


def bench_one(calc, atoms: Atoms, runs: int = 5, warmup: int = 2) -> np.ndarray:
    """Benchmark a single (calculator, atoms) pair.

    Returns array of per-run times in milliseconds.
    """
    a = atoms.copy()
    a.calc = calc

    # Warmup
    for _ in range(warmup):
        a.calc.results = {}
        _ = a.get_potential_energy()
        _ = a.get_forces()

    # Timed runs
    times = []
    for _ in range(runs):
        a.calc.results = {}
        t0 = time.perf_counter()
        _ = a.get_potential_energy()
        _ = a.get_forces()
        times.append(time.perf_counter() - t0)

    return np.array(times) * 1000  # ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark of all MACE Foundation Models on MLX vs PyTorch CPU.",
    )
    parser.add_argument("--models", nargs="*", help="Specific model names to test (default: all)")
    parser.add_argument("--systems", nargs="*", help="Specific system names to test (default: all)")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs per benchmark (default: 5)")
    parser.add_argument("--no-torch", action="store_true", help="Skip PyTorch comparison")
    parser.add_argument("--quick", action="store_true", help="Quick mode: small systems only (<=54 atoms)")
    parser.add_argument("--correctness", action="store_true", help="Also check correctness vs PyTorch")
    args = parser.parse_args()

    # Filter models
    models = ALL_MODELS
    if args.models:
        models = [m for m in models if m[0] in args.models]

    # Filter systems
    systems = SYSTEMS
    if args.systems:
        systems = [s for s in systems if s[0] in args.systems]
    if args.quick:
        # Keep only systems with <=54 atoms
        quick_names = {"H2O", "Si-2", "Si-16", "Cu-27", "Si-54"}
        systems = [s for s in systems if s[0] in quick_names]

    from mace_mlx.calculators import MACEMLXCalculator

    calc_torch_fn = None
    if not args.no_torch:
        try:
            from mace.calculators.foundations_models import mace_mp as torch_mace_mp
            calc_torch_fn = torch_mace_mp
        except ImportError:
            print("WARNING: mace-torch not installed -- skipping PyTorch comparison.\n")
            args.no_torch = True

    # Header
    print("=" * 100)
    print("MACE-MLX Comprehensive Benchmark")
    print(f"Models: {len(models)}, Systems: {len(systems)}, Runs: {args.runs}")
    print("=" * 100)

    results: list[tuple] = []
    total_models = len(models)

    for m_idx, (model_name, head, desc) in enumerate(models, 1):
        print(f"\n{'─' * 80}")
        print(f"[{m_idx}/{total_models}] Model: {desc}")
        print(f"{'─' * 80}")

        # Load MLX calculator
        kw_m: dict = {"model_path": model_name}
        if head:
            kw_m["head"] = head
        try:
            t_load = time.perf_counter()
            calc_m = MACEMLXCalculator(**kw_m)
            load_ms = (time.perf_counter() - t_load) * 1000
            print(f"  MLX loaded in {load_ms:.0f} ms")
        except Exception as e:
            print(f"  SKIP (MLX load failed): {e}")
            continue

        # Load PyTorch calculator
        calc_t = None
        if not args.no_torch and calc_torch_fn is not None:
            kw_t: dict = {"model": model_name, "device": "cpu", "default_dtype": "float32"}
            if head:
                kw_t["head"] = head
            try:
                calc_t = calc_torch_fn(**kw_t)
            except Exception as e:
                print(f"  PyTorch unavailable: {e}")

        # Print column header
        if calc_t:
            hdr = f"  {'System':12s} {'Atoms':>5s} {'MLX (ms)':>10s} {'Torch (ms)':>12s} {'Ratio':>7s}"
            if args.correctness:
                hdr += f" {'dE (eV)':>10s} {'dF (eV/A)':>10s}"
            print(hdr)
        else:
            print(f"  {'System':12s} {'Atoms':>5s} {'MLX (ms)':>10s}")

        for sys_name, sys_fn, sys_desc in systems:
            atoms = sys_fn()
            n = len(atoms)

            try:
                t_m = bench_one(calc_m, atoms, runs=args.runs)
                mlx_mean = float(np.mean(t_m))

                if calc_t:
                    t_t = bench_one(calc_t, atoms, runs=args.runs)
                    torch_mean = float(np.mean(t_t))
                    ratio = torch_mean / mlx_mean

                    line = f"  {sys_name:12s} {n:5d} {mlx_mean:9.1f}ms {torch_mean:11.1f}ms {ratio:6.2f}x"

                    if args.correctness:
                        a_m = atoms.copy()
                        a_m.calc = calc_m
                        a_t = atoms.copy()
                        a_t.calc = calc_t
                        dE = abs(float(a_m.get_potential_energy()) - float(a_t.get_potential_energy()))
                        dF = float(np.max(np.abs(a_m.get_forces() - a_t.get_forces())))
                        line += f" {dE:10.2e} {dF:10.2e}"

                    print(line)
                    results.append((model_name, head, sys_name, n, mlx_mean, torch_mean, ratio))
                else:
                    print(f"  {sys_name:12s} {n:5d} {mlx_mean:9.1f}ms")

            except Exception as e:
                print(f"  {sys_name:12s} {n:5d}  ERROR: {str(e)[:60]}")

    # Summary
    if results:
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")

        ratios = [r[6] for r in results]
        wins = sum(1 for r in ratios if r >= 1.0)
        total = len(ratios)
        print(f"MLX wins: {wins}/{total} ({wins / total * 100:.0f}%)")
        print(f"Speedup range: {min(ratios):.2f}x - {max(ratios):.2f}x")
        print(f"Geometric mean speedup: {np.exp(np.mean(np.log(ratios))):.2f}x")

        # Per-model summary
        print(f"\n{'Model':25s} {'Min':>6s} {'Median':>7s} {'Max':>6s}")
        print("-" * 50)
        seen: set[tuple] = set()
        for model_name, head, *_ in results:
            key = (model_name, head)
            if key in seen:
                continue
            seen.add(key)
            model_ratios = [r[6] for r in results if (r[0], r[1]) == key]
            h = f" ({head[:8]})" if head else ""
            print(f"{model_name + h:25s} {min(model_ratios):5.2f}x {np.median(model_ratios):6.2f}x {max(model_ratios):5.2f}x")


if __name__ == "__main__":
    main()
