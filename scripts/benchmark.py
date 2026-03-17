"""Benchmark MACE-MLX vs PyTorch MACE on various system sizes.

Usage:
    python scripts/benchmark.py [--runs N] [--no-torch] [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from ase import Atoms
from ase.build import bulk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ms(val: float) -> str:
    """Format milliseconds with 1 decimal."""
    return f"{val:8.1f}"


def _header() -> None:
    print(f"\n{'=' * 100}")


def _separator() -> None:
    print(f"{'─' * 100}")


# ---------------------------------------------------------------------------
# Per-component timing
# ---------------------------------------------------------------------------

def _time_neighbor_list(atoms: Atoms, calc, n_runs: int = 10) -> float:
    """Time just the neighbor list construction (ms)."""
    times: list[float] = []
    for _ in range(n_runs):
        # Force rebuild by clearing cache
        calc._nl_cache = None
        t0 = time.perf_counter()
        calc._get_neighbor_list(atoms)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.mean(times)) * 1000


def _count_edges(atoms: Atoms, calc) -> int:
    """Count edges in the neighbor list."""
    src, _, _ = calc._get_neighbor_list(atoms)
    return len(src)


# ---------------------------------------------------------------------------
# Benchmark one system
# ---------------------------------------------------------------------------

def benchmark_system(
    atoms: Atoms,
    name: str,
    calc_mlx,
    calc_torch=None,
    n_runs: int = 10,
) -> dict:
    """Benchmark MLX (and optionally PyTorch) on *atoms*.

    Returns a dict with timing and accuracy results including per-component
    breakdown (neighbor list, forward, backward).
    """
    result: dict = {"name": name, "n_atoms": len(atoms)}

    # ── Count edges ────────────────────────────────────────────────────
    n_edges = _count_edges(atoms, calc_mlx)
    result["n_edges"] = n_edges

    # ── Neighbor list timing ──────────────────────────────────────────
    nl_ms = _time_neighbor_list(atoms, calc_mlx, n_runs=n_runs)
    result["nl_ms"] = nl_ms

    # ── MLX total ─────────────────────────────────────────────────────
    atoms_mlx = atoms.copy()
    atoms_mlx.calc = calc_mlx

    # Warmup (2 calls)
    for _ in range(2):
        atoms_mlx.calc.results = {}
        _ = atoms_mlx.get_potential_energy()
        _ = atoms_mlx.get_forces()

    mlx_times: list[float] = []
    for _ in range(n_runs):
        atoms_mlx.calc.results = {}
        # Force NL rebuild each iteration to measure total cost accurately
        calc_mlx._nl_cache = None
        t0 = time.perf_counter()
        _ = atoms_mlx.get_potential_energy()
        _ = atoms_mlx.get_forces()
        t1 = time.perf_counter()
        mlx_times.append(t1 - t0)

    mlx_mean = np.mean(mlx_times) * 1000
    mlx_std = np.std(mlx_times) * 1000
    mlx_min = np.min(mlx_times) * 1000
    result["mlx_mean_ms"] = mlx_mean
    result["mlx_std_ms"] = mlx_std
    result["mlx_min_ms"] = mlx_min

    # Model time = total - NL
    result["model_ms"] = max(0.0, mlx_mean - nl_ms)

    # Final MLX values for correctness check
    atoms_mlx.calc.results = {}
    e_mlx = atoms_mlx.get_potential_energy()
    f_mlx = atoms_mlx.get_forces()
    result["e_mlx"] = e_mlx

    # ── PyTorch ────────────────────────────────────────────────────────
    if calc_torch is not None:
        atoms_torch = atoms.copy()
        atoms_torch.calc = calc_torch

        # Warmup
        for _ in range(2):
            atoms_torch.calc.results = {}
            _ = atoms_torch.get_potential_energy()
            _ = atoms_torch.get_forces()

        torch_times: list[float] = []
        for _ in range(n_runs):
            atoms_torch.calc.results = {}
            t0 = time.perf_counter()
            _ = atoms_torch.get_potential_energy()
            _ = atoms_torch.get_forces()
            t1 = time.perf_counter()
            torch_times.append(t1 - t0)

        torch_mean = np.mean(torch_times) * 1000
        torch_std = np.std(torch_times) * 1000
        torch_min = np.min(torch_times) * 1000
        result["torch_mean_ms"] = torch_mean
        result["torch_std_ms"] = torch_std
        result["torch_min_ms"] = torch_min

        # Correctness
        atoms_torch.calc.results = {}
        e_torch = atoms_torch.get_potential_energy()
        f_torch = atoms_torch.get_forces()

        result["e_diff"] = abs(e_mlx - e_torch)
        result["f_max_diff"] = float(np.max(np.abs(f_mlx - f_torch)))
        result["speedup"] = torch_mean / mlx_mean if mlx_mean > 0 else float("nan")

    return result


def print_result(r: dict, show_torch: bool) -> None:
    """Pretty-print one benchmark result (detailed view)."""
    n = r["n_atoms"]
    edges = r.get("n_edges", "?")
    nl = r.get("nl_ms", 0)
    model = r.get("model_ms", 0)
    total = r["mlx_mean_ms"]

    print(f"\n  {r['name']}  ({n} atoms, {edges} edges)")
    print(f"    NL: {nl:.1f} ms | Model: {model:.1f} ms | Total: {total:.1f} ms "
          f"(+/- {r['mlx_std_ms']:.1f} ms)")

    if show_torch and "torch_mean_ms" in r:
        print(f"    PyTorch (CPU): {r['torch_mean_ms']:.1f} ms  |  "
              f"Speedup: {r['speedup']:.2f}x")
        print(f"    dE: {r['e_diff']:.2e} eV  |  dF max: {r['f_max_diff']:.2e} eV/A")


def print_summary_table(results: list[dict], show_torch: bool) -> None:
    """Print a compact summary table."""
    _header()
    print("  MACE-MLX Performance Benchmark — Summary")
    _header()

    if show_torch:
        hdr = (
            f"  {'System':<16} {'Atoms':>5} {'Edges':>6} "
            f"{'NL(ms)':>7} {'Model(ms)':>9} {'Total(ms)':>9} "
            f"{'Torch(ms)':>9} {'Ratio':>7} "
            f"{'dE(eV)':>10} {'dF max':>10}"
        )
        print(hdr)
        _separator()
        for r in results:
            torch_ms = r.get("torch_mean_ms", float("nan"))
            speedup = r.get("speedup", float("nan"))
            e_diff = r.get("e_diff", float("nan"))
            f_diff = r.get("f_max_diff", float("nan"))
            print(
                f"  {r['name']:<16} {r['n_atoms']:>5} {r.get('n_edges', 0):>6} "
                f"{r.get('nl_ms', 0):>7.1f} {r.get('model_ms', 0):>9.1f} "
                f"{r['mlx_mean_ms']:>9.1f} "
                f"{torch_ms:>9.1f} {speedup:>6.2f}x "
                f"{e_diff:>10.2e} {f_diff:>10.2e}"
            )
    else:
        hdr = (
            f"  {'System':<16} {'Atoms':>5} {'Edges':>6} "
            f"{'NL(ms)':>7} {'Model(ms)':>9} {'Total(ms)':>9} {'min(ms)':>8}"
        )
        print(hdr)
        _separator()
        for r in results:
            print(
                f"  {r['name']:<16} {r['n_atoms']:>5} {r.get('n_edges', 0):>6} "
                f"{r.get('nl_ms', 0):>7.1f} {r.get('model_ms', 0):>9.1f} "
                f"{r['mlx_mean_ms']:>9.1f} {r['mlx_min_ms']:>8.1f}"
            )

    _header()

    # Scaling analysis
    print("\n  Scaling Analysis (atoms vs total time):")
    _separator()
    if len(results) > 1:
        base = results[0]
        for r in results[1:]:
            atom_ratio = r["n_atoms"] / base["n_atoms"] if base["n_atoms"] > 0 else 0
            time_ratio = r["mlx_mean_ms"] / base["mlx_mean_ms"] if base["mlx_mean_ms"] > 0 else 0
            print(
                f"    {base['name']:>14} -> {r['name']:<14}: "
                f"atoms {atom_ratio:>6.1f}x, time {time_ratio:>6.1f}x"
            )
    print()


# ---------------------------------------------------------------------------
# Build test systems
# ---------------------------------------------------------------------------

def build_systems(quick: bool = False) -> list[tuple[Atoms, str]]:
    """Return list of (atoms, label) for benchmarking.

    Args:
        quick: If True, only include small/medium systems.
    """
    systems: list[tuple[Atoms, str]] = []

    # Small molecule
    water = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )
    systems.append((water, "Water (H2O)"))

    # Si unit cell
    systems.append((bulk("Si", "diamond", a=5.43), "Si unit"))

    # Medium systems
    systems.append((bulk("Cu", "fcc", a=3.6) * (2, 2, 2), "Cu 2x2x2"))
    systems.append((bulk("Si", "diamond", a=5.43) * (2, 2, 2), "Si 2x2x2"))
    systems.append((bulk("Cu", "fcc", a=3.6) * (3, 3, 3), "Cu 3x3x3"))
    systems.append((bulk("Si", "diamond", a=5.43) * (3, 3, 3), "Si 3x3x3"))
    systems.append((bulk("Al", "fcc", a=4.05) * (3, 3, 3), "Al 3x3x3"))

    if quick:
        return systems

    # Large systems
    systems.append((bulk("Si", "diamond", a=5.43) * (4, 4, 4), "Si 4x4x4"))
    systems.append((bulk("Cu", "fcc", a=3.6) * (4, 4, 4), "Cu 4x4x4"))
    systems.append((bulk("Si", "diamond", a=5.43) * (5, 5, 5), "Si 5x5x5"))
    systems.append((bulk("Cu", "fcc", a=3.6) * (5, 5, 5), "Cu 5x5x5"))
    systems.append((bulk("Si", "diamond", a=5.43) * (6, 6, 6), "Si 6x6x6"))
    systems.append((bulk("Cu", "fcc", a=3.6) * (6, 6, 6), "Cu 6x6x6"))
    systems.append((bulk("Si", "diamond", a=5.43) * (7, 7, 7), "Si 7x7x7"))
    systems.append((bulk("Fe", "bcc", a=2.87) * (5, 5, 5), "Fe 5x5x5"))
    systems.append((bulk("Al", "fcc", a=4.05) * (5, 5, 5), "Al 5x5x5"))

    return systems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MACE-MLX vs PyTorch MACE",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of timed iterations per system (default: 10)",
    )
    parser.add_argument(
        "--no-torch", action="store_true",
        help="Skip PyTorch comparison (MLX only)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Only benchmark small/medium systems (skip large)",
    )
    args = parser.parse_args()

    show_torch = not args.no_torch

    # ── Load calculators ───────────────────────────────────────────────
    print("Loading MACE-MLX calculator ...", flush=True)
    from mace_mlx.calculators import MACEMLXCalculator, _USE_MATSCIPY
    calc_mlx = MACEMLXCalculator(model_path="small", skin=0.0)  # no cache for fair benchmark

    calc_torch = None
    if show_torch:
        print("Loading PyTorch MACE calculator ...", flush=True)
        try:
            from mace.calculators import mace_mp
            calc_torch = mace_mp(model="small", device="cpu", default_dtype="float32")
        except ImportError:
            print("WARNING: mace-torch not installed — skipping PyTorch comparison.\n")
            show_torch = False

    # ── Header ─────────────────────────────────────────────────────────
    nl_backend = "matscipy (C)" if _USE_MATSCIPY else "ASE (Python)"
    print(f"\n{'#' * 100}")
    print(f"  MACE-MLX Performance Benchmark")
    print(f"  Model:           MACE-MP-0 Small")
    print(f"  NL backend:      {nl_backend}")
    print(f"  Runs:            {args.runs} per system (+ 2 warmup)")
    if show_torch:
        print(f"  Compare:         MLX (Apple GPU) vs PyTorch (CPU)")
    else:
        print(f"  Mode:            MLX only")
    print(f"{'#' * 100}")

    # ── Run benchmarks ─────────────────────────────────────────────────
    systems = build_systems(quick=args.quick)
    results: list[dict] = []

    total = len(systems)
    for idx, (atoms, name) in enumerate(systems, 1):
        print(f"\n  [{idx}/{total}] Benchmarking {name} ({len(atoms)} atoms) ...",
              flush=True)
        try:
            r = benchmark_system(
                atoms, name,
                calc_mlx=calc_mlx,
                calc_torch=calc_torch,
                n_runs=args.runs,
            )
            print_result(r, show_torch)
            results.append(r)
        except Exception as exc:
            print(f"\n  {name}: FAILED — {exc}")

    # ── Summary ────────────────────────────────────────────────────────
    if results:
        print_summary_table(results, show_torch)


if __name__ == "__main__":
    main()
