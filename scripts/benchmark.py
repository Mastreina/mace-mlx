"""Benchmark MACE-MLX vs PyTorch MACE (CPU & MPS/GPU) on various system sizes.

Usage:
    python scripts/benchmark.py [--runs N] [--no-torch] [--no-mps] [--quick]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from ase import Atoms
from ase.build import bulk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

W = 110  # table width


def _header() -> None:
    print(f"\n{'=' * W}")


def _separator() -> None:
    print(f"{'─' * W}")


def _get_mps_sync():
    """Return ``torch.mps.synchronize`` if available, else *None*."""
    try:
        import torch
        if hasattr(torch.mps, "synchronize"):
            return torch.mps.synchronize
    except (ImportError, AttributeError):
        pass
    return None


def _patch_torch_for_mps():
    """Monkey-patch ``torch.Tensor.double()`` for MPS compatibility.

    MPS doesn't support float64.  This makes ``.double()`` a no-op
    (returns ``.float()``) for tensors already on MPS, preserving the
    original behaviour for CPU / CUDA tensors.

    The MACE forward pass calls ``.double()`` for numerical precision in
    energy summation — on MPS we stay in float32 (acceptable for a
    benchmark).
    """
    import torch

    if getattr(torch.Tensor, "_mps_patched", False):
        return
    _orig_double = torch.Tensor.double

    def _safe_double(self, *args, **kwargs):
        if self.device.type == "mps":
            return self.float(*args, **kwargs)
        return _orig_double(self, *args, **kwargs)

    torch.Tensor.double = _safe_double
    torch.Tensor._mps_patched = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Per-component timing (MLX only)
# ---------------------------------------------------------------------------

def _time_neighbor_list(atoms: Atoms, calc, n_runs: int = 10) -> float:
    """Time just the neighbor list construction (ms)."""
    times: list[float] = []
    for _ in range(n_runs):
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
# Generic torch benchmark helper
# ---------------------------------------------------------------------------

def _bench_torch_calc(
    calc, atoms: Atoms, n_runs: int, sync_fn=None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Benchmark a PyTorch calculator.

    *sync_fn* is called before the first timing call and around each
    iteration to ensure MPS work is complete before reading the clock.

    Returns ``(times_ms, energy, forces)``.
    """
    a = atoms.copy()
    a.calc = calc

    # Warmup
    for _ in range(2):
        a.calc.results = {}
        _ = a.get_potential_energy()
        _ = a.get_forces()
    if sync_fn:
        sync_fn()

    times: list[float] = []
    for _ in range(n_runs):
        a.calc.results = {}
        if sync_fn:
            sync_fn()
        t0 = time.perf_counter()
        _ = a.get_potential_energy()
        _ = a.get_forces()
        if sync_fn:
            sync_fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # Final values for correctness
    a.calc.results = {}
    e = float(a.get_potential_energy())
    f = a.get_forces()

    return np.array(times) * 1000, e, f


# ---------------------------------------------------------------------------
# Benchmark one system
# ---------------------------------------------------------------------------

def benchmark_system(
    atoms: Atoms,
    name: str,
    calc_mlx,
    calc_torch_cpu=None,
    calc_torch_mps=None,
    n_runs: int = 10,
) -> dict:
    """Benchmark MLX and optionally PyTorch (CPU / MPS) on *atoms*."""
    result: dict = {"name": name, "n_atoms": len(atoms)}

    # Edge count & NL timing
    result["n_edges"] = _count_edges(atoms, calc_mlx)
    result["nl_ms"] = _time_neighbor_list(atoms, calc_mlx, n_runs=n_runs)

    # ── MLX ──────────────────────────────────────────────────────────────
    atoms_mlx = atoms.copy()
    atoms_mlx.calc = calc_mlx

    for _ in range(2):
        atoms_mlx.calc.results = {}
        _ = atoms_mlx.get_potential_energy()
        _ = atoms_mlx.get_forces()

    mlx_times: list[float] = []
    for _ in range(n_runs):
        atoms_mlx.calc.results = {}
        calc_mlx._nl_cache = None  # force NL rebuild each iteration
        t0 = time.perf_counter()
        _ = atoms_mlx.get_potential_energy()
        _ = atoms_mlx.get_forces()
        t1 = time.perf_counter()
        mlx_times.append(t1 - t0)

    mlx_mean = float(np.mean(mlx_times)) * 1000
    result["mlx_mean_ms"] = mlx_mean
    result["mlx_std_ms"] = float(np.std(mlx_times)) * 1000
    result["mlx_min_ms"] = float(np.min(mlx_times)) * 1000
    result["model_ms"] = max(0.0, mlx_mean - result["nl_ms"])

    # Final MLX values
    atoms_mlx.calc.results = {}
    e_mlx = float(atoms_mlx.get_potential_energy())
    f_mlx = atoms_mlx.get_forces()
    result["e_mlx"] = e_mlx

    # ── PyTorch CPU ──────────────────────────────────────────────────────
    if calc_torch_cpu is not None:
        t_ms, e_cpu, f_cpu = _bench_torch_calc(calc_torch_cpu, atoms, n_runs)
        result["cpu_mean_ms"] = float(np.mean(t_ms))
        result["cpu_std_ms"] = float(np.std(t_ms))
        result["cpu_min_ms"] = float(np.min(t_ms))
        result["cpu_e_diff"] = abs(e_mlx - e_cpu)
        result["cpu_f_max_diff"] = float(np.max(np.abs(f_mlx - f_cpu)))
        result["speedup_cpu"] = (
            result["cpu_mean_ms"] / mlx_mean if mlx_mean > 0 else float("nan")
        )

    # ── PyTorch MPS ──────────────────────────────────────────────────────
    if calc_torch_mps is not None:
        sync_fn = _get_mps_sync()
        t_ms, e_mps, f_mps = _bench_torch_calc(
            calc_torch_mps, atoms, n_runs, sync_fn=sync_fn,
        )
        result["mps_mean_ms"] = float(np.mean(t_ms))
        result["mps_std_ms"] = float(np.std(t_ms))
        result["mps_min_ms"] = float(np.min(t_ms))
        result["mps_e_diff"] = abs(e_mlx - e_mps)
        result["mps_f_max_diff"] = float(np.max(np.abs(f_mlx - f_mps)))
        result["speedup_mps"] = (
            result["mps_mean_ms"] / mlx_mean if mlx_mean > 0 else float("nan")
        )

    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_result(r: dict, has_cpu: bool, has_mps: bool) -> None:
    """Pretty-print one benchmark result (detailed view)."""
    n = r["n_atoms"]
    edges = r.get("n_edges", "?")
    nl = r.get("nl_ms", 0)
    model = r.get("model_ms", 0)
    total = r["mlx_mean_ms"]

    print(f"\n  {r['name']}  ({n} atoms, {edges} edges)")
    print(
        f"    MLX:  NL {nl:.1f} ms | Model {model:.1f} ms | "
        f"Total {total:.1f} ms (± {r['mlx_std_ms']:.1f})"
    )

    if has_cpu and "cpu_mean_ms" in r:
        print(
            f"    CPU:  {r['cpu_mean_ms']:.1f} ms  |  "
            f"MLX vs CPU: {r['speedup_cpu']:.2f}x"
        )
        print(
            f"          dE: {r['cpu_e_diff']:.2e} eV  |  "
            f"dF max: {r['cpu_f_max_diff']:.2e} eV/A"
        )

    if has_mps and "mps_mean_ms" in r:
        print(
            f"    MPS:  {r['mps_mean_ms']:.1f} ms  |  "
            f"MLX vs MPS: {r['speedup_mps']:.2f}x"
        )
        print(
            f"          dE: {r['mps_e_diff']:.2e} eV  |  "
            f"dF max: {r['mps_f_max_diff']:.2e} eV/A"
        )


def print_summary_table(results: list[dict], has_cpu: bool, has_mps: bool) -> None:
    """Print a compact summary table."""

    if has_cpu and has_mps:
        # ── Three-way comparison ─────────────────────────────────────────
        _header()
        print("  MACE-MLX Performance Benchmark — Summary  (MLX vs CPU vs MPS)")
        _header()
        print(
            f"  {'System':<16} {'Atoms':>5} {'Edges':>6} "
            f"{'MLX(ms)':>9} {'CPU(ms)':>9} {'MPS(ms)':>9} "
            f"{'vs CPU':>7} {'vs MPS':>7} "
            f"{'dE(eV)':>10} {'dF max':>10}"
        )
        _separator()
        for r in results:
            print(
                f"  {r['name']:<16} {r['n_atoms']:>5} {r.get('n_edges', 0):>6} "
                f"{r['mlx_mean_ms']:>9.1f} "
                f"{r.get('cpu_mean_ms', float('nan')):>9.1f} "
                f"{r.get('mps_mean_ms', float('nan')):>9.1f} "
                f"{r.get('speedup_cpu', float('nan')):>6.2f}x "
                f"{r.get('speedup_mps', float('nan')):>6.2f}x "
                f"{r.get('cpu_e_diff', float('nan')):>10.2e} "
                f"{r.get('cpu_f_max_diff', float('nan')):>10.2e}"
            )
        _header()

    elif has_cpu:
        # ── Two-way: MLX vs CPU ──────────────────────────────────────────
        _header()
        print("  MACE-MLX Performance Benchmark — Summary  (MLX vs CPU)")
        _header()
        print(
            f"  {'System':<16} {'Atoms':>5} {'Edges':>6} "
            f"{'NL(ms)':>7} {'Model(ms)':>9} {'MLX(ms)':>9} "
            f"{'CPU(ms)':>9} {'vs CPU':>7} "
            f"{'dE(eV)':>10} {'dF max':>10}"
        )
        _separator()
        for r in results:
            print(
                f"  {r['name']:<16} {r['n_atoms']:>5} {r.get('n_edges', 0):>6} "
                f"{r.get('nl_ms', 0):>7.1f} {r.get('model_ms', 0):>9.1f} "
                f"{r['mlx_mean_ms']:>9.1f} "
                f"{r.get('cpu_mean_ms', float('nan')):>9.1f} "
                f"{r.get('speedup_cpu', float('nan')):>6.2f}x "
                f"{r.get('cpu_e_diff', float('nan')):>10.2e} "
                f"{r.get('cpu_f_max_diff', float('nan')):>10.2e}"
            )
        _header()

    else:
        # ── MLX only ─────────────────────────────────────────────────────
        _header()
        print("  MACE-MLX Performance Benchmark — Summary  (MLX only)")
        _header()
        print(
            f"  {'System':<16} {'Atoms':>5} {'Edges':>6} "
            f"{'NL(ms)':>7} {'Model(ms)':>9} {'Total(ms)':>9} {'min(ms)':>8}"
        )
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
            atom_ratio = (
                r["n_atoms"] / base["n_atoms"] if base["n_atoms"] > 0 else 0
            )
            time_ratio = (
                r["mlx_mean_ms"] / base["mlx_mean_ms"]
                if base["mlx_mean_ms"] > 0
                else 0
            )
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
        description="Benchmark MACE-MLX vs PyTorch MACE (CPU & MPS/GPU)",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of timed iterations per system (default: 10)",
    )
    parser.add_argument(
        "--no-torch", action="store_true",
        help="Skip all PyTorch comparisons (MLX only)",
    )
    parser.add_argument(
        "--no-mps", action="store_true",
        help="Skip PyTorch MPS/GPU comparison",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Only benchmark small/medium systems (skip large)",
    )
    args = parser.parse_args()

    has_torch = not args.no_torch
    has_mps = has_torch and not args.no_mps

    # ── Load calculators ───────────────────────────────────────────────
    print("Loading MACE-MLX calculator ...", flush=True)
    from mace_mlx.calculators import MACEMLXCalculator, _USE_MATSCIPY

    calc_mlx = MACEMLXCalculator(model_path="small", skin=0.0)

    calc_torch_cpu = None
    calc_torch_mps = None
    mace_mp = None  # keep reference for MPS loading

    if has_torch:
        print("Loading PyTorch MACE calculator (CPU) ...", flush=True)
        try:
            from mace.calculators import mace_mp as _mace_mp

            mace_mp = _mace_mp
            calc_torch_cpu = mace_mp(
                model="small", device="cpu", default_dtype="float32",
            )
        except ImportError:
            print(
                "WARNING: mace-torch not installed — "
                "skipping PyTorch comparison.\n"
            )
            has_torch = False
            has_mps = False

    if has_mps and mace_mp is not None:
        try:
            import torch

            if torch.backends.mps.is_available():
                print("Loading PyTorch MACE calculator (MPS/GPU) ...", flush=True)
                try:
                    # Direct MPS loading (works if checkpoint is float32)
                    calc_torch_mps = mace_mp(
                        model="small", device="mps", default_dtype="float32",
                    )
                except Exception:
                    # MPS can't handle float64 → load on CPU, convert, move
                    calc_torch_mps = mace_mp(
                        model="small", device="cpu", default_dtype="float32",
                    )
                    mps_dev = torch.device("mps")
                    for m in calc_torch_mps.models:
                        m.to(mps_dev)
                    calc_torch_mps.device = mps_dev
                    print("  (loaded via CPU → MPS transfer)")
            else:
                print("WARNING: MPS not available — skipping GPU comparison.\n")
                has_mps = False
        except Exception as e:
            print(f"WARNING: Failed to load MPS model: {e}\n")
            has_mps = False

    if calc_torch_mps is not None:
        _patch_torch_for_mps()

    # ── Header ─────────────────────────────────────────────────────────
    nl_backend = "matscipy (C)" if _USE_MATSCIPY else "ASE (Python)"
    backends = ["MLX (Apple GPU)"]
    if has_torch:
        backends.append("PyTorch (CPU)")
    if has_mps:
        backends.append("PyTorch (MPS/GPU)")

    print(f"\n{'#' * W}")
    print("  MACE-MLX Performance Benchmark")
    print(f"  Model:           MACE-MP-0 Small")
    print(f"  NL backend:      {nl_backend}")
    print(f"  Runs:            {args.runs} per system (+ 2 warmup)")
    print(f"  Compare:         {' vs '.join(backends)}")
    print(f"{'#' * W}")

    # ── Run benchmarks ─────────────────────────────────────────────────
    systems = build_systems(quick=args.quick)
    results: list[dict] = []

    total = len(systems)
    for idx, (atoms, name) in enumerate(systems, 1):
        print(
            f"\n  [{idx}/{total}] Benchmarking {name} ({len(atoms)} atoms) ...",
            flush=True,
        )
        try:
            r = benchmark_system(
                atoms,
                name,
                calc_mlx=calc_mlx,
                calc_torch_cpu=calc_torch_cpu if has_torch else None,
                calc_torch_mps=calc_torch_mps if has_mps else None,
                n_runs=args.runs,
            )
            print_result(r, has_cpu=has_torch, has_mps=has_mps)
            results.append(r)
        except Exception as exc:
            print(f"\n  {name}: FAILED — {exc}")

    # ── Summary ────────────────────────────────────────────────────────
    if results:
        print_summary_table(results, has_cpu=has_torch, has_mps=has_mps)


if __name__ == "__main__":
    main()
