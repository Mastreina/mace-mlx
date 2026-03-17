"""Comprehensive benchmark of all MACE Foundation Models: MLX vs PyTorch (CPU & MPS/GPU).

Tests every available model across multiple system sizes, reporting:
- Inference time (energy + forces)
- Accuracy vs PyTorch reference
- Per-backend comparison (MLX, CPU, MPS)

Usage:
    python scripts/benchmark_all_models.py --quick --correctness --runs 5
    python scripts/benchmark_all_models.py --models small medium large --runs 5 --correctness
    python scripts/benchmark_all_models.py --no-torch --runs 3
    python scripts/benchmark_all_models.py --no-mps --runs 5
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
    (returns ``.float()``) for tensors already on MPS.
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


def bench_one(
    calc, atoms: Atoms, runs: int = 5, warmup: int = 2, sync_fn=None,
) -> np.ndarray:
    """Benchmark a single (calculator, atoms) pair.

    *sync_fn* is called around each timed iteration to flush async work
    (needed for accurate MPS timing).

    Returns array of per-run times in milliseconds.
    """
    a = atoms.copy()
    a.calc = calc

    # Warmup
    for _ in range(warmup):
        a.calc.results = {}
        _ = a.get_potential_energy()
        _ = a.get_forces()
    if sync_fn:
        sync_fn()

    # Timed runs
    times = []
    for _ in range(runs):
        a.calc.results = {}
        if sync_fn:
            sync_fn()
        t0 = time.perf_counter()
        _ = a.get_potential_energy()
        _ = a.get_forces()
        if sync_fn:
            sync_fn()
        times.append(time.perf_counter() - t0)

    return np.array(times) * 1000  # ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Comprehensive benchmark of all MACE Foundation Models: "
            "MLX vs PyTorch (CPU & MPS/GPU)."
        ),
    )
    parser.add_argument("--models", nargs="*", help="Specific model names to test (default: all)")
    parser.add_argument("--systems", nargs="*", help="Specific system names to test (default: all)")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs per benchmark (default: 5)")
    parser.add_argument("--no-torch", action="store_true", help="Skip all PyTorch comparisons")
    parser.add_argument("--no-mps", action="store_true", help="Skip PyTorch MPS/GPU comparison")
    parser.add_argument("--quick", action="store_true", help="Quick mode: small systems only (<=54 atoms)")
    parser.add_argument("--correctness", action="store_true", help="Also check correctness vs PyTorch CPU")
    args = parser.parse_args()

    has_torch = not args.no_torch
    has_mps = has_torch and not args.no_mps

    # Filter models / systems
    models = ALL_MODELS
    if args.models:
        models = [m for m in models if m[0] in args.models]

    systems = SYSTEMS
    if args.systems:
        systems = [s for s in systems if s[0] in args.systems]
    if args.quick:
        quick_names = {"H2O", "Si-2", "Si-16", "Cu-27", "Si-54"}
        systems = [s for s in systems if s[0] in quick_names]

    from mace_mlx.calculators import MACEMLXCalculator

    calc_torch_fn = None
    if has_torch:
        try:
            from mace.calculators.foundations_models import mace_mp as torch_mace_mp
            calc_torch_fn = torch_mace_mp
        except ImportError:
            print("WARNING: mace-torch not installed -- skipping PyTorch comparison.\n")
            has_torch = False
            has_mps = False

    # Check MPS availability once
    mps_sync = None
    if has_mps:
        try:
            import torch
            if torch.backends.mps.is_available():
                mps_sync = _get_mps_sync()
                _patch_torch_for_mps()
            else:
                print("WARNING: MPS not available — skipping GPU comparison.\n")
                has_mps = False
        except (ImportError, AttributeError):
            print("WARNING: Cannot check MPS — skipping GPU comparison.\n")
            has_mps = False

    # Header
    backends = ["MLX"]
    if has_torch:
        backends.append("CPU")
    if has_mps:
        backends.append("MPS/GPU")
    print("=" * 100)
    print("MACE-MLX Comprehensive Benchmark")
    print(
        f"Models: {len(models)}, Systems: {len(systems)}, "
        f"Runs: {args.runs}, Backends: {' / '.join(backends)}"
    )
    print("=" * 100)

    results: list[dict] = []
    total_models = len(models)

    for m_idx, (model_name, head, desc) in enumerate(models, 1):
        print(f"\n{'─' * 80}")
        print(f"[{m_idx}/{total_models}] Model: {desc}")
        print(f"{'─' * 80}")

        # ── Load MLX calculator ──────────────────────────────────────────
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

        # ── Load PyTorch CPU calculator ──────────────────────────────────
        calc_cpu = None
        if has_torch and calc_torch_fn is not None:
            kw_t: dict = {
                "model": model_name, "device": "cpu",
                "default_dtype": "float32",
            }
            if head:
                kw_t["head"] = head
            try:
                calc_cpu = calc_torch_fn(**kw_t)
            except Exception as e:
                print(f"  PyTorch CPU unavailable: {e}")

        # ── Load PyTorch MPS calculator ──────────────────────────────────
        calc_mps = None
        if has_mps and calc_torch_fn is not None:
            kw_mps: dict = {
                "model": model_name, "device": "mps",
                "default_dtype": "float32",
            }
            if head:
                kw_mps["head"] = head
            try:
                try:
                    calc_mps = calc_torch_fn(**kw_mps)
                except Exception:
                    # MPS can't handle float64 → load on CPU, convert, move
                    import torch
                    kw_cpu: dict = dict(kw_mps, device="cpu")
                    calc_mps = calc_torch_fn(**kw_cpu)
                    mps_dev = torch.device("mps")
                    for m in calc_mps.models:
                        m.to(mps_dev)
                    calc_mps.device = mps_dev
            except Exception as e:
                print(f"  PyTorch MPS unavailable: {e}")
                calc_mps = None

        # ── Column header ────────────────────────────────────────────────
        if calc_cpu and calc_mps:
            hdr = (
                f"  {'System':12s} {'Atoms':>5s} "
                f"{'MLX(ms)':>9s} {'CPU(ms)':>9s} {'MPS(ms)':>9s} "
                f"{'vs CPU':>7s} {'vs MPS':>7s}"
            )
            if args.correctness:
                hdr += f" {'dE(eV)':>10s} {'dF(eV/A)':>10s}"
        elif calc_cpu:
            hdr = (
                f"  {'System':12s} {'Atoms':>5s} "
                f"{'MLX(ms)':>9s} {'CPU(ms)':>9s} {'vs CPU':>7s}"
            )
            if args.correctness:
                hdr += f" {'dE(eV)':>10s} {'dF(eV/A)':>10s}"
        else:
            hdr = f"  {'System':12s} {'Atoms':>5s} {'MLX(ms)':>9s}"
        print(hdr)

        # ── Benchmark each system ────────────────────────────────────────
        for sys_name, sys_fn, _sys_desc in systems:
            atoms = sys_fn()
            n = len(atoms)

            try:
                t_m = bench_one(calc_m, atoms, runs=args.runs)
                mlx_mean = float(np.mean(t_m))

                # CPU
                cpu_mean = None
                ratio_cpu = None
                if calc_cpu:
                    t_cpu = bench_one(calc_cpu, atoms, runs=args.runs)
                    cpu_mean = float(np.mean(t_cpu))
                    ratio_cpu = (
                        cpu_mean / mlx_mean if mlx_mean > 0 else float("nan")
                    )

                # MPS
                mps_mean = None
                ratio_mps = None
                if calc_mps:
                    t_mps = bench_one(
                        calc_mps, atoms, runs=args.runs, sync_fn=mps_sync,
                    )
                    mps_mean = float(np.mean(t_mps))
                    ratio_mps = (
                        mps_mean / mlx_mean if mlx_mean > 0 else float("nan")
                    )

                # Build output line
                if calc_cpu and calc_mps:
                    line = (
                        f"  {sys_name:12s} {n:5d} "
                        f"{mlx_mean:8.1f}ms {cpu_mean:8.1f}ms {mps_mean:8.1f}ms "
                        f"{ratio_cpu:6.2f}x {ratio_mps:6.2f}x"
                    )
                elif calc_cpu:
                    line = (
                        f"  {sys_name:12s} {n:5d} "
                        f"{mlx_mean:8.1f}ms {cpu_mean:8.1f}ms "
                        f"{ratio_cpu:6.2f}x"
                    )
                else:
                    line = f"  {sys_name:12s} {n:5d} {mlx_mean:8.1f}ms"

                # Correctness check (vs CPU reference)
                if args.correctness and calc_cpu:
                    a_m = atoms.copy()
                    a_m.calc = calc_m
                    a_t = atoms.copy()
                    a_t.calc = calc_cpu
                    dE = abs(
                        float(a_m.get_potential_energy())
                        - float(a_t.get_potential_energy())
                    )
                    dF = float(
                        np.max(np.abs(a_m.get_forces() - a_t.get_forces()))
                    )
                    line += f" {dE:10.2e} {dF:10.2e}"

                print(line)
                results.append({
                    "model": model_name,
                    "head": head,
                    "system": sys_name,
                    "n_atoms": n,
                    "mlx_ms": mlx_mean,
                    "cpu_ms": cpu_mean,
                    "mps_ms": mps_mean,
                    "ratio_cpu": ratio_cpu,
                    "ratio_mps": ratio_mps,
                })

            except Exception as e:
                print(f"  {sys_name:12s} {n:5d}  ERROR: {str(e)[:60]}")

    # ── Summary ──────────────────────────────────────────────────────────
    if not results:
        return

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    # CPU aggregate stats
    cpu_ratios = [r["ratio_cpu"] for r in results if r["ratio_cpu"] is not None]
    if cpu_ratios:
        cpu_wins = sum(1 for r in cpu_ratios if r >= 1.0)
        print(f"\nMLX vs CPU:")
        print(f"  MLX wins: {cpu_wins}/{len(cpu_ratios)} "
              f"({cpu_wins / len(cpu_ratios) * 100:.0f}%)")
        print(f"  Speedup range: {min(cpu_ratios):.2f}x — {max(cpu_ratios):.2f}x")
        print(f"  Geometric mean: {np.exp(np.mean(np.log(cpu_ratios))):.2f}x")

    # MPS aggregate stats
    mps_ratios = [r["ratio_mps"] for r in results if r["ratio_mps"] is not None]
    if mps_ratios:
        mps_wins = sum(1 for r in mps_ratios if r >= 1.0)
        print(f"\nMLX vs MPS/GPU:")
        print(f"  MLX wins: {mps_wins}/{len(mps_ratios)} "
              f"({mps_wins / len(mps_ratios) * 100:.0f}%)")
        print(f"  Speedup range: {min(mps_ratios):.2f}x — {max(mps_ratios):.2f}x")
        print(f"  Geometric mean: {np.exp(np.mean(np.log(mps_ratios))):.2f}x")

    # Per-model breakdown
    def _print_per_model(label: str, ratio_key: str) -> None:
        vals = [r[ratio_key] for r in results if r[ratio_key] is not None]
        if not vals:
            return
        print(f"\nPer-model {label}:")
        print(f"  {'Model':25s} {'Min':>6s} {'Median':>7s} {'Max':>6s}")
        print(f"  {'-' * 47}")
        seen: set[tuple] = set()
        for r in results:
            key = (r["model"], r["head"])
            if key in seen:
                continue
            seen.add(key)
            m_ratios = [
                x[ratio_key]
                for x in results
                if (x["model"], x["head"]) == key and x[ratio_key] is not None
            ]
            if not m_ratios:
                continue
            h = f" ({r['head'][:8]})" if r["head"] else ""
            lbl = r["model"] + h
            print(
                f"  {lbl:25s} "
                f"{min(m_ratios):5.2f}x "
                f"{np.median(m_ratios):6.2f}x "
                f"{max(m_ratios):5.2f}x"
            )

    if cpu_ratios:
        _print_per_model("MLX vs CPU", "ratio_cpu")
    if mps_ratios:
        _print_per_model("MLX vs MPS", "ratio_mps")


if __name__ == "__main__":
    main()
