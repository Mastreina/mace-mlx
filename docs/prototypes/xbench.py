"""Cross-implementation benchmark: mace-torch (cpu/mps) vs mace-mlx
(v0.2.0 / v0.3.0 / current sparse-SC worktree).

One configuration per process. mace-mlx version is selected via
PYTHONPATH (this script lives outside the repo so cwd does not shadow
it); the resolved package path/version is printed and recorded.

Timing: identical to ab_bench.py -- warmup 3 + 10 runs median of
clear-cache energy+forces on the same rattled structure (seed 42).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.build import bulk

OUT = Path(__file__).parent / "xbench_results.json"

SYSTEMS = {
    "Si216": lambda: bulk("Si", "diamond", a=5.43, cubic=True) * (3, 3, 3),
    "Si1000": lambda: bulk("Si", "diamond", a=5.43, cubic=True) * (5, 5, 5),
    "Si2000": lambda: bulk("Si", "diamond", a=5.43, cubic=True) * (5, 5, 10),
}


def save(key, payload):
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[key] = payload
    OUT.write_text(json.dumps(data, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)  # result key prefix
    ap.add_argument("--backend", choices=["torch", "mlx"], required=True)
    ap.add_argument("--device", default="gpu")  # torch: cpu|mps; mlx: gpu
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--model", required=True)
    ap.add_argument("--system", choices=list(SYSTEMS), required=True)
    args = ap.parse_args()
    key = f"{args.tag}/{args.model}/{args.system}"

    version = ""
    try:
        if args.backend == "torch":
            import torch
            from mace.calculators import mace_mp
            import mace
            version = f"mace-torch {getattr(mace, '__version__', '?')}"
            if args.device == "mps":
                # mace-torch does not support MPS out of the box: the fp64
                # checkpoint cannot be mapped to MPS, and forward hardcodes
                # .double() energy accumulation. Workaround (reference
                # numbers only): load on CPU as fp32, move to MPS, and
                # degrade .double() to .float() on MPS tensors.
                _orig_double = torch.Tensor.double

                def _mps_safe_double(self):
                    return (self.float() if self.device.type == "mps"
                            else _orig_double(self))

                torch.Tensor.double = _mps_safe_double
                version += " (MPS workaround)"
                calc = mace_mp(model=args.model, device="cpu",
                               default_dtype=args.dtype)
                if getattr(calc, "models", None):
                    calc.models = [m.to("mps") for m in calc.models]
                if hasattr(calc, "model"):
                    calc.model = calc.model.to("mps")
                calc.device = torch.device("mps")
            else:
                calc = mace_mp(model=args.model, device=args.device,
                               default_dtype=args.dtype)
        else:
            import mace_mlx
            version = (f"mace-mlx {getattr(mace_mlx, '__version__', '?')} "
                       f"@ {Path(mace_mlx.__file__).parent.parent}")
            from mace_mlx.calculators import mace_mp
            calc = mace_mp(model=args.model, default_dtype=args.dtype)
        print(f"loaded: {version}", flush=True)

        atoms = SYSTEMS[args.system]()
        rng = np.random.default_rng(42)
        atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
        atoms.calc = calc

        for _ in range(3):
            calc.results = {}
            atoms.get_potential_energy()
            atoms.get_forces()

        ts = []
        for _ in range(10):
            calc.results = {}
            t0 = time.perf_counter()
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            ts.append(time.perf_counter() - t0)
        e2e_ms = float(np.median(ts)) * 1e3

        print(f"RESULT {key}: e2e={e2e_ms:.1f}ms  E={e:.4f} eV  "
              f"|F|max={np.abs(f).max():.4f}", flush=True)
        save(key, {"e2e_ms": e2e_ms, "energy_ev": float(e),
                   "fmax": float(np.abs(f).max()), "natoms": len(atoms),
                   "version": version, "device": args.device,
                   "dtype": args.dtype})
    except Exception as e:
        print(f"FAILED {key}: {type(e).__name__}: {e}", flush=True)
        save(key, {"error": f"{type(e).__name__}: {e}", "version": version,
                   "device": args.device, "dtype": args.dtype})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
