"""Timing script for the fused Metal kernel prototypes. RUN SERIALLY (one
process per config; GPU must be otherwise idle).

Modes
  --which scx    block-level SC main-contraction A/B (ref = production
                 selector-GEMM sparse path, fused = Metal kernel X)
  --which mul21  block-level layer-1 conv_tp A/B (ref = production
                 _batched_mul21_forward, fused = single-kernel mji)
  --which e2e    full calculator step (energy+forces, NL cache warm) with
                 the fused path(s) monkeypatched in

Timing: warmup 3 + 10 runs median; peak memory reset after warmup and read
after the timed loop. Results appended to team_metal_bench_results.json.

Suggested serial plan (each line = one process):
  python team_metal_bench.py --print-plan
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/mastreina/Desktop/mace-mlx")

RESULTS = HERE / "team_metal_bench_results.json"
MODEL_DIR = "/Users/mastreina/.cache/mace_mlx/medium-mpa-0/v2"


def timed(fn, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    mx.reset_peak_memory()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3, mx.get_peak_memory() / 1e6


def save(key, ms, peak_mb, extra=None):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = {"ms": ms, "peak_mb": peak_mb, **(extra or {})}
    RESULTS.write_text(json.dumps(data, indent=1))
    print(f"RESULT {key}: {ms:.2f} ms  peak={peak_mb:.0f} MB", flush=True)


# ---------------------------------------------------------------- scx block
def bench_scx(args):
    import team_metal_proto_scx as P

    con = P.build_contraction(args.lout)
    fused = P.make_fused_x(con) if args.variant == "fused" else None
    rng = np.random.default_rng(1)
    b = args.b
    features = mx.array(rng.normal(size=(b, 128, 16)).astype(np.float32))
    onehot_np = np.zeros((b, 89), dtype=np.float32)
    onehot_np[np.arange(b), rng.integers(0, 89, size=b)] = 1.0
    onehot = mx.array(onehot_np)
    cot = mx.array(
        rng.normal(size=(b, 128 * (2 * args.lout + 1))).astype(np.float32)
    )
    mx.eval(features, onehot, cot)

    if args.variant == "fused":
        def fwd_fn(x):
            return P.call_unrolled_fused(con, fused, x, onehot)
    else:
        def fwd_fn(x):
            return con(x, onehot)

    if args.mode == "fwd":
        f = fwd_fn
        if args.compile:
            f = mx.compile(f)

        def step():
            mx.eval(f(features))
    else:  # fwdbwd: force-style, d/dfeatures only (dW branch dead)
        vag = mx.value_and_grad(lambda x: (fwd_fn(x) * cot).sum())
        if args.compile:
            vag = mx.compile(vag)

        def step():
            v, g = vag(features)
            mx.eval(v, g)

    ms, peak = timed(step)
    save(f"scx/lout{args.lout}/b{args.b}/{args.mode}/"
         f"{'c' if args.compile else 'nc'}/{args.variant}", ms, peak)


# -------------------------------------------------------------- mul21 block
def bench_mul21(args):
    import team_metal_proto_mul21 as P
    from mace_mlx.model import load_model

    model = load_model(MODEL_DIR)
    tp = model.interactions[1].conv_tp
    assert tp._batched_mul21
    fused = P.make_fused_mul21(tp) if args.variant == "fused" else None

    rng = np.random.default_rng(2)
    E = args.edges
    x1 = mx.array(rng.normal(size=(E, tp.irreps_in1.dim)).astype(np.float32))
    x2 = mx.array(rng.normal(size=(E, tp.irreps_in2.dim)).astype(np.float32))
    w = mx.array(rng.normal(size=(E, tp.weight_numel)).astype(np.float32) * 0.3)
    outdim = P._slot_metadata(tp)[1]
    cot = mx.array(rng.normal(size=(E, outdim)).astype(np.float32))
    mx.eval(x1, x2, w, cot)

    fn = fused if args.variant == "fused" else tp

    if args.mode == "fwd":
        f = (lambda a, b2, c: fn(a, b2, c))
        if args.compile:
            f = mx.compile(f)

        def step():
            mx.eval(f(x1, x2, w))
    else:  # fwdbwd: all three inputs live (force path)
        vag = mx.value_and_grad(
            lambda a, b2, c: (fn(a, b2, c) * cot).sum(), argnums=(0, 1, 2)
        )
        if args.compile:
            vag = mx.compile(vag)

        def step():
            v, g = vag(x1, x2, w)
            mx.eval(v, g)

    ms, peak = timed(step)
    save(f"mul21/E{args.edges}/{args.mode}/"
         f"{'c' if args.compile else 'nc'}/{args.variant}", ms, peak)


# --------------------------------------------------------------------- e2e
def _patch_scx(calc):
    import team_metal_proto_scx as P
    from mace_mlx.symmetric_contraction import Contraction

    orig = Contraction._call_unrolled

    def patched(self, features, element_onehot):
        fx = getattr(self, "_fused_x", None)
        if fx is None or features.shape[0] == 0:
            return orig(self, features, element_onehot)
        return P.call_unrolled_fused(self, fx, features, element_onehot)

    Contraction._call_unrolled = patched
    n = 0
    for prod in calc.model.products:
        for con in prod.symmetric_contractions.contractions:
            if con._use_sparse_main:
                con._fused_x = P.make_fused_x(con)
                n += 1
    print(f"patched {n} SC contractions with fused X kernel", flush=True)


def _patch_mul21(calc):
    import team_metal_proto_mul21 as P
    from mace_mlx.tensor_product import TensorProduct

    orig = TensorProduct.__call__

    def patched(self, x1, x2, weight=None):
        f = getattr(self, "_fused_m21", None)
        if f is not None and weight is not None and x1.ndim == 2 and x1.shape[0] > 0:
            return f(x1, x2, weight)
        return orig(self, x1, x2, weight)

    TensorProduct.__call__ = patched
    n = 0
    for inter in calc.model.interactions:
        tp = inter.conv_tp
        if getattr(tp, "_batched_mul21", False):
            tp._fused_m21 = P.make_fused_mul21(tp)
            n += 1
    print(f"patched {n} conv_tp blocks with fused mul21 kernel", flush=True)


def bench_e2e(args):
    from ase.build import bulk

    from mace_mlx.calculators import mace_mp

    systems = {
        "Si216": lambda: bulk("Si", "diamond", a=5.43, cubic=True) * (3, 3, 3),
        "Si1000": lambda: bulk("Si", "diamond", a=5.43, cubic=True) * (5, 5, 5),
        "Si2000": lambda: bulk("Si", "diamond", a=5.43, cubic=True) * (5, 5, 10),
    }
    calc = mace_mp(model=args.model, default_dtype=args.dtype)
    if args.variant in ("scx", "both"):
        _patch_scx(calc)
    if args.variant in ("mul21", "both"):
        _patch_mul21(calc)

    atoms = systems[args.system]()
    rng = np.random.default_rng(42)
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
    atoms.calc = calc

    def step():
        calc.results = {}
        atoms.get_potential_energy()
        atoms.get_forces()

    ms, peak = timed(step)
    e = float(atoms.get_potential_energy())
    fmax = float(np.abs(atoms.get_forces()).max())
    save(f"e2e/{args.model}/{args.system}/{args.dtype}/{args.variant}",
         ms, peak, {"energy_eV": e, "fmax": fmax, "natoms": len(atoms)})


PLAN = """\
# block-level (fwd and fwd+bwd, compiled, ref vs fused)
{py} {me} --which scx --lout 0 --mode fwd --variant ref
{py} {me} --which scx --lout 0 --mode fwd --variant fused
{py} {me} --which scx --lout 0 --mode fwdbwd --variant ref
{py} {me} --which scx --lout 0 --mode fwdbwd --variant fused
{py} {me} --which scx --lout 1 --mode fwd --variant ref
{py} {me} --which scx --lout 1 --mode fwd --variant fused
{py} {me} --which scx --lout 1 --mode fwdbwd --variant ref
{py} {me} --which scx --lout 1 --mode fwdbwd --variant fused
{py} {me} --which mul21 --mode fwd --variant ref
{py} {me} --which mul21 --mode fwd --variant fused
{py} {me} --which mul21 --mode fwdbwd --variant ref
{py} {me} --which mul21 --mode fwdbwd --variant fused
# end-to-end (medium/Si1000; ref -> single-patch -> both)
{py} {me} --which e2e --variant ref
{py} {me} --which e2e --variant scx
{py} {me} --which e2e --variant mul21
{py} {me} --which e2e --variant both
# optional: larger system for the memory story
{py} {me} --which e2e --system Si2000 --variant ref
{py} {me} --which e2e --system Si2000 --variant both
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["scx", "mul21", "e2e"])
    ap.add_argument("--variant",
                    choices=["ref", "fused", "scx", "mul21", "both"],
                    default="ref")
    ap.add_argument("--mode", choices=["fwd", "fwdbwd"], default="fwdbwd")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--lout", type=int, default=1, choices=[0, 1])
    ap.add_argument("--b", type=int, default=1000)
    ap.add_argument("--edges", type=int, default=46000)
    ap.add_argument("--model", default="medium-mpa-0")
    ap.add_argument("--system", default="Si1000",
                    choices=["Si216", "Si1000", "Si2000"])
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16"])
    ap.add_argument("--print-plan", action="store_true")
    args = ap.parse_args()

    if args.print_plan:
        py = "/Users/mastreina/Desktop/mace-mlx/.venv/bin/python"
        print(PLAN.format(py=py, me=str(Path(__file__).resolve())))
        return

    if args.which == "scx":
        bench_scx(args)
    elif args.which == "mul21":
        bench_mul21(args)
    elif args.which == "e2e":
        bench_e2e(args)
    else:
        raise SystemExit("--which required (or --print-plan)")


if __name__ == "__main__":
    main()
