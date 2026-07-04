"""Timing harness for the hand-written-VJP conv_tp prototypes.

One variant per process (mx.compile caches by function object id; fresh
function objects per process avoid trace pollution across variants).
Timing protocol matches teamA: warmup 3 + median of 10, mx.eval sync,
fwd and f+b3 (grads wrt x1, x2, w with loss = out.sum() -- same scalarizer
as the teamA baseline numbers), peak memory via reset/get_peak_memory.

Variants:
  base  current production path: tp._batched_mul21_forward, autograd (=B3)
  loop  tp._loop_forward, autograd (pre-batching baseline)
  b2    fwd_split, autograd  (control: shows fwd gain without custom VJP;
        its autograd backward has per-slot slice VJPs and should be poor)
  v1    fwd_split + hand-written split VJP
  v2    fwd_unified + hand-written unified VJP
Add --compile to wrap both the fwd callable and the value_and_grad in
mx.compile (matches the e2e calculator, which compiles the whole step).

Suggested serial run (from repo root, GPU otherwise idle):
  for v in base b2 v1 v2; do
    .venv/bin/python scratchpad/team_convtp_bench.py --variant $v --E 46000
  done
  ... repeat with --E 9936, and optionally --compile for base/v1/v2.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from team_convtp_vjp import (  # noqa: E402
    MODEL_DIRS, ConvTPConsts, fwd_split, load_tp, make_inputs, make_v1,
    make_v2,
)


def bench(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True,
                    choices=["base", "loop", "b2", "v1", "v2"])
    ap.add_argument("--model", choices=list(MODEL_DIRS), default="medium")
    ap.add_argument("--E", type=int, default=46000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    tp = load_tp(args.model)
    x1, x2, w, _R = make_inputs(tp, args.E, args.seed)

    if args.variant == "base":
        fn = lambda a, b, ww: tp._batched_mul21_forward(a, b, ww)  # noqa: E731
    elif args.variant == "loop":
        fn = lambda a, b, ww: tp._loop_forward(a, b, ww)           # noqa: E731
    else:
        c = ConvTPConsts(tp)
        if args.variant == "b2":
            fn = lambda a, b, ww: fwd_split(c, a, b, ww)           # noqa: E731
        elif args.variant == "v1":
            fn = make_v1(c)
        else:
            fn = make_v2(c)

    # sanity: fwd agreement with the production path (cheap, once)
    ref = tp._batched_mul21_forward(x1, x2, w)
    out = fn(x1, x2, w)
    mx.eval(ref, out)
    maxerr = float(mx.max(mx.abs(out - ref)).item())

    fwd_call = fn
    vag = mx.value_and_grad(
        lambda a, b, ww: fn(a, b, ww).sum(), argnums=(0, 1, 2))
    if args.compile:
        fwd_call = mx.compile(lambda a, b, ww: fn(a, b, ww))
        vag = mx.compile(vag)

    def run_fwd():
        mx.eval(fwd_call(x1, x2, w))

    def run_fb3():
        l, gs = vag(x1, x2, w)
        mx.eval(l, gs)

    mx.reset_peak_memory()
    t_f = bench(run_fwd, n=args.n, warmup=args.warmup)
    m_f = mx.get_peak_memory() / 1e6
    mx.reset_peak_memory()
    t_fb3 = bench(run_fb3, n=args.n, warmup=args.warmup)
    m_fb3 = mx.get_peak_memory() / 1e6

    tag = args.variant + ("+compile" if args.compile else "")
    print(f"RESULT model={args.model} E={args.E} variant={tag:12s} "
          f"maxerr={maxerr:.2e} fwd={t_f:8.2f}ms ({m_f:7.0f}MB)  "
          f"f+b3={t_fb3:8.2f}ms ({m_fb3:7.0f}MB)", flush=True)


if __name__ == "__main__":
    main()
