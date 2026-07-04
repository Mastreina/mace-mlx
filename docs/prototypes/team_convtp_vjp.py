"""Hand-written VJP prototype for the batched layer-2 conv_tp (_batched_mul21).

Motivation (teamA_convtp_batch.md section 5): the B3 batched forward is 69.7 ms
but f+b3 (grads wrt x1, x2, w) is 249 ms @ E=46000 -- the autograd backward
dominates. Traffic breakdown of the autograd backward shows ~60% of backward
bytes go into the assembly slice-VJPs (10 zero-pads to full (E,mul,K) size plus
9 accumulation adds, ~18 GB) and duplicated elementwise materializations.

The hand-written VJP replaces:
  - 10 pad + 9 add assembly VJP  ->  10 slice-reshapes + 2 concats (inverse
    permutation of the forward assembly), ~1.9 GB instead of ~18 GB;
  - the multiply-VJP + broadcast-VJP chains of the scalar group  ->  two
    aligned elementwise products consumed by batched GEMMs (k- and u-
    reductions become bmm against xs / x1s instead of materialize+sum);
  - saved big residuals (A, W0, W1, P0)  ->  recomputation from primals via
    3 cheap GEMMs (2.6 GB traffic, ~6 GFLOP), enabling a B2-style forward
    (fastest measured fwd: 45.9 ms vs B3 69.7 ms) with no autograd penalty.

Two variants:
  v1 "split":   scalar group and l>0 group kept separate (mirrors B2/B3
                structure). Backward = 6 bmm/GEMM + 5 big elementwise.
  v2 "unified": strategy-C style single bmm over x1_cat (E,mul,1+d1) and
                Mf (E,1+d1,Ktot); fewer kernels, slightly more bytes.

All closure-captured constants are numpy-built leaf mx.arrays, mx.eval'ed at
construction (defense against the MLX 0.31.2 compile lazy-capture bug, see
docs/prototypes/repro_compile_lazy_capture.py).

Check mode (this file as a script): numerical validation of fwd / dx1 / dx2 /
dw for v1, v2, and their mx.compile'd forms against autograd references
(_batched_mul21_forward always; _loop_forward with --loop-ref). Uses a fixed
random projection R for the loss so cotangents are generic, and reports
max|delta| / max|ref| per tensor. Threshold 1e-5.

Timing lives in team_convtp_bench.py (one variant per process).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

CACHE = Path.home() / ".cache" / "mace_mlx"
MODEL_DIRS = {"medium": "medium-mpa-0", "large": "large"}


class ConvTPConsts:
    """Constants for the hand-written-VJP variants, derived from a
    TensorProduct that passed the _batched_mul21 guards.

    Prototype-only extra assumptions (asserted):
      - x1 layout is [scalar block | l>0 block], adjacent and covering, so
        dx1 assembles with one concat (true for medium/large layer 2;
        landing code would fall back to zero-pad otherwise);
      - within each group, segment offsets are increasing in output-slot
        order (so cotangent regrouping is a plain slot-order concat).
    """

    def __init__(self, tp):
        assert tp._batched_mul21, "TP did not take the batched_mul21 path"
        self.n = tp._bm21_n
        self.mul = tp._bm21_mul
        self.d1 = tp._bm21_d1
        self.K0 = tp._bm21_K0
        self.K1 = tp._bm21_K1
        self.Ktot = self.K0 + self.K1
        self.sl0 = tp._bm21_sl_scal
        self.sl1 = tp._bm21_sl_m21
        self.x1_dim = tp.irreps_in1.dim
        self.x2_dim = tp.irreps_in2.dim
        self.out_dim = tp.irreps_out.dim
        self.wn = tp.weight_numel
        assert self.wn == self.n * self.mul

        # x1 layout guard: [scal | m21] adjacent, covering
        assert self.sl0 is not None, "prototype assumes a scalar group"
        assert self.sl0.start == 0
        assert self.sl0.stop == self.sl1.start
        assert self.sl1.stop == self.x1_dim
        assert self.sl0.stop - self.sl0.start == self.mul
        assert self.sl1.stop - self.sl1.start == self.mul * self.d1

        S = np.array(tp._bm21_S, dtype=np.float32)
        G1 = np.array(tp._bm21_G1, dtype=np.float32)
        T0 = np.array(tp._bm21_T0, dtype=np.float32)
        T1 = np.array(tp._bm21_T1, dtype=np.float32)

        # slot_meta: (is_scal, group_off, d, out_col_start, ktot_col, inst)
        self.slot_meta: list[tuple[bool, int, int, int, int, int]] = []
        GF = np.zeros((self.x2_dim, (1 + self.d1) * self.Ktot), np.float32)
        Tf = np.zeros((self.n, self.Ktot), np.float32)
        cs = 0
        col = 0
        exp_off = {True: 0, False: 0}  # group offsets must be slot-ordered
        for is_scal, o, d in tp._bm21_slots:
            assert o == exp_off[is_scal], "group offsets not slot-ordered"
            exp_off[is_scal] += d
            if is_scal:
                inst = int(np.argmax(T0[:, o]))
                GF[:, col:col + d] = S[:, o:o + d]
                Tf[inst, col:col + d] = 1.0
            else:
                inst = int(np.argmax(T1[:, o]))
                for m in range(self.d1):
                    GF[:, (1 + m) * self.Ktot + col:
                       (1 + m) * self.Ktot + col + d] = \
                        G1[:, m * self.K1 + o: m * self.K1 + o + d]
                Tf[inst, col:col + d] = 1.0
            self.slot_meta.append((is_scal, o, d, cs, col, inst))
            cs += self.mul * d
            col += d
        assert cs == self.out_dim and col == self.Ktot

        # All closure constants as leaf arrays, eval'ed NOW (MLX 0.31.2
        # compile lazy-capture bug defense). Transposed copies prebuilt so
        # no derived array is ever captured un-evaluated.
        def leaf(a):
            return mx.array(np.ascontiguousarray(a.astype(np.float32)))

        self.S, self.St = leaf(S), leaf(S.T)
        self.G1, self.G1t = leaf(G1), leaf(G1.T)
        self.T0, self.T0t = leaf(T0), leaf(T0.T)
        self.T1, self.T1t = leaf(T1), leaf(T1.T)
        self.GF, self.GFt = leaf(GF), leaf(GF.T)
        self.Tf, self.Tft = leaf(Tf), leaf(Tf.T)
        mx.eval(self.S, self.St, self.G1, self.G1t, self.T0, self.T0t,
                self.T1, self.T1t, self.GF, self.GFt, self.Tf, self.Tft)


# ---------------------------------------------------------------------------
# forwards (plain, autograd-differentiable)
# ---------------------------------------------------------------------------

def fwd_split(c: ConvTPConsts, x1, x2, w):
    """B2-style forward: x2-first contraction, per-slot segment weights.

    Fastest measured forward (45.9 ms vs B3 69.7 ms @ E=46000): weights are
    applied on the (E,mul,d) slices during assembly, skipping the W0/W1
    expansion GEMMs and the big pre-assembly elementwise products. Its
    autograd backward is poor (per-slot slice VJPs) -- which is exactly
    what the hand-written VJP replaces.
    """
    E = x1.shape[0]
    wr = w.reshape(E, c.n, c.mul)
    M1 = (x2 @ c.G1).reshape(E, c.d1, c.K1)
    x1m = x1[:, c.sl1].reshape(E, c.mul, c.d1)
    A = x1m @ M1                                   # (E, mul, K1)
    xs = x2 @ c.S                                  # (E, K0)
    x1s = x1[:, c.sl0]                             # (E, mul)
    P0 = x1s[:, :, None] * xs[:, None, :]          # (E, mul, K0)
    parts = []
    for is_scal, o, d, _cs, _col, inst in c.slot_meta:
        src = P0 if is_scal else A
        seg = src[:, :, o:o + d] * wr[:, inst, :, None]
        parts.append(seg.reshape(E, c.mul * d))
    return mx.concatenate(parts, axis=-1)


def fwd_unified(c: ConvTPConsts, x1, x2, w):
    """Strategy-C style forward: one bmm covers both groups."""
    E = x1.shape[0]
    wr = w.reshape(E, c.n, c.mul)
    Mf = (x2 @ c.GF).reshape(E, 1 + c.d1, c.Ktot)
    x1c = mx.concatenate(
        [x1[:, c.sl0].reshape(E, c.mul, 1),
         x1[:, c.sl1].reshape(E, c.mul, c.d1)], axis=-1)   # (E, mul, 1+d1)
    B = x1c @ Mf                                            # (E, mul, Ktot)
    parts = []
    for _is_scal, _o, d, _cs, col, inst in c.slot_meta:
        seg = B[:, :, col:col + d] * wr[:, inst, :, None]
        parts.append(seg.reshape(E, c.mul * d))
    return mx.concatenate(parts, axis=-1)


# ---------------------------------------------------------------------------
# hand-written VJPs
# ---------------------------------------------------------------------------

def _regroup_split(c: ConvTPConsts, g, E):
    """Inverse of the forward assembly: dout (E, out_dim) -> g0, g1.

    Within each group the segment offsets increase in slot order (asserted
    at construction), so a slot-order concat reproduces the group layout.
    """
    g0_parts, g1_parts = [], []
    for is_scal, _o, d, cs, _col, _inst in c.slot_meta:
        gi = g[:, cs:cs + c.mul * d].reshape(E, c.mul, d)
        (g0_parts if is_scal else g1_parts).append(gi)
    g0 = mx.concatenate(g0_parts, axis=-1)   # (E, mul, K0)
    g1 = mx.concatenate(g1_parts, axis=-1)   # (E, mul, K1)
    return g0, g1


def make_v1(c: ConvTPConsts):
    """split fwd + hand-written split VJP.

    Backward math (path weights are baked into S/G1; T0/T1 are 0/1):
      out1[e,u,K] = w1[e,u,K] * sum_m x1m[e,u,m] M1[e,m,K],  w1 = w_t @ T1
      out0[e,u,k] = w0[e,u,k] * x1s[e,u] * xs[e,k],          w0 = w_t @ T0

      dx1m = (g1*w1) @ M1^T                    bmm      (E,mul,d1)
      dM1  = x1m^T @ (g1*w1)                   bmm      (E,d1,K1)
      dwt1 = (g1*A) @ T1^T,  A = x1m @ M1      GEMM     (E,mul,n)
      dx1s = (g0*w0) @ xs[:,:,None]            bmm      (E,mul)
      dxs  = x1s[:,None,:] @ (g0*w0)           bmm      (E,K0)
      dwt0 = ((g0*xs) @ T0^T) * x1s[:,:,None]  GEMM+ew  (E,mul,n)
      dx2  = dM1.flat @ G1^T + dxs @ S^T
      dw   = (dwt1+dwt0).transpose(0,2,1).flat
      dx1  = concat([dx1s, dx1m.flat])         (adjacent covering slices)
    """
    @mx.custom_function
    def f(x1, x2, w):
        return fwd_split(c, x1, x2, w)

    @f.vjp
    def f_vjp(primals, g, _output):
        x1, x2, w = primals
        E = x1.shape[0]
        g0, g1 = _regroup_split(c, g, E)

        # recomputed cheap intermediates (no residuals saved)
        w_t = w.reshape(E, c.n, c.mul).transpose(0, 2, 1)   # (E, mul, n)
        M1 = (x2 @ c.G1).reshape(E, c.d1, c.K1)
        x1m = x1[:, c.sl1].reshape(E, c.mul, c.d1)
        xs = x2 @ c.S
        x1s = x1[:, c.sl0]

        # l>0 group
        W1 = w_t @ c.T1                                     # (E, mul, K1)
        G1W = g1 * W1
        dx1m = G1W @ M1.transpose(0, 2, 1)                  # (E, mul, d1)
        dM1 = x1m.transpose(0, 2, 1) @ G1W                  # (E, d1, K1)
        A = x1m @ M1
        dwt1 = (g1 * A) @ c.T1t                             # (E, mul, n)

        # scalar group
        W0 = w_t @ c.T0                                     # (E, mul, K0)
        G0W = g0 * W0
        dx1s = (G0W @ xs[:, :, None])[:, :, 0]              # (E, mul)
        dxs = (x1s[:, None, :] @ G0W)[:, 0, :]              # (E, K0)
        dwt0 = ((g0 * xs[:, None, :]) @ c.T0t) * x1s[:, :, None]

        dw = (dwt1 + dwt0).transpose(0, 2, 1).reshape(E, c.wn)
        dx2 = dM1.reshape(E, c.d1 * c.K1) @ c.G1t + dxs @ c.St
        dx1 = mx.concatenate([dx1s, dx1m.reshape(E, c.mul * c.d1)], axis=-1)
        return dx1, dx2, dw

    return f


def make_v2(c: ConvTPConsts):
    """unified fwd + hand-written unified VJP (fewer, bigger kernels).

      B = x1c @ Mf; out[slot] = B_seg * w_inst
      dx1c = (G*Wf) @ Mf^T          bmm   (E,mul,1+d1)
      dMf  = x1c^T @ (G*Wf)         bmm   (E,1+d1,Ktot)
      dw_t = (G*B) @ Tf^T           GEMM  (E,mul,n)
      dx2  = dMf.flat @ GF^T
    """
    @mx.custom_function
    def f(x1, x2, w):
        return fwd_unified(c, x1, x2, w)

    @f.vjp
    def f_vjp(primals, g, _output):
        x1, x2, w = primals
        E = x1.shape[0]
        parts = [g[:, cs:cs + c.mul * d].reshape(E, c.mul, d)
                 for _s, _o, d, cs, _col, _i in c.slot_meta]
        G = mx.concatenate(parts, axis=-1)                  # (E, mul, Ktot)

        w_t = w.reshape(E, c.n, c.mul).transpose(0, 2, 1)
        Mf = (x2 @ c.GF).reshape(E, 1 + c.d1, c.Ktot)
        x1c = mx.concatenate(
            [x1[:, c.sl0].reshape(E, c.mul, 1),
             x1[:, c.sl1].reshape(E, c.mul, c.d1)], axis=-1)

        Wf = w_t @ c.Tf                                     # (E, mul, Ktot)
        GW = G * Wf
        dx1c = GW @ Mf.transpose(0, 2, 1)                   # (E, mul, 1+d1)
        dMf = x1c.transpose(0, 2, 1) @ GW                   # (E, 1+d1, Ktot)
        dx2 = dMf.reshape(E, (1 + c.d1) * c.Ktot) @ c.GFt
        B = x1c @ Mf
        dw = ((G * B) @ c.Tft).transpose(0, 2, 1).reshape(E, c.wn)
        dx1 = mx.concatenate(
            [dx1c[:, :, 0], dx1c[:, :, 1:].reshape(E, c.mul * c.d1)],
            axis=-1)
        return dx1, dx2, dw

    return f


# ---------------------------------------------------------------------------
# construction helpers shared with the bench script
# ---------------------------------------------------------------------------

def load_tp(model_name: str):
    from mace_mlx.model import load_model
    model = load_model(str(CACHE / MODEL_DIRS[model_name] / "v2"))
    tp = model.interactions[1].conv_tp
    assert tp._batched_mul21
    return tp


def make_inputs(tp, E: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1 = mx.array(rng.normal(size=(E, tp.irreps_in1.dim)).astype(np.float32))
    x2 = mx.array(rng.normal(size=(E, tp.irreps_in2.dim)).astype(np.float32))
    w = mx.array(rng.normal(size=(E, tp.weight_numel)).astype(np.float32))
    R = mx.array(rng.normal(size=(E, tp.irreps_out.dim)).astype(np.float32))
    mx.eval(x1, x2, w, R)
    return x1, x2, w, R


def relerr(a: mx.array, b: mx.array) -> float:
    d = float(mx.max(mx.abs(a - b)).item())
    s = float(mx.max(mx.abs(b)).item())
    return d / (s + 1e-30)


# ---------------------------------------------------------------------------
# numerical check
# ---------------------------------------------------------------------------

def run_check(model_name: str, E: int, seed: int, loop_ref: bool) -> bool:
    t_start = time.perf_counter()
    tp = load_tp(model_name)
    c = ConvTPConsts(tp)
    x1, x2, w, R = make_inputs(tp, E, seed)
    print(f"== check {model_name} inter[1].conv_tp  E={E}  mul={c.mul}  "
          f"seed={seed} ==", flush=True)

    def loss_of(fn):
        return lambda a, b, ww: (fn(a, b, ww) * R).sum()

    # reference: autograd through the production batched forward
    ref_fn = tp._batched_mul21_forward
    ref_out = ref_fn(x1, x2, w)
    mx.eval(ref_out)
    vag_ref = mx.value_and_grad(loss_of(ref_fn), argnums=(0, 1, 2))
    lr, (dx1_r, dx2_r, dw_r) = vag_ref(x1, x2, w)
    mx.eval(lr, dx1_r, dx2_r, dw_r)
    print(f"  [ref bm21 autograd done, {time.perf_counter()-t_start:.1f}s]",
          flush=True)

    if loop_ref:
        loop_out = tp._loop_forward(x1, x2, w)
        mx.eval(loop_out)
        e = relerr(ref_out, loop_out)
        print(f"  bm21-fwd vs loop-fwd            relerr {e:.2e}", flush=True)
        vag_loop = mx.value_and_grad(
            loss_of(tp._loop_forward), argnums=(0, 1, 2))
        ll, (dx1_l, dx2_l, dw_l) = vag_loop(x1, x2, w)
        mx.eval(ll, dx1_l, dx2_l, dw_l)
        print(f"  bm21-grads vs loop-grads        dx1 {relerr(dx1_r, dx1_l):.2e}"
              f"  dx2 {relerr(dx2_r, dx2_l):.2e}"
              f"  dw {relerr(dw_r, dw_l):.2e}", flush=True)

    ok = True
    variants = {
        "v1": make_v1(c),
        "v2": make_v2(c),
    }
    for name, fn in variants.items():
        for compiled in (False, True):
            tag = name + ("+compile" if compiled else "        ")
            vag = mx.value_and_grad(loss_of(fn), argnums=(0, 1, 2))
            if compiled:
                vag = mx.compile(vag)
                out = mx.compile(lambda a, b, ww: fn(a, b, ww))(x1, x2, w)
            else:
                out = fn(x1, x2, w)
            mx.eval(out)
            e_f = relerr(out, ref_out)
            lv, (dx1_v, dx2_v, dw_v) = vag(x1, x2, w)
            mx.eval(lv, dx1_v, dx2_v, dw_v)
            e_l = abs(float(lv.item()) - float(lr.item())) / (
                abs(float(lr.item())) + 1e-30)
            e_x1 = relerr(dx1_v, dx1_r)
            e_x2 = relerr(dx2_v, dx2_r)
            e_w = relerr(dw_v, dw_r)
            good = max(e_f, e_x1, e_x2, e_w) < 1e-5
            ok &= good
            flag = "OK " if good else "FAIL"
            print(f"  {tag}  fwd {e_f:.2e}  loss {e_l:.2e}  "
                  f"dx1 {e_x1:.2e}  dx2 {e_x2:.2e}  dw {e_w:.2e}  [{flag}]",
                  flush=True)

    # nan/inf guard (compile lazy-capture bug would show up here)
    for arr, nm in [(out, "fwd"), (dx1_v, "dx1"), (dx2_v, "dx2"),
                    (dw_v, "dw")]:
        n_bad = int(mx.sum(~mx.isfinite(arr)).item())
        if n_bad:
            print(f"  WARNING: {nm} has {n_bad} non-finite values", flush=True)
            ok = False

    print(f"  [total {time.perf_counter()-t_start:.1f}s]  "
          f"{'ALL OK' if ok else 'FAILURES PRESENT'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(MODEL_DIRS), default="medium")
    ap.add_argument("--E", type=int, nargs="+", default=[9936, 46000])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loop-ref", action="store_true",
                    help="also cross-check against _loop_forward autograd")
    args = ap.parse_args()

    all_ok = True
    for E in args.E:
        all_ok &= run_check(args.model, E, args.seed, args.loop_ref)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
