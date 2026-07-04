"""Prototype D4: padded sparse SC with hand-written custom VJP.

Two-stage design:
  stage 1 (plain autograd): T = (W_sel @ V).reshape(b,c,P,w)
      backward dW_sel = dT @ V.T is a plain transposed GEMM, and under
      MLX lazy evaluation it is only computed when weight gradients are
      actually requested (dead node otherwise).
  stage 2 (custom_function): out = g(T, feats) = (T * F).sum(-1),
      F = feats-expansion to the padded (P, width) layout.
      Hand-written VJP:
        df = ((dout[...,None] * T).reshape(b,c,P*w)) @ Sel.T   # segment-sum GEMM
        dT = dout[...,None] * F                                 # recompute F
      No scatter anywhere; take's default scatter-add VJP is never
      triggered because the VJP is ours.

Variants (one per process; compile-cache discipline):
  vjpb       - F via selector GEMM (feats @ Sel)
  vjpb_take  - F via mx.take inside the custom_function (fwd only)
  vjpb_c     - vjpb with mx.compile around the whole contraction call

Checks per contraction:
  fwd max_err vs dense; df max_err vs dense-autograd df;
  dW_sel max_err vs dense-autograd dW_sel (validates the dT link);
  fwd / fwd+bwd(x) / fwd+bwd(x, W_sel) times and peak memory.
"""
import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mace_mlx.model import load_model

CACHE = Path.home() / ".cache" / "mace_mlx"
MODEL_DIRS = {"small": "small", "medium": "medium-mpa-0"}


def bench(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def build_padded(contr, use_take):
    U = np.array(contr._u_matrices[contr.correlation])
    i_dim, k_dim = U.shape[-2], U.shape[-1]
    P = int(np.prod(U.shape[:-2]))
    U2 = U.reshape(P, i_dim, k_dim)
    col_nz = np.abs(U2).max(axis=2) > 1e-12
    width = int(col_nz.sum(axis=1).max())
    V = np.zeros((k_dim, P * width), dtype=np.float32)
    Sel = np.zeros((i_dim, P * width), dtype=np.float32)
    i_flat = np.zeros(P * width, dtype=np.int32)  # pad slots -> 0 (T is 0 there)
    for p in range(P):
        for w, i in enumerate(np.nonzero(col_nz[p])[0]):
            V[:, p * width + w] = U2[p, i, :]
            Sel[i, p * width + w] = 1.0
            i_flat[p * width + w] = i

    V_mx = mx.stop_gradient(mx.array(V))
    Sel_mx = mx.stop_gradient(mx.array(Sel))
    SelT_mx = mx.stop_gradient(mx.array(Sel.T.copy()))
    i_flat_mx = mx.array(i_flat)

    def expand_gemm(feats, b, c):
        return (feats @ Sel_mx).reshape(b, c, P, width)

    def expand_take(feats, b, c):
        return mx.take(feats, i_flat_mx, axis=2).reshape(b, c, P, width)

    expand = expand_take if use_take else expand_gemm

    @mx.custom_function
    def g(T4, feats):
        b, c = feats.shape[0], feats.shape[1]
        F = expand(feats, b, c)
        return (T4 * F).sum(axis=-1)

    @g.vjp
    def g_vjp(primals, cotan, output):
        T4, feats = primals
        b, c = feats.shape[0], feats.shape[1]
        d4 = cotan[..., None]  # (b,c,P,1)
        df = (d4 * T4).reshape(b, c, P * width) @ SelT_mx  # (b,c,i)
        dT = d4 * expand(feats, b, c)  # only evaluated if W grads needed
        return dT, df

    def main_fn(W_sel, feats):
        b, c, _ = W_sel.shape
        T4 = (W_sel @ V_mx).reshape(b, c, P, width)
        return g(T4, feats)

    return main_fn, dict(P=P, width=width)


def sparse_call(contr, main_fn, features, element_onehot, W_max_ck=None):
    """Forward with sparse main step. W_max_ck overrides the cached
    weight matrix so gradients w.r.t. it can be tested."""
    contr._ensure_weight_caches()
    b, num_c, _ = features.shape
    W_ck = W_max_ck if W_max_ck is not None else contr._W_max_ck

    k_m = contr.weights_max.shape[1]
    W_sel_ck = (element_onehot @ W_ck).reshape(b, num_c, k_m)
    out = main_fn(W_sel_ck, features)

    feat_col = features[:, :, :, None]
    for idx in range(contr._unrolled_n_lower):
        k_i = contr.weights[idx].shape[1]
        W_sel_i = (element_onehot @ contr._W_lower_ck[idx]).reshape(b, num_c, k_i)
        c_tensor = W_sel_i @ contr._u_lower_2d_t[idx]
        c_tensor = c_tensor + out
        i_d = contr._unrolled_lower_i_dims[idx]
        prefix_outer = contr._unrolled_lower_prefix_sizes[idx] // i_d
        c_4d = c_tensor.reshape(b, num_c, prefix_outer, i_d)
        out = (c_4d @ feat_col).reshape(b, num_c, prefix_outer)

    return out.reshape(b, -1)


def dense_call(contr, features, element_onehot, W_max_ck=None):
    """Current production path, with optional W_max_ck override (so the
    same weight-gradient test can run against the dense reference)."""
    if W_max_ck is None:
        return contr(features, element_onehot)
    contr._ensure_weight_caches()
    b, num_c, _ = features.shape
    k_m = contr.weights_max.shape[1]
    W_sel_ck = (element_onehot @ W_max_ck).reshape(b, num_c, k_m)
    WU = (W_sel_ck @ contr._u_main_wf).reshape(
        b, num_c, contr._u_main_prefix_size, contr._u_main_i_dim
    )
    feat_col = features[:, :, :, None]
    out = (WU @ feat_col).reshape(b, num_c, contr._u_main_prefix_size)
    for idx in range(contr._unrolled_n_lower):
        k_i = contr.weights[idx].shape[1]
        W_sel_i = (element_onehot @ contr._W_lower_ck[idx]).reshape(b, num_c, k_i)
        c_tensor = W_sel_i @ contr._u_lower_2d_t[idx]
        c_tensor = c_tensor + out
        i_d = contr._unrolled_lower_i_dims[idx]
        prefix_outer = contr._unrolled_lower_prefix_sizes[idx] // i_d
        c_4d = c_tensor.reshape(b, num_c, prefix_outer, i_d)
        out = (c_4d @ feat_col).reshape(b, num_c, prefix_outer)
    return out.reshape(b, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_DIRS), required=True)
    ap.add_argument("--variant", required=True,
                    choices=["vjpb", "vjpb_take", "vjpb_c"])
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()

    model = load_model(str(CACHE / MODEL_DIRS[args.model] / "v2"))
    sc = model.products[0].symmetric_contractions
    b = args.batch
    rng = np.random.default_rng(0)
    feats_flat = mx.array(rng.normal(size=(b, sc.irreps_in.dim)).astype(np.float32))
    z_np = rng.integers(0, sc.num_elements, size=b).astype(np.int32)
    onehot = np.zeros((b, sc.num_elements), dtype=np.float32)
    onehot[np.arange(b), z_np] = 1.0
    oh = mx.array(onehot)

    blocks = []
    for idx, (mul, ir_dim) in enumerate(sc._ir_dims):
        blocks.append(feats_flat[..., sc._slices[idx]].reshape(b, mul, ir_dim))
    x = mx.concatenate(blocks, axis=-1)
    mx.eval(x, oh)

    print(f"== {args.model} products[0], b={b}, variant={args.variant} ==",
          flush=True)
    for ci, contr in enumerate(sc.contractions):
        use_take = args.variant == "vjpb_take"
        main_fn, meta = build_padded(contr, use_take)
        print(f"  contraction[{ci}] lout={contr.ir_out.l} "
              f"P={meta['P']} width={meta['width']}", flush=True)

        fn = lambda f: sparse_call(contr, main_fn, f, oh)
        if args.variant == "vjpb_c":
            fn = mx.compile(fn)

        # --- correctness: fwd, df, dW ---
        ref = contr(x, oh)
        out = fn(x)
        mx.eval(ref, out)
        err_f = float(mx.max(mx.abs(out - ref)).item())

        df_ref = mx.grad(lambda xx: dense_call(contr, xx, oh).sum())(x)
        df_new = mx.grad(lambda xx: fn(xx).sum())(x)
        mx.eval(df_ref, df_new)
        err_df = float(mx.max(mx.abs(df_new - df_ref)).item())
        scale_df = float(mx.max(mx.abs(df_ref)).item())

        contr._ensure_weight_caches()
        Wck0 = contr._W_max_ck
        dW_ref = mx.grad(
            lambda ww: dense_call(contr, x, oh, W_max_ck=ww).sum())(Wck0)
        dW_new = mx.grad(
            lambda ww: sparse_call(contr, main_fn, x, oh, W_max_ck=ww).sum())(Wck0)
        mx.eval(dW_ref, dW_new)
        err_dw = float(mx.max(mx.abs(dW_new - dW_ref)).item())
        scale_dw = float(mx.max(mx.abs(dW_ref)).item())

        print(f"    err fwd={err_f:.2e}  df={err_df:.2e} (|df|max {scale_df:.1f})"
              f"  dW={err_dw:.2e} (|dW|max {scale_dw:.1f})", flush=True)

        # --- perf ---
        def fwd():
            mx.eval(fn(x))

        vag = mx.value_and_grad(lambda xx: fn(xx).sum())

        def fwdbwd():
            l, g_ = vag(x)
            mx.eval(l, g_)

        # grads w.r.t. BOTH x and W (training-like; checks lazy-pruning cost)
        vag2 = mx.value_and_grad(
            lambda xx, ww: sparse_call(contr, main_fn, xx, oh, W_max_ck=ww).sum(),
            argnums=(0, 1))

        def fwdbwd2():
            l, gs = vag2(x, Wck0)
            mx.eval(l, gs)

        mx.reset_peak_memory()
        t_f = bench(fwd)
        m_f = mx.get_peak_memory() / 1e6
        mx.reset_peak_memory()
        t_fb = bench(fwdbwd)
        m_fb = mx.get_peak_memory() / 1e6
        mx.reset_peak_memory()
        t_fb2 = bench(fwdbwd2)
        m_fb2 = mx.get_peak_memory() / 1e6
        print(f"    RESULT {args.variant:9s} lout={contr.ir_out.l} "
              f"fwd={t_f:7.2f}ms ({m_f:6.0f}MB)  "
              f"f+b(x)={t_fb:7.2f}ms ({m_fb:6.0f}MB)  "
              f"f+b(x,W)={t_fb2:7.2f}ms ({m_fb2:6.0f}MB)", flush=True)


if __name__ == "__main__":
    main()
