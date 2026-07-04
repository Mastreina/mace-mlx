"""Prototype D5: (i,k)-row-compressed bilinear sparse symmetric contraction.

Key structure: U2 (P, i, k) has only nrow nonzero (i,k) rows
(lout=0: 99/368, lout=1: 233/816). Rewrite the main contraction as a
bilinear form over those rows:

    X[b,c,r]  = f[b,c,i_r] * W[b,c,k_r]      r = 1..nrow
              = (f @ SelI) * (W_sel @ SelK)   two thin GEMMs + aligned mul
    out[b,c,p] = X @ U_rows                   U_rows[r,p] = U2[p,i_r,k_r]

Everything is GEMM + aligned elementwise: no broadcast 4D tensors, no
gather/scatter, no custom_function (autograd VJPs are transposed GEMMs;
dW link is a dead node under lazy eval when only forces are needed).
FLOPs (lout=1, b=1000, c=128): 46G main + 5G expand vs 163G dense.

Variants: rowc / rowc_c (compile), one per process.
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


def build_rowcomp(contr):
    U = np.array(contr._u_matrices[contr.correlation])
    i_dim, k_dim = U.shape[-2], U.shape[-1]
    P = int(np.prod(U.shape[:-2]))
    U2 = U.reshape(P, i_dim, k_dim)
    row_nz = np.abs(U2).max(axis=0) > 1e-12  # (i_dim, k_dim)
    i_r, k_r = np.nonzero(row_nz)
    nrow = len(i_r)
    SelI = np.zeros((i_dim, nrow), dtype=np.float32)
    SelI[i_r, np.arange(nrow)] = 1.0
    SelK = np.zeros((k_dim, nrow), dtype=np.float32)
    SelK[k_r, np.arange(nrow)] = 1.0
    U_rows = U2[:, i_r, k_r].T.copy().astype(np.float32)  # (nrow, P)
    return dict(
        SelI=mx.stop_gradient(mx.array(SelI)),
        SelK=mx.stop_gradient(mx.array(SelK)),
        U_rows=mx.stop_gradient(mx.array(U_rows)),
        nrow=nrow, P=P,
    )


def main_rowc(s, W_sel, feats):
    X = (feats @ s["SelI"]) * (W_sel @ s["SelK"])  # (b,c,nrow)
    return X @ s["U_rows"]  # (b,c,P)


def sparse_call(contr, s, features, element_onehot, W_max_ck=None):
    contr._ensure_weight_caches()
    b, num_c, _ = features.shape
    W_ck = W_max_ck if W_max_ck is not None else contr._W_max_ck

    k_m = contr.weights_max.shape[1]
    W_sel_ck = (element_onehot @ W_ck).reshape(b, num_c, k_m)
    out = main_rowc(s, W_sel_ck, features)

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


def dense_call(contr, features, element_onehot, W_max_ck):
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
    ap.add_argument("--variant", required=True, choices=["rowc", "rowc_c"])
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
        s = build_rowcomp(contr)
        print(f"  contraction[{ci}] lout={contr.ir_out.l} "
              f"P={s['P']} nrow={s['nrow']}", flush=True)

        fn = lambda f: sparse_call(contr, s, f, oh)
        if args.variant == "rowc_c":
            fn = mx.compile(fn)

        ref = contr(x, oh)
        mx.eval(ref)  # eval BEFORE running variant (isolate any bad kernels)
        out = fn(x)
        mx.eval(out)
        err_f = float(mx.max(mx.abs(out - ref)).item())

        contr._ensure_weight_caches()
        Wck0 = contr._W_max_ck
        df_ref = mx.grad(lambda xx: dense_call(contr, xx, oh, Wck0).sum())(x)
        mx.eval(df_ref)
        df_new = mx.grad(lambda xx: fn(xx).sum())(x)
        mx.eval(df_new)
        err_df = float(mx.max(mx.abs(df_new - df_ref)).item())

        dW_ref = mx.grad(
            lambda ww: dense_call(contr, x, oh, W_max_ck=ww).sum())(Wck0)
        mx.eval(dW_ref)
        dW_new = mx.grad(
            lambda ww: sparse_call(contr, s, x, oh, W_max_ck=ww).sum())(Wck0)
        mx.eval(dW_new)
        err_dw = float(mx.max(mx.abs(dW_new - dW_ref)).item())

        print(f"    err fwd={err_f:.2e}  df={err_df:.2e}  dW={err_dw:.2e}",
              flush=True)

        def fwd():
            mx.eval(fn(x))

        vag = mx.value_and_grad(lambda xx: fn(xx).sum())

        def fwdbwd():
            l, g_ = vag(x)
            mx.eval(l, g_)

        vag2 = mx.value_and_grad(
            lambda xx, ww: sparse_call(contr, s, xx, oh, W_max_ck=ww).sum(),
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
        print(f"    RESULT {args.variant:7s} lout={contr.ir_out.l} "
              f"fwd={t_f:7.2f}ms ({m_f:6.0f}MB)  "
              f"f+b(x)={t_fb:7.2f}ms ({m_fb:6.0f}MB)  "
              f"f+b(x,W)={t_fb2:7.2f}ms ({m_fb2:6.0f}MB)", flush=True)


if __name__ == "__main__":
    main()
