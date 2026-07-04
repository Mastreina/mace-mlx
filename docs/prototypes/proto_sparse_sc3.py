"""Prototype D3: selector-matrix sparse symmetric contraction (no gather).

Replaces the dense main-contraction GEMM with constant 0/1 selector-matrix
GEMMs so that autograd's VJP is a transposed GEMM (segment-sum) instead of
the scatter-add produced for mx.take (the reason all v1/v2 variants lost
on backward).

Variants (run ONE per process to avoid mx.compile's function-id cache
pollution; see teamA_convtp_batch.md measurement notes):
  dense    - current production path (baseline)
  dense_c  - baseline + mx.compile
  sela     - v2 column compression with both takes replaced by selector
             GEMMs: G = (W@U_cols) * (f@SelF); out = G @ Agg (ncol,P).
             Expected to lose on lout=1 FLOPs; measured for the record.
  selb     - padded per-p layout: for each prefix slot p, at most `width`
             nonzero (i, k-vector) columns (width computed from U's nnz):
               T = (W_sel @ V).reshape(b,c,P,width)    V   (k, P*width)
               F = (feats @ Sel).reshape(b,c,P,width)  Sel (i, P*width) 0/1
               out = (T * F).sum(-1)
             Pure GEMM + elementwise; no take/scatter anywhere.
  selb_c   - selb + mx.compile

Timing: warmup 3 + 10 runs median, mx.eval each call. Peak memory via
mx.reset_peak_memory()/mx.get_peak_memory(), measured separately for
fwd and fwd+bwd (T and F are saved by autograd for backward, so the
fwd+bwd peak is the number that matters).
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


def build_colsel(contr):
    """Variant a: column compression + selector GEMMs (no takes)."""
    U = np.array(contr._u_matrices[contr.correlation])
    i_dim, k_dim = U.shape[-2], U.shape[-1]
    P = int(np.prod(U.shape[:-2]))
    U2 = U.reshape(P, i_dim, k_dim)
    U_wf = np.moveaxis(U2, -1, 0).reshape(k_dim, P * i_dim)
    cols = np.nonzero(np.abs(U_wf).max(axis=0) > 1e-12)[0]
    ncol = len(cols)
    p_of_col = cols // i_dim
    i_of_col = cols % i_dim
    SelF = np.zeros((i_dim, ncol), dtype=np.float32)
    SelF[i_of_col, np.arange(ncol)] = 1.0
    Agg = np.zeros((ncol, P), dtype=np.float32)
    Agg[np.arange(ncol), p_of_col] = 1.0
    return dict(
        U_cols=mx.array(U_wf[:, cols].astype(np.float32)),
        SelF=mx.array(SelF), Agg=mx.array(Agg), ncol=ncol, P=P,
    )


def build_padded(contr):
    """Variant b: padded per-p layout, width computed from U's nnz."""
    U = np.array(contr._u_matrices[contr.correlation])
    i_dim, k_dim = U.shape[-2], U.shape[-1]
    P = int(np.prod(U.shape[:-2]))
    U2 = U.reshape(P, i_dim, k_dim)
    col_nz = np.abs(U2).max(axis=2) > 1e-12  # (P, i_dim)
    width = int(col_nz.sum(axis=1).max())
    V = np.zeros((k_dim, P * width), dtype=np.float32)
    Sel = np.zeros((i_dim, P * width), dtype=np.float32)
    for p in range(P):
        for w, i in enumerate(np.nonzero(col_nz[p])[0]):
            V[:, p * width + w] = U2[p, i, :]
            Sel[i, p * width + w] = 1.0
    nnz_cols = int(col_nz.sum())
    return dict(
        V=mx.array(V), Sel=mx.array(Sel), width=width, P=P,
        pad_frac=1.0 - nnz_cols / (P * width),
    )


def main_sela(s, W_sel, feats):
    G = (W_sel @ s["U_cols"]) * (feats @ s["SelF"])  # (b,c,ncol)
    return G @ s["Agg"]  # (b,c,P)


def main_selb(s, W_sel, feats):
    b, c, _ = W_sel.shape
    T = (W_sel @ s["V"]).reshape(b, c, s["P"], s["width"])
    F = (feats @ s["Sel"]).reshape(b, c, s["P"], s["width"])
    return (T * F).sum(axis=-1)  # (b,c,P)


def sparse_call(contr, struct, main_fn, features, element_onehot):
    """Contraction forward with sparse main step; lower steps unchanged."""
    contr._ensure_weight_caches()
    b, num_c, _ = features.shape

    k_m = contr.weights_max.shape[1]
    W_sel_ck = (element_onehot @ contr._W_max_ck).reshape(b, num_c, k_m)
    out = main_fn(struct, W_sel_ck, features)

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


def make_fn(variant, contr, oh):
    """Fresh function object per (variant, contraction)."""
    if variant == "dense":
        return lambda f: contr(f, oh)
    if variant == "dense_c":
        return mx.compile(lambda f: contr(f, oh))
    if variant == "sela":
        s = build_colsel(contr)
        print(f"    sela struct: ncol={s['ncol']} P={s['P']}", flush=True)
        return lambda f: sparse_call(contr, s, main_sela, f, oh)
    if variant in ("selb", "selb_c"):
        s = build_padded(contr)
        print(f"    selb struct: P={s['P']} width={s['width']} "
              f"pad_frac={s['pad_frac']:.2f}", flush=True)
        fn = lambda f: sparse_call(contr, s, main_selb, f, oh)
        return mx.compile(fn) if variant == "selb_c" else fn
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_DIRS), required=True)
    ap.add_argument("--variant", required=True,
                    choices=["dense", "dense_c", "sela", "selb", "selb_c"])
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
        print(f"  contraction[{ci}] lout={contr.ir_out.l} corr={contr.correlation}",
              flush=True)
        ref = contr(x, oh)
        mx.eval(ref)

        fn = make_fn(args.variant, contr, oh)
        out = fn(x)
        mx.eval(out)
        err = float(mx.max(mx.abs(out - ref)).item())

        def fwd():
            mx.eval(fn(x))

        def loss(xx):
            return fn(xx).sum()

        vag = mx.value_and_grad(loss)

        def fwdbwd():
            l, g = vag(x)
            mx.eval(l, g)

        mx.reset_peak_memory()
        t_f = bench(fwd)
        m_f = mx.get_peak_memory() / 1e6
        mx.reset_peak_memory()
        t_fb = bench(fwdbwd)
        m_fb = mx.get_peak_memory() / 1e6
        print(f"    RESULT {args.variant:8s} lout={contr.ir_out.l} "
              f"fwd={t_f:7.2f}ms ({m_f:7.0f}MB)  "
              f"fwd+bwd={t_fb:7.2f}ms ({m_fb:7.0f}MB)  max_err={err:.2e}",
              flush=True)


if __name__ == "__main__":
    main()
