"""Prototype D2: column-compressed sparse symmetric contraction.

Key insight: U_wf (k, prefix*i) has mostly-zero COLUMNS. Instead of the
full GEMM into (b,c,prefix*i), do a GEMM only into the nonzero columns
(b,c,ncol), multiply by gathered features, then accumulate columns into
their prefix slots with a small fixed-width loop (pure adds, no scatter).
"""
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mace_mlx.model import load_model

HERE = Path(__file__).parent


def bench(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def reset_peak():
    try:
        mx.reset_peak_memory()
    except AttributeError:
        mx.metal.reset_peak_memory()


def get_peak():
    try:
        return mx.get_peak_memory()
    except AttributeError:
        return mx.metal.get_peak_memory()


class ColCompressed:
    def __init__(self, contr):
        U = np.array(contr._u_matrices[contr.correlation])
        prefix_shape = U.shape[:-2]
        i_dim, k_dim = U.shape[-2], U.shape[-1]
        P = int(np.prod(prefix_shape))
        # U_wf layout: (k, P*i) with column index col = p*i_dim + i
        U_wf = np.moveaxis(U.reshape(P, i_dim, k_dim), -1, 0).reshape(k_dim, P * i_dim)
        col_nnz = np.abs(U_wf).max(axis=0) > 1e-12
        cols = np.nonzero(col_nnz)[0]
        self.ncol = len(cols)
        self.P, self.i_dim, self.k_dim = P, i_dim, k_dim
        self.U_cols = mx.array(U_wf[:, cols].astype(np.float32))  # (k, ncol)
        p_of_col = (cols // i_dim).astype(np.int32)
        i_of_col = (cols % i_dim).astype(np.int32)
        self.i_col = mx.array(i_of_col)
        # padded per-p tables of column positions
        counts = np.bincount(p_of_col, minlength=P)
        self.width = int(counts.max())
        col_tab = np.full((P, self.width), -1, dtype=np.int64)
        fill = np.zeros(P, dtype=np.int64)
        for n, p in enumerate(p_of_col):
            col_tab[p, fill[p]] = n
            fill[p] += 1
        # map -1 padding to a dummy column; give it zero weight via mask
        self.mask = mx.array((col_tab >= 0).astype(np.float32))  # (P, width)
        col_tab[col_tab < 0] = 0
        self.col_tab = mx.array(col_tab.astype(np.int32))


def main_colcomp(cc, W_sel, feats):
    # GEMM only over nonzero columns: (b,c,k) @ (k,ncol) -> (b,c,ncol)
    WU = W_sel @ cc.U_cols
    G = WU * mx.take(feats, cc.i_col, axis=2)  # (b,c,ncol)
    # accumulate columns into prefix slots: fixed-width loop of takes
    out = None
    for w in range(cc.width):
        idx = cc.col_tab[:, w]        # (P,)
        m = cc.mask[:, w]             # (P,)
        term = mx.take(G, idx, axis=2) * m
        out = term if out is None else out + term
    return out  # (b, c, P)


def sparse_call(contr, cc, features, element_onehot):
    contr._ensure_weight_caches()
    b, num_c, _ = features.shape
    n_lower = contr._unrolled_n_lower

    k_m = contr.weights_max.shape[1]
    W_sel_ck = (element_onehot @ contr._W_max_ck).reshape(b, num_c, k_m)
    out = main_colcomp(cc, W_sel_ck, features)

    feat_col = features[:, :, :, None]
    for idx in range(n_lower):
        k_i = contr.weights[idx].shape[1]
        W_sel_i = (element_onehot @ contr._W_lower_ck[idx]).reshape(b, num_c, k_i)
        c_tensor = W_sel_i @ contr._u_lower_2d_t[idx]
        c_tensor = c_tensor + out
        i_d = contr._unrolled_lower_i_dims[idx]
        prefix_outer = contr._unrolled_lower_prefix_sizes[idx] // i_d
        c_4d = c_tensor.reshape(b, num_c, prefix_outer, i_d)
        out = (c_4d @ feat_col).reshape(b, num_c, prefix_outer)

    return out.reshape(b, -1)


for model_name in ["mp0-small", "mpa0-medium"]:
    print(f"\n=== {model_name} product[0] ===", flush=True)
    model = load_model(str(HERE / "models" / model_name))
    sc = model.products[0].symmetric_contractions
    in_dim = sc.irreps_in.dim
    num_el = sc.num_elements

    b = 1000
    rng = np.random.default_rng(0)
    feats_flat = mx.array(rng.normal(size=(b, in_dim)).astype(np.float32))
    z_np = rng.integers(0, num_el, size=b).astype(np.int32)
    onehot = np.zeros((b, num_el), dtype=np.float32)
    onehot[np.arange(b), z_np] = 1.0
    oh = mx.array(onehot)

    blocks = []
    for idx, (mul, ir_dim) in enumerate(sc._ir_dims):
        blocks.append(feats_flat[..., sc._slices[idx]].reshape(b, mul, ir_dim))
    x = mx.concatenate(blocks, axis=-1)
    mx.eval(x, oh)

    for ci, contr in enumerate(sc.contractions):
        cc = ColCompressed(contr)
        print(f"  contraction[{ci}] lout={contr.ir_out.l}: P={cc.P} i={cc.i_dim} "
              f"k={cc.k_dim} ncol={cc.ncol}/{cc.P*cc.i_dim} width={cc.width}", flush=True)

        ref = contr(x, oh)
        mx.eval(ref)

        # also test compiled variants
        plain = lambda f, c=contr, s=cc: sparse_call(c, s, f, oh)
        comp = mx.compile(plain)
        dense_c = mx.compile(lambda f, c=contr: c(f, oh))
        variants = {
            "dense_current": lambda f, c=contr: c(f, oh),
            "dense+compile": dense_c,
            "colcomp": plain,
            "colcomp+compile": comp,
        }
        for name, fn in variants.items():
            try:
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

                reset_peak(); t_f = bench(fwd); m_f = get_peak() / 1e6
                reset_peak(); t_fb = bench(fwdbwd); m_fb = get_peak() / 1e6
                print(f"    {name:16s} fwd={t_f:7.2f}ms ({m_f:6.0f}MB)  "
                      f"fwd+bwd={t_fb:7.2f}ms ({m_fb:6.0f}MB)  max_err={err:.2e}", flush=True)
            except Exception as e:
                print(f"    {name:16s} FAILED: {type(e).__name__}: {e}", flush=True)
