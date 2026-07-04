"""Repro: mx.compile reads garbage from lazy closure-captured arrays
(MLX 0.31.2, Apple M4 Pro, GPU).

Trigger: the compiled function references, via closure, DERIVED arrays
(non-leaf, with a pending compute graph -- here the transpose/reshape
chain built by _ensure_weight_caches) that were created BEFORE the
compile trace and never mx.eval'd. The compiled graph then produces
nan/inf/unstable values ("cache+lower" and "lower only" configs below);
the same code with the caches eval'd first, or with leaf (numpy-born)
constants only, is correct. NOT related to mx.custom_function -- this
repro uses plain autograd ops only (proto_sparse_sc4's vjpb_c variant
originally surfaced it with a custom_function in the graph, which was a
red herring).

Expected output:
  cache+lower (sc4 shape)    fwd max_err=nan ...
  lower only                 fwd max_err=nan finite=False
  cache only                 fwd max_err=0.000e+00 finite=True
  main only                  fwd max_err=0.000e+00 finite=True
"""
from pathlib import Path

import mlx.core as mx
import numpy as np

from mace_mlx.model import load_model

CACHE = Path.home() / ".cache" / "mace_mlx"

model = load_model(str(CACHE / "medium-mpa-0" / "v2"))
sc = model.products[0].symmetric_contractions
contr = sc.contractions[0]  # lout=0 (nan case)
b = 1000
rng = np.random.default_rng(0)
feats_flat = mx.array(rng.normal(size=(b, sc.irreps_in.dim)).astype(np.float32))
z = rng.integers(0, sc.num_elements, size=b)
oh_np = np.zeros((b, sc.num_elements), dtype=np.float32)
oh_np[np.arange(b), z] = 1.0
oh = mx.array(oh_np)
blocks = []
for idx, (mul, ir_dim) in enumerate(sc._ir_dims):
    blocks.append(feats_flat[..., sc._slices[idx]].reshape(b, mul, ir_dim))
x = mx.concatenate(blocks, axis=-1)
mx.eval(x, oh)

# padded structure (same as sc4 build_padded, GEMM expand)
U = np.array(contr._u_matrices[contr.correlation])
i_dim, k_dim = U.shape[-2], U.shape[-1]
P = int(np.prod(U.shape[:-2]))
U2 = U.reshape(P, i_dim, k_dim)
col_nz = np.abs(U2).max(axis=2) > 1e-12
width = int(col_nz.sum(axis=1).max())
V = np.zeros((k_dim, P * width), dtype=np.float32)
Sel = np.zeros((i_dim, P * width), dtype=np.float32)
for p in range(P):
    for w, i in enumerate(np.nonzero(col_nz[p])[0]):
        V[:, p * width + w] = U2[p, i, :]
        Sel[i, p * width + w] = 1.0
V_mx = mx.stop_gradient(mx.array(V))
Sel_mx = mx.stop_gradient(mx.array(Sel))
SelT_mx = mx.stop_gradient(mx.array(Sel.T.copy()))


def make_g():
    def g(T4, feats):  # plain function, no custom_function at all
        bb, cc = feats.shape[0], feats.shape[1]
        F = (feats @ Sel_mx).reshape(bb, cc, P, width)
        return (T4 * F).sum(axis=-1)

    return g


contr._ensure_weight_caches()  # caches built but NOT evaluated -> lazy
k_m = contr.weights_max.shape[1]
c = contr.num_features


def full_call(feats, g, with_cache_call, with_lower):
    if with_cache_call:
        contr._ensure_weight_caches()  # side-effectful call inside trace
    W_sel = (oh @ contr._W_max_ck).reshape(b, c, k_m)
    T4 = (W_sel @ V_mx).reshape(b, c, P, width)
    out = g(T4, feats)
    if with_lower:
        feat_col = feats[:, :, :, None]
        for idx in range(contr._unrolled_n_lower):
            k_i = contr.weights[idx].shape[1]
            W_i = (oh @ contr._W_lower_ck[idx]).reshape(b, c, k_i)
            ct = W_i @ contr._u_lower_2d_t[idx] + out
            i_d = contr._unrolled_lower_i_dims[idx]
            po = contr._unrolled_lower_prefix_sizes[idx] // i_d
            out = (ct.reshape(b, c, po, i_d) @ feat_col).reshape(b, c, po)
    return out.reshape(b, -1)


configs = [
    ("cache+lower (sc4 shape)", True, True),
    ("lower only", False, True),
    ("cache only", True, False),
    ("main only", False, False),
]
for name, wc, wl in configs:
    g = make_g()
    dense_ref = contr(x, oh)   # dense path first, big WU buffer -> cache pool
    comp = mx.compile(lambda f, g=g, wc=wc, wl=wl: full_call(f, g, wc, wl))
    out = comp(x)
    mx.eval(dense_ref, out)    # same eval batch, like sc4
    plain = lambda f, g=g, wc=wc, wl=wl: full_call(f, g, wc, wl)
    ref = plain(x)
    mx.eval(ref)
    err = float(mx.max(mx.abs(out - ref)).item())
    print(f"  {name:26s} fwd max_err={err:.3e} "
          f"finite={bool(mx.all(mx.isfinite(out)))}", flush=True)
