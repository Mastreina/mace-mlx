"""Micro-benchmark: where do the 83ms of sparse-SC fwd+bwd go?

Individually times each kernel of the vjpb pipeline (lout=1 shapes,
b=1000, c=128, P=768, w=3, i=16, k=51) plus the lower-order steps,
so we know what to attack to reach the 58ms acceptance line.
"""
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mace_mlx.model import load_model

CACHE = Path.home() / ".cache" / "mace_mlx"


def bench(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


model = load_model(str(CACHE / "medium-mpa-0" / "v2"))
sc = model.products[0].symmetric_contractions
contr = sc.contractions[1]  # lout=1
b, c = 1000, contr.num_features
i_dim = contr._u_main_i_dim
k_dim = contr._u_main_k_dim

U = np.array(contr._u_matrices[contr.correlation])
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

V_mx = mx.array(V)
Sel_mx = mx.array(Sel)
SelT_mx = mx.array(Sel.T.copy())

rng = np.random.default_rng(0)
W_sel = mx.array(rng.normal(size=(b, c, k_dim)).astype(np.float32))
feats = mx.array(rng.normal(size=(b, c, i_dim)).astype(np.float32))
dout = mx.array(rng.normal(size=(b, c, P)).astype(np.float32))
mx.eval(W_sel, feats, dout)

print(f"shapes: b={b} c={c} P={P} w={width} i={i_dim} k={k_dim}", flush=True)

# forward pieces
T4 = (W_sel @ V_mx).reshape(b, c, P, width)
F4 = (feats @ Sel_mx).reshape(b, c, P, width)
mx.eval(T4, F4)
G2 = (dout[..., None] * T4).reshape(b, c, P * width)
mx.eval(G2)

pieces = {
    "fwd T GEMM (b,c,k)@(k,Pw)": lambda: mx.eval(W_sel @ V_mx),
    "fwd F GEMM (b,c,i)@(i,Pw)": lambda: mx.eval(feats @ Sel_mx),
    "fwd (T*F).sum(-1)": lambda: mx.eval((T4 * F4).sum(axis=-1)),
    "bwd d4*T elementwise": lambda: mx.eval(dout[..., None] * T4),
    "bwd df GEMM (b,c,Pw)@(Pw,i)": lambda: mx.eval(G2 @ SelT_mx),
}
total = 0.0
for name, fn in pieces.items():
    t = bench(fn)
    total += t
    print(f"  {name:34s} {t:7.2f} ms", flush=True)
print(f"  {'sum of main-step kernels':34s} {total:7.2f} ms", flush=True)

# lower-step cost: full dense f+b minus dense main-step-only f+b
oh_np = np.zeros((b, sc.num_elements), dtype=np.float32)
oh_np[np.arange(b), rng.integers(0, sc.num_elements, size=b)] = 1.0
oh = mx.array(oh_np)
feats_flat = mx.array(rng.normal(size=(b, sc.irreps_in.dim)).astype(np.float32))
blocks = []
for idx, (mul, ir_dim) in enumerate(sc._ir_dims):
    blocks.append(feats_flat[..., sc._slices[idx]].reshape(b, mul, ir_dim))
x = mx.concatenate(blocks, axis=-1)
mx.eval(x, oh)
contr._ensure_weight_caches()


def main_only(xx):
    W = (oh @ contr._W_max_ck).reshape(b, c, k_dim)
    T = (W @ V_mx).reshape(b, c, P, width)
    F = (xx @ Sel_mx).reshape(b, c, P, width)
    return (T * F).sum(axis=-1).sum()


def lower_only(xx):
    # exact lower chain, fed by a constant in place of the main output
    out = mx.stop_gradient(mx.zeros((b, c, P), dtype=xx.dtype))
    feat_col = xx[:, :, :, None]
    for idx in range(contr._unrolled_n_lower):
        k_i = contr.weights[idx].shape[1]
        W_sel_i = (oh @ contr._W_lower_ck[idx]).reshape(b, c, k_i)
        c_tensor = W_sel_i @ contr._u_lower_2d_t[idx]
        c_tensor = c_tensor + out
        i_d = contr._unrolled_lower_i_dims[idx]
        prefix_outer = contr._unrolled_lower_prefix_sizes[idx] // i_d
        c_4d = c_tensor.reshape(b, c, prefix_outer, i_d)
        out = (c_4d @ feat_col).reshape(b, c, prefix_outer)
    return out.sum()


for name, f in [("main-only (selb autograd)", main_only),
                ("lower-only", lower_only)]:
    vag = mx.value_and_grad(f)

    def fb():
        l, g = vag(x)
        mx.eval(l, g)

    t_fb = bench(fb)
    print(f"  {name:34s} f+b={t_fb:7.2f} ms", flush=True)
