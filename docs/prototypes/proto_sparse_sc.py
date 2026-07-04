"""Prototype D: sparse symmetric contraction (exploit 99.5% U sparsity).

Only the MAIN (nu=correlation) contraction materializes the huge
(b, c, prefix*i) intermediate; lower-nu steps are cheap. We replace:
    WU = (W_sel @ U_wf).reshape(b,c,prefix,i);  out = WU @ f[...,None]
with a sparse evaluation over U's nonzeros (p, i, k, val):
    G[b,c,n] = W_sel[b,c,k_n] * f[b,c,i_n] * val_n
    out[b,c,p] = segment_sum_n->p (G)
Variants for the segment sum:
    S-matmul:   G @ S  with S (nnz, prefix) 0/1 pattern
    at-scatter: zeros.at[:, :, p_idx].add(G)
    csr-loop:   pad nnz per p to fixed width, loop width, pure gathers
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


class SparseMain:
    """Precomputed sparse structure of the main U tensor."""

    def __init__(self, contr):
        U = np.array(contr._u_matrices[contr.correlation])
        # shape (*prefix, i, k)
        prefix_shape = U.shape[:-2]
        i_dim, k_dim = U.shape[-2], U.shape[-1]
        P = int(np.prod(prefix_shape))
        U2 = U.reshape(P, i_dim, k_dim)
        p_idx, i_idx, k_idx = np.nonzero(np.abs(U2) > 1e-12)
        vals = U2[p_idx, i_idx, k_idx].astype(np.float32)
        self.nnz = len(vals)
        self.P, self.i_dim, self.k_dim = P, i_dim, k_dim
        self.prefix_shape = prefix_shape
        self.p_idx = mx.array(p_idx.astype(np.int32))
        self.i_idx = mx.array(i_idx.astype(np.int32))
        self.k_idx = mx.array(k_idx.astype(np.int32))
        self.vals = mx.array(vals)
        # S pattern (nnz, P)
        S = np.zeros((self.nnz, P), dtype=np.float32)
        S[np.arange(self.nnz), p_idx] = 1.0
        self.S = mx.array(S)
        # padded CSR: width = max nnz per p
        order = np.argsort(p_idx, kind="stable")
        counts = np.bincount(p_idx, minlength=P)
        self.width = int(counts.max())
        k_tab = np.zeros((P, self.width), dtype=np.int32)
        i_tab = np.zeros((P, self.width), dtype=np.int32)
        v_tab = np.zeros((P, self.width), dtype=np.float32)
        fill = np.zeros(P, dtype=np.int64)
        for n in order:
            p = p_idx[n]
            w = fill[p]
            k_tab[p, w] = k_idx[n]
            i_tab[p, w] = i_idx[n]
            v_tab[p, w] = vals[n]
            fill[p] += 1
        self.k_tab = mx.array(k_tab)
        self.i_tab = mx.array(i_tab)
        self.v_tab = mx.array(v_tab)


def main_smatmul(sp, W_sel, feats):
    G = mx.take(W_sel, sp.k_idx, axis=2) * mx.take(feats, sp.i_idx, axis=2) * sp.vals
    return G @ sp.S  # (b, c, P)


def main_scatter(sp, W_sel, feats):
    G = mx.take(W_sel, sp.k_idx, axis=2) * mx.take(feats, sp.i_idx, axis=2) * sp.vals
    b, c = G.shape[0], G.shape[1]
    out = mx.zeros((b, c, sp.P), dtype=G.dtype)
    return out.at[:, :, sp.p_idx].add(G)


def main_csr(sp, W_sel, feats):
    out = None
    for w in range(sp.width):
        Gw = (mx.take(W_sel, sp.k_tab[:, w], axis=2)
              * mx.take(feats, sp.i_tab[:, w], axis=2)
              * sp.v_tab[:, w])
        out = Gw if out is None else out + Gw
    return out  # (b, c, P)


def sparse_call(contr, sp, features, element_onehot, main_fn):
    """Contraction forward with sparse main step; lower steps unchanged."""
    contr._ensure_weight_caches()
    b, num_c, coupling_i = features.shape
    n_lower = contr._unrolled_n_lower

    k_m = contr.weights_max.shape[1]
    W_sel_ck = (element_onehot @ contr._W_max_ck).reshape(b, num_c, k_m)

    out = main_fn(sp, W_sel_ck, features)  # (b, c, prefix_size) -- sparse!

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
    mx.eval(feats_flat, oh)

    # reshape to (b, c, coupling) like SymmetricContraction.__call__
    blocks = []
    for idx, (mul, ir_dim) in enumerate(sc._ir_dims):
        blocks.append(feats_flat[..., sc._slices[idx]].reshape(b, mul, ir_dim))
    x = mx.concatenate(blocks, axis=-1)
    mx.eval(x)

    for ci, contr in enumerate(sc.contractions):
        sp = SparseMain(contr)
        print(f"  contraction[{ci}] lout={contr.ir_out.l}: prefix={sp.P} i={sp.i_dim} "
              f"k={sp.k_dim} nnz={sp.nnz} csr_width={sp.width}", flush=True)

        ref = contr(x, oh)
        mx.eval(ref)

        variants = {
            "dense_current": lambda f, c=contr: c(f, oh),
            "sparse_smatmul": lambda f, c=contr, s=sp: sparse_call(c, s, f, oh, main_smatmul),
            "sparse_scatter": lambda f, c=contr, s=sp: sparse_call(c, s, f, oh, main_scatter),
            "sparse_csr": lambda f, c=contr, s=sp: sparse_call(c, s, f, oh, main_csr),
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
                print(f"    {name:15s} fwd={t_f:7.2f}ms ({m_f:6.0f}MB)  "
                      f"fwd+bwd={t_fb:7.2f}ms ({m_fb:6.0f}MB)  max_err={err:.2e}", flush=True)
            except Exception as e:
                print(f"    {name:15s} FAILED: {type(e).__name__}: {e}", flush=True)
