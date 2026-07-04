"""Prototype C: SymmetricContraction optimizations.

Loads real SC modules from converted mpa0-medium and mp0-small models.
  C1 baseline: current unrolled path
  C2 mx.checkpoint: recompute in backward (memory vs time tradeoff)
  C3 gather weight-select: W_ck[z_idx] instead of onehot @ W_flat
  C4 U-matrix sparsity statistics (motivates sparse contraction sketch)
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


def gather_call(contr, features, z_idx):
    """Variant of Contraction._call_unrolled with gather weight selection."""
    contr._ensure_weight_caches()
    b, num_c, coupling_i = features.shape
    n_lower = contr._unrolled_n_lower

    k_m = contr.weights_max.shape[1]
    W_sel_ck = mx.take(contr._W_max_ck, z_idx, axis=0).reshape(b, num_c, k_m)

    prefix_size = contr._u_main_prefix_size
    i_dim = contr._u_main_i_dim
    WU = (W_sel_ck @ contr._u_main_wf).reshape(b, num_c, prefix_size, i_dim)
    feat_col = features[:, :, :, None]
    out = (WU @ feat_col).reshape(b, num_c, prefix_size)

    for idx in range(n_lower):
        k_i = contr.weights[idx].shape[1]
        W_sel_i = mx.take(contr._W_lower_ck[idx], z_idx, axis=0).reshape(b, num_c, k_i)
        c_tensor = W_sel_i @ contr._u_lower_2d_t[idx]
        c_tensor = c_tensor + out
        i_d = contr._unrolled_lower_i_dims[idx]
        prefix_outer = contr._unrolled_lower_prefix_sizes[idx] // i_d
        c_4d = c_tensor.reshape(b, num_c, prefix_outer, i_d)
        out = (c_4d @ feat_col).reshape(b, num_c, prefix_outer)

    return out.reshape(b, -1)


def sc_gather(sc, features, z_idx):
    batch_size = features.shape[0]
    blocks = []
    for idx, (mul, ir_dim) in enumerate(sc._ir_dims):
        block = features[..., sc._slices[idx]].reshape(batch_size, mul, ir_dim)
        blocks.append(block)
    x = mx.concatenate(blocks, axis=-1)
    outs = [gather_call(c, x, z_idx) for c in sc.contractions]
    return mx.concatenate(outs, axis=-1)


for model_name in ["mpa0-medium", "mp0-small"]:
    print(f"\n=== {model_name} (product[0].symmetric_contractions) ===", flush=True)
    model = load_model(str(HERE / "models" / model_name))
    sc = model.products[0].symmetric_contractions
    in_dim = sc.irreps_in.dim
    num_el = sc.num_elements

    # C4: U sparsity stats
    for ci, c in enumerate(sc.contractions):
        for nu, U in c._u_matrices.items():
            U_np = np.array(U)
            nnz = np.count_nonzero(np.abs(U_np) > 1e-12)
            print(f"  U[lout={c.ir_out.l}, nu={nu}] shape={U_np.shape} "
                  f"nnz={nnz}/{U_np.size} ({100*nnz/U_np.size:.1f}%)", flush=True)

    for b in [1000]:
        rng = np.random.default_rng(0)
        feats = mx.array(rng.normal(size=(b, in_dim)).astype(np.float32))
        z_np = rng.integers(0, num_el, size=b).astype(np.int32)
        z_idx = mx.array(z_np)
        onehot = np.zeros((b, num_el), dtype=np.float32)
        onehot[np.arange(b), z_np] = 1.0
        oh = mx.array(onehot)
        mx.eval(feats, z_idx, oh)

        ref = sc(feats, oh)
        mx.eval(ref)

        variants = {
            "C1_current": lambda f: sc(f, oh),
            "C2_checkpoint": lambda f: mx.checkpoint(lambda ff: sc(ff, oh))(f),
            "C3_gather": lambda f: sc_gather(sc, f, z_idx),
        }
        print(f"  --- b={b} ---")
        for name, fn in variants.items():
            try:
                out = fn(feats)
                mx.eval(out)
                err = float(mx.max(mx.abs(out - ref)).item())

                def fwd():
                    mx.eval(fn(feats))

                def loss(x):
                    return fn(x).sum()
                vag = mx.value_and_grad(loss)

                def fwdbwd():
                    l, g = vag(feats)
                    mx.eval(l, g)

                reset_peak(); t_f = bench(fwd); m_f = get_peak() / 1e6
                reset_peak(); t_fb = bench(fwdbwd); m_fb = get_peak() / 1e6
                print(f"  {name:14s} fwd={t_f:7.2f}ms ({m_f:6.0f}MB)  "
                      f"fwd+bwd={t_fb:7.2f}ms ({m_fb:6.0f}MB)  max_err={err:.2e}", flush=True)
            except Exception as e:
                print(f"  {name:14s} FAILED: {type(e).__name__}: {e}", flush=True)
