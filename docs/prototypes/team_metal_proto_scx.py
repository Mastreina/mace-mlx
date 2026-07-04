"""Prototype 1: fused Metal kernel for the SC X-construction.

Current production (sparse SC main step, symmetric_contraction.py):
    X = (features @ SelI) * (W_sel_ck @ SelK)   # two selector GEMMs, 3 materializations
    out = X @ U_rows

Fused: one kernel computes X[b,c,r] = f[b,c,i_r] * W[b,c,k_r] directly
(dual gather + multiply). Backward via mx.custom_function:
    df[b,c,i] = sum_{r: i_r=i} dX[b,c,r] * W[b,c,k_r]   (CSR over i, no atomics)
    dW[b,c,k] = sum_{r: k_r=k} dX[b,c,r] * f[b,c,i_r]   (CSR over k, separate kernel
                                                          so it is pruned in force-only backward)

Validation (this script, GPU, small-b to stay light):
    vs the production Contraction._call_unrolled for real medium-mpa-0
    contraction configs (lout=0 nrow=99, lout=1 nrow=233), fp32:
    fwd, d/dfeatures (force path), d/dweights (training path),
    plain and under mx.compile(mx.value_and_grad(...)); fp16 smoke.
"""
from __future__ import annotations

import argparse
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/mastreina/Desktop/mace-mlx")

from mace_mlx.irreps import Irrep, Irreps
from mace_mlx.symmetric_contraction import Contraction

# ---------------------------------------------------------------------------
# Kernels (compiled lazily by MLX on first call; template ints specialize)
# ---------------------------------------------------------------------------

_SRC_FWD = """
    uint elem = thread_position_in_grid.x;      // bc * NROW + r
    uint r = elem % NROW;
    uint bc = elem / NROW;
    X[elem] = (T)((float)f[bc * IDIM + idx_i[r]] * (float)w[bc * KDIM + idx_k[r]]);
"""

_SRC_DF = """
    uint elem = thread_position_in_grid.x;      // bc * IDIM + i
    uint i = elem % IDIM;
    uint bc = elem / IDIM;
    float acc = 0.0f;
    for (uint s = starts_i[i]; s < starts_i[i + 1]; ++s) {
        uint r = rows_by_i[s];
        acc += (float)dX[bc * NROW + r] * (float)w[bc * KDIM + idx_k[r]];
    }
    df[elem] = (T)acc;
"""

_SRC_DW = """
    uint elem = thread_position_in_grid.x;      // bc * KDIM + k
    uint k = elem % KDIM;
    uint bc = elem / KDIM;
    float acc = 0.0f;
    for (uint s = starts_k[k]; s < starts_k[k + 1]; ++s) {
        uint r = rows_by_k[s];
        acc += (float)dX[bc * NROW + r] * (float)f[bc * IDIM + idx_i[r]];
    }
    dw[elem] = (T)acc;
"""

_k_fwd = mx.fast.metal_kernel(
    name="scx_fused_fwd",
    input_names=["f", "w", "idx_i", "idx_k"],
    output_names=["X"],
    source=_SRC_FWD,
)
_k_df = mx.fast.metal_kernel(
    name="scx_fused_df",
    input_names=["dX", "w", "idx_k", "rows_by_i", "starts_i"],
    output_names=["df"],
    source=_SRC_DF,
)
_k_dw = mx.fast.metal_kernel(
    name="scx_fused_dw",
    input_names=["dX", "f", "idx_i", "rows_by_k", "starts_k"],
    output_names=["dw"],
    source=_SRC_DW,
)

_TG = (256, 1, 1)


def make_fused_x(contraction: Contraction):
    """Build a custom_function computing X for one Contraction's constants.

    Returns fused_x(f, W) with f (b, c, i_dim), W (b, c, k_dim) -> X (b, c, nrow).
    Constants (indices, CSR) are derived from the production SelI/SelK and
    eval'ed immediately (mx.compile lazy-closure-capture defense).
    """
    assert contraction._use_sparse_main
    sel_i = np.array(contraction._sp_sel_i)  # (i_dim, nrow)
    sel_k = np.array(contraction._sp_sel_k)  # (k_dim, nrow)
    i_dim, nrow = sel_i.shape
    k_dim = sel_k.shape[0]
    i_of_r = sel_i.argmax(axis=0).astype(np.uint32)
    k_of_r = sel_k.argmax(axis=0).astype(np.uint32)

    def csr(idx, dim):
        order = np.argsort(idx, kind="stable").astype(np.uint32)
        starts = np.searchsorted(idx[order], np.arange(dim + 1)).astype(np.uint32)
        return mx.array(order), mx.array(starts)

    idx_i = mx.array(i_of_r)
    idx_k = mx.array(k_of_r)
    rows_by_i, starts_i = csr(i_of_r, i_dim)
    rows_by_k, starts_k = csr(k_of_r, k_dim)
    mx.eval(idx_i, idx_k, rows_by_i, starts_i, rows_by_k, starts_k)

    tmpl = [("NROW", nrow), ("IDIM", i_dim), ("KDIM", k_dim)]

    @mx.custom_function
    def fused_x(f, w):
        b, c = f.shape[0], f.shape[1]
        return _k_fwd(
            inputs=[f, w, idx_i, idx_k],
            template=[("T", f.dtype)] + tmpl,
            grid=(b * c * nrow, 1, 1), threadgroup=_TG,
            output_shapes=[(b, c, nrow)], output_dtypes=[f.dtype],
        )[0]

    @fused_x.vjp
    def _vjp(primals, cotan, output):
        f, w = primals
        b, c = f.shape[0], f.shape[1]
        df = _k_df(
            inputs=[cotan, w, idx_k, rows_by_i, starts_i],
            template=[("T", f.dtype)] + tmpl,
            grid=(b * c * i_dim, 1, 1), threadgroup=_TG,
            output_shapes=[(b, c, i_dim)], output_dtypes=[f.dtype],
        )[0]
        dw = _k_dw(
            inputs=[cotan, f, idx_i, rows_by_k, starts_k],
            template=[("T", f.dtype)] + tmpl,
            grid=(b * c * k_dim, 1, 1), threadgroup=_TG,
            output_shapes=[(b, c, k_dim)], output_dtypes=[f.dtype],
        )[0]
        return df, dw

    return fused_x


def call_unrolled_fused(con: Contraction, fused_x, features, element_onehot):
    """Mirror of Contraction._call_unrolled with the X line swapped."""
    con._ensure_weight_caches()
    b, num_c, _ = features.shape
    k_m = con.weights_max.shape[1]
    W_sel_ck = (element_onehot @ con._W_max_ck).reshape(b, num_c, k_m)
    feat_col = features[:, :, :, None]

    X = fused_x(features, W_sel_ck)          # <<< fused kernel
    out = X @ con._sp_u_rows

    for idx in range(con._unrolled_n_lower):
        k_i = con.weights[idx].shape[1]
        W_sel_i = (element_onehot @ con._W_lower_ck[idx]).reshape(b, num_c, k_i)
        c_tensor = W_sel_i @ con._u_lower_2d_t[idx] + out
        i_d = con._unrolled_lower_i_dims[idx]
        prefix_outer = con._unrolled_lower_prefix_sizes[idx] // i_d
        c_4d = c_tensor.reshape(b, num_c, prefix_outer, i_d)
        out = (c_4d @ feat_col).reshape(b, num_c, prefix_outer)

    return out.reshape(b, -1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

MEDIUM_IRREPS_IN = "128x0e+128x1o+128x2e+128x3o"


def build_contraction(lout: int, num_elements: int = 89, seed: int = 0):
    con = Contraction(
        irreps_in=Irreps(MEDIUM_IRREPS_IN),
        ir_out=Irrep(f"{lout}{'e' if lout % 2 == 0 else 'o'}"),
        correlation=3,
        num_elements=num_elements,
    )
    # Deterministic weights of realistic magnitude
    rng = np.random.default_rng(seed)
    con.weights_max = mx.array(
        rng.normal(size=con.weights_max.shape).astype(np.float32)
    ) / con.weights_max.shape[1]
    con.weights = [
        mx.array(rng.normal(size=w.shape).astype(np.float32)) / w.shape[1]
        for w in con.weights
    ]
    mx.eval(con.weights_max, con.weights)
    return con


def validate(lout: int, b: int, num_elements: int = 89) -> list[tuple[str, float, float]]:
    rng = np.random.default_rng(42 + lout)
    con = build_contraction(lout, num_elements)
    fused = make_fused_x(con)

    f_np = rng.normal(size=(b, 128, 16)).astype(np.float32)
    onehot_np = np.zeros((b, num_elements), dtype=np.float32)
    onehot_np[np.arange(b), rng.integers(0, num_elements, size=b)] = 1.0
    features = mx.array(f_np)
    onehot = mx.array(onehot_np)
    cot_np = rng.normal(size=(b, 128 * (2 * lout + 1))).astype(np.float32)
    cot = mx.array(cot_np)
    mx.eval(features, onehot, cot)

    rows: list[tuple[str, float, float]] = []

    def rel(err, refv):
        return err / max(float(mx.max(mx.abs(refv)).item()), 1e-30)

    # forward
    out_ref = con(features, onehot)
    out_fused = call_unrolled_fused(con, fused, features, onehot)
    mx.eval(out_ref, out_fused)
    e = float(mx.max(mx.abs(out_ref - out_fused)).item())
    rows.append(("fwd", e, rel(e, out_ref)))

    # d/dfeatures (force path)
    def loss_ref(x):
        return (con(x, onehot) * cot).sum()

    def loss_fused(x):
        return (call_unrolled_fused(con, fused, x, onehot) * cot).sum()

    gr = mx.grad(loss_ref)(features)
    gf = mx.grad(loss_fused)(features)
    mx.eval(gr, gf)
    e = float(mx.max(mx.abs(gr - gf)).item())
    rows.append(("d/dfeatures", e, rel(e, gr)))

    # d/dweights_max (training path; exercises the dW kernel)
    def lossw_ref(wm):
        con.weights_max = wm
        con._cached_wmax_id = None  # invalidate cache
        return (con(features, onehot) * cot).sum()

    def lossw_fused(wm):
        con.weights_max = wm
        con._cached_wmax_id = None
        return (call_unrolled_fused(con, fused, features, onehot) * cot).sum()

    wm0 = con.weights_max
    gwr = mx.grad(lossw_ref)(wm0)
    gwf = mx.grad(lossw_fused)(wm0)
    mx.eval(gwr, gwf)
    con.weights_max = wm0
    con._cached_wmax_id = None
    e = float(mx.max(mx.abs(gwr - gwf)).item())
    rows.append(("d/dweights", e, rel(e, gwr)))

    # compile(value_and_grad(fused)) vs plain ref grad
    vag = mx.compile(mx.value_and_grad(loss_fused))
    v_c, g_c = vag(features)
    v_p = loss_ref(features)
    mx.eval(v_c, g_c, v_p)
    e = float(mx.max(mx.abs(g_c - gr)).item())
    ev = abs(float(v_c.item()) - float(v_p.item())) / max(abs(float(v_p.item())), 1e-30)
    rows.append(("compile(vag) grad", e, rel(e, gr)))
    rows.append(("compile(vag) value rel", ev, ev))

    # fp16 smoke: fused fp16 vs ref fp16 (same-precision comparison)
    f16 = features.astype(mx.float16)
    con16 = build_contraction(lout, num_elements)
    con16.weights_max = con.weights_max
    con16.weights = list(con.weights)
    con16.set_dtype(mx.float16)
    fused16 = make_fused_x(con16)
    out16_ref = con16(f16, onehot.astype(mx.float16))
    out16_fused = call_unrolled_fused(con16, fused16, f16, onehot.astype(mx.float16))
    mx.eval(out16_ref, out16_fused)
    e = float(mx.max(mx.abs(out16_ref - out16_fused)).item())
    rows.append(("fp16 fwd (vs fp16 ref)", e, rel(e, out16_ref)))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=256, help="batch (atoms); keep small")
    args = ap.parse_args()

    all_ok = True
    for lout in (0, 1):
        print(f"--- lout={lout} (b={args.b}) ---")
        for name, err, relerr in validate(lout, args.b):
            # abs<1e-5 (force-level) or rel<1e-6 (fp32 sum-reordering floor,
            # for large-magnitude weight grads); fp16 judged on rel only.
            if "fp16" in name:
                ok = relerr < 2e-2
            else:
                ok = err < 1e-5 or relerr < 1e-6
            all_ok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:24s} "
                  f"abs={err:.3e} rel={relerr:.3e}")
    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
