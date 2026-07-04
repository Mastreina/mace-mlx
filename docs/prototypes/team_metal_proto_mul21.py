"""Prototype 2: fused Metal kernel for the second-layer conv_tp (batched_mul21).

This is the current #1 whole-step hotspot (medium/large models, layer-1
interaction). Production forward (_batched_mul21_forward) materializes, per
edge (E=46k, medium: mul=128, K0=16, K1=24, out=5120):

    w_t   = weight.reshape(E,10,128).T      (E,128,10)   23.6 MB
    M1    = (x2 @ G1)                       (E,3,24)     13.2 MB
    g1    = x1_m @ M1        batched GEMM   (E,128,24)  565 MB
    wexp1 = w_t @ T1         selector GEMM  (E,128,24)  565 MB
    out1  = g1 * wexp1                      (E,128,24)  565 MB
    xs    = x2 @ S                          (E,16)        2.9 MB
    wexp0 = w_t @ T0                        (E,128,16)  377 MB
    out0  = x1_s * wexp0 * xs               (E,128,16)  377 MB (+ temp)
    mji   = concat(10 slot slices)          (E,5120)    942 MB write + 942 read

Fused: ONE kernel (threadgroup per edge, thread per channel u) reads
x1 (E,512), M1 (E,72), xs (E,16), w (E,1280 raw layout -- no transpose) and
writes mji (E,5120) directly in final slot layout. The d1=3 dot product and
the selector-expansion multiply happen in registers; M1/xs live in
threadgroup memory. x2 enters through two tiny GEMMs (M1 = x2@G1, xs = x2@S)
kept as MLX ops inside the custom_function.

Backward: one 4-output kernel computes dx1, dM1, dxs, dw in a single pass
over the cotangent (in-kernel recompute, simdgroup+threadgroup reductions for
dM1/dxs); dx2 = dM1 @ G1^T + dxs @ S^T as MLX GEMMs. All three inputs are
live in the force path (x1 <- node feats, x2 = SH, w <- radial MLP), so a
single multi-output kernel loses nothing to dead-branch pruning.

Validation vs production TensorProduct (real medium-mpa-0 layer-1 conv_tp
loaded from cache): fwd, d/dx1, d/dx2, d/dweight, compile(value_and_grad),
fp16 smoke. Small E to stay GPU-light.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/mastreina/Desktop/mace-mlx")

_TGSIZE_NOTE = "threadgroup size = mul (128) -> 4 simdgroups"


def _slot_metadata(tp):
    """Recover per-output-slot metadata from a batched_mul21 TensorProduct."""
    insts = tp._instructions
    n = len(insts)
    scal = [i for i in range(n) if tp._cg_scalars[i] is not None]
    # segment offsets, rebuilt exactly as _setup_batched_mul21 does
    seg = {}
    off = 0
    for i in scal:
        seg[i] = off
        off += insts[i].ir_out_dim
    off = 0
    for i in range(n):
        if i not in seg:
            seg[i] = off
            off += insts[i].ir_out_dim
    slot_order = sorted(range(n), key=lambda i: insts[i].i_out)
    mul = tp._bm21_mul
    slots = []
    out_off = 0
    for i in slot_order:
        d = insts[i].ir_out_dim
        slots.append(dict(inst=i, scal=(i in scal), seg=seg[i], d=d, off=out_off))
        out_off += mul * d
    return slots, out_off


def _gen_header(slots, name_suffix):
    n = len(slots)
    arr = lambda key: ", ".join(str(int(s[key])) for s in slots)
    return f"""
constant uint NSLOT_{name_suffix} = {n};
constant uint SLOT_INST_{name_suffix}[{n}] = {{{arr('inst')}}};
constant uint SLOT_SCAL_{name_suffix}[{n}] = {{{", ".join("1" if s["scal"] else "0" for s in slots)}}};
constant uint SLOT_SEG_{name_suffix}[{n}]  = {{{arr('seg')}}};
constant uint SLOT_D_{name_suffix}[{n}]    = {{{arr('d')}}};
constant uint SLOT_OFF_{name_suffix}[{n}]  = {{{arr('off')}}};
"""


_SRC_FWD_TMPL = """
    threadgroup float M1s[D1 * K1];
    threadgroup float xss[K0];
    uint e = threadgroup_position_in_grid.y;
    uint u = thread_position_in_threadgroup.x;

    for (uint t = u; t < D1 * K1; t += MUL) M1s[t] = (float)M1[e * (D1 * K1) + t];
    for (uint t = u; t < K0; t += MUL)      xss[t] = (float)xs[e * K0 + t];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    float x1v[D1];
    for (uint m = 0; m < D1; ++m)
        x1v[m] = (float)x1[e * X1DIM + SL_M21 + u * D1 + m];
    float x1s = (float)x1[e * X1DIM + SL_SCAL + u];

    for (uint s = 0; s < NSLOT_{sfx}; ++s) {{
        float w_su = (float)w[e * WDIM + SLOT_INST_{sfx}[s] * MUL + u];
        uint d = SLOT_D_{sfx}[s];
        uint seg = SLOT_SEG_{sfx}[s];
        uint base = e * OUTDIM + SLOT_OFF_{sfx}[s] + u * d;
        if (SLOT_SCAL_{sfx}[s] != 0) {{
            float a = x1s * w_su;
            for (uint k = 0; k < d; ++k)
                mji[base + k] = (T)(a * xss[seg + k]);
        }} else {{
            for (uint k = 0; k < d; ++k) {{
                float acc = 0.0f;
                for (uint m = 0; m < D1; ++m)
                    acc += x1v[m] * M1s[m * K1 + seg + k];
                mji[base + k] = (T)(w_su * acc);
            }}
        }}
    }}
"""

# Backward: one pass over the cotangent per edge.
#   dw[e,i,u]   = sum_k cot * (fwd factor without w)
#   dx1v[m]     = sum over m21 slots of w_su * sum_k cot * M1s[m*K1+seg+k]
#   dx1s        = sum over scal slots of w_su * sum_k cot * xss[seg+k]
#   dM1[m,c]    = sum_u cot * w_su * x1v[m]      (threadgroup reduction)
#   dxs[c]      = sum_u cot * w_su * x1s         (threadgroup reduction)
# Reductions: simd_sum within each of the (MUL/32) simdgroups, partials in
# threadgroup memory, first D1*d (or d) threads combine and write.
_SRC_BWD_TMPL = """
    threadgroup float M1s[D1 * K1];
    threadgroup float xss[K0];
    threadgroup float red[(MUL / 32) * D1 * DMAX];
    uint e = threadgroup_position_in_grid.y;
    uint u = thread_position_in_threadgroup.x;
    uint simd_id = u / 32;

    for (uint t = u; t < D1 * K1; t += MUL) M1s[t] = (float)M1[e * (D1 * K1) + t];
    for (uint t = u; t < K0; t += MUL)      xss[t] = (float)xs[e * K0 + t];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    float x1v[D1];
    for (uint m = 0; m < D1; ++m)
        x1v[m] = (float)x1[e * X1DIM + SL_M21 + u * D1 + m];
    float x1s = (float)x1[e * X1DIM + SL_SCAL + u];

    float dx1v[D1] = {{0.0f}};
    float dx1s = 0.0f;

    for (uint s = 0; s < NSLOT_{sfx}; ++s) {{
        uint i = SLOT_INST_{sfx}[s];
        float w_su = (float)w[e * WDIM + i * MUL + u];
        uint d = SLOT_D_{sfx}[s];
        uint seg = SLOT_SEG_{sfx}[s];
        uint base = e * OUTDIM + SLOT_OFF_{sfx}[s] + u * d;

        float g[DMAX];
        for (uint k = 0; k < d; ++k) g[k] = (float)cot[base + k];

        float dw_acc = 0.0f;
        if (SLOT_SCAL_{sfx}[s] != 0) {{
            // forward: out = x1s * w * xss[seg+k]
            float gx = 0.0f;                    // sum_k g*xss
            for (uint k = 0; k < d; ++k) {{
                dw_acc += g[k] * x1s * xss[seg + k];
                gx += g[k] * xss[seg + k];
            }}
            dx1s += w_su * gx;
            // dxs[seg+k] = sum_u g[k]*w*x1s
            for (uint k = 0; k < d; ++k) {{
                float v = metal::simd_sum(g[k] * w_su * x1s);
                if ((u & 31) == 0) red[simd_id * D1 * DMAX + k] = v;
            }}
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (u < d) {{
                float acc = 0.0f;
                for (uint q = 0; q < MUL / 32; ++q)
                    acc += red[q * D1 * DMAX + u];
                dxs_out[e * K0 + seg + u] = (T)acc;
            }}
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        }} else {{
            // forward: out = w * sum_m x1v[m]*M1s[m*K1+seg+k]
            for (uint k = 0; k < d; ++k) {{
                float dot = 0.0f;
                for (uint m = 0; m < D1; ++m)
                    dot += x1v[m] * M1s[m * K1 + seg + k];
                dw_acc += g[k] * dot;
                for (uint m = 0; m < D1; ++m)
                    dx1v[m] += g[k] * w_su * M1s[m * K1 + seg + k];
            }}
            // dM1[m, seg+k] = sum_u g[k]*w*x1v[m]
            for (uint m = 0; m < D1; ++m) {{
                for (uint k = 0; k < d; ++k) {{
                    float v = metal::simd_sum(g[k] * w_su * x1v[m]);
                    if ((u & 31) == 0) red[simd_id * D1 * DMAX + m * d + k] = v;
                }}
            }}
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (u < D1 * d) {{
                uint m = u / d, k = u % d;
                float acc = 0.0f;
                for (uint q = 0; q < MUL / 32; ++q)
                    acc += red[q * D1 * DMAX + m * d + k];
                dM1_out[e * (D1 * K1) + m * K1 + seg + k] = (T)acc;
            }}
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        }}
        dw_out[e * WDIM + i * MUL + u] = (T)dw_acc;
    }}

    for (uint m = 0; m < D1; ++m)
        dx1_out[e * X1DIM + SL_M21 + u * D1 + m] = (T)dx1v[m];
    dx1_out[e * X1DIM + SL_SCAL + u] = (T)dx1s;
"""


def make_fused_mul21(tp):
    """Build fused_mul21(x1_gathered, x2, weight) -> mji for one TensorProduct.

    tp must have _batched_mul21 set up. The scalar group must be non-empty
    and x1 slots must cover the full x1 row (true for the real medium/large
    layer-1 conv_tp; asserted below).
    """
    assert tp._batched_mul21 and tp._bm21_S is not None
    slots, outdim = _slot_metadata(tp)
    mul, d1 = tp._bm21_mul, tp._bm21_d1
    K0, K1 = tp._bm21_K0, tp._bm21_K1
    n = tp._bm21_n
    sl_scal, sl_m21 = tp._bm21_sl_scal, tp._bm21_sl_m21
    x1dim = tp.irreps_in1.dim
    wdim = n * mul
    dmax = max(s["d"] for s in slots)
    assert mul % 32 == 0

    G1 = tp._bm21_G1  # (x2_dim, d1*K1)
    S = tp._bm21_S    # (x2_dim, K0)
    mx.eval(G1, S)

    cfg = f"{mul}_{d1}_{K0}_{K1}_{n}_{outdim}_{x1dim}_{sl_scal.start}_{sl_m21.start}"
    sfx = hashlib.md5(
        (cfg + str([tuple(sorted(s.items())) for s in slots])).encode()
    ).hexdigest()[:8]
    header = _gen_header(slots, sfx)

    k_fwd = mx.fast.metal_kernel(
        name=f"mul21_fwd_{sfx}",
        input_names=["x1", "M1", "xs", "w"],
        output_names=["mji"],
        source=_SRC_FWD_TMPL.format(sfx=sfx),
        header=header,
    )
    k_bwd = mx.fast.metal_kernel(
        name=f"mul21_bwd_{sfx}",
        input_names=["cot", "x1", "M1", "xs", "w"],
        output_names=["dx1_out", "dM1_out", "dxs_out", "dw_out"],
        source=_SRC_BWD_TMPL.format(sfx=sfx),
        header=header,
    )

    tmpl = [
        ("MUL", mul), ("D1", d1), ("K0", K0), ("K1", K1),
        ("X1DIM", x1dim), ("WDIM", wdim), ("OUTDIM", outdim),
        ("SL_SCAL", sl_scal.start), ("SL_M21", sl_m21.start),
        ("DMAX", dmax),
    ]

    def run_fwd(x1, M1, xs, w):
        E = x1.shape[0]
        return k_fwd(
            inputs=[x1, M1, xs, w],
            template=[("T", x1.dtype)] + tmpl,
            grid=(mul, E, 1), threadgroup=(mul, 1, 1),
            output_shapes=[(E, outdim)], output_dtypes=[x1.dtype],
        )[0]

    @mx.custom_function
    def fused_mul21(x1, x2, w):
        M1 = x2 @ G1
        xs = x2 @ S
        return run_fwd(x1, M1, xs, w)

    @fused_mul21.vjp
    def _vjp(primals, cotan, output):
        x1, x2, w = primals
        E = x1.shape[0]
        M1 = x2 @ G1
        xs = x2 @ S
        dx1, dM1, dxs, dw = k_bwd(
            inputs=[cotan, x1, M1, xs, w],
            template=[("T", x1.dtype)] + tmpl,
            grid=(mul, E, 1), threadgroup=(mul, 1, 1),
            output_shapes=[(E, x1dim), (E, d1 * K1), (E, K0), (E, wdim)],
            output_dtypes=[x1.dtype] * 4,
        )
        dx2 = dM1 @ G1.T + dxs @ S.T
        return dx1, dx2, dw

    return fused_mul21


# ---------------------------------------------------------------------------
# Validation vs production _batched_mul21_forward (real medium layer-1 conv_tp)
# ---------------------------------------------------------------------------


def validate(E: int) -> bool:
    from mace_mlx.model import load_model

    model = load_model("/Users/mastreina/.cache/mace_mlx/medium-mpa-0/v2")
    tp = model.interactions[1].conv_tp
    assert tp._batched_mul21
    fused = make_fused_mul21(tp)

    rng = np.random.default_rng(7)
    x1 = mx.array(rng.normal(size=(E, tp.irreps_in1.dim)).astype(np.float32))
    x2 = mx.array(rng.normal(size=(E, tp.irreps_in2.dim)).astype(np.float32))
    w = mx.array(
        rng.normal(size=(E, tp.weight_numel)).astype(np.float32) * 0.3
    )
    outdim = _slot_metadata(tp)[1]
    cot = mx.array(rng.normal(size=(E, outdim)).astype(np.float32))
    mx.eval(x1, x2, w, cot)

    ok_all = True

    def report(name, a, b, tol_abs=1e-5, tol_rel=1e-6):
        nonlocal ok_all
        err = float(mx.max(mx.abs(a - b)).item())
        ref = max(float(mx.max(mx.abs(b)).item()), 1e-30)
        ok = err < tol_abs or err / ref < tol_rel
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s} abs={err:.3e} rel={err/ref:.3e}")

    out_ref = tp(x1, x2, w)
    out_fused = fused(x1, x2, w)
    mx.eval(out_ref, out_fused)
    report("fwd", out_fused, out_ref)

    def loss_ref(a, b, c):
        return (tp(a, b, c) * cot).sum()

    def loss_fused(a, b, c):
        return (fused(a, b, c) * cot).sum()

    gr = mx.grad(loss_ref, argnums=(0, 1, 2))(x1, x2, w)
    gf = mx.grad(loss_fused, argnums=(0, 1, 2))(x1, x2, w)
    mx.eval(gr, gf)
    report("d/dx1", gf[0], gr[0])
    report("d/dx2 (SH)", gf[1], gr[1])
    report("d/dweight", gf[2], gr[2])

    vag = mx.compile(mx.value_and_grad(loss_fused, argnums=(0, 1, 2)))
    v_c, g_c = vag(x1, x2, w)
    mx.eval(v_c, g_c)
    report("compile(vag) d/dx1", g_c[0], gr[0])
    report("compile(vag) d/dx2", g_c[1], gr[1])
    report("compile(vag) d/dw", g_c[2], gr[2])

    # fp16 smoke vs fp16 reference (same-precision)
    x1h, x2h, wh = (t.astype(mx.float16) for t in (x1, x2, w))
    out_h_ref = tp(x1h, x2h, wh)
    out_h = fused(x1h, x2h, wh)
    mx.eval(out_h_ref, out_h)
    err = float(mx.max(mx.abs(out_h - out_h_ref)).item())
    ref = max(float(mx.max(mx.abs(out_h_ref)).item()), 1e-30)
    ok = err / ref < 2e-2
    ok_all &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {'fp16 fwd (vs fp16 ref)':22s} "
          f"abs={err:.3e} rel={err/ref:.3e}")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=int, default=2000, help="edge count; keep small")
    args = ap.parse_args()
    print(f"--- medium-mpa-0 layer-1 conv_tp, E={args.edges} ({_TGSIZE_NOTE}) ---")
    ok = validate(args.edges)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
