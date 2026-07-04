"""Prototype: restructuring the SC lower-order (nu < correlation) iterations.

Variants (self-contained, library files untouched):
  baseline  library Contraction.__call__ as-is (sparse main + dense lower)
  addmm     lower add fused into the GEMM epilogue via mx.addmm
            (c_tensor = out + W_sel @ U_t in one kernel, no separate
            (b,c,prefix) elementwise add)
  distrib   distributive-law rewrite: out_next = RC(W,f) + out_4d @ f_col
            where RC is the row-compressed bilinear form over U_lower's
            nonzero (k,i) pairs (24/3 pairs for lout=1, 16/1 for lout=0).
            The (b,c,prefix) intermediate c_tensor is never materialized;
            the add moves from prefix (768) down to prefix_outer (48).

Usage:
  --check          numerical check vs baseline (small b, runs GPU briefly)
  --bench          timing (warmup 3 + 10 median, peak memory)  [main session only]
  --variant {baseline,addmm,distrib}
  --contraction {0,1,2}  0 = products[0] lout=0, 1 = products[0] lout=1,
                         2 = products[1] lout=0
  --scope {contraction,sc}  single contraction vs whole SymmetricContraction
  --mode {full,lower_only}  lower_only replicates the legacy 16.4 ms probe
                            (main output replaced by stop_gradient zeros)
  --b N            batch size (default 1000 for bench, 64 for check)
"""
import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mace_mlx.model import load_model

CACHE = Path.home() / ".cache" / "mace_mlx"


# ---------------------------------------------------------------- constants
def build_lower_rc_consts(contr):
    """Row-compressed constants for each lower iteration.

    U_lower viewed as U'(k, prefix_outer, i): extract nonzero (k, i) pairs
    (max over prefix_outer) and build SelK (k, nnz), SelI (i, nnz),
    U_rows (nnz, prefix_outer). Same construction as the sparse main path.
    """
    consts = []
    for idx in range(contr._unrolled_n_lower):
        U2t = np.array(contr._u_lower_2d_t[idx])  # (k, prefix)
        k_l = U2t.shape[0]
        i_d = contr._unrolled_lower_i_dims[idx]
        po = contr._unrolled_lower_prefix_sizes[idx] // i_d
        U3 = U2t.reshape(k_l, po, i_d)
        k_r, i_r = np.nonzero(np.abs(U3).max(axis=1) > 1e-12)
        nnz = len(k_r)
        sel_k = np.zeros((k_l, nnz), dtype=np.float32)
        sel_k[k_r, np.arange(nnz)] = 1.0
        sel_i = np.zeros((i_d, nnz), dtype=np.float32)
        sel_i[i_r, np.arange(nnz)] = 1.0
        u_rows = U3[k_r, :, i_r].astype(np.float32)  # (nnz, po)
        entry = (
            mx.array(sel_i),
            mx.array(sel_k),
            mx.array(u_rows),
            po,
            i_d,
            nnz,
        )
        mx.eval(*entry[:3])  # avoid lazy-closure-capture hazards
        consts.append(entry)
    return consts


# ---------------------------------------------------------------- variants
def make_fwd(contr, variant, rc_consts=None, lower_only=False):
    """Return fwd(x, oh) -> (b, c*ir_out_dim) for one Contraction."""
    n_lower = contr._unrolled_n_lower
    k_m = contr.weights_max.shape[1]
    lower_i = contr._unrolled_lower_i_dims
    lower_p = contr._unrolled_lower_prefix_sizes

    def main_out(x, oh, b, c):
        if lower_only:
            return mx.stop_gradient(
                mx.zeros((b, c, contr._u_main_prefix_size), dtype=x.dtype)
            )
        W_sel_ck = (oh @ contr._W_max_ck).reshape(b, c, k_m)
        X = (x @ contr._sp_sel_i) * (W_sel_ck @ contr._sp_sel_k)
        return X @ contr._sp_u_rows  # (b, c, prefix)

    if variant == "baseline":
        if lower_only:
            def fwd(x, oh):
                b, c, _ = x.shape
                out = main_out(x, oh, b, c)
                feat_col = x[..., None]
                for idx in range(n_lower):
                    k_i = contr.weights[idx].shape[1]
                    W_sel_i = (oh @ contr._W_lower_ck[idx]).reshape(b, c, k_i)
                    c_t = W_sel_i @ contr._u_lower_2d_t[idx] + out
                    i_d = lower_i[idx]
                    po = lower_p[idx] // i_d
                    out = (c_t.reshape(b, c, po, i_d) @ feat_col).reshape(b, c, po)
                return out.reshape(b, -1)
            return fwd
        contr._ensure_weight_caches()
        return lambda x, oh: contr(x, oh)  # library path

    if variant == "addmm":
        def fwd(x, oh):
            b, c, _ = x.shape
            out = main_out(x, oh, b, c)
            feat_col = x[..., None]
            for idx in range(n_lower):
                k_i = contr.weights[idx].shape[1]
                W_sel_i = (oh @ contr._W_lower_ck[idx]).reshape(b, c, k_i)
                # fused: out + W_sel_i @ U_t in one GEMM epilogue
                c_t = mx.addmm(out, W_sel_i, contr._u_lower_2d_t[idx])
                i_d = lower_i[idx]
                po = lower_p[idx] // i_d
                out = (c_t.reshape(b, c, po, i_d) @ feat_col).reshape(b, c, po)
            return out.reshape(b, -1)
        return fwd

    if variant == "distrib":
        assert rc_consts is not None
        def fwd(x, oh):
            b, c, _ = x.shape
            out = main_out(x, oh, b, c)
            feat_col = x[..., None]
            for idx in range(n_lower):
                sel_i, sel_k, u_rows, po, i_d, _ = rc_consts[idx]
                k_i = contr.weights[idx].shape[1]
                W_sel_i = (oh @ contr._W_lower_ck[idx]).reshape(b, c, k_i)
                # RC term: row-compressed bilinear over U_lower nnz (k,i)
                Xl = (x @ sel_i) * (W_sel_i @ sel_k)  # (b,c,nnz)
                rc = Xl @ u_rows  # (b,c,po)
                # coupling term: contract previous out with features directly
                coup = (out.reshape(b, c, po, i_d) @ feat_col).reshape(b, c, po)
                out = rc + coup  # add at po, not prefix
            return out.reshape(b, -1)
        return fwd

    raise ValueError(variant)


# ---------------------------------------------------------------- helpers
def get_contraction(model, which):
    if which == 0:
        return model.products[0].symmetric_contractions, 0
    if which == 1:
        return model.products[0].symmetric_contractions, 1
    if which == 2:
        return model.products[1].symmetric_contractions, 0
    raise ValueError(which)


def make_inputs(sc, b, seed=0):
    rng = np.random.default_rng(seed)
    c = sc._num_features
    x = mx.array(rng.normal(size=(b, c, sc._coupling_dim)).astype(np.float32))
    oh_np = np.zeros((b, sc.num_elements), dtype=np.float32)
    oh_np[np.arange(b), rng.integers(0, sc.num_elements, size=b)] = 1.0
    oh = mx.array(oh_np)
    x_flat = mx.array(rng.normal(size=(b, sc.irreps_in.dim)).astype(np.float32))
    mx.eval(x, oh, x_flat)
    return x, oh, x_flat


def bench(fn, warmup=3, n=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


# ---------------------------------------------------------------- check
def run_check(model, b):
    print(f"numerical check vs baseline, b={b}")
    ok = True
    for which in (0, 1, 2):
        sc, ci = get_contraction(model, which)
        contr = sc.contractions[ci]
        contr._ensure_weight_caches()
        x, oh, _ = make_inputs(sc, b, seed=which)
        rng = np.random.default_rng(100 + which)

        f_base = make_fwd(contr, "baseline")
        ref = f_base(x, oh)
        mx.eval(ref)
        cot = mx.array(rng.normal(size=ref.shape).astype(np.float32))
        mx.eval(cot)

        def loss_of(fwd):
            return lambda xx: (fwd(xx, oh) * cot).sum()

        ref_l, ref_g = mx.value_and_grad(loss_of(f_base))(x)
        mx.eval(ref_l, ref_g)
        ref_np = np.array(ref)
        refg_np = np.array(ref_g)
        scale_f = np.abs(ref_np).max()
        scale_g = np.abs(refg_np).max()

        rc_consts = build_lower_rc_consts(contr)
        for variant in ("addmm", "distrib"):
            fwd = make_fwd(contr, variant, rc_consts=rc_consts)
            out = fwd(x, oh)
            l, g = mx.value_and_grad(loss_of(fwd))(x)
            mx.eval(out, l, g)
            df = np.abs(np.array(out) - ref_np).max()
            dg = np.abs(np.array(g) - refg_np).max()
            rel_f = df / max(scale_f, 1e-30)
            rel_g = dg / max(scale_g, 1e-30)
            status = "OK " if (rel_f < 1e-5 and rel_g < 1e-5) else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"  contraction {which} ({'lout=' + str(contr.ir_out.l)}) "
                  f"{variant:8s} fwd max abs {df:.2e} (rel {rel_f:.2e})  "
                  f"dfeat max abs {dg:.2e} (rel {rel_g:.2e})  {status}")
    print("check:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- bench
def run_bench(model, args):
    sc, ci = get_contraction(model, args.contraction)
    contr = sc.contractions[ci]
    contr._ensure_weight_caches()
    x, oh, x_flat = make_inputs(sc, args.b)
    lower_only = args.mode == "lower_only"
    rc_consts = (
        build_lower_rc_consts(contr) if args.variant == "distrib" else None
    )

    if args.scope == "sc":
        if args.variant != "baseline" or lower_only:
            # whole-SC scope with variant lower paths
            fwds = []
            for cc in sc.contractions:
                cc._ensure_weight_caches()
                rcc = (
                    build_lower_rc_consts(cc)
                    if args.variant == "distrib" else None
                )
                fwds.append(
                    make_fwd(cc, args.variant, rc_consts=rcc,
                             lower_only=lower_only)
                )

            def sc_fwd(feats_flat, ohh):
                bsz = feats_flat.shape[0]
                blocks = []
                for idx, (mul, ir_dim) in enumerate(sc._ir_dims):
                    blocks.append(
                        feats_flat[..., sc._slices[idx]].reshape(bsz, mul, ir_dim)
                    )
                xx = mx.concatenate(blocks, axis=-1)
                return mx.concatenate([f(xx, ohh) for f in fwds], axis=-1)

            fwd = lambda xx: sc_fwd(xx, oh)
        else:
            fwd = lambda xx: sc(xx, oh)
        inp = x_flat
        label = f"SC[{'p0' if args.contraction < 2 else 'p1'}]"
    else:
        f1 = make_fwd(contr, args.variant, rc_consts=rc_consts,
                      lower_only=lower_only)
        fwd = lambda xx: f1(xx, oh)
        inp = x
        label = f"contraction {args.contraction} (lout={contr.ir_out.l})"

    def fwd_only():
        mx.eval(fwd(inp))

    vag = mx.value_and_grad(lambda xx: fwd(xx).sum())

    def fb():
        l, g = vag(inp)
        mx.eval(l, g)

    print(f"bench {label} variant={args.variant} mode={args.mode} "
          f"b={args.b} scope={args.scope}")
    mx.reset_peak_memory()
    t_f = bench(fwd_only)
    peak_f = mx.get_peak_memory() / 1e6
    mx.reset_peak_memory()
    t_fb = bench(fb)
    peak_fb = mx.get_peak_memory() / 1e6
    print(f"  fwd  {t_f:8.2f} ms   peak {peak_f:8.0f} MB")
    print(f"  f+b  {t_fb:8.2f} ms   peak {peak_fb:8.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline",
                    choices=["baseline", "addmm", "distrib"])
    ap.add_argument("--contraction", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--scope", default="contraction",
                    choices=["contraction", "sc"])
    ap.add_argument("--mode", default="full", choices=["full", "lower_only"])
    ap.add_argument("--b", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    model = load_model(str(CACHE / "medium-mpa-0" / "v2"))
    if args.check:
        b = args.b or 64
        if not run_check(model, b):
            raise SystemExit(1)
    if args.bench:
        args.b = args.b or 1000
        run_bench(model, args)


if __name__ == "__main__":
    main()
