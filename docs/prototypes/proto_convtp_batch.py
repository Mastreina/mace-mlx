"""Prototype F: batched layer-2 conv_tp (10-instruction uvu TP with mul2=1).

Strategies benchmarked against the current _loop_forward:
  A: literal claim from OPTIMIZATION_REVIEW 4.1 — horizontally concat the six
     cg_mul2_1 matrices into (3, 106), one matmul t = x1_1o @ cg_cat, then
     per-segment x2 contraction; scalar group batched via a small (16, 16)
     scale matrix. Same FLOPs as loop, fewer launches / x1 reads.
  B: contraction reorder ("x2-first") — contract x2 with CG first into a tiny
     per-edge matrix M1 (E, 3, 24) via one (16 x 72) constant matmul, then ONE
     batched matmul x1_1o @ M1. Cuts the dominant FLOP/traffic term ~4.4x.
  C: like B but folds the 0e block in too: x1_cat (E, mul, 4) @ M (E, 4, 40),
     one bmm covers all 10 instructions.

All strategies apply weights via a single take-gather + multiply and assemble
output by slot-order concat (no zeros + accumulate).
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
# Pin baseline code to the clean git HEAD (4d07c83) checkout — the main
# worktree is being refactored concurrently.
sys.path.insert(0, str(HERE / "head-checkout"))

import mlx.core as mx
import numpy as np

import mace_mlx
assert "head-checkout" in mace_mlx.__file__, mace_mlx.__file__
from mace_mlx.model import load_model


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


class BatchedConvTP:
    """Precompute constants for batched variants of a mul2=1 uvu TensorProduct.

    Assumes (asserted): every instruction is uvu, has_weight, mul2=1,
    weight_shape (mul, 1) with a common mul, unique i_out per instruction,
    and is either a cg-scalar instruction (ir1_dim=1) or a cg_mul2_1
    instruction (ir1_dim=3 here).
    """

    def __init__(self, tp):
        insts = tp._instructions
        n = len(insts)
        self.n_inst = n
        sl1 = tp.irreps_in1.slices
        sl2 = tp.irreps_in2.slices
        self.x2_dim = tp.irreps_in2.dim
        self.mul = insts[0].mul1
        for i, inst in enumerate(insts):
            assert inst.connection_mode == "uvu" and inst.has_weight
            assert inst.mul2 == 1 and inst.weight_shape == (self.mul, 1)
            assert tp._cg_scalars[i] is not None or tp._cg_mul2_1[i] is not None
        i_outs = [inst.i_out for inst in insts]
        assert len(set(i_outs)) == n == len(tp.irreps_out)
        # slot order: instructions sorted by i_out
        self.slot_order = sorted(range(n), key=lambda i: insts[i].i_out)

        scal = [i for i in range(n) if tp._cg_scalars[i] is not None]
        m21 = [i for i in range(n) if tp._cg_mul2_1[i] is not None]
        self.scal, self.m21 = scal, m21
        # single source block per group
        (i0,) = {insts[i].i_in1 for i in scal}
        (i1,) = {insts[i].i_in1 for i in m21}
        self.sl_0e, self.sl_1o = sl1[i0], sl1[i1]
        self.d1 = insts[m21[0]].ir1_dim  # 3
        assert all(insts[i].ir1_dim == self.d1 for i in m21)

        do = {i: insts[i].ir_out_dim for i in range(n)}
        self.K0 = sum(do[i] for i in scal)
        self.K1 = sum(do[i] for i in m21)

        # ---- scalar group: S (x2_dim, K0), out0 = x1_0e * w * (x2 @ S) ----
        S = np.zeros((self.x2_dim, self.K0), np.float32)
        off = 0
        self.scal_segs = {}  # inst -> (offset, do)
        ids0 = []
        for i in scal:
            inst = insts[i]
            s2 = sl2[inst.i_in2].start
            c = tp._cg_scalars[i] * inst.path_weight
            for k in range(do[i]):
                S[s2 + k, off + k] = c
            self.scal_segs[i] = (off, do[i])
            ids0 += [i] * do[i]
            off += do[i]
        self.S = mx.stop_gradient(mx.array(S))
        self.ids0 = mx.array(np.array(ids0, np.int32))

        # ---- m21 group (B): G1 (x2_dim, d1*K1), M1 = (x2@G1) -> (E,d1,K1) ----
        G1 = np.zeros((self.x2_dim, self.d1 * self.K1), np.float32)
        off = 0
        self.m21_segs = {}
        ids1 = []
        for i in m21:
            inst = insts[i]
            s2 = sl2[inst.i_in2].start
            cg = np.array(tp._cg_tensors[i])  # (d1, d2, do)
            pw = inst.path_weight
            for m in range(self.d1):
                for j in range(inst.ir2_dim):
                    for k in range(do[i]):
                        G1[s2 + j, m * self.K1 + off + k] = cg[m, j, k] * pw
            self.m21_segs[i] = (off, do[i])
            ids1 += [i] * do[i]
            off += do[i]
        self.G1 = mx.stop_gradient(mx.array(G1))
        self.ids1 = mx.array(np.array(ids1, np.int32))

        # ---- strategy C: G_full (x2_dim, (1+d1)*Ktot), cols in slot order ----
        self.Ktot = self.K0 + self.K1
        dcat = 1 + self.d1
        GF = np.zeros((self.x2_dim, dcat * self.Ktot), np.float32)
        off = 0
        idsF = []
        self.slot_segs = []  # (inst, offset, do) in slot order
        for i in self.slot_order:
            inst = insts[i]
            s2 = sl2[inst.i_in2].start
            pw = inst.path_weight
            if i in self.scal_segs:
                c = tp._cg_scalars[i] * pw
                for k in range(do[i]):
                    GF[s2 + k, 0 * self.Ktot + off + k] = c
            else:
                cg = np.array(tp._cg_tensors[i])
                for m in range(self.d1):
                    for j in range(inst.ir2_dim):
                        for k in range(do[i]):
                            GF[s2 + j, (1 + m) * self.Ktot + off + k] = cg[m, j, k] * pw
            self.slot_segs.append((i, off, do[i]))
            idsF += [i] * do[i]
            off += do[i]
        self.GF = mx.stop_gradient(mx.array(GF))
        self.idsF = mx.array(np.array(idsF, np.int32))

        # 0/1 selector matrices: replace take-gather with GEMM so the VJP is
        # the transposed GEMM (efficient segment-sum) instead of scatter-add.
        T0 = np.zeros((n, self.K0), np.float32)
        for col, i in enumerate(ids0):
            T0[i, col] = 1.0
        T1 = np.zeros((n, self.K1), np.float32)
        for col, i in enumerate(ids1):
            T1[i, col] = 1.0
        self.T0 = mx.stop_gradient(mx.array(T0))
        self.T1 = mx.stop_gradient(mx.array(T1))

        # ---- strategy A: concat cg_mul2_1 (pw baked in), per-seg contraction ----
        cats = []
        self.a_segs = []  # (inst, col_off, d2, do)
        col = 0
        for i in m21:
            inst = insts[i]
            cgm = np.array(tp._cg_mul2_1[i]) * inst.path_weight  # (d1, d2*do)
            cats.append(cgm)
            self.a_segs.append((i, col, inst.ir2_dim, do[i]))
            col += inst.ir2_dim * do[i]
        self.cg_cat = mx.stop_gradient(mx.array(np.concatenate(cats, 1)))
        self.in2_slices = sl2
        self.insts = insts

    # -- shared pieces --
    def _w_t(self, weight):
        E = weight.shape[0]
        return weight.reshape(E, self.n_inst, self.mul).transpose(0, 2, 1)

    def _assemble(self, E, get_seg):
        parts = []
        for i, off, d in self.slot_segs:
            parts.append(get_seg(i, off, d).reshape(E, self.mul * d))
        return mx.concatenate(parts, axis=-1)

    # -- strategy A --
    def forward_A(self, x1, x2, weight):
        E = x1.shape[0]
        w_t = self._w_t(weight)  # (E, mul, n_inst)
        x1_1o = x1[:, self.sl_1o].reshape(E, self.mul, self.d1)
        t = x1_1o @ self.cg_cat  # (E, mul, 106) — single matmul (the claim)
        seg_out = {}
        for i, col, d2, d in self.a_segs:
            ts = t[:, :, col : col + d2 * d].reshape(E, self.mul, d2, d)
            x2s = x2[:, self.in2_slices[self.insts[i].i_in2]]  # (E, d2)
            tp_i = (ts * x2s[:, None, :, None]).sum(axis=-2)  # (E, mul, d)
            seg_out[i] = tp_i * w_t[:, :, i : i + 1]
        # scalar group batched
        x1_0e = x1[:, self.sl_0e].reshape(E, self.mul, 1)
        xs = x2 @ self.S  # (E, K0)
        w0 = mx.take(w_t, self.ids0, axis=-1)  # (E, mul, K0)
        out0 = x1_0e * w0 * xs[:, None, :]
        for i, (off, d) in self.scal_segs.items():
            seg_out[i] = out0[:, :, off : off + d]
        return self._assemble(E, lambda i, off, d: seg_out[i])

    # -- strategy B --
    def forward_B(self, x1, x2, weight):
        E = x1.shape[0]
        w_t = self._w_t(weight)
        # m21 group: contract x2 into CG first, then ONE batched matmul
        M1 = (x2 @ self.G1).reshape(E, self.d1, self.K1)
        x1_1o = x1[:, self.sl_1o].reshape(E, self.mul, self.d1)
        out1 = (x1_1o @ M1) * mx.take(w_t, self.ids1, axis=-1)  # (E, mul, K1)
        # scalar group
        x1_0e = x1[:, self.sl_0e].reshape(E, self.mul, 1)
        xs = x2 @ self.S
        out0 = x1_0e * mx.take(w_t, self.ids0, axis=-1) * xs[:, None, :]

        def get_seg(i, off, d):
            if i in self.scal_segs:
                o, _ = self.scal_segs[i]
                return out0[:, :, o : o + d]
            o, _ = self.m21_segs[i]
            return out1[:, :, o : o + d]

        return self._assemble(E, get_seg)

    # -- strategy B2: weight applied per-segment (slice broadcast, no take) --
    def forward_B2(self, x1, x2, weight):
        E = x1.shape[0]
        wr = weight.reshape(E, self.n_inst, self.mul)
        M1 = (x2 @ self.G1).reshape(E, self.d1, self.K1)
        x1_1o = x1[:, self.sl_1o].reshape(E, self.mul, self.d1)
        out1 = x1_1o @ M1  # (E, mul, K1) — weights NOT applied yet
        x1_0e = x1[:, self.sl_0e].reshape(E, self.mul, 1)
        xs = x2 @ self.S
        out0 = x1_0e * xs[:, None, :]  # (E, mul, K0)

        parts = []
        for i, off, d in self.slot_segs:
            w_i = wr[:, i, :, None]  # (E, mul, 1)
            if i in self.scal_segs:
                o, _ = self.scal_segs[i]
                seg = out0[:, :, o : o + d] * w_i
            else:
                o, _ = self.m21_segs[i]
                seg = out1[:, :, o : o + d] * w_i
            parts.append(seg.reshape(E, self.mul * d))
        return mx.concatenate(parts, axis=-1)

    # -- strategy B3: like B but weight gather via 0/1 selector GEMM --
    def forward_B3(self, x1, x2, weight):
        E = x1.shape[0]
        w_t = self._w_t(weight)  # (E, mul, n_inst)
        M1 = (x2 @ self.G1).reshape(E, self.d1, self.K1)
        x1_1o = x1[:, self.sl_1o].reshape(E, self.mul, self.d1)
        out1 = (x1_1o @ M1) * (w_t @ self.T1)  # (E, mul, K1)
        x1_0e = x1[:, self.sl_0e].reshape(E, self.mul, 1)
        xs = x2 @ self.S
        out0 = x1_0e * (w_t @ self.T0) * xs[:, None, :]

        def get_seg(i, off, d):
            if i in self.scal_segs:
                o, _ = self.scal_segs[i]
                return out0[:, :, o : o + d]
            o, _ = self.m21_segs[i]
            return out1[:, :, o : o + d]

        return self._assemble(E, get_seg)

    # -- strategy C --
    def forward_C(self, x1, x2, weight):
        E = x1.shape[0]
        w_t = self._w_t(weight)
        x1_cat = mx.concatenate(
            [
                x1[:, self.sl_0e].reshape(E, self.mul, 1),
                x1[:, self.sl_1o].reshape(E, self.mul, self.d1),
            ],
            axis=-1,
        )  # (E, mul, 4)
        M = (x2 @ self.GF).reshape(E, 1 + self.d1, self.Ktot)
        out = (x1_cat @ M) * mx.take(w_t, self.idsF, axis=-1)  # (E, mul, 40)
        parts = []
        for _, off, d in self.slot_segs:
            parts.append(out[:, :, off : off + d].reshape(E, self.mul * d))
        return mx.concatenate(parts, axis=-1)


def run_block_bench(model_name, E, seed=0):
    model = load_model(str(HERE / "models" / model_name))
    tp = model.interactions[1].conv_tp
    bat = BatchedConvTP(tp)
    in1_dim = tp.irreps_in1.dim
    wn = tp.weight_numel
    rng = np.random.default_rng(seed)
    x1 = mx.array(rng.normal(size=(E, in1_dim)).astype(np.float32))
    x2 = mx.array(rng.normal(size=(E, bat.x2_dim)).astype(np.float32))
    w = mx.array(rng.normal(size=(E, wn)).astype(np.float32))
    mx.eval(x1, x2, w)

    ref = tp(x1, x2, w)
    mx.eval(ref)

    variants = {
        "loop (current)": lambda a, b, c: tp(a, b, c),
        "loop+compile": mx.compile(lambda a, b, c: tp._loop_forward(a, b, c)),
        "A concat-cg": bat.forward_A,
        "B x2-first": bat.forward_B,
        "B2 seg-weight": bat.forward_B2,
        "B3 selector-mm": bat.forward_B3,
        "C single-bmm": bat.forward_C,
        "B2+compile": mx.compile(bat.forward_B2),
        "B3+compile": mx.compile(bat.forward_B3),
    }
    print(f"\n=== {model_name} inter[1].conv_tp  E={E}  (mul={bat.mul}) ===", flush=True)
    print(f"{'variant':16s} {'maxerr':>9s} {'fwd ms':>8s} {'fwd MB':>8s} "
          f"{'f+b ms':>8s} {'f+b MB':>8s} {'f+b3 ms':>8s} {'f+b3 MB':>8s}")
    results = {}
    for name, fn in variants.items():
        out = fn(x1, x2, w)
        mx.eval(out)
        err = float(mx.max(mx.abs(out - ref)).item())

        def fwd():
            mx.eval(fn(x1, x2, w))

        vag = mx.value_and_grad(lambda a: fn(a, x2, w).sum())

        def fwdbwd():
            l, g = vag(x1)
            mx.eval(l, g)

        # grad wrt all three inputs — closer to the real model backward
        # (in the model, x2=SH(positions) and w=RadialMLP(lengths) both need grads)
        vag3 = mx.value_and_grad(
            lambda a, b, c: fn(a, b, c).sum(), argnums=(0, 1, 2)
        )

        def fwdbwd3():
            l, g = vag3(x1, x2, w)
            mx.eval(l, g)

        reset_peak(); t_f = bench(fwd); m_f = get_peak() / 1e6
        reset_peak(); t_fb = bench(fwdbwd); m_fb = get_peak() / 1e6
        reset_peak(); t_fb3 = bench(fwdbwd3); m_fb3 = get_peak() / 1e6
        results[name] = (err, t_f, m_f, t_fb, m_fb, t_fb3, m_fb3)
        print(f"{name:16s} {err:9.2e} {t_f:8.2f} {m_f:8.0f} {t_fb:8.2f} {m_fb:8.0f} "
              f"{t_fb3:8.2f} {m_fb3:8.0f}", flush=True)
    base = results["loop (current)"]
    for name, r in results.items():
        if name != "loop (current)":
            print(f"  {name:16s} fwd x{base[1]/r[1]:5.2f}  f+b x{base[3]/r[3]:5.2f}  "
                  f"f+b3 x{base[5]/r[5]:5.2f}  mem(f+b3) x{base[6]/r[6]:4.2f}")
    return results


if __name__ == "__main__":
    import sys
    configs = [("mpa0-medium", 9936), ("mpa0-medium", 46000), ("mp0-large", 46000)]
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        configs = [("mpa0-medium", 46000)]
    for model_name, E in configs:
        run_block_bench(model_name, E)
