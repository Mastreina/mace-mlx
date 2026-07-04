"""Tensor product of irreducible representations.

Implements the core equivariant operation combining two irreps features
using precomputed Clebsch-Gordan coefficients.

Performance notes:
    When ir1_dim == 1 (scalar input), the CG tensor is (1, d, d) and the
    three-input einsum collapses. Two fast paths are used:

    a) CG[0] = c * I_d (unrotated): scalar multiply, no einsum needed.
       out = c * x1 * x2

    b) CG[0] = M (rotated, general matrix): one matmul replaces the einsum.
       x2_rot = x2 @ M.T, then same scalar-like multiply with x2_rot.

    These fast-paths are critical for MACE conv_tp where hidden_irreps are
    scalars (e.g. "128x0e") and the CG may be pre-rotated to absorb the
    SH basis rotation.
"""

from __future__ import annotations

import hashlib

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mace_mlx.clebsch_gordan import wigner_3j
from mace_mlx.irreps import Irrep, Irreps

# --- Fused Metal kernels for the batched_mul21 conv_tp path ---------------
# One forward kernel (threadgroup per edge, thread per channel u) produces
# the output directly in final slot layout: the w_t transpose, M1 broadcast,
# selector expansion, multiplies and 10-slot assembly all happen in
# registers/threadgroup memory, eliminating ~3.4 GB of materialized
# intermediates per 46k edges (medium). One 4-output backward kernel
# computes dx1/dM1/dxs/dw in a single pass over the cotangent (in-kernel
# recompute, simdgroup reductions); dx2 folds back through two small GEMMs.
# Measured (M4 Pro, medium layer-1, E=46000): block fwd 9.3x, fwd+bwd 7.2x,
# whole-step medium/Si1000 2.5x. Numerics vs the GEMM path: <=2.5e-7 rel.

_METAL_MUL21_FWD_SRC = """
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

_METAL_MUL21_BWD_SRC = """
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
            float gx = 0.0f;
            for (uint k = 0; k < d; ++k) {{
                dw_acc += g[k] * x1s * xss[seg + k];
                gx += g[k] * xss[seg + k];
            }}
            dx1s += w_su * gx;
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
            for (uint k = 0; k < d; ++k) {{
                float dot = 0.0f;
                for (uint m = 0; m < D1; ++m)
                    dot += x1v[m] * M1s[m * K1 + seg + k];
                dw_acc += g[k] * dot;
                for (uint m = 0; m < D1; ++m)
                    dx1v[m] += g[k] * w_su * M1s[m * K1 + seg + k];
            }}
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


class Instruction:
    """Single tensor product instruction describing one coupling path."""

    __slots__ = (
        "i_in1",
        "i_in2",
        "i_out",
        "connection_mode",
        "has_weight",
        "path_weight",
        "weight_shape",
        "mul1",
        "mul2",
        "mul_out",
        "ir1_dim",
        "ir2_dim",
        "ir_out_dim",
    )

    def __init__(
        self,
        i_in1: int,
        i_in2: int,
        i_out: int,
        connection_mode: str,
        has_weight: bool,
        path_weight: float = 1.0,
    ):
        self.i_in1 = i_in1
        self.i_in2 = i_in2
        self.i_out = i_out
        self.connection_mode = connection_mode
        self.has_weight = has_weight
        self.path_weight = path_weight


class TensorProduct(nn.Module):
    """General tensor product with configurable instructions.

    Supports external weights (for use with RadialMLP in InteractionBlock)
    and internal weights (for FullyConnectedTensorProduct).

    Connection modes:
        - "uvw": out[w,k] = sum_{u,v} W[u,v,w] * CG_tp(x1[u], x2[v])
        - "uvu": out[u,k] = sum_v    W[u,v]   * CG_tp(x1[u], x2[v])
        - "uuu": out[u,k] =          W[u]     * CG_tp(x1[u], x2[u])
        - "uuw": out[w,k] = sum_u    W[u,w]   * CG_tp(x1[u], x2[u])
    """

    def __init__(
        self,
        irreps_in1: str | Irreps,
        irreps_in2: str | Irreps,
        irreps_out: str | Irreps,
        instructions: list[tuple],
        shared_weights: bool = False,
        internal_weights: bool = False,
    ):
        super().__init__()
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        self._shared_weights = shared_weights
        self._internal_weights = internal_weights

        self._instructions: list[Instruction] = []
        self._cg_tensors: list[mx.array] = []
        # Map from instruction index -> internal weight index (only for has_weight instructions)
        self._weight_indices: list[int | None] = []

        internal_weight_list = []
        weight_idx = 0

        for inst_tuple in instructions:
            i_in1, i_in2, i_out, mode, has_weight = inst_tuple[:5]
            path_weight = inst_tuple[5] if len(inst_tuple) > 5 else 1.0

            mul1, ir1 = self.irreps_in1[i_in1]
            mul2, ir2 = self.irreps_in2[i_in2]
            mul_out, ir_out = self.irreps_out[i_out]

            # Selection rule check
            assert abs(ir1.l - ir2.l) <= ir_out.l <= ir1.l + ir2.l, (
                f"Selection rule violated: |{ir1.l}-{ir2.l}| <= {ir_out.l} <= {ir1.l}+{ir2.l}"
            )
            assert ir1.p * ir2.p == ir_out.p, (
                f"Parity violated: {ir1.p}*{ir2.p} != {ir_out.p}"
            )

            # Precompute coupling coefficient (wigner_3j, unit-norm convention
            # matching e3nn's TensorProduct)
            w3j = wigner_3j(ir1.l, ir2.l, ir_out.l)

            self._cg_tensors.append(mx.stop_gradient(mx.array(w3j.astype(np.float32))))

            # Weight shape from mode
            if mode == "uvw":
                weight_shape = (mul1, mul2, mul_out) if has_weight else None
            elif mode == "uvu":
                assert mul1 == mul_out, f"uvu requires mul_in1==mul_out, got {mul1}!={mul_out}"
                weight_shape = (mul1, mul2) if has_weight else None
            elif mode == "uuu":
                assert mul1 == mul2 == mul_out, (
                    f"uuu requires mul_in1==mul_in2==mul_out, got {mul1},{mul2},{mul_out}"
                )
                weight_shape = (mul1,) if has_weight else None
            elif mode == "uuw":
                assert mul1 == mul2, f"uuw requires mul_in1==mul_in2, got {mul1}!={mul2}"
                weight_shape = (mul1, mul_out) if has_weight else None
            else:
                raise ValueError(f"Unknown connection mode: {mode}")

            inst = Instruction(i_in1, i_in2, i_out, mode, has_weight, path_weight)
            inst.weight_shape = weight_shape
            inst.mul1 = mul1
            inst.mul2 = mul2
            inst.mul_out = mul_out
            inst.ir1_dim = ir1.dim
            inst.ir2_dim = ir2.dim
            inst.ir_out_dim = ir_out.dim

            if has_weight and weight_shape is not None:
                self._weight_indices.append(weight_idx)
                if internal_weights:
                    w = mx.random.normal(shape=weight_shape)
                    internal_weight_list.append(w)
                weight_idx += 1
            else:
                self._weight_indices.append(None)

            self._instructions.append(inst)

        if internal_weights and internal_weight_list:
            self.weights = internal_weight_list

        # Precompute fast-path info for ir1_dim==1.
        # When ir1_dim==1, CG is (1, d, d) and the 3-input einsum collapses:
        #   einsum("...u1,...vj,1jk->...uvk") = x1[...,u,:,None] * (x2 @ cg[0].T)[...,None,v,:]
        # Two sub-cases:
        #   a) cg[0] = c * I_d (scalar identity) → c * x1 * x2 (no matmul)
        #   b) cg[0] is general (d,d) matrix → x2 @ cg_2d_T (one matmul)
        self._cg_scalars: list[float | None] = []
        self._cg_2d_T: list[mx.array | None] = []
        # Precompute fast-path info for mul2==1 with general CG.
        # When mul2==1 the v-dimension is trivial and the 3-input einsum
        #   einsum("...ui,...j,ijk->...uk", x1, x2, cg)
        # can be decomposed into:
        #   t = x1 @ cg_mat           (matmul, shape B,u,ir2*k)
        #   tp = (t * x2[...,None,:,None]).sum(-2)  (broadcast + reduce)
        # This is much faster than the fused 3-input einsum.
        self._cg_mul2_1: list[mx.array | None] = []
        for inst_idx, inst in enumerate(self._instructions):
            cg = self._cg_tensors[inst_idx]
            if inst.ir1_dim == 1 and inst.ir2_dim == inst.ir_out_dim:
                cg_mat = np.array(cg[0])  # (d, d)
                c = cg_mat[0, 0]
                if np.allclose(cg_mat, c * np.eye(inst.ir2_dim), atol=1e-6):
                    self._cg_scalars.append(float(c))
                    self._cg_2d_T.append(None)
                else:
                    # General matrix: store transposed for matmul fast path
                    self._cg_scalars.append(None)
                    self._cg_2d_T.append(mx.stop_gradient(mx.array(cg_mat.T.astype(np.float32))))
            else:
                self._cg_scalars.append(None)
                self._cg_2d_T.append(None)
            # mul2==1 matmul decomposition: reshape CG (i,j,k) -> (i, j*k)
            if (
                inst.mul2 == 1
                and inst.ir1_dim > 1
                and self._cg_scalars[inst_idx] is None
                and self._cg_2d_T[inst_idx] is None
            ):
                cg_np = np.array(cg)  # (ir1_dim, ir2_dim, ir_out_dim)
                cg_reshaped = cg_np.reshape(inst.ir1_dim, inst.ir2_dim * inst.ir_out_dim)
                self._cg_mul2_1.append(
                    mx.stop_gradient(mx.array(cg_reshaped.astype(np.float32)))
                )
            else:
                self._cg_mul2_1.append(None)

        # Batched UVU scalar fast path: when ALL instructions share:
        #   - uvu mode, has_weight, external weights, ir1_dim=1, mul2=1
        #   - same i_in1 (same input block for x1)
        #   - each maps to a unique i_out (no accumulation needed)
        #   - cg_scalar or cg_2d_T fast path available
        # This batches 4 separate matmul+multiply sequences into 1,
        # dramatically simplifying the autograd graph.
        self._batched_uvu_scalar = False
        if (
            len(self._instructions) > 1
            and not internal_weights
            and all(
                inst.connection_mode == "uvu"
                and inst.has_weight
                and inst.ir1_dim == 1
                and inst.mul2 == 1
                and (self._cg_scalars[i] is not None or self._cg_2d_T[i] is not None)
                for i, inst in enumerate(self._instructions)
            )
        ):
            # Check same i_in1 for all instructions
            i_in1_set = {inst.i_in1 for inst in self._instructions}
            # Check each instruction maps to a unique i_out
            i_out_list = [inst.i_out for inst in self._instructions]
            if len(i_in1_set) == 1 and len(i_out_list) == len(set(i_out_list)):
                self._batched_uvu_scalar = True
                self._batched_i_in1 = self._instructions[0].i_in1
                self._batched_mul1 = self._instructions[0].mul1

                # Build block-diagonal CG rotation matrix (transposed).
                # For scalar CG (c*I), the block is c*I_d.
                # For general CG, the block is cg_2d_T.
                # Total size: sum of ir2_dim across instructions.
                total_ir2 = sum(inst.ir2_dim for inst in self._instructions)
                cg_block_np = np.zeros((total_ir2, total_ir2), dtype=np.float32)
                row = 0
                for i, inst in enumerate(self._instructions):
                    d = inst.ir2_dim
                    if self._cg_scalars[i] is not None:
                        cg_block_np[row:row+d, row:row+d] = (
                            self._cg_scalars[i] * np.eye(d, dtype=np.float32)
                        )
                    else:
                        cg_block_np[row:row+d, row:row+d] = np.array(
                            self._cg_2d_T[i]
                        )
                    row += d
                self._cg_block_diag = mx.stop_gradient(mx.array(cg_block_np))

                # Store path weights as a vector for broadcast multiply.
                # Shape: (total_ir2,) with each instruction's path_weight
                # repeated ir_out_dim times.
                pw_np = np.ones(total_ir2, dtype=np.float32)
                offset = 0
                for inst in self._instructions:
                    if inst.path_weight != 1.0:
                        pw_np[offset:offset+inst.ir_out_dim] = inst.path_weight
                    offset += inst.ir_out_dim
                self._batched_path_weights = mx.stop_gradient(mx.array(pw_np))
                self._batched_has_nonunit_pw = any(
                    inst.path_weight != 1.0 for inst in self._instructions
                )

                # Store the output slot mapping and ir_dim sizes for
                # reassembly of the output.
                self._batched_ir_dims = [inst.ir_out_dim for inst in self._instructions]
                self._batched_i_outs = [inst.i_out for inst in self._instructions]
                self._batched_total_ir2 = total_ir2

                # Store per-instruction weight size for slicing
                self._batched_w_sizes = [
                    int(np.prod(inst.weight_shape))
                    for inst in self._instructions
                ]

        # Batched mul2=1 fast path ("x2-first" contraction reordering).
        # Applies to the second-layer conv_tp of L>0 models (mixed 0e+1o node
        # features): all instructions are uvu / has_weight / mul2=1 with a
        # common multiplicity, each classified as either cg-scalar (0e source)
        # or cg_mul2_1 (l>0 source). Instead of one matmul chain per
        # instruction, x2 is contracted with the (path-weighted) CG constants
        # FIRST — producing a small (E, d1, K1) matrix with no mul factor —
        # followed by a single batched matmul against the l>0 source block.
        # Weight expansion uses constant 0/1 selector GEMMs rather than
        # mx.take so the VJP is a transposed GEMM (segment-sum) instead of a
        # scatter-add. Measured on mpa0-medium layer 2: block forward ~3.4x,
        # full-gradient ~1.4x, whole-step ~1.2x.
        self._batched_mul21 = False
        self._metal_mul21 = None
        if (
            not self._batched_uvu_scalar
            and len(self._instructions) > 1
            and not internal_weights
            and all(
                inst.connection_mode == "uvu"
                and inst.has_weight
                and inst.mul2 == 1
                and inst.mul1 == self._instructions[0].mul1
                and inst.weight_shape == (inst.mul1, 1)
                and (self._cg_scalars[i] is not None
                     or self._cg_mul2_1[i] is not None)
                for i, inst in enumerate(self._instructions)
            )
        ):
            self._setup_batched_mul21()
            if self._batched_mul21:
                self._setup_metal_mul21()

    def _setup_batched_mul21(self) -> None:
        insts = self._instructions
        n = len(insts)
        i_outs = [inst.i_out for inst in insts]
        # Require unique i_out covering every output slot (true for the
        # conv_tp layouts produced by tp_out_irreps_with_instructions).
        if len(set(i_outs)) != n or n != len(self.irreps_out):
            return

        scal = [i for i in range(n) if self._cg_scalars[i] is not None]
        m21 = [i for i in range(n) if self._cg_scalars[i] is None]
        if not m21:
            return
        # Single source block per group; uniform ir1_dim in the m21 group.
        if len({insts[i].i_in1 for i in m21}) != 1:
            return
        if scal and len({insts[i].i_in1 for i in scal}) != 1:
            return
        d1 = insts[m21[0]].ir1_dim
        if any(insts[i].ir1_dim != d1 for i in m21):
            return

        sl1 = self.irreps_in1.slices
        sl2 = self.irreps_in2.slices
        x2_dim = self.irreps_in2.dim
        mul = insts[0].mul1

        K0 = sum(insts[i].ir_out_dim for i in scal)
        K1 = sum(insts[i].ir_out_dim for i in m21)

        # Scalar group: out0 = x1_0e * (w @ T0) * (x2 @ S)
        S = np.zeros((x2_dim, K0), dtype=np.float32)
        scal_segs = {}
        ids0 = []
        off = 0
        for i in scal:
            inst = insts[i]
            s2 = sl2[inst.i_in2].start
            c = self._cg_scalars[i] * inst.path_weight
            for k in range(inst.ir_out_dim):
                S[s2 + k, off + k] = c
            scal_segs[i] = (off, inst.ir_out_dim)
            ids0 += [i] * inst.ir_out_dim
            off += inst.ir_out_dim

        # mul2_1 group: M1 = (x2 @ G1).reshape(E, d1, K1); out1 = x1_l @ M1
        G1 = np.zeros((x2_dim, d1 * K1), dtype=np.float32)
        m21_segs = {}
        ids1 = []
        off = 0
        for i in m21:
            inst = insts[i]
            s2 = sl2[inst.i_in2].start
            cg = np.array(self._cg_tensors[i])  # (d1, d2, do)
            pw = inst.path_weight
            do = inst.ir_out_dim
            for m in range(d1):
                for j in range(inst.ir2_dim):
                    for k in range(do):
                        G1[s2 + j, m * K1 + off + k] = cg[m, j, k] * pw
            m21_segs[i] = (off, do)
            ids1 += [i] * do
            off += do

        # 0/1 selector matrices: weight expansion as GEMM (VJP = segment-sum)
        T0 = np.zeros((n, K0), dtype=np.float32)
        for col, i in enumerate(ids0):
            T0[i, col] = 1.0
        T1 = np.zeros((n, K1), dtype=np.float32)
        for col, i in enumerate(ids1):
            T1[i, col] = 1.0

        self._batched_mul21 = True
        self._bm21_n = n
        self._bm21_mul = mul
        self._bm21_d1 = d1
        self._bm21_K0 = K0
        self._bm21_K1 = K1
        self._bm21_sl_scal = sl1[insts[scal[0]].i_in1] if scal else None
        self._bm21_sl_m21 = sl1[insts[m21[0]].i_in1]
        self._bm21_S = mx.stop_gradient(mx.array(S)) if scal else None
        self._bm21_G1 = mx.stop_gradient(mx.array(G1))
        self._bm21_T0 = mx.stop_gradient(mx.array(T0)) if scal else None
        self._bm21_T1 = mx.stop_gradient(mx.array(T1))
        # Output assembly: (is_scalar_group, segment offset, ir_out_dim)
        # in output-slot order.
        slot_order = sorted(range(n), key=lambda i: insts[i].i_out)
        self._bm21_slots = [
            (i in scal_segs,
             (scal_segs[i][0] if i in scal_segs else m21_segs[i][0]),
             insts[i].ir_out_dim)
            for i in slot_order
        ]

    def _batched_mul21_forward(
        self, x1: mx.array, x2: mx.array, weight: mx.array
    ) -> mx.array:
        batch_shape = x1.shape[:-1]
        E = 1
        for s in batch_shape:
            E *= s
        x1 = x1.reshape(E, x1.shape[-1])
        x2 = x2.reshape(E, x2.shape[-1])
        mul, d1 = self._bm21_mul, self._bm21_d1

        # (E, n_inst*mul) -> (E, mul, n_inst)
        w_t = weight.reshape(E, self._bm21_n, mul).transpose(0, 2, 1)

        # l>0 group: x2-first contraction, then one batched matmul
        M1 = (x2 @ self._bm21_G1).reshape(E, d1, self._bm21_K1)
        x1_m = x1[:, self._bm21_sl_m21].reshape(E, mul, d1)
        out1 = (x1_m @ M1) * (w_t @ self._bm21_T1)  # (E, mul, K1)

        out0 = None
        if self._bm21_S is not None:
            xs = x2 @ self._bm21_S  # (E, K0)
            x1_s = x1[:, self._bm21_sl_scal].reshape(E, mul, 1)
            out0 = x1_s * (w_t @ self._bm21_T0) * xs[:, None, :]

        parts = []
        for is_scal, off, d in self._bm21_slots:
            src = out0 if is_scal else out1
            parts.append(src[:, :, off:off + d].reshape(E, mul * d))
        out = mx.concatenate(parts, axis=-1)
        return out.reshape(*batch_shape, out.shape[-1])

    def _setup_metal_mul21(self) -> None:
        """Build the fused Metal kernels for the batched_mul21 path.

        GPU-only fast path on top of _batched_mul21 (guards below fall back
        to the GEMM path). Sets self._metal_mul21 to a custom_function
        (x1, x2, w) -> mji or leaves it None.
        """
        insts = self._instructions
        n = self._bm21_n
        mul, d1 = self._bm21_mul, self._bm21_d1
        K0, K1 = self._bm21_K0, self._bm21_K1
        # Kernel assumptions: a scalar group exists, threadgroup = mul
        # (simd_sum reductions need mul % 32 == 0; Apple GPUs cap
        # threadgroups at 1024).
        if self._bm21_S is None or mul % 32 != 0 or mul > 1024:
            return
        sl_scal, sl_m21 = self._bm21_sl_scal, self._bm21_sl_m21
        x1dim = self.irreps_in1.dim
        # dx1 is written only through the two source slices; they must tile
        # the x1 row exactly or backward would leave uninitialized gaps.
        spans = sorted(
            [(sl_scal.start, sl_scal.stop), (sl_m21.start, sl_m21.stop)]
        )
        if (
            spans[0][0] != 0
            or spans[0][1] != spans[1][0]
            or spans[1][1] != x1dim
        ):
            return

        # Per-slot metadata in output-slot order (same segment offsets as
        # _setup_batched_mul21 builds for scal_segs/m21_segs).
        scal_set = {i for i in range(n) if self._cg_scalars[i] is not None}
        seg = {}
        off = 0
        for i in range(n):
            if i in scal_set:
                seg[i] = off
                off += insts[i].ir_out_dim
        off = 0
        for i in range(n):
            if i not in scal_set:
                seg[i] = off
                off += insts[i].ir_out_dim
        slot_order = sorted(range(n), key=lambda i: insts[i].i_out)
        slots = []
        outdim = 0
        for i in slot_order:
            d = insts[i].ir_out_dim
            slots.append(
                dict(inst=i, scal=(i in scal_set), seg=seg[i], d=d, off=outdim)
            )
            outdim += mul * d
        if outdim != self.irreps_out.dim:
            return
        dmax = max(s["d"] for s in slots)
        wdim = n * mul

        # Constants are read by the compiled graph via closure over self;
        # materialize them now (MLX 0.31.2 reads garbage from lazy derived
        # arrays captured before their first eval).
        mx.eval(self._bm21_G1, self._bm21_S)

        cfg = (
            f"{mul}_{d1}_{K0}_{K1}_{n}_{outdim}_{x1dim}"
            f"_{sl_scal.start}_{sl_m21.start}"
            + str([tuple(sorted(s.items())) for s in slots])
        )
        sfx = hashlib.md5(cfg.encode()).hexdigest()[:8]
        arr = lambda key: ", ".join(str(int(s[key])) for s in slots)
        header = f"""
constant uint NSLOT_{sfx} = {n};
constant uint SLOT_INST_{sfx}[{n}] = {{{arr('inst')}}};
constant uint SLOT_SCAL_{sfx}[{n}] = {{{", ".join("1" if s["scal"] else "0" for s in slots)}}};
constant uint SLOT_SEG_{sfx}[{n}]  = {{{arr('seg')}}};
constant uint SLOT_D_{sfx}[{n}]    = {{{arr('d')}}};
constant uint SLOT_OFF_{sfx}[{n}]  = {{{arr('off')}}};
"""
        k_fwd = mx.fast.metal_kernel(
            name=f"mul21_fwd_{sfx}",
            input_names=["x1", "M1", "xs", "w"],
            output_names=["mji"],
            source=_METAL_MUL21_FWD_SRC.format(sfx=sfx),
            header=header,
        )
        k_bwd = mx.fast.metal_kernel(
            name=f"mul21_bwd_{sfx}",
            input_names=["cot", "x1", "M1", "xs", "w"],
            output_names=["dx1_out", "dM1_out", "dxs_out", "dw_out"],
            source=_METAL_MUL21_BWD_SRC.format(sfx=sfx),
            header=header,
        )
        tmpl = [
            ("MUL", mul), ("D1", d1), ("K0", K0), ("K1", K1),
            ("X1DIM", x1dim), ("WDIM", wdim), ("OUTDIM", outdim),
            ("SL_SCAL", sl_scal.start), ("SL_M21", sl_m21.start),
            ("DMAX", dmax),
        ]

        # G1/S are read from self at call time so dtype conversion
        # (set_dtype/_convert_private_arrays) is picked up; only Python
        # object references are closed over, not mx.arrays.
        @mx.custom_function
        def fused_mul21(x1, x2, w):
            E = x1.shape[0]
            M1 = x2 @ self._bm21_G1
            xs = x2 @ self._bm21_S
            return k_fwd(
                inputs=[x1, M1, xs, w],
                template=[("T", x1.dtype)] + tmpl,
                grid=(mul, E, 1),
                threadgroup=(mul, 1, 1),
                output_shapes=[(E, outdim)],
                output_dtypes=[x1.dtype],
            )[0]

        @fused_mul21.vjp
        def _fused_mul21_vjp(primals, cotan, output):
            x1, x2, w = primals
            E = x1.shape[0]
            M1 = x2 @ self._bm21_G1
            xs = x2 @ self._bm21_S
            dx1, dM1, dxs, dw = k_bwd(
                inputs=[cotan, x1, M1, xs, w],
                template=[("T", x1.dtype)] + tmpl,
                grid=(mul, E, 1),
                threadgroup=(mul, 1, 1),
                output_shapes=[
                    (E, x1dim), (E, d1 * K1), (E, K0), (E, wdim)
                ],
                output_dtypes=[x1.dtype] * 4,
            )
            dx2 = dM1 @ self._bm21_G1.T + dxs @ self._bm21_S.T
            return dx1, dx2, dw

        self._metal_mul21 = fused_mul21

    def _metal_mul21_forward(
        self, x1: mx.array, x2: mx.array, weight: mx.array
    ) -> mx.array:
        batch_shape = x1.shape[:-1]
        E = 1
        for s in batch_shape:
            E *= s
        out = self._metal_mul21(
            x1.reshape(E, x1.shape[-1]),
            x2.reshape(E, x2.shape[-1]),
            weight.reshape(E, weight.shape[-1]),
        )
        return out.reshape(*batch_shape, out.shape[-1])

    @property
    def weight_numel(self) -> int:
        """Total number of weight parameters needed (for external weights)."""
        total = 0
        for inst in self._instructions:
            if inst.has_weight and inst.weight_shape is not None:
                total += int(np.prod(inst.weight_shape))
        return total

    def __call__(
        self, x1: mx.array, x2: mx.array, weight: mx.array | None = None
    ) -> mx.array:
        """Forward pass.

        Args:
            x1: (..., irreps_in1.dim)
            x2: (..., irreps_in2.dim)
            weight: external weights (..., weight_numel) when not using internal_weights
        Returns:
            (..., irreps_out.dim)
        """
        if self._batched_uvu_scalar and weight is not None:
            return self._batched_forward(x1, x2, weight)
        if (
            self._metal_mul21 is not None
            and weight is not None
            and mx.default_device().type == mx.DeviceType.gpu
        ):
            return self._metal_mul21_forward(x1, x2, weight)
        if self._batched_mul21 and weight is not None:
            return self._batched_mul21_forward(x1, x2, weight)
        return self._loop_forward(x1, x2, weight)

    def _batched_forward(
        self, x1: mx.array, x2: mx.array, weight: mx.array
    ) -> mx.array:
        """Batched forward for uvu-scalar instructions.

        All instructions have ir1_dim=1, mul2=1, uvu mode with external
        weights. We batch the CG rotation into one block-diagonal matmul
        and keep weight application as simple slicing (faster than gather).

        Per-instruction:
            x2_rot_i = x2_slice @ cg_2d_T_i      → batched into one matmul
            out_i = x1 * w_i * x2_rot_i           → simple broadcast

        This reduces 4 CG matmuls to 1 and simplifies the autograd graph
        from ~16 ops to ~7 ops.
        """
        batch_shape = x1.shape[:-1]
        mul1 = self._batched_mul1

        # x1_block: (..., mul1, 1) — same for all instructions
        slices_in1 = self.irreps_in1.slices
        x1_block = x1[..., slices_in1[self._batched_i_in1]]
        x1_block = x1_block.reshape(*batch_shape, mul1, 1)

        # Step 1: CG rotation — one matmul replaces N separate matmuls
        # x2: (..., total_ir2), CG_block_diag: (total_ir2, total_ir2)
        x2_rot = x2 @ self._cg_block_diag  # (..., total_ir2)

        # Apply path weights (baked into x2_rot to avoid extra op later)
        if self._batched_has_nonunit_pw:
            x2_rot = x2_rot * self._batched_path_weights

        # Step 2: For each instruction, compute x1 * w * x2_rot via broadcast.
        # w_i: (..., mul1) → (..., mul1, 1)
        # x2_rot_i: (..., d_i) → (..., 1, d_i)
        # out_i: (..., mul1, d_i) → (..., mul1 * d_i)
        out_blocks = []
        w_offset = 0
        x2_offset = 0
        for inst_idx in range(len(self._instructions)):
            d_i = self._batched_ir_dims[inst_idx]
            w_size = self._batched_w_sizes[inst_idx]

            w_i = weight[..., w_offset : w_offset + w_size]
            w_i = w_i.reshape(*batch_shape, mul1, 1)
            w_offset += w_size

            x2_rot_i = x2_rot[..., x2_offset : x2_offset + d_i]
            x2_rot_i = mx.expand_dims(x2_rot_i, axis=-2)
            x2_offset += d_i

            out_i = (x1_block * w_i) * x2_rot_i
            out_blocks.append(out_i.reshape(*batch_shape, mul1 * d_i))

        # Assemble output
        num_out_slots = len(self.irreps_out)
        i_outs = self._batched_i_outs
        if i_outs == list(range(num_out_slots)):
            return mx.concatenate(out_blocks, axis=-1)

        # General case: place blocks into correct output slots
        out_parts = [None] * num_out_slots
        for idx, i_out in enumerate(i_outs):
            out_parts[i_out] = out_blocks[idx]
        for i in range(num_out_slots):
            if out_parts[i] is None:
                mulir = self.irreps_out[i]
                out_parts[i] = mx.zeros(
                    (*batch_shape, mulir.mul * mulir.ir.dim)
                )
        return mx.concatenate(out_parts, axis=-1)

    def _loop_forward(
        self, x1: mx.array, x2: mx.array, weight: mx.array | None = None
    ) -> mx.array:
        """Original loop-based forward pass."""
        batch_shape = x1.shape[:-1]
        slices_in1 = self.irreps_in1.slices
        slices_in2 = self.irreps_in2.slices

        # Initialize output accumulator per irreps_out slot
        out_parts = [
            mx.zeros((*batch_shape, mulir.mul * mulir.ir.dim), dtype=x1.dtype)
            for mulir in self.irreps_out
        ]

        weight_offset = 0

        for inst_idx, inst in enumerate(self._instructions):
            cg = self._cg_tensors[inst_idx]

            # Extract input blocks
            x1_block = x1[..., slices_in1[inst.i_in1]]
            x2_block = x2[..., slices_in2[inst.i_in2]]
            x1_block = x1_block.reshape(*batch_shape, inst.mul1, inst.ir1_dim)
            x2_block = x2_block.reshape(*batch_shape, inst.mul2, inst.ir2_dim)

            # Resolve weight
            w = None
            if inst.has_weight:
                widx = self._weight_indices[inst_idx]
                if self._internal_weights and widx is not None:
                    w = self.weights[widx]
                elif weight is not None:
                    w_size = int(np.prod(inst.weight_shape))
                    w_flat = weight[..., weight_offset : weight_offset + w_size]
                    if len(batch_shape) > 0:
                        w = w_flat.reshape(*batch_shape, *inst.weight_shape)
                    else:
                        w = w_flat.reshape(*inst.weight_shape)
                    weight_offset += w_size
                else:
                    raise ValueError(
                        "External weights required but not provided"
                    )

            # Compute tensor product per mode
            cg_scalar = self._cg_scalars[inst_idx]
            cg_2d_T = self._cg_2d_T[inst_idx]
            cg_mul2_1 = self._cg_mul2_1[inst_idx]
            out_block = self._compute_mode(
                inst, x1_block, x2_block, cg, w, batch_shape, cg_scalar, cg_2d_T,
                cg_mul2_1,
            )

            # Apply path weight and accumulate
            if inst.path_weight != 1.0:
                out_block = out_block * inst.path_weight

            out_flat = out_block.reshape(*batch_shape, inst.mul_out * inst.ir_out_dim)
            out_parts[inst.i_out] = out_parts[inst.i_out] + out_flat

        return mx.concatenate(out_parts, axis=-1)

    @staticmethod
    def _compute_mode(
        inst: Instruction,
        x1_block: mx.array,
        x2_block: mx.array,
        cg: mx.array,
        w: mx.array | None,
        batch_shape: tuple,
        cg_scalar: float | None = None,
        cg_2d_T: mx.array | None = None,
        cg_mul2_1: mx.array | None = None,
    ) -> mx.array:
        """Compute tensor product for a single instruction.

        Fast paths for ir1_dim==1:
            cg_scalar: CG is c * identity -> broadcast multiply
            cg_2d_T:   CG[0] is general matrix -> matmul x2 @ cg_2d_T

        Fast path for mul2==1, ir1_dim>1:
            cg_mul2_1: CG reshaped to (ir1_dim, ir2_dim*ir_out_dim) for matmul
            Decomposes 3-input einsum into matmul + broadcast + reduce.
        """
        mode = inst.connection_mode

        if mode == "uvw":
            if cg_scalar is not None:
                # ir1_dim==1, CG = c*I: tp = c * x1[...,u,0:1,None] * x2[...,None,v,k]
                tp = cg_scalar * x1_block[..., :, 0:1, None] * x2_block[..., None, :, :]
            elif cg_2d_T is not None:
                # ir1_dim==1, CG = general matrix: rotate x2 then broadcast
                x2_rot = x2_block @ cg_2d_T  # (..., v, k)
                tp = x1_block[..., :, 0:1, None] * x2_rot[..., None, :, :]
            else:
                tp = mx.einsum("...ui,...vj,ijk->...uvk", x1_block, x2_block, cg)
            if w is not None:
                if w.ndim == len(batch_shape) + 3:
                    out_block = mx.einsum("...uvk,...uvw->...wk", tp, w)
                else:
                    out_block = mx.einsum("...uvk,uvw->...wk", tp, w)
            else:
                out_block = tp.sum(axis=(-3, -2))
                out_block = out_block.reshape(*batch_shape, 1, inst.ir_out_dim)

        elif mode == "uvu":
            if cg_scalar is not None:
                # ir1_dim==1, CG = c*I
                if w is not None:
                    if inst.mul2 == 1:
                        # v-dim is trivial: (..., u, 1) * (..., 1, k) broadcast
                        wx2 = w * x2_block
                    elif w.ndim == len(batch_shape) + 2:
                        wx2 = mx.einsum("...uv,...vk->...uk", w, x2_block)
                    else:
                        wx2 = mx.einsum("uv,...vk->...uk", w, x2_block)
                    out_block = (cg_scalar * x1_block) * wx2
                    return out_block
                else:
                    x2_sum = x2_block.sum(axis=-2, keepdims=True)
                    out_block = cg_scalar * x1_block * x2_sum
                    out_block = mx.broadcast_to(
                        out_block, (*batch_shape, inst.mul1, inst.ir_out_dim)
                    )
                    return out_block
            elif cg_2d_T is not None:
                # ir1_dim==1, CG = general matrix: rotate x2 then same pattern
                x2_rot = x2_block @ cg_2d_T  # (..., v, k)
                if w is not None:
                    if w.ndim == len(batch_shape) + 2:
                        wx2 = mx.einsum("...uv,...vk->...uk", w, x2_rot)
                    else:
                        wx2 = mx.einsum("uv,...vk->...uk", w, x2_rot)
                    out_block = x1_block * wx2
                    return out_block
                else:
                    x2_sum = x2_rot.sum(axis=-2, keepdims=True)
                    out_block = x1_block * x2_sum
                    out_block = mx.broadcast_to(
                        out_block, (*batch_shape, inst.mul1, inst.ir_out_dim)
                    )
                    return out_block
            elif cg_mul2_1 is not None:
                # mul2==1 matmul decomposition: replaces 3-input einsum.
                # x1: (..., mul1, ir1_dim), x2: (..., 1, ir2_dim)
                # cg_mul2_1: (ir1_dim, ir2_dim * ir_out_dim)
                # Step 1: t = x1 @ cg_mul2_1  -> (..., mul1, ir2_dim * ir_out_dim)
                t = x1_block @ cg_mul2_1  # matmul, the expensive part done efficiently
                # Step 2: reshape -> (..., mul1, ir2_dim, ir_out_dim)
                t = t.reshape(*batch_shape, inst.mul1, inst.ir2_dim, inst.ir_out_dim)
                # Step 3: contract with x2 over ir2_dim.
                # x2_vec: (..., ir2_dim) from squeezing v=1 dim
                x2_vec = x2_block[..., 0, :]  # (..., ir2_dim)
                # Broadcast: x2_vec -> (..., 1, ir2_dim, 1)
                tp = (t * x2_vec[..., None, :, None]).sum(axis=-2)
                # tp: (..., mul1, ir_out_dim)
                if w is not None:
                    # uvu with mul2=1: w shape is (..., mul1, 1) or (mul1, 1)
                    # out = tp * w[..., :, 0:1]  (broadcast over ir_out_dim)
                    if w.ndim == len(batch_shape) + 2:
                        out_block = tp * w[..., :, 0:1]
                    else:
                        out_block = tp * w[:, 0:1]
                else:
                    out_block = tp
                return out_block
            else:
                tp = mx.einsum("...ui,...vj,ijk->...uvk", x1_block, x2_block, cg)
                if w is not None:
                    if w.ndim == len(batch_shape) + 2:
                        out_block = mx.einsum("...uvk,...uv->...uk", tp, w)
                    else:
                        out_block = mx.einsum("...uvk,uv->...uk", tp, w)
                else:
                    out_block = tp.sum(axis=-2)

        elif mode == "uuu":
            if cg_scalar is not None:
                tp = cg_scalar * x1_block * x2_block
            elif cg_2d_T is not None:
                x2_rot = x2_block @ cg_2d_T
                tp = x1_block * x2_rot
            else:
                tp = mx.einsum("...ui,...uj,ijk->...uk", x1_block, x2_block, cg)
            if w is not None:
                if w.ndim == len(batch_shape) + 1:
                    out_block = tp * w[..., :, None]
                else:
                    out_block = tp * w[:, None]
            else:
                out_block = tp

        elif mode == "uuw":
            if cg_scalar is not None:
                tp = cg_scalar * x1_block * x2_block
            elif cg_2d_T is not None:
                x2_rot = x2_block @ cg_2d_T
                tp = x1_block * x2_rot
            else:
                tp = mx.einsum("...ui,...uj,ijk->...uk", x1_block, x2_block, cg)
            if w is not None:
                if w.ndim == len(batch_shape) + 2:
                    out_block = mx.einsum("...uk,...uw->...wk", tp, w)
                else:
                    out_block = mx.einsum("...uk,uw->...wk", tp, w)
            else:
                out_block = tp.sum(axis=-2).reshape(*batch_shape, 1, inst.ir_out_dim)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return out_block


class FullyConnectedTensorProduct(nn.Module):
    """Fully connected tensor product with internal weights.

    Generates all valid coupling paths (satisfying selection rules)
    between irreps_in1, irreps_in2 and irreps_out, each with a
    learnable weight in "uvw" mode.

    Path normalization matches e3nn: path_weight = 1/sqrt(sum_alpha / ir_out.dim)
    where sum_alpha = sum of mul1*mul2 over all paths to the same output slot.
    """

    def __init__(
        self,
        irreps_in1: str | Irreps,
        irreps_in2: str | Irreps,
        irreps_out: str | Irreps,
    ):
        super().__init__()
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)

        # Generate all valid instructions (matching e3nn iteration order:
        # for i_in1, for i_in2, for each valid i_out from the coupling)
        instructions = []
        for i_in1, (mul_in1, ir_in1) in enumerate(irreps_in1):
            for i_in2, (mul_in2, ir_in2) in enumerate(irreps_in2):
                for i_out, (mul_out, ir_out) in enumerate(irreps_out):
                    if ir_in1.p * ir_in2.p != ir_out.p:
                        continue
                    if not (abs(ir_in1.l - ir_in2.l) <= ir_out.l <= ir_in1.l + ir_in2.l):
                        continue
                    instructions.append((i_in1, i_in2, i_out, "uvw", True))

        # Compute path normalization (matching e3nn FCTP convention)
        # For each output slot: path_weight = 1/sqrt(sum_alpha / ir_out.dim)
        # where alpha per path = mul_in1 * mul_in2 (for uvw mode)
        alpha_per_out: dict[int, float] = {}
        for inst_tuple in instructions:
            i_in1, i_in2, i_out = inst_tuple[0], inst_tuple[1], inst_tuple[2]
            mul1 = irreps_in1[i_in1].mul
            mul2 = irreps_in2[i_in2].mul
            alpha_per_out[i_out] = alpha_per_out.get(i_out, 0.0) + mul1 * mul2

        normalized_instructions = []
        for inst_tuple in instructions:
            i_in1, i_in2, i_out, mode, has_weight = inst_tuple
            ir_out_dim = irreps_out[i_out].ir.dim
            total_alpha = alpha_per_out[i_out]
            path_weight = (ir_out_dim / total_alpha) ** 0.5 if total_alpha > 0 else 1.0
            normalized_instructions.append(
                (i_in1, i_in2, i_out, mode, has_weight, path_weight)
            )

        self.tp = TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=normalized_instructions,
            internal_weights=True,
        )
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out

        # Detect scalar-only fast path: single instruction, all ir_dim=1,
        # "uvw" mode with internal weights and CG = scalar identity.
        # Common case: skip_tp in interaction blocks (e.g. 128x0e x 89x0e -> 128x0e).
        # ONLY activate when BOTH inputs are purely scalar (all l=0).
        self._scalar_fctp_fast = False
        all_scalar_in1 = all(ir.l == 0 for _, ir in irreps_in1)
        all_scalar_in2 = all(ir.l == 0 for _, ir in irreps_in2)
        all_scalar_out = all(ir.l == 0 for _, ir in irreps_out)
        if (
            all_scalar_in1
            and all_scalar_in2
            and all_scalar_out
            and len(self.tp._instructions) == 1
            and len(normalized_instructions) == 1
        ):
            inst = self.tp._instructions[0]
            if (
                inst.connection_mode == "uvw"
                and inst.ir1_dim == 1
                and inst.ir2_dim == 1
                and inst.ir_out_dim == 1
                and self.tp._cg_scalars[0] is not None
            ):
                self._scalar_fctp_fast = True
                cg_s = self.tp._cg_scalars[0]
                pw = inst.path_weight
                self._scalar_fctp_scale = cg_s * pw
                # Precompute W transposed for two-step matmul:
                # W shape: (u, v, w) -> W_vuw shape: (v, u*w)
                self._scalar_fctp_u = inst.mul1
                self._scalar_fctp_w = inst.mul_out

        # Detect scalar-in2 fast path: in2 is all-scalar, but in1/out may
        # have L>0 irreps. When in2 is all 0e, only l_in1 == l_out couplings
        # survive (via l ⊗ 0e -> l), and CG is scalar-identity for each l.
        # Output irreps not covered by any instruction are zero-padded.
        # This avoids the full TP loop and its 3-input einsum for non-scalar
        # instructions (which are in fact never generated).
        # Common case: skip_tp "128x0e+128x1o x 89x0e -> 128x0e+128x1o"
        # where only the 0e instruction is generated.
        self._scalar_in2_fast = False
        if (
            not self._scalar_fctp_fast
            and all_scalar_in2
            and len(self.tp._instructions) >= 1
            and all(
                inst.connection_mode == "uvw"
                and inst.ir2_dim == 1
                and inst.ir1_dim == inst.ir_out_dim
                for inst in self.tp._instructions
            )
        ):
            # For l (x) 0e -> l couplings the CG tensor's [:, 0, :] slice is
            # c * identity for ALL l (wigner_3j(l,0,l) = I/sqrt(2l+1)), not
            # just l=0 — check it directly instead of relying on
            # tp._cg_scalars (which is only populated when ir1_dim == 1).
            cg_diag_scales: list[float] = []
            all_diag = True
            for idx, inst in enumerate(self.tp._instructions):
                cg_m = np.array(self.tp._cg_tensors[idx])[:, 0, :]
                c = float(cg_m[0, 0])
                if not np.allclose(
                    cg_m, c * np.eye(inst.ir1_dim), atol=1e-6
                ):
                    all_diag = False
                    break
                cg_diag_scales.append(c)

            if all_diag:
                self._scalar_in2_fast = True
                # Precompute scale per instruction: cg_diag * path_weight
                self._si2_scales = []
                self._si2_inst_info = []
                for idx, inst in enumerate(self.tp._instructions):
                    scale = cg_diag_scales[idx] * inst.path_weight
                    self._si2_scales.append(scale)
                    self._si2_inst_info.append((
                        inst.mul1, inst.mul2, inst.mul_out, inst.ir1_dim,
                        inst.i_in1, inst.i_in2, inst.i_out,
                    ))
                self._si2_out_dim = irreps_out.dim
                self._si2_out_irreps = irreps_out
                # Lazily-built cache of transposed weight views (see
                # _ensure_si2_weight_cache).
                self._si2_W_vuw: list[mx.array] | None = None
                self._si2_cached_wids: tuple | None = None

    def _ensure_transposed_weight_cache(self) -> list[mx.array]:
        """Cache (v, u*w)-reshaped views of the constant internal weights.

        The lazy graph is rebuilt every step; without this cache the
        transpose+reshape re-materializes a multi-MB weight copy per call.
        The cached arrays are ordinary lazy arrays — MLX materializes them
        once on first eval and reuses the buffer afterwards. Keyed by weight
        identity so replacing the weights invalidates the cache.
        """
        wids = tuple(id(w) for w in self.tp.weights)
        if getattr(self, "_wvuw_cached_ids", None) != wids:
            cache = []
            for W in self.tp.weights:
                mul1, mul2, mul_out = W.shape
                cache.append(
                    mx.transpose(W, axes=(1, 0, 2)).reshape(mul2, mul1 * mul_out)
                )
            self._wvuw_cache = cache
            self._wvuw_cached_ids = wids
        return self._wvuw_cache

    def __call__(self, x1: mx.array, x2: mx.array) -> mx.array:
        if self._scalar_fctp_fast:
            # Scalar-only fast path: 2 matmuls instead of outer product + einsum.
            # result[...,w] = scale * sum_u sum_v x1[...,u] * x2[...,v] * W[u,v,w]
            u, w = self._scalar_fctp_u, self._scalar_fctp_w
            batch_shape = x1.shape[:-1]
            x1_flat = x1.reshape(-1, x1.shape[-1])
            x2_flat = x2.reshape(-1, x2.shape[-1])
            # Step 1: contract x2 over v dimension
            W_vuw = self._ensure_transposed_weight_cache()[0]  # (v, u*w)
            tmp = (x2_flat @ W_vuw).reshape(-1, u, w)  # (B, u, w)
            # Step 2: contract x1 over u dimension via batched matmul
            result = (x1_flat[:, None, :] @ tmp).squeeze(-2)  # (B, w)
            result = result.reshape(*batch_shape, w)
            return self._scalar_fctp_scale * result
        if self._scalar_in2_fast:
            return self._scalar_in2_forward(x1, x2)
        return self.tp(x1, x2)

    def _scalar_in2_forward(self, x1: mx.array, x2: mx.array) -> mx.array:
        """Fast path for FCTP when in2 is all-scalar (0e).

        Each instruction couples l x 0e -> l with CG = c * identity.
        The TP simplifies to:
            out[w, m] = scale * sum_u sum_v x1[u, m] * x2[v] * W[u, v, w]
        which is a per-m broadcast of the scalar FCTP operation.
        Output irreps without any contributing instruction get zeros.
        """
        batch_shape = x1.shape[:-1]
        batch_size = 1
        for s in batch_shape:
            batch_size *= s

        slices_in1 = self.tp.irreps_in1.slices
        slices_in2 = self.tp.irreps_in2.slices
        slices_out = self._si2_out_irreps.slices

        x1_flat = x1.reshape(batch_size, x1.shape[-1])
        x2_flat = x2.reshape(batch_size, x2.shape[-1])

        w_cache = self._ensure_transposed_weight_cache()

        # Build output parts: one per output irrep slot
        num_out_slots = len(self._si2_out_irreps)
        out_parts: list[mx.array | None] = [None] * num_out_slots

        for idx in range(len(self._si2_scales)):
            scale = self._si2_scales[idx]
            mul1, mul2, mul_out, ir_dim, i_in1, i_in2, i_out = self._si2_inst_info[idx]

            # x1_block: (B, mul1, ir_dim)
            x1_block = x1_flat[:, slices_in1[i_in1]].reshape(batch_size, mul1, ir_dim)
            # x2_block: (B, mul2) -- scalar, no m-components
            x2_block = x2_flat[:, slices_in2[i_in2]]

            # Step 1: sum_v x2[v] * W[u, v, w] -> (B, mul1, mul_out)
            W_vuw = w_cache[self.tp._weight_indices[idx]]  # (mul2, mul1*mul_out)
            tmp = (x2_block @ W_vuw).reshape(batch_size, mul1, mul_out)

            # Step 2: sum_u x1[u, m] * tmp[u, w] -> out[w, m]
            # = (x1_block^T @ tmp) in the (mul1) dimension
            if ir_dim == 1:
                out_block = (x1_block.transpose(0, 2, 1) @ tmp).reshape(
                    *batch_shape, mul_out
                )
            else:
                out_block = x1_block.transpose(0, 2, 1) @ tmp  # (B, ir_dim, mul_out)
                out_block = out_block.transpose(0, 2, 1).reshape(
                    *batch_shape, mul_out * ir_dim
                )

            out_block = scale * out_block

            if out_parts[i_out] is None:
                out_parts[i_out] = out_block
            else:
                out_parts[i_out] = out_parts[i_out] + out_block

        # Fill in zero blocks for output slots with no contributing instructions
        for i in range(num_out_slots):
            if out_parts[i] is None:
                mulir = self._si2_out_irreps[i]
                out_parts[i] = mx.zeros(
                    (*batch_shape, mulir.mul * mulir.ir.dim), dtype=x1.dtype
                )

        return mx.concatenate(out_parts, axis=-1)
