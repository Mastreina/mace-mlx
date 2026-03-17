"""Equivariant linear layer for O(3) representations.

Maps between irreps while preserving equivariance: only irreps of the
same type (same l and p) can be linearly mixed, and the weight is shared
across the 2l+1 magnetic quantum numbers.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mace_mlx.irreps import Irreps, Irrep


class _Instruction(NamedTuple):
    i_in: int
    i_out: int
    ir_dim: int
    mul_in: int
    mul_out: int
    path_weight: float  # normalization factor applied to the output


class EquivariantLinear(nn.Module):
    """Equivariant linear map between irreps representations.

    For each matching pair of input/output irreps with the same (l, p),
    creates a weight matrix of shape (mul_in, mul_out).  The weight is
    shared across the 2l+1 magnetic quantum numbers — this is what
    makes the layer equivariant.

    Multiple input irreps of the same type can contribute to the same
    output irrep; their contributions are summed.  Path normalisation
    (element-wise, same as e3nn default) ensures proper scaling.
    """

    def __init__(self, irreps_in: str | Irreps, irreps_out: str | Irreps):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)

        # Build instructions in the same order as e3nn:
        # outer loop over i_in, inner loop over i_out.
        instructions: list[_Instruction] = []
        weights: list[mx.array] = []

        # Pre-compute total input multiplicity for each output slot
        # (for path normalisation).
        total_input_mul: dict[int, int] = {}
        for i_out, (mul_out, ir_out) in enumerate(self.irreps_out):
            total_input_mul[i_out] = sum(
                mul_in
                for mul_in, ir_in in self.irreps_in
                if ir_in == ir_out
            )

        for i_in, (mul_in, ir_in) in enumerate(self.irreps_in):
            for i_out, (mul_out, ir_out) in enumerate(self.irreps_out):
                if ir_in != ir_out:
                    continue

                total = total_input_mul[i_out]
                path_weight = 1.0 / math.sqrt(total) if total > 0 else 1.0

                # Initialise weight ~ N(0, 1)  (path_weight handles scaling)
                w = mx.random.normal(shape=(mul_in, mul_out))
                instructions.append(
                    _Instruction(i_in, i_out, ir_in.dim, mul_in, mul_out, path_weight)
                )
                weights.append(w)

        self.instructions = instructions
        self.weights = weights

        # Detect scalar-only fast path: all irreps are l=0 (dim=1) and
        # there is exactly one instruction mapping one input slot to one
        # output slot.  In that case the forward pass collapses to a
        # single matmul: output = path_weight * (x @ weight).
        self._scalar_fast_path = False
        self._scalar_weight: mx.array | None = None
        if (
            len(instructions) == 1
            and instructions[0].ir_dim == 1
            and len(self.irreps_in) == 1
            and len(self.irreps_out) == 1
        ):
            self._scalar_fast_path = True
            pw = instructions[0].path_weight
            self._scalar_path_weight = pw

        # Detect multi-irrep matmul fast path: all instructions have the
        # same (mul_in, mul_out) and each maps a unique i_in to a unique
        # i_out with no accumulation.  The einsum "...ui,uw->...wi" is
        # replaced by a single matmul over flattened m-components.
        #
        # Two sub-cases:
        # (a) "uniform 1-to-1": N instructions, each with its own weight
        #     of shape (mul_in, mul_out). All have the same mul_in, mul_out
        #     but different ir_dim. Each maps a unique i_in -> unique i_out.
        #     No accumulation needed. We process all blocks via matmul on
        #     (B * total_m, mul_in) @ (mul_in, mul_out) where total_m is the
        #     sum of ir_dim across instructions, BUT only when all share a
        #     single weight. When weights differ per instruction, we must
        #     use separate matmuls per instruction.
        #
        # (b) "accumulate": multiple instructions map to the same i_out.
        #     We group by i_out and sum contributions. Each group with the
        #     same i_out has the same (mul_in, mul_out, ir_dim, path_weight).
        self._multi_irrep_matmul = False
        if (
            not self._scalar_fast_path
            and len(instructions) > 1
        ):
            self._setup_multi_irrep_fast_path(instructions)

    def _setup_multi_irrep_fast_path(
        self, instructions: list[_Instruction]
    ) -> None:
        """Detect and set up the multi-irrep matmul fast path.

        For instructions that share the same (mul_in, mul_out), replace
        the per-instruction einsum with a single matmul by treating all
        m-components as a batch dimension.
        """
        mul_in0 = instructions[0].mul_in
        mul_out0 = instructions[0].mul_out

        # Check all instructions share the same mul_in, mul_out
        if not all(
            inst.mul_in == mul_in0 and inst.mul_out == mul_out0
            for inst in instructions
        ):
            return

        # Check 1-to-1 mapping: each i_in appears once, each i_out appears once
        i_in_list = [inst.i_in for inst in instructions]
        i_out_list = [inst.i_out for inst in instructions]
        i_in_unique = len(i_in_list) == len(set(i_in_list))
        i_out_unique = len(i_out_list) == len(set(i_out_list))

        if i_in_unique and i_out_unique:
            # Case (a): 1-to-1, separate matmul per instruction
            # but each matmul is just (B*ir_dim, mul_in) @ (mul_in, mul_out)
            # which is more efficient than einsum
            self._multi_irrep_matmul = True
            self._mi_mul_in = mul_in0
            self._mi_mul_out = mul_out0
            # Store slices and ir_dims for fast iteration
            slices_in = self.irreps_in.slices
            slices_out = self.irreps_out.slices
            self._mi_in_slices = [slices_in[inst.i_in] for inst in instructions]
            self._mi_out_slices = [slices_out[inst.i_out] for inst in instructions]
            self._mi_ir_dims = [inst.ir_dim for inst in instructions]
            self._mi_path_weights = [inst.path_weight for inst in instructions]
            # Check if all path weights are the same (common case)
            self._mi_uniform_pw = all(
                abs(pw - self._mi_path_weights[0]) < 1e-10
                for pw in self._mi_path_weights
            )
            self._mi_pw0 = self._mi_path_weights[0]
            return

        # Case (b): accumulation needed — group by i_out
        from collections import defaultdict
        groups: dict[int, list[int]] = defaultdict(list)
        for idx, inst in enumerate(instructions):
            groups[inst.i_out].append(idx)

        # All instructions still must share (mul_in, mul_out) and path_weight
        # within each group must be the same
        all_same_pw_in_group = True
        for i_out, idxs in groups.items():
            pws = [instructions[i].path_weight for i in idxs]
            if not all(abs(pw - pws[0]) < 1e-10 for pw in pws):
                all_same_pw_in_group = False
                break

        if not all_same_pw_in_group:
            return

        self._multi_irrep_matmul = True
        self._mi_mul_in = mul_in0
        self._mi_mul_out = mul_out0

        slices_in = self.irreps_in.slices
        slices_out = self.irreps_out.slices
        self._mi_in_slices = [slices_in[inst.i_in] for inst in instructions]
        self._mi_out_slices = [slices_out[inst.i_out] for inst in instructions]
        self._mi_ir_dims = [inst.ir_dim for inst in instructions]
        self._mi_path_weights = [inst.path_weight for inst in instructions]
        self._mi_uniform_pw = all(
            abs(pw - self._mi_path_weights[0]) < 1e-10
            for pw in self._mi_path_weights
        )
        self._mi_pw0 = self._mi_path_weights[0]

        # Store grouping info for accumulation
        self._mi_1to1 = i_in_unique and i_out_unique
        if not (i_in_unique and i_out_unique):
            # Store group info: for each i_out, which instruction indices
            self._mi_groups = dict(groups)
            self._mi_i_out_order = list(self.irreps_out.slices)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: tensor of shape ``(..., irreps_in.dim)``

        Returns:
            Tensor of shape ``(..., irreps_out.dim)``
        """
        # Fast path: scalar-only with single instruction -> one matmul
        if self._scalar_fast_path:
            return (self._scalar_path_weight * x) @ self.weights[0]

        # Fast path: multi-irrep with matmul instead of einsum
        if self._multi_irrep_matmul:
            return self._multi_irrep_forward(x)

        return self._loop_forward(x)

    def _multi_irrep_forward(self, x: mx.array) -> mx.array:
        """Multi-irrep fast path using matmul instead of einsum.

        For each instruction, replaces:
            einsum("...ui,uw->...wi", x_block, w)
        with:
            (x_flat @ w).reshape(...)
        where x_flat has shape (B * ir_dim, mul_in).

        This is equivalent but avoids the einsum overhead and enables
        better MLX kernel fusion.
        """
        batch_shape = x.shape[:-1]
        batch_size = 1
        for s in batch_shape:
            batch_size *= s
        mul_in = self._mi_mul_in
        mul_out = self._mi_mul_out

        out_parts = []
        for idx in range(len(self.instructions)):
            w = self.weights[idx]
            sl = self._mi_in_slices[idx]
            ir_dim = self._mi_ir_dims[idx]

            # Extract block and reshape: (..., mul_in * ir_dim) -> (B * ir_dim, mul_in)
            x_block = x[..., sl]
            x_flat = x_block.reshape(batch_size, mul_in, ir_dim)
            # Transpose to (batch_size, ir_dim, mul_in) then flatten first two dims
            x_flat = x_flat.transpose(0, 2, 1).reshape(batch_size * ir_dim, mul_in)

            # Matmul: (B * ir_dim, mul_in) @ (mul_in, mul_out) -> (B * ir_dim, mul_out)
            out_flat = x_flat @ w

            # Reshape back: (B, ir_dim, mul_out) -> (B, mul_out, ir_dim) -> (B, mul_out * ir_dim)
            out_block = out_flat.reshape(batch_size, ir_dim, mul_out)
            out_block = out_block.transpose(0, 2, 1).reshape(*batch_shape, mul_out * ir_dim)

            out_parts.append(out_block)

        # Apply path weights and handle accumulation
        if hasattr(self, '_mi_1to1') and not self._mi_1to1:
            return self._accumulate_groups(out_parts, batch_shape)

        # 1-to-1 case: apply path_weight and concatenate in output order
        if self._mi_uniform_pw:
            result = mx.concatenate(out_parts, axis=-1)
            return self._mi_pw0 * result
        else:
            scaled = []
            for idx, part in enumerate(out_parts):
                scaled.append(self._mi_path_weights[idx] * part)
            return mx.concatenate(scaled, axis=-1)

    def _accumulate_groups(
        self, out_parts: list[mx.array], batch_shape: tuple
    ) -> mx.array:
        """Handle accumulation when multiple inputs map to the same output."""
        slices_out = self.irreps_out.slices
        result_parts: list[mx.array] = [
            mx.zeros((*batch_shape, mulir.mul * mulir.ir.dim))
            for mulir in self.irreps_out
        ]

        for idx, inst in enumerate(self.instructions):
            pw = self._mi_path_weights[idx]
            result_parts[inst.i_out] = result_parts[inst.i_out] + pw * out_parts[idx]

        return mx.concatenate(result_parts, axis=-1)

    def _loop_forward(self, x: mx.array) -> mx.array:
        """Original loop-based forward pass (fallback)."""
        batch_shape = x.shape[:-1]
        slices_in = self.irreps_in.slices

        out_parts: list[mx.array] = [
            mx.zeros((*batch_shape, mulir.mul * mulir.ir.dim))
            for mulir in self.irreps_out
        ]

        for idx, inst in enumerate(self.instructions):
            w = self.weights[idx]

            x_block = x[..., slices_in[inst.i_in]]
            x_block = x_block.reshape(*batch_shape, inst.mul_in, inst.ir_dim)

            out_block = mx.einsum("...ui,uw->...wi", x_block, w)
            out_block = inst.path_weight * out_block
            out_block = out_block.reshape(*batch_shape, inst.mul_out * inst.ir_dim)

            out_parts[inst.i_out] = out_parts[inst.i_out] + out_block

        if len(out_parts) == 0:
            return mx.zeros((*batch_shape, 0))

        return mx.concatenate(out_parts, axis=-1)
