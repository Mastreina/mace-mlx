"""Gated activation preserving equivariance.

Splits input into: [scalars_to_activate, gate_scalars, non_scalar_irreps]
- scalars_to_activate: apply activation function directly
- gate_scalars: one scalar per non-scalar irrep block, apply sigmoid/tanh
- non_scalar_irreps: multiply by corresponding gate scalar
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from mace_mlx.irreps import Irreps


class Gate(nn.Module):
    """Gated activation preserving equivariance.

    Input layout:  [scalars_to_activate | gate_scalars | non_scalar_irreps]
    Output layout: [activated_scalars | gated_features]

    The total number of gate scalars must equal the total number of
    non-scalar irrep copies (sum of multiplicities in irreps_gated).
    """

    def __init__(
        self,
        irreps_scalars: Irreps | str,
        act_scalars: list[Callable],
        irreps_gates: Irreps | str,
        act_gates: list[Callable],
        irreps_gated: Irreps | str,
    ):
        super().__init__()
        self.irreps_scalars = Irreps(irreps_scalars)
        self.irreps_gates = Irreps(irreps_gates)
        self.irreps_gated = Irreps(irreps_gated)

        # Validate: all scalars must be l=0
        for mul, ir in self.irreps_scalars:
            assert ir.l == 0, f"irreps_scalars must be l=0, got {ir}"
        for mul, ir in self.irreps_gates:
            assert ir.l == 0, f"irreps_gates must be l=0, got {ir}"

        # Validate: number of gate scalars == number of gated irreps
        num_gate_scalars = sum(mul for mul, _ in self.irreps_gates)
        num_gated_irreps = sum(mul for mul, _ in self.irreps_gated)
        assert num_gate_scalars == num_gated_irreps, (
            f"Number of gate scalars ({num_gate_scalars}) must match "
            f"number of gated irreps ({num_gated_irreps})"
        )

        # Store activations
        self.act_scalars = act_scalars
        self.act_gates = act_gates

        # Input irreps = scalars + gates + gated
        self.irreps_in = self.irreps_scalars + self.irreps_gates + self.irreps_gated

        # Output irreps = scalars + gated (gates are consumed)
        self.irreps_out = self.irreps_scalars + self.irreps_gated

        # Precompute dimensions
        self._dim_scalars = self.irreps_scalars.dim if len(self.irreps_scalars) > 0 else 0
        self._dim_gates = self.irreps_gates.dim if len(self.irreps_gates) > 0 else 0

    def __call__(self, features: mx.array) -> mx.array:
        # 1. Split features into scalar, gate, and gated parts
        offset = 0
        scalar_parts = features[..., offset : offset + self._dim_scalars]
        offset += self._dim_scalars
        gate_parts = features[..., offset : offset + self._dim_gates]
        offset += self._dim_gates
        gated_parts = features[..., offset:]

        # 2. Apply activation to scalars (block by block)
        activated_scalars = self._apply_block_activations(
            scalar_parts, self.irreps_scalars, self.act_scalars
        )

        # 3. Apply gate activation to gate scalars
        activated_gates = self._apply_block_activations(
            gate_parts, self.irreps_gates, self.act_gates
        )

        # 4. Multiply each gated block by its gate scalar
        if self._dim_gates > 0:
            gated_out = self._apply_gates(activated_gates, gated_parts)
        else:
            gated_out = None

        # 5. Concatenate [activated_scalars, gated_features]
        parts = []
        if self._dim_scalars > 0:
            parts.append(activated_scalars)
        if gated_out is not None:
            parts.append(gated_out)

        if not parts:
            return mx.zeros((*features.shape[:-1], 0))

        return mx.concatenate(parts, axis=-1)

    @staticmethod
    def _apply_block_activations(
        x: mx.array, irreps: Irreps, activations: list[Callable]
    ) -> mx.array:
        """Apply per-block activations to scalar features.

        When all blocks use the same activation, applies it once to the
        entire tensor to avoid unnecessary slicing and concatenation.
        """
        if irreps.dim == 0:
            return x
        # When all blocks share the same activation, apply it once
        first_act = activations[0]
        all_same = True
        for idx in range(len(irreps)):
            act = activations[idx] if idx < len(activations) else activations[-1]
            if act is not first_act:
                all_same = False
                break
        if all_same:
            return first_act(x)
        parts = []
        offset = 0
        for idx, (mul, ir) in enumerate(irreps):
            d = mul * ir.dim
            block = x[..., offset : offset + d]
            act = activations[idx] if idx < len(activations) else activations[-1]
            parts.append(act(block))
            offset += d
        return mx.concatenate(parts, axis=-1)

    def _apply_gates(self, gates: mx.array, gated: mx.array) -> mx.array:
        """Multiply each gated irrep block by its corresponding gate scalar.

        Vectorized: processes all multiplicities of each irrep type in one
        broadcast multiply instead of looping per-copy. This reduces the
        number of autograd nodes from sum(mul) to len(irreps_gated).
        """
        parts = []
        gate_offset = 0
        gated_offset = 0
        for mul, ir in self.irreps_gated:
            ir_dim = ir.dim
            total_dim = mul * ir_dim
            # All gate scalars for this irrep type: (..., mul)
            g_block = gates[..., gate_offset : gate_offset + mul]
            # All gated features: (..., mul * ir_dim) -> (..., mul, ir_dim)
            gated_block = gated[..., gated_offset : gated_offset + total_dim]
            gated_reshaped = gated_block.reshape(
                *gated.shape[:-1], mul, ir_dim
            )
            # Broadcast multiply: (..., mul, 1) * (..., mul, ir_dim)
            result = g_block[..., :, None] * gated_reshaped
            # Flatten: (..., mul * ir_dim)
            parts.append(result.reshape(*gated.shape[:-1], total_dim))
            gate_offset += mul
            gated_offset += total_dim
        return mx.concatenate(parts, axis=-1)
