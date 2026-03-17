"""MACE building blocks for equivariant message passing.

Implements node embedding, atomic energies, interaction blocks,
product basis, and readout blocks.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from mace_mlx.gate import Gate
from mace_mlx.irreps import Irrep, Irreps, MulIr
from mace_mlx.linear import EquivariantLinear
from mace_mlx.radial import make_radial_mlp, make_radial_mlp_with_layernorm
from mace_mlx.tensor_product import FullyConnectedTensorProduct, TensorProduct
from mace_mlx.kernels import gather_tp_scatter
from mace_mlx.utils import SILU_NORM_FACTOR, scatter_sum, tp_out_irreps_with_instructions


def _can_fuse_scalar_tp(tp: TensorProduct) -> bool:
    """Check if a TensorProduct can use the fused gather-TP-scatter kernel.

    The fused kernel applies when:
    - There is exactly one instruction
    - The instruction uses "uvu" mode with external weights
    - All irreps dimensions are 1 (scalar, l=0)
    - CG coefficient is a scalar (identity fast-path)
    - path_weight is 1.0

    These conditions hold for MACE's conv_tp when hidden_irreps = Nx0e
    and irreps_sh starts with "0e" (the 0e component couples 0e x 0e -> 0e).
    """
    if len(tp._instructions) != 1:
        return False
    inst = tp._instructions[0]
    if inst.connection_mode != "uvu":
        return False
    if inst.ir1_dim != 1 or inst.ir2_dim != 1 or inst.ir_out_dim != 1:
        return False
    if tp._cg_scalars[0] is None or abs(tp._cg_scalars[0] - 1.0) > 1e-6:
        return False
    if abs(inst.path_weight - 1.0) > 1e-6:
        return False
    if inst.mul2 != 1:
        return False
    return True


class LinearNodeEmbeddingBlock(nn.Module):
    """Embed one-hot atomic attributes into irreps features.

    Uses EquivariantLinear to map from one-hot species (scalar irreps)
    to the desired output irreps.
    """

    def __init__(self, irreps_in: Irreps | str, irreps_out: Irreps | str):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.linear = EquivariantLinear(self.irreps_in, self.irreps_out)

    def __call__(self, node_attrs: mx.array) -> mx.array:
        """
        Args:
            node_attrs: (num_atoms, irreps_in.dim) one-hot or scalar features
        Returns:
            (num_atoms, irreps_out.dim)
        """
        return self.linear(node_attrs)


class AtomicEnergiesBlock(nn.Module):
    """Per-element atomic energies (baseline)."""

    def __init__(self, atomic_energies: mx.array):
        super().__init__()
        self.atomic_energies = atomic_energies
        self.freeze(keys=["atomic_energies"])

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (num_atoms, num_elements) one-hot encoding
        Returns:
            (num_atoms,) or (num_atoms, num_heads) baseline energy per atom
        """
        return x @ mx.atleast_2d(self.atomic_energies).T


class RealAgnosticInteractionBlock(nn.Module):
    """MACE interaction block with equivariant message passing.

    Pipeline:
    1. linear_up: EquivariantLinear (expand node features)
    2. conv_tp: TensorProduct (combine with edge spherical harmonics, external weights)
    3. conv_tp_weights: RadialMLP (edge distances -> TP weights)
    4. scatter_sum: aggregate messages to nodes
    5. linear: EquivariantLinear (contract)
    6. skip_tp: FullyConnectedTensorProduct (skip connection with node attributes)
    """

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str,
        irreps_sh: Irreps | str,
        num_radial: int,
        radial_MLP: list[int],
        avg_num_neighbors: float,
        num_species: int,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        irreps_sh = Irreps(irreps_sh)
        self.avg_num_neighbors = avg_num_neighbors

        # Node attributes irreps (one-hot encoding = num_species x 0e)
        node_attrs_irreps = Irreps(f"{num_species}x0e")

        # 1. linear_up: expand node features
        self.linear_up = EquivariantLinear(self.irreps_in, self.irreps_in)

        # 2. conv_tp: tensor product with edge spherical harmonics
        #    Pre-rotate CG coefficients to absorb the SH basis rotation,
        #    so SH can be passed in standard m-ordering without runtime rotation.
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.irreps_in, irreps_sh, self.irreps_out
        )
        self.conv_tp = TensorProduct(
            self.irreps_in,
            irreps_sh,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,

        )
        self._irreps_mid = irreps_mid

        # 3. conv_tp_weights: radial MLP for TP weights
        self.conv_tp_weights = make_radial_mlp(
            [num_radial] + radial_MLP + [self.conv_tp.weight_numel]
        )

        # 4. linear: contract after aggregation
        self.linear = EquivariantLinear(irreps_mid, self.irreps_out)

        # 5. skip_tp: skip connection with node attributes
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out, node_attrs_irreps, self.irreps_out
        )

        # Detect scalar-only TP for fused kernel fast path
        self._can_fuse = _can_fuse_scalar_tp(self.conv_tp)
        self._use_fused_kernel = False

    def set_fused_kernel(self, enabled: bool = True) -> None:
        """Enable/disable the fused Metal gather-TP-scatter kernel.

        Only effective for scalar-only TP (hidden_irreps = Nx0e).
        The Metal kernel has no autograd support, so this should only
        be enabled during inference.
        """
        self._use_fused_kernel = enabled and self._can_fuse

    def __call__(
        self,
        node_feats: mx.array,
        node_attrs: mx.array,
        edge_attrs: mx.array,
        edge_feats: mx.array,
        edge_index: mx.array,
        cutoff: mx.array | None = None,
    ) -> tuple[mx.array, None]:
        """
        Args:
            node_feats: (num_atoms, irreps_in.dim)
            node_attrs: (num_atoms, num_species) one-hot
            edge_attrs: (num_edges, irreps_sh.dim) spherical harmonics
            edge_feats: (num_edges, num_radial) radial features
            edge_index: (2, num_edges) [sender, receiver]
            cutoff: (num_edges, 1) cutoff values or None

        Returns:
            (message, None) where message is (num_atoms, irreps_out.dim)
        """
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        # 1. Expand node features
        node_feats_up = self.linear_up(node_feats)

        # 2. Compute TP weights from radial features
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff

        # 3-4. Message computation + aggregation
        if self._use_fused_kernel:
            # Fused Metal kernel: gather + multiply + scatter in one pass
            message = gather_tp_scatter(
                node_feats_up, tp_weights, edge_attrs[:, 0],
                sender, receiver, num_nodes,
                use_metal=False,
            )
        else:
            # Standard path: gather -> TP -> scatter
            mji = self.conv_tp(node_feats_up[sender], edge_attrs, tp_weights)
            message = scatter_sum(mji, receiver, num_nodes)

        # 5. Contract
        message = self.linear(message) / self.avg_num_neighbors

        # 6. Skip connection
        message = self.skip_tp(message, node_attrs)

        return message, None


class RealAgnosticResidualInteractionBlock(nn.Module):
    """MACE residual interaction block.

    Same as RealAgnosticInteractionBlock but returns (message, skip_connection)
    for residual architecture. The skip connection is computed before the main
    message passing, using the original node features.
    """

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str,
        irreps_sh: Irreps | str,
        num_radial: int,
        radial_MLP: list[int],
        avg_num_neighbors: float,
        num_species: int,
        hidden_irreps: Irreps | str | None = None,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        irreps_sh = Irreps(irreps_sh)
        self.avg_num_neighbors = avg_num_neighbors

        if hidden_irreps is None:
            hidden_irreps = self.irreps_out
        hidden_irreps = Irreps(hidden_irreps)

        node_attrs_irreps = Irreps(f"{num_species}x0e")

        # Skip connection TP (before message passing)
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_in, node_attrs_irreps, hidden_irreps
        )

        # 1. linear_up
        self.linear_up = EquivariantLinear(self.irreps_in, self.irreps_in)

        # 2. conv_tp — CG coefficients pre-rotated to absorb SH basis rotation
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.irreps_in, irreps_sh, self.irreps_out
        )
        self.conv_tp = TensorProduct(
            self.irreps_in,
            irreps_sh,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,

        )
        self._irreps_mid = irreps_mid

        # 3. conv_tp_weights
        self.conv_tp_weights = make_radial_mlp(
            [num_radial] + radial_MLP + [self.conv_tp.weight_numel]
        )

        # 4. linear
        self.linear = EquivariantLinear(irreps_mid, self.irreps_out)

        # Detect scalar-only TP for fused kernel fast path
        self._can_fuse = _can_fuse_scalar_tp(self.conv_tp)
        self._use_fused_kernel = False

    def set_fused_kernel(self, enabled: bool = True) -> None:
        """Enable/disable the fused Metal gather-TP-scatter kernel.

        Only effective for scalar-only TP (hidden_irreps = Nx0e).
        The Metal kernel has no autograd support, so this should only
        be enabled during inference.
        """
        self._use_fused_kernel = enabled and self._can_fuse

    def __call__(
        self,
        node_feats: mx.array,
        node_attrs: mx.array,
        edge_attrs: mx.array,
        edge_feats: mx.array,
        edge_index: mx.array,
        cutoff: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Returns:
            (message, skip_connection)
            message: (num_atoms, irreps_out.dim)
            skip_connection: (num_atoms, hidden_irreps.dim)
        """
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        # Skip connection BEFORE message passing
        sc = self.skip_tp(node_feats, node_attrs)

        # 1. Expand
        node_feats_up = self.linear_up(node_feats)

        # 2. TP weights
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff

        # 3-4. Message computation + aggregation
        if self._use_fused_kernel:
            # Fused Metal kernel: gather + multiply + scatter in one pass
            message = gather_tp_scatter(
                node_feats_up, tp_weights, edge_attrs[:, 0],
                sender, receiver, num_nodes,
                use_metal=False,
            )
        else:
            # Standard path: gather -> TP -> scatter
            mji = self.conv_tp(node_feats_up[sender], edge_attrs, tp_weights)
            message = scatter_sum(mji, receiver, num_nodes)

        # 5. Contract
        message = self.linear(message) / self.avg_num_neighbors

        return message, sc


class RealAgnosticDensityInteractionBlock(nn.Module):
    """MACE density interaction block (non-residual).

    Same as RealAgnosticInteractionBlock but uses per-node density
    normalization instead of dividing by avg_num_neighbors.

    Density normalization:
        edge_density = tanh(density_fn(edge_feats) ** 2)  per-edge (N_edges, 1)
        density = scatter_sum(edge_density, receiver)     per-node (N_nodes, 1)
        message = linear(message) / (density + 1)
    """

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str,
        irreps_sh: Irreps | str,
        num_radial: int,
        radial_MLP: list[int],
        avg_num_neighbors: float,
        num_species: int,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        irreps_sh = Irreps(irreps_sh)
        self.avg_num_neighbors = avg_num_neighbors

        node_attrs_irreps = Irreps(f"{num_species}x0e")

        # 1. linear_up
        self.linear_up = EquivariantLinear(self.irreps_in, self.irreps_in)

        # 2. conv_tp
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.irreps_in, irreps_sh, self.irreps_out
        )
        self.conv_tp = TensorProduct(
            self.irreps_in,
            irreps_sh,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,

        )
        self._irreps_mid = irreps_mid

        # 3. conv_tp_weights
        self.conv_tp_weights = make_radial_mlp(
            [num_radial] + radial_MLP + [self.conv_tp.weight_numel]
        )

        # 4. density_fn: single linear layer (num_radial -> 1), no activation
        self.density_fn = nn.Linear(num_radial, 1, bias=False)

        # 5. linear
        self.linear = EquivariantLinear(irreps_mid, self.irreps_out)

        # 6. skip_tp
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out, node_attrs_irreps, self.irreps_out
        )

        # Fused kernel detection
        self._can_fuse = _can_fuse_scalar_tp(self.conv_tp)
        self._use_fused_kernel = False

    def set_fused_kernel(self, enabled: bool = True) -> None:
        """Enable/disable the fused Metal gather-TP-scatter kernel."""
        self._use_fused_kernel = enabled and self._can_fuse

    def __call__(
        self,
        node_feats: mx.array,
        node_attrs: mx.array,
        edge_attrs: mx.array,
        edge_feats: mx.array,
        edge_index: mx.array,
        cutoff: mx.array | None = None,
    ) -> tuple[mx.array, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        # 1. Expand
        node_feats_up = self.linear_up(node_feats)

        # 2. TP weights
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff

        # 3. Density normalization
        edge_density = mx.tanh(self.density_fn(edge_feats) ** 2)  # (N_edges, 1)
        if cutoff is not None:
            edge_density = edge_density * cutoff
        density = scatter_sum(edge_density, receiver, num_nodes)  # (N_nodes, 1)

        # 4. Message computation + aggregation
        if self._use_fused_kernel:
            message = gather_tp_scatter(
                node_feats_up, tp_weights, edge_attrs[:, 0],
                sender, receiver, num_nodes,
                use_metal=False,
            )
        else:
            mji = self.conv_tp(node_feats_up[sender], edge_attrs, tp_weights)
            message = scatter_sum(mji, receiver, num_nodes)

        # 5. Contract with density normalization
        message = self.linear(message) / (density + 1)

        # 6. Skip connection
        message = self.skip_tp(message, node_attrs)

        return message, None


class RealAgnosticDensityResidualInteractionBlock(nn.Module):
    """MACE density residual interaction block.

    Same as RealAgnosticResidualInteractionBlock but uses per-node density
    normalization instead of dividing by avg_num_neighbors.
    Returns (message, skip_connection) for residual architecture.
    """

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str,
        irreps_sh: Irreps | str,
        num_radial: int,
        radial_MLP: list[int],
        avg_num_neighbors: float,
        num_species: int,
        hidden_irreps: Irreps | str | None = None,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        irreps_sh = Irreps(irreps_sh)
        self.avg_num_neighbors = avg_num_neighbors

        if hidden_irreps is None:
            hidden_irreps = self.irreps_out
        hidden_irreps = Irreps(hidden_irreps)

        node_attrs_irreps = Irreps(f"{num_species}x0e")

        # Skip connection TP
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_in, node_attrs_irreps, hidden_irreps
        )

        # 1. linear_up
        self.linear_up = EquivariantLinear(self.irreps_in, self.irreps_in)

        # 2. conv_tp
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.irreps_in, irreps_sh, self.irreps_out
        )
        self.conv_tp = TensorProduct(
            self.irreps_in,
            irreps_sh,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,

        )
        self._irreps_mid = irreps_mid

        # 3. conv_tp_weights
        self.conv_tp_weights = make_radial_mlp(
            [num_radial] + radial_MLP + [self.conv_tp.weight_numel]
        )

        # 4. density_fn: single linear layer (num_radial -> 1), no activation
        self.density_fn = nn.Linear(num_radial, 1, bias=False)

        # 5. linear
        self.linear = EquivariantLinear(irreps_mid, self.irreps_out)

        # Fused kernel detection
        self._can_fuse = _can_fuse_scalar_tp(self.conv_tp)
        self._use_fused_kernel = False

    def set_fused_kernel(self, enabled: bool = True) -> None:
        """Enable/disable the fused Metal gather-TP-scatter kernel."""
        self._use_fused_kernel = enabled and self._can_fuse

    def __call__(
        self,
        node_feats: mx.array,
        node_attrs: mx.array,
        edge_attrs: mx.array,
        edge_feats: mx.array,
        edge_index: mx.array,
        cutoff: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        # Skip connection BEFORE message passing
        sc = self.skip_tp(node_feats, node_attrs)

        # 1. Expand
        node_feats_up = self.linear_up(node_feats)

        # 2. TP weights
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff

        # 3. Density normalization
        edge_density = mx.tanh(self.density_fn(edge_feats) ** 2)  # (N_edges, 1)
        if cutoff is not None:
            edge_density = edge_density * cutoff
        density = scatter_sum(edge_density, receiver, num_nodes)  # (N_nodes, 1)

        # 4. Message computation + aggregation
        if self._use_fused_kernel:
            message = gather_tp_scatter(
                node_feats_up, tp_weights, edge_attrs[:, 0],
                sender, receiver, num_nodes,
                use_metal=False,
            )
        else:
            mji = self.conv_tp(node_feats_up[sender], edge_attrs, tp_weights)
            message = scatter_sum(mji, receiver, num_nodes)

        # 5. Contract with density normalization
        message = self.linear(message) / (density + 1)

        return message, sc


class EquivariantProductBasisBlock(nn.Module):
    """Symmetric contraction + optional skip + linear.

    Uses SymmetricContraction if available, otherwise a placeholder.
    When num_elements=1, the product is element-agnostic (used by mh-1 family).
    """

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str,
        correlation: int,
        num_elements: int,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self._agnostic = (num_elements == 1)

        from mace_mlx.symmetric_contraction import SymmetricContraction

        self.symmetric_contractions = SymmetricContraction(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            correlation=correlation,
            num_elements=num_elements,
        )

        self.linear = EquivariantLinear(self.irreps_out, self.irreps_out)

    def __call__(
        self,
        node_feats: mx.array,
        node_attrs: mx.array,
        sc: mx.array | None = None,
    ) -> mx.array:
        # For agnostic product, use all-ones node_attrs (element-independent)
        if self._agnostic:
            node_attrs = mx.ones((node_feats.shape[0], 1), dtype=node_feats.dtype)
        node_feats = self.symmetric_contractions(node_feats, node_attrs)

        if sc is not None:
            return self.linear(node_feats) + sc

        return self.linear(node_feats)


class LinearReadoutBlock(nn.Module):
    """Linear readout to scalar output."""

    def __init__(self, irreps_in: Irreps | str, irreps_out: Irreps | str = "0e"):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.linear = EquivariantLinear(self.irreps_in, self.irreps_out)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


class NonLinearReadoutBlock(nn.Module):
    """Non-linear readout: Linear -> Gate -> [mask_head] -> Linear.

    For multi-head models (num_heads > 1), mask_head zeros out all
    MLP features except those belonging to the active head before
    the final linear layer, matching PyTorch MACE behavior.
    """

    def __init__(
        self,
        irreps_in: Irreps | str,
        MLP_irreps: Irreps | str,
        gate_fn: Callable = nn.silu,
        irreps_out: Irreps | str = "0e",
        num_heads: int = 1,
    ):
        super().__init__()
        irreps_in = Irreps(irreps_in)
        MLP_irreps = Irreps(MLP_irreps)
        irreps_out = Irreps(irreps_out)
        self._num_heads = num_heads
        self._features_per_head = MLP_irreps.dim // max(num_heads, 1)

        # For NonLinearReadout in MACE, MLP_irreps is typically scalars only
        # e.g. "16x0e". We apply a scalar activation (gate).
        # Build the Gate: all scalars, no gated part
        irreps_scalars = MLP_irreps
        self.linear_1 = EquivariantLinear(irreps_in, MLP_irreps)
        self.non_linearity = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate_fn] * len(irreps_scalars),
            irreps_gates=Irreps(""),
            act_gates=[],
            irreps_gated=Irreps(""),
        )
        self.linear_2 = EquivariantLinear(MLP_irreps, irreps_out)

        # Precompute head masks for multi-head models
        if self._num_heads > 1:
            fph = self._features_per_head
            total_dim = MLP_irreps.dim
            self._head_masks = []
            for h in range(self._num_heads):
                mask = mx.zeros(total_dim)
                start = h * fph
                end = start + fph
                mask = mask.at[start:end].add(1.0)
                self._head_masks.append(mask)

    def __call__(self, x: mx.array, head_idx: int = 0) -> mx.array:
        x = self.linear_1(x)
        x = self.non_linearity(x)
        # Mask: zero out non-selected head's features before linear_2
        if self._num_heads > 1:
            x = x * self._head_masks[head_idx]
        return self.linear_2(x)


class RealAgnosticResidualNonLinearInteractionBlock(nn.Module):
    """MACE residual interaction block with non-linear gating and density normalization.

    Used by mh-1 model family. Key differences from RealAgnosticResidualInteractionBlock:
    - source_embedding / target_embedding: linear projections of one-hot species,
      concatenated with radial features to form augmented edge features
    - conv_tp_weights uses RadialMLP (LayerNorm + bias) instead of FullyConnectedNet
    - density_fn: RadialMLP that computes per-edge density for normalization
    - alpha/beta: learnable parameters for density normalization
    - Equivariant non-linearity (silu for scalars, sigmoid gating for L>0)
    - linear_res: residual linear skip from linear_up output
    - linear_1/linear_2: additional linear layers around the non-linearity
    - skip_tp: a simple Linear (not FullyConnectedTensorProduct)
    - edge_irreps: separate irreps for linear_up output (may differ from irreps_in)
    """

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str,
        irreps_sh: Irreps | str,
        num_radial: int,
        radial_MLP: list[int],
        avg_num_neighbors: float,
        num_species: int,
        hidden_irreps: Irreps | str | None = None,
        edge_irreps: Irreps | str | None = None,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        irreps_sh = Irreps(irreps_sh)
        self.avg_num_neighbors = avg_num_neighbors

        if hidden_irreps is None:
            hidden_irreps = self.irreps_out
        hidden_irreps = Irreps(hidden_irreps)

        if edge_irreps is None:
            edge_irreps = self.irreps_in
        edge_irreps = Irreps(edge_irreps)

        node_attrs_irreps = Irreps(f"{num_species}x0e")

        # Scalar feature dimension for source/target embedding
        node_scalar_dim = self.irreps_in.count(Irrep("0e"))
        node_scalar_irreps = Irreps(f"{node_scalar_dim}x0e")

        # source/target embedding: Linear projections of one-hot -> scalar features
        self.source_embedding = EquivariantLinear(node_attrs_irreps, node_scalar_irreps)
        self.target_embedding = EquivariantLinear(node_attrs_irreps, node_scalar_irreps)

        # Skip connection: simple Linear (not FCTP like standard blocks)
        self.skip_tp = EquivariantLinear(self.irreps_in, hidden_irreps)

        # linear_up: project to edge_irreps
        self.linear_up = EquivariantLinear(self.irreps_in, edge_irreps)

        # conv_tp: tensor product with SH
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            edge_irreps, irreps_sh, self.irreps_out
        )
        self.conv_tp = TensorProduct(
            edge_irreps,
            irreps_sh,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,

        )
        self._irreps_mid = irreps_mid

        # conv_tp_weights: RadialMLP (with LayerNorm + bias)
        # Input dim = num_radial + 2 * node_scalar_dim (augmented edge features)
        augmented_input_dim = num_radial + 2 * node_scalar_dim
        self.conv_tp_weights = make_radial_mlp_with_layernorm(
            [augmented_input_dim] + radial_MLP + [self.conv_tp.weight_numel]
        )

        # density_fn: RadialMLP (with LayerNorm + bias)
        self.density_fn = make_radial_mlp_with_layernorm(
            [augmented_input_dim, 64, 1]
        )

        # alpha/beta: learnable density normalization params
        self.alpha = mx.array(20.0)
        self.beta = mx.array(0.0)

        # Equivariant non-linearity setup
        # Split target irreps into scalars (silu) and non-scalars (sigmoid gating)
        irreps_scalars = Irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]
        )
        irreps_gated = Irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]
        )
        irreps_gates = Irreps(
            [(mul, Irrep("0e")) for mul, _ in irreps_gated]
        )

        # Input to Gate: scalars + gate_scalars + gated
        # Gate produces: activated_scalars + gated_output
        # e3nn Gate wraps activations with normalize2mom (1/sqrt(E[f(x)^2])).
        # Bake these factors into the activation functions.
        _SIGMOID_NORM = 1.8467055569904827

        def norm_silu(x):
            return nn.silu(x) * SILU_NORM_FACTOR

        def norm_sigmoid(x):
            return mx.sigmoid(x) * _SIGMOID_NORM

        self.equivariant_nonlin = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[norm_silu] * len(irreps_scalars),
            irreps_gates=irreps_gates,
            act_gates=[norm_sigmoid] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        # irreps_nonlin = input irreps to the gate (scalars + gates + gated)
        self._irreps_nonlin = self.equivariant_nonlin.irreps_in

        # linear_res: residual from edge_irreps to irreps_nonlin
        self.linear_res = EquivariantLinear(edge_irreps, self._irreps_nonlin)

        # linear_1: contract message from irreps_mid to irreps_nonlin
        self.linear_1 = EquivariantLinear(irreps_mid, self._irreps_nonlin)

        # linear_2: post-nonlinearity projection
        self.linear_2 = EquivariantLinear(self.irreps_out, self.irreps_out)


    def __call__(
        self,
        node_feats: mx.array,
        node_attrs: mx.array,
        edge_attrs: mx.array,
        edge_feats: mx.array,
        edge_index: mx.array,
        cutoff: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Returns:
            (message, skip_connection)
        """
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        # Skip connection (simple linear, no TP with node_attrs)
        sc = self.skip_tp(node_feats)

        # linear_up
        node_feats_up = self.linear_up(node_feats)

        # Residual from linear_up output
        node_feats_res = self.linear_res(node_feats_up)

        # Source/target embeddings from one-hot species
        src_emb = self.source_embedding(node_attrs)
        tgt_emb = self.target_embedding(node_attrs)

        # Augment edge features: [radial_feats | src_emb[sender] | tgt_emb[receiver]]
        augmented_edge_feats = mx.concatenate(
            [edge_feats, src_emb[sender], tgt_emb[receiver]],
            axis=-1,
        )

        # TP weights from augmented edge features
        tp_weights = self.conv_tp_weights(augmented_edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff

        # Density normalization
        edge_density = mx.tanh(self.density_fn(augmented_edge_feats) ** 2)
        if cutoff is not None:
            edge_density = edge_density * cutoff
        density = scatter_sum(edge_density, receiver, num_nodes)  # (N_nodes, 1)

        # Message computation: TP + scatter
        mji = self.conv_tp(node_feats_up[sender], edge_attrs, tp_weights)
        message = scatter_sum(mji, receiver, num_nodes)

        # Contract with density normalization
        message = self.linear_1(message) / (density * self.beta + self.alpha)

        # Add residual
        message = message + node_feats_res

        # Equivariant non-linearity
        message = self.equivariant_nonlin(message)

        # Post-nonlinearity linear
        message = self.linear_2(message)

        return message, sc
