"""Complete MACE model implementation for MLX.

Implements MACE and ScaleShiftMACE models with the same architecture
as the PyTorch MACE, enabling direct weight transfer from trained models.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mace_mlx.blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    RealAgnosticResidualNonLinearInteractionBlock,
)
from mace_mlx.irreps import Irreps
from mace_mlx.radial import RadialEmbeddingBlock
from mace_mlx.spherical_harmonics import spherical_harmonics, _e3nn_rotation_matrix
from mace_mlx.utils import get_edge_vectors_and_lengths, scatter_sum


class MACE(nn.Module):
    """MACE equivariant message passing neural network.

    Architecture:
    1. Compute edge vectors/lengths from positions + edge_index + shifts
    2. e0 = atomic_energies(node_attrs) -> per-atom baseline
    3. node_feats = node_embedding(node_attrs)
    4. edge_attrs = spherical_harmonics(vectors) (rotated to e3nn basis)
    5. edge_feats = radial_embedding(lengths)
    6. For each interaction layer:
       a. node_feats, sc = interaction(node_feats, node_attrs, edge_attrs, edge_feats, edge_index)
       b. node_feats = product(node_feats, node_attrs, sc)
       c. node_energy = readout(node_feats)
    7. total_energy = e0 + sum(readout_energies)
    """

    def __init__(
        self,
        r_max: float,
        num_bessel: int = 10,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 3,
        num_interactions: int = 2,
        hidden_irreps: str = "128x0e",
        correlation: int | list[int] = 3,
        num_elements: int = 89,
        atomic_energies: mx.array | None = None,
        avg_num_neighbors: float = 1.0,
        radial_MLP: list[int] | None = None,
        gate: str = "silu",
        first_interaction_nonresidual: bool = False,
        distance_transform: dict | None = None,
        use_density_normalization: bool = False,
        heads: list[str] | None = None,
        interaction_cls: str = "RealAgnosticResidualInteractionBlock",
        edge_irreps: str | None = None,
        use_agnostic_product: bool = False,
        apply_cutoff: bool = True,
    ):
        super().__init__()
        self.r_max = r_max
        self.max_ell = max_ell
        self.num_interactions = num_interactions

        # Multi-head support
        self.heads = heads or ["Default"]
        self.num_heads = len(self.heads)
        # Active head index (set by calculator or user)
        self._head_idx = 0

        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        hidden_irreps = Irreps(hidden_irreps)

        # Correlation can be a single int or per-layer list
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions

        # Node attributes irreps (one-hot encoding)
        node_attrs_irreps = Irreps(f"{num_elements}x0e")

        # Spherical harmonics irreps
        sh_irreps = Irreps.spherical_harmonics(max_ell)

        # Interaction output irreps: same multiplicity as hidden but all l values
        # For hidden_irreps = "128x0e", interaction_irreps = "128x0e + 128x1o + 128x2e + 128x3o"
        num_features = hidden_irreps[0].mul  # multiplicity of first (scalar) block
        interaction_irreps = Irreps(
            [(num_features, ir) for _, ir in sh_irreps]
        )

        # Scalar-only hidden irreps (for the last layer and node_embedding)
        hidden_irreps_scalar = Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l == 0]
        )

        # Per-layer hidden_irreps:
        # All layers except the last use the full hidden_irreps (may include L>0).
        # The last layer uses only the scalar part.
        hidden_irreps_per_layer = []
        for i in range(num_interactions):
            if i < num_interactions - 1:
                hidden_irreps_per_layer.append(hidden_irreps)
            else:
                hidden_irreps_per_layer.append(hidden_irreps_scalar)

        # Readout output irreps: num_heads x 0e for multi-head, 0e for single-head
        readout_out_irreps = Irreps(f"{self.num_heads}x0e")
        # MLP hidden irreps for the NonLinearReadoutBlock: scale by num_heads
        mlp_hidden_mul = 16 * self.num_heads
        mlp_irreps = Irreps(f"{mlp_hidden_mul}x0e")

        # Parse edge_irreps for NonLinear interaction blocks
        edge_irreps_parsed = Irreps(edge_irreps) if edge_irreps is not None else None

        # 1. Atomic energies
        if atomic_energies is None:
            atomic_energies = mx.zeros(num_elements)
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        # 2. Node embedding (always outputs scalars only)
        self.node_embedding = LinearNodeEmbeddingBlock(
            node_attrs_irreps, hidden_irreps_scalar
        )

        # 3. Radial embedding (with optional distance transform for 0b family)
        self._has_distance_transform = distance_transform is not None
        self._apply_cutoff = apply_cutoff
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff,
        )
        # Placeholder for atomic numbers (set by load_model via z_table).
        # Required by AgnesiTransform to map one-hot -> covalent radii.
        self._atomic_numbers: mx.array | None = None

        # Precompute block-diagonal SH basis rotation matrix.
        # Our spherical_harmonics uses standard m-ordering, while e3nn
        # (and thus the trained MACE weights) uses a different ordering.
        # This block-diagonal matrix rotates each l-block from standard
        # to e3nn ordering. Applied once to the SH before the interaction
        # blocks, so the TP and all downstream operations use e3nn basis.
        import numpy as np
        total_sh_dim = sh_irreps.dim
        sh_rot_np = np.eye(total_sh_dim, dtype=np.float32)
        offset = 0
        for _, ir in sh_irreps:
            d = ir.dim
            if ir.l > 0:
                D = _e3nn_rotation_matrix(ir.l).astype(np.float32)
                # Store D.T: the forward pass computes edge_attrs @ _sh_rotation,
                # and the correct basis change is Y @ D.T (cf. _to_e3nn_basis).
                sh_rot_np[offset:offset+d, offset:offset+d] = D.T
            offset += d
        self._sh_rotation = mx.array(sh_rot_np)

        # Resolve interaction block class
        use_nonlinear_interaction = (
            interaction_cls == "RealAgnosticResidualNonLinearInteractionBlock"
        )

        # 4. Interaction blocks + product blocks + readout blocks
        self.interactions = []
        self.products = []
        self.readouts = []

        # Track input irreps for each layer: starts as scalar from node_embedding,
        # then becomes the output of the previous layer's product block.
        layer_input_irreps = hidden_irreps_scalar

        for i in range(num_interactions):
            layer_hidden = hidden_irreps_per_layer[i]

            if use_nonlinear_interaction:
                # NonLinear interaction block (mh-1 family)
                # First layer uses scalar-only edge_irreps when input is scalar-only
                if i == 0 and edge_irreps_parsed is not None:
                    layer_edge_irreps = Irreps(
                        [(mul, ir) for mul, ir in edge_irreps_parsed if ir.l == 0]
                    )
                else:
                    layer_edge_irreps = edge_irreps_parsed

                inter = RealAgnosticResidualNonLinearInteractionBlock(
                    irreps_in=layer_input_irreps,
                    irreps_out=interaction_irreps,
                    irreps_sh=sh_irreps,
                    num_radial=num_bessel,
                    radial_MLP=radial_MLP,
                    avg_num_neighbors=avg_num_neighbors,
                    num_species=num_elements,
                    hidden_irreps=layer_hidden,
                    edge_irreps=str(layer_edge_irreps) if layer_edge_irreps else None,
                )
            elif i == 0 and first_interaction_nonresidual:
                # Use non-residual block for first interaction when flagged
                # (0b/0b2/mpa-0 family); all other layers use residual block.
                # Density variants use per-node density normalization instead
                # of dividing by avg_num_neighbors (0b2/mpa-0 family).
                NonResBlock = (
                    RealAgnosticDensityInteractionBlock
                    if use_density_normalization
                    else RealAgnosticInteractionBlock
                )
                inter = NonResBlock(
                    irreps_in=layer_input_irreps,
                    irreps_out=interaction_irreps,
                    irreps_sh=sh_irreps,
                    num_radial=num_bessel,
                    radial_MLP=radial_MLP,
                    avg_num_neighbors=avg_num_neighbors,
                    num_species=num_elements,
                )
            else:
                ResBlock = (
                    RealAgnosticDensityResidualInteractionBlock
                    if use_density_normalization
                    else RealAgnosticResidualInteractionBlock
                )
                inter = ResBlock(
                    irreps_in=layer_input_irreps,
                    irreps_out=interaction_irreps,
                    irreps_sh=sh_irreps,
                    num_radial=num_bessel,
                    radial_MLP=radial_MLP,
                    avg_num_neighbors=avg_num_neighbors,
                    num_species=num_elements,
                    hidden_irreps=layer_hidden,
                )
            self.interactions.append(inter)

            # Product basis: contracts interaction_irreps -> layer_hidden
            # Agnostic product uses num_elements=1 (element-independent weights)
            prod_num_elements = 1 if use_agnostic_product else num_elements
            prod = EquivariantProductBasisBlock(
                irreps_in=interaction_irreps,
                irreps_out=layer_hidden,
                correlation=correlation[i],
                num_elements=prod_num_elements,
            )
            self.products.append(prod)

            # Readout: linear for all but last, nonlinear for last
            if i < num_interactions - 1:
                readout = LinearReadoutBlock(
                    irreps_in=layer_hidden,
                    irreps_out=readout_out_irreps,
                )
            else:
                readout = NonLinearReadoutBlock(
                    irreps_in=layer_hidden,
                    MLP_irreps=mlp_irreps,
                    gate_fn=nn.silu,
                    irreps_out=readout_out_irreps,
                    num_heads=self.num_heads,
                )
            self.readouts.append(readout)

            # Next layer's input is this layer's output
            layer_input_irreps = layer_hidden

    def _embed_radial(
        self,
        lengths: mx.array,
        node_attrs: mx.array,
        edge_index: mx.array,
    ) -> tuple[mx.array, mx.array | None]:
        """Compute radial embedding, forwarding extra args for distance transforms.

        Returns:
            (edge_feats, cutoff_or_None) tuple. cutoff is None when
            apply_cutoff=True (cutoff already baked into edge_feats).
        """
        if self._has_distance_transform:
            result = self.radial_embedding(
                lengths, node_attrs, edge_index, self._atomic_numbers
            )
        else:
            result = self.radial_embedding(lengths)
        if isinstance(result, tuple):
            return result
        return result, None

    def set_dtype(self, dtype, predicate=None):
        """Override to also convert _sh_rotation (private attr)."""
        super().set_dtype(dtype, predicate)
        if hasattr(self, "_sh_rotation") and self._sh_rotation.dtype != dtype:
            self._sh_rotation = self._sh_rotation.astype(dtype)

    def set_fused_kernel(self, enabled: bool = True) -> None:
        """Enable/disable fused Metal gather-TP-scatter kernels in all interaction blocks.

        The Metal kernels eliminate intermediate tensor allocations in the
        gather -> TensorProduct -> scatter hot path. Only effective for
        scalar-only TP (hidden_irreps = Nx0e). No autograd support, so
        this should be enabled for inference only.

        Args:
            enabled: True to enable, False to disable.
        """
        for inter in self.interactions:
            if hasattr(inter, "set_fused_kernel"):
                inter.set_fused_kernel(enabled)

    def _readout_select_head(self, readout_output: mx.array) -> mx.array:
        """Select the active head from readout output.

        Args:
            readout_output: (num_atoms, num_heads) from readout block

        Returns:
            (num_atoms,) selected head energy
        """
        if self.num_heads == 1:
            return readout_output.squeeze(-1)
        return readout_output[:, self._head_idx]

    def _e0_select_head(self, e0_output: mx.array) -> mx.array:
        """Select the active head from atomic energies output.

        Args:
            e0_output: (num_atoms, num_heads) from AtomicEnergiesBlock

        Returns:
            (num_atoms,) selected head e0
        """
        if self.num_heads == 1:
            return e0_output.squeeze(-1)
        return e0_output[:, self._head_idx]

    def __call__(
        self,
        positions: mx.array,
        node_attrs: mx.array,
        edge_index: mx.array,
        shifts: mx.array,
        cell: mx.array | None = None,
        batch: mx.array | None = None,
        num_graphs: int = 1,
    ) -> dict:
        """Forward pass.

        Args:
            positions: (num_atoms, 3) atom positions
            node_attrs: (num_atoms, num_elements) one-hot encoding
            edge_index: (2, num_edges) [senders, receivers]
            shifts: (num_edges, 3) periodic shift vectors
            cell: (3, 3) unit cell or None
            batch: (num_atoms,) graph membership indices or None
            num_graphs: number of graphs in the batch (avoids batch.max().item())

        Returns:
            dict with 'energy', 'node_energy', 'forces'
        """
        num_atoms = positions.shape[0]
        if batch is None:
            batch = mx.zeros(num_atoms, dtype=mx.int32)

        # stop_gradient on inputs that don't depend on positions —
        # prevents autograd from tracking unnecessary computation branches
        node_attrs = mx.stop_gradient(node_attrs)
        shifts = mx.stop_gradient(shifts)
        if cell is not None:
            cell = mx.stop_gradient(cell)

        # 1. Edge vectors and lengths
        vectors, lengths = get_edge_vectors_and_lengths(
            positions, edge_index, shifts, cell
        )

        # 2. Atomic energies (baseline, position-independent)
        node_e0 = self._e0_select_head(
            self.atomic_energies_fn(node_attrs)
        )  # (num_atoms,)

        # 3. Node embedding (position-independent, stop_gradient since
        #    dE/dpositions doesn't need gradients through this branch)
        node_feats = mx.stop_gradient(self.node_embedding(node_attrs))

        # 4. Spherical harmonics in standard m-ordering, then rotate to
        #    e3nn ordering so that the TP and downstream weights are consistent
        edge_attrs = spherical_harmonics(
            self.max_ell, vectors, normalize=True, normalization="component"
        )
        edge_attrs = edge_attrs @ self._sh_rotation

        # 5. Radial embedding
        edge_feats, cutoff = self._embed_radial(lengths, node_attrs, edge_index)

        # 6. Interaction loop
        node_energies_list = [node_e0]
        node_feats_list = []

        for i in range(self.num_interactions):
            node_feats, sc = self.interactions[i](
                node_feats, node_attrs, edge_attrs, edge_feats, edge_index,
                cutoff=cutoff,
            )
            node_feats = self.products[i](node_feats, node_attrs, sc=sc)
            node_feats_list.append(node_feats)

        # 7. Readouts — each produces (num_atoms, num_heads), select active head
        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            # Pass head_idx to NonLinearReadoutBlock for mask_head
            if hasattr(readout, '_num_heads') and readout._num_heads > 1:
                raw = readout(node_feats_list[feat_idx], head_idx=self._head_idx)
            else:
                raw = readout(node_feats_list[feat_idx])
            node_es = self._readout_select_head(raw)  # (num_atoms,)
            node_energies_list.append(node_es)

        # 8. Accumulate energies — cast to float32 before summing for
        # numerical stability with FP16/BF16 weights.
        node_energy = mx.stack(
            [e.astype(mx.float32) for e in node_energies_list], axis=0
        ).sum(axis=0)

        # Per-graph energy
        energy = scatter_sum(
            node_energy[:, None], batch, num_graphs
        ).squeeze(-1)

        return {
            "energy": energy,
            "node_energy": node_energy,
        }

    def _forward_from_vectors(
        self,
        vectors: mx.array,
        lengths: mx.array,
        node_attrs: mx.array,
        edge_index: mx.array,
        batch: mx.array,
        num_graphs: int = 1,
    ) -> mx.array:
        """Compute total energy from pre-computed edge vectors and lengths.

        Used by the calculator for stress computation where vectors/lengths
        already incorporate strain and must remain differentiable.

        Args:
            vectors: (num_edges, 3) edge displacement vectors
            lengths: (num_edges, 1) edge distances
            node_attrs: (num_atoms, num_elements) one-hot encoding
            edge_index: (2, num_edges)
            batch: (num_atoms,) graph membership
            num_graphs: number of graphs

        Returns:
            Scalar total energy.
        """
        node_attrs = mx.stop_gradient(node_attrs)

        # Atomic energies (baseline)
        node_e0 = self._e0_select_head(self.atomic_energies_fn(node_attrs))

        # Node embedding
        node_feats = mx.stop_gradient(self.node_embedding(node_attrs))

        # Spherical harmonics (rotate to e3nn basis)
        edge_attrs = spherical_harmonics(
            self.max_ell, vectors, normalize=True, normalization="component"
        )
        edge_attrs = edge_attrs @ self._sh_rotation

        # Radial embedding
        edge_feats, cutoff = self._embed_radial(lengths, node_attrs, edge_index)

        # Interaction loop
        node_energies_list = [node_e0]
        node_feats_list = []

        for i in range(self.num_interactions):
            node_feats, sc = self.interactions[i](
                node_feats, node_attrs, edge_attrs, edge_feats, edge_index,
                cutoff=cutoff,
            )
            node_feats = self.products[i](node_feats, node_attrs, sc=sc)
            node_feats_list.append(node_feats)

        # Readouts
        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            if hasattr(readout, '_num_heads') and readout._num_heads > 1:
                raw = readout(node_feats_list[feat_idx], head_idx=self._head_idx)
            else:
                raw = readout(node_feats_list[feat_idx])
            node_es = self._readout_select_head(raw)
            node_energies_list.append(node_es)

        # Accumulate energies in float32 for numerical stability
        node_energy = mx.stack(
            [e.astype(mx.float32) for e in node_energies_list], axis=0
        ).sum(axis=0)

        energy = scatter_sum(
            node_energy[:, None], batch, num_graphs
        ).squeeze(-1)
        return energy.sum()

    def _forward_from_vectors_with_node_energy(
        self,
        vectors: mx.array,
        lengths: mx.array,
        node_attrs: mx.array,
        edge_index: mx.array,
        batch: mx.array,
        num_graphs: int = 1,
    ) -> tuple[mx.array, mx.array]:
        """Like _forward_from_vectors but also returns node_energy.

        Returns:
            (scalar_energy, node_energy) tuple.
        """
        node_attrs = mx.stop_gradient(node_attrs)

        # Atomic energies (baseline)
        node_e0 = self._e0_select_head(self.atomic_energies_fn(node_attrs))

        # Node embedding
        node_feats = mx.stop_gradient(self.node_embedding(node_attrs))

        # Spherical harmonics (rotate to e3nn basis)
        edge_attrs = spherical_harmonics(
            self.max_ell, vectors, normalize=True, normalization="component"
        )
        edge_attrs = edge_attrs @ self._sh_rotation

        # Radial embedding
        edge_feats, cutoff = self._embed_radial(lengths, node_attrs, edge_index)

        # Interaction loop
        node_energies_list = [node_e0]
        node_feats_list = []

        for i in range(self.num_interactions):
            node_feats, sc = self.interactions[i](
                node_feats, node_attrs, edge_attrs, edge_feats, edge_index,
                cutoff=cutoff,
            )
            node_feats = self.products[i](node_feats, node_attrs, sc=sc)
            node_feats_list.append(node_feats)

        # Readouts — each produces (num_atoms, num_heads), select active head
        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            # Pass head_idx to NonLinearReadoutBlock for mask_head
            if hasattr(readout, '_num_heads') and readout._num_heads > 1:
                raw = readout(node_feats_list[feat_idx], head_idx=self._head_idx)
            else:
                raw = readout(node_feats_list[feat_idx])
            node_es = self._readout_select_head(raw)  # (num_atoms,)
            node_energies_list.append(node_es)

        # Accumulate energies in float32 for numerical stability
        node_energy = mx.stack(
            [e.astype(mx.float32) for e in node_energies_list], axis=0
        ).sum(axis=0)

        energy = scatter_sum(
            node_energy[:, None], batch, num_graphs
        ).squeeze(-1)
        return energy.sum(), node_energy

    def energy_fn(
        self,
        positions: mx.array,
        node_attrs: mx.array,
        edge_index: mx.array,
        shifts: mx.array,
        cell: mx.array | None = None,
        batch: mx.array | None = None,
        num_graphs: int = 1,
    ) -> mx.array:
        """Compute total energy (scalar) for use with mx.grad."""
        result = self(positions, node_attrs, edge_index, shifts, cell, batch, num_graphs)
        return result["energy"].sum()


class ScaleShiftMACE(MACE):
    """MACE with scale-shift applied to interaction energies.

    The scale and shift are applied to the sum of readout energies
    (excluding the atomic baseline e0):
        interaction_energy = scale * sum(readout_energies) + shift * num_atoms

    For multi-head models, scale and shift can be vectors (one per head).
    """

    def __init__(
        self,
        scale: float | list[float] = 1.0,
        shift: float | list[float] = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Store as lists for multi-head indexing; scalars become 1-element lists
        if isinstance(scale, (int, float)):
            self._scale_list = [float(scale)]
        else:
            self._scale_list = [float(s) for s in scale]
        if isinstance(shift, (int, float)):
            self._shift_list = [float(shift)]
        else:
            self._shift_list = [float(s) for s in shift]

    @property
    def scale_val(self) -> float:
        """Return scale for the active head."""
        idx = self._head_idx if self._head_idx < len(self._scale_list) else 0
        return self._scale_list[idx]

    @property
    def shift_val(self) -> float:
        """Return shift for the active head."""
        idx = self._head_idx if self._head_idx < len(self._shift_list) else 0
        return self._shift_list[idx]

    def _ss_forward_core(
        self,
        node_attrs: mx.array,
        node_feats: mx.array,
        edge_attrs: mx.array,
        edge_feats: mx.array,
        edge_index: mx.array,
        batch: mx.array,
        num_graphs: int,
        cutoff: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Shared core of ScaleShiftMACE forward pass.

        Returns:
            (node_e0, node_inter_es, total_energy, node_energy, inter_e)
        """
        # Atomic energies (baseline, position-independent)
        node_e0 = self._e0_select_head(self.atomic_energies_fn(node_attrs))

        # Interaction loop
        node_es_list = []
        node_feats_list = []

        for i in range(self.num_interactions):
            node_feats, sc = self.interactions[i](
                node_feats, node_attrs, edge_attrs, edge_feats, edge_index,
                cutoff=cutoff,
            )
            node_feats = self.products[i](node_feats, node_attrs, sc=sc)
            node_feats_list.append(node_feats)

        # Readouts
        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            if hasattr(readout, '_num_heads') and readout._num_heads > 1:
                raw = readout(node_feats_list[feat_idx], head_idx=self._head_idx)
            else:
                raw = readout(node_feats_list[feat_idx])
            node_es = self._readout_select_head(raw)
            node_es_list.append(node_es)

        # Scale-shift on interaction energies (accumulate in float32)
        node_inter_es = mx.stack(
            [e.astype(mx.float32) for e in node_es_list], axis=0
        ).sum(axis=0)
        node_inter_es = self.scale_val * node_inter_es + self.shift_val

        # Total node energy = e0 + scaled interaction (both in float32)
        node_energy = node_e0.astype(mx.float32) + node_inter_es

        # Per-graph energies
        e0 = scatter_sum(node_e0.astype(mx.float32)[:, None], batch, num_graphs).squeeze(-1)
        inter_e = scatter_sum(
            node_inter_es[:, None], batch, num_graphs
        ).squeeze(-1)
        total_energy = e0 + inter_e

        return node_e0, node_inter_es, total_energy, node_energy, inter_e

    def __call__(
        self,
        positions: mx.array,
        node_attrs: mx.array,
        edge_index: mx.array,
        shifts: mx.array,
        cell: mx.array | None = None,
        batch: mx.array | None = None,
        num_graphs: int = 1,
    ) -> dict:
        num_atoms = positions.shape[0]
        if batch is None:
            batch = mx.zeros(num_atoms, dtype=mx.int32)

        # stop_gradient on inputs that don't depend on positions
        node_attrs = mx.stop_gradient(node_attrs)
        shifts = mx.stop_gradient(shifts)
        if cell is not None:
            cell = mx.stop_gradient(cell)

        # 1. Edge vectors and lengths
        vectors, lengths = get_edge_vectors_and_lengths(
            positions, edge_index, shifts, cell
        )

        # 2. Node embedding
        node_feats = mx.stop_gradient(self.node_embedding(node_attrs))

        # 3. Spherical harmonics (rotate to e3nn basis)
        edge_attrs = spherical_harmonics(
            self.max_ell, vectors, normalize=True, normalization="component"
        )
        edge_attrs = edge_attrs @ self._sh_rotation

        # 4. Radial embedding
        edge_feats, cutoff = self._embed_radial(lengths, node_attrs, edge_index)

        # 5. Core forward (interactions + readouts + scale-shift)
        _, _, total_energy, node_energy, inter_e = self._ss_forward_core(
            node_attrs, node_feats, edge_attrs, edge_feats, edge_index, batch, num_graphs,
            cutoff=cutoff,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
        }

    def _forward_from_vectors(
        self,
        vectors: mx.array,
        lengths: mx.array,
        node_attrs: mx.array,
        edge_index: mx.array,
        batch: mx.array,
        num_graphs: int = 1,
    ) -> mx.array:
        """Compute total energy from pre-computed edge vectors and lengths.

        ScaleShiftMACE variant: applies scale/shift to interaction energies.
        """
        node_attrs = mx.stop_gradient(node_attrs)

        # Node embedding
        node_feats = mx.stop_gradient(self.node_embedding(node_attrs))

        # Spherical harmonics (rotate to e3nn basis)
        edge_attrs = spherical_harmonics(
            self.max_ell, vectors, normalize=True, normalization="component"
        )
        edge_attrs = edge_attrs @ self._sh_rotation

        # Radial embedding
        edge_feats, cutoff = self._embed_radial(lengths, node_attrs, edge_index)

        _, _, total_energy, _, _ = self._ss_forward_core(
            node_attrs, node_feats, edge_attrs, edge_feats, edge_index, batch, num_graphs,
            cutoff=cutoff,
        )
        return total_energy.sum()

    def _forward_from_vectors_with_node_energy(
        self,
        vectors: mx.array,
        lengths: mx.array,
        node_attrs: mx.array,
        edge_index: mx.array,
        batch: mx.array,
        num_graphs: int = 1,
    ) -> tuple[mx.array, mx.array]:
        """Like _forward_from_vectors but also returns node_energy."""
        node_attrs = mx.stop_gradient(node_attrs)

        # Node embedding
        node_feats = mx.stop_gradient(self.node_embedding(node_attrs))

        # Spherical harmonics (rotate to e3nn basis)
        edge_attrs = spherical_harmonics(
            self.max_ell, vectors, normalize=True, normalization="component"
        )
        edge_attrs = edge_attrs @ self._sh_rotation

        # Radial embedding
        edge_feats, cutoff = self._embed_radial(lengths, node_attrs, edge_index)

        _, _, total_energy, node_energy, _ = self._ss_forward_core(
            node_attrs, node_feats, edge_attrs, edge_feats, edge_index, batch, num_graphs,
            cutoff=cutoff,
        )
        return total_energy.sum(), node_energy

    def energy_fn(
        self,
        positions: mx.array,
        node_attrs: mx.array,
        edge_index: mx.array,
        shifts: mx.array,
        cell: mx.array | None = None,
        batch: mx.array | None = None,
        num_graphs: int = 1,
    ) -> mx.array:
        result = self(positions, node_attrs, edge_index, shifts, cell, batch, num_graphs)
        return result["energy"].sum()


_DTYPE_MAP = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}


def _convert_private_arrays(model: nn.Module, dtype: mx.Dtype) -> None:
    """Convert all private mx.array attributes in the model tree to target dtype.

    nn.Module.set_dtype only converts 'valid parameters' (public attrs).
    This function also converts private attrs (prefixed with '_') that are
    mx.array with a floating dtype -- e.g. precomputed CG coefficients,
    U matrices, rotation matrices.

    MLX Module inherits from dict, so attributes are stored in the module's
    dict (module.keys()) and also in __dict__. We check both.
    """
    def _convert_value(val):
        """Convert a single value, returning the converted version."""
        if isinstance(val, mx.array) and mx.issubdtype(val.dtype, mx.floating):
            return val.astype(dtype) if val.dtype != dtype else val
        return val

    for _, module in model.named_modules():
        # Check module dict items (MLX Module is a dict subclass)
        for key in list(module.keys()):
            val = module[key]
            if isinstance(val, mx.array) and mx.issubdtype(val.dtype, mx.floating):
                if val.dtype != dtype:
                    module[key] = val.astype(dtype)
            elif isinstance(val, dict):
                for k, v in val.items():
                    converted = _convert_value(v)
                    if converted is not v:
                        val[k] = converted
            elif isinstance(val, list):
                for i, v in enumerate(val):
                    converted = _convert_value(v)
                    if converted is not v:
                        val[i] = converted

        # Also check __dict__ (instance attributes not in the module dict)
        for attr_name in list(module.__dict__.keys()):
            val = module.__dict__[attr_name]
            if isinstance(val, mx.array) and mx.issubdtype(val.dtype, mx.floating):
                if val.dtype != dtype:
                    module.__dict__[attr_name] = val.astype(dtype)


def load_model(
    model_dir: str, dtype: str = "float32"
) -> MACE | ScaleShiftMACE:
    """Load a converted MACE model from a directory.

    Expects:
        model_dir/config.json — model hyperparameters
        model_dir/weights.npz — converted weight arrays

    Args:
        model_dir: path to directory with config.json and weights.npz
        dtype: "float32", "float16", or "bfloat16"

    Returns:
        Instantiated MACE or ScaleShiftMACE with loaded weights.
    """
    model_dir = Path(model_dir)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    model_type = config.pop("model_type", "ScaleShiftMACE")
    scale = config.pop("scale", 1.0)
    shift = config.pop("shift", 0.0)
    z_table = config.pop("z_table", None)
    heads = config.get("heads", None)

    # Convert atomic_energies back to mx.array
    if "atomic_energies" in config and config["atomic_energies"] is not None:
        config["atomic_energies"] = mx.array(config["atomic_energies"])

    if model_type == "ScaleShiftMACE":
        model = ScaleShiftMACE(scale=scale, shift=shift, **config)
    else:
        model = MACE(**config)

    # Load weights
    weights = mx.load(str(model_dir / "weights.npz"))
    # Convert flat key dict to nested dict for model.load_weights
    model.load_weights(list(weights.items()))

    # Convert model parameters to target dtype
    if dtype in ("float16", "bfloat16"):
        mx_dtype = _DTYPE_MAP[dtype]
        model.set_dtype(mx_dtype)
        # Also convert private constant arrays (CG coefficients, U matrices, etc.)
        _convert_private_arrays(model, mx_dtype)

    # Store z_table and compute dtype on model for calculator use
    if z_table is not None:
        model.z_table = z_table
        model._atomic_numbers = mx.array(z_table, dtype=mx.int32)
    model._compute_dtype = _DTYPE_MAP.get(dtype, mx.float32)

    return model
