"""Tests for Gate, utils, and MACE building blocks."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mace_mlx.gate import Gate
from mace_mlx.irreps import Irreps
from mace_mlx.utils import (
    get_edge_vectors_and_lengths,
    scatter_sum,
    tp_out_irreps_with_instructions,
)
from mace_mlx.blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)


# ============================================================
# scatter_sum tests
# ============================================================


class TestScatterSum:
    def test_basic_shape(self):
        src = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = mx.array([0, 1, 0])
        out = scatter_sum(src, index, 3)
        mx.eval(out)
        assert out.shape == (3, 2)

    def test_basic_values(self):
        src = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = mx.array([0, 1, 0])
        out = scatter_sum(src, index, 3)
        mx.eval(out)
        expected = np.array([[6.0, 8.0], [3.0, 4.0], [0.0, 0.0]])
        np.testing.assert_allclose(np.array(out), expected, atol=1e-6)

    def test_all_same_index(self):
        src = mx.ones((5, 3))
        index = mx.zeros((5,), dtype=mx.int32)
        out = scatter_sum(src, index, 2)
        mx.eval(out)
        np.testing.assert_allclose(np.array(out[0]), [5.0, 5.0, 5.0], atol=1e-6)
        np.testing.assert_allclose(np.array(out[1]), [0.0, 0.0, 0.0], atol=1e-6)

    def test_empty_source(self):
        src = mx.zeros((0, 4))
        index = mx.array([], dtype=mx.int32)
        out = scatter_sum(src, index, 3)
        mx.eval(out)
        assert out.shape == (3, 4)
        np.testing.assert_allclose(np.array(out), 0.0, atol=1e-10)

    def test_gradient_flow(self):
        """CRITICAL: Verify mx.grad works through scatter_sum."""
        src = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = mx.array([0, 1, 0])

        def f(s):
            return scatter_sum(s, index, 2).sum()

        grad_fn = mx.grad(f)
        grads = grad_fn(src)
        mx.eval(grads)
        assert grads.shape == src.shape
        # Each src element contributes exactly once to the sum,
        # so gradient should be all 1s
        np.testing.assert_allclose(np.array(grads), 1.0, atol=1e-6)

    def test_gradient_selective(self):
        """Test that gradient flows correctly with different index patterns."""
        src = mx.array([[1.0], [2.0], [3.0], [4.0]])
        index = mx.array([0, 0, 1, 1])

        def f(s):
            result = scatter_sum(s, index, 2)
            # Only take the sum of index-0 bucket
            return result[0].sum()

        grad_fn = mx.grad(f)
        grads = grad_fn(src)
        mx.eval(grads)
        # Only indices 0 and 1 contribute to result[0]
        expected = np.array([[1.0], [1.0], [0.0], [0.0]])
        np.testing.assert_allclose(np.array(grads), expected, atol=1e-6)

    def test_gradient_with_downstream_operations(self):
        """Test gradient through scatter_sum -> further ops."""
        index = mx.array([0, 1, 0])

        def f(s):
            aggregated = scatter_sum(s, index, 2)  # (2, 2)
            return (aggregated ** 2).sum()

        src = mx.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        grad_fn = mx.grad(f)
        grads = grad_fn(src)
        mx.eval(grads)
        assert grads.shape == src.shape
        # aggregated = [[2, 0], [0, 1]]
        # loss = 4 + 0 + 0 + 1 = 5
        # d(loss)/d(aggregated) = [[4, 0], [0, 2]]
        # d(loss)/d(src[0]) = d/d(agg[0]) = [4, 0]
        # d(loss)/d(src[1]) = d/d(agg[1]) = [0, 2]
        # d(loss)/d(src[2]) = d/d(agg[0]) = [4, 0]
        expected = np.array([[4.0, 0.0], [0.0, 2.0], [4.0, 0.0]])
        np.testing.assert_allclose(np.array(grads), expected, atol=1e-6)


# ============================================================
# get_edge_vectors_and_lengths tests
# ============================================================


class TestEdgeVectors:
    def test_basic(self):
        positions = mx.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        edge_index = mx.array([[0, 1], [1, 2]])  # edges: 0->1, 1->2
        shifts = mx.zeros((2, 3))

        vectors, lengths = get_edge_vectors_and_lengths(
            positions, edge_index, shifts
        )
        mx.eval(vectors, lengths)

        assert vectors.shape == (2, 3)
        assert lengths.shape == (2, 1)
        # Edge 0->1: pos[1] - pos[0] = [1, 0, 0]
        np.testing.assert_allclose(np.array(vectors[0]), [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(np.array(lengths[0]), [1.0], atol=1e-6)

    def test_with_cell(self):
        positions = mx.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
        edge_index = mx.array([[0], [1]])
        shifts = mx.array([[1.0, 0.0, 0.0]])
        cell = mx.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])

        vectors, lengths = get_edge_vectors_and_lengths(
            positions, edge_index, shifts, cell
        )
        mx.eval(vectors, lengths)
        # vector = (0.9 - 0.0) + [1,0,0] @ cell = 0.9 + 2.0 = 2.9
        np.testing.assert_allclose(np.array(vectors[0, 0]), 2.9, atol=1e-6)


# ============================================================
# compute_forces tests
# ============================================================


class TestComputeForces:
    def test_harmonic_potential(self):
        """Test forces from a simple harmonic potential E = 0.5 * sum(r^2)."""
        positions = mx.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])

        def energy_fn(pos):
            return 0.5 * mx.sum(pos * pos)

        forces = -mx.grad(energy_fn)(positions)
        mx.eval(forces)
        # F = -dE/dr = -r
        np.testing.assert_allclose(
            np.array(forces), -np.array(positions), atol=1e-5
        )


# ============================================================
# tp_out_irreps_with_instructions tests
# ============================================================


class TestTpOutIrreps:
    def test_basic(self):
        irreps1 = Irreps("4x0e + 4x1o")
        irreps2 = Irreps("0e + 1o + 2e")
        target = Irreps("4x0e + 4x1o + 4x2e")

        irreps_out, instructions = tp_out_irreps_with_instructions(
            irreps1, irreps2, target
        )

        assert len(instructions) > 0
        assert irreps_out.dim > 0

        # All instructions should use "uvu" mode
        for inst in instructions:
            assert inst[3] == "uvu"

    def test_respects_target(self):
        """Only irreps in target should appear in the output."""
        irreps1 = Irreps("2x0e + 2x1o")
        irreps2 = Irreps("0e + 1o")
        target = Irreps("2x0e")

        irreps_out, instructions = tp_out_irreps_with_instructions(
            irreps1, irreps2, target
        )

        # All output irreps should be 0e
        for mul, ir in irreps_out:
            assert ir.l == 0 and ir.p == 1


# ============================================================
# Gate tests
# ============================================================


class TestGate:
    def test_scalar_only(self):
        """Gate with only scalars (no gated part)."""
        gate = Gate(
            irreps_scalars=Irreps("4x0e"),
            act_scalars=[nn.silu],
            irreps_gates=Irreps(""),
            act_gates=[],
            irreps_gated=Irreps(""),
        )
        x = mx.random.normal((5, 4))
        out = gate(x)
        mx.eval(out)
        assert out.shape == (5, 4)
        # Should match direct silu application
        expected = nn.silu(x)
        mx.eval(expected)
        np.testing.assert_allclose(np.array(out), np.array(expected), atol=1e-6)

    def test_with_gated_features(self):
        """Gate with scalars and gated non-scalar features."""
        # 4 scalars activated + 2 gate scalars + 2x1o gated
        gate = Gate(
            irreps_scalars=Irreps("4x0e"),
            act_scalars=[nn.silu],
            irreps_gates=Irreps("2x0e"),
            act_gates=[mx.sigmoid],
            irreps_gated=Irreps("2x1o"),
        )
        # Input dim = 4 + 2 + 2*3 = 12
        assert gate.irreps_in.dim == 12
        # Output dim = 4 + 2*3 = 10
        assert gate.irreps_out.dim == 10

        x = mx.random.normal((3, 12))
        out = gate(x)
        mx.eval(out)
        assert out.shape == (3, 10)

    def test_output_shape(self):
        """Test various configurations for correct output shapes."""
        gate = Gate(
            irreps_scalars=Irreps("8x0e"),
            act_scalars=[nn.silu],
            irreps_gates=Irreps("4x0e"),
            act_gates=[mx.sigmoid],
            irreps_gated=Irreps("2x1o + 2x2e"),
        )
        # gates=4 = 2+2 (2 copies of 1o + 2 copies of 2e)
        x = mx.random.normal((5, gate.irreps_in.dim))
        out = gate(x)
        mx.eval(out)
        assert out.shape == (5, gate.irreps_out.dim)

    def test_scalar_passthrough(self):
        """With identity activation on scalars, should pass through."""
        gate = Gate(
            irreps_scalars=Irreps("2x0e"),
            act_scalars=[lambda x: x],
            irreps_gates=Irreps(""),
            act_gates=[],
            irreps_gated=Irreps(""),
        )
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        out = gate(x)
        mx.eval(out)
        np.testing.assert_allclose(np.array(out), np.array(x), atol=1e-6)

    def test_batch_dimension(self):
        """Gate should work with various batch shapes."""
        gate = Gate(
            irreps_scalars=Irreps("2x0e"),
            act_scalars=[nn.silu],
            irreps_gates=Irreps("1x0e"),
            act_gates=[mx.sigmoid],
            irreps_gated=Irreps("1x1o"),
        )
        # Batch of 10
        x = mx.random.normal((10, gate.irreps_in.dim))
        out = gate(x)
        mx.eval(out)
        assert out.shape == (10, gate.irreps_out.dim)


# ============================================================
# LinearNodeEmbeddingBlock tests
# ============================================================


class TestLinearNodeEmbedding:
    def test_forward_shape(self):
        block = LinearNodeEmbeddingBlock(
            irreps_in=Irreps("4x0e"),
            irreps_out=Irreps("8x0e + 4x1o"),
        )
        # One-hot species: 4 species
        node_attrs = mx.random.normal((10, 4))
        out = block(node_attrs)
        mx.eval(out)
        expected_dim = Irreps("8x0e + 4x1o").dim  # 8 + 12 = 20
        assert out.shape == (10, expected_dim)


# ============================================================
# AtomicEnergiesBlock tests
# ============================================================


class TestAtomicEnergies:
    def test_basic(self):
        energies = mx.array([-1.0, -2.0, -3.0])
        block = AtomicEnergiesBlock(energies)
        # 3-element one-hot
        species_onehot = mx.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        out = block(species_onehot)
        mx.eval(out)
        expected = np.array([[-1.0], [-2.0], [-3.0], [-1.0]])
        np.testing.assert_allclose(np.array(out), expected, atol=1e-6)


# ============================================================
# Readout block tests
# ============================================================


class TestReadoutBlocks:
    def test_linear_readout(self):
        block = LinearReadoutBlock(
            irreps_in=Irreps("8x0e + 4x1o"),
            irreps_out=Irreps("1x0e"),
        )
        x = mx.random.normal((5, Irreps("8x0e + 4x1o").dim))
        out = block(x)
        mx.eval(out)
        assert out.shape == (5, 1)

    def test_nonlinear_readout(self):
        block = NonLinearReadoutBlock(
            irreps_in=Irreps("16x0e"),
            MLP_irreps=Irreps("8x0e"),
            gate_fn=nn.silu,
            irreps_out=Irreps("1x0e"),
        )
        x = mx.random.normal((5, 16))
        out = block(x)
        mx.eval(out)
        assert out.shape == (5, 1)


# ============================================================
# InteractionBlock tests
# ============================================================


class TestInteractionBlock:
    @pytest.fixture
    def small_graph(self):
        """Small 3-atom graph for testing."""
        num_atoms = 3
        num_edges = 4
        num_species = 2
        num_radial = 8
        irreps_in = Irreps("4x0e + 4x1o")
        irreps_out = Irreps("4x0e + 4x1o")
        irreps_sh = Irreps("0e + 1o")

        mx.random.seed(42)
        node_feats = mx.random.normal((num_atoms, irreps_in.dim))
        # One-hot species
        node_attrs = mx.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        edge_attrs = mx.random.normal((num_edges, irreps_sh.dim))
        edge_feats = mx.random.normal((num_edges, num_radial))
        edge_index = mx.array([
            [0, 0, 1, 2],
            [1, 2, 0, 1],
        ])

        return {
            "node_feats": node_feats,
            "node_attrs": node_attrs,
            "edge_attrs": edge_attrs,
            "edge_feats": edge_feats,
            "edge_index": edge_index,
            "irreps_in": irreps_in,
            "irreps_out": irreps_out,
            "irreps_sh": irreps_sh,
            "num_radial": num_radial,
            "num_species": num_species,
            "num_atoms": num_atoms,
        }

    def test_interaction_block_shape(self, small_graph):
        block = RealAgnosticInteractionBlock(
            irreps_in=small_graph["irreps_in"],
            irreps_out=small_graph["irreps_out"],
            irreps_sh=small_graph["irreps_sh"],
            num_radial=small_graph["num_radial"],
            radial_MLP=[16, 16],
            avg_num_neighbors=2.0,
            num_species=small_graph["num_species"],
        )
        msg, sc = block(
            small_graph["node_feats"],
            small_graph["node_attrs"],
            small_graph["edge_attrs"],
            small_graph["edge_feats"],
            small_graph["edge_index"],
        )
        mx.eval(msg)
        assert msg.shape == (small_graph["num_atoms"], small_graph["irreps_out"].dim)
        assert sc is None

    def test_residual_interaction_block_shape(self, small_graph):
        hidden_irreps = Irreps("4x0e")
        block = RealAgnosticResidualInteractionBlock(
            irreps_in=small_graph["irreps_in"],
            irreps_out=small_graph["irreps_out"],
            irreps_sh=small_graph["irreps_sh"],
            num_radial=small_graph["num_radial"],
            radial_MLP=[16, 16],
            avg_num_neighbors=2.0,
            num_species=small_graph["num_species"],
            hidden_irreps=hidden_irreps,
        )
        msg, sc = block(
            small_graph["node_feats"],
            small_graph["node_attrs"],
            small_graph["edge_attrs"],
            small_graph["edge_feats"],
            small_graph["edge_index"],
        )
        mx.eval(msg, sc)
        assert msg.shape == (small_graph["num_atoms"], small_graph["irreps_out"].dim)
        assert sc.shape == (small_graph["num_atoms"], hidden_irreps.dim)

    def test_interaction_gradient_flow(self, small_graph):
        """Verify gradient flows through the interaction block."""
        block = RealAgnosticInteractionBlock(
            irreps_in=small_graph["irreps_in"],
            irreps_out=small_graph["irreps_out"],
            irreps_sh=small_graph["irreps_sh"],
            num_radial=small_graph["num_radial"],
            radial_MLP=[16, 16],
            avg_num_neighbors=2.0,
            num_species=small_graph["num_species"],
        )

        def loss_fn(node_feats):
            msg, _ = block(
                node_feats,
                small_graph["node_attrs"],
                small_graph["edge_attrs"],
                small_graph["edge_feats"],
                small_graph["edge_index"],
            )
            return mx.sum(msg)

        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(small_graph["node_feats"])
        mx.eval(grads)
        assert grads.shape == small_graph["node_feats"].shape
        # Gradients should be finite and non-zero
        grads_np = np.array(grads)
        assert np.all(np.isfinite(grads_np))
        assert np.any(np.abs(grads_np) > 1e-10)


# ============================================================
# EquivariantProductBasisBlock tests
# ============================================================


class TestProductBasisBlock:
    def test_forward_shape(self):
        irreps_in = Irreps("4x0e + 4x1o")
        irreps_out = Irreps("4x0e + 4x1o")
        num_elements = 2

        block = EquivariantProductBasisBlock(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=2,
            num_elements=num_elements,
        )

        node_feats = mx.random.normal((3, irreps_in.dim))
        node_attrs = mx.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

        out = block(node_feats, node_attrs)
        mx.eval(out)
        assert out.shape == (3, irreps_out.dim)

    def test_with_skip_connection(self):
        irreps_in = Irreps("4x0e + 4x1o")
        irreps_out = Irreps("4x0e + 4x1o")
        num_elements = 2

        block = EquivariantProductBasisBlock(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=2,
            num_elements=num_elements,
        )

        node_feats = mx.random.normal((3, irreps_in.dim))
        node_attrs = mx.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        sc = mx.random.normal((3, irreps_out.dim))

        out = block(node_feats, node_attrs, sc=sc)
        mx.eval(out)
        assert out.shape == (3, irreps_out.dim)


# ============================================================
# Full chain test
# ============================================================


class TestFullChain:
    def test_embedding_interaction_readout(self):
        """Test the full chain: embedding -> interaction -> readout."""
        num_atoms = 4
        num_edges = 6
        num_species = 2
        num_radial = 8
        irreps_hidden = Irreps("4x0e + 4x1o")
        irreps_sh = Irreps("0e + 1o")

        mx.random.seed(123)

        # 1. Node embedding
        embedding = LinearNodeEmbeddingBlock(
            irreps_in=Irreps(f"{num_species}x0e"),
            irreps_out=irreps_hidden,
        )
        node_attrs = mx.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        node_feats = embedding(node_attrs)
        mx.eval(node_feats)
        assert node_feats.shape == (num_atoms, irreps_hidden.dim)

        # 2. Interaction
        interaction = RealAgnosticInteractionBlock(
            irreps_in=irreps_hidden,
            irreps_out=irreps_hidden,
            irreps_sh=irreps_sh,
            num_radial=num_radial,
            radial_MLP=[16, 16],
            avg_num_neighbors=3.0,
            num_species=num_species,
        )
        edge_attrs = mx.random.normal((num_edges, irreps_sh.dim))
        edge_feats = mx.random.normal((num_edges, num_radial))
        edge_index = mx.array([
            [0, 0, 1, 1, 2, 3],
            [1, 2, 0, 3, 3, 0],
        ])

        node_feats_out, _ = interaction(
            node_feats, node_attrs, edge_attrs, edge_feats, edge_index
        )
        mx.eval(node_feats_out)
        assert node_feats_out.shape == (num_atoms, irreps_hidden.dim)

        # 3. Linear readout
        readout = LinearReadoutBlock(
            irreps_in=irreps_hidden,
            irreps_out=Irreps("1x0e"),
        )
        energies = readout(node_feats_out)
        mx.eval(energies)
        assert energies.shape == (num_atoms, 1)

        # 4. Total energy
        total_energy = mx.sum(energies)
        mx.eval(total_energy)
        assert total_energy.shape == ()

    def test_residual_chain(self):
        """Test residual interaction -> product basis -> readout."""
        num_atoms = 3
        num_edges = 4
        num_species = 2
        num_radial = 8
        irreps_hidden = Irreps("4x0e + 4x1o")
        irreps_sh = Irreps("0e + 1o")

        mx.random.seed(456)

        # Embedding
        embedding = LinearNodeEmbeddingBlock(
            irreps_in=Irreps(f"{num_species}x0e"),
            irreps_out=irreps_hidden,
        )
        node_attrs = mx.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        node_feats = embedding(node_attrs)

        # Residual interaction
        interaction = RealAgnosticResidualInteractionBlock(
            irreps_in=irreps_hidden,
            irreps_out=irreps_hidden,
            irreps_sh=irreps_sh,
            num_radial=num_radial,
            radial_MLP=[16, 16],
            avg_num_neighbors=2.0,
            num_species=num_species,
            hidden_irreps=irreps_hidden,
        )
        edge_attrs = mx.random.normal((num_edges, irreps_sh.dim))
        edge_feats = mx.random.normal((num_edges, num_radial))
        edge_index = mx.array([
            [0, 0, 1, 2],
            [1, 2, 0, 1],
        ])

        message, sc = interaction(
            node_feats, node_attrs, edge_attrs, edge_feats, edge_index
        )
        mx.eval(message, sc)

        # Product basis with skip connection
        product_basis = EquivariantProductBasisBlock(
            irreps_in=irreps_hidden,
            irreps_out=irreps_hidden,
            correlation=2,
            num_elements=num_species,
        )
        node_feats_out = product_basis(message, node_attrs, sc=sc)
        mx.eval(node_feats_out)
        assert node_feats_out.shape == (num_atoms, irreps_hidden.dim)

        # Nonlinear readout (using only scalars)
        readout = NonLinearReadoutBlock(
            irreps_in=irreps_hidden,
            MLP_irreps=Irreps("8x0e"),
            gate_fn=nn.silu,
            irreps_out=Irreps("1x0e"),
        )
        energies = readout(node_feats_out)
        mx.eval(energies)
        assert energies.shape == (num_atoms, 1)
