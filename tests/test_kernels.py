"""Tests for fused Metal kernels in mace_mlx/kernels.py."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mace_mlx.irreps import Irreps
from mace_mlx.kernels import (
    fused_gather_tp_scatter_scalar,
    fused_gather_tp_scatter_scalar_metal,
    gather_tp_scatter,
    scatter_sum_metal,
)
from mace_mlx.utils import scatter_sum


# ============================================================
# scatter_sum_metal tests
# ============================================================


class TestScatterSumMetal:
    def test_basic_values(self):
        src = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = mx.array([0, 1, 0], dtype=mx.int32)
        out = scatter_sum_metal(src, index, 3)
        mx.eval(out)
        expected = np.array([[6.0, 8.0], [3.0, 4.0], [0.0, 0.0]])
        np.testing.assert_allclose(np.array(out), expected, atol=1e-5)

    def test_all_same_index(self):
        src = mx.ones((5, 3))
        index = mx.zeros((5,), dtype=mx.int32)
        out = scatter_sum_metal(src, index, 2)
        mx.eval(out)
        np.testing.assert_allclose(np.array(out[0]), [5.0, 5.0, 5.0], atol=1e-5)
        np.testing.assert_allclose(np.array(out[1]), [0.0, 0.0, 0.0], atol=1e-5)

    def test_matches_reference(self):
        """Metal scatter_sum matches at[].add() reference."""
        mx.random.seed(42)
        src = mx.random.normal((1000, 64))
        index = mx.random.randint(0, 100, shape=(1000,)).astype(mx.int32)

        ref = scatter_sum(src, index, 100)
        metal = scatter_sum_metal(src, index, 100)
        mx.eval(ref, metal)
        np.testing.assert_allclose(
            np.array(metal), np.array(ref), atol=1e-4, rtol=1e-4
        )


# ============================================================
# fused_gather_tp_scatter tests
# ============================================================


class TestFusedGatherTPScatter:
    @pytest.fixture
    def graph_data(self):
        """Typical MACE-like graph data for testing."""
        mx.random.seed(42)
        num_atoms = 50
        num_edges = 200
        D = 32

        node_feats = mx.random.normal((num_atoms, D))
        tp_weights = mx.random.normal((num_edges, D))
        edge_attr_0e = mx.random.normal((num_edges,))
        sender = mx.random.randint(0, num_atoms, shape=(num_edges,)).astype(mx.int32)
        receiver = mx.random.randint(0, num_atoms, shape=(num_edges,)).astype(mx.int32)

        return {
            "node_feats": node_feats,
            "tp_weights": tp_weights,
            "edge_attr_0e": edge_attr_0e,
            "sender": sender,
            "receiver": receiver,
            "num_nodes": num_atoms,
        }

    def _reference(self, data):
        """Reference implementation: gather -> multiply -> scatter."""
        mji = (
            data["node_feats"][data["sender"]]
            * data["tp_weights"]
            * data["edge_attr_0e"][:, None]
        )
        return scatter_sum(mji, data["receiver"], data["num_nodes"])

    def test_pure_mlx_matches_reference(self, graph_data):
        ref = self._reference(graph_data)
        result = fused_gather_tp_scatter_scalar(**graph_data)
        mx.eval(ref, result)
        np.testing.assert_allclose(
            np.array(result), np.array(ref), atol=1e-5, rtol=1e-5
        )

    def test_metal_matches_reference(self, graph_data):
        ref = self._reference(graph_data)
        result = fused_gather_tp_scatter_scalar_metal(**graph_data)
        mx.eval(ref, result)
        np.testing.assert_allclose(
            np.array(result), np.array(ref), atol=1e-4, rtol=1e-4
        )

    def test_dispatcher_metal(self, graph_data):
        ref = self._reference(graph_data)
        result = gather_tp_scatter(**graph_data, use_metal=True)
        mx.eval(ref, result)
        np.testing.assert_allclose(
            np.array(result), np.array(ref), atol=1e-4, rtol=1e-4
        )

    def test_dispatcher_pure_mlx(self, graph_data):
        ref = self._reference(graph_data)
        result = gather_tp_scatter(**graph_data, use_metal=False)
        mx.eval(ref, result)
        np.testing.assert_allclose(
            np.array(result), np.array(ref), atol=1e-5, rtol=1e-5
        )

    def test_output_shape(self, graph_data):
        result = gather_tp_scatter(**graph_data, use_metal=True)
        mx.eval(result)
        assert result.shape == (
            graph_data["num_nodes"],
            graph_data["node_feats"].shape[1],
        )

    def test_empty_graph(self):
        """Edge case: no edges."""
        node_feats = mx.random.normal((5, 8))
        tp_weights = mx.zeros((0, 8))
        edge_attr_0e = mx.array([], dtype=mx.float32)
        sender = mx.array([], dtype=mx.int32)
        receiver = mx.array([], dtype=mx.int32)

        result = fused_gather_tp_scatter_scalar(
            node_feats, tp_weights, edge_attr_0e, sender, receiver, 5
        )
        mx.eval(result)
        assert result.shape == (5, 8)
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-10)

    def test_large_graph(self):
        """Stress test with large graph."""
        mx.random.seed(123)
        num_atoms = 1000
        num_edges = 10000
        D = 128

        node_feats = mx.random.normal((num_atoms, D))
        tp_weights = mx.random.normal((num_edges, D))
        edge_attr_0e = mx.random.normal((num_edges,))
        sender = mx.random.randint(0, num_atoms, shape=(num_edges,)).astype(mx.int32)
        receiver = mx.random.randint(0, num_atoms, shape=(num_edges,)).astype(mx.int32)

        ref_mji = node_feats[sender] * tp_weights * edge_attr_0e[:, None]
        ref = scatter_sum(ref_mji, receiver, num_atoms)

        result = fused_gather_tp_scatter_scalar_metal(
            node_feats, tp_weights, edge_attr_0e, sender, receiver, num_atoms
        )
        mx.eval(ref, result)
        np.testing.assert_allclose(
            np.array(result), np.array(ref), atol=1e-4, rtol=1e-4
        )

    def test_pure_mlx_gradient(self):
        """Verify autograd works through the pure-MLX fused path."""
        mx.random.seed(42)
        num_atoms = 10
        num_edges = 20
        D = 8

        node_feats = mx.random.normal((num_atoms, D))
        tp_weights = mx.random.normal((num_edges, D))
        edge_attr_0e = mx.random.normal((num_edges,))
        sender = mx.random.randint(0, num_atoms, shape=(num_edges,)).astype(mx.int32)
        receiver = mx.random.randint(0, num_atoms, shape=(num_edges,)).astype(mx.int32)

        def f(nf):
            result = fused_gather_tp_scatter_scalar(
                nf, tp_weights, edge_attr_0e, sender, receiver, num_atoms
            )
            return result.sum()

        grad_fn = mx.grad(f)
        grads = grad_fn(node_feats)
        mx.eval(grads)
        assert grads.shape == node_feats.shape
        grads_np = np.array(grads)
        assert np.all(np.isfinite(grads_np))

    def test_metal_no_gradient(self):
        """Metal kernel should raise on backward pass."""
        mx.random.seed(42)
        node_feats = mx.random.normal((5, 4))
        tp_weights = mx.random.normal((10, 4))
        edge_attr_0e = mx.random.normal((10,))
        sender = mx.random.randint(0, 5, shape=(10,)).astype(mx.int32)
        receiver = mx.random.randint(0, 5, shape=(10,)).astype(mx.int32)

        def f(nf):
            return fused_gather_tp_scatter_scalar_metal(
                nf, tp_weights, edge_attr_0e, sender, receiver, 5
            ).sum()

        grad_fn = mx.grad(f)
        with pytest.raises(ValueError, match="Not implemented"):
            grads = grad_fn(node_feats)
            mx.eval(grads)


# ============================================================
# Integration: fused kernel in InteractionBlock
# ============================================================


class TestInteractionBlockFused:
    @pytest.fixture
    def scalar_graph(self):
        """Graph with scalar-only irreps (fusible case)."""
        num_atoms = 10
        num_edges = 30
        num_species = 2
        num_radial = 8
        irreps_in = Irreps("8x0e")
        irreps_out = Irreps("8x0e")
        irreps_sh = Irreps("0e + 1o + 2e")

        mx.random.seed(42)
        node_feats = mx.random.normal((num_atoms, irreps_in.dim))
        node_attrs_np = np.zeros((num_atoms, num_species), dtype=np.float32)
        for i in range(num_atoms):
            node_attrs_np[i, i % num_species] = 1.0
        node_attrs = mx.array(node_attrs_np)
        edge_attrs = mx.random.normal((num_edges, irreps_sh.dim))
        edge_feats = mx.random.normal((num_edges, num_radial))
        sender = mx.random.randint(0, num_atoms, shape=(num_edges,)).astype(mx.int32)
        receiver = mx.random.randint(0, num_atoms, shape=(num_edges,)).astype(mx.int32)
        edge_index = mx.stack([sender, receiver])

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
            "num_nodes": num_atoms,
        }

    def test_can_fuse_scalar_only(self, scalar_graph):
        from mace_mlx.blocks import RealAgnosticInteractionBlock

        block = RealAgnosticInteractionBlock(
            irreps_in=scalar_graph["irreps_in"],
            irreps_out=scalar_graph["irreps_out"],
            irreps_sh=scalar_graph["irreps_sh"],
            num_radial=scalar_graph["num_radial"],
            radial_MLP=[16, 16],
            avg_num_neighbors=3.0,
            num_species=scalar_graph["num_species"],
        )
        assert block._can_fuse is True

    def test_cannot_fuse_higher_l(self):
        from mace_mlx.blocks import RealAgnosticInteractionBlock

        block = RealAgnosticInteractionBlock(
            irreps_in=Irreps("4x0e + 4x1o"),
            irreps_out=Irreps("4x0e + 4x1o"),
            irreps_sh=Irreps("0e + 1o + 2e"),
            num_radial=8,
            radial_MLP=[16, 16],
            avg_num_neighbors=3.0,
            num_species=2,
        )
        assert block._can_fuse is False

    def test_fused_matches_standard(self, scalar_graph):
        """Fused kernel produces same result as standard path."""
        from mace_mlx.blocks import RealAgnosticInteractionBlock

        block = RealAgnosticInteractionBlock(
            irreps_in=scalar_graph["irreps_in"],
            irreps_out=scalar_graph["irreps_out"],
            irreps_sh=scalar_graph["irreps_sh"],
            num_radial=scalar_graph["num_radial"],
            radial_MLP=[16, 16],
            avg_num_neighbors=3.0,
            num_species=scalar_graph["num_species"],
        )

        # Standard path
        msg_std, _ = block(
            scalar_graph["node_feats"],
            scalar_graph["node_attrs"],
            scalar_graph["edge_attrs"],
            scalar_graph["edge_feats"],
            scalar_graph["edge_index"],
        )
        mx.eval(msg_std)

        # Fused path
        block.set_fused_kernel(True)
        assert block._use_fused_kernel is True
        msg_fused, _ = block(
            scalar_graph["node_feats"],
            scalar_graph["node_attrs"],
            scalar_graph["edge_attrs"],
            scalar_graph["edge_feats"],
            scalar_graph["edge_index"],
        )
        mx.eval(msg_fused)

        np.testing.assert_allclose(
            np.array(msg_fused), np.array(msg_std), atol=1e-4, rtol=1e-4
        )

    def test_residual_fused_matches_standard(self, scalar_graph):
        """Fused kernel in residual block produces same result."""
        from mace_mlx.blocks import RealAgnosticResidualInteractionBlock

        block = RealAgnosticResidualInteractionBlock(
            irreps_in=scalar_graph["irreps_in"],
            irreps_out=scalar_graph["irreps_out"],
            irreps_sh=scalar_graph["irreps_sh"],
            num_radial=scalar_graph["num_radial"],
            radial_MLP=[16, 16],
            avg_num_neighbors=3.0,
            num_species=scalar_graph["num_species"],
            hidden_irreps=scalar_graph["irreps_in"],
        )

        # Standard path
        msg_std, sc_std = block(
            scalar_graph["node_feats"],
            scalar_graph["node_attrs"],
            scalar_graph["edge_attrs"],
            scalar_graph["edge_feats"],
            scalar_graph["edge_index"],
        )
        mx.eval(msg_std, sc_std)

        # Fused path
        block.set_fused_kernel(True)
        msg_fused, sc_fused = block(
            scalar_graph["node_feats"],
            scalar_graph["node_attrs"],
            scalar_graph["edge_attrs"],
            scalar_graph["edge_feats"],
            scalar_graph["edge_index"],
        )
        mx.eval(msg_fused, sc_fused)

        np.testing.assert_allclose(
            np.array(msg_fused), np.array(msg_std), atol=1e-4, rtol=1e-4
        )
        # Skip connection should be identical (not affected by fusion)
        np.testing.assert_allclose(
            np.array(sc_fused), np.array(sc_std), atol=1e-6
        )

    def test_set_fused_kernel_noop_higher_l(self):
        """set_fused_kernel is a no-op for non-fusible blocks."""
        from mace_mlx.blocks import RealAgnosticInteractionBlock

        block = RealAgnosticInteractionBlock(
            irreps_in=Irreps("4x0e + 4x1o"),
            irreps_out=Irreps("4x0e + 4x1o"),
            irreps_sh=Irreps("0e + 1o + 2e"),
            num_radial=8,
            radial_MLP=[16, 16],
            avg_num_neighbors=3.0,
            num_species=2,
        )
        block.set_fused_kernel(True)
        assert block._use_fused_kernel is False
