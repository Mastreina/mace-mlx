"""Tests for radial basis functions and cutoff envelopes."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch

from mace_mlx.radial import (
    BesselBasis,
    GaussianBasis,
    PolynomialCutoff,
    RadialEmbeddingBlock,
    make_radial_mlp,
)


# ---------------------------------------------------------------------------
# 1. BesselBasis — shape and manual computation
# ---------------------------------------------------------------------------
class TestBesselBasis:
    def test_output_shape(self):
        bb = BesselBasis(r_max=6.0, num_basis=8)
        x = mx.array([0.5, 1.0, 2.0, 3.0])
        out = bb(x)
        assert out.shape == (4, 8)

    def test_output_shape_2d(self):
        bb = BesselBasis(r_max=6.0, num_basis=8)
        x = mx.array([[0.5], [1.0], [2.0]])
        out = bb(x)
        assert out.shape == (3, 8)

    def test_values_manual(self):
        r_max = 6.0
        num_basis = 8
        bb = BesselBasis(r_max=r_max, num_basis=num_basis)
        x = mx.array([1.0, 2.0, 3.0])
        out = bb(x)
        mx.eval(out)
        out_np = np.array(out)

        prefactor = math.sqrt(2.0 / r_max)
        for i, xi in enumerate([1.0, 2.0, 3.0]):
            for n in range(num_basis):
                w = (n + 1) * math.pi / r_max
                expected = prefactor * math.sin(w * xi) / xi
                np.testing.assert_allclose(
                    out_np[i, n], expected, atol=1e-6, rtol=1e-5
                )

    def test_x_zero_no_nan(self):
        bb = BesselBasis(r_max=6.0, num_basis=8)
        x = mx.array([0.0])
        out = bb(x)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out)))

    def test_trainable_flag(self):
        bb_frozen = BesselBasis(r_max=6.0, num_basis=8, trainable=False)
        bb_train = BesselBasis(r_max=6.0, num_basis=8, trainable=True)
        # Frozen model should have no trainable parameters
        assert len(bb_frozen.trainable_parameters()) == 0
        # Trainable model should have bessel_weights as trainable
        trainable = bb_train.trainable_parameters()
        assert any(
            p.shape == (8,)
            for leaf in trainable.values()
            for p in (leaf if isinstance(leaf, list) else [leaf])
            if isinstance(p, mx.array)
        )


# ---------------------------------------------------------------------------
# 2. PolynomialCutoff — boundary values and smoothness
# ---------------------------------------------------------------------------
class TestPolynomialCutoff:
    def test_at_zero(self):
        pc = PolynomialCutoff(r_max=6.0, p=6)
        out = pc(mx.array([0.0]))
        mx.eval(out)
        np.testing.assert_allclose(np.array(out)[0], 1.0, atol=1e-7)

    def test_at_r_max(self):
        pc = PolynomialCutoff(r_max=6.0, p=6)
        out = pc(mx.array([6.0]))
        mx.eval(out)
        np.testing.assert_allclose(np.array(out)[0], 0.0, atol=1e-7)

    def test_beyond_r_max(self):
        pc = PolynomialCutoff(r_max=6.0, p=6)
        out = pc(mx.array([7.0, 10.0, 100.0]))
        mx.eval(out)
        np.testing.assert_allclose(np.array(out), 0.0, atol=1e-7)

    def test_smooth_in_between(self):
        pc = PolynomialCutoff(r_max=6.0, p=6)
        x = mx.linspace(0.01, 5.99, 100)
        out = pc(x)
        mx.eval(out)
        out_np = np.array(out)
        # All values should be between 0 and 1
        assert np.all(out_np >= -1e-7)
        assert np.all(out_np <= 1.0 + 1e-7)
        # Should be monotonically decreasing
        diffs = np.diff(out_np)
        assert np.all(diffs <= 1e-6), "Cutoff should be monotonically decreasing"

    def test_different_p_values(self):
        for p in [2, 4, 6, 8]:
            pc = PolynomialCutoff(r_max=5.0, p=p)
            out_0 = pc(mx.array([0.0]))
            out_rmax = pc(mx.array([5.0]))
            mx.eval(out_0, out_rmax)
            np.testing.assert_allclose(np.array(out_0)[0], 1.0, atol=1e-7)
            np.testing.assert_allclose(np.array(out_rmax)[0], 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# 3. RadialEmbeddingBlock — output shape
# ---------------------------------------------------------------------------
class TestRadialEmbeddingBlock:
    def test_output_shape(self):
        block = RadialEmbeddingBlock(r_max=6.0, num_bessel=8, num_polynomial_cutoff=6)
        edges = mx.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
        out = block(edges)
        mx.eval(out)
        assert out.shape == (6, 8)

    def test_output_shape_2d_input(self):
        block = RadialEmbeddingBlock(r_max=6.0, num_bessel=8)
        edges = mx.array([[0.5], [1.0], [2.0]])
        out = block(edges)
        mx.eval(out)
        assert out.shape == (3, 8)

    def test_zero_at_r_max(self):
        block = RadialEmbeddingBlock(r_max=6.0, num_bessel=8, num_polynomial_cutoff=6)
        edges = mx.array([6.0, 7.0])
        out = block(edges)
        mx.eval(out)
        np.testing.assert_allclose(np.array(out), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. RadialMLP — forward pass shape
# ---------------------------------------------------------------------------
class TestRadialMLP:
    def test_forward_shape(self):
        mlp = make_radial_mlp([8, 64, 64, 16])
        x = mx.ones((10, 8))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (10, 16)

    def test_forward_shape_single(self):
        mlp = make_radial_mlp([4, 32, 1])
        x = mx.ones((5, 4))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (5, 1)

    def test_single_layer(self):
        mlp = make_radial_mlp([8, 16])
        x = mx.ones((3, 8))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (3, 16)


# ---------------------------------------------------------------------------
# 5. GaussianBasis
# ---------------------------------------------------------------------------
class TestGaussianBasis:
    def test_output_shape(self):
        gb = GaussianBasis(r_max=6.0, num_basis=128)
        x = mx.array([0.5, 1.0, 3.0])
        out = gb(x)
        mx.eval(out)
        assert out.shape == (3, 128)

    def test_peak_at_center(self):
        gb = GaussianBasis(r_max=6.0, num_basis=7)
        # centers are at 0, 1, 2, 3, 4, 5, 6
        x = mx.array([3.0])
        out = gb(x)
        mx.eval(out)
        out_np = np.array(out)[0]
        # Peak should be at center index 3
        assert np.argmax(out_np) == 3
        np.testing.assert_allclose(out_np[3], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Cross-validation against PyTorch MACE
# ---------------------------------------------------------------------------
class TestCrossValidation:
    """Compare MLX outputs against PyTorch MACE reference implementation."""

    @pytest.fixture
    def distances(self):
        return [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

    @pytest.fixture
    def r_max(self):
        return 6.0

    def test_bessel_vs_torch(self, distances, r_max):
        from mace.modules.radial import BesselBasis as TorchBessel

        num_basis = 8
        mlx_bb = BesselBasis(r_max=r_max, num_basis=num_basis)
        torch_bb = TorchBessel(r_max=r_max, num_basis=num_basis, trainable=False)
        torch_bb.eval()

        x_mlx = mx.array(distances)[:, None]  # (N, 1) as PyTorch expects
        x_torch = torch.tensor(distances, dtype=torch.float32).unsqueeze(-1)

        out_mlx = mlx_bb(x_mlx)
        mx.eval(out_mlx)
        with torch.no_grad():
            out_torch = torch_bb(x_torch)

        np.testing.assert_allclose(
            np.array(out_mlx),
            out_torch.numpy(),
            atol=1e-5,
            rtol=1e-4,
            err_msg="BesselBasis MLX vs PyTorch mismatch",
        )

    def test_cutoff_vs_torch(self, distances, r_max):
        from mace.modules.radial import PolynomialCutoff as TorchCutoff

        mlx_pc = PolynomialCutoff(r_max=r_max, p=6)
        torch_pc = TorchCutoff(r_max=r_max, p=6)
        torch_pc.eval()

        x_mlx = mx.array(distances)
        x_torch = torch.tensor(distances, dtype=torch.float32).unsqueeze(-1)

        out_mlx = mlx_pc(x_mlx)
        mx.eval(out_mlx)
        with torch.no_grad():
            out_torch = torch_pc(x_torch)

        np.testing.assert_allclose(
            np.array(out_mlx),
            out_torch.numpy().squeeze(-1),
            atol=1e-5,
            rtol=1e-4,
            err_msg="PolynomialCutoff MLX vs PyTorch mismatch",
        )

    def test_cutoff_vs_torch_beyond_rmax(self, r_max):
        from mace.modules.radial import PolynomialCutoff as TorchCutoff

        beyond = [6.0, 6.5, 7.0, 10.0]
        mlx_pc = PolynomialCutoff(r_max=r_max, p=6)
        torch_pc = TorchCutoff(r_max=r_max, p=6)
        torch_pc.eval()

        x_mlx = mx.array(beyond)
        x_torch = torch.tensor(beyond, dtype=torch.float32).unsqueeze(-1)

        out_mlx = mlx_pc(x_mlx)
        mx.eval(out_mlx)
        with torch.no_grad():
            out_torch = torch_pc(x_torch)

        np.testing.assert_allclose(
            np.array(out_mlx),
            out_torch.numpy().squeeze(-1),
            atol=1e-7,
            err_msg="PolynomialCutoff beyond r_max mismatch",
        )

    def test_gaussian_vs_torch(self, distances, r_max):
        from mace.modules.radial import GaussianBasis as TorchGaussian

        num_basis = 16
        mlx_gb = GaussianBasis(r_max=r_max, num_basis=num_basis)
        torch_gb = TorchGaussian(r_max=r_max, num_basis=num_basis, trainable=False)
        torch_gb.eval()

        x_mlx = mx.array(distances)[:, None]
        x_torch = torch.tensor(distances, dtype=torch.float32).unsqueeze(-1)

        out_mlx = mlx_gb(x_mlx)
        mx.eval(out_mlx)
        with torch.no_grad():
            out_torch = torch_gb(x_torch)

        np.testing.assert_allclose(
            np.array(out_mlx),
            out_torch.numpy(),
            atol=1e-5,
            rtol=1e-4,
            err_msg="GaussianBasis MLX vs PyTorch mismatch",
        )
