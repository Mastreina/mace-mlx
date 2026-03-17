"""Tests for SymmetricContraction module.

Validates output shapes, gradient flow, and cross-validates
against the reference MACE PyTorch implementation.
"""

import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import mlx.core as mx
import numpy as np
import pytest
import torch
from e3nn import o3

from mace_mlx.irreps import Irrep, Irreps
from mace_mlx.symmetric_contraction import (
    Contraction,
    SymmetricContraction,
    _build_einsum_strings,
)


# ---------------------------------------------------------------------------
# 1. Einsum string construction
# ---------------------------------------------------------------------------


class TestEinsumStrings:
    """Verify the einsum equation strings match MACE's pattern."""

    def test_corr2_scalar(self):
        main, lower = _build_einsum_strings(correlation=2, ir_out_lmax=0)
        assert main == "wik,ekc,bci,be->bcw"
        assert len(lower) == 1
        assert lower[0][0] == "wk,ekc,be->bcw"
        assert lower[0][1] == "bci,bci->bc"

    def test_corr3_scalar(self):
        main, lower = _build_einsum_strings(correlation=3, ir_out_lmax=0)
        assert main == "wxik,ekc,bci,be->bcwx"
        assert len(lower) == 2
        # nu=2
        assert lower[0][0] == "wxk,ekc,be->bcwx"
        assert lower[0][1] == "bcwi,bci->bcw"
        # nu=1
        assert lower[1][0] == "wk,ekc,be->bcw"
        assert lower[1][1] == "bci,bci->bc"

    def test_corr2_vector(self):
        main, lower = _build_einsum_strings(correlation=2, ir_out_lmax=1)
        assert main == "wxik,ekc,bci,be->bcwx"
        assert len(lower) == 1
        assert lower[0][0] == "wxk,ekc,be->bcwx"
        assert lower[0][1] == "bcwi,bci->bcw"


# ---------------------------------------------------------------------------
# 2. Output shape tests
# ---------------------------------------------------------------------------


class TestOutputShape:
    """Verify output shapes for various configurations."""

    @pytest.mark.parametrize(
        "irreps_in,irreps_out,correlation,num_elements,batch",
        [
            ("2x0e + 2x1o", "2x0e", 2, 3, 5),
            ("4x0e + 4x1o", "4x0e", 2, 5, 10),
            ("2x0e + 2x1o", "2x0e", 3, 3, 5),
            ("2x0e + 2x1o + 2x2e", "2x0e", 2, 4, 8),
            ("4x0e + 4x1o + 4x2e", "4x0e", 3, 3, 6),
        ],
    )
    def test_shape(self, irreps_in, irreps_out, correlation, num_elements, batch):
        sc = SymmetricContraction(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
        )
        irreps_in_obj = Irreps(irreps_in)
        irreps_out_obj = Irreps(irreps_out)

        x = mx.random.normal(shape=(batch, irreps_in_obj.dim))
        y_logits = mx.random.normal(shape=(batch, num_elements))
        y = mx.softmax(y_logits, axis=-1)

        out = sc(x, y)
        num_features = irreps_in_obj.count(Irrep("0e"))
        # Output dim = num_features * irreps_out.num_irreps (one contraction per output irrep)
        # Actually: each contraction produces num_features * ir_out.dim
        # and there is one contraction per MulIr in irreps_out
        expected_dim = sum(
            num_features * mulir.ir.dim for mulir in irreps_out_obj
        )
        assert out.shape == (batch, expected_dim), (
            f"Expected ({batch}, {expected_dim}), got {out.shape}"
        )

    def test_scalar_output_single_element(self):
        sc = SymmetricContraction(
            irreps_in="1x0e + 1x1o",
            irreps_out="1x0e",
            correlation=2,
            num_elements=1,
        )
        x = mx.random.normal(shape=(3, 4))
        y = mx.ones((3, 1))
        out = sc(x, y)
        assert out.shape == (3, 1)


# ---------------------------------------------------------------------------
# 3. Gradient flow tests
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify gradients flow through the contraction."""

    def test_grad_wrt_features(self):
        sc = SymmetricContraction(
            irreps_in="2x0e + 2x1o",
            irreps_out="2x0e",
            correlation=2,
            num_elements=3,
        )
        x = mx.random.normal(shape=(4, 8))
        y = mx.softmax(mx.random.normal(shape=(4, 3)), axis=-1)

        def loss_fn(x_):
            return mx.sum(sc(x_, y))

        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(x)
        assert grads.shape == x.shape
        # Gradients should be finite and not all zero
        assert mx.all(mx.isfinite(grads))
        assert not mx.all(grads == 0)

    def test_grad_wrt_weights(self):
        sc = SymmetricContraction(
            irreps_in="2x0e + 2x1o",
            irreps_out="2x0e",
            correlation=2,
            num_elements=3,
        )
        x = mx.random.normal(shape=(4, 8))
        y = mx.softmax(mx.random.normal(shape=(4, 3)), axis=-1)

        def loss_fn(params):
            sc.update(params)
            return mx.sum(sc(x, y))

        grads = mx.grad(loss_fn)(sc.parameters())

        # Check that at least the contraction weights have gradients
        has_grad = False
        def check_grads(g, prefix=""):
            nonlocal has_grad
            if isinstance(g, dict):
                for k, v in g.items():
                    check_grads(v, f"{prefix}.{k}")
            elif isinstance(g, list):
                for i, v in enumerate(g):
                    check_grads(v, f"{prefix}[{i}]")
            elif isinstance(g, mx.array):
                if mx.any(g != 0):
                    has_grad = True

        check_grads(grads)
        assert has_grad, "No non-zero gradients found in any weight"

    def test_grad_correlation_3(self):
        sc = SymmetricContraction(
            irreps_in="2x0e + 2x1o",
            irreps_out="2x0e",
            correlation=3,
            num_elements=3,
        )
        x = mx.random.normal(shape=(4, 8))
        y = mx.softmax(mx.random.normal(shape=(4, 3)), axis=-1)

        def loss_fn(x_):
            return mx.sum(sc(x_, y))

        grads = mx.grad(loss_fn)(x)
        assert grads.shape == x.shape
        assert mx.all(mx.isfinite(grads))


# ---------------------------------------------------------------------------
# 4. Cross-validation against MACE (the critical test)
# ---------------------------------------------------------------------------


class TestCrossValidationMACE:
    """Compare MLX implementation against MACE PyTorch reference.

    This is the most important test: create a MACE SymmetricContraction,
    copy U matrices and weights to our implementation, feed same input,
    compare output.
    """

    def _create_and_compare(
        self, irreps_in_str, irreps_out_str, correlation, num_elements, batch, seed=42
    ):
        """Helper: build both MACE and MLX contractions, copy weights, compare."""
        from mace.modules.symmetric_contraction import (
            SymmetricContraction as MACESym,
        )

        torch.manual_seed(seed)

        mace_irreps_in = o3.Irreps(irreps_in_str)
        mace_irreps_out = o3.Irreps(irreps_out_str)

        mace_sc = MACESym(
            irreps_in=mace_irreps_in,
            irreps_out=mace_irreps_out,
            correlation=correlation,
            num_elements=num_elements,
        )

        # Build MLX module
        mlx_sc = SymmetricContraction(
            irreps_in=irreps_in_str,
            irreps_out=irreps_out_str,
            correlation=correlation,
            num_elements=num_elements,
        )

        # Copy weights from MACE to MLX for each contraction
        for c_idx, (mace_c, mlx_c) in enumerate(
            zip(mace_sc.contractions, mlx_sc.contractions)
        ):
            # Copy weights_max
            w_max_np = mace_c.weights_max.detach().numpy()
            mlx_c.weights_max = mx.array(w_max_np)

            # Copy lower-order weights
            for w_idx in range(len(mlx_c.weights)):
                w_np = mace_c.weights[w_idx].detach().numpy()
                mlx_c.weights[w_idx] = mx.array(w_np)

            # Verify U matrices match
            for nu in range(1, correlation + 1):
                mace_U = mace_c.U_tensors(nu).numpy()
                mlx_U = np.array(mlx_c._u_matrices[nu])
                np.testing.assert_allclose(
                    mlx_U,
                    mace_U,
                    atol=1e-5,
                    err_msg=f"U_matrix_{nu} mismatch for contraction {c_idx}",
                )

        # Create matching inputs
        num_features = mace_irreps_in.count((0, 1))
        coupling_irreps = o3.Irreps([ir.ir for ir in mace_irreps_in])
        coupling_dim = coupling_irreps.dim

        torch.manual_seed(seed + 1)
        x_torch = torch.randn(batch, num_features, coupling_dim)
        y_torch = torch.nn.functional.softmax(
            torch.randn(batch, num_elements), dim=-1
        )

        x_np = x_torch.numpy()
        y_np = y_torch.numpy()

        # MACE forward
        with torch.no_grad():
            mace_out = mace_sc.contractions[0](x_torch, y_torch)
            mace_outs = []
            for mc in mace_sc.contractions:
                mace_outs.append(mc(x_torch, y_torch))
            mace_full = torch.cat(mace_outs, dim=-1).numpy()

        # MLX forward: input features need to be in flat format for SymmetricContraction
        # But internally it reshapes. We need flat (batch, irreps_in.dim).
        # Actually, the SymmetricContraction.__call__ expects (batch, irreps_in.dim)
        # and reshapes internally. But the MACE contraction directly takes
        # (batch, num_features, coupling_dim). So let's test at the Contraction level too.

        # Test individual contractions directly
        x_mlx = mx.array(x_np)
        y_mlx = mx.array(y_np)

        for c_idx, (mc, mlx_c) in enumerate(
            zip(mace_sc.contractions, mlx_sc.contractions)
        ):
            with torch.no_grad():
                mace_out_c = mc(x_torch, y_torch).numpy()
            mlx_out_c = np.array(mlx_c(x_mlx, y_mlx))
            np.testing.assert_allclose(
                mlx_out_c,
                mace_out_c,
                atol=1e-4,
                rtol=1e-4,
                err_msg=(
                    f"Contraction {c_idx} output mismatch\n"
                    f"MLX: {mlx_out_c[:2]}\n"
                    f"MACE: {mace_out_c[:2]}"
                ),
            )

        # Test full SymmetricContraction
        # For the full module, input is (batch, irreps_in.dim) in irreps-major layout
        # x_np is (batch, num_features, coupling_dim) in feature-major layout
        # We need to convert to irreps-major: for each (mul, ir) block, all copies contiguous
        irreps_in = Irreps(irreps_in_str)
        blocks_irreps = []
        offset = 0
        for mul, ir in irreps_in:
            # In feature-major (batch, num_features, coupling_dim):
            # block data is at x_np[:, :, offset:offset+ir.dim]
            block = x_np[:, :, offset : offset + ir.dim]  # (batch, num_features, ir.dim)
            blocks_irreps.append(block.reshape(batch, mul * ir.dim))
            offset += ir.dim
        x_flat = np.concatenate(blocks_irreps, axis=-1)  # (batch, irreps_in.dim)
        x_flat_mlx = mx.array(x_flat)
        mlx_full = np.array(mlx_sc(x_flat_mlx, y_mlx))

        np.testing.assert_allclose(
            mlx_full,
            mace_full,
            atol=1e-4,
            rtol=1e-4,
            err_msg="Full SymmetricContraction output mismatch",
        )

    def test_corr2_scalar_simple(self):
        """Correlation=2, scalar output, small irreps."""
        self._create_and_compare(
            irreps_in_str="2x0e + 2x1o",
            irreps_out_str="2x0e",
            correlation=2,
            num_elements=3,
            batch=5,
        )

    def test_corr3_scalar_simple(self):
        """Correlation=3, scalar output, small irreps."""
        self._create_and_compare(
            irreps_in_str="2x0e + 2x1o",
            irreps_out_str="2x0e",
            correlation=3,
            num_elements=3,
            batch=5,
        )

    def test_corr2_with_l2(self):
        """Correlation=2, scalar output, includes l=2."""
        self._create_and_compare(
            irreps_in_str="2x0e + 2x1o + 2x2e",
            irreps_out_str="2x0e",
            correlation=2,
            num_elements=4,
            batch=6,
        )

    def test_corr2_more_features(self):
        """Correlation=2, scalar output, more features."""
        self._create_and_compare(
            irreps_in_str="4x0e + 4x1o",
            irreps_out_str="4x0e",
            correlation=2,
            num_elements=3,
            batch=8,
        )

    def test_corr3_more_features(self):
        """Correlation=3, more features, more elements."""
        self._create_and_compare(
            irreps_in_str="4x0e + 4x1o",
            irreps_out_str="4x0e",
            correlation=3,
            num_elements=5,
            batch=8,
        )

    def test_corr2_vector_output(self):
        """Correlation=2, vector (1o) output."""
        self._create_and_compare(
            irreps_in_str="2x0e + 2x1o",
            irreps_out_str="2x1o",
            correlation=2,
            num_elements=3,
            batch=5,
        )

    def test_corr2_mixed_output(self):
        """Correlation=2, mixed scalar + vector output."""
        self._create_and_compare(
            irreps_in_str="2x0e + 2x1o",
            irreps_out_str="2x0e + 2x1o",
            correlation=2,
            num_elements=3,
            batch=5,
        )

    def test_single_element(self):
        """Single chemical element."""
        self._create_and_compare(
            irreps_in_str="2x0e + 2x1o",
            irreps_out_str="2x0e",
            correlation=2,
            num_elements=1,
            batch=4,
        )

    def test_larger_realistic(self):
        """More realistic MACE-like configuration."""
        self._create_and_compare(
            irreps_in_str="8x0e + 8x1o + 8x2e",
            irreps_out_str="8x0e",
            correlation=2,
            num_elements=4,
            batch=10,
        )

    def test_corr3_with_l2(self):
        """Correlation=3 with l=2 input."""
        self._create_and_compare(
            irreps_in_str="2x0e + 2x1o + 2x2e",
            irreps_out_str="2x0e",
            correlation=3,
            num_elements=3,
            batch=5,
        )


# ---------------------------------------------------------------------------
# 5. Edge cases and properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Additional property tests."""

    def test_zero_features_zero_output(self):
        """Zero input features should produce zero output."""
        sc = SymmetricContraction(
            irreps_in="2x0e + 2x1o",
            irreps_out="2x0e",
            correlation=2,
            num_elements=3,
        )
        x = mx.zeros((4, 8))
        y = mx.softmax(mx.random.normal(shape=(4, 3)), axis=-1)
        out = sc(x, y)
        np.testing.assert_allclose(np.array(out), 0.0, atol=1e-7)

    def test_deterministic(self):
        """Same input should give same output (no randomness in forward)."""
        sc = SymmetricContraction(
            irreps_in="2x0e + 2x1o",
            irreps_out="2x0e",
            correlation=2,
            num_elements=3,
        )
        x = mx.random.normal(shape=(4, 8))
        y = mx.softmax(mx.random.normal(shape=(4, 3)), axis=-1)
        out1 = np.array(sc(x, y))
        out2 = np.array(sc(x, y))
        np.testing.assert_array_equal(out1, out2)

    def test_batch_independence(self):
        """Each sample in batch should be processed independently."""
        sc = SymmetricContraction(
            irreps_in="2x0e + 2x1o",
            irreps_out="2x0e",
            correlation=2,
            num_elements=3,
        )
        x = mx.random.normal(shape=(4, 8))
        y = mx.softmax(mx.random.normal(shape=(4, 3)), axis=-1)

        # Process full batch
        out_full = np.array(sc(x, y))

        # Process each sample individually
        for i in range(4):
            out_single = np.array(sc(x[i : i + 1], y[i : i + 1]))
            np.testing.assert_allclose(
                out_single[0],
                out_full[i],
                atol=1e-6,
                err_msg=f"Batch independence violated at sample {i}",
            )

    def test_u_matrices_frozen(self):
        """U matrices should not appear in trainable parameters."""
        import mlx.nn as nn

        sc = SymmetricContraction(
            irreps_in="2x0e + 2x1o",
            irreps_out="2x0e",
            correlation=2,
            num_elements=3,
        )
        params = sc.trainable_parameters()
        leaves = nn.utils.tree_flatten(params)
        param_count = sum(v.size for _, v in leaves if isinstance(v, mx.array))
        assert param_count > 0, "Should have trainable parameters"

        # U matrices should NOT appear in trainable params
        param_keys = [k for k, _ in leaves]
        for key in param_keys:
            assert "_u_matrices" not in key, (
                f"U matrix found in trainable params: {key}"
            )
