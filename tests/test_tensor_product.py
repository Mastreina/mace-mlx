"""Tests for TensorProduct and FullyConnectedTensorProduct.

Validates against e3nn reference implementation, checks selection rules,
output shapes, external weights, and equivariance.
"""

import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import mlx.core as mx
import numpy as np
import pytest
import torch

from mace_mlx.irreps import Irreps
from mace_mlx.tensor_product import (
    FullyConnectedTensorProduct,
    TensorProduct,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_mlx(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().numpy().astype(np.float32))


def _to_numpy(x: mx.array) -> np.ndarray:
    return np.array(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Compare FullyConnectedTensorProduct against e3nn
# ---------------------------------------------------------------------------


class TestFCTPVsE3nn:
    """Cross-validate our FCTP against e3nn's FullyConnectedTensorProduct."""

    @pytest.mark.parametrize(
        "irreps_in1,irreps_in2,irreps_out",
        [
            ("2x0e", "1x0e", "2x0e"),
            ("4x0e + 2x1o", "0e + 1o", "4x0e + 2x1o"),
            ("2x0e + 1x1o", "1x0e + 1x1o", "2x0e + 1x1o"),
            ("3x0e + 2x1o + 1x2e", "1x0e + 1x1o", "3x0e + 2x1o + 1x2e"),
            ("1x0e", "1x0e", "1x0e"),
            ("1x1o", "1x1o", "1x0e + 1x1o + 1x2e"),
        ],
    )
    def test_matches_e3nn(self, irreps_in1, irreps_in2, irreps_out):
        from e3nn.o3 import FullyConnectedTensorProduct as E3nnFCTP

        torch.manual_seed(42)
        e3nn_tp = E3nnFCTP(irreps_in1, irreps_in2, irreps_out)

        our_tp = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)

        # Copy weights from e3nn to ours.
        # e3nn stores all weights in a single flat tensor.
        # Our TensorProduct stores weights as a list in self.tp.weights,
        # one per has_weight instruction, in instruction order.
        e3nn_weight = e3nn_tp.weight.detach().numpy().astype(np.float32)
        offset = 0
        for i, inst in enumerate(our_tp.tp._instructions):
            if inst.has_weight and inst.weight_shape is not None:
                w_size = int(np.prod(inst.weight_shape))
                w_np = e3nn_weight[offset : offset + w_size].reshape(inst.weight_shape)
                widx = our_tp.tp._weight_indices[i]
                our_tp.tp.weights[widx] = mx.array(w_np)
                offset += w_size

        assert offset == len(e3nn_weight), (
            f"Weight count mismatch: consumed {offset}, total {len(e3nn_weight)}"
        )

        # Compare outputs
        torch.manual_seed(123)
        x1_torch = torch.randn(5, e3nn_tp.irreps_in1.dim)
        x2_torch = torch.randn(5, e3nn_tp.irreps_in2.dim)

        e3nn_out = e3nn_tp(x1_torch, x2_torch).detach().numpy()

        x1_mlx = _to_mlx(x1_torch)
        x2_mlx = _to_mlx(x2_torch)
        our_out = _to_numpy(our_tp(x1_mlx, x2_mlx))

        np.testing.assert_allclose(
            our_out,
            e3nn_out,
            atol=1e-5,
            rtol=1e-4,
            err_msg=f"FCTP mismatch for {irreps_in1} x {irreps_in2} -> {irreps_out}",
        )

    def test_instruction_order_matches_e3nn(self):
        """Verify instruction generation order matches e3nn."""
        from e3nn.o3 import FullyConnectedTensorProduct as E3nnFCTP

        irreps_in1, irreps_in2, irreps_out = "4x0e + 2x1o", "0e + 1o", "4x0e + 2x1o"

        e3nn_tp = E3nnFCTP(irreps_in1, irreps_in2, irreps_out)
        our_tp = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)

        assert len(our_tp.tp._instructions) == len(e3nn_tp.instructions)

        for ours, theirs in zip(our_tp.tp._instructions, e3nn_tp.instructions):
            assert ours.i_in1 == theirs.i_in1
            assert ours.i_in2 == theirs.i_in2
            assert ours.i_out == theirs.i_out
            assert ours.connection_mode == theirs.connection_mode
            np.testing.assert_allclose(
                ours.path_weight,
                theirs.path_weight,
                atol=1e-6,
                err_msg=f"path_weight mismatch at ({ours.i_in1},{ours.i_in2},{ours.i_out})",
            )


# ---------------------------------------------------------------------------
# 2. External weights
# ---------------------------------------------------------------------------


class TestExternalWeights:
    """Test TensorProduct with external weights."""

    def test_output_changes_with_weights(self):
        irreps_in1 = "2x0e + 1x1o"
        irreps_in2 = "1x0e + 1x1o"
        irreps_out = "2x0e + 1x1o"

        instructions = []
        ir1 = Irreps(irreps_in1)
        ir2 = Irreps(irreps_in2)
        iro = Irreps(irreps_out)
        for i_out, (mul_out, ir_out) in enumerate(iro):
            for i_in1, (mul_in1, ir_in1) in enumerate(ir1):
                for i_in2, (mul_in2, ir_in2) in enumerate(ir2):
                    if ir_in1.p * ir_in2.p != ir_out.p:
                        continue
                    if not (abs(ir_in1.l - ir_in2.l) <= ir_out.l <= ir_in1.l + ir_in2.l):
                        continue
                    instructions.append((i_in1, i_in2, i_out, "uvw", True))

        tp = TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            internal_weights=False,
        )

        mx.random.seed(42)
        x1 = mx.random.normal(shape=(3, ir1.dim))
        x2 = mx.random.normal(shape=(3, ir2.dim))

        w1 = mx.random.normal(shape=(3, tp.weight_numel))
        w2 = mx.random.normal(shape=(3, tp.weight_numel))

        out1 = tp(x1, x2, weight=w1)
        out2 = tp(x1, x2, weight=w2)

        # Different weights should give different outputs
        diff = mx.abs(out1 - out2).max()
        assert float(diff) > 1e-3, "Different weights should produce different outputs"

    def test_weight_numel_matches(self):
        """weight_numel should match the sum of all weight shapes."""
        instructions = [
            (0, 0, 0, "uvw", True),
            (0, 1, 1, "uvw", True),
        ]
        tp = TensorProduct("3x0e + 2x1o", "1x0e + 1x1o", "3x0e + 2x1o", instructions)
        expected = 3 * 1 * 3 + 3 * 1 * 2  # (mul1*mul2*mul_out) for each
        assert tp.weight_numel == expected

    def test_no_weight_instructions(self):
        """Instructions without weights should not contribute to weight_numel."""
        instructions = [
            (0, 0, 0, "uuu", False),
        ]
        tp = TensorProduct("2x0e", "2x0e", "2x0e", instructions)
        assert tp.weight_numel == 0

        x1 = mx.ones((1, 2))
        x2 = mx.ones((1, 2))
        out = tp(x1, x2)
        assert out.shape == (1, 2)


# ---------------------------------------------------------------------------
# 3. Selection rules
# ---------------------------------------------------------------------------


class TestSelectionRules:
    """Verify only valid (l1, l2, l3) combinations produce output."""

    def test_invalid_triangle_raises(self):
        """Invalid triangle inequality should raise at construction."""
        with pytest.raises(AssertionError):
            TensorProduct("1x0e", "1x0e", "1x2e", [(0, 0, 0, "uvw", True)])

    def test_invalid_parity_raises(self):
        """Mismatched parity should raise at construction."""
        with pytest.raises(AssertionError):
            TensorProduct("1x0e", "1x0e", "1x0o", [(0, 0, 0, "uvw", True)])

    def test_valid_couplings(self):
        """All valid couplings for 1o x 1o should be 0e, 1e, 2e."""
        from mace_mlx.irreps import Irrep

        ir1o = Irrep("1o")
        products = ir1o * ir1o
        expected_ls = {0, 1, 2}
        assert {ir.l for ir in products} == expected_ls
        assert all(ir.p == 1 for ir in products)  # odd * odd = even


# ---------------------------------------------------------------------------
# 4. Output shape
# ---------------------------------------------------------------------------


class TestOutputShape:
    """Test output shapes for various irreps configurations."""

    @pytest.mark.parametrize(
        "irreps_in1,irreps_in2,irreps_out,batch",
        [
            ("1x0e", "1x0e", "1x0e", (1,)),
            ("4x0e + 2x1o", "0e + 1o", "4x0e + 2x1o", (5,)),
            ("3x0e + 2x1o + 1x2e", "1x0e + 1x1o", "3x0e + 2x1o + 1x2e", (10,)),
            ("2x0e", "2x0e", "2x0e", (3, 4)),
        ],
    )
    def test_output_shape(self, irreps_in1, irreps_in2, irreps_out, batch):
        tp = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)
        ir1 = Irreps(irreps_in1)
        ir2 = Irreps(irreps_in2)
        iro = Irreps(irreps_out)

        # Set deterministic weights for shape test
        for i in range(len(tp.tp.weights)):
            tp.tp.weights[i] = mx.ones_like(tp.tp.weights[i])

        x1 = mx.ones((*batch, ir1.dim))
        x2 = mx.ones((*batch, ir2.dim))
        out = tp(x1, x2)
        assert out.shape == (*batch, iro.dim)

    def test_no_batch_dim(self):
        """Single sample without batch dimension."""
        tp = FullyConnectedTensorProduct("1x0e", "1x0e", "1x0e")
        for i in range(len(tp.tp.weights)):
            tp.tp.weights[i] = mx.ones_like(tp.tp.weights[i])
        x1 = mx.ones((1,))
        x2 = mx.ones((1,))
        out = tp(x1, x2)
        assert out.shape == (1,)


# ---------------------------------------------------------------------------
# 5. Connection modes
# ---------------------------------------------------------------------------


class TestConnectionModes:
    """Test all four connection modes produce correct output shapes."""

    def test_uvu_mode(self):
        # uvu: mul_in1 must equal mul_out
        instructions = [(0, 0, 0, "uvu", True)]
        tp = TensorProduct(
            "3x0e", "2x0e", "3x0e",
            instructions, internal_weights=True,
        )
        x1 = mx.ones((1, 3))
        x2 = mx.ones((1, 2))
        out = tp(x1, x2)
        assert out.shape == (1, 3)

    def test_uuu_mode(self):
        instructions = [(0, 0, 0, "uuu", True)]
        tp = TensorProduct(
            "3x0e", "3x0e", "3x0e",
            instructions, internal_weights=True,
        )
        x1 = mx.ones((1, 3))
        x2 = mx.ones((1, 3))
        out = tp(x1, x2)
        assert out.shape == (1, 3)

    def test_uuw_mode(self):
        # uuw: mul_in1 must equal mul_in2
        instructions = [(0, 0, 0, "uuw", True)]
        tp = TensorProduct(
            "3x0e", "3x0e", "2x0e",
            instructions, internal_weights=True,
        )
        x1 = mx.ones((1, 3))
        x2 = mx.ones((1, 3))
        out = tp(x1, x2)
        assert out.shape == (1, 2)

    def test_uvw_mode(self):
        instructions = [(0, 0, 0, "uvw", True)]
        tp = TensorProduct(
            "3x0e", "2x0e", "4x0e",
            instructions, internal_weights=True,
        )
        x1 = mx.ones((1, 3))
        x2 = mx.ones((1, 2))
        out = tp(x1, x2)
        assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# 6. Equivariance test
# ---------------------------------------------------------------------------


class TestEquivariance:
    """Test that TP(D1 @ x1, D2 @ x2) = D3 @ TP(x1, x2) for rotations."""

    def _wigner_D(self, l: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
        """Real Wigner D-matrix using our CG module."""
        from mace_mlx.clebsch_gordan import _wigner_D_real

        return _wigner_D_real(l, alpha, beta, gamma)

    def _rotate_irreps(self, x: np.ndarray, irreps: Irreps, alpha, beta, gamma):
        """Apply rotation to an irreps feature vector."""
        out = np.zeros_like(x)
        slices = irreps.slices
        for i, (mul, ir) in enumerate(irreps):
            D = self._wigner_D(ir.l, alpha, beta, gamma)
            block = x[..., slices[i]].reshape(*x.shape[:-1], mul, ir.dim)
            rotated = np.einsum("ij,...kj->...ki", D, block)
            out[..., slices[i]] = rotated.reshape(*x.shape[:-1], mul * ir.dim)
        return out

    def test_equivariance_scalar_output(self):
        """Scalar output should be rotation-invariant: TP(Dx1, Dx2) = TP(x1,x2)."""
        tp = FullyConnectedTensorProduct("2x1o", "1x1o", "2x0e")
        for i in range(len(tp.tp.weights)):
            tp.tp.weights[i] = mx.ones_like(tp.tp.weights[i])

        np.random.seed(42)
        x1 = np.random.randn(5, tp.irreps_in1.dim).astype(np.float32)
        x2 = np.random.randn(5, tp.irreps_in2.dim).astype(np.float32)

        alpha, beta, gamma = 1.23, 0.78, 2.45

        out_original = _to_numpy(tp(mx.array(x1), mx.array(x2)))

        x1_rot = self._rotate_irreps(x1, tp.irreps_in1, alpha, beta, gamma).astype(np.float32)
        x2_rot = self._rotate_irreps(x2, tp.irreps_in2, alpha, beta, gamma).astype(np.float32)
        out_rotated_input = _to_numpy(tp(mx.array(x1_rot), mx.array(x2_rot)))

        np.testing.assert_allclose(
            out_original,
            out_rotated_input,
            atol=1e-5,
            err_msg="Scalar output should be rotation-invariant",
        )

    def test_equivariance_vector_output(self):
        """Vector output: TP(Dx1, Dx2) = D @ TP(x1, x2)."""
        tp = FullyConnectedTensorProduct("2x0e + 1x1o", "1x0e + 1x1o", "2x0e + 1x1o")
        for i in range(len(tp.tp.weights)):
            tp.tp.weights[i] = mx.ones_like(tp.tp.weights[i])

        np.random.seed(42)
        x1 = np.random.randn(5, tp.irreps_in1.dim).astype(np.float32)
        x2 = np.random.randn(5, tp.irreps_in2.dim).astype(np.float32)

        alpha, beta, gamma = 1.23, 0.78, 2.45

        out_original = _to_numpy(tp(mx.array(x1), mx.array(x2)))

        x1_rot = self._rotate_irreps(x1, tp.irreps_in1, alpha, beta, gamma).astype(np.float32)
        x2_rot = self._rotate_irreps(x2, tp.irreps_in2, alpha, beta, gamma).astype(np.float32)
        out_rotated_input = _to_numpy(tp(mx.array(x1_rot), mx.array(x2_rot)))

        out_rotate_output = self._rotate_irreps(
            out_original, tp.irreps_out, alpha, beta, gamma
        )

        np.testing.assert_allclose(
            out_rotated_input,
            out_rotate_output,
            atol=1e-4,
            err_msg="TP(D@x1, D@x2) should equal D@TP(x1, x2)",
        )
