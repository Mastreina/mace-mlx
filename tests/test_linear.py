"""Tests for EquivariantLinear layer."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mace_mlx.linear import EquivariantLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: mx.array) -> np.ndarray:
    return np.array(x, copy=False)


def _setup_e3nn():
    """Import e3nn with the torch.load safe-globals workaround."""
    import torch.serialization
    torch.serialization.add_safe_globals([slice])
    import torch
    from e3nn.o3 import Linear as E3nnLinear
    return torch, E3nnLinear


# ---------------------------------------------------------------------------
# 1. Cross-validation against e3nn
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "irreps_in, irreps_out",
    [
        ("4x0e + 2x1o", "3x0e + 1x1o"),
        ("2x0e + 1x1o + 1x2e", "3x0e + 2x1o + 1x2e"),
        ("4x0e + 3x0e", "2x0e"),                        # multiple inputs → same output
        ("1x0e", "1x0e"),                                # simplest case
        ("3x1o + 2x2e", "1x1o + 4x2e"),
    ],
)
def test_cross_validate_e3nn(irreps_in: str, irreps_out: str):
    """Copy weights from e3nn Linear into our EquivariantLinear, verify outputs match."""
    torch, E3nnLinear = _setup_e3nn()

    # --- Build e3nn layer ---
    e3nn_lin = E3nnLinear(irreps_in, irreps_out)

    # --- Build our layer ---
    mlx_lin = EquivariantLinear(irreps_in, irreps_out)

    # --- Copy weights ---
    # e3nn stores all weights as a single flat 1D tensor, laid out in
    # instruction order.  Each instruction's block has size mul_in * mul_out.
    flat_w = e3nn_lin.weight.detach().numpy()
    offset = 0
    new_weights = []
    for inst in mlx_lin.instructions:
        numel = inst.mul_in * inst.mul_out
        block = flat_w[offset : offset + numel].reshape(inst.mul_in, inst.mul_out)
        new_weights.append(mx.array(block))
        offset += numel
    assert offset == flat_w.size, (
        f"Weight size mismatch: consumed {offset}, e3nn has {flat_w.size}"
    )
    mlx_lin.weights = new_weights

    # --- Forward pass ---
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((5, mlx_lin.irreps_in.dim)).astype(np.float32)

    x_mlx = mx.array(x_np)
    x_torch = torch.from_numpy(x_np)

    out_mlx = _to_numpy(mlx_lin(x_mlx))
    out_e3nn = e3nn_lin(x_torch).detach().numpy()

    np.testing.assert_allclose(out_mlx, out_e3nn, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "irreps_in, irreps_out, batch_shape",
    [
        ("4x0e + 2x1o", "3x0e + 1x1o", (8,)),
        ("1x0e", "5x0e", (2, 3)),
        ("2x1o + 1x2e", "1x1o + 2x2e", ()),
        ("10x0e", "10x0e", (4,)),
    ],
)
def test_output_shape(irreps_in, irreps_out, batch_shape):
    lin = EquivariantLinear(irreps_in, irreps_out)
    in_dim = lin.irreps_in.dim
    out_dim = lin.irreps_out.dim
    x = mx.random.normal(shape=(*batch_shape, in_dim))
    y = lin(x)
    assert y.shape == (*batch_shape, out_dim)


# ---------------------------------------------------------------------------
# 3. No-matching irreps → output is zero
# ---------------------------------------------------------------------------

def test_no_matching_irreps():
    lin = EquivariantLinear("3x0e", "2x1o")
    x = mx.ones((4, lin.irreps_in.dim))
    y = lin(x)
    np.testing.assert_allclose(_to_numpy(y), 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# 4. Scalar-only case behaves like a standard linear layer
# ---------------------------------------------------------------------------

def test_scalar_is_linear():
    """For 0e-only irreps, equivariant linear should match a plain matrix multiply."""
    lin = EquivariantLinear("4x0e", "3x0e")
    assert len(lin.instructions) == 1
    inst = lin.instructions[0]
    w = _to_numpy(lin.weights[0])  # (4, 3)

    x = mx.random.normal(shape=(5, 4))
    y = lin(x)

    # Expected: path_weight * (x @ w)
    expected = inst.path_weight * (_to_numpy(x) @ w)
    np.testing.assert_allclose(_to_numpy(y), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Equivariance for scalar irreps
# ---------------------------------------------------------------------------

def test_equivariance_scalar():
    """Scalars (0e) are rotation-invariant; f(x) should be the same
    regardless of any "rotation" (which is identity for scalars)."""
    lin = EquivariantLinear("3x0e", "2x0e")
    x = mx.random.normal(shape=(4, 3))
    y1 = lin(x)
    y2 = lin(x)  # same input → same output (deterministic)
    np.testing.assert_allclose(_to_numpy(y1), _to_numpy(y2), atol=1e-8)


# ---------------------------------------------------------------------------
# 6. Weight sharing across m components (equivariance structure check)
# ---------------------------------------------------------------------------

def test_weight_shared_across_m():
    """The weight matrix is shared across the 2l+1 m-components.
    Verify by checking that the operation on each m-component uses
    the same weight."""
    lin = EquivariantLinear("2x1o", "3x1o")
    # 1o has dim=3.  Input dim = 2*3 = 6, Output dim = 3*3 = 9
    assert lin.irreps_in.dim == 6
    assert lin.irreps_out.dim == 9

    # Construct input where only one m-component is nonzero at a time
    # and verify the linear relationship is the same across m.
    w = _to_numpy(lin.weights[0])  # (2, 3)
    pw = lin.instructions[0].path_weight

    for m in range(3):  # m components of 1o
        # Input: put [a, b] in the m-th component of the 2x1o block
        x = np.zeros((1, 6), dtype=np.float32)
        x[0, 0 * 3 + m] = 1.0  # first multiplicity, m-th component
        x[0, 1 * 3 + m] = 2.0  # second multiplicity, m-th component

        y = _to_numpy(lin(mx.array(x)))

        # Expected output at m-th component of each output multiplicity
        inp_vec = np.array([1.0, 2.0])
        expected_out = pw * (w.T @ inp_vec)  # shape (3,)

        for k in range(3):  # output multiplicities
            assert abs(y[0, k * 3 + m] - expected_out[k]) < 1e-6, (
                f"m={m}, k={k}: got {y[0, k*3+m]}, expected {expected_out[k]}"
            )
            # Other m-components should be zero
            for m2 in range(3):
                if m2 != m:
                    assert abs(y[0, k * 3 + m2]) < 1e-8


# ---------------------------------------------------------------------------
# 7. Multiple input blocks contributing to same output
# ---------------------------------------------------------------------------

def test_multiple_inputs_same_output():
    """When multiple input blocks have the same irrep type as an output,
    their contributions should be summed."""
    lin = EquivariantLinear("2x0e + 3x0e", "1x0e")

    # Two instructions: (i_in=0, i_out=0) and (i_in=1, i_out=0)
    assert len(lin.instructions) == 2
    assert lin.instructions[0].i_in == 0
    assert lin.instructions[0].i_out == 0
    assert lin.instructions[1].i_in == 1
    assert lin.instructions[1].i_out == 0

    w0 = _to_numpy(lin.weights[0])  # (2, 1)
    w1 = _to_numpy(lin.weights[1])  # (3, 1)
    pw0 = lin.instructions[0].path_weight
    pw1 = lin.instructions[1].path_weight

    # path_weight should be the same (both go to same output)
    assert abs(pw0 - pw1) < 1e-10
    pw = pw0  # 1/sqrt(2+3) = 1/sqrt(5)
    assert abs(pw - 1.0 / math.sqrt(5)) < 1e-10

    x = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    y = _to_numpy(lin(x))

    # Manual: contribution from first block (2x0e)
    c0 = pw * (np.array([1.0, 2.0]) @ w0)  # shape (1,)
    # contribution from second block (3x0e)
    c1 = pw * (np.array([3.0, 4.0, 5.0]) @ w1)  # shape (1,)
    expected = c0 + c1

    np.testing.assert_allclose(y[0], expected.flatten(), atol=1e-6)


# ---------------------------------------------------------------------------
# 8. Batched forward
# ---------------------------------------------------------------------------

def test_batched_forward():
    """Verify that batched and single-element forward give the same results."""
    lin = EquivariantLinear("3x0e + 2x1o", "2x0e + 1x1o")
    rng = np.random.default_rng(123)
    x_np = rng.standard_normal((4, lin.irreps_in.dim)).astype(np.float32)

    # Batched
    y_batch = _to_numpy(lin(mx.array(x_np)))

    # One by one
    for i in range(4):
        y_single = _to_numpy(lin(mx.array(x_np[i:i+1])))
        np.testing.assert_allclose(y_batch[i], y_single[0], atol=1e-6)


# ---------------------------------------------------------------------------
# 9. Empty irreps
# ---------------------------------------------------------------------------

def test_empty_irreps_out():
    lin = EquivariantLinear("3x0e", "")
    x = mx.ones((2, 3))
    y = lin(x)
    assert y.shape == (2, 0)


def test_empty_irreps_in():
    lin = EquivariantLinear("", "3x0e")
    x = mx.ones((2, 0))
    y = lin(x)
    assert y.shape == (2, 3)
    np.testing.assert_allclose(_to_numpy(y), 0.0, atol=1e-8)
