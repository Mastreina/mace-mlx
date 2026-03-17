"""Tests for real spherical harmonics via CG recursion."""

import math

import mlx.core as mx
import numpy as np
import pytest

from mace_mlx.spherical_harmonics import (
    _to_e3nn_basis,
    spherical_harmonics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _e3nn_available() -> bool:
    try:
        import torch  # noqa: F401
        import torch.serialization

        torch.serialization.add_safe_globals([slice])
        from e3nn.o3 import spherical_harmonics as _  # noqa: F401

        return True
    except Exception:
        return False


requires_e3nn = pytest.mark.skipif(
    not _e3nn_available(), reason="e3nn not installed or incompatible torch"
)


def _e3nn_sh(lmax: int, vectors_np: np.ndarray, normalize: bool, normalization: str):
    """Reference e3nn spherical harmonics (returns numpy array)."""
    import torch
    import torch.serialization

    torch.serialization.add_safe_globals([slice])
    from e3nn.o3 import spherical_harmonics as e3nn_sh_fn

    vectors_torch = torch.tensor(vectors_np, dtype=torch.float64)
    ls = list(range(lmax + 1))
    ref = e3nn_sh_fn(ls, vectors_torch, normalize=normalize, normalization=normalization)
    return ref.numpy()


# ---------------------------------------------------------------------------
# Test: output shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lmax", [0, 1, 2, 3, 4, 6, 10])
def test_output_shape(lmax, random_vectors):
    """Output shape must be (..., (lmax+1)^2)."""
    vecs = mx.array(random_vectors.astype(np.float32))
    Y = spherical_harmonics(lmax, vecs)
    assert Y.shape == (100, (lmax + 1) ** 2)


def test_output_shape_batched(rng):
    """Verify batched input shapes."""
    vecs = mx.array(rng.standard_normal((4, 5, 3)).astype(np.float32))
    Y = spherical_harmonics(3, vecs)
    assert Y.shape == (4, 5, 16)


def test_output_shape_single():
    """Single vector input."""
    vecs = mx.array([1.0, 0.0, 0.0]).reshape(1, 3)
    Y = spherical_harmonics(2, vecs)
    assert Y.shape == (1, 9)


# ---------------------------------------------------------------------------
# Test: l=0 is constant
# ---------------------------------------------------------------------------


def test_l0_is_constant(random_vectors):
    """Y_0 = 1 for all vectors (component normalization)."""
    vecs = mx.array(random_vectors.astype(np.float32))
    Y = spherical_harmonics(0, vecs, normalize=True, normalization="component")
    np.testing.assert_allclose(np.array(Y[:, 0]), 1.0, atol=1e-6)


def test_l0_norm_normalization(random_vectors):
    """Y_0 = 1 for norm normalization too (since 2*0+1 = 1)."""
    vecs = mx.array(random_vectors.astype(np.float32))
    Y = spherical_harmonics(0, vecs, normalize=True, normalization="norm")
    np.testing.assert_allclose(np.array(Y[:, 0]), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: l=1 proportional to [y, z, x] (standard m-ordering)
# ---------------------------------------------------------------------------


def test_l1_standard_ordering(random_unit_vectors):
    """l=1 component-normalised SH = sqrt(3) * [y, z, x]."""
    vecs = mx.array(random_unit_vectors.astype(np.float32))
    Y = spherical_harmonics(1, vecs, normalize=False, normalization="component")
    Y_np = np.array(Y)

    sqrt3 = math.sqrt(3)
    expected = sqrt3 * np.stack(
        [random_unit_vectors[:, 1], random_unit_vectors[:, 2], random_unit_vectors[:, 0]],
        axis=-1,
    ).astype(np.float32)

    np.testing.assert_allclose(Y_np[:, 1:4], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Test: component normalization  ||Y_l||^2 = 2l+1  on the sphere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lmax", [1, 2, 3, 4, 6])
def test_component_norm(lmax, rng):
    """||Y_l||^2 averaged over random unit vectors ≈ 2l+1."""
    n = 5000
    vecs_np = rng.standard_normal((n, 3))
    vecs = mx.array(vecs_np.astype(np.float32))
    Y = spherical_harmonics(lmax, vecs, normalize=True, normalization="component")
    Y_np = np.array(Y)

    for l in range(lmax + 1):
        start = l * l
        end = (l + 1) * (l + 1)
        Yl = Y_np[:, start:end]
        avg_norm_sq = np.mean(np.sum(Yl**2, axis=-1))
        np.testing.assert_allclose(avg_norm_sq, 2 * l + 1, rtol=0.05)


# ---------------------------------------------------------------------------
# Test: norm normalization  ||Y_l|| = 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lmax", [1, 2, 3, 5])
def test_norm_normalization(lmax, rng):
    """||Y_l||^2 averaged over random unit vectors ≈ 1 for norm normalization."""
    n = 5000
    vecs_np = rng.standard_normal((n, 3))
    vecs = mx.array(vecs_np.astype(np.float32))
    Y = spherical_harmonics(lmax, vecs, normalize=True, normalization="norm")
    Y_np = np.array(Y)

    for l in range(lmax + 1):
        start = l * l
        end = (l + 1) * (l + 1)
        Yl = Y_np[:, start:end]
        avg_norm_sq = np.mean(np.sum(Yl**2, axis=-1))
        np.testing.assert_allclose(avg_norm_sq, 1.0, rtol=0.05)


# ---------------------------------------------------------------------------
# Test: compare against e3nn
# ---------------------------------------------------------------------------


@requires_e3nn
@pytest.mark.parametrize("lmax", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("normalization", ["component", "norm"])
def test_against_e3nn(lmax, normalization, rng):
    """Cross-validate against e3nn.o3.spherical_harmonics."""
    vectors_np = rng.standard_normal((100, 3))
    vectors_mlx = mx.array(vectors_np.astype(np.float32))

    # Our SH (standard m-ordering)
    ours = spherical_harmonics(
        lmax, vectors_mlx, normalize=True, normalization=normalization
    )
    # Transform to e3nn basis for comparison
    ours_e3nn = _to_e3nn_basis(ours, lmax)
    ours_np = np.array(ours_e3nn)

    # e3nn reference
    ref = _e3nn_sh(lmax, vectors_np, normalize=True, normalization=normalization)

    np.testing.assert_allclose(
        ours_np,
        ref.astype(np.float32),
        atol=1e-4,
        err_msg=f"Mismatch at lmax={lmax}, normalization={normalization}",
    )


@requires_e3nn
def test_against_e3nn_no_normalize(rng):
    """Cross-validate without vector normalization."""
    vectors_np = rng.standard_normal((50, 3))
    vectors_mlx = mx.array(vectors_np.astype(np.float32))
    lmax = 4

    ours = spherical_harmonics(
        lmax, vectors_mlx, normalize=False, normalization="component"
    )
    ours_e3nn = _to_e3nn_basis(ours, lmax)
    ours_np = np.array(ours_e3nn)

    ref = _e3nn_sh(lmax, vectors_np, normalize=False, normalization="component")

    np.testing.assert_allclose(
        ours_np,
        ref.astype(np.float32),
        atol=1e-3,
    )


# ---------------------------------------------------------------------------
# Test: equivariance  Y(Rv) = D(R) Y(v)
# ---------------------------------------------------------------------------


def _random_rotation(rng) -> np.ndarray:
    """Random 3x3 proper rotation matrix via QR decomposition."""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


@pytest.mark.parametrize("lmax", [1, 2, 3, 4])
def test_equivariance(lmax, rng):
    """Verify Y(Rv) = D(R) Y(v) for a random rotation R."""
    from mace_mlx.clebsch_gordan import _wigner_D_real
    from scipy.spatial.transform import Rotation

    R = _random_rotation(rng)

    # Euler angles for the Wigner D matrix.
    # _wigner_D_real uses beta with opposite sign from scipy's ZYZ convention.
    alpha, beta, gamma = Rotation.from_matrix(R).as_euler("ZYZ")

    vectors_np = rng.standard_normal((50, 3))
    rotated_np = vectors_np @ R.T  # Rv for each row vector

    vectors_mlx = mx.array(vectors_np.astype(np.float32))
    rotated_mlx = mx.array(rotated_np.astype(np.float32))

    Y_v = spherical_harmonics(lmax, vectors_mlx, normalize=True)
    Y_Rv = spherical_harmonics(lmax, rotated_mlx, normalize=True)

    Y_v_np = np.array(Y_v)
    Y_Rv_np = np.array(Y_Rv)

    # Apply D(R) block-diagonally to Y(v)
    Y_v_rotated = np.zeros_like(Y_v_np)
    for l in range(lmax + 1):
        D_l = _wigner_D_real(l, alpha, -beta, gamma).astype(np.float32)
        start = l * l
        end = (l + 1) * (l + 1)
        Y_v_rotated[:, start:end] = Y_v_np[:, start:end] @ D_l.T

    np.testing.assert_allclose(
        Y_Rv_np,
        Y_v_rotated,
        atol=1e-4,
        err_msg=f"Equivariance violated at lmax={lmax}",
    )


# ---------------------------------------------------------------------------
# Test: zero vector handling
# ---------------------------------------------------------------------------


def test_zero_vector():
    """Zero-length vector should not produce NaN when normalize=True."""
    vecs = mx.zeros((1, 3))
    Y = spherical_harmonics(3, vecs, normalize=True)
    assert not mx.any(mx.isnan(Y)).item()


# ---------------------------------------------------------------------------
# Test: invalid normalization
# ---------------------------------------------------------------------------


def test_invalid_normalization():
    """Raise on unsupported normalization string."""
    with pytest.raises(ValueError, match="normalization"):
        spherical_harmonics(2, mx.zeros((1, 3)), normalization="integral")
