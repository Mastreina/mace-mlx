"""Real spherical harmonics via Clebsch-Gordan recursion on MLX.

Computes Y_l^m for l=0..lmax using the recursion:
    Y_l = einsum(Y_{l-1}, Y_1, CG(l-1,1,l)) * sqrt((2l+1)/(3l))

The output uses standard m-ordering (m=-l,...,l) which is consistent
with the CG coefficients in clebsch_gordan.py.  The Wigner-D matrix
for the coordinate-swap rotation (pi, pi/2, 3pi/2) converts to the
e3nn basis when needed for cross-validation.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import numpy as np

from mace_mlx.clebsch_gordan import _wigner_D_real, so3_clebsch_gordan


# ---------------------------------------------------------------------------
# Precomputed coupling tensors (numpy, cached)
# ---------------------------------------------------------------------------

# Euler angles for the standard-to-e3nn basis rotation
_E3NN_ALPHA, _E3NN_BETA, _E3NN_GAMMA = math.pi, math.pi / 2, 3 * math.pi / 2


@lru_cache(maxsize=None)
def _cg_coupling(l: int) -> np.ndarray:
    """CG coupling tensor for l-1,1 -> l with component normalisation factor.

    Returns numpy float32 array of shape (2l-1, 3, 2l+1) ready for einsum.
    The normalisation factor sqrt((2l+1)/(3l)) is baked in.
    """
    cg = so3_clebsch_gordan(l - 1, 1, l)  # float64
    correction = math.sqrt((2 * l + 1) / (3 * l))
    return (cg * correction).astype(np.float32)


_cg_coupling_mx_cache: dict[int, mx.array] = {}


def _cg_coupling_mx(l: int) -> mx.array:
    """Return cached MLX array of CG coupling tensor for l."""
    if l not in _cg_coupling_mx_cache:
        _cg_coupling_mx_cache[l] = mx.array(_cg_coupling(l))
    return _cg_coupling_mx_cache[l]


@lru_cache(maxsize=None)
def _e3nn_rotation_matrix(l: int) -> np.ndarray:
    """Wigner D matrix that rotates standard m-ordering to e3nn ordering.

    Returns numpy float64 array of shape (2l+1, 2l+1).
    """
    return _wigner_D_real(l, _E3NN_ALPHA, _E3NN_BETA, _E3NN_GAMMA)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def spherical_harmonics(
    lmax: int,
    vectors: mx.array,
    normalize: bool = True,
    normalization: str = "component",
) -> mx.array:
    """Compute real spherical harmonics Y_l^m for l=0..lmax.

    Uses CG recursion in standard m-ordering (m=-l,...,l), consistent
    with ``so3_clebsch_gordan`` and ``wigner_3j`` in this package.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum (l goes from 0 to lmax inclusive).
    vectors : mx.array
        Input 3-D vectors, shape ``(..., 3)``.
    normalize : bool
        If True, normalise input vectors to unit length before evaluation.
    normalization : str
        ``"component"`` — each component has variance 1 on the sphere,
        i.e. ``||Y_l||^2 = 2l+1``.  This is the MACE / e3nn default.
        ``"norm"`` — ``||Y_l|| = 1`` on the sphere.

    Returns
    -------
    mx.array
        Shape ``(..., (lmax+1)^2)`` containing all harmonics concatenated.
    """
    if normalization not in ("component", "norm"):
        raise ValueError(
            f"normalization must be 'component' or 'norm', got '{normalization}'"
        )

    # --- normalise input vectors ------------------------------------------
    if normalize:
        r = mx.sqrt(mx.sum(vectors * vectors, axis=-1, keepdims=True))
        r = mx.maximum(r, mx.stop_gradient(mx.array(1e-20)))  # avoid division by zero
        unit = vectors / r
    else:
        unit = vectors

    x = unit[..., 0]
    y = unit[..., 1]
    z = unit[..., 2]

    # --- l = 0 ------------------------------------------------------------
    Y_blocks: list[mx.array] = []
    Y0 = mx.ones_like(x)[..., None]  # (..., 1)
    Y_blocks.append(Y0)

    if lmax == 0:
        return _apply_norm(mx.concatenate(Y_blocks, axis=-1), lmax, normalization)

    # --- l = 1 (standard m-ordering: m=-1 -> y, m=0 -> z, m=+1 -> x) -----
    sqrt3 = math.sqrt(3.0)
    Y1 = mx.stack([sqrt3 * y, sqrt3 * z, sqrt3 * x], axis=-1)  # (..., 3)
    Y_blocks.append(Y1)

    if lmax == 1:
        return _apply_norm(mx.concatenate(Y_blocks, axis=-1), lmax, normalization)

    # --- l >= 2: CG recursion ---------------------------------------------
    Y_prev = Y1
    for l in range(2, lmax + 1):
        cg = mx.stop_gradient(_cg_coupling_mx(l))  # (2l-1, 3, 2l+1)
        Y_new = mx.einsum("...i,...j,ijk->...k", Y_prev, Y1, cg)
        Y_blocks.append(Y_new)
        Y_prev = Y_new

    out = mx.concatenate(Y_blocks, axis=-1)  # (..., (lmax+1)^2)
    return _apply_norm(out, lmax, normalization)


def _apply_norm(out: mx.array, lmax: int, normalization: str) -> mx.array:
    """Convert from component normalisation to the requested one."""
    if normalization == "component":
        return out
    # norm: divide each l-block by sqrt(2l+1)
    scales = []
    for l in range(lmax + 1):
        scales.extend([1.0 / math.sqrt(2 * l + 1)] * (2 * l + 1))
    return out * mx.array(scales, dtype=out.dtype)


# ---------------------------------------------------------------------------
# Utility: convert to e3nn basis (for testing / cross-validation)
# ---------------------------------------------------------------------------


def _to_e3nn_basis(Y: mx.array, lmax: int) -> mx.array:
    """Rotate SH from standard m-ordering to the e3nn basis.

    This applies the Wigner-D matrix for the coordinate-swap rotation
    ``(x,y,z) -> (y,z,x)`` to each l-block independently.

    Used for cross-validation against ``e3nn.o3.spherical_harmonics``.
    """
    blocks = []
    for l in range(lmax + 1):
        start = l * l
        end = (l + 1) * (l + 1)
        Yl = Y[..., start:end]  # (..., 2l+1)
        D = mx.array(_e3nn_rotation_matrix(l).astype(np.float32))  # (2l+1, 2l+1)
        # Y_e3nn = Y_std @ D^T  (D transforms column vectors, we have row layout)
        Yl_e3nn = mx.einsum("...i,ji->...j", Yl, D)
        blocks.append(Yl_e3nn)
    return mx.concatenate(blocks, axis=-1)
