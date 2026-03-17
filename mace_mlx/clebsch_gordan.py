"""Clebsch-Gordan coefficients and related coupling utilities.

Pure-numpy implementation for O(3) coupling coefficients used in MACE.
All functions use float64 precision and are cached with lru_cache.

The real Wigner 3-j symbols are computed via the eigenvalue method
(same approach as e3nn) to guarantee matching sign conventions.
"""

from __future__ import annotations

import functools
from math import factorial, sqrt

import numpy as np
from scipy.linalg import eigh, expm

from mace_mlx.irreps import Irrep, Irreps


# ---------------------------------------------------------------------------
# Internal: Real Wigner D-matrices for the eigenvalue method
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _complex_to_real_unitary(l: int) -> np.ndarray:
    """Unitary matrix U from complex to real spherical harmonics basis.

    |Y_real> = U |Y_complex>

    Uses the Condon-Shortley phase convention.
    """
    dim = 2 * l + 1
    U = np.zeros((dim, dim), dtype=np.complex128)
    for m in range(-l, l + 1):
        if m > 0:
            U[l + m, l + m] = (-1) ** m / sqrt(2)
            U[l + m, l - m] = 1.0 / sqrt(2)
        elif m == 0:
            U[l, l] = 1.0
        else:  # m < 0
            U[l + m, l - m] = -1j * (-1) ** abs(m) / sqrt(2)
            U[l + m, l + m] = 1j / sqrt(2)
    return U


@functools.lru_cache(maxsize=None)
def _J_matrix_real(l: int) -> np.ndarray:
    """The generator i*J_y in the real spherical harmonics basis.

    exp(beta * J) gives the real Wigner small-d matrix for y-rotation.
    """
    dim = 2 * l + 1
    # Build i*J_y in complex |l,m> basis
    J_complex = np.zeros((dim, dim), dtype=np.float64)
    for m in range(-l, l + 1):
        if m + 1 <= l:
            J_complex[m + l + 1, m + l] = sqrt(l * (l + 1) - m * (m + 1)) / 2
        if m - 1 >= -l:
            J_complex[m - 1 + l, m + l] = -sqrt(l * (l + 1) - m * (m - 1)) / 2
    # Transform to real basis
    U = _complex_to_real_unitary(l)
    J_real = U @ J_complex @ U.conj().T
    return J_real.real


def _Rz_real(l: int, angle: float) -> np.ndarray:
    """z-rotation matrix in real spherical harmonics basis."""
    dim = 2 * l + 1
    M = np.zeros((dim, dim), dtype=np.float64)
    inds = np.arange(dim)
    reversed_inds = np.arange(dim - 1, -1, -1)
    frequencies = np.arange(l, -l - 1, -1, dtype=np.float64)
    M[inds, reversed_inds] = np.sin(frequencies * angle)
    M[inds, inds] = np.cos(frequencies * angle)
    return M


def _wigner_D_real(l: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Real Wigner D matrix D^l(alpha, beta, gamma)."""
    J = _J_matrix_real(l)
    return _Rz_real(l, alpha) @ expm(beta * J) @ _Rz_real(l, gamma)


# ---------------------------------------------------------------------------
# 1. SU(2) Clebsch-Gordan coefficients (complex basis)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def su2_clebsch_gordan(j1: int, j2: int, j3: int) -> np.ndarray:
    """Compute SU(2) Clebsch-Gordan coefficients C^{j3}_{m1,m2,m3}.

    Uses the explicit Racah formula (sum over k).

    Returns
    -------
    np.ndarray of shape (2*j1+1, 2*j2+1, 2*j3+1), complex128
        C[m1+j1, m2+j2, m3+j3] = <j1,m1; j2,m2 | j3,m3>
    """
    assert isinstance(j1, int) and j1 >= 0
    assert isinstance(j2, int) and j2 >= 0
    assert isinstance(j3, int) and j3 >= 0

    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return np.zeros((2 * j1 + 1, 2 * j2 + 1, 2 * j3 + 1), dtype=np.complex128)

    C = np.zeros((2 * j1 + 1, 2 * j2 + 1, 2 * j3 + 1), dtype=np.complex128)

    for m1 in range(-j1, j1 + 1):
        for m2 in range(-j2, j2 + 1):
            m3 = m1 + m2
            if abs(m3) > j3:
                continue

            prefactor = sqrt(
                (2 * j3 + 1)
                * factorial(j1 + j2 - j3)
                * factorial(j1 - j2 + j3)
                * factorial(-j1 + j2 + j3)
                * factorial(j1 + m1)
                * factorial(j1 - m1)
                * factorial(j2 + m2)
                * factorial(j2 - m2)
                * factorial(j3 + m3)
                * factorial(j3 - m3)
                / factorial(j1 + j2 + j3 + 1)
            )

            s = 0.0
            for k in range(
                max(0, j2 - j3 - m1, j1 + m2 - j3),
                min(j1 + j2 - j3, j1 - m1, j2 + m2) + 1,
            ):
                a1 = k
                a2 = j1 + j2 - j3 - k
                a3 = j1 - m1 - k
                a4 = j2 + m2 - k
                a5 = j3 - j2 + m1 + k
                a6 = j3 - j1 - m2 + k
                if a1 < 0 or a2 < 0 or a3 < 0 or a4 < 0 or a5 < 0 or a6 < 0:
                    continue
                s += (-1) ** k / (
                    factorial(a1)
                    * factorial(a2)
                    * factorial(a3)
                    * factorial(a4)
                    * factorial(a5)
                    * factorial(a6)
                )

            C[m1 + j1, m2 + j2, m3 + j3] = prefactor * s

    return C


# ---------------------------------------------------------------------------
# 2. Change-of-basis matrix: real <-> complex spherical harmonics
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def change_basis_real_to_complex(l: int) -> np.ndarray:
    """Unitary matrix Q: real -> complex spherical harmonics.

    Y_complex = Q @ Y_real

    Returns shape (2l+1, 2l+1) complex128.
    """
    # Q = U^dagger where U is complex-to-real
    return _complex_to_real_unitary(l).conj().T


# ---------------------------------------------------------------------------
# 3. Real Wigner 3-j symbols (matching e3nn sign convention)
# ---------------------------------------------------------------------------

# Random Euler angles used in the eigenvalue method (same as e3nn)
_RANDOM_ANGLES = np.array(
    [
        [4.41301023, 5.56684102, 4.59384642],
        [4.93325116, 6.12697327, 4.14574096],
        [0.53878964, 4.09050444, 5.36539036],
        [2.16017393, 3.48835314, 5.55174441],
        [2.52385107, 0.29089583, 3.90040975],
    ]
)


@functools.lru_cache(maxsize=None)
def _wigner_3j_sorted(l1: int, l2: int, l3: int) -> np.ndarray:
    """Compute real Wigner 3-j symbol for sorted l1 <= l2 <= l3.

    Uses the eigenvalue method: find the 1-D null space of
    (D1 x D2 x D3 - I) summed over random rotations, matching
    the e3nn convention exactly.
    """
    assert l1 <= l2 <= l3
    if l3 > l1 + l2 or l3 < l2 - l1:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1), dtype=np.float64)

    n = (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1)

    B = np.zeros((n, n), dtype=np.float64)
    I = np.eye(n, dtype=np.float64)
    for abc in _RANDOM_ANGLES:
        D1 = _wigner_D_real(l1, *abc)
        D2 = _wigner_D_real(l2, *abc)
        D3 = _wigner_D_real(l3, *abc)
        D = np.einsum("il,jm,kn->ijklmn", D1, D2, D3).reshape(n, n)
        diff = D - I
        B += diff.T @ diff

    eigenvalues, eigenvectors = eigh(B)
    assert eigenvalues[0] < 1e-10, (
        f"Smallest eigenvalue = {eigenvalues[0]} for ({l1},{l2},{l3})"
    )

    Q = eigenvectors[:, 0].reshape(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
    Q[np.abs(Q) < 1e-14] = 0.0

    # Sign convention (matching e3nn):
    # Prefer Q[l1, l2, l3] > 0 (center element, all m=0)
    # If that's zero, use first nonzero element
    center = Q[l1, l2, l3]
    if center != 0:
        if center < 0:
            Q = -Q
    else:
        first_nonzero = next(x for x in Q.flatten() if x != 0)
        if first_nonzero < 0:
            Q = -Q

    return Q


@functools.lru_cache(maxsize=None)
def wigner_3j(l1: int, l2: int, l3: int) -> np.ndarray:
    """Real Wigner 3-j symbols matching e3nn's convention.

    For all valid (l1, l2, l3) with |l1-l2| <= l3 <= l1+l2.
    Unsorted cases are derived from sorted base cases using the same
    transposition rules as e3nn, ensuring sign-consistent results.

    Returns shape (2*l1+1, 2*l2+1, 2*l3+1) float64, with norm 1.
    """
    if l3 < abs(l1 - l2) or l3 > l1 + l2:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1), dtype=np.float64)

    # Apply e3nn's transposition rules from sorted base cases
    sign = (-1) ** (l1 + l2 + l3)

    if l1 <= l2 <= l3:
        return _wigner_3j_sorted(l1, l2, l3).copy()
    if l1 <= l3 <= l2:
        base = _wigner_3j_sorted(l1, l3, l2)
        return np.ascontiguousarray(base.transpose(0, 2, 1) * sign)
    if l2 <= l1 <= l3:
        base = _wigner_3j_sorted(l2, l1, l3)
        return np.ascontiguousarray(base.transpose(1, 0, 2) * sign)
    if l3 <= l2 <= l1:
        base = _wigner_3j_sorted(l3, l2, l1)
        return np.ascontiguousarray(base.transpose(2, 1, 0) * sign)
    if l2 <= l3 <= l1:
        base = _wigner_3j_sorted(l2, l3, l1)
        return np.ascontiguousarray(base.transpose(2, 0, 1))
    if l3 <= l1 <= l2:
        base = _wigner_3j_sorted(l3, l1, l2)
        return np.ascontiguousarray(base.transpose(1, 2, 0))

    raise RuntimeError(f"Unreachable: ({l1}, {l2}, {l3})")


# ---------------------------------------------------------------------------
# 4. SO(3) real Clebsch-Gordan coefficients
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def so3_clebsch_gordan(l1: int, l2: int, l3: int) -> np.ndarray:
    """Real CG coefficients = wigner_3j * sqrt(2*l3+1).

    This is the "component-normalized" real CG coefficient.

    Returns shape (2*l1+1, 2*l2+1, 2*l3+1) float64.
    """
    return wigner_3j(l1, l2, l3) * sqrt(2 * l3 + 1)


# ---------------------------------------------------------------------------
# 5. U_matrix_real — coupling matrices for MACE SymmetricContraction
# ---------------------------------------------------------------------------


def _wigner_nj(
    irrepss: list[Irreps],
    normalization: str = "component",
    filter_ir_mid: list[Irrep] | None = None,
) -> list[tuple[Irrep, object, np.ndarray]]:
    """Compute n-body coupling matrices recursively.

    Mirrors the MACE/e3nn _wigner_nj function using numpy.

    Returns
    -------
    list of (ir_out, path_info, coupling_matrix)
        coupling_matrix shape: (ir_out.dim, *[irreps.dim for irreps in irrepss])
    """
    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        eye = np.eye(irreps.dim, dtype=np.float64)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret.append((ir, sl, eye[sl]))
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                w3j = wigner_3j(ir_out.l, ir_left.l, ir.l)
                if normalization == "component":
                    w3j = w3j * sqrt(ir_out.dim)
                elif normalization == "norm":
                    w3j = w3j * sqrt(ir_left.dim) * sqrt(ir.dim)

                C = np.einsum(
                    "jk,ijl->ikl", C_left.reshape(ir_left.dim, -1), w3j
                )
                C = C.reshape(
                    ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim
                )

                for u in range(mul):
                    E = np.zeros(
                        (
                            ir_out.dim,
                            *(irreps.dim for irreps in irrepss_left),
                            irreps_right.dim,
                        ),
                        dtype=np.float64,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret.append((ir_out, (path_left, sl), E))
            i += mul * ir.dim

    return sorted(ret, key=lambda x: (x[0].l, x[0].p))


def U_matrix_real(
    irreps_in: str | Irreps,
    irreps_out: str | Irreps,
    correlation: int,
    normalization: str = "component",
    filter_ir_mid: list | None = None,
) -> list:
    """Compute U matrices for MACE's SymmetricContraction.

    Mirrors the MACE U_matrix_real function (non-cuequivariance path).

    Parameters
    ----------
    irreps_in : str or Irreps
        Input irreps, e.g. "32x0e + 16x1o".
    irreps_out : str or Irreps
        Output irreps, e.g. "32x0e".
    correlation : int
        Correlation order (body order - 1).
    normalization : str
        "component" (default) or "norm".
    filter_ir_mid : list or None
        Filter for intermediate irreps.

    Returns
    -------
    list
        Alternating [ir_string, U_tensor, ir_string, U_tensor, ...].
        Each U_tensor has shape:
        - For scalar output (ir.dim==1): (*[irreps_in.dim]*correlation, n_paths)
        - Otherwise: (ir.dim, *[irreps_in.dim]*correlation, n_paths)
    """
    irreps_out = Irreps(irreps_out)
    irreps_in = Irreps(irreps_in)
    irrepss = [irreps_in] * correlation

    if filter_ir_mid is not None:
        filter_ir_mid = [Irrep(ir) for ir in filter_ir_mid]

    if correlation == 4 and filter_ir_mid is None:
        filter_ir_mid = [Irrep(i, 1 if i % 2 == 0 else -1) for i in range(12)]

    wigners = _wigner_nj(irrepss, normalization, filter_ir_mid)

    # Group by output irrep, accumulate coupling paths
    current_ir = wigners[0][0] if wigners else None
    out = []
    stack = np.empty(0)

    for ir, _, base_o3 in wigners:
        if ir in [mulir.ir for mulir in irreps_out] and ir == current_ir:
            piece = base_o3.squeeze()[..., np.newaxis]
            stack = (
                np.concatenate((stack, piece), axis=-1)
                if stack.size > 0
                else piece
            )
            last_ir = current_ir
        elif ir in [mulir.ir for mulir in irreps_out] and ir != current_ir:
            if stack.size > 0:
                out.append(repr(last_ir))
                out.append(stack)
            stack = base_o3.squeeze()[..., np.newaxis]
            current_ir = ir
            last_ir = ir
        else:
            current_ir = ir

    try:
        out.append(repr(last_ir))
        out.append(stack)
    except UnboundLocalError:
        # No matching irreps at all
        first_dim = irreps_out.dim
        if first_dim != 1:
            size = (first_dim, *([irreps_in.dim] * correlation), 1)
        else:
            size = (*([irreps_in.dim] * correlation), 1)
        out = [repr(list(irreps_out)[0].ir), np.zeros(size, dtype=np.float64)]

    return out
