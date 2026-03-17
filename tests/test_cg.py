"""Tests for Clebsch-Gordan coefficients and Wigner 3-j symbols.

Compares against e3nn reference implementation and verifies
mathematical properties (orthogonality, symmetry, normalization).
"""

import os

import numpy as np
import pytest

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from mace_mlx.clebsch_gordan import (
    change_basis_real_to_complex,
    so3_clebsch_gordan,
    su2_clebsch_gordan,
    wigner_3j,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

L_MAX = 3  # Test all l1, l2, l3 up to this value


def _valid_triples(lmax: int):
    """Generate all valid (l1, l2, l3) triples with l1, l2 <= lmax."""
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                yield l1, l2, l3


# ---------------------------------------------------------------------------
# 1. Compare wigner_3j against e3nn for all valid triples
# ---------------------------------------------------------------------------


class TestWigner3jVsE3nn:
    """Compare our wigner_3j against e3nn.o3.wigner_3j."""

    @pytest.fixture(autouse=True)
    def _import_e3nn(self):
        from e3nn.o3 import wigner_3j as e3nn_w3j

        self.e3nn_w3j = e3nn_w3j

    @pytest.mark.parametrize("l1,l2,l3", list(_valid_triples(L_MAX)))
    def test_matches_e3nn(self, l1, l2, l3):
        ours = wigner_3j(l1, l2, l3)
        theirs = self.e3nn_w3j(l1, l2, l3).numpy()

        assert ours.shape == theirs.shape
        np.testing.assert_allclose(
            ours,
            theirs,
            atol=5e-8,
            err_msg=f"wigner_3j({l1},{l2},{l3}) mismatch",
        )

    @pytest.mark.parametrize("l1,l2,l3", list(_valid_triples(L_MAX)))
    def test_normalization(self, l1, l2, l3):
        """||3j||^2 should be 1 for valid triples."""
        w = wigner_3j(l1, l2, l3)
        norm_sq = np.sum(w**2)
        np.testing.assert_allclose(
            norm_sq, 1.0, atol=1e-12, err_msg=f"norm({l1},{l2},{l3})"
        )


# ---------------------------------------------------------------------------
# 2. CG orthogonality
# ---------------------------------------------------------------------------


class TestCGOrthogonality:
    """Verify CG orthogonality relation:

    sum_{m3} CG(l1,m1;l2,m2;l3,m3) * CG(l1,m1';l2,m2';l3,m3)
        = delta(m1,m1') * delta(m2,m2')  (summed over l3 too)

    In matrix form for fixed l3:
    sum_{m3} C[:,:,m3] * C[:,:,m3]^T contracted appropriately.
    """

    @pytest.mark.parametrize("l1,l2", [(l1, l2) for l1 in range(L_MAX + 1) for l2 in range(L_MAX + 1)])
    def test_completeness(self, l1, l2):
        """sum_{l3} sum_{m3} CG[m1,m2,m3] CG[m1',m2',m3] = delta."""
        d1, d2 = 2 * l1 + 1, 2 * l2 + 1
        result = np.zeros((d1, d2, d1, d2), dtype=np.float64)

        for l3 in range(abs(l1 - l2), l1 + l2 + 1):
            cg = so3_clebsch_gordan(l1, l2, l3)
            # cg shape: (d1, d2, d3)
            result += np.einsum("ijk,lmk->ijlm", cg, cg)

        expected = np.einsum("il,jm->ijlm", np.eye(d1), np.eye(d2))
        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-12,
            err_msg=f"CG completeness failed for l1={l1}, l2={l2}",
        )


# ---------------------------------------------------------------------------
# 3. Symmetry properties
# ---------------------------------------------------------------------------


class TestSymmetryProperties:
    """Verify various symmetry properties of Wigner 3-j symbols."""

    @pytest.mark.parametrize("l1,l2,l3", list(_valid_triples(L_MAX)))
    def test_norm_invariance_under_permutation(self, l1, l2, l3):
        """The norm should be 1 regardless of argument order."""
        w = wigner_3j(l1, l2, l3)
        assert np.abs(np.sum(w**2) - 1.0) < 1e-12

    @pytest.mark.parametrize("l1,l2,l3", list(_valid_triples(L_MAX)))
    def test_column_swap_symmetry(self, l1, l2, l3):
        """Swapping first two columns: W(l1,l2,l3) relates to W(l2,l1,l3)
        by transpose(0,1) and phase (-1)^(l1+l2+l3)."""
        w_12 = wigner_3j(l1, l2, l3)
        w_21 = wigner_3j(l2, l1, l3)
        sign = (-1) ** (l1 + l2 + l3)
        np.testing.assert_allclose(
            w_12,
            w_21.transpose(1, 0, 2) * sign,
            atol=5e-8,
            err_msg=f"Column swap symmetry failed for ({l1},{l2},{l3})",
        )

    @pytest.mark.parametrize("l1,l2,l3", list(_valid_triples(L_MAX)))
    def test_cyclic_permutation(self, l1, l2, l3):
        """Cyclic permutation: W(l1,l2,l3) relates to W(l2,l3,l1)."""
        # Only test if all permutations satisfy triangle inequality
        if l1 < abs(l2 - l3) or l1 > l2 + l3:
            return
        w_123 = wigner_3j(l1, l2, l3)
        w_231 = wigner_3j(l2, l3, l1)
        # Cyclic permutation: equivalent to two transpositions, each with phase
        # Net effect: no phase change for cyclic permutation
        np.testing.assert_allclose(
            w_123,
            w_231.transpose(2, 0, 1),
            atol=5e-8,
            err_msg=f"Cyclic permutation failed for ({l1},{l2},{l3})",
        )


# ---------------------------------------------------------------------------
# 4. Complex CG basic properties
# ---------------------------------------------------------------------------


class TestSU2CG:
    """Basic properties of complex SU(2) CG coefficients."""

    def test_trivial_coupling(self):
        """<0,0; l,m | l,m> = 1 for all l, m."""
        for l in range(L_MAX + 1):
            C = su2_clebsch_gordan(0, l, l)
            assert C.shape == (1, 2 * l + 1, 2 * l + 1)
            np.testing.assert_allclose(
                C[0], np.eye(2 * l + 1), atol=1e-14
            )

    def test_m_selection_rule(self):
        """CG is zero unless m1 + m2 = m3."""
        C = su2_clebsch_gordan(1, 1, 2)
        for m1 in range(-1, 2):
            for m2 in range(-1, 2):
                for m3 in range(-2, 3):
                    if m1 + m2 != m3:
                        assert (
                            abs(C[m1 + 1, m2 + 1, m3 + 2]) < 1e-15
                        ), f"Non-zero CG for m1+m2!=m3: ({m1},{m2},{m3})"

    def test_triangle_violation(self):
        """CG should be zero for invalid triangles."""
        C = su2_clebsch_gordan(1, 1, 3)
        np.testing.assert_allclose(C, 0, atol=1e-15)


# ---------------------------------------------------------------------------
# 5. Change of basis matrix properties
# ---------------------------------------------------------------------------


class TestChangeBasis:
    """Properties of the real-to-complex change of basis matrix."""

    @pytest.mark.parametrize("l", range(L_MAX + 1))
    def test_unitarity(self, l):
        Q = change_basis_real_to_complex(l)
        dim = 2 * l + 1
        np.testing.assert_allclose(
            Q @ Q.conj().T, np.eye(dim), atol=1e-14
        )
        np.testing.assert_allclose(
            Q.conj().T @ Q, np.eye(dim), atol=1e-14
        )

    def test_l0(self):
        """For l=0, Q should be identity."""
        Q = change_basis_real_to_complex(0)
        np.testing.assert_allclose(Q, [[1.0]], atol=1e-15)


# ---------------------------------------------------------------------------
# 6. U_matrix_real basic tests
# ---------------------------------------------------------------------------


class TestUMatrix:
    """Basic tests for U_matrix_real."""

    def test_correlation_1_scalar(self):
        """For correlation=1, U is just the identity selector for matching irreps."""
        from mace_mlx.clebsch_gordan import U_matrix_real

        result = U_matrix_real("1x0e", "1x0e", correlation=1)
        assert len(result) >= 2
        # Should contain the irrep label and a tensor
        assert isinstance(result[0], str)
        assert isinstance(result[1], np.ndarray)

    def test_correlation_2_simple(self):
        """Correlation 2 with simple irreps should produce valid U matrices."""
        from mace_mlx.clebsch_gordan import U_matrix_real

        result = U_matrix_real("1x0e + 1x1o", "1x0e", correlation=2)
        assert len(result) >= 2
        U = result[1]
        # U should have shape (irreps_in.dim^2, n_paths)
        assert U.ndim >= 2
        # All values should be finite
        assert np.all(np.isfinite(U))

    def test_u_matrix_matches_mace(self):
        """Compare U_matrix_real with MACE reference for a simple case."""
        from mace_mlx.clebsch_gordan import U_matrix_real

        from mace.tools.cg import U_matrix_real as mace_U_matrix_real

        irreps_in = "1x0e + 1x1o"
        irreps_out = "1x0e"

        for corr in [1, 2, 3]:
            ours = U_matrix_real(irreps_in, irreps_out, correlation=corr)
            theirs = mace_U_matrix_real(irreps_in, irreps_out, correlation=corr)

            # Both return lists of [ir_str, tensor, ...]
            assert len(ours) == len(theirs), (
                f"corr={corr}: len {len(ours)} vs {len(theirs)}"
            )

            for i in range(0, len(ours), 2):
                our_tensor = ours[i + 1]
                their_tensor = theirs[i + 1].numpy()
                np.testing.assert_allclose(
                    our_tensor,
                    their_tensor,
                    atol=1e-6,
                    err_msg=f"U_matrix corr={corr}, ir={ours[i]}",
                )
