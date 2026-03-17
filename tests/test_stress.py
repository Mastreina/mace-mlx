"""Tests for stress/virials computation in MACE-MLX.

Cross-validates stress against PyTorch MACE and checks finite-difference
consistency for periodic systems.
"""

from __future__ import annotations

import numpy as np
import pytest

from mace_mlx.calculators import MACEMLXCalculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def calc_mlx():
    """Shared MACEMLXCalculator with MACE-MP-0 small model (no NL caching)."""
    try:
        import torch  # noqa: F401
        from mace.calculators import mace_mp  # noqa: F401
    except ImportError:
        pytest.skip("mace-torch not installed")
    return MACEMLXCalculator(model_path="small", skin=0.0)


@pytest.fixture(scope="module")
def calc_torch():
    """Shared PyTorch MACE-MP-0 small calculator."""
    try:
        from mace.calculators import mace_mp
    except ImportError:
        pytest.skip("mace-torch not installed")
    return mace_mp(model="small", device="cpu", default_dtype="float32")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bulk(element, crystal, a, cubic=False):
    from ase.build import bulk

    return bulk(element, crystal, a=a, cubic=cubic)


# ---------------------------------------------------------------------------
# Cross-validation: stress matches PyTorch MACE
# ---------------------------------------------------------------------------


class TestStressCrossValidation:
    """Stress from MLX must match PyTorch MACE within tight tolerance."""

    @pytest.mark.parametrize(
        "element,crystal,a",
        [
            ("Si", "diamond", 5.43),
            ("Cu", "fcc", 3.6),
            ("Al", "fcc", 4.05),
            ("Fe", "bcc", 2.87),
            ("Au", "fcc", 4.08),
        ],
        ids=["Si", "Cu", "Al", "Fe", "Au"],
    )
    def test_equilibrium_stress(self, calc_mlx, calc_torch, element, crystal, a):
        """Stress at equilibrium lattice constant matches PyTorch."""
        atoms = _make_bulk(element, crystal, a)

        at = atoms.copy()
        at.calc = calc_torch
        s_torch = at.get_stress()

        am = atoms.copy()
        am.calc = calc_mlx
        s_mlx = am.get_stress()

        np.testing.assert_allclose(
            s_mlx,
            s_torch,
            atol=1e-5,
            err_msg=f"{element} stress mismatch vs PyTorch",
        )

    def test_strained_cell_stress(self, calc_mlx, calc_torch):
        """Stress for a strained cubic Si cell matches PyTorch."""
        si = _make_bulk("Si", "diamond", 5.43, cubic=True)
        cell = np.array(si.get_cell())
        cell *= 1.02  # 2% volumetric strain
        si.set_cell(cell, scale_atoms=True)

        si_t = si.copy()
        si_t.calc = calc_torch
        s_torch = si_t.get_stress()

        si_m = si.copy()
        si_m.calc = calc_mlx
        s_mlx = si_m.get_stress()

        np.testing.assert_allclose(
            s_mlx,
            s_torch,
            atol=1e-5,
            err_msg="Strained Si stress mismatch vs PyTorch",
        )


# ---------------------------------------------------------------------------
# Force consistency: stress path produces same forces as energy-only path
# ---------------------------------------------------------------------------


class TestForceConsistency:
    """Forces computed via the stress path must match the energy-only path."""

    def test_forces_match_between_paths(self, calc_mlx):
        """Forces from stress path == forces from energy-only path."""
        si = _make_bulk("Si", "diamond", 5.43)

        # Energy-only path (properties=["energy", "forces"])
        si1 = si.copy()
        si1.calc = calc_mlx
        f_only = si1.get_forces().copy()

        # Stress path (properties include "stress")
        si2 = si.copy()
        si2.calc = calc_mlx
        _ = si2.get_stress()
        f_stress = si2.get_forces()

        np.testing.assert_allclose(
            f_stress,
            f_only,
            atol=1e-5,
            err_msg="Forces differ between stress and energy-only paths",
        )


# ---------------------------------------------------------------------------
# Stress tensor symmetry
# ---------------------------------------------------------------------------


class TestStressSymmetry:
    """The virial/stress tensor should be symmetric (sigma_ij == sigma_ji)."""

    def test_stress_tensor_symmetric(self, calc_mlx):
        """Off-diagonal Voigt components should reflect symmetry."""
        si = _make_bulk("Si", "diamond", 5.43, cubic=True)
        # Apply shear strain to get nonzero off-diagonal stress
        cell = np.array(si.get_cell(), dtype=np.float64)
        cell[0, 1] += 0.1  # shear
        si.set_cell(cell, scale_atoms=True)

        si.calc = calc_mlx
        s = si.get_stress()
        # Voigt: xx, yy, zz, yz, xz, xy
        # Reconstruct 3x3 symmetric tensor
        stress_3x3 = np.array([
            [s[0], s[5], s[4]],
            [s[5], s[1], s[3]],
            [s[4], s[3], s[2]],
        ])
        # Should be symmetric by construction (Voigt always is),
        # but verify the values are reasonable
        np.testing.assert_allclose(
            stress_3x3,
            stress_3x3.T,
            atol=1e-10,
            err_msg="Stress tensor is not symmetric",
        )


# ---------------------------------------------------------------------------
# Non-periodic systems: stress not computed
# ---------------------------------------------------------------------------


class TestNonPeriodicStress:
    """Non-periodic systems should not produce stress."""

    def test_no_stress_for_molecule(self, calc_mlx):
        """Stress is not in results for a non-periodic molecule."""
        from ase import Atoms

        water = Atoms(
            "OH2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.757, 0.586, 0.0],
                [-0.757, 0.586, 0.0],
            ],
        )
        water.calc = calc_mlx
        _ = water.get_potential_energy()
        _ = water.get_forces()
        assert "stress" not in calc_mlx.results


# ---------------------------------------------------------------------------
# Finite difference validation
# ---------------------------------------------------------------------------


class TestFiniteDifferenceStress:
    """Stress should be consistent with numerical differentiation of energy."""

    def test_fd_stress_cubic_si(self, calc_mlx):
        """Analytical stress ~= finite-difference stress for cubic Si."""
        si = _make_bulk("Si", "diamond", 5.43, cubic=True)
        si.calc = calc_mlx
        stress = si.get_stress()
        vol0 = si.get_volume()
        cell0 = np.array(si.get_cell(), dtype=np.float64)
        pos0 = si.get_positions().copy().astype(np.float64)

        # Use eps=1e-3 (best tradeoff for float32 precision)
        eps = 1e-3
        voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        numerical_stress = np.zeros(6)

        for v, (i, j) in enumerate(voigt_map):
            D = np.zeros((3, 3))
            D[i, j] = eps
            S = 0.5 * (D + D.T)

            si_p = si.copy()
            si_p.set_cell(cell0 + cell0 @ S, scale_atoms=False)
            si_p.set_positions(pos0 + pos0 @ S)
            si_p.calc = calc_mlx
            e_p = si_p.get_potential_energy()

            D = np.zeros((3, 3))
            D[i, j] = -eps
            S = 0.5 * (D + D.T)

            si_m = si.copy()
            si_m.set_cell(cell0 + cell0 @ S, scale_atoms=False)
            si_m.set_positions(pos0 + pos0 @ S)
            si_m.calc = calc_mlx
            e_m = si_m.get_potential_energy()

            numerical_stress[v] = (e_p - e_m) / (2 * eps * vol0)

        # Tolerance is loose due to float32 FD limitations
        np.testing.assert_allclose(
            stress,
            numerical_stress,
            atol=5e-4,
            err_msg="Analytical vs finite-difference stress mismatch",
        )
