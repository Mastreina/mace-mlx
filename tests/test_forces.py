"""Force consistency tests via finite differences.

Verify that analytical forces (-dE/dr) are consistent with numerical
finite-difference forces for both periodic and non-periodic systems.
"""

from __future__ import annotations

import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk

from mace_mlx.calculators import MACEMLXCalculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mlx_calc():
    """Shared MACEMLXCalculator with MACE-MP-0 small model."""
    try:
        import torch  # noqa: F401
        from mace.calculators import mace_mp  # noqa: F401
    except ImportError:
        pytest.skip("mace-torch not installed")
    return MACEMLXCalculator(model_path="small")


# ---------------------------------------------------------------------------
# Test: finite difference forces (periodic)
# ---------------------------------------------------------------------------


class TestForceConsistencyPeriodic:
    """Verify forces are consistent with energy via finite differences for periodic systems."""

    @pytest.mark.parametrize("material,crystal,a", [
        ("Si", "diamond", 5.43),
        ("Cu", "fcc", 3.6),
    ])
    def test_finite_difference_forces_periodic(self, mlx_calc, material, crystal, a):
        atoms = bulk(material, crystal, a=a)
        atoms.calc = mlx_calc
        analytical_forces = atoms.get_forces()

        dx = 0.001
        numerical_forces = np.zeros_like(analytical_forces)
        for i in range(len(atoms)):
            for j in range(3):
                atoms_plus = atoms.copy()
                atoms_plus.calc = MACEMLXCalculator(model_path="small")
                pos = atoms_plus.get_positions()
                pos[i, j] += dx
                atoms_plus.set_positions(pos)
                e_plus = atoms_plus.get_potential_energy()

                atoms_minus = atoms.copy()
                atoms_minus.calc = MACEMLXCalculator(model_path="small")
                pos = atoms_minus.get_positions()
                pos[i, j] -= dx
                atoms_minus.set_positions(pos)
                e_minus = atoms_minus.get_potential_energy()

                numerical_forces[i, j] = -(e_plus - e_minus) / (2 * dx)

        np.testing.assert_allclose(
            analytical_forces, numerical_forces, atol=1e-2,
            err_msg=f"Forces not consistent with energy for {material}",
        )


# ---------------------------------------------------------------------------
# Test: finite difference forces (non-periodic molecules)
# ---------------------------------------------------------------------------


class TestForceConsistencyMolecule:
    """Verify forces via finite differences for non-periodic molecules."""

    @pytest.mark.parametrize("formula,positions", [
        ("H2O", [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]),
        ("CO2", [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]),
    ])
    def test_finite_difference_forces_molecule(self, mlx_calc, formula, positions):
        atoms = Atoms(formula, positions=positions)
        atoms.calc = mlx_calc
        analytical_forces = atoms.get_forces()

        dx = 0.001
        numerical_forces = np.zeros_like(analytical_forces)
        for i in range(len(atoms)):
            for j in range(3):
                atoms_plus = atoms.copy()
                atoms_plus.calc = MACEMLXCalculator(model_path="small")
                pos = atoms_plus.get_positions()
                pos[i, j] += dx
                atoms_plus.set_positions(pos)
                e_plus = atoms_plus.get_potential_energy()

                atoms_minus = atoms.copy()
                atoms_minus.calc = MACEMLXCalculator(model_path="small")
                pos = atoms_minus.get_positions()
                pos[i, j] -= dx
                atoms_minus.set_positions(pos)
                e_minus = atoms_minus.get_potential_energy()

                numerical_forces[i, j] = -(e_plus - e_minus) / (2 * dx)

        np.testing.assert_allclose(
            analytical_forces, numerical_forces, atol=1e-2,
            err_msg=f"Forces not consistent with energy for {formula}",
        )


# ---------------------------------------------------------------------------
# Test: Newton's third law
# ---------------------------------------------------------------------------


class TestNewtonThirdLaw:
    """Verify that total force sums to zero (translational invariance)."""

    @pytest.mark.parametrize("material,crystal,a", [
        ("Si", "diamond", 5.43),
        ("Cu", "fcc", 3.6),
        ("Al", "fcc", 4.05),
    ])
    def test_total_force_zero_periodic(self, mlx_calc, material, crystal, a):
        """Total force on periodic system should sum to zero."""
        atoms = bulk(material, crystal, a=a)
        atoms.calc = mlx_calc
        f = atoms.get_forces()
        total_force = np.sum(f, axis=0)
        np.testing.assert_allclose(
            total_force, 0.0, atol=1e-4,
            err_msg=f"{material}: total force not zero",
        )

    @pytest.mark.parametrize("formula,positions", [
        ("H2O", [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]),
        ("CH4", [[0, 0, 0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
                 [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629]]),
    ])
    def test_total_force_zero_molecule(self, mlx_calc, formula, positions):
        """Total force on isolated molecule should sum to zero."""
        atoms = Atoms(formula, positions=positions)
        atoms.calc = mlx_calc
        f = atoms.get_forces()
        total_force = np.sum(f, axis=0)
        np.testing.assert_allclose(
            total_force, 0.0, atol=1e-4,
            err_msg=f"{formula}: total force not zero",
        )


# ---------------------------------------------------------------------------
# Test: translational invariance
# ---------------------------------------------------------------------------


class TestTranslationalInvariance:
    """Verify energy is invariant under rigid translations."""

    @pytest.mark.parametrize("shift", [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.5, 0.0],
        [1.0, 1.0, 1.0],
    ])
    def test_translation_invariance_molecule(self, mlx_calc, shift):
        """Energy should not change when all atoms are shifted uniformly."""
        atoms = Atoms("H2O", positions=[
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0],
        ])

        atoms.calc = mlx_calc
        e_original = atoms.get_potential_energy()

        shifted = atoms.copy()
        shifted.positions += np.array(shift)
        shifted.calc = mlx_calc
        e_shifted = shifted.get_potential_energy()

        np.testing.assert_allclose(
            e_original, e_shifted, atol=1e-6,
            err_msg=f"Energy changed after translation by {shift}",
        )

    @pytest.mark.parametrize("shift", [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
    ])
    def test_translation_invariance_forces_molecule(self, mlx_calc, shift):
        """Forces should not change when all atoms are shifted uniformly."""
        atoms = Atoms("H2O", positions=[
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0],
        ])

        atoms.calc = mlx_calc
        f_original = atoms.get_forces()

        shifted = atoms.copy()
        shifted.positions += np.array(shift)
        shifted.calc = mlx_calc
        f_shifted = shifted.get_forces()

        np.testing.assert_allclose(
            f_original, f_shifted, atol=1e-4,
            err_msg=f"Forces changed after translation by {shift}",
        )


# ---------------------------------------------------------------------------
# Test: force continuity
# ---------------------------------------------------------------------------


class TestForceContinuity:
    """Verify forces change smoothly with position."""

    def test_forces_continuous_molecule(self, mlx_calc):
        """Small position change should cause small force change."""
        atoms = Atoms("H2O", positions=[
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0],
        ])
        atoms.calc = mlx_calc
        f0 = atoms.get_forces()

        perturbed = atoms.copy()
        perturbed.positions[0, 0] += 0.001
        perturbed.calc = mlx_calc
        f1 = perturbed.get_forces()

        max_diff = np.max(np.abs(f1 - f0))
        assert max_diff < 1.0, (
            f"Force change too large for 0.001 A perturbation: {max_diff:.4f}"
        )

    def test_forces_continuous_periodic(self, mlx_calc):
        """Small position change in periodic system should cause small force change."""
        si = bulk("Si", "diamond", a=5.43)
        si.calc = mlx_calc
        f0 = si.get_forces()

        perturbed = si.copy()
        perturbed.positions[0, 0] += 0.001
        perturbed.calc = mlx_calc
        f1 = perturbed.get_forces()

        max_diff = np.max(np.abs(f1 - f0))
        assert max_diff < 1.0, (
            f"Force change too large for 0.001 A perturbation: {max_diff:.4f}"
        )
