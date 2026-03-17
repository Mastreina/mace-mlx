"""Tests for MACEMLXCalculator: ASE interface and force validation."""

from __future__ import annotations

import numpy as np
import pytest

from mace_mlx.calculators import MACEMLXCalculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def calc_small():
    """Shared MACEMLXCalculator with MACE-MP-0 small model."""
    try:
        import torch  # noqa: F401
        from mace.calculators import mace_mp  # noqa: F401
    except ImportError:
        pytest.skip("mace-torch not installed")
    return MACEMLXCalculator(model_path="small")


@pytest.fixture(scope="module")
def calc_torch_small():
    """Shared PyTorch MACE-MP-0 small calculator."""
    try:
        from mace.calculators import mace_mp
    except ImportError:
        pytest.skip("mace-torch not installed")
    return mace_mp(model="small", device="cpu", default_dtype="float32")


# ---------------------------------------------------------------------------
# Test: water molecule
# ---------------------------------------------------------------------------


class TestWaterMolecule:
    """Compare MACE-MLX vs PyTorch MACE on a water molecule."""

    @staticmethod
    def _make_water():
        from ase import Atoms

        return Atoms(
            "OH2",
            positions=[
                [0.0, 0.0, 0.0],       # O
                [0.757, 0.586, 0.0],    # H
                [-0.757, 0.586, 0.0],   # H
            ],
        )

    def test_energy_and_forces(self, calc_small, calc_torch_small):
        """Energy and forces should match PyTorch within tolerance."""
        water = self._make_water()

        # MLX
        water_mlx = water.copy()
        water_mlx.calc = calc_small
        e_mlx = water_mlx.get_potential_energy()
        f_mlx = water_mlx.get_forces()

        # PyTorch
        water_torch = water.copy()
        water_torch.calc = calc_torch_small
        e_torch = water_torch.get_potential_energy()
        f_torch = water_torch.get_forces()

        print(f"Water energy  MLX={e_mlx:.6f}  Torch={e_torch:.6f}  "
              f"diff={abs(e_mlx - e_torch):.2e}")
        print(f"Water forces max diff={np.max(np.abs(f_mlx - f_torch)):.2e}")

        np.testing.assert_allclose(e_mlx, e_torch, atol=1e-3,
                                   err_msg="Water energy mismatch")
        np.testing.assert_allclose(f_mlx, f_torch, atol=1e-2,
                                   err_msg="Water forces mismatch")


# ---------------------------------------------------------------------------
# Test: bulk silicon (periodic)
# ---------------------------------------------------------------------------


class TestBulkSilicon:
    """Compare MACE-MLX vs PyTorch MACE on periodic bulk Si."""

    @staticmethod
    def _make_si():
        from ase.build import bulk
        return bulk("Si", "diamond", a=5.43)

    def test_energy_and_forces(self, calc_small, calc_torch_small):
        """Energy and forces should match PyTorch for periodic system."""
        si = self._make_si()

        # MLX
        si_mlx = si.copy()
        si_mlx.calc = calc_small
        e_mlx = si_mlx.get_potential_energy()
        f_mlx = si_mlx.get_forces()

        # PyTorch
        si_torch = si.copy()
        si_torch.calc = calc_torch_small
        e_torch = si_torch.get_potential_energy()
        f_torch = si_torch.get_forces()

        print(f"Si energy  MLX={e_mlx:.6f}  Torch={e_torch:.6f}  "
              f"diff={abs(e_mlx - e_torch):.2e}")
        print(f"Si forces max diff={np.max(np.abs(f_mlx - f_torch)):.2e}")

        assert f_mlx.shape == (2, 3)
        assert isinstance(e_mlx, float)

        np.testing.assert_allclose(e_mlx, e_torch, atol=1e-3,
                                   err_msg="Si energy mismatch")
        np.testing.assert_allclose(f_mlx, f_torch, atol=1e-2,
                                   err_msg="Si forces mismatch")


# ---------------------------------------------------------------------------
# Test: force consistency (finite differences)
# ---------------------------------------------------------------------------


class TestForceConsistency:
    """Verify analytical forces are consistent with numerical finite differences."""

    @staticmethod
    def _make_water():
        from ase import Atoms

        return Atoms(
            "OH2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.757, 0.586, 0.0],
                [-0.757, 0.586, 0.0],
            ],
        )

    def test_finite_difference(self, calc_small):
        """Analytical forces ≈ -(E(+dx) - E(-dx)) / (2*dx)."""
        water = self._make_water()
        water.calc = calc_small
        forces = water.get_forces()

        dx = 0.001  # Angstrom
        numerical_forces = np.zeros_like(forces)

        for i in range(len(water)):
            for j in range(3):
                # +dx
                w_plus = water.copy()
                w_plus.calc = calc_small
                pos_p = w_plus.get_positions()
                pos_p[i, j] += dx
                w_plus.set_positions(pos_p)
                e_plus = w_plus.get_potential_energy()

                # -dx
                w_minus = water.copy()
                w_minus.calc = calc_small
                pos_m = w_minus.get_positions()
                pos_m[i, j] -= dx
                w_minus.set_positions(pos_m)
                e_minus = w_minus.get_potential_energy()

                numerical_forces[i, j] = -(e_plus - e_minus) / (2 * dx)

        print(f"Analytical forces:\n{forces}")
        print(f"Numerical forces:\n{numerical_forces}")
        print(f"Max diff: {np.max(np.abs(forces - numerical_forces)):.2e}")

        np.testing.assert_allclose(
            forces, numerical_forces, atol=1e-2,
            err_msg="Analytical vs numerical force mismatch",
        )
