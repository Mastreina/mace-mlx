"""Multi-material correctness tests: cross-validate MLX vs PyTorch MACE.

Tests energy and forces for diverse bulk materials, molecules, supercells,
edge cases, and MD trajectory consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk


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
    from mace_mlx.calculators import MACEMLXCalculator
    return MACEMLXCalculator(model_path="small")


@pytest.fixture(scope="module")
def torch_calc():
    """Shared PyTorch MACE-MP-0 small calculator."""
    try:
        from mace.calculators import mace_mp
    except ImportError:
        pytest.skip("mace-torch not installed")
    return mace_mp(model="small", device="cpu", default_dtype="float32")


# ---------------------------------------------------------------------------
# Test: bulk materials
# ---------------------------------------------------------------------------


class TestMaterialsCorrectness:
    """Cross-validate MLX vs PyTorch across diverse bulk materials."""

    @pytest.mark.parametrize("material,crystal,a", [
        ("Si", "diamond", 5.43),
        ("Cu", "fcc", 3.6),
        ("Al", "fcc", 4.05),
        ("Fe", "bcc", 2.87),
        ("Ni", "fcc", 3.52),
        ("Au", "fcc", 4.08),
        ("Ti", "hcp", 2.95),
    ])
    def test_bulk_materials(self, mlx_calc, torch_calc, material, crystal, a):
        if crystal == "hcp":
            atoms = bulk(material, crystal, a=a, c=a * 1.633)
        else:
            atoms = bulk(material, crystal, a=a)

        a_mlx = atoms.copy()
        a_mlx.calc = mlx_calc
        a_torch = atoms.copy()
        a_torch.calc = torch_calc

        e_mlx = a_mlx.get_potential_energy()
        e_torch = a_torch.get_potential_energy()
        f_mlx = a_mlx.get_forces()
        f_torch = a_torch.get_forces()

        np.testing.assert_allclose(
            e_mlx, e_torch, atol=1e-3,
            err_msg=f"{material} energy mismatch",
        )
        np.testing.assert_allclose(
            f_mlx, f_torch, atol=1e-2,
            err_msg=f"{material} forces mismatch",
        )

    @pytest.mark.parametrize("material,crystal,a", [
        ("Si", "diamond", 5.43),
        ("Cu", "fcc", 3.6),
        ("Al", "fcc", 4.05),
        ("Fe", "bcc", 2.87),
        ("Ni", "fcc", 3.52),
        ("Au", "fcc", 4.08),
    ])
    def test_bulk_energy_is_negative(self, mlx_calc, material, crystal, a):
        """Bulk materials at equilibrium should have negative energy."""
        atoms = bulk(material, crystal, a=a)
        atoms.calc = mlx_calc
        e = atoms.get_potential_energy()
        assert e < 0, f"{material}: energy {e:.4f} should be negative"

    @pytest.mark.parametrize("material,crystal,a", [
        ("Si", "diamond", 5.43),
        ("Cu", "fcc", 3.6),
        ("Al", "fcc", 4.05),
    ])
    def test_bulk_forces_near_zero_at_equilibrium(self, mlx_calc, material, crystal, a):
        """Forces on atoms in perfect crystal should be near zero."""
        atoms = bulk(material, crystal, a=a)
        atoms.calc = mlx_calc
        f = atoms.get_forces()
        f_max = np.max(np.abs(f))
        assert f_max < 0.5, (
            f"{material}: max force {f_max:.4f} should be small at equilibrium"
        )


# ---------------------------------------------------------------------------
# Test: molecules
# ---------------------------------------------------------------------------


class TestMolecules:
    """Cross-validate MLX vs PyTorch for isolated molecules."""

    @pytest.mark.parametrize("formula,positions", [
        ("H2O", [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]]),
        ("CH4", [[0, 0, 0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
                 [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629]]),
        ("CO2", [[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]),
        ("NH3", [[0, 0, 0], [0, 0.939, 0.381], [0.813, -0.470, 0.381],
                 [-0.813, -0.470, 0.381]]),
    ])
    def test_molecules(self, mlx_calc, torch_calc, formula, positions):
        atoms = Atoms(formula, positions=positions)

        a_mlx = atoms.copy()
        a_mlx.calc = mlx_calc
        a_torch = atoms.copy()
        a_torch.calc = torch_calc

        np.testing.assert_allclose(
            a_mlx.get_potential_energy(),
            a_torch.get_potential_energy(),
            atol=1e-3,
            err_msg=f"{formula} energy mismatch",
        )
        np.testing.assert_allclose(
            a_mlx.get_forces(),
            a_torch.get_forces(),
            atol=1e-2,
            err_msg=f"{formula} forces mismatch",
        )

    @pytest.mark.parametrize("formula,positions", [
        ("H2O", [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]]),
        ("CH4", [[0, 0, 0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
                 [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629]]),
        ("CO2", [[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]),
    ])
    def test_molecule_force_shapes(self, mlx_calc, formula, positions):
        """Forces shape must match atom count."""
        atoms = Atoms(formula, positions=positions)
        atoms.calc = mlx_calc
        f = atoms.get_forces()
        assert f.shape == (len(atoms), 3)

    @pytest.mark.parametrize("formula,positions", [
        ("H2O", [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]]),
        ("CO2", [[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]),
    ])
    def test_molecule_energy_finite(self, mlx_calc, formula, positions):
        """Energy should be finite for all test molecules."""
        atoms = Atoms(formula, positions=positions)
        atoms.calc = mlx_calc
        e = atoms.get_potential_energy()
        assert np.isfinite(e), f"{formula}: energy {e} is not finite"


# ---------------------------------------------------------------------------
# Test: supercells
# ---------------------------------------------------------------------------


class TestSupercells:
    """Test larger supercells for cross-validation."""

    @pytest.mark.parametrize("n", [2, 3])
    def test_si_supercells(self, mlx_calc, torch_calc, n):
        si = bulk("Si", "diamond", a=5.43) * (n, n, n)
        a_mlx = si.copy()
        a_mlx.calc = mlx_calc
        a_torch = si.copy()
        a_torch.calc = torch_calc

        np.testing.assert_allclose(
            a_mlx.get_potential_energy(),
            a_torch.get_potential_energy(),
            atol=1e-2,
        )
        np.testing.assert_allclose(
            a_mlx.get_forces(),
            a_torch.get_forces(),
            atol=1e-2,
        )

    @pytest.mark.parametrize("n", [2, 3])
    def test_cu_supercells(self, mlx_calc, torch_calc, n):
        cu = bulk("Cu", "fcc", a=3.6) * (n, n, n)
        a_mlx = cu.copy()
        a_mlx.calc = mlx_calc
        a_torch = cu.copy()
        a_torch.calc = torch_calc

        np.testing.assert_allclose(
            a_mlx.get_potential_energy(),
            a_torch.get_potential_energy(),
            atol=1e-2,
        )
        np.testing.assert_allclose(
            a_mlx.get_forces(),
            a_torch.get_forces(),
            atol=1e-2,
        )

    def test_energy_scales_with_atoms(self, mlx_calc):
        """Energy should scale approximately linearly with number of atoms."""
        si1 = bulk("Si", "diamond", a=5.43)
        si8 = bulk("Si", "diamond", a=5.43) * (2, 2, 2)

        si1.calc = mlx_calc
        si8.calc = mlx_calc

        e1 = si1.get_potential_energy()
        e8 = si8.get_potential_energy()

        # 8x more atoms -> energy should be ~8x (within 20%)
        ratio = e8 / e1
        assert abs(ratio - 8.0) < 2.0, (
            f"Energy scaling: E1={e1:.4f}, E8={e8:.4f}, ratio={ratio:.2f}"
        )


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_atom(self, mlx_calc):
        """Single atom should have zero forces."""
        atom = Atoms("Si", positions=[[0, 0, 0]])
        atom.calc = mlx_calc
        e = atom.get_potential_energy()
        f = atom.get_forces()
        assert isinstance(e, float)
        assert f.shape == (1, 3)
        np.testing.assert_allclose(f, 0.0, atol=1e-10)

    def test_two_atoms_far_apart(self, mlx_calc):
        """Two atoms beyond cutoff should have zero interaction forces."""
        atoms = Atoms("SiSi", positions=[[0, 0, 0], [100, 0, 0]])
        atoms.calc = mlx_calc
        f = atoms.get_forces()
        np.testing.assert_allclose(f, 0.0, atol=1e-10)

    def test_distorted_cell(self, mlx_calc, torch_calc):
        """Test with non-cubic cell (sheared)."""
        si = bulk("Si", "diamond", a=5.43)
        cell = si.cell.copy()
        cell[0, 1] = 0.3
        si.set_cell(cell, scale_atoms=True)

        a_mlx = si.copy()
        a_mlx.calc = mlx_calc
        a_torch = si.copy()
        a_torch.calc = torch_calc

        np.testing.assert_allclose(
            a_mlx.get_potential_energy(),
            a_torch.get_potential_energy(),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            a_mlx.get_forces(),
            a_torch.get_forces(),
            atol=1e-2,
        )

    def test_compressed_cell(self, mlx_calc, torch_calc):
        """Test with compressed unit cell."""
        si = bulk("Si", "diamond", a=5.0)  # smaller than equilibrium 5.43

        a_mlx = si.copy()
        a_mlx.calc = mlx_calc
        a_torch = si.copy()
        a_torch.calc = torch_calc

        np.testing.assert_allclose(
            a_mlx.get_potential_energy(),
            a_torch.get_potential_energy(),
            atol=1e-3,
        )

    def test_expanded_cell(self, mlx_calc, torch_calc):
        """Test with expanded unit cell."""
        si = bulk("Si", "diamond", a=5.8)  # larger than equilibrium 5.43

        a_mlx = si.copy()
        a_mlx.calc = mlx_calc
        a_torch = si.copy()
        a_torch.calc = torch_calc

        np.testing.assert_allclose(
            a_mlx.get_potential_energy(),
            a_torch.get_potential_energy(),
            atol=1e-3,
        )

    def test_different_element_single_atoms(self, mlx_calc):
        """Different single atoms should have different energies (different E0)."""
        e_vals = {}
        for elem in ["H", "C", "N", "O", "Si"]:
            atom = Atoms(elem, positions=[[0, 0, 0]])
            atom.calc = mlx_calc
            e_vals[elem] = atom.get_potential_energy()

        # At least some energies should differ
        energies = list(e_vals.values())
        assert len(set(round(e, 6) for e in energies)) > 1, (
            f"All single-atom energies are the same: {e_vals}"
        )


# ---------------------------------------------------------------------------
# Test: MD trajectory consistency
# ---------------------------------------------------------------------------


class TestMDTrajectory:
    """Test MD trajectory consistency."""

    def test_nve_trajectory(self, mlx_calc):
        """Run 5-step NVE MD and verify energy conservation."""
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet
        from ase import units

        si = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
        si.calc = mlx_calc

        MaxwellBoltzmannDistribution(si, temperature_K=300)

        dyn = VelocityVerlet(si, timestep=1.0 * units.fs)

        energies = []
        for _ in range(5):
            dyn.run(1)
            e = si.get_potential_energy() + si.get_kinetic_energy()
            energies.append(e)

        energies = np.array(energies)
        drift = np.max(np.abs(energies - energies[0]))
        assert drift < 0.5, f"Energy drift {drift:.4f} eV in 5 NVE steps"


# ---------------------------------------------------------------------------
# Test: calculator API
# ---------------------------------------------------------------------------


class TestCalculatorAPI:
    """Test calculator interface properties."""

    def test_results_contain_energy(self, mlx_calc):
        si = bulk("Si", "diamond", a=5.43)
        si.calc = mlx_calc
        _ = si.get_potential_energy()
        assert "energy" in si.calc.results

    def test_results_contain_forces(self, mlx_calc):
        si = bulk("Si", "diamond", a=5.43)
        si.calc = mlx_calc
        _ = si.get_forces()
        assert "forces" in si.calc.results

    def test_results_contain_free_energy(self, mlx_calc):
        si = bulk("Si", "diamond", a=5.43)
        si.calc = mlx_calc
        _ = si.get_potential_energy()
        assert "free_energy" in si.calc.results
        assert si.calc.results["free_energy"] == si.calc.results["energy"]

    def test_implemented_properties(self, mlx_calc):
        assert "energy" in mlx_calc.implemented_properties
        assert "forces" in mlx_calc.implemented_properties
        assert "free_energy" in mlx_calc.implemented_properties

    def test_multiple_calls_consistent(self, mlx_calc):
        """Calling get_potential_energy twice gives same result."""
        si = bulk("Si", "diamond", a=5.43)
        si.calc = mlx_calc
        e1 = si.get_potential_energy()
        # Reset to force recalculation
        si.calc.results = {}
        e2 = si.get_potential_energy()
        np.testing.assert_allclose(e1, e2, atol=1e-10)

    def test_copy_atoms_independent(self, mlx_calc):
        """Calculations on copies should be independent."""
        si = bulk("Si", "diamond", a=5.43)

        si1 = si.copy()
        si1.calc = mlx_calc
        e1 = si1.get_potential_energy()

        si2 = si.copy()
        si2.positions[0] += [0.1, 0, 0]
        si2.calc = mlx_calc
        e2 = si2.get_potential_energy()

        assert e1 != e2, "Different positions should give different energies"
