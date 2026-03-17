"""Realistic systems tests: defects, surfaces, interfaces, and disordered structures.

Cross-validates MLX vs PyTorch MACE on physically realistic and numerically
challenging systems including point defects, surfaces, adsorbates, strained
crystals, disordered alloys, and mixed periodicity edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from ase import Atoms, Atom
from ase.build import bulk, fcc111, fcc100, add_adsorbate


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
# Helpers
# ---------------------------------------------------------------------------


def _compare(atoms, mlx_calc, torch_calc, e_atol=1e-3, f_atol=1e-3,
             label=""):
    """Run energy/forces on both calculators and assert close."""
    a_mlx = atoms.copy()
    a_mlx.calc = mlx_calc
    a_torch = atoms.copy()
    a_torch.calc = torch_calc

    e_mlx = a_mlx.get_potential_energy()
    e_torch = a_torch.get_potential_energy()
    f_mlx = a_mlx.get_forces()
    f_torch = a_torch.get_forces()

    np.testing.assert_allclose(
        e_mlx, e_torch, atol=e_atol,
        err_msg=f"{label} energy mismatch (MLX={e_mlx:.6f}, Torch={e_torch:.6f})",
    )
    np.testing.assert_allclose(
        f_mlx, f_torch, atol=f_atol,
        err_msg=f"{label} forces mismatch (max diff={np.max(np.abs(f_mlx - f_torch)):.2e})",
    )


# ---------------------------------------------------------------------------
# 1. Point defects
# ---------------------------------------------------------------------------


class TestDefects:
    """Cross-validate defect structures: vacancies and interstitials."""

    def test_si_vacancy(self, mlx_calc, torch_calc):
        """Si 3x3x3 diamond with one vacancy (53 atoms)."""
        si_vac = bulk("Si", "diamond", a=5.43) * (3, 3, 3)
        del si_vac[0]
        assert len(si_vac) == 53
        _compare(si_vac, mlx_calc, torch_calc, label="Si vacancy")

    def test_cu_vacancy(self, mlx_calc, torch_calc):
        """Cu 3x3x3 FCC with one vacancy (26 atoms)."""
        cu_vac = bulk("Cu", "fcc", a=3.6) * (3, 3, 3)
        del cu_vac[0]
        assert len(cu_vac) == 26
        _compare(cu_vac, mlx_calc, torch_calc, label="Cu vacancy")

    def test_fe_c_interstitial(self, mlx_calc, torch_calc):
        """Fe 3x3x3 BCC with C interstitial at octahedral site (28 atoms)."""
        fe_inter = bulk("Fe", "bcc", a=2.87) * (3, 3, 3)
        fe_inter.append(Atom("C", position=fe_inter.cell.diagonal() / 2 * 0.33))
        assert len(fe_inter) == 28
        _compare(fe_inter, mlx_calc, torch_calc, label="Fe-C interstitial")


# ---------------------------------------------------------------------------
# 2. Surfaces (slabs)
# ---------------------------------------------------------------------------


class TestSurfaces:
    """Cross-validate surface slab structures."""

    def test_cu111(self, mlx_calc, torch_calc):
        """Cu(111) 3x3x3 slab with 10 A vacuum."""
        cu111 = fcc111("Cu", size=(3, 3, 3), vacuum=10.0)
        _compare(cu111, mlx_calc, torch_calc, label="Cu(111)")

    def test_cu100(self, mlx_calc, torch_calc):
        """Cu(100) 3x3x3 slab with 10 A vacuum."""
        cu100 = fcc100("Cu", size=(3, 3, 3), vacuum=10.0)
        _compare(cu100, mlx_calc, torch_calc, label="Cu(100)")

    def test_si100(self, mlx_calc, torch_calc):
        """Si(100) 2x2x3 slab with 10 A vacuum."""
        from ase.build import diamond100
        si100 = diamond100("Si", size=(2, 2, 3), vacuum=10.0)
        _compare(si100, mlx_calc, torch_calc, label="Si(100)")

    def test_al111(self, mlx_calc, torch_calc):
        """Al(111) 3x3x4 slab with 10 A vacuum."""
        al111 = fcc111("Al", size=(3, 3, 4), vacuum=10.0)
        _compare(al111, mlx_calc, torch_calc, label="Al(111)")


# ---------------------------------------------------------------------------
# 3. Adsorbates on surfaces
# ---------------------------------------------------------------------------


class TestAdsorbates:
    """Cross-validate adsorbate-on-surface systems."""

    def test_co_on_cu111(self, mlx_calc, torch_calc):
        """CO on Cu(111) surface — ontop site."""
        slab = fcc111("Cu", size=(3, 3, 3), vacuum=10.0)
        add_adsorbate(slab, "C", height=1.9, position="ontop")
        add_adsorbate(slab, "O", height=3.1, position="ontop")
        _compare(slab, mlx_calc, torch_calc, label="CO/Cu(111)")

    def test_h_on_al111(self, mlx_calc, torch_calc):
        """H on Al(111) surface — fcc hollow site."""
        al_slab = fcc111("Al", size=(3, 3, 3), vacuum=10.0)
        add_adsorbate(al_slab, "H", height=1.5, position="fcc")
        _compare(al_slab, mlx_calc, torch_calc, label="H/Al(111)")


# ---------------------------------------------------------------------------
# 4. Distorted / strained structures
# ---------------------------------------------------------------------------


class TestDistorted:
    """Cross-validate distorted and strained crystal structures."""

    def test_random_perturbation(self, mlx_calc, torch_calc):
        """Si 3x3x3 diamond with random perturbations (0.1 A RMS)."""
        si_perturbed = bulk("Si", "diamond", a=5.43) * (3, 3, 3)
        rng = np.random.default_rng(42)
        si_perturbed.positions += rng.normal(0, 0.1, si_perturbed.positions.shape)
        _compare(si_perturbed, mlx_calc, torch_calc,
                 label="Si perturbed")

    def test_compressed(self, mlx_calc, torch_calc):
        """Compressed Si crystal (a=5.0 vs equilibrium 5.43)."""
        si_compressed = bulk("Si", "diamond", a=5.0)
        _compare(si_compressed, mlx_calc, torch_calc,
                 label="Si compressed")

    def test_expanded(self, mlx_calc, torch_calc):
        """Expanded Si crystal (a=6.0 vs equilibrium 5.43)."""
        si_expanded = bulk("Si", "diamond", a=6.0)
        _compare(si_expanded, mlx_calc, torch_calc,
                 label="Si expanded")

    def test_sheared(self, mlx_calc, torch_calc):
        """Si 2x2x2 diamond with large shear deformation."""
        si_shear = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
        cell = si_shear.cell.copy()
        cell[0, 1] = 1.0  # large shear
        si_shear.set_cell(cell, scale_atoms=True)
        _compare(si_shear, mlx_calc, torch_calc, label="Si sheared")


# ---------------------------------------------------------------------------
# 5. Disordered / amorphous-like structures
# ---------------------------------------------------------------------------


class TestDisordered:
    """Cross-validate disordered and multi-component structures."""

    def test_random_alloy(self, mlx_calc, torch_calc):
        """CuNi random alloy — 27-atom FCC with random Ni substitutions."""
        cu_ni = bulk("Cu", "fcc", a=3.56) * (3, 3, 3)
        rng = np.random.default_rng(123)
        for i in rng.choice(27, 13, replace=False):
            cu_ni.symbols[i] = "Ni"
        _compare(cu_ni, mlx_calc, torch_calc, label="CuNi alloy")

    def test_liquid_like(self, mlx_calc, torch_calc):
        """Heavily perturbed Si positions (liquid-like snapshot)."""
        si_liquid = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
        rng = np.random.default_rng(99)
        si_liquid.positions += rng.normal(0, 0.5, si_liquid.positions.shape)
        _compare(si_liquid, mlx_calc, torch_calc, label="Si liquid-like")

    def test_high_entropy_alloy(self, mlx_calc, torch_calc):
        """CuNiAlFe high-entropy alloy — 27-atom FCC, 4 element types."""
        hea = bulk("Cu", "fcc", a=3.6) * (3, 3, 3)
        elements = ["Cu", "Ni", "Al", "Fe"] * 7  # 28 elements, use first 27
        for i in range(27):
            hea.symbols[i] = elements[i]
        _compare(hea, mlx_calc, torch_calc, label="HEA CuNiAlFe")


# ---------------------------------------------------------------------------
# 6. Mixed periodicity and edge cases
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Test numerical edge cases: mixed periodicity, close atoms, boundaries."""

    def test_1d_periodic(self, mlx_calc, torch_calc):
        """1D periodic Si chain (pbc=[True, False, False])."""
        atoms_1d = Atoms(
            "Si4",
            positions=[[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [4.5, 0, 0]],
            cell=[6, 20, 20],
            pbc=[True, False, False],
        )
        _compare(atoms_1d, mlx_calc, torch_calc, label="1D periodic Si")

    def test_2d_periodic(self, mlx_calc, torch_calc):
        """2D periodic Cu sheet (pbc=[True, True, False])."""
        atoms_2d = fcc111("Cu", size=(4, 4, 1), vacuum=15.0)
        atoms_2d.pbc = [True, True, False]
        _compare(atoms_2d, mlx_calc, torch_calc, label="2D periodic Cu")

    def test_close_atoms(self, mlx_calc, torch_calc):
        """Two Si atoms closer than equilibrium (1.5 A)."""
        close_atoms = Atoms("SiSi", positions=[[0, 0, 0], [1.5, 0, 0]])
        _compare(close_atoms, mlx_calc, torch_calc, label="Close Si-Si")

    def test_boundary_atoms(self, mlx_calc, torch_calc):
        """Atom very close to cell boundary (periodic image test)."""
        si_boundary = bulk("Si", "diamond", a=5.43)
        si_boundary.positions[0] = [0.001, 0.001, 0.001]
        _compare(si_boundary, mlx_calc, torch_calc,
                 label="Si boundary atom")


# ---------------------------------------------------------------------------
# 7. Larger realistic systems
# ---------------------------------------------------------------------------


class TestLargerSystems:
    """Cross-validate larger multi-species systems."""

    def test_nacl_supercell(self, mlx_calc, torch_calc):
        """NaCl 3x3x3 supercell (216 atoms, rocksalt)."""
        from ase.spacegroup import crystal as ase_crystal
        nacl = ase_crystal(
            ["Na", "Cl"],
            [(0, 0, 0), (0.5, 0.5, 0.5)],
            spacegroup=225,
            cellpar=[5.64, 5.64, 5.64, 90, 90, 90],
        ) * (3, 3, 3)
        assert len(nacl) == 216
        _compare(nacl, mlx_calc, torch_calc, label="NaCl 3x3x3")

    def test_steel_fe_c(self, mlx_calc, torch_calc):
        """Fe-C steel-like: 54 Fe + 4 C at octahedral interstitials (58 atoms)."""
        a = 2.87
        steel = bulk("Fe", "bcc", a=a, cubic=True) * (3, 3, 3)
        # Octahedral interstitial sites in BCC: edge midpoints
        for pos in [
            [a / 2, 0, 0],
            [0, a / 2, 0],
            [a, a, a / 2],
            [2 * a, a / 2, 2 * a],
        ]:
            steel.append(Atom("C", position=pos))
        assert len(steel) == 58
        _compare(steel, mlx_calc, torch_calc, label="Fe-C steel")

    def test_mgo_supercell(self, mlx_calc, torch_calc):
        """MgO 3x3x3 rocksalt supercell (216 atoms)."""
        from ase.spacegroup import crystal as ase_crystal
        mgo = ase_crystal(
            ["Mg", "O"],
            [(0, 0, 0), (0.5, 0.5, 0.5)],
            spacegroup=225,
            cellpar=[4.212, 4.212, 4.212, 90, 90, 90],
        )
        mgo_slab = mgo * (3, 3, 3)
        assert len(mgo_slab) == 216
        _compare(mgo_slab, mlx_calc, torch_calc, label="MgO 3x3x3")
