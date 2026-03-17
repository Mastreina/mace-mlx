"""Multi-element and complex crystalline system tests.

Cross-validates MACE-MLX vs PyTorch MACE for multi-element crystals,
alloys, oxides, supercells, and larger organic molecules.
"""

from __future__ import annotations

import math

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
# Structure builders — each returns an Atoms object or raises an exception
# ---------------------------------------------------------------------------


def _build_nacl():
    """NaCl rock salt — ionic, 2 species."""
    from ase.spacegroup import crystal
    return crystal(
        ["Na", "Cl"], [(0, 0, 0), (0.5, 0.5, 0.5)],
        spacegroup=225, cellpar=[5.64, 5.64, 5.64, 90, 90, 90],
    )


def _build_mgo():
    """MgO periclase — simple oxide, 2 species."""
    from ase.spacegroup import crystal
    return crystal(
        ["Mg", "O"], [(0, 0, 0), (0.5, 0.5, 0.5)],
        spacegroup=225, cellpar=[4.212] * 3 + [90] * 3,
    )


def _build_tio2():
    """TiO2 rutile — transition-metal oxide, 2 species."""
    from ase.spacegroup import crystal
    return crystal(
        ["Ti", "O"], [(0, 0, 0), (0.3, 0.3, 0)],
        spacegroup=136, cellpar=[4.594, 4.594, 2.959, 90, 90, 90],
    )


def _build_sio2():
    """SiO2 alpha-quartz — 3D covalent oxide, 2 species."""
    from ase.spacegroup import crystal
    return crystal(
        ["Si", "O"], [(0.4697, 0, 0), (0.4135, 0.2669, 0.1191)],
        spacegroup=152, cellpar=[4.916, 4.916, 5.405, 90, 90, 120],
    )


def _build_caco3():
    """CaCO3 calcite — 3 species."""
    from ase.spacegroup import crystal
    return crystal(
        ["Ca", "C", "O"],
        [(0, 0, 0), (0, 0, 0.25), (0.2578, 0, 0.25)],
        spacegroup=167, cellpar=[4.99, 4.99, 17.06, 90, 90, 120],
    )


def _build_srtio3():
    """SrTiO3 perovskite — 3 species."""
    from ase.spacegroup import crystal
    return crystal(
        ["Sr", "Ti", "O", "O", "O"],
        [(0, 0, 0), (0.5, 0.5, 0.5),
         (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        spacegroup=221, cellpar=[3.905] * 3 + [90] * 3,
    )


def _build_fe2o3():
    """Fe2O3 hematite — magnetic oxide."""
    from ase.spacegroup import crystal
    return crystal(
        ["Fe", "O"], [(0, 0, 0.3553), (0.3059, 0, 0.25)],
        spacegroup=167, cellpar=[5.038, 5.038, 13.772, 90, 90, 120],
    )


def _build_gan():
    """GaN wurtzite — semiconductor, hexagonal."""
    from ase.spacegroup import crystal
    return crystal(
        ["Ga", "N"], [(1 / 3, 2 / 3, 0), (1 / 3, 2 / 3, 0.385)],
        spacegroup=186, cellpar=[3.189, 3.189, 5.185, 90, 90, 120],
    )


def _build_batio3():
    """BaTiO3 perovskite — ferroelectric."""
    from ase.spacegroup import crystal
    return crystal(
        ["Ba", "Ti", "O", "O", "O"],
        [(0, 0, 0), (0.5, 0.5, 0.5),
         (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        spacegroup=221, cellpar=[4.01] * 3 + [90] * 3,
    )


def _build_zno():
    """ZnO wurtzite."""
    from ase.spacegroup import crystal
    return crystal(
        ["Zn", "O"], [(1 / 3, 2 / 3, 0), (1 / 3, 2 / 3, 0.382)],
        spacegroup=186, cellpar=[3.25, 3.25, 5.207, 90, 90, 120],
    )


def _build_cuzn():
    """CuZn brass — B2 structure."""
    from ase.spacegroup import crystal
    return crystal(
        ["Cu", "Zn"], [(0, 0, 0), (0.5, 0.5, 0.5)],
        spacegroup=221, cellpar=[2.95] * 3 + [90] * 3,
    )


def _build_nial():
    """NiAl B2 intermetallic."""
    from ase.spacegroup import crystal
    return crystal(
        ["Ni", "Al"], [(0, 0, 0), (0.5, 0.5, 0.5)],
        spacegroup=221, cellpar=[2.887] * 3 + [90] * 3,
    )


def _build_feni():
    """FeNi random alloy — FCC with mixed species."""
    feni = bulk("Ni", "fcc", a=3.56) * (2, 2, 2)  # 32 atoms
    symbols = list(feni.get_chemical_symbols())
    # Replace half with Fe
    for i in [0, 2, 5, 7]:
        symbols[i] = "Fe"
    feni.set_chemical_symbols(symbols)
    return feni


def _build_nacl_super():
    """NaCl 2x2x2 supercell — 64 atoms."""
    return _build_nacl() * (2, 2, 2)


def _build_mgo_super():
    """MgO 3x3x3 supercell — 54 atoms."""
    return _build_mgo() * (3, 3, 3)


def _build_srtio3_super():
    """SrTiO3 2x2x2 supercell — 40 atoms."""
    return _build_srtio3() * (2, 2, 2)


def _build_ethanol():
    """Ethanol C2H5OH — 9 atoms, 3 species."""
    return Atoms("C2H6O", positions=[
        [0.0, 0.0, 0.0],       # C
        [1.52, 0.0, 0.0],      # C
        [2.09, 1.21, 0.0],     # O
        [-0.54, 0.0, 1.03],    # H
        [-0.54, 0.0, -1.03],   # H
        [-0.54, -1.03, 0.0],   # H
        [2.09, -0.54, 1.03],   # H
        [2.09, -0.54, -1.03],  # H
        [3.05, 1.21, 0.0],     # H (OH)
    ])


def _build_benzene():
    """Benzene C6H6 — planar ring, 12 atoms."""
    positions = []
    for i in range(6):
        angle = i * math.pi / 3
        positions.append([1.4 * math.cos(angle), 1.4 * math.sin(angle), 0])
    for i in range(6):
        angle = i * math.pi / 3
        positions.append([2.48 * math.cos(angle), 2.48 * math.sin(angle), 0])
    return Atoms("C6H6", positions=positions)


def _build_acetic_acid():
    """Acetic acid CH3COOH — 8 atoms, 3 species."""
    return Atoms("C2H4O2", positions=[
        [0.0, 0.0, 0.0],        # C (methyl)
        [1.52, 0.0, 0.0],       # C (carboxyl)
        [2.09, 1.08, 0.4],      # O (=O)
        [2.09, -1.08, -0.4],    # O (-OH)
        [-0.54, 0.0, 1.03],     # H
        [-0.54, 0.94, -0.52],   # H
        [-0.54, -0.94, -0.52],  # H
        [3.05, -1.08, -0.4],    # H (OH)
    ])


# ---------------------------------------------------------------------------
# Safe builder wrapper — catches spacegroup/build errors
# ---------------------------------------------------------------------------


def _safe_build(builder_fn, name):
    """Try to build a structure; skip test if it fails."""
    try:
        atoms = builder_fn()
        if atoms is None or len(atoms) == 0:
            pytest.skip(f"{name}: builder returned empty structure")
        return atoms
    except Exception as e:
        pytest.skip(f"{name}: structure build failed — {e}")


# ---------------------------------------------------------------------------
# Helper: check if all elements are in the z_table
# ---------------------------------------------------------------------------


def _check_z_table(atoms, calc, name):
    """Skip test if any element is missing from the calculator's z_table."""
    z_table = getattr(calc, "z_table", None)
    if z_table is None:
        return
    for z in set(atoms.get_atomic_numbers()):
        if int(z) not in z_table:
            pytest.skip(
                f"{name}: element Z={z} not in model's z_table"
            )


# ---------------------------------------------------------------------------
# Helper: run comparison between MLX and PyTorch
# ---------------------------------------------------------------------------


def _compare_energy_forces(
    atoms, mlx_calc, torch_calc, name,
    energy_atol=1e-3, forces_atol=1e-2,
):
    """Cross-validate energy and forces between MLX and PyTorch."""
    _check_z_table(atoms, mlx_calc, name)

    a_mlx = atoms.copy()
    a_mlx.calc = mlx_calc
    a_torch = atoms.copy()
    a_torch.calc = torch_calc

    e_mlx = a_mlx.get_potential_energy()
    e_torch = a_torch.get_potential_energy()
    f_mlx = a_mlx.get_forces()
    f_torch = a_torch.get_forces()

    e_diff = abs(e_mlx - e_torch)
    f_max_diff = np.max(np.abs(f_mlx - f_torch))
    print(
        f"{name:20s}  natoms={len(atoms):3d}  "
        f"E_mlx={e_mlx:12.6f}  E_torch={e_torch:12.6f}  "
        f"dE={e_diff:.2e}  dF_max={f_max_diff:.2e}"
    )

    np.testing.assert_allclose(
        e_mlx, e_torch, atol=energy_atol,
        err_msg=f"{name} energy mismatch",
    )
    np.testing.assert_allclose(
        f_mlx, f_torch, atol=forces_atol,
        err_msg=f"{name} forces mismatch",
    )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestMultiElementCrystals:
    """Cross-validate MLX vs PyTorch for multi-element crystalline systems."""

    @pytest.mark.parametrize("name,builder", [
        ("NaCl", _build_nacl),
        ("MgO", _build_mgo),
        ("TiO2", _build_tio2),
        ("SiO2", _build_sio2),
        ("CaCO3", _build_caco3),
        ("SrTiO3", _build_srtio3),
        ("Fe2O3", _build_fe2o3),
        ("GaN", _build_gan),
        ("BaTiO3", _build_batio3),
        ("ZnO", _build_zno),
    ])
    def test_energy_forces_match(self, mlx_calc, torch_calc, name, builder):
        atoms = _safe_build(builder, name)
        _compare_energy_forces(atoms, mlx_calc, torch_calc, name)

    @pytest.mark.parametrize("name,builder", [
        ("NaCl", _build_nacl),
        ("MgO", _build_mgo),
        ("TiO2", _build_tio2),
        ("SrTiO3", _build_srtio3),
    ])
    def test_energy_is_negative(self, mlx_calc, name, builder):
        """Multi-element crystals at equilibrium should have negative total energy."""
        atoms = _safe_build(builder, name)
        _check_z_table(atoms, mlx_calc, name)
        atoms.calc = mlx_calc
        e = atoms.get_potential_energy()
        assert e < 0, f"{name}: energy {e:.4f} should be negative"

    @pytest.mark.parametrize("name,builder", [
        ("NaCl", _build_nacl),
        ("MgO", _build_mgo),
        ("SrTiO3", _build_srtio3),
    ])
    def test_forces_near_zero(self, mlx_calc, name, builder):
        """Forces on atoms in perfect crystals should be small."""
        atoms = _safe_build(builder, name)
        _check_z_table(atoms, mlx_calc, name)
        atoms.calc = mlx_calc
        f = atoms.get_forces()
        f_max = np.max(np.abs(f))
        assert f_max < 1.0, (
            f"{name}: max force {f_max:.4f} eV/A too large for equilibrium structure"
        )


class TestAlloys:
    """Cross-validate MLX vs PyTorch for alloy / intermetallic systems."""

    @pytest.mark.parametrize("name,builder", [
        ("CuZn", _build_cuzn),
        ("NiAl", _build_nial),
        ("FeNi", _build_feni),
    ])
    def test_energy_forces_match(self, mlx_calc, torch_calc, name, builder):
        atoms = _safe_build(builder, name)
        _compare_energy_forces(atoms, mlx_calc, torch_calc, name)

    @pytest.mark.parametrize("name,builder", [
        ("CuZn", _build_cuzn),
        ("NiAl", _build_nial),
    ])
    def test_energy_is_negative(self, mlx_calc, name, builder):
        atoms = _safe_build(builder, name)
        _check_z_table(atoms, mlx_calc, name)
        atoms.calc = mlx_calc
        e = atoms.get_potential_energy()
        assert e < 0, f"{name}: energy {e:.4f} should be negative"


class TestMultiElementSupercells:
    """Cross-validate MLX vs PyTorch for larger multi-element supercells."""

    @pytest.mark.parametrize("name,builder", [
        ("NaCl-2x2x2", _build_nacl_super),
        ("MgO-3x3x3", _build_mgo_super),
        ("SrTiO3-2x2x2", _build_srtio3_super),
    ])
    def test_energy_forces_match(self, mlx_calc, torch_calc, name, builder):
        atoms = _safe_build(builder, name)
        _compare_energy_forces(
            atoms, mlx_calc, torch_calc, name,
            energy_atol=1e-2, forces_atol=1e-2,
        )

    def test_nacl_energy_scales(self, mlx_calc):
        """Energy should scale linearly with number of unit cells."""
        nacl1 = _safe_build(_build_nacl, "NaCl")
        _check_z_table(nacl1, mlx_calc, "NaCl")
        nacl1.calc = mlx_calc
        e1 = nacl1.get_potential_energy()
        n1 = len(nacl1)

        nacl2 = _safe_build(_build_nacl_super, "NaCl-2x2x2")
        nacl2.calc = mlx_calc
        e2 = nacl2.get_potential_energy()
        n2 = len(nacl2)

        # Per-atom energy should be roughly the same
        e_per_atom_1 = e1 / n1
        e_per_atom_2 = e2 / n2
        diff = abs(e_per_atom_1 - e_per_atom_2)
        print(
            f"NaCl per-atom energy: unit={e_per_atom_1:.4f}  "
            f"super={e_per_atom_2:.4f}  diff={diff:.4f}"
        )
        assert diff < 0.1, (
            f"Per-atom energy mismatch: {e_per_atom_1:.4f} vs {e_per_atom_2:.4f}"
        )


class TestLargerMolecules:
    """Cross-validate MLX vs PyTorch for larger organic-like molecules."""

    @pytest.mark.parametrize("name,builder", [
        ("Ethanol", _build_ethanol),
        ("Benzene", _build_benzene),
        ("AceticAcid", _build_acetic_acid),
    ])
    def test_energy_forces_match(self, mlx_calc, torch_calc, name, builder):
        atoms = _safe_build(builder, name)
        _compare_energy_forces(atoms, mlx_calc, torch_calc, name)

    @pytest.mark.parametrize("name,builder", [
        ("Ethanol", _build_ethanol),
        ("Benzene", _build_benzene),
        ("AceticAcid", _build_acetic_acid),
    ])
    def test_force_shapes(self, mlx_calc, name, builder):
        """Force array shape must match (natoms, 3)."""
        atoms = _safe_build(builder, name)
        _check_z_table(atoms, mlx_calc, name)
        atoms.calc = mlx_calc
        f = atoms.get_forces()
        assert f.shape == (len(atoms), 3), (
            f"{name}: expected forces shape ({len(atoms)}, 3), got {f.shape}"
        )

    @pytest.mark.parametrize("name,builder", [
        ("Ethanol", _build_ethanol),
        ("Benzene", _build_benzene),
        ("AceticAcid", _build_acetic_acid),
    ])
    def test_total_force_zero(self, mlx_calc, name, builder):
        """Total force on an isolated molecule should sum to zero."""
        atoms = _safe_build(builder, name)
        _check_z_table(atoms, mlx_calc, name)
        atoms.calc = mlx_calc
        f = atoms.get_forces()
        total = np.sum(f, axis=0)
        np.testing.assert_allclose(
            total, 0.0, atol=1e-3,
            err_msg=f"{name}: total force not zero — {total}",
        )

    @pytest.mark.parametrize("name,builder", [
        ("Ethanol", _build_ethanol),
        ("Benzene", _build_benzene),
        ("AceticAcid", _build_acetic_acid),
    ])
    def test_energy_finite(self, mlx_calc, name, builder):
        """Energy should be finite for all molecules."""
        atoms = _safe_build(builder, name)
        _check_z_table(atoms, mlx_calc, name)
        atoms.calc = mlx_calc
        e = atoms.get_potential_energy()
        assert np.isfinite(e), f"{name}: energy {e} is not finite"
