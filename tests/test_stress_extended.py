"""Extended stress validation tests for MACE-MLX.

Cross-validates stress against PyTorch MACE for multi-element and strained
systems, validates stress via finite differences, and checks equation-of-state
pressure-volume consistency.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from ase.spacegroup import crystal

from mace_mlx.calculators import MACEMLXCalculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mlx_calc():
    """Shared MACEMLXCalculator with MACE-MP-0 small model (no NL caching)."""
    try:
        import torch  # noqa: F401
        from mace.calculators import mace_mp  # noqa: F401
    except ImportError:
        pytest.skip("mace-torch not installed")
    return MACEMLXCalculator(model_path="small", skin=0.0)


@pytest.fixture(scope="module")
def torch_calc():
    """Shared PyTorch MACE-MP-0 small calculator."""
    try:
        from mace.calculators import mace_mp
    except ImportError:
        pytest.skip("mace-torch not installed")
    return mace_mp(model="small", device="cpu", default_dtype="float32")


# ---------------------------------------------------------------------------
# System builders
# ---------------------------------------------------------------------------


def _make_strained_si():
    """Si diamond with 2% uniaxial strain along x."""
    si = bulk("Si", "diamond", a=5.43)
    cell = np.array(si.get_cell(), dtype=np.float64)
    cell[0] *= 1.02  # 2% strain along x
    si.set_cell(cell, scale_atoms=True)
    return si


def _make_nacl():
    return crystal(
        ["Na", "Cl"],
        [(0, 0, 0), (0.5, 0.5, 0.5)],
        spacegroup=225,
        cellpar=[5.64] * 3 + [90] * 3,
    )


def _make_mgo():
    return crystal(
        ["Mg", "O"],
        [(0, 0, 0), (0.5, 0.5, 0.5)],
        spacegroup=225,
        cellpar=[4.212] * 3 + [90] * 3,
    )


# Systems for cross-validation
_stress_systems = [
    ("Si diamond", bulk("Si", "diamond", a=5.43)),
    ("Cu fcc", bulk("Cu", "fcc", a=3.6)),
    ("Al fcc", bulk("Al", "fcc", a=4.05)),
    ("Fe bcc", bulk("Fe", "bcc", a=2.87)),
    ("Si 2x2x2", bulk("Si", "diamond", a=5.43) * (2, 2, 2)),
    ("Cu 2x2x2", bulk("Cu", "fcc", a=3.6) * (2, 2, 2)),
    ("NaCl", _make_nacl()),
    ("MgO", _make_mgo()),
    ("Si strained", _make_strained_si()),
]


# ---------------------------------------------------------------------------
# 1. Stress cross-validation for complex systems
# ---------------------------------------------------------------------------


class TestStressCrossValidation:
    """Stress from MLX must match PyTorch MACE across diverse systems."""

    @pytest.mark.parametrize(
        "name,atoms",
        _stress_systems,
        ids=[s[0] for s in _stress_systems],
    )
    def test_stress_matches_pytorch(self, mlx_calc, torch_calc, name, atoms):
        a1 = atoms.copy()
        a1.calc = mlx_calc
        s_mlx = a1.get_stress()

        a2 = atoms.copy()
        a2.calc = torch_calc
        s_torch = a2.get_stress()

        print(f"\n  {name}: max|diff| = {np.max(np.abs(s_mlx - s_torch)):.2e}")

        np.testing.assert_allclose(
            s_mlx,
            s_torch,
            atol=1e-4,
            err_msg=f"{name} stress mismatch vs PyTorch",
        )


# ---------------------------------------------------------------------------
# 2. Stress finite difference validation
# ---------------------------------------------------------------------------


class TestStressFiniteDifference:
    """Verify stress = dE/d_eps / V via finite differences on the strain tensor."""

    def test_stress_finite_difference(self, mlx_calc):
        """Analytical stress should match numerical finite-difference stress."""
        si = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
        si.calc = mlx_calc
        stress_analytical = si.get_stress()
        vol0 = si.get_volume()
        cell0 = np.array(si.get_cell(), dtype=np.float64)
        pos0 = si.get_positions().copy().astype(np.float64)

        eps = 1e-3
        voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        stress_numerical = np.zeros(6)

        for v, (i, j) in enumerate(voigt_map):
            # +eps
            D_plus = np.zeros((3, 3))
            D_plus[i, j] = eps
            S_plus = 0.5 * (D_plus + D_plus.T)

            si_p = si.copy()
            si_p.set_cell(cell0 + cell0 @ S_plus, scale_atoms=False)
            si_p.set_positions(pos0 + pos0 @ S_plus)
            si_p.calc = mlx_calc
            e_plus = si_p.get_potential_energy()

            # -eps
            D_minus = np.zeros((3, 3))
            D_minus[i, j] = -eps
            S_minus = 0.5 * (D_minus + D_minus.T)

            si_m = si.copy()
            si_m.set_cell(cell0 + cell0 @ S_minus, scale_atoms=False)
            si_m.set_positions(pos0 + pos0 @ S_minus)
            si_m.calc = mlx_calc
            e_minus = si_m.get_potential_energy()

            stress_numerical[v] = (e_plus - e_minus) / (2 * eps * vol0)

        print(f"\n  Analytical: {stress_analytical}")
        print(f"  Numerical:  {stress_numerical}")
        print(f"  Max diff:   {np.max(np.abs(stress_analytical - stress_numerical)):.2e}")

        np.testing.assert_allclose(
            stress_analytical,
            stress_numerical,
            atol=1e-3,
            err_msg="Analytical vs finite-difference stress mismatch",
        )


# ---------------------------------------------------------------------------
# 3. Equation of state (pressure-volume)
# ---------------------------------------------------------------------------


class TestEquationOfState:
    """Compute E(V) curve and verify P = -dE/dV matches stress."""

    def test_equation_of_state(self, mlx_calc):
        """Hydrostatic pressure from stress should match -dE/dV."""
        si = bulk("Si", "diamond", a=5.43)

        volumes = []
        energies = []
        pressures_from_stress = []

        for scale in np.linspace(0.95, 1.05, 11):
            atoms = si.copy()
            atoms.set_cell(si.cell * scale, scale_atoms=True)
            atoms.calc = mlx_calc

            e = atoms.get_potential_energy()
            s = atoms.get_stress()  # Voigt: xx, yy, zz, yz, xz, xy
            p = -(s[0] + s[1] + s[2]) / 3  # hydrostatic pressure

            volumes.append(atoms.get_volume())
            energies.append(e)
            pressures_from_stress.append(p)

        V = np.array(volumes)
        E = np.array(energies)
        P_numerical = -np.gradient(E, V)
        P_stress = np.array(pressures_from_stress)

        # Compare inner points (avoid boundary gradient artifacts)
        print(f"\n  P_stress (inner):    {P_stress[2:-2]}")
        print(f"  P_numerical (inner): {P_numerical[2:-2]}")
        print(f"  Max diff: {np.max(np.abs(P_stress[2:-2] - P_numerical[2:-2])):.4f}")

        np.testing.assert_allclose(
            P_stress[2:-2],
            P_numerical[2:-2],
            atol=0.05,
            err_msg="Pressure from stress vs -dE/dV mismatch",
        )
