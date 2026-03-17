"""MD stability and geometry optimization tests for MACE-MLX.

Tests NVE energy conservation, trajectory cross-validation with PyTorch,
geometry optimization, and cell optimization with stress.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

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


@pytest.fixture(scope="module")
def torch_calc():
    """Shared PyTorch MACE-MP-0 small calculator."""
    try:
        from mace.calculators import mace_mp
    except ImportError:
        pytest.skip("mace-torch not installed")
    return mace_mp(model="small", device="cpu", default_dtype="float32")


# ---------------------------------------------------------------------------
# NVE energy conservation
# ---------------------------------------------------------------------------


class TestNVEEnergyConservation:
    """NVE MD should conserve total energy."""

    def test_nve_energy_conservation_water(self, mlx_calc):
        """10-step NVE for water molecule with small timestep."""
        water = Atoms(
            "OH2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.757, 0.586, 0.0],
                [-0.757, 0.586, 0.0],
            ],
        )
        water.calc = mlx_calc

        MaxwellBoltzmannDistribution(water, temperature_K=100)
        dyn = VelocityVerlet(water, timestep=0.5 * units.fs)

        energies = []
        for _ in range(10):
            dyn.run(1)
            e_total = water.get_potential_energy() + water.get_kinetic_energy()
            energies.append(e_total)

        energies = np.array(energies)
        drift = np.max(energies) - np.min(energies)
        print(f"\n  Water NVE drift: {drift:.6f} eV over 10 steps at 0.5 fs")
        assert drift < 0.1, f"Energy drift {drift:.4f} eV too large for water NVE"

    def test_nve_energy_conservation_bulk_si(self, mlx_calc):
        """10-step NVE for bulk Si 2x2x2."""
        si = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
        si.calc = mlx_calc

        MaxwellBoltzmannDistribution(si, temperature_K=300)
        dyn = VelocityVerlet(si, timestep=0.5 * units.fs)

        energies = []
        for _ in range(10):
            dyn.run(1)
            e_total = si.get_potential_energy() + si.get_kinetic_energy()
            energies.append(e_total)

        energies = np.array(energies)
        drift = np.max(energies) - np.min(energies)
        print(f"\n  Si 2x2x2 NVE drift: {drift:.6f} eV over 10 steps at 0.5 fs")
        assert drift < 0.05, f"Energy drift {drift:.4f} eV exceeds 50 meV"


# ---------------------------------------------------------------------------
# Trajectory cross-validation
# ---------------------------------------------------------------------------


class TestTrajectoryComparison:
    """NVE trajectories from MLX and PyTorch should match closely."""

    def test_nve_trajectory_matches_pytorch(self, mlx_calc, torch_calc):
        """Run 5 NVE steps with both MLX and PyTorch, compare trajectories."""
        si = bulk("Si", "diamond", a=5.43) * (2, 2, 2)

        # Use deterministic initialization
        rng = np.random.default_rng(42)
        MaxwellBoltzmannDistribution(si, temperature_K=100, rng=rng)
        init_positions = si.positions.copy()
        init_velocities = si.get_velocities().copy()

        # MLX trajectory
        si_mlx = si.copy()
        si_mlx.set_velocities(init_velocities)
        si_mlx.calc = mlx_calc
        dyn_mlx = VelocityVerlet(si_mlx, timestep=1.0 * units.fs)
        pos_mlx = [si_mlx.positions.copy()]
        for _ in range(5):
            dyn_mlx.run(1)
            pos_mlx.append(si_mlx.positions.copy())

        # PyTorch trajectory
        si_torch = si.copy()
        si_torch.set_velocities(init_velocities)
        si_torch.calc = torch_calc
        dyn_torch = VelocityVerlet(si_torch, timestep=1.0 * units.fs)
        pos_torch = [si_torch.positions.copy()]
        for _ in range(5):
            dyn_torch.run(1)
            pos_torch.append(si_torch.positions.copy())

        # Compare trajectories
        for step in range(6):
            rms = np.sqrt(np.mean((pos_mlx[step] - pos_torch[step]) ** 2))
            print(f"  Step {step}: RMS deviation {rms:.6f} A")
            assert rms < 0.01, f"Step {step}: RMS deviation {rms:.6f} A too large"


# ---------------------------------------------------------------------------
# Geometry optimization
# ---------------------------------------------------------------------------


class TestGeometryOptimization:
    """Test that geometry optimization converges."""

    def test_geometry_optimization_distorted_si(self, mlx_calc):
        """BFGS geometry optimization on slightly distorted Si."""
        from ase.optimize import BFGS

        si = bulk("Si", "diamond", a=5.43)
        si.positions[0] += [0.1, 0.05, -0.05]
        si.calc = mlx_calc

        opt = BFGS(si, logfile=None)
        converged = opt.run(fmax=0.05, steps=20)

        f_max = np.max(np.abs(si.get_forces()))
        print(f"\n  Distorted Si opt: fmax = {f_max:.4f} eV/A, converged = {converged}")
        assert f_max < 0.05, f"Forces not converged: fmax = {f_max:.4f}"

    def test_geometry_optimization_water(self, mlx_calc):
        """BFGS geometry optimization on slightly distorted water."""
        from ase.optimize import BFGS

        water = Atoms(
            "OH2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.85, 0.5, 0.0],  # slightly distorted
                [-0.85, 0.5, 0.0],
            ],
        )
        water.calc = mlx_calc

        opt = BFGS(water, logfile=None)
        opt.run(fmax=0.05, steps=30)

        f_max = np.max(np.abs(water.get_forces()))
        print(f"\n  Water opt: fmax = {f_max:.4f} eV/A")
        assert f_max < 0.05, f"Water optimization did not converge: fmax = {f_max:.4f}"


# ---------------------------------------------------------------------------
# Cell optimization
# ---------------------------------------------------------------------------


class TestCellOptimization:
    """Test cell optimization with stress (ExpCellFilter)."""

    def test_cell_optimization_si(self, mlx_calc):
        """Cell optimization should recover Si lattice parameter near 5.43 A."""
        from ase.filters import ExpCellFilter
        from ase.optimize import BFGS

        si = bulk("Si", "diamond", a=5.3)  # slightly wrong lattice param
        si.calc = mlx_calc

        ecf = ExpCellFilter(si)
        opt = BFGS(ecf, logfile=None)
        opt.run(fmax=0.05, steps=30)

        # Primitive diamond cell: a_prim = a_cubic / sqrt(2)
        # Recover cubic lattice parameter from volume: V = a^3 / 4 for diamond
        vol_opt = si.get_volume()
        a_cubic_opt = (vol_opt * 4) ** (1 / 3)
        print(f"\n  Si cell opt: a_cubic = {a_cubic_opt:.3f} A (expected ~5.43)")
        assert abs(a_cubic_opt - 5.43) < 0.2, (
            f"Optimized a_cubic = {a_cubic_opt:.3f}, expected ~5.43"
        )

    def test_cell_optimization_stress_small(self, mlx_calc):
        """After cell optimization, stress should be small."""
        from ase.filters import ExpCellFilter
        from ase.optimize import BFGS

        si = bulk("Si", "diamond", a=5.3)
        si.calc = mlx_calc

        ecf = ExpCellFilter(si)
        opt = BFGS(ecf, logfile=None)
        opt.run(fmax=0.05, steps=30)

        stress = si.get_stress()
        max_stress = np.max(np.abs(stress))
        print(f"\n  Si post-opt stress: max = {max_stress:.4f} eV/A^3")
        # Stress should be reasonably small after optimization
        assert max_stress < 0.1, (
            f"Stress too large after cell opt: {max_stress:.4f} eV/A^3"
        )
