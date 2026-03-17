"""Large system correctness and timing tests for MACE-MLX.

Tests energy/force correctness for systems up to 2000 atoms against PyTorch
MACE, verifies output shapes, and checks sub-quadratic scaling.
"""

from __future__ import annotations

import time

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
# System builders
# ---------------------------------------------------------------------------


def _make_nacl():
    return crystal(
        ["Na", "Cl"],
        [(0, 0, 0), (0.5, 0.5, 0.5)],
        spacegroup=225,
        cellpar=[5.64] * 3 + [90] * 3,
    )


# Systems: (name, atoms, expected_natoms)
_large_systems = [
    ("Si 4x4x4", bulk("Si", "diamond", a=5.43) * (4, 4, 4), 128),
    ("Cu 4x4x4", bulk("Cu", "fcc", a=3.6) * (4, 4, 4), 64),
    ("Al 4x4x4", bulk("Al", "fcc", a=4.05) * (4, 4, 4), 64),
    ("Fe 4x4x4", bulk("Fe", "bcc", a=2.87) * (4, 4, 4), 64),
    ("NaCl 3x3x3", _make_nacl() * (3, 3, 3), 216),
    ("Si 6x6x6", bulk("Si", "diamond", a=5.43) * (6, 6, 6), 432),
    ("Cu 7x7x7", bulk("Cu", "fcc", a=3.6) * (7, 7, 7), 343),
]


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


class TestLargeSystemCorrectness:
    """Cross-validate energy and forces for large systems."""

    @pytest.mark.parametrize(
        "name,atoms,expected_natoms",
        _large_systems,
        ids=[s[0] for s in _large_systems],
    )
    def test_energy_matches_pytorch(
        self, mlx_calc, torch_calc, name, atoms, expected_natoms
    ):
        """Energy matches PyTorch for large systems (tolerance scales with size)."""
        assert len(atoms) == expected_natoms, (
            f"{name}: expected {expected_natoms} atoms, got {len(atoms)}"
        )
        # Scale tolerance with system size (accumulation errors)
        atol_e = 0.01 * (expected_natoms / 100)
        atol_f = 0.02

        a_mlx = atoms.copy()
        a_mlx.calc = mlx_calc
        e_mlx = a_mlx.get_potential_energy()
        f_mlx = a_mlx.get_forces()

        a_torch = atoms.copy()
        a_torch.calc = torch_calc
        e_torch = a_torch.get_potential_energy()
        f_torch = a_torch.get_forces()

        e_diff = abs(e_mlx - e_torch)
        f_max_diff = float(np.max(np.abs(f_mlx - f_torch)))

        print(
            f"\n  {name} ({expected_natoms} atoms): "
            f"E diff = {e_diff:.2e} eV, F max diff = {f_max_diff:.2e} eV/A"
        )

        np.testing.assert_allclose(
            e_mlx,
            e_torch,
            atol=atol_e,
            err_msg=f"{name}: energy mismatch",
        )
        np.testing.assert_allclose(
            f_mlx,
            f_torch,
            atol=atol_f,
            err_msg=f"{name}: forces mismatch",
        )

    @pytest.mark.timeout(300)
    def test_2000_atoms_runs(self, mlx_calc):
        """Verify 2000-atom system completes without error."""
        si = bulk("Si", "diamond", a=5.43) * (10, 10, 10)
        assert len(si) == 2000
        si.calc = mlx_calc

        t0 = time.perf_counter()
        e = si.get_potential_energy()
        f = si.get_forces()
        elapsed = time.perf_counter() - t0

        print(f"\n  Si 10x10x10 (2000 atoms): E = {e:.4f} eV, time = {elapsed:.1f} s")

        assert isinstance(e, float)
        assert np.isfinite(e)
        assert f.shape == (2000, 3)
        assert np.all(np.isfinite(f))

    def test_force_shapes(self, mlx_calc):
        """Force arrays have correct shapes for all large systems."""
        for name, atoms, expected_natoms in _large_systems:
            a = atoms.copy()
            a.calc = mlx_calc
            f = a.get_forces()
            assert f.shape == (expected_natoms, 3), (
                f"{name}: expected ({expected_natoms}, 3), got {f.shape}"
            )

    def test_energy_scales_linearly(self, mlx_calc):
        """Energy should scale approximately linearly with atom count."""
        si1 = bulk("Si", "diamond", a=5.43)
        si1.calc = mlx_calc
        e1 = si1.get_potential_energy()

        for n in [2, 3, 4]:
            si_n = bulk("Si", "diamond", a=5.43) * (n, n, n)
            si_n.calc = mlx_calc
            e_n = si_n.get_potential_energy()
            natoms = len(si_n)
            expected_ratio = natoms / len(si1)
            actual_ratio = e_n / e1

            print(
                f"\n  Si {n}x{n}x{n} ({natoms} atoms): "
                f"ratio = {actual_ratio:.2f} (expected ~{expected_ratio:.0f})"
            )

            assert abs(actual_ratio - expected_ratio) < expected_ratio * 0.3, (
                f"Energy scaling: E1={e1:.4f}, E{n}={e_n:.4f}, "
                f"ratio={actual_ratio:.2f}, expected ~{expected_ratio:.0f}"
            )


# ---------------------------------------------------------------------------
# Timing / scaling tests
# ---------------------------------------------------------------------------


class TestLargeSystemTiming:
    """Verify inference time scales sub-quadratically with system size."""

    def test_scaling_subquadratic(self, mlx_calc):
        """Time should scale sub-quadratically with system size."""
        times = {}
        for n in [2, 4, 6]:
            si = bulk("Si", "diamond", a=5.43) * (n, n, n)
            natoms = len(si)
            si.calc = mlx_calc

            # Warmup
            _ = si.get_potential_energy()
            _ = si.get_forces()

            # Timed run
            mlx_calc.results = {}
            t0 = time.perf_counter()
            _ = si.get_potential_energy()
            _ = si.get_forces()
            times[natoms] = time.perf_counter() - t0

        print("\n  Scaling results:")
        for natoms, t in sorted(times.items()):
            print(f"    {natoms} atoms: {t:.3f} s")

        n_small = sorted(times.keys())[0]
        n_large = sorted(times.keys())[-1]

        ratio_time = times[n_large] / times[n_small]
        ratio_atoms = n_large / n_small

        print(
            f"  Ratio: {n_large}/{n_small} atoms = {ratio_atoms:.0f}x, "
            f"time = {ratio_time:.1f}x"
        )

        # Sub-quadratic: time ratio should be less than atom_ratio^1.5
        max_ratio = ratio_atoms ** 1.5
        assert ratio_time < max_ratio, (
            f"Scaling looks super-quadratic: {ratio_time:.1f}x time for "
            f"{ratio_atoms:.0f}x atoms (limit {max_ratio:.0f}x)"
        )
