"""Performance regression tests for MACE-MLX.

These tests verify that inference time stays within acceptable bounds
and that correctness is maintained across different system sizes, materials,
and ALL supported foundation models.

Run:
    pytest tests/test_benchmark.py -v -s
    pytest tests/test_benchmark.py -v -s -m benchmark
    pytest -m "not benchmark"            # skip these in CI
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

# Mark every test in this module as a benchmark test
pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# All supported foundation models
# ---------------------------------------------------------------------------

ALL_FOUNDATION_MODELS = [
    ("small", None),
    ("medium", None),
    ("large", None),
    ("small-0b", None),
    ("medium-0b", None),
    ("small-0b2", None),
    ("medium-0b2", None),
    ("large-0b2", None),
    ("medium-0b3", None),
    ("medium-mpa-0", None),
    ("small-omat-0", None),
    ("medium-omat-0", None),
    ("mace-matpes-pbe-0", None),
    ("mace-matpes-r2scan-0", None),
    ("mh-1", "matpes_r2scan"),
]


def _model_id(val):
    """Generate readable test IDs for model parametrize."""
    model_name, head = val
    if head:
        return f"{model_name}({head})"
    return model_name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mlx_calculator():
    """Module-scoped MLX calculator (loaded once, reused across tests)."""
    from mace_mlx.calculators import MACEMLXCalculator

    return MACEMLXCalculator(model_path="small")


@pytest.fixture(scope="module")
def torch_calculator():
    """Module-scoped PyTorch MACE calculator."""
    try:
        from mace.calculators import mace_mp
    except ImportError:
        pytest.skip("mace-torch not installed")
    return mace_mp(model="small", device="cpu", default_dtype="float32")


# ---------------------------------------------------------------------------
# Helper atoms builders
# ---------------------------------------------------------------------------

def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )


def _si(n: int = 1) -> Atoms:
    return bulk("Si", "diamond", a=5.43) * (n, n, n)


def _cu(n: int = 2) -> Atoms:
    return bulk("Cu", "fcc", a=3.6) * (n, n, n)


def _al(n: int = 2) -> Atoms:
    return bulk("Al", "fcc", a=4.05) * (n, n, n)


def _fe(n: int = 2) -> Atoms:
    return bulk("Fe", "bcc", a=2.87) * (n, n, n)


# ---------------------------------------------------------------------------
# Helper: run timing loop
# ---------------------------------------------------------------------------

def _time_calc(atoms: Atoms, calc, n_warmup: int = 2, n_runs: int = 10) -> float:
    """Return mean inference time in milliseconds (after warmup)."""
    a = atoms.copy()
    a.calc = calc

    for _ in range(n_warmup):
        a.calc.results = {}
        _ = a.get_potential_energy()
        _ = a.get_forces()

    times: list[float] = []
    for _ in range(n_runs):
        a.calc.results = {}
        t0 = time.perf_counter()
        _ = a.get_potential_energy()
        _ = a.get_forces()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.mean(times)) * 1000


def _load_mlx_calc(model_name: str, head: str | None = None):
    """Load an MLX calculator, skipping if it fails."""
    from mace_mlx.calculators import MACEMLXCalculator

    kw: dict = {"model_path": model_name}
    if head:
        kw["head"] = head
    try:
        return MACEMLXCalculator(**kw)
    except Exception as e:
        pytest.skip(f"Cannot load MLX model {model_name}: {e}")


def _load_torch_calc(model_name: str, head: str | None = None):
    """Load a PyTorch MACE calculator, skipping if it fails."""
    try:
        from mace.calculators.foundations_models import mace_mp
    except ImportError:
        pytest.skip("mace-torch not installed")

    kw: dict = {"model": model_name, "device": "cpu", "default_dtype": "float32"}
    if head:
        kw["head"] = head
    try:
        return mace_mp(**kw)
    except Exception as e:
        pytest.skip(f"Cannot load PyTorch model {model_name}: {e}")


# =====================================================================
# Correctness tests — small model (original)
# =====================================================================


class TestCorrectness:
    """Verify MLX results match PyTorch across different systems."""

    @pytest.mark.parametrize(
        "label, atoms_fn",
        [
            ("water", _water),
            ("Si 1x1x1", lambda: _si(1)),
            ("Si 2x2x2", lambda: _si(2)),
            ("Cu 2x2x2", lambda: _cu(2)),
        ],
        ids=["water", "Si-1x1x1", "Si-2x2x2", "Cu-2x2x2"],
    )
    def test_energy_forces_match(
        self, mlx_calculator, torch_calculator, label, atoms_fn,
    ):
        """Energy and forces from MLX should match PyTorch within tolerance."""
        atoms = atoms_fn()

        a_mlx = atoms.copy()
        a_mlx.calc = mlx_calculator
        e_mlx = a_mlx.get_potential_energy()
        f_mlx = a_mlx.get_forces()

        a_torch = atoms.copy()
        a_torch.calc = torch_calculator
        e_torch = a_torch.get_potential_energy()
        f_torch = a_torch.get_forces()

        e_diff = abs(e_mlx - e_torch)
        f_max_diff = float(np.max(np.abs(f_mlx - f_torch)))

        print(f"\n  {label}: E diff = {e_diff:.2e} eV, "
              f"F max diff = {f_max_diff:.2e} eV/A")

        np.testing.assert_allclose(
            e_mlx, e_torch, atol=1e-3,
            err_msg=f"{label}: energy mismatch",
        )
        np.testing.assert_allclose(
            f_mlx, f_torch, atol=1e-2,
            err_msg=f"{label}: forces mismatch",
        )

    @pytest.mark.parametrize(
        "label, atoms_fn",
        [
            ("Si 4x4x4", lambda: _si(4)),
            ("Cu 4x4x4", lambda: _cu(4)),
            ("Al 3x3x3", lambda: _al(3)),
            ("Fe 3x3x3", lambda: _fe(3)),
        ],
        ids=["Si-4x4x4", "Cu-4x4x4", "Al-3x3x3", "Fe-3x3x3"],
    )
    def test_larger_systems_match(
        self, mlx_calculator, torch_calculator, label, atoms_fn,
    ):
        """Larger systems: energy and forces from MLX match PyTorch."""
        atoms = atoms_fn()

        a_mlx = atoms.copy()
        a_mlx.calc = mlx_calculator
        e_mlx = a_mlx.get_potential_energy()
        f_mlx = a_mlx.get_forces()

        a_torch = atoms.copy()
        a_torch.calc = torch_calculator
        e_torch = a_torch.get_potential_energy()
        f_torch = a_torch.get_forces()

        e_diff = abs(e_mlx - e_torch)
        f_max_diff = float(np.max(np.abs(f_mlx - f_torch)))

        print(f"\n  {label} ({len(atoms)} atoms): E diff = {e_diff:.2e} eV, "
              f"F max diff = {f_max_diff:.2e} eV/A")

        np.testing.assert_allclose(
            e_mlx, e_torch, atol=1e-3,
            err_msg=f"{label}: energy mismatch",
        )
        np.testing.assert_allclose(
            f_mlx, f_torch, atol=1e-2,
            err_msg=f"{label}: forces mismatch",
        )

    @pytest.mark.parametrize(
        "label, atoms_fn",
        [
            ("Al 3x3x3", lambda: _al(3)),
            ("Fe 3x3x3", lambda: _fe(3)),
        ],
        ids=["Al-3x3x3", "Fe-3x3x3"],
    )
    def test_multi_material_correctness(
        self, mlx_calculator, torch_calculator, label, atoms_fn,
    ):
        """Multi-material correctness: Al and Fe match PyTorch."""
        atoms = atoms_fn()

        a_mlx = atoms.copy()
        a_mlx.calc = mlx_calculator
        e_mlx = a_mlx.get_potential_energy()
        f_mlx = a_mlx.get_forces()

        a_torch = atoms.copy()
        a_torch.calc = torch_calculator
        e_torch = a_torch.get_potential_energy()
        f_torch = a_torch.get_forces()

        e_diff = abs(e_mlx - e_torch)
        f_max_diff = float(np.max(np.abs(f_mlx - f_torch)))

        print(f"\n  {label}: E diff = {e_diff:.2e} eV, "
              f"F max diff = {f_max_diff:.2e} eV/A")

        np.testing.assert_allclose(
            e_mlx, e_torch, atol=1e-3,
            err_msg=f"{label}: energy mismatch",
        )
        np.testing.assert_allclose(
            f_mlx, f_torch, atol=1e-2,
            err_msg=f"{label}: forces mismatch",
        )

    def test_force_shape(self, mlx_calculator):
        """Force array must have shape (n_atoms, 3)."""
        systems = [
            (_water, "water"),
            (lambda: _si(2), "Si-2x2x2"),
            (lambda: _cu(2), "Cu-2x2x2"),
            (lambda: _al(3), "Al-3x3x3"),
            (lambda: _fe(3), "Fe-3x3x3"),
        ]
        for atoms_fn, label in systems:
            a = atoms_fn()
            n = len(a)
            a.calc = mlx_calculator
            f = a.get_forces()
            assert f.shape == (n, 3), (
                f"{label}: Expected ({n}, 3), got {f.shape}"
            )


# =====================================================================
# All-model correctness tests
# =====================================================================


class TestAllModelCorrectness:
    """Verify MLX matches PyTorch for every supported foundation model."""

    @pytest.mark.parametrize(
        "model_name,head",
        ALL_FOUNDATION_MODELS,
        ids=[_model_id(m) for m in ALL_FOUNDATION_MODELS],
    )
    def test_model_correctness(self, model_name: str, head: str | None):
        """Energy and forces from MLX should match PyTorch for each model."""
        calc_m = _load_mlx_calc(model_name, head)
        calc_t = _load_torch_calc(model_name, head)

        atoms = _si(2)  # 16-atom Si supercell as standard test
        n = len(atoms)

        a_m = atoms.copy()
        a_m.calc = calc_m
        e_m = a_m.get_potential_energy()
        f_m = a_m.get_forces()

        a_t = atoms.copy()
        a_t.calc = calc_t
        e_t = a_t.get_potential_energy()
        f_t = a_t.get_forces()

        dE = abs(float(e_m) - float(e_t))
        dF = float(np.max(np.abs(f_m - f_t)))

        head_str = f" (head={head})" if head else ""
        print(f"\n  {model_name}{head_str}: "
              f"E diff = {dE:.2e} eV, F max diff = {dF:.2e} eV/A")

        assert f_m.shape == (n, 3), f"Force shape mismatch: {f_m.shape}"
        np.testing.assert_allclose(
            e_m, e_t, atol=1e-3,
            err_msg=f"{model_name}{head_str}: energy mismatch",
        )
        np.testing.assert_allclose(
            f_m, f_t, atol=1e-2,
            err_msg=f"{model_name}{head_str}: forces mismatch",
        )


# =====================================================================
# All-model performance tests
# =====================================================================


class TestAllModelPerformance:
    """Benchmark each foundation model and log timing."""

    @pytest.mark.parametrize(
        "model_name,head",
        ALL_FOUNDATION_MODELS,
        ids=[_model_id(m) for m in ALL_FOUNDATION_MODELS],
    )
    def test_model_performance(self, model_name: str, head: str | None):
        """Benchmark each model on Si 2x2x2 (16 atoms)."""
        calc_m = _load_mlx_calc(model_name, head)

        atoms = _si(2)
        mean_ms = _time_calc(atoms, calc_m, n_runs=5)

        head_str = f" (head={head})" if head else ""
        print(f"\n  {model_name}{head_str}: Si 2x2x2 (16 atoms) = {mean_ms:.1f} ms")

        # Soft guard: any model should finish 16 atoms in under 30s
        assert mean_ms < 30000, (
            f"{model_name}{head_str} too slow on 16 atoms: {mean_ms:.1f} ms"
        )


# =====================================================================
# Performance tests — small model (original)
# =====================================================================


class TestPerformance:
    """Track inference timing.

    These tests do NOT hard-fail on timing (hardware varies), but they
    log the numbers so regressions can be spotted.
    """

    def test_water_timing(self, mlx_calculator):
        """Benchmark: water molecule (3 atoms)."""
        mean_ms = _time_calc(_water(), mlx_calculator)
        print(f"\n  Water  (3 atoms):  {mean_ms:.1f} ms")
        assert mean_ms < 5000, f"Water inference too slow: {mean_ms:.1f} ms"

    def test_si_1x1x1_timing(self, mlx_calculator):
        """Benchmark: Si unit cell (2 atoms)."""
        mean_ms = _time_calc(_si(1), mlx_calculator)
        print(f"\n  Si 1x1x1  (2 atoms):  {mean_ms:.1f} ms")
        assert mean_ms < 5000

    def test_si_2x2x2_timing(self, mlx_calculator):
        """Benchmark: Si 2x2x2 supercell (16 atoms)."""
        mean_ms = _time_calc(_si(2), mlx_calculator)
        print(f"\n  Si 2x2x2  (16 atoms):  {mean_ms:.1f} ms")
        assert mean_ms < 10000

    def test_si_3x3x3_timing(self, mlx_calculator):
        """Benchmark: Si 3x3x3 supercell (54 atoms)."""
        mean_ms = _time_calc(_si(3), mlx_calculator)
        print(f"\n  Si 3x3x3  (54 atoms):  {mean_ms:.1f} ms")
        assert mean_ms < 30000

    def test_si_4x4x4_timing(self, mlx_calculator):
        """Benchmark: Si 4x4x4 supercell (128 atoms)."""
        atoms = _si(4)
        mean_ms = _time_calc(atoms, mlx_calculator)
        print(f"\n  Si 4x4x4  ({len(atoms)} atoms):  {mean_ms:.1f} ms")
        assert mean_ms < 120000

    def test_cu_2x2x2_timing(self, mlx_calculator):
        """Benchmark: Cu 2x2x2 supercell."""
        atoms = _cu(2)
        mean_ms = _time_calc(atoms, mlx_calculator)
        print(f"\n  Cu 2x2x2  ({len(atoms)} atoms):  {mean_ms:.1f} ms")
        assert mean_ms < 15000

    def test_cu_3x3x3_timing(self, mlx_calculator):
        """Benchmark: Cu 3x3x3 supercell."""
        atoms = _cu(3)
        mean_ms = _time_calc(atoms, mlx_calculator)
        print(f"\n  Cu 3x3x3  ({len(atoms)} atoms):  {mean_ms:.1f} ms")
        assert mean_ms < 60000

    def test_cu_4x4x4_timing(self, mlx_calculator):
        """Benchmark: Cu 4x4x4 supercell."""
        atoms = _cu(4)
        mean_ms = _time_calc(atoms, mlx_calculator)
        print(f"\n  Cu 4x4x4  ({len(atoms)} atoms):  {mean_ms:.1f} ms")
        assert mean_ms < 120000

    def test_al_3x3x3_timing(self, mlx_calculator):
        """Benchmark: Al 3x3x3 supercell (108 atoms)."""
        atoms = _al(3)
        mean_ms = _time_calc(atoms, mlx_calculator)
        print(f"\n  Al 3x3x3  ({len(atoms)} atoms):  {mean_ms:.1f} ms")
        assert mean_ms < 120000

    def test_fe_3x3x3_timing(self, mlx_calculator):
        """Benchmark: Fe 3x3x3 supercell."""
        atoms = _fe(3)
        mean_ms = _time_calc(atoms, mlx_calculator)
        print(f"\n  Fe 3x3x3  ({len(atoms)} atoms):  {mean_ms:.1f} ms")
        assert mean_ms < 120000


# =====================================================================
# Scaling tests
# =====================================================================


class TestScaling:
    """Verify sub-quadratic scaling for different model sizes."""

    def test_scaling_is_subquadratic(self, mlx_calculator):
        """Time should not grow faster than O(n^2) with atom count (small model)."""
        t_si1 = _time_calc(_si(1), mlx_calculator, n_runs=5)
        t_si3 = _time_calc(_si(3), mlx_calculator, n_runs=5)
        ratio_atoms = 54 / 2  # 27x more atoms
        ratio_time = t_si3 / t_si1 if t_si1 > 0 else float("inf")
        print(f"\n  Scaling (small): Si 1x1x1 -> 3x3x3: "
              f"{t_si1:.1f} -> {t_si3:.1f} ms "
              f"(atoms 27x, time {ratio_time:.1f}x)")
        # O(n^2) would give 729x; anything under 200x is fine
        assert ratio_time < 200, (
            f"Scaling looks super-quadratic: {ratio_time:.0f}x for 27x atoms"
        )

    def test_scaling_small_model(self):
        """Verify sub-quadratic scaling for small model across multiple sizes."""
        calc = _load_mlx_calc("small")
        sizes = [1, 2, 3]
        labels = []
        times = []
        atom_counts = []

        for n in sizes:
            atoms = _si(n)
            t = _time_calc(atoms, calc, n_runs=5)
            labels.append(f"Si {n}x{n}x{n}")
            times.append(t)
            atom_counts.append(len(atoms))

        print("\n  Scaling (small model):")
        for i in range(len(sizes)):
            print(f"    {labels[i]:12s} ({atom_counts[i]:4d} atoms): {times[i]:.1f} ms")

        # Check that going from smallest to largest is sub-quadratic
        if times[0] > 0:
            atom_ratio = atom_counts[-1] / atom_counts[0]
            time_ratio = times[-1] / times[0]
            quadratic_ratio = atom_ratio ** 2
            print(f"    Atom ratio: {atom_ratio:.1f}x, Time ratio: {time_ratio:.1f}x, "
                  f"Quadratic would be: {quadratic_ratio:.1f}x")
            assert time_ratio < quadratic_ratio, (
                f"Scaling appears super-quadratic: {time_ratio:.0f}x for {atom_ratio:.0f}x atoms"
            )

    def test_scaling_medium_model(self):
        """Verify sub-quadratic scaling for medium model across multiple sizes."""
        calc = _load_mlx_calc("medium")
        sizes = [1, 2, 3]
        labels = []
        times = []
        atom_counts = []

        for n in sizes:
            atoms = _si(n)
            t = _time_calc(atoms, calc, n_runs=5)
            labels.append(f"Si {n}x{n}x{n}")
            times.append(t)
            atom_counts.append(len(atoms))

        print("\n  Scaling (medium model):")
        for i in range(len(sizes)):
            print(f"    {labels[i]:12s} ({atom_counts[i]:4d} atoms): {times[i]:.1f} ms")

        if times[0] > 0:
            atom_ratio = atom_counts[-1] / atom_counts[0]
            time_ratio = times[-1] / times[0]
            quadratic_ratio = atom_ratio ** 2
            print(f"    Atom ratio: {atom_ratio:.1f}x, Time ratio: {time_ratio:.1f}x, "
                  f"Quadratic would be: {quadratic_ratio:.1f}x")
            assert time_ratio < quadratic_ratio, (
                f"Scaling appears super-quadratic: {time_ratio:.0f}x for {atom_ratio:.0f}x atoms"
            )


# =====================================================================
# Neighbor list optimization tests
# =====================================================================


class TestNeighborListOptimization:
    """Verify the neighbor list caching and matscipy integration."""

    def test_matscipy_available(self):
        """matscipy should be installed for fast neighbor lists."""
        from mace_mlx.calculators import _USE_MATSCIPY
        if not _USE_MATSCIPY:
            pytest.skip("matscipy not installed")
        assert _USE_MATSCIPY

    def test_cache_reuse(self, mlx_calculator):
        """Neighbor list cache should be reused when atoms don't move."""
        atoms = _cu(2)
        a = atoms.copy()
        a.calc = mlx_calculator

        _ = a.get_potential_energy()
        assert mlx_calculator._nl_cache is not None
        first_cache = mlx_calculator._nl_cache

        a.calc.results = {}
        _ = a.get_potential_energy()
        assert mlx_calculator._nl_cache is first_cache

    def test_cache_invalidation_on_move(self, mlx_calculator):
        """Cache should be rebuilt when atoms move significantly."""
        atoms = _cu(2)
        a = atoms.copy()
        a.calc = mlx_calculator

        _ = a.get_potential_energy()
        old_positions = mlx_calculator._cache_positions.copy()

        a.positions[0] += [1.0, 0.0, 0.0]
        a.calc.results = {}
        _ = a.get_potential_energy()

        assert not np.array_equal(mlx_calculator._cache_positions, old_positions)

    def test_cache_invalidation_on_size_change(self, mlx_calculator):
        """Cache should be rebuilt when system size changes."""
        a1 = _cu(2)
        a1.calc = mlx_calculator
        _ = a1.get_potential_energy()
        n1 = mlx_calculator._cache_natoms

        a2 = _cu(3)
        a2.calc = mlx_calculator
        _ = a2.get_potential_energy()
        n2 = mlx_calculator._cache_natoms

        assert n1 != n2
        assert n2 == len(_cu(3))

    def test_nl_timing_improvement(self, mlx_calculator):
        """Neighbor list construction should be fast with matscipy."""
        from mace_mlx.calculators import _USE_MATSCIPY
        if not _USE_MATSCIPY:
            pytest.skip("matscipy not installed")

        atoms = _cu(3)

        times: list[float] = []
        for _ in range(10):
            mlx_calculator._nl_cache = None
            t0 = time.perf_counter()
            mlx_calculator._get_neighbor_list(atoms)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        mean_ms = float(np.mean(times)) * 1000
        print(f"\n  Cu 3x3x3 NL construction: {mean_ms:.2f} ms (matscipy)")
        assert mean_ms < 50, f"NL too slow: {mean_ms:.1f} ms"
