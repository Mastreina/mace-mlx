"""End-to-end tests for MACE model and checkpoint converter."""

from __future__ import annotations

import tempfile

import mlx.core as mx
import numpy as np
import pytest

from mace_mlx.irreps import Irreps
from mace_mlx.model import MACE, ScaleShiftMACE


class TestMACEModel:
    """Test MACE model construction and forward pass."""

    def test_mace_construction(self):
        """Test that MACE model can be constructed."""
        model = MACE(
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=2,
            num_interactions=2,
            hidden_irreps="16x0e",
            correlation=2,
            num_elements=4,
            avg_num_neighbors=3.0,
            radial_MLP=[16, 16],
        )
        assert len(model.interactions) == 2
        assert len(model.products) == 2
        assert len(model.readouts) == 2

    def test_mace_forward_shape(self):
        """Test MACE forward pass output shapes."""
        mx.random.seed(42)
        model = MACE(
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=2,
            num_interactions=2,
            hidden_irreps="16x0e",
            correlation=2,
            num_elements=4,
            avg_num_neighbors=3.0,
            radial_MLP=[16, 16],
        )

        num_atoms = 5
        num_edges = 8
        positions = mx.random.normal((num_atoms, 3))
        node_attrs = mx.zeros((num_atoms, 4))
        node_attrs = node_attrs.at[mx.arange(num_atoms), mx.array([0, 1, 2, 3, 0])].add(1.0)

        edge_index = mx.array([
            [0, 0, 1, 1, 2, 3, 3, 4],
            [1, 2, 0, 3, 4, 0, 2, 1],
        ])
        shifts = mx.zeros((num_edges, 3))

        output = model(positions, node_attrs, edge_index, shifts)
        mx.eval(output["energy"], output["node_energy"])

        assert output["energy"].shape == (1,)
        assert output["node_energy"].shape == (num_atoms,)

    def test_scaleshiftmace_forward(self):
        """Test ScaleShiftMACE forward pass."""
        mx.random.seed(42)
        model = ScaleShiftMACE(
            scale=0.5,
            shift=0.1,
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=2,
            num_interactions=2,
            hidden_irreps="16x0e",
            correlation=2,
            num_elements=4,
            avg_num_neighbors=3.0,
            radial_MLP=[16, 16],
        )

        num_atoms = 3
        positions = mx.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        node_attrs = mx.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])
        edge_index = mx.array([
            [0, 0, 1, 1, 2, 2],
            [1, 2, 0, 2, 0, 1],
        ])
        shifts = mx.zeros((6, 3))

        output = model(positions, node_attrs, edge_index, shifts)
        mx.eval(output["energy"], output["node_energy"])

        assert output["energy"].shape == (1,)
        assert output["node_energy"].shape == (num_atoms,)
        assert "interaction_energy" in output

    def test_energy_finite(self):
        """Test that energy output is finite."""
        mx.random.seed(123)
        model = MACE(
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=1,
            num_interactions=1,
            hidden_irreps="8x0e",
            correlation=2,
            num_elements=2,
            avg_num_neighbors=2.0,
            radial_MLP=[16],
        )

        positions = mx.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        node_attrs = mx.array([[1.0, 0.0], [0.0, 1.0]])
        edge_index = mx.array([[0, 1], [1, 0]])
        shifts = mx.zeros((2, 3))

        output = model(positions, node_attrs, edge_index, shifts)
        mx.eval(output["energy"])

        energy = output["energy"].item()
        assert np.isfinite(energy), f"Energy is not finite: {energy}"


class TestMACEMPConversion:
    """Test conversion and inference with MACE-MP-0 small model.

    These tests require the mace-torch package and will download the
    MACE-MP-0 small model (~50MB) on first run.
    """

    @pytest.fixture(scope="class")
    def converted_model(self):
        """Convert MACE-MP-0 small and return (mlx_model, torch_model)."""
        try:
            import torch
            from mace.calculators import mace_mp
        except ImportError:
            pytest.skip("mace-torch not installed")

        calc = mace_mp(model="small", device="cpu", default_dtype="float32")
        torch_model = calc.models[0]
        torch_model.eval()

        # Convert
        from mace_mlx.converter import convert_mace_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            config = convert_mace_checkpoint("small", tmpdir)

            # Load converted model
            from mace_mlx.model import load_model

            mlx_model = load_model(tmpdir)

        return mlx_model, torch_model

    @pytest.fixture
    def water_molecule(self):
        """Build a water molecule graph."""
        # H2O: O at origin, H at (0.757, 0.586, 0), H at (-0.757, 0.586, 0)
        positions_np = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.757, 0.586, 0.0],
                [-0.757, 0.586, 0.0],
            ],
            dtype=np.float32,
        )
        atomic_numbers = [8, 1, 1]

        # All-pairs edges
        senders = [0, 0, 1, 1, 2, 2]
        receivers = [1, 2, 0, 2, 0, 1]
        edge_index_np = np.array([senders, receivers], dtype=np.int32)
        shifts_np = np.zeros((6, 3), dtype=np.float32)

        return {
            "positions": positions_np,
            "atomic_numbers": atomic_numbers,
            "edge_index": edge_index_np,
            "shifts": shifts_np,
        }

    def _run_torch_model(self, torch_model, molecule):
        """Run PyTorch MACE model on a molecule."""
        import torch

        positions = torch.tensor(
            molecule["positions"], dtype=torch.float32, requires_grad=True
        )

        # Build one-hot node_attrs
        z_table = torch_model.atomic_numbers.tolist()
        num_elements = len(z_table)
        num_atoms = len(molecule["atomic_numbers"])

        node_attrs = torch.zeros(num_atoms, num_elements, dtype=torch.float32)
        for i, z in enumerate(molecule["atomic_numbers"]):
            node_attrs[i, z_table.index(z)] = 1.0

        edge_index = torch.tensor(molecule["edge_index"], dtype=torch.long)
        shifts = torch.tensor(molecule["shifts"], dtype=torch.float32)
        batch = torch.zeros(num_atoms, dtype=torch.long)

        data = {
            "positions": positions,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "shifts": shifts,
            "batch": batch,
            "ptr": torch.tensor([0, num_atoms], dtype=torch.long),
            "cell": torch.zeros(3, 3, dtype=torch.float32),
            "head": torch.zeros(1, dtype=torch.long),
        }

        output = torch_model(data, training=False, compute_force=True)

        return {
            "energy": output["energy"].detach().item(),
            "node_energy": output["node_energy"].detach().numpy(),
            "forces": output["forces"].detach().numpy()
            if output["forces"] is not None
            else None,
        }

    def _run_mlx_model(self, mlx_model, molecule, torch_model):
        """Run MLX MACE model on a molecule."""
        z_table = torch_model.atomic_numbers.tolist()
        num_elements = len(z_table)
        num_atoms = len(molecule["atomic_numbers"])

        positions = mx.array(molecule["positions"])
        node_attrs = mx.zeros((num_atoms, num_elements))
        for i, z in enumerate(molecule["atomic_numbers"]):
            idx = z_table.index(z)
            node_attrs = node_attrs.at[i, idx].add(1.0)

        edge_index = mx.array(molecule["edge_index"])
        shifts = mx.array(molecule["shifts"])

        output = mlx_model(positions, node_attrs, edge_index, shifts)
        mx.eval(output["energy"], output["node_energy"])

        return {
            "energy": output["energy"].item(),
            "node_energy": np.array(output["node_energy"]),
        }

    def test_energy_match(self, converted_model, water_molecule):
        """Test that MLX energy matches PyTorch for water molecule."""
        mlx_model, torch_model = converted_model

        torch_result = self._run_torch_model(torch_model, water_molecule)
        mlx_result = self._run_mlx_model(mlx_model, water_molecule, torch_model)

        print(f"PyTorch energy: {torch_result['energy']:.6f}")
        print(f"MLX energy:     {mlx_result['energy']:.6f}")
        print(f"Diff:           {abs(torch_result['energy'] - mlx_result['energy']):.6e}")

        np.testing.assert_allclose(
            mlx_result["energy"],
            torch_result["energy"],
            atol=1e-3,
            rtol=1e-3,
            err_msg="Energy mismatch between PyTorch and MLX",
        )

    def test_node_energy_match(self, converted_model, water_molecule):
        """Test that per-atom energies match."""
        mlx_model, torch_model = converted_model

        torch_result = self._run_torch_model(torch_model, water_molecule)
        mlx_result = self._run_mlx_model(mlx_model, water_molecule, torch_model)

        print(f"PyTorch node energies: {torch_result['node_energy']}")
        print(f"MLX node energies:     {mlx_result['node_energy']}")

        np.testing.assert_allclose(
            mlx_result["node_energy"],
            torch_result["node_energy"],
            atol=1e-3,
            rtol=1e-3,
            err_msg="Node energy mismatch",
        )

    def test_different_molecules(self, converted_model):
        """Test with different molecules to verify generalization."""
        mlx_model, torch_model = converted_model

        # Methane: C at origin, 4 H atoms
        molecules = [
            {
                "positions": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.629, 0.629, 0.629],
                        [-0.629, -0.629, 0.629],
                        [-0.629, 0.629, -0.629],
                        [0.629, -0.629, -0.629],
                    ],
                    dtype=np.float32,
                ),
                "atomic_numbers": [6, 1, 1, 1, 1],
                "edge_index": np.array(
                    [
                        [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 1, 2, 3, 4, 0, 2, 3, 0, 3, 0],
                        [1, 2, 3, 4, 0, 2, 3, 0, 3, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
                    ],
                    dtype=np.int32,
                ),
                "shifts": np.zeros((20, 3), dtype=np.float32),
            },
        ]

        for mol in molecules:
            torch_result = self._run_torch_model(torch_model, mol)
            mlx_result = self._run_mlx_model(mlx_model, mol, torch_model)

            np.testing.assert_allclose(
                mlx_result["energy"],
                torch_result["energy"],
                atol=1e-3,
                rtol=1e-3,
            )


class TestMACEMPAllSizes:
    """Parametrized tests for small, medium, and large MACE-MP-0 models.

    Verifies that energy and forces from MLX match PyTorch for each model size.
    """

    @pytest.fixture(scope="class", params=["small", "medium", "large"])
    def calc_pair(self, request):
        """Return (mlx_calc, torch_calc) for the given model size."""
        model_size = request.param
        try:
            import torch  # noqa: F401
            from mace.calculators import mace_mp
        except ImportError:
            pytest.skip("mace-torch not installed")

        from mace_mlx.calculators import MACEMLXCalculator

        mlx_calc = MACEMLXCalculator(model_path=model_size)
        torch_calc = mace_mp(
            model=model_size, device="cpu", default_dtype="float32"
        )
        return mlx_calc, torch_calc, model_size

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

    @staticmethod
    def _make_si():
        from ase.build import bulk

        return bulk("Si", "diamond", a=5.43)

    def test_water_energy(self, calc_pair):
        """Energy for water molecule matches PyTorch."""
        mlx_calc, torch_calc, size = calc_pair
        water = self._make_water()

        w_mlx = water.copy()
        w_mlx.calc = mlx_calc
        w_torch = water.copy()
        w_torch.calc = torch_calc

        e_mlx = w_mlx.get_potential_energy()
        e_torch = w_torch.get_potential_energy()

        print(f"[{size}] Water energy: MLX={e_mlx:.6f} Torch={e_torch:.6f} "
              f"diff={abs(e_mlx - e_torch):.2e}")

        np.testing.assert_allclose(
            e_mlx, e_torch, atol=1e-3,
            err_msg=f"[{size}] Water energy mismatch",
        )

    def test_water_forces(self, calc_pair):
        """Forces for water molecule match PyTorch."""
        mlx_calc, torch_calc, size = calc_pair
        water = self._make_water()

        w_mlx = water.copy()
        w_mlx.calc = mlx_calc
        w_torch = water.copy()
        w_torch.calc = torch_calc

        f_mlx = w_mlx.get_forces()
        f_torch = w_torch.get_forces()
        max_diff = np.abs(f_mlx - f_torch).max()

        print(f"[{size}] Water force max diff: {max_diff:.2e}")

        np.testing.assert_allclose(
            f_mlx, f_torch, atol=1e-2,
            err_msg=f"[{size}] Water force mismatch",
        )

    def test_si_energy(self, calc_pair):
        """Energy for bulk Si matches PyTorch."""
        mlx_calc, torch_calc, size = calc_pair
        si = self._make_si()

        si_mlx = si.copy()
        si_mlx.calc = mlx_calc
        si_torch = si.copy()
        si_torch.calc = torch_calc

        e_mlx = si_mlx.get_potential_energy()
        e_torch = si_torch.get_potential_energy()

        print(f"[{size}] Si energy: MLX={e_mlx:.6f} Torch={e_torch:.6f} "
              f"diff={abs(e_mlx - e_torch):.2e}")

        np.testing.assert_allclose(
            e_mlx, e_torch, atol=1e-3,
            err_msg=f"[{size}] Si energy mismatch",
        )
