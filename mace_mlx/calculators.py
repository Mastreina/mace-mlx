"""ASE Calculator interface for MACE-MLX.

Provides MACEMLXCalculator: a drop-in replacement for mace.calculators.MACECalculator
on macOS, using MLX for GPU-accelerated inference on Apple Silicon.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from mace_mlx.model import load_model

# Prefer matscipy (C-compiled, ~10-30x faster) over ASE's pure-Python version.
from ase.neighborlist import primitive_neighbor_list as _ase_nl

try:
    from matscipy.neighbours import neighbour_list as _matscipy_nl

    _USE_MATSCIPY = True
except ImportError:
    _USE_MATSCIPY = False


def _build_neighbor_list(
    atoms, r_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build neighbor list, returning (edge_src, edge_dst, shifts).

    Uses matscipy when available (C-compiled, ~10-30x faster than ASE).
    Falls back to ASE for non-periodic systems (matscipy requires an
    invertible cell matrix).
    """
    if _USE_MATSCIPY and atoms.cell.rank == 3:
        return _matscipy_nl("ijS", atoms, r_max)
    return _ase_nl("ijS", atoms.pbc, atoms.cell, atoms.positions, r_max)


class MACEMLXCalculator(Calculator):
    """ASE Calculator using MACE-MLX for energy and force predictions.

    Drop-in replacement for mace.calculators.MACECalculator on macOS.
    Uses MLX for GPU-accelerated inference on Apple Silicon.

    Args:
        model_path: Path to a converted MLX model directory, or one of
            "small", "medium", "large", "mh-1" to auto-convert a MACE-MP model.
        device: "gpu" or "cpu" (MLX device selection).
        default_dtype: "float32" or "float16".
        head: Head name for multi-head models (e.g., "matpes_r2scan").
            If not specified, defaults to the first head.
        skin: Skin distance (Angstrom) for neighbor list caching.  The
            neighbor list is rebuilt only when the maximum atomic
            displacement exceeds skin / 2.  Set to 0 to disable caching.
    """

    implemented_properties = ["energy", "energies", "free_energy", "node_energy", "forces", "stress"]

    # Class-level cache for converted MACE-MP models
    _converted_cache: dict[str, str] = {}

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "gpu",
        default_dtype: str = "float32",
        head: str | None = None,
        skin: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if model_path is None:
            model_path = "small"

        model_dir = self._resolve_model(model_path)
        self.model = load_model(model_dir, dtype=default_dtype)

        # Multi-head support
        self._head = head
        if hasattr(self.model, "heads") and self.model.num_heads > 1:
            if self._head is None:
                self._head = self.model.heads[0]
            if self._head not in self.model.heads:
                raise ValueError(
                    f"Head '{self._head}' not found in model heads: {self.model.heads}"
                )
            self.model._head_idx = self.model.heads.index(self._head)

        self.r_max = float(self.model.r_max)
        self.z_table: list[int] = getattr(self.model, "z_table", None)
        self._compute_dtype = getattr(self.model, "_compute_dtype", mx.float32)

        # Precompute z -> index mapping for fast one-hot encoding
        if self.z_table is not None:
            self._z_to_idx = {int(z): i for i, z in enumerate(self.z_table)}
        else:
            self._z_to_idx = None

        # Neighbor list cache
        self._skin = skin
        self._nl_cache: dict | None = None
        self._cache_positions: np.ndarray | None = None
        self._cache_cell: np.ndarray | None = None
        self._cache_pbc: np.ndarray | None = None
        self._cache_natoms: int = 0

        # Cached numpy edge arrays (reused when NL cache hits)
        self._cached_edge_index_np: np.ndarray | None = None
        self._cached_shifts_np: np.ndarray | None = None

        # Cached batch MLX array (reused when num_atoms matches)
        self._cached_batch_mx: mx.array | None = None

        if device == "cpu":
            mx.set_default_device(mx.cpu)
        else:
            mx.set_default_device(mx.gpu)

        # Enable fused gather-TP-scatter (autograd-safe pure-MLX path)
        self.model.set_fused_kernel(True)

    # ------------------------------------------------------------------ #
    # Model resolution
    # ------------------------------------------------------------------ #

    @classmethod
    def _resolve_model(cls, model_path: str) -> str:
        """Return path to an MLX model directory.

        If model_path is "small"/"medium"/"large", convert the corresponding
        MACE-MP-0 checkpoint (cached across instances).  Otherwise treat it
        as an already-converted directory path.
        """
        # Check if it's a named MACE-MP model
        try:
            from mace.calculators.foundations_models import mace_mp_names

            is_named = model_path in mace_mp_names
        except ImportError:
            is_named = model_path in ("small", "medium", "large")

        if is_named:
            if model_path in cls._converted_cache:
                cached = cls._converted_cache[model_path]
                if Path(cached).exists():
                    return cached

            from mace_mlx.converter import convert_mace_checkpoint

            tmpdir = tempfile.mkdtemp(prefix=f"mace_mlx_{model_path}_")
            convert_mace_checkpoint(model_path, tmpdir)
            cls._converted_cache[model_path] = tmpdir
            return tmpdir

        return model_path

    # ------------------------------------------------------------------ #
    # Neighbor list with caching
    # ------------------------------------------------------------------ #

    def _needs_neighbor_update(self, atoms) -> bool:
        """Check if the neighbor list needs rebuilding."""
        if self._nl_cache is None:
            return True
        if len(atoms) != self._cache_natoms:
            return True
        # Cell or PBC changed
        if not np.array_equal(atoms.pbc, self._cache_pbc):
            return True
        if not np.allclose(np.array(atoms.cell), self._cache_cell, atol=1e-10):
            return True
        # Displacement check
        if self._skin <= 0:
            return True
        max_disp = np.max(
            np.linalg.norm(atoms.positions - self._cache_positions, axis=1)
        )
        return max_disp > self._skin / 2

    def _get_neighbor_list(self, atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached or freshly-built neighbor list."""
        if self._needs_neighbor_update(atoms):
            edge_src, edge_dst, shifts = _build_neighbor_list(
                atoms, self.r_max + self._skin
            )
            self._nl_cache = {
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "shifts": shifts,
            }
            self._cache_positions = atoms.positions.copy()
            self._cache_cell = np.array(atoms.cell, dtype=np.float64)
            self._cache_pbc = atoms.pbc.copy()
            self._cache_natoms = len(atoms)
            # Recompute derived numpy arrays when NL is rebuilt
            self._cached_edge_index_np = np.stack([edge_src, edge_dst]).astype(np.int32)
            self._cached_shifts_np = shifts.astype(np.float32)
        return (
            self._nl_cache["edge_src"],
            self._nl_cache["edge_dst"],
            self._nl_cache["shifts"],
        )

    # ------------------------------------------------------------------ #
    # ASE calculate
    # ------------------------------------------------------------------ #

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        """Compute energy and forces for the given atoms object."""
        super().calculate(atoms, properties, system_changes)

        # -- Neighbor list ------------------------------------------------
        edge_src, edge_dst, shifts_frac = self._get_neighbor_list(atoms)
        edge_index_np = self._cached_edge_index_np
        shifts_np = self._cached_shifts_np

        # -- One-hot node attributes --------------------------------------
        atomic_numbers = atoms.get_atomic_numbers()
        num_atoms = len(atomic_numbers)

        if self._z_to_idx is not None:
            num_elements = len(self.z_table)
            z_to_idx = self._z_to_idx
        else:
            # Fallback: treat atomic numbers directly as element indices
            num_elements = int(getattr(self.model, "num_elements", 89))
            z_to_idx = {z: z for z in range(num_elements)}

        # Validate all z values upfront
        for z in atomic_numbers:
            z_int = int(z)
            if z_int not in z_to_idx:
                import ase.data
                raise ValueError(
                    f"Atomic number {z_int} ({ase.data.chemical_symbols[z_int]}) "
                    f"is not supported by this model. "
                    f"Supported: {sorted(z_to_idx.keys())}"
                )
        indices = np.array([z_to_idx[int(z)] for z in atomic_numbers], dtype=np.int32)
        node_attrs_np = np.zeros((num_atoms, num_elements), dtype=np.float32)
        node_attrs_np[np.arange(num_atoms), indices] = 1.0

        # -- Convert to mx.array ------------------------------------------
        # Positions always float32 for accurate force gradients via autograd.
        # Other float inputs match the model's compute dtype.
        positions_mx = mx.array(atoms.get_positions().astype(np.float32))
        node_attrs_mx = mx.array(node_attrs_np).astype(self._compute_dtype)
        edge_index_mx = mx.array(edge_index_np)
        shifts_mx = mx.array(shifts_np).astype(self._compute_dtype)

        cell_np = np.array(atoms.get_cell(), dtype=np.float32)
        has_cell = atoms.cell.rank > 0
        cell_mx = mx.array(cell_np).astype(self._compute_dtype) if has_cell else None

        if self._cached_batch_mx is None or self._cached_batch_mx.shape[0] != num_atoms:
            self._cached_batch_mx = mx.zeros(num_atoms, dtype=mx.int32)
        batch_mx = self._cached_batch_mx

        # -- Determine whether to compute stress ----------------------------
        compute_stress = "stress" in (properties or self.implemented_properties)
        compute_stress = compute_stress and atoms.cell.rank == 3

        # -- Energy, forces, and optionally stress --------------------------
        if compute_stress:
            energy, forces, stress, node_energy = self._compute_energy_forces_stress(
                positions_mx,
                node_attrs_mx,
                edge_index_mx,
                shifts_mx,
                cell_mx,
                batch_mx,
                num_graphs=1,
            )
            self.results["stress"] = np.array(stress)
        else:
            energy, forces, node_energy = self._compute_energy_and_forces(
                positions_mx,
                node_attrs_mx,
                edge_index_mx,
                shifts_mx,
                cell_mx,
                batch_mx,
                num_graphs=1,
            )

        self.results["energy"] = float(energy.item())
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = np.array(forces)
        # Per-atom energy decomposition (node_energy / energies)
        self.results["node_energy"] = np.array(node_energy)
        self.results["energies"] = self.results["node_energy"]

    # ------------------------------------------------------------------ #
    # Energy + forces via autograd
    # ------------------------------------------------------------------ #

    def _compute_energy_and_forces(
        self,
        positions: mx.array,
        node_attrs: mx.array,
        edge_index: mx.array,
        shifts: mx.array,
        cell: mx.array | None,
        batch: mx.array,
        num_graphs: int = 1,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Compute energy (scalar), forces (num_atoms, 3), and node_energy via value_and_grad."""
        compute_dtype = self._compute_dtype

        # Stop gradient on non-position inputs to reduce autograd graph cost
        node_attrs = mx.stop_gradient(node_attrs)
        edge_index = mx.stop_gradient(edge_index)
        shifts = mx.stop_gradient(shifts)
        if cell is not None:
            cell = mx.stop_gradient(cell)
        batch = mx.stop_gradient(batch)

        # Mutable container to capture node_energy from the forward pass
        captured = {}

        def energy_fn(pos):
            # Cast positions to model dtype for FP16/BF16 speedup.
            # Autograd flows back through astype to produce float32 gradients.
            pos_cast = pos.astype(compute_dtype) if compute_dtype != mx.float32 else pos
            result = self.model(
                pos_cast, node_attrs, edge_index, shifts, cell, batch, num_graphs
            )
            captured["node_energy"] = result["node_energy"]
            return result["energy"].sum()

        # value_and_grad computes forward+backward in one pass,
        # avoiding redundant forward computation vs separate grad() call
        energy, grads = mx.value_and_grad(energy_fn)(positions)
        forces = -grads
        node_energy = captured["node_energy"]
        mx.eval(energy, forces, node_energy)
        return energy, forces, node_energy

    # ------------------------------------------------------------------ #
    # Energy + forces + stress via displacement tensor
    # ------------------------------------------------------------------ #

    def _compute_energy_forces_stress(
        self,
        positions: mx.array,
        node_attrs: mx.array,
        edge_index: mx.array,
        shifts: mx.array,
        cell: mx.array,
        batch: mx.array,
        num_graphs: int = 1,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Compute energy, forces, stress, and node_energy via the displacement tensor approach.

        Follows the symmetric displacement method from PyTorch MACE:
        1. Create symmetric displacement tensor (3x3) as differentiable input
        2. Apply strain to positions and cell
        3. Differentiate energy w.r.t. positions (forces) and displacement (virials)
        4. stress = virials / volume in Voigt notation

        Returns:
            (energy, forces, stress_voigt, node_energy) where stress_voigt is (6,)
            in Voigt order: xx, yy, zz, yz, xz, xy
        """
        model = self.model
        compute_dtype = self._compute_dtype

        # Stop gradient on non-position inputs to reduce autograd graph cost
        node_attrs = mx.stop_gradient(node_attrs)
        edge_index = mx.stop_gradient(edge_index)
        shifts = mx.stop_gradient(shifts)
        if cell is not None:
            cell = mx.stop_gradient(cell)
        batch = mx.stop_gradient(batch)

        # Mutable container to capture node_energy from the forward pass
        captured = {}

        def energy_fn(pos, displacement):
            # Cast to compute dtype for FP16/BF16 speedup
            if compute_dtype != mx.float32:
                pos = pos.astype(compute_dtype)
                displacement = displacement.astype(compute_dtype)

            # Symmetric displacement (matching PyTorch MACE convention)
            sym_displacement = 0.5 * (displacement + displacement.T)

            # Apply strain to positions: pos' = pos + pos @ sym_displacement
            # Matches PyTorch: einsum("be,bec->bc", pos, disp) = pos @ disp
            pos_strained = pos + pos @ sym_displacement

            # Apply strain to cell: cell' = cell + cell @ sym_displacement
            # Matches PyTorch: torch.matmul(cell, symmetric_displacement)
            cell_strained = cell + cell @ sym_displacement

            # Compute strained edge vectors directly
            sender = edge_index[0]
            receiver = edge_index[1]
            vectors = pos_strained[receiver] - pos_strained[sender]
            # shifts are integer lattice vectors; multiply by strained cell
            vectors = vectors + shifts @ cell_strained

            lengths = mx.sqrt(mx.sum(vectors * vectors, axis=-1, keepdims=True))

            # Use _forward_from_vectors_with_node_energy to also capture node_energy
            energy, node_energy = model._forward_from_vectors_with_node_energy(
                vectors, lengths, node_attrs, edge_index, batch, num_graphs
            )
            captured["node_energy"] = node_energy
            return energy

        displacement = mx.zeros((3, 3))

        vag = mx.value_and_grad(energy_fn, argnums=(0, 1))
        energy, (grad_pos, grad_disp) = vag(positions, displacement)

        forces = -grad_pos
        node_energy = captured["node_energy"]
        # Virials = -dE/d_displacement (for storage/reporting)
        # Stress = dE/d_displacement / volume (matches PyTorch MACE convention:
        # stress is computed from raw gradient before sign flip)

        # Convert gradient to stress in Voigt notation
        # 3x3 determinant (mlx.linalg has no det)
        volume = mx.abs(
            cell[0, 0] * (cell[1, 1] * cell[2, 2] - cell[1, 2] * cell[2, 1])
            - cell[0, 1] * (cell[1, 0] * cell[2, 2] - cell[1, 2] * cell[2, 0])
            + cell[0, 2] * (cell[1, 0] * cell[2, 1] - cell[1, 1] * cell[2, 0])
        )
        stress_tensor = grad_disp / volume

        # Voigt: xx, yy, zz, yz, xz, xy
        stress_voigt = mx.array([
            stress_tensor[0, 0],
            stress_tensor[1, 1],
            stress_tensor[2, 2],
            stress_tensor[1, 2],
            stress_tensor[0, 2],
            stress_tensor[0, 1],
        ])

        mx.eval(energy, forces, stress_voigt, node_energy)
        return energy, forces, stress_voigt, node_energy


def mace_mp(model: str = "small", device: str = "gpu",
            default_dtype: str = "float32", head: str | None = None,
            **kwargs) -> MACEMLXCalculator:
    """Load a MACE-MP foundation model.

    Drop-in replacement for ``mace.calculators.mace_mp``.

    Args:
        model: Model name — "small", "medium", "large", "mh-1", etc.
        device: "gpu" (default, Apple Silicon) or "cpu".
        default_dtype: "float32" (default) or "float16".
        head: Head name for multi-head models (e.g., "matpes_r2scan").
        **kwargs: Passed to MACEMLXCalculator.

    Returns:
        MACEMLXCalculator with the requested MACE-MP model loaded.
    """
    return MACEMLXCalculator(model_path=model, device=device,
                              default_dtype=default_dtype, head=head, **kwargs)


def mace_off(model: str = "small", device: str = "gpu",
             default_dtype: str = "float32", **kwargs) -> MACEMLXCalculator:
    """Load a MACE-OFF organic force field model.

    Drop-in replacement for ``mace.calculators.mace_off``.

    Args:
        model: Model size — "small", "medium", or "large".
        device: "gpu" (default, Apple Silicon) or "cpu".
        default_dtype: "float32" (default) or "float16".
        **kwargs: Passed to MACEMLXCalculator.

    Returns:
        MACEMLXCalculator with the requested MACE-OFF model loaded.
    """
    return MACEMLXCalculator(model_path=f"off-{model}", device=device,
                              default_dtype=default_dtype, **kwargs)
