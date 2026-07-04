"""ASE Calculator interface for MACE-MLX.

Provides MACEMLXCalculator: a drop-in replacement for mace.calculators.MACECalculator
on macOS, using MLX for GPU-accelerated inference on Apple Silicon.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from mace_mlx.model import load_model

# Bump when converter output changes to invalidate cached conversions.
# v2: adds ZBL pair_repulsion extraction.
_CONVERTER_CACHE_VERSION = 2


def _conversion_cache_root() -> Path:
    """Directory for persistently cached converted models."""
    env = os.environ.get("MACE_MLX_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "mace_mlx"

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


_DEVICE_MAP = {"cuda": "gpu", "mps": "gpu", "gpu": "gpu", "cpu": "cpu", "": "gpu"}


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
        use_compile: Wrap the energy+forces step in mx.compile (default
            True). Each distinct edge count traces once and is cached;
            set False to disable (e.g. for debugging).
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
        use_compile: bool = True,
        **kwargs,
    ):
        # Accept mace-torch model_paths parameter
        if model_path is None:
            model_path = kwargs.pop("model_paths", None)
            if isinstance(model_path, (list, tuple)):
                if len(model_path) > 1:
                    raise NotImplementedError(
                        "MACE-MLX does not support committee models yet; "
                        f"got {len(model_path)} model paths. Pass a single "
                        "model, or use mace-torch for committee inference."
                    )
                model_path = model_path[0] if model_path else None
        if model_path is None:
            # Match mace-torch's mace_mp default (medium-mpa-0)
            model_path = "medium-mpa-0"

        if default_dtype == "float64":
            warnings.warn(
                "MLX does not support float64 on GPU; falling back to "
                "float32. Energies/forces will differ from mace-torch's "
                "float64 default at the float32 precision level."
            )
            default_dtype = "float32"

        # Map mace-torch device names to MLX equivalents
        device = _DEVICE_MAP.get(device, device)

        super().__init__(**kwargs)

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
            num_elements = int(getattr(self.model, "num_elements", 89))
            self._z_to_idx = {z: z for z in range(num_elements)}

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
        # Cached MLX edge arrays (uploaded once per NL rebuild)
        self._cached_edge_index_mx: mx.array | None = None
        self._cached_shifts_mx: mx.array | None = None

        # Cached batch MLX array (reused when num_atoms matches)
        self._cached_batch_mx: mx.array | None = None

        # Cached one-hot node attributes (rebuilt only when species change)
        self._cached_node_attrs_mx: mx.array | None = None
        self._cached_numbers: np.ndarray | None = None

        # Once stress has been explicitly requested for a periodic system,
        # compute it in the same forward/backward pass as energy+forces on
        # every subsequent step. Otherwise ASE's get_stress() after
        # get_forces() triggers a second full calculation per MD step.
        self._stress_always = False

        # Compiled step functions (built lazily; rebuilt on head change)
        self._use_compile = use_compile
        self._step_fn = None
        self._step_fn_head: int | None = None
        self._stress_step_fn = None
        self._stress_step_fn_head: int | None = None

        if device == "cpu":
            mx.set_default_device(mx.cpu)
        else:
            mx.set_default_device(mx.gpu)

    # ------------------------------------------------------------------ #
    # Model resolution
    # ------------------------------------------------------------------ #

    @classmethod
    def _resolve_model(cls, model_path: str) -> str:
        """Return path to an MLX model directory.

        Named models ("small", "medium-mpa-0", "off-medium", ...) are
        converted once and cached persistently under
        ``~/.cache/mace_mlx/<name>/v<version>/`` (override the root with
        the ``MACE_MLX_CACHE_DIR`` environment variable). Anything else is
        treated as an already-converted directory path.
        """
        is_named = model_path.startswith("off-")
        if not is_named:
            try:
                from mace.calculators.foundations_models import mace_mp_names

                is_named = model_path in mace_mp_names
            except ImportError:
                is_named = model_path in ("small", "medium", "large",
                                          "medium-mpa-0")

        if not is_named:
            return model_path

        if model_path in cls._converted_cache:
            cached = cls._converted_cache[model_path]
            if Path(cached).exists():
                return cached

        target = (
            _conversion_cache_root() / model_path
            / f"v{_CONVERTER_CACHE_VERSION}"
        )
        if (target / "config.json").exists() and (target / "weights.npz").exists():
            cls._converted_cache[model_path] = str(target)
            return str(target)

        # Convert into a temp dir on the same filesystem, then rename
        # atomically so concurrent processes never see a half-written cache.
        target.parent.mkdir(parents=True, exist_ok=True)
        tmpdir = tempfile.mkdtemp(
            prefix=f".{model_path}-converting-", dir=str(target.parent)
        )
        try:
            from mace_mlx.converter import convert_mace_checkpoint

            convert_mace_checkpoint(model_path, tmpdir)
        except ImportError as e:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise ImportError(
                f"Converting the named model '{model_path}' requires torch "
                "and mace-torch (one-time download + weight conversion; the "
                "result is cached). Install them with: "
                "pip install torch mace-torch"
            ) from e
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

        try:
            os.rename(tmpdir, target)
        except OSError:
            # Another process finished first — use its result.
            shutil.rmtree(tmpdir, ignore_errors=True)
            if not (target / "weights.npz").exists():
                raise
        cls._converted_cache[model_path] = str(target)
        return str(target)

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
            # Recompute derived arrays when NL is rebuilt; the MLX uploads
            # are reused across steps while the NL cache stays valid.
            self._cached_edge_index_np = np.stack([edge_src, edge_dst]).astype(np.int32)
            self._cached_shifts_np = shifts.astype(np.float32)
            self._cached_edge_index_mx = mx.array(self._cached_edge_index_np)
            self._cached_shifts_mx = mx.array(self._cached_shifts_np)
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
        self._get_neighbor_list(atoms)
        edge_index_mx = self._cached_edge_index_mx
        shifts_mx = self._cached_shifts_mx

        # -- One-hot node attributes (cached while species are unchanged) --
        atomic_numbers = atoms.get_atomic_numbers()
        num_atoms = len(atomic_numbers)

        if (
            self._cached_node_attrs_mx is None
            or self._cached_numbers is None
            or not np.array_equal(atomic_numbers, self._cached_numbers)
        ):
            if self.z_table is not None:
                num_elements = len(self.z_table)
            else:
                num_elements = int(getattr(self.model, "num_elements", 89))
            z_to_idx = self._z_to_idx

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
            indices = np.array(
                [z_to_idx[int(z)] for z in atomic_numbers], dtype=np.int32
            )
            node_attrs_np = np.zeros((num_atoms, num_elements), dtype=np.float32)
            node_attrs_np[np.arange(num_atoms), indices] = 1.0
            self._cached_node_attrs_mx = mx.array(
                node_attrs_np, dtype=self._compute_dtype
            )
            self._cached_numbers = atomic_numbers.copy()
        node_attrs_mx = self._cached_node_attrs_mx

        # -- Convert to mx.array ------------------------------------------
        # All geometric inputs (positions, shifts, cell) stay float32; the
        # model casts derived edge features to its compute dtype internally.
        positions_mx = mx.array(atoms.get_positions().astype(np.float32))

        has_cell = atoms.cell.rank > 0
        cell_mx = (
            mx.array(np.array(atoms.get_cell(), dtype=np.float32))
            if has_cell else None
        )

        if self._cached_batch_mx is None or self._cached_batch_mx.shape[0] != num_atoms:
            self._cached_batch_mx = mx.zeros(num_atoms, dtype=mx.int32)
        batch_mx = self._cached_batch_mx

        # -- Determine whether to compute stress ----------------------------
        # Sticky: once stress is explicitly requested for a periodic system,
        # keep computing it in the same pass as energy+forces (a stress-aware
        # MD loop would otherwise trigger two full calculations per step).
        if properties is not None and "stress" in properties and atoms.cell.rank == 3:
            self._stress_always = True
        compute_stress = "stress" in (properties or self.implemented_properties)
        compute_stress = (compute_stress or self._stress_always) and atoms.cell.rank == 3

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

    def _get_step_fn(self):
        """Return the (optionally compiled) energy+forces step function.

        The step is a pure function so it can be wrapped with mx.compile:
        the first call per input-shape traces and optimizes the graph, and
        subsequent MD steps replay the cached tape in C++ instead of
        rebuilding hundreds of lazy ops in Python. A neighbor-list rebuild
        changes num_edges and triggers a cheap retrace (~one step's time).

        Note: the model (weights, active head) is captured as a constant —
        the compiled function is rebuilt if the active head changes.
        """
        head = getattr(self.model, "_head_idx", 0)
        if self._step_fn is None or self._step_fn_head != head:
            model = self.model

            def energy_fn(pos, node_attrs, edge_index, shifts, cell, batch):
                # Positions stay float32: casting absolute coordinates to
                # half precision before the edge-vector subtraction destroys
                # force accuracy. The model casts derived features internally.
                out = model(pos, node_attrs, edge_index, shifts, cell, batch, 1)
                return out["energy"].sum(), out["node_energy"]

            # value_and_grad computes forward+backward in one pass; the aux
            # (node_energy) rides along without a side-effecting closure.
            vag = mx.value_and_grad(energy_fn)
            self._step_fn_raw = vag
            self._step_fn = mx.compile(vag) if self._use_compile else vag
            self._step_fn_head = head
        return self._step_fn

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
        step = self._get_step_fn()
        if edge_index.shape[1] == 0:
            # MLX 0.31.2 segfaults executing compiled Metal graphs with
            # zero-size inputs; the zero-edge case is trivial, run it eagerly.
            step = self._step_fn_raw
        (energy, node_energy), grads = step(
            positions, node_attrs, edge_index, shifts, cell, batch
        )
        forces = -grads
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
        step = self._get_stress_step_fn()
        if edge_index.shape[1] == 0:
            # See _compute_energy_and_forces: avoid compiled zero-size graphs
            step = self._stress_step_fn_raw
        displacement = mx.zeros((3, 3))
        (energy, node_energy), (grad_pos, grad_disp) = step(
            positions, displacement, node_attrs, edge_index, shifts, cell, batch
        )
        forces = -grad_pos
        mx.eval(energy, forces, grad_disp, node_energy)

        # Post-processing on the host: volume + Voigt assembly are 3x3
        # scalar ops that don't belong in the GPU graph.
        # Stress = dE/d_displacement / volume (matches PyTorch MACE
        # convention: computed from the raw gradient before sign flip).
        cell_np = np.array(cell)
        volume = abs(np.linalg.det(cell_np))
        stress_np = np.array(grad_disp) / volume
        # Voigt: xx, yy, zz, yz, xz, xy
        stress_voigt = stress_np[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
        return energy, forces, stress_voigt, node_energy

    def _get_stress_step_fn(self):
        """Return the (optionally compiled) energy+forces+virial step function."""
        head = getattr(self.model, "_head_idx", 0)
        if self._stress_step_fn is None or self._stress_step_fn_head != head:
            model = self.model

            def energy_fn(pos, displacement, node_attrs, edge_index, shifts,
                          cell, batch):
                # The strain math runs in float32 (positions/cell precision);
                # the model casts derived edge features internally.
                # Symmetric displacement (matching PyTorch MACE convention)
                sym_displacement = 0.5 * (displacement + displacement.T)

                # Apply strain: pos' = pos + pos @ disp, cell' = cell + cell @ disp
                pos_strained = pos + pos @ sym_displacement
                cell_strained = cell + cell @ sym_displacement

                # Strained edge vectors (shifts are integer lattice vectors)
                sender = edge_index[0]
                receiver = edge_index[1]
                vectors = pos_strained[receiver] - pos_strained[sender]
                vectors = vectors + shifts @ cell_strained

                # Clamped like utils.get_edge_vectors_and_lengths (finite
                # VJP for degenerate zero-length edges)
                sq = mx.sum(vectors * vectors, axis=-1, keepdims=True)
                lengths = mx.sqrt(mx.maximum(sq, 1e-24))

                energy, node_energy = model._forward_from_vectors_with_node_energy(
                    vectors, lengths, node_attrs, edge_index, batch, 1
                )
                return energy, node_energy

            vag = mx.value_and_grad(energy_fn, argnums=(0, 1))
            self._stress_step_fn_raw = vag
            self._stress_step_fn = mx.compile(vag) if self._use_compile else vag
            self._stress_step_fn_head = head
        return self._stress_step_fn


def mace_mp(model: str | None = None, device: str = "gpu",
            default_dtype: str = "float32", head: str | None = None,
            dispersion: bool = False, return_raw_model: bool = False,
            **kwargs) -> MACEMLXCalculator:
    """Load a MACE-MP foundation model.

    Drop-in replacement for ``mace.calculators.mace_mp``.

    Args:
        model: Model name — "small", "medium", "large", "mh-1", etc.
            Defaults to "medium-mpa-0", matching mace-torch's default.
        device: "gpu" (default, Apple Silicon) or "cpu".
        default_dtype: "float32" (default) or "float16".
        head: Head name for multi-head models (e.g., "matpes_r2scan").
        dispersion: Ignored (D3 dispersion not supported in MLX).
        return_raw_model: Not supported; raises NotImplementedError.
        **kwargs: Passed to MACEMLXCalculator.

    Returns:
        MACEMLXCalculator with the requested MACE-MP model loaded.
    """
    if return_raw_model:
        raise NotImplementedError("return_raw_model is not supported in MACE-MLX")
    if dispersion:
        warnings.warn(
            "MACE-MLX does not support D3 dispersion corrections. "
            "Ignoring dispersion=True."
        )
    # Pop mace-torch dispersion kwargs silently
    for key in ("damping", "dispersion_xc", "dispersion_cutoff"):
        kwargs.pop(key, None)
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


def mace_anicc(**kwargs) -> MACEMLXCalculator:
    """MACE-ANI is not yet supported in MACE-MLX."""
    raise NotImplementedError(
        "mace_anicc is not yet supported in MACE-MLX. "
        "Use mace-torch for ANI models."
    )


def mace_omol(**kwargs) -> MACEMLXCalculator:
    """MACE-OMOL is not yet supported in MACE-MLX."""
    raise NotImplementedError(
        "mace_omol is not yet supported in MACE-MLX. "
        "Use mace-torch for OMOL models."
    )


# Drop-in compatibility alias
MACECalculator = MACEMLXCalculator
