# MACE-MLX

GPU-accelerated [MACE](https://github.com/ACEsuit/mace) interatomic potential inference on Apple Silicon, powered by [MLX](https://github.com/ml-explore/mlx).

## Why MACE-MLX?

MACE is the state-of-the-art machine learning interatomic potential, but on macOS:
- PyTorch's MPS backend doesn't work with MACE (float64 incompatibility, e3nn JIT issues)
- Inference is limited to CPU only

MACE-MLX solves this by reimplementing MACE's inference engine in MLX, Apple's ML framework optimized for Apple Silicon. **Result: 1.4x--4.8x faster than PyTorch CPU across all models.**

## Supported Models

All MACE Foundation Models are supported:

| Model Family | Variants | Status |
|---|---|---|
| MACE-MP-0 | small, medium, large | Supported |
| MACE-MP-0b | small, medium | Supported |
| MACE-MP-0b2 | small, medium, large | Supported |
| MACE-MP-0b3 | medium | Supported |
| MACE-MPA-0 | medium (new default) | Supported |
| MACE-OMAT-0 | small, medium | Supported |
| MACE-MatPES | PBE, R2SCAN | Supported |
| MACE-MH-1 | 6 heads (multi-head) | Supported |

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python** >= 3.10
- Models are auto-downloaded on first use (requires internet)

## Installation

```bash
pip install mace-mlx
```

Or with [uv](https://docs.astral.sh/uv/):
```bash
uv pip install mace-mlx
```

ASE and matscipy are included as core dependencies.

For development:
```bash
git clone https://github.com/ACEsuit/mace-mlx
cd mace-mlx
pip install -e ".[dev]"
```

## Quick Start

### Drop-in replacement for PyTorch MACE

```python
# Before (PyTorch MACE)
from mace.calculators import mace_mp
calc = mace_mp(model="small", device="cpu")

# After (MACE-MLX) — just change the import
from mace_mlx.calculators import mace_mp
calc = mace_mp(model="small")
```

### Full example

```python
from ase.build import bulk
from mace_mlx.calculators import mace_mp

# Create calculator (auto-downloads model on first use)
calc = mace_mp(model="medium-mpa-0")

# Use with any ASE Atoms object
si = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
si.calc = calc

energy = si.get_potential_energy()     # eV
forces = si.get_forces()              # eV/Ang
stress = si.get_stress()              # eV/Ang^3 (Voigt notation)

print(f"Energy: {energy:.4f} eV")
print(f"Max force: {forces.max():.4f} eV/Ang")
```

### Multi-head models

```python
from mace_mlx.calculators import mace_mp

# MACE-MH-1 with specific head
calc = mace_mp(model="mh-1", head="matpes_r2scan")
```

### Molecular dynamics

```python
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from mace_mlx.calculators import mace_mp

atoms = bulk('Cu', 'fcc', a=3.6) * (3, 3, 3)
atoms.calc = mace_mp(model="small")
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
dyn.run(100)  # 100 steps of NVE MD
```

### Geometry optimization

```python
from ase.build import bulk
from ase.optimize import BFGS
from mace_mlx.calculators import mace_mp

si = bulk('Si', 'diamond', a=5.5)  # slightly wrong lattice constant
si.calc = mace_mp(model="small")

opt = BFGS(si)
opt.run(fmax=0.01)
print(f"Optimized energy: {si.get_potential_energy():.4f} eV")
```

## Performance

Benchmarks on Apple Silicon, energy + forces computation:

### MACE-MP-0 Small (scalar hidden features)
| System | Atoms | MLX (ms) | PyTorch CPU (ms) | Speedup |
|--------|-------|----------|-------------------|---------|
| Water | 3 | 3.5 | 6.6 | 1.9x |
| Si 2x2x2 | 16 | 6 | 22 | 3.7x |
| Si 3x3x3 | 54 | 15 | 33 | 2.2x |
| Si 5x5x5 | 250 | 62 | 100 | 1.6x |
| Si 8x8x8 | 1024 | 264 | 364 | 1.4x |

### MACE-MPA-0 Medium (L>0 features, new default model)
| System | Atoms | MLX (ms) | PyTorch CPU (ms) | Speedup |
|--------|-------|----------|-------------------|---------|
| Si 2x2x2 | 16 | 14 | 41 | 3.0x |
| Si 5x5x5 | 250 | 195 | 368 | 1.9x |

### MACE-MH-1 Multi-head
| System | Atoms | MLX (ms) | PyTorch CPU (ms) | Speedup |
|--------|-------|----------|-------------------|---------|
| Si 2x2x2 | 16 | 25 | 74 | 2.9x |
| Si 5x5x5 | 250 | 388 | 776 | 2.0x |

## API Reference

### `mace_mp(model, device, default_dtype, head)`

Factory function matching `mace.calculators.mace_mp`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"small"` | Model name: `"small"`, `"medium"`, `"large"`, `"medium-mpa-0"`, `"mh-1"`, etc. |
| `device` | str | `"gpu"` | `"gpu"` (Apple Silicon) or `"cpu"` |
| `default_dtype` | str | `"float32"` | `"float32"` recommended |
| `head` | str \| None | `None` | Head name for multi-head models |

### `MACEMLXCalculator`

ASE Calculator class. Same parameters as `mace_mp()` plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | `"small"` | Model name or path to converted model |
| `skin` | float | `0.5` | Neighbor list cache skin distance (Ang) |

### `convert_mace_checkpoint(model_path, output_dir)`

Convert a PyTorch MACE checkpoint to MLX format.

## How It Works

MACE-MLX reimplements the complete MACE inference pipeline in MLX:

- **Equivariant operations**: Irreps, spherical harmonics, tensor products, symmetric contractions -- all built from scratch using CG coefficients + einsum/matmul
- **Automatic weight conversion**: Loads PyTorch MACE checkpoints and converts weights to MLX format
- **Force computation**: Via `mx.value_and_grad` (MLX autograd)
- **Stress/virials**: Via symmetric displacement tensor approach
- **Optimizations**: Batched CG matmul, scalar fast paths, vectorized Gate, Metal kernels, neighbor list caching (matscipy)

## Citation

If you use MACE-MLX, please cite the original MACE papers:

```bibtex
@article{batatia2022mace,
  title={MACE: Higher order equivariant message passing neural networks for fast and accurate force fields},
  author={Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor NC and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## License

MIT
