# MACE-MLX

Drop-in MLX replacement for [MACE](https://github.com/ACEsuit/mace) on Apple Silicon. **2-4x faster** than PyTorch CPU.

## Install

```bash
pip install mace-mlx
```

Named foundation models ("small", "medium-mpa-0", "off-medium", ...) are
downloaded and converted from the PyTorch checkpoints once, which needs
torch + mace-torch at conversion time:

```bash
pip install "mace-mlx[convert]"
```

The converted model is cached under `~/.cache/mace_mlx/` (override with
`MACE_MLX_CACHE_DIR`), so later runs — and environments without torch that
share the cache — load instantly.

For development:
```bash
git clone https://github.com/Mastreina/mace-mlx
cd mace-mlx
pip install -e ".[dev]"
```

## Quick Start

Change one import line -- everything else stays the same:

```python
# Before (PyTorch MACE)
from mace.calculators import mace_mp

# After (MACE-MLX)
from mace_mlx.calculators import mace_mp
```

Complete example:

```python
from ase.build import bulk
from mace_mlx.calculators import mace_mp

calc = mace_mp(model="medium-mpa-0")

si = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
si.calc = calc

energy = si.get_potential_energy()     # eV
forces = si.get_forces()              # eV/Ang
stress = si.get_stress()              # eV/Ang^3 (Voigt)

print(f"Energy: {energy:.4f} eV")
print(f"Max force: {forces.max():.4f} eV/Ang")
```

## Supported Models

| Model Family | Variants | Status |
|---|---|---|
| MACE-MP-0 | small, medium, large | Supported |
| MACE-MP-0b | small, medium | Supported |
| MACE-MP-0b2 | small, medium, large | Supported |
| MACE-MP-0b3 | medium | Supported |
| MACE-MPA-0 | medium (default) | Supported |
| MACE-OMAT-0 | small, medium | Supported |
| MACE-MatPES | PBE, R2SCAN | Supported |
| MACE-MH-1 | 6 heads (multi-head) | Supported |
| MACE-OFF23 | small, medium, large (`mace_off`) | Supported |

The mpa-0/0b/0b2/0b3 family's ZBL pair repulsion is included, so
short-range/high-pressure configurations match mace-torch.

## Performance

MACE-MP-0 Small on Apple Silicon (energy + forces, v0.2.0 measurements):

| System | Atoms | MLX (ms) | CPU (ms) | MPS (ms) |
|--------|-------|----------|----------|----------|
| Water | 3 | 3.5 | 8.0 | 16.7 |
| Si 2x2x2 | 16 | 4.1 | 16.3 | 17.5 |
| Cu 3x3x3 | 27 | 7.6 | 25.7 | 21.3 |
| Si 3x3x3 | 54 | 10.9 | 31.7 | 25.5 |
| Al 3x3x3 | 27 | 6.0 | 21.5 | 19.8 |

Since v0.2.0, per-step time improved further on top of the numbers above
(measured on M4 Pro, same systems/models): medium models (`medium-mpa-0`
default) run **1.3-1.5x faster** (e.g. Si 1000 atoms: 678 -> 503 ms), large
1.2-1.3x, small 1.04-1.07x, via a batched second-layer tensor product,
`mx.compile`, and per-step caching. See `docs/OPTIMIZATION_REVIEW.md`.

## API

**`mace_mp(model=None, device="gpu", default_dtype="float32", head=None)`**
Factory function matching `mace.calculators.mace_mp`, including the default
model (`medium-mpa-0`). Auto-downloads and converts models on first use.

**`mace_off(model="small", device="gpu", default_dtype="float32")`**
Factory function for MACE-OFF organic chemistry models.

**`MACECalculator`** (alias: `MACEMLXCalculator`)
ASE Calculator class. Accepts the same parameters plus `model_path`, `skin`
(neighbor list cache distance, default 0.5 Ang) and `use_compile`
(mx.compile the energy+forces step, default True).

## Differences vs mace-torch

- `default_dtype` defaults to float32 (MLX has no float64 on GPU; passing
  `"float64"` warns and falls back to float32). Expect float32-level
  agreement (~1e-5 eV/A in forces) against torch's float64 results.
- `float16` runs the feature path in half precision while keeping geometry,
  radial basis, E0, and energy accumulation in float32 (~0.6-1.2 meV/atom
  vs float32; no speed advantage on M-series, treat as experimental).
- Committee models (a list in `model_paths`) are not supported and raise.
- `dispersion=True` is ignored — combine with a CPU D3 calculator via
  ASE's `SumCalculator` if needed.
- `return_raw_model` is not supported.
- Once `get_stress()` has been called on a periodic system, stress is
  computed in the same forward/backward pass as energy+forces on every
  subsequent step (NPT-friendly; one calculation per MD step).

## Citation

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
