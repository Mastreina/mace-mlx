# MACE-MLX

Drop-in MLX replacement for [MACE](https://github.com/ACEsuit/mace) on Apple Silicon. **2-4x faster** than PyTorch CPU.

## Install

```bash
pip install mace-mlx
```

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
| MACE-MPA-0 | medium (new default) | Supported |
| MACE-OMAT-0 | small, medium | Supported |
| MACE-MatPES | PBE, R2SCAN | Supported |
| MACE-MH-1 | 6 heads (multi-head) | Supported |

## Performance

MACE-MP-0 Small on Apple Silicon (energy + forces):

| System | Atoms | MLX (ms) | CPU (ms) | MPS (ms) |
|--------|-------|----------|----------|----------|
| Water | 3 | 3.5 | 8.0 | 16.7 |
| Si 2x2x2 | 16 | 4.1 | 16.3 | 17.5 |
| Cu 3x3x3 | 27 | 7.6 | 25.7 | 21.3 |
| Si 3x3x3 | 54 | 10.9 | 31.7 | 25.5 |
| Al 3x3x3 | 27 | 6.0 | 21.5 | 19.8 |

## API

**`mace_mp(model="small", device="gpu", default_dtype="float32", head=None)`**
Factory function matching `mace.calculators.mace_mp`. Auto-downloads models on first use.

**`mace_off(model="small", device="gpu", default_dtype="float32")`**
Factory function for MACE-OFF organic chemistry models.

**`MACECalculator`** (alias: `MACEMLXCalculator`)
ASE Calculator class. Accepts the same parameters plus `model_path` and `skin` (neighbor list cache distance, default 0.5 Ang).

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
