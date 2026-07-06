# mace-mlx

Drop-in MLX replacement for [MACE](https://github.com/ACEsuit/mace) on Apple Silicon. 

Fully optimized by `claude-fable-5`.

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

Energy + forces per step on Apple M4 Pro (48 GB), rattled bulk Si,
`medium-mpa-0` (the default model), fp32:

| Configuration | Si 1000 atoms | Si 2000 atoms |
|---|---:|---:|
| mace-torch cpu, float64 (its default) | 2101 ms | 4293 ms |
| mace-torch cpu, float32 | 1181 ms | 2355 ms |
| mace-mlx 0.3.0 | 535 ms | 1543 ms |
| mace-mlx 0.4.0 (sparse symmetric contraction) | 379 ms | 763 ms |
| **mace-mlx 0.5.0 (fused Metal kernels)** | **137 ms** | **275 ms** |

That is ~15x over mace-torch's official default and ~9x at equal (fp32)
precision, with peak memory 5.3 GB (Si1000) / 8.3 GB (Si2000). mace-torch's
own MPS backend does not run out of the box (float64 checkpoints and a
hardcoded `.double()` in forward). `default_dtype="float16"` gives a
further ~1.45x where its accuracy fits (see below). Smaller L=0 models
(`small`) gain less from the fused kernels (~110 ms / Si1000). Benchmarks
and raw data: `docs/prototypes/`.

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
  radial basis, E0, and energy accumulation in float32. It is ~1.45x faster
  and validated per use case (details in
  `docs/prototypes/team_fp16_report.md`): fine for NVT/NPT MD and
  relaxations down to fmax≈0.01 eV/A (force error <=1% rel-RMS,
  ~1 meV/atom near equilibrium); avoid for phonons/Hessians (finite-
  difference force constants), tight relaxations (fmax<0.005), and
  absolute-energy comparisons of highly strained structures (systematic
  shifts up to ~10 meV/atom observed).
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
