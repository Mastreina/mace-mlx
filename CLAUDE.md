# MACE-MLX Development Guide

## Quick Start
```bash
git clone https://github.com/ACEsuit/mace-mlx
cd mace-mlx
pip install -e ".[dev]"
pytest tests/ -m "not benchmark"
```

## Project Structure
```
mace_mlx/
├── irreps.py              # O(3) irreducible representations
├── clebsch_gordan.py       # CG coefficients and U matrices
├── spherical_harmonics.py  # Real spherical harmonics (CG recursion)
├── tensor_product.py       # TensorProduct + FullyConnectedTensorProduct
├── linear.py               # Equivariant linear layer
├── gate.py                 # Gate activation
├── symmetric_contraction.py # MACE body-order contraction
├── radial.py               # Bessel/Gaussian basis + polynomial cutoff
├── blocks.py               # Interaction, Product, Readout blocks
├── model.py                # MACE + ScaleShiftMACE models
├── converter.py            # PyTorch → MLX weight converter
├── calculators.py          # ASE Calculator + mace_mp/mace_off
├── kernels.py              # Custom Metal kernels
└── utils.py                # scatter_sum, edge geometry, constants
```

## Running Tests
```bash
pytest tests/ -q                     # All tests
pytest tests/ -m "not benchmark"     # Skip slow benchmarks
pytest tests/test_model.py -v        # Specific module
```

## Adding a New Model
1. Load in PyTorch, check architecture: `type(model).__name__`, interaction block types
2. Update `converter.py`: `_extract_config`, `_load_torch_model`, weight mapping
3. Update `model.py` if new block type needed
4. Add end-to-end test in `tests/test_model.py`
5. Cross-validate: energy and forces must match PyTorch within 1e-3

## Key Design Decisions
- SH uses standard m-ordering; e3nn basis rotation applied via precomputed matrix in model forward
- TensorProduct uses batched CG block-diagonal matmul for scalar inputs
- SymmetricContraction uses weights-first matmul decomposition
- Forces via `mx.value_and_grad` (forward+backward in one pass)
- Neighbor list uses matscipy when available (~30x faster than ASE)
- `mx.stop_gradient` on non-position-dependent inputs to reduce backward cost
