"""MACE-MLX: MACE interatomic potential inference on Apple Silicon via MLX."""

__version__ = "0.1.0"

from mace_mlx.calculators import MACEMLXCalculator, mace_mp, mace_off
from mace_mlx.converter import convert_mace_checkpoint
from mace_mlx.model import MACE, ScaleShiftMACE, load_model
