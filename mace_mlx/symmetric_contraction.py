"""Symmetric contraction for MACE body-order correlations.

MLX implementation matching MACE's SymmetricContraction module.
Uses precomputed U matrices (from CG coefficients) to contract
node features into body-order invariants/equivariants.

Performance: The forward pass uses a "weights-first" matmul decomposition
that contracts the parameter dimension (k) with element-selected weights
before contracting the coupling dimension (i) with features. This reduces
intermediate tensor size from O(batch * features * prefix * k) to
O(batch * features * prefix * i), yielding ~3x speedup for L>0 outputs
on Apple Silicon by reducing memory bandwidth pressure.

Reference: Batatia et al., MACE: Higher Order Equivariant Message Passing
Neural Networks for Fast and Accurate Force Fields, Eq. 10 and 11.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mace_mlx.clebsch_gordan import U_matrix_real
from mace_mlx.irreps import Irrep, Irreps

ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


def _build_einsum_strings(
    correlation: int, ir_out_lmax: int
) -> tuple[str, list[tuple[str, str]]]:
    """Build einsum equation strings for each contraction step.

    Returns the main (highest-nu) equation and a list of
    (weighting_eq, features_eq) for lower nu values.
    """
    min_l = min(ir_out_lmax, 1)

    # Highest correlation (main equation)
    prefix = [ALPHABET[j] for j in range(correlation + min_l - 1)]
    main_eq = "".join(prefix) + "ik,ekc,bci,be->bc" + "".join(prefix)

    lower_eqs = []
    for i in range(correlation - 1, 0, -1):
        # Weighting equation
        prefix_w = [ALPHABET[j] for j in range(i + min_l)]
        weighting_eq = "".join(prefix_w) + "k,ekc,be->bc" + "".join(prefix_w)

        # Features equation
        prefix_f = [ALPHABET[j] for j in range(i - 1 + min_l)]
        features_eq = "bc" + "".join(prefix_f) + "i,bci->bc" + "".join(prefix_f)

        lower_eqs.append((weighting_eq, features_eq))

    return main_eq, lower_eqs


class Contraction(nn.Module):
    """Single contraction for one output irrep.

    Implements the recursive contraction using precomputed U matrices
    and learnable element-dependent weights.

    The forward pass uses a "weights-first" matmul decomposition:

    Original main einsum: "...ik,ekc,bci,be->bc..."
    Weights-first decomposition:
      1. W_sel = onehot @ W_reshaped          -- select weights by element
      2. WU = W_sel @ U_wf                    -- contract k: (b,c,k) @ (k,prefix*i)
      3. result = WU_reshaped @ feat_col      -- contract i: (b,c,prefix,i) @ (b,c,i,1)

    This contracts k (parameter dim, typically 23-51) before i (coupling dim,
    typically 16), producing smaller intermediates than the i-first approach.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        ir_out: Irrep,
        correlation: int,
        num_elements: int,
    ):
        super().__init__()
        self.num_features = irreps_in.count(Irrep("0e"))
        self.coupling_irreps = Irreps(
            [(1, mulir.ir) for mulir in irreps_in]
        )
        self.correlation = correlation
        self.ir_out = ir_out
        self.num_elements = num_elements

        ir_out_irreps = Irreps([(1, ir_out)])

        # Precompute U matrices and build einsum equations
        self._u_matrices = {}
        self._path_weights = []  # whether each nu has nonzero U

        # Precomputed U matrix reshapes for weights-first matmul decomposition
        self._u_main_wf = None  # (k, prefix*i) for W-first main einsum
        self._u_main_prefix_shape = None
        self._u_main_i_dim = 0
        self._u_main_k_dim = 0
        self._u_lower_2d = []  # (prefix, k) for weight einsums
        self._u_lower_prefix_shapes = []

        for nu in range(1, correlation + 1):
            U_list = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=ir_out_irreps,
                correlation=nu,
            )
            U_np = U_list[-1]  # Last element is the tensor
            self._path_weights.append(np.any(np.abs(U_np) > 1e-15))
            self._u_matrices[nu] = mx.array(U_np, dtype=mx.float32)

        # Build einsum strings (kept for reference/testing)
        self._main_eq, self._lower_eqs = _build_einsum_strings(
            correlation, ir_out.l
        )

        # Precompute U matrix reshapes for the main einsum.
        # U matrices are constants (CG-derived), so stop_gradient prevents
        # autograd from tracking unnecessary backward computation through them.
        U_main = self._u_matrices[correlation]
        u_shape = U_main.shape
        # Last two dims are (i, k), everything before is prefix
        self._u_main_prefix_shape = u_shape[:-2]
        self._u_main_prefix_size = math.prod(self._u_main_prefix_shape)
        i_dim = u_shape[-2]
        k_dim = u_shape[-1]
        self._u_main_i_dim = i_dim
        self._u_main_k_dim = k_dim

        # Weights-first U reshape: move k to front, flatten to (k, prefix*i)
        # Original U shape: (*prefix, i, k)
        # Step: move k to front -> (k, *prefix, i) -> reshape to (k, prefix*i)
        U_wf = mx.moveaxis(U_main, -1, 0)  # (k, *prefix, i)
        self._u_main_wf = mx.stop_gradient(
            U_wf.reshape(k_dim, self._u_main_prefix_size * i_dim)
        )

        # Precompute U reshapes for lower-order weight einsums
        for nu in range(correlation - 1, 0, -1):
            U_lower = self._u_matrices[nu]
            u_shape_l = U_lower.shape
            # Last dim is k, everything before is prefix
            prefix_shape_l = u_shape_l[:-1]
            prefix_size_l = math.prod(prefix_shape_l)
            k_dim_l = u_shape_l[-1]
            U_2d_l = mx.stop_gradient(U_lower.reshape(prefix_size_l, k_dim_l))  # (prefix, k)
            self._u_lower_2d.append(U_2d_l)
            self._u_lower_prefix_shapes.append(prefix_shape_l)

        # Create weight parameters
        # weights_max: for the highest correlation order
        num_params_max = self._u_matrices[correlation].shape[-1]
        self.weights_max = mx.random.normal(
            shape=(num_elements, num_params_max, self.num_features)
        ) / num_params_max

        # weights: for lower correlation orders (stored in reverse order:
        # weights[0] corresponds to nu=correlation-1, etc.)
        self.weights = []
        for i, nu in enumerate(range(correlation - 1, 0, -1)):
            num_params = self._u_matrices[nu].shape[-1]
            w = mx.random.normal(
                shape=(num_elements, num_params, self.num_features)
            ) / num_params
            self.weights.append(w)

        # Zero out weights for correlation orders with zero U matrices
        # _path_weights[nu-1] tells whether nu has nonzero paths
        # _path_weights is indexed 0..correlation-1 for nu=1..correlation
        if not self._path_weights[correlation - 1]:
            self.weights_max = mx.zeros_like(self.weights_max)
        for i, nu in enumerate(range(correlation - 1, 0, -1)):
            if not self._path_weights[nu - 1]:
                self.weights[i] = mx.zeros_like(self.weights[i])

    def set_dtype(self, dtype, predicate=None):
        """Override to also convert U matrices (private attrs excluded by default)."""
        super().set_dtype(dtype, predicate)
        # Convert precomputed U matrices to match parameter dtype
        if self._u_main_wf is not None and self._u_main_wf.dtype != dtype:
            self._u_main_wf = self._u_main_wf.astype(dtype)
        self._u_lower_2d = [u.astype(dtype) for u in self._u_lower_2d]
        for nu, u in self._u_matrices.items():
            self._u_matrices[nu] = u.astype(dtype)

    def __call__(self, features: mx.array, element_onehot: mx.array) -> mx.array:
        """Forward pass: recursive contraction using matmul decomposition.

        Args:
            features: (batch, num_features, coupling_dim) node features
            element_onehot: (batch, num_elements) one-hot element encoding

        Returns:
            (batch, num_features * ir_out.dim) contracted features
        """
        b, num_c, coupling_i = features.shape

        # Step 1: Highest correlation order
        # Original: einsum("...ik,ekc,bci,be->bc...", U, W, features, onehot)
        # Decomposed into: features @ U_2d, then contract with W_sel
        out = self._contract_main(features, element_onehot, b, num_c)

        # Steps 2..correlation: lower orders, accumulate
        for i in range(len(self._lower_eqs)):
            # Weight contraction: einsum("...k,ekc,be->bc...", U, W, onehot)
            c_tensor = self._contract_weight(i, element_onehot, b, num_c)
            c_tensor = c_tensor + out

            # Feature contraction: einsum("bc...i,bci->bc...", c_tensor, features)
            out = self._contract_features(c_tensor, features, b, num_c)

        return out.reshape(b, -1)

    def _select_weights(
        self, W: mx.array, element_onehot: mx.array, b: int,
    ) -> mx.array:
        """Select/blend weights by element via matmul.

        Computes W_sel[b, k, c] = sum_e element_onehot[b, e] * W[e, k, c]
        using a single matmul: (b, e) @ (e, k*c) -> (b, k*c) -> (b, k, c).
        """
        e, k, c = W.shape
        W_flat = W.reshape(e, k * c)  # (e, k*c)
        W_sel = (element_onehot @ W_flat).reshape(b, k, c)  # (b, k, c)
        return W_sel

    def _contract_main(
        self, features: mx.array, element_onehot: mx.array,
        b: int, num_c: int,
    ) -> mx.array:
        """Main einsum via weights-first matmul decomposition.

        Original: einsum("...ik,ekc,bci,be->bc...", U, W, features, onehot)
        where U has shape (*prefix, i, k).

        Weights-first decomposition (contracts k before i to reduce
        intermediate tensor size from b*c*prefix*k to b*c*prefix*i):
          1. W_sel = onehot @ W_reshaped        -- (b, c, k) via matmul
          2. WU = W_sel @ U_wf                  -- (b, c, prefix*i) via matmul
          3. result = WU_reshaped @ feat_col    -- (b, c, prefix, 1) via matmul
        """
        prefix_shape = self._u_main_prefix_shape
        prefix_size = self._u_main_prefix_size
        i_dim = self._u_main_i_dim
        k_dim = self._u_main_k_dim

        # 1. Select weights by element: (b, k, c) -> (b, c, k)
        W_sel = self._select_weights(self.weights_max, element_onehot, b)
        W_perm = mx.transpose(W_sel, axes=(0, 2, 1))  # (b, c, k)

        # 2. Contract k with U: (b, c, k) @ (k, prefix*i) -> (b, c, prefix*i)
        #    Then reshape to (b, c, prefix, i) for the next matmul.
        WU = (W_perm @ self._u_main_wf).reshape(
            b, num_c, prefix_size, i_dim
        )

        # 3. Contract i with features:
        #    (b, c, prefix, i) @ (b, c, i, 1) -> (b, c, prefix, 1)
        feat_col = features[:, :, :, None]  # (b, c, i, 1)
        result = (WU @ feat_col).reshape(b, num_c, *prefix_shape)

        return result

    def _contract_weight(
        self, lower_idx: int, element_onehot: mx.array,
        b: int, num_c: int,
    ) -> mx.array:
        """Weight einsum decomposed into matmul.

        Original: einsum("...k,ekc,be->bc...", U, W, onehot)
        where U has shape (*prefix, k).

        Decomposition:
          1. W_sel = onehot @ W_reshaped        -- (b, k, c) via matmul
          2. result = W_sel_perm @ U_2d.T       -- (b, c, prefix) via matmul
        """
        U_2d = self._u_lower_2d[lower_idx]  # (prefix, k)
        prefix_shape = self._u_lower_prefix_shapes[lower_idx]
        W_lower = self.weights[lower_idx]

        # 1. Select weights by element
        W_sel = self._select_weights(W_lower, element_onehot, b)

        # 2. Contract: (b, c, k) @ (k, prefix) = (b, c, prefix)
        W_perm = mx.transpose(W_sel, axes=(0, 2, 1))  # (b, c, k)
        result = W_perm @ U_2d.T  # (b, c, prefix)
        result = result.reshape(b, num_c, *prefix_shape)

        return result

    @staticmethod
    def _contract_features(
        c_tensor: mx.array, features: mx.array,
        b: int, num_c: int,
    ) -> mx.array:
        """Feature einsum decomposed into batched matmul.

        Original: einsum("bc...i,bci->bc...", c_tensor, features)

        Keeps (b, c) separate to avoid creating huge b*c batch dimension,
        letting the GPU handle fewer, larger matmul tiles.
          (b, c, prefix, i) @ (b, c, i, 1) -> (b, c, prefix, 1)
        """
        prefix_shape = c_tensor.shape[2:-1]
        prefix_size = math.prod(prefix_shape)
        i_dim = c_tensor.shape[-1]

        c_4d = c_tensor.reshape(b, num_c, prefix_size, i_dim)  # (b, c, prefix, i)
        feat_4d = features[:, :, :, None]  # (b, c, i, 1)
        result = (c_4d @ feat_4d).reshape(b, num_c, *prefix_shape)

        return result


class SymmetricContraction(nn.Module):
    """MACE symmetric contraction for body-order correlation.

    For each output irrep, creates a Contraction sub-module that
    implements the multi-body correlation using precomputed U matrices.
    """

    def __init__(
        self,
        irreps_in: str | Irreps,
        irreps_out: str | Irreps,
        correlation: int,
        num_elements: int,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.correlation = correlation
        self.num_elements = num_elements

        self.contractions = []
        for mul, ir in self.irreps_out:
            self.contractions.append(
                Contraction(
                    irreps_in=self.irreps_in,
                    ir_out=ir,
                    correlation=correlation,
                    num_elements=num_elements,
                )
            )

        # Precompute metadata for __call__
        self._num_features = self.irreps_in.count(Irrep("0e"))
        coupling_irreps = Irreps([(1, mulir.ir) for mulir in self.irreps_in])
        self._coupling_dim = coupling_irreps.dim
        self._slices = self.irreps_in.slices
        self._ir_dims = [(mul, ir.dim) for mul, ir in self.irreps_in]

    def __call__(
        self, features: mx.array, element_onehot: mx.array
    ) -> mx.array:
        """Forward pass.

        Args:
            features: (num_atoms, irreps_in.dim) node features in irreps-major
                layout (all copies of each irrep block are contiguous).
                Internally reordered to feature-major layout and reshaped
                to (num_atoms, num_features, coupling_dim).
            element_onehot: (num_atoms, num_elements) one-hot element encoding.

        Returns:
            (num_atoms, irreps_out.dim) contracted features.
        """
        batch_size = features.shape[0]

        # Reorder from irreps-major to feature-major layout:
        # irreps-major: [f0_0e, f1_0e, ..., fN_0e, f0_1o_m-1, f0_1o_m0, ...]
        # feature-major: [f0_0e, f0_1o_m-1, f0_1o_m0, ..., f1_0e, f1_1o_m-1, ...]
        blocks = []
        slices = self._slices
        for idx, (mul, ir_dim) in enumerate(self._ir_dims):
            # Extract block: (batch, mul * ir_dim) -> (batch, mul, ir_dim)
            block = features[..., slices[idx]]
            block = block.reshape(batch_size, mul, ir_dim)
            blocks.append(block)

        # Stack along the coupling dimension: each block has (batch, mul, ir.dim)
        # We want (batch, mul, coupling_dim) by concatenating ir.dim parts
        x = mx.concatenate(blocks, axis=-1)  # (batch, num_features, coupling_dim)

        outs = [c(x, element_onehot) for c in self.contractions]
        return mx.concatenate(outs, axis=-1)
