"""Utility functions for MACE-MLX.

Provides scatter_sum, edge geometry computation, force computation via
autograd, and tensor product instruction generation.
"""

from __future__ import annotations

import math

import mlx.core as mx

from mace_mlx.irreps import Irrep, Irreps

# normalize2mom constant for SiLU: 1 / sqrt(E[silu(x)^2]) for x ~ N(0,1)
# Computed via numerical integration; matches e3nn exactly.
SILU_NORM_FACTOR = 1.6791768074035645


def scatter_sum(src: mx.array, index: mx.array, dim_size: int) -> mx.array:
    """Scatter-add: out[index[i]] += src[i].

    Args:
        src: (N, ...) source tensor
        index: (N,) destination indices
        dim_size: size of output dimension 0

    Returns:
        (dim_size, ...) result
    """
    out = mx.zeros((dim_size, *src.shape[1:]), dtype=src.dtype)
    return out.at[index].add(src)


def get_edge_vectors_and_lengths(
    positions: mx.array,
    edge_index: mx.array,
    shifts: mx.array,
    cell: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Compute edge displacement vectors and their lengths.

    Args:
        positions: (num_atoms, 3)
        edge_index: (2, num_edges) — [senders, receivers]
        shifts: (num_edges, 3) periodic boundary shift vectors
        cell: (3, 3) unit cell matrix, or None for non-periodic systems

    Returns:
        vectors: (num_edges, 3) displacement vectors (receiver - sender + shift)
        lengths: (num_edges, 1) distances
    """
    sender = edge_index[0]  # (num_edges,)
    receiver = edge_index[1]  # (num_edges,)

    # Displacement = positions[receiver] - positions[sender]
    vectors = positions[receiver] - positions[sender]

    # Apply periodic boundary shifts
    if cell is not None and shifts is not None:
        # shifts: (num_edges, 3) integer lattice shifts
        # cell:   (3, 3) lattice vectors
        vectors = vectors + shifts @ cell
    elif shifts is not None:
        vectors = vectors + shifts

    # Compute lengths
    lengths = mx.sqrt(mx.sum(vectors * vectors, axis=-1, keepdims=True))
    return vectors, lengths


def tp_out_irreps_with_instructions(
    irreps1: Irreps | str,
    irreps2: Irreps | str,
    target_irreps: Irreps | str,
) -> tuple[Irreps, list[tuple]]:
    """Generate tensor product output irreps and instructions.

    Used by InteractionBlock to determine the output irreps and
    instruction list for the convolution tensor product.
    This matches MACE's tp_out_irreps_with_instructions function.

    The connection mode is "uvu": out[u,k] = sum_v W[u,v] * CG_tp(x1[u], x2[v])
    where mul_out == mul_in1 (the multiplicity of irreps1).

    Args:
        irreps1: First input irreps (node features, expanded)
        irreps2: Second input irreps (edge spherical harmonics)
        target_irreps: Desired output irreps to filter against

    Returns:
        irreps_out: Sorted output Irreps
        instructions: List of (i_in1, i_in2, i_out, mode, has_weight, path_weight)
    """
    irreps1 = Irreps(irreps1)
    irreps2 = Irreps(irreps2)
    target_irreps = Irreps(target_irreps)

    trainable = True
    irreps_out_list = []
    instructions = []

    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (mul2, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:
                if ir_out in target_irreps:
                    k = len(irreps_out_list)
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable, mul2))

    # Compute path_weight matching e3nn convention:
    # For uvu mode: alpha = mul2 per path
    # path_weight = sqrt(ir_out.dim / sum_alpha) for all paths to same output
    alpha_per_out: dict[int, float] = {}
    for inst in instructions:
        i_out = inst[2]
        mul2 = inst[5]
        alpha_per_out[i_out] = alpha_per_out.get(i_out, 0.0) + mul2

    normalized_instructions = []
    for inst in instructions:
        i_in1, i_in2, i_out, mode, train, mul2 = inst
        ir_out_dim = irreps_out_list[i_out][1].dim
        total_alpha = alpha_per_out[i_out]
        pw = math.sqrt(ir_out_dim / total_alpha) if total_alpha > 0 else 1.0
        normalized_instructions.append((i_in1, i_in2, i_out, mode, train, pw))

    # Sort the output irreps
    irreps_out = Irreps(irreps_out_list)
    sort_result = irreps_out.sort()
    irreps_out_sorted = sort_result.irreps
    inv = sort_result.inv

    # Permute the output indexes of the instructions to match sorted irreps
    instructions = [
        (i_in1, i_in2, inv[i_out], mode, train, pw)
        for i_in1, i_in2, i_out, mode, train, pw in normalized_instructions
    ]
    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out_sorted, instructions
