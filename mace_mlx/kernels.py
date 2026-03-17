"""Fused Metal kernels for MACE-MLX hot-path operations.

Provides GPU-fused implementations of gather -> TensorProduct -> scatter
that eliminate intermediate tensor allocations. Two implementations:

1. Metal kernel (forward-only, no autograd): Uses atomic scatter for
   maximum GPU utilization during inference.
2. Pure-MLX fallback (autograd-compatible): Used during training or
   when Metal kernels are unavailable.

The scalar-only fast path (hidden_irreps = Nx0e) is the primary target,
as it's the most common MACE configuration and has the simplest fusion.
"""

from __future__ import annotations

import mlx.core as mx


# ---------------------------------------------------------------------------
# Metal kernel: fused gather-multiply-scatter for scalar TP (inference only)
# ---------------------------------------------------------------------------

_FUSED_SCATTER_SCALAR_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint D = node_feats_shape[1];
    uint edge_id = tid / D;
    uint feat_id = tid % D;

    int src_node = sender[edge_id];
    int dst_node = receiver[edge_id];

    float nf = node_feats[src_node * D + feat_id];
    float w  = weights[edge_id * D + feat_id];
    float ea = edge_attr_0e[edge_id];

    float val = nf * w * ea;

    atomic_fetch_add_explicit(
        &out[dst_node * D + feat_id],
        val,
        memory_order_relaxed
    );
"""

_FUSED_SCATTER_SCALAR_KERNEL: object | None = None


def _get_fused_scatter_scalar_kernel():
    """Lazily compile the fused gather-multiply-scatter Metal kernel."""
    global _FUSED_SCATTER_SCALAR_KERNEL
    if _FUSED_SCATTER_SCALAR_KERNEL is None:
        _FUSED_SCATTER_SCALAR_KERNEL = mx.fast.metal_kernel(
            name="fused_gather_tp_scatter_scalar",
            input_names=["node_feats", "weights", "edge_attr_0e",
                         "sender", "receiver"],
            output_names=["out"],
            source=_FUSED_SCATTER_SCALAR_SOURCE,
            ensure_row_contiguous=True,
            atomic_outputs=True,
        )
    return _FUSED_SCATTER_SCALAR_KERNEL


def fused_gather_tp_scatter_scalar_metal(
    node_feats: mx.array,
    tp_weights: mx.array,
    edge_attr_0e: mx.array,
    sender: mx.array,
    receiver: mx.array,
    num_nodes: int,
) -> mx.array:
    """Fused gather -> scalar-TP -> scatter using a Metal kernel.

    Computes:
        message[receiver[e], u] += node_feats[sender[e], u]
                                   * tp_weights[e, u]
                                   * edge_attr_0e[e]

    This replaces the 3-step sequence:
        1. mji = node_feats[sender] * tp_weights * edge_attr_0e[:, None]
        2. message = scatter_sum(mji, receiver, num_nodes)
    eliminating the (num_edges, D) intermediate tensor.

    Note: No autograd support (Metal custom kernels). Use for inference only.

    Args:
        node_feats: (num_nodes, D) node features (already linear_up'd)
        tp_weights: (num_edges, D) tensor product weights from radial MLP
        edge_attr_0e: (num_edges,) scalar (0e) component of edge attrs
        sender: (num_edges,) sender node indices (int32)
        receiver: (num_edges,) receiver node indices (int32)
        num_nodes: number of nodes in the graph

    Returns:
        (num_nodes, D) aggregated messages
    """
    kernel = _get_fused_scatter_scalar_kernel()
    num_edges = sender.shape[0]
    D = node_feats.shape[1]
    total_threads = num_edges * D
    threadgroup_size = min(256, total_threads)

    outputs = kernel(
        inputs=[node_feats, tp_weights, edge_attr_0e, sender, receiver],
        output_shapes=[(num_nodes, D)],
        output_dtypes=[node_feats.dtype],
        grid=(total_threads, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        init_value=0.0,
    )
    return outputs[0]


# ---------------------------------------------------------------------------
# Metal kernel: fused scatter-add (standalone, for non-scalar TP cases)
# ---------------------------------------------------------------------------

_SCATTER_ADD_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint num_cols = src_shape[1];
    uint edge_id = tid / num_cols;
    uint feat_id = tid % num_cols;
    int node_id = idx[edge_id];
    atomic_fetch_add_explicit(
        &out[node_id * num_cols + feat_id],
        src[edge_id * num_cols + feat_id],
        memory_order_relaxed
    );
"""

_SCATTER_ADD_KERNEL: object | None = None


def _get_scatter_add_kernel():
    """Lazily compile the atomic scatter-add Metal kernel."""
    global _SCATTER_ADD_KERNEL
    if _SCATTER_ADD_KERNEL is None:
        _SCATTER_ADD_KERNEL = mx.fast.metal_kernel(
            name="scatter_add",
            input_names=["src", "idx"],
            output_names=["out"],
            source=_SCATTER_ADD_SOURCE,
            ensure_row_contiguous=True,
            atomic_outputs=True,
        )
    return _SCATTER_ADD_KERNEL


def scatter_sum_metal(
    src: mx.array, index: mx.array, dim_size: int
) -> mx.array:
    """Scatter-add using a Metal kernel with atomics.

    Equivalent to scatter_sum in utils.py but uses GPU atomics.
    No autograd support -- inference only.

    Args:
        src: (N, D) source tensor
        index: (N,) destination indices (int32)
        dim_size: size of output dimension 0

    Returns:
        (dim_size, D) result
    """
    kernel = _get_scatter_add_kernel()
    N, D = src.shape[0], src.shape[1]
    total_threads = N * D
    threadgroup_size = min(256, total_threads)

    outputs = kernel(
        inputs=[src, index],
        output_shapes=[(dim_size, D)],
        output_dtypes=[src.dtype],
        grid=(total_threads, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        init_value=0.0,
    )
    return outputs[0]


# ---------------------------------------------------------------------------
# Pure-MLX fused gather-TP-scatter (autograd-compatible)
# ---------------------------------------------------------------------------

def fused_gather_tp_scatter_scalar(
    node_feats: mx.array,
    tp_weights: mx.array,
    edge_attr_0e: mx.array,
    sender: mx.array,
    receiver: mx.array,
    num_nodes: int,
) -> mx.array:
    """Fused gather -> scalar-TP -> scatter using pure MLX ops.

    Autograd-compatible version of the Metal kernel. Still avoids
    materializing the full intermediate tensor when possible.

    Computes:
        message[receiver[e], u] += node_feats[sender[e], u]
                                   * tp_weights[e, u]
                                   * edge_attr_0e[e]

    Args:
        node_feats: (num_nodes, D) node features
        tp_weights: (num_edges, D) tensor product weights
        edge_attr_0e: (num_edges,) scalar edge attrs
        sender: (num_edges,) sender indices
        receiver: (num_edges,) receiver indices
        num_nodes: number of nodes

    Returns:
        (num_nodes, D) aggregated messages
    """
    # Gather, multiply, scatter -- relies on MLX lazy eval for fusion
    mji = node_feats[sender] * tp_weights * edge_attr_0e[:, None]
    out = mx.zeros((num_nodes, mji.shape[1]), dtype=mji.dtype)
    return out.at[receiver].add(mji)


# ---------------------------------------------------------------------------
# Dispatcher: picks Metal or pure-MLX based on context
# ---------------------------------------------------------------------------

def gather_tp_scatter(
    node_feats: mx.array,
    tp_weights: mx.array,
    edge_attr_0e: mx.array,
    sender: mx.array,
    receiver: mx.array,
    num_nodes: int,
    *,
    use_metal: bool = True,
) -> mx.array:
    """Dispatch to Metal or pure-MLX fused gather-TP-scatter.

    Automatically selects the Metal kernel for inference (better perf
    on large graphs) or the pure-MLX path when autograd is needed.

    Args:
        node_feats: (num_nodes, D) node features
        tp_weights: (num_edges, D) tensor product weights
        edge_attr_0e: (num_edges,) scalar edge attrs
        sender: (num_edges,) sender indices
        receiver: (num_edges,) receiver indices
        num_nodes: number of nodes
        use_metal: if True, use Metal kernel (no autograd); else pure MLX

    Returns:
        (num_nodes, D) aggregated messages
    """
    if use_metal:
        return fused_gather_tp_scatter_scalar_metal(
            node_feats, tp_weights, edge_attr_0e,
            sender, receiver, num_nodes,
        )
    return fused_gather_tp_scatter_scalar(
        node_feats, tp_weights, edge_attr_0e,
        sender, receiver, num_nodes,
    )
