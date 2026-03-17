"""Convert PyTorch MACE checkpoints to MLX format.

Handles weight format differences between e3nn/PyTorch MACE and our MLX
implementation, including:
- e3nn Linear flat weight layout -> per-instruction weight list
- e3nn FullyConnectedNet normalization baking -> nn.Linear weights
- FullyConnectedTensorProduct flat weight -> per-instruction weight list
- TensorProduct weight ordering
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from mace_mlx.utils import SILU_NORM_FACTOR as _SILU_NORM_FACTOR


def convert_mace_checkpoint(
    model_path: str,
    output_dir: str,
    dtype: str = "float32",
) -> dict:
    """Convert a PyTorch MACE checkpoint to MLX format.

    Args:
        model_path: Path to PyTorch MACE .pt checkpoint file, or "small"/"medium"/"large"
                     to download MACE-MP models.
        output_dir: Directory to save config.json and weights.npz
        dtype: "float32" or "float16"

    Returns:
        Config dict with model hyperparameters.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    model = _load_torch_model(model_path)
    state_dict = model.state_dict()

    # Extract config
    config = _extract_config(model, state_dict)

    # Determine model type
    model_type = type(model).__name__
    config["model_type"] = model_type

    if model_type == "ScaleShiftMACE":
        scale_t = state_dict["scale_shift.scale"]
        shift_t = state_dict["scale_shift.shift"]
        if scale_t.numel() > 1:
            config["scale"] = scale_t.tolist()
            config["shift"] = shift_t.tolist()
        else:
            config["scale"] = scale_t.item()
            config["shift"] = shift_t.item()

    # Create MLX model to get parameter structure
    from mace_mlx.model import MACE, ScaleShiftMACE as MLXScaleShiftMACE

    mlx_config = {
        k: v
        for k, v in config.items()
        if k not in ("model_type", "scale", "shift", "z_table")
    }
    # Convert atomic_energies to mx.array
    if mlx_config.get("atomic_energies") is not None:
        mlx_config["atomic_energies"] = mx.array(mlx_config["atomic_energies"])

    if model_type == "ScaleShiftMACE":
        mlx_model = MLXScaleShiftMACE(
            scale=config["scale"], shift=config["shift"], **mlx_config
        )
    else:
        mlx_model = MACE(**mlx_config)

    # Map weights
    weight_dict = _map_weights(model, state_dict, mlx_model, config)

    # Save
    np_dtype = np.float32 if dtype == "float32" else np.float16
    save_dict = {}
    for k, v in weight_dict.items():
        if isinstance(v, mx.array):
            arr = np.array(v)
        else:
            arr = np.asarray(v)
        if arr.dtype in (np.float32, np.float64):
            arr = arr.astype(np_dtype)
        save_dict[k] = mx.array(arr)

    # Filter out empty arrays that can't be serialized
    save_dict = {k: v for k, v in save_dict.items() if v.size > 0}
    mx.savez(str(output_dir / "weights.npz"), **save_dict)

    # Save config (convert numpy arrays to lists for JSON)
    config_json = {}
    for k, v in config.items():
        if isinstance(v, np.ndarray):
            config_json[k] = v.tolist()
        elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], np.floating):
            config_json[k] = [float(x) for x in v]
        else:
            config_json[k] = v

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)

    print(f"Saved MLX model to {output_dir}")
    print(f"  Config: {output_dir / 'config.json'}")
    print(f"  Weights: {output_dir / 'weights.npz'}")
    print(f"  Model type: {model_type}")
    print(f"  Parameters: {sum(v.size for v in save_dict.values()):,}")

    return config


def _load_torch_model(model_path: str):
    """Load PyTorch MACE model from path or download MACE-MP/OFF."""
    import torch

    # Check if it's a named MACE-MP model
    try:
        from mace.calculators.foundations_models import mace_mp_names

        is_named_mp = model_path in mace_mp_names
    except ImportError:
        is_named_mp = model_path in ("small", "medium", "large")

    if is_named_mp:
        from mace.calculators import mace_mp

        # Multi-head models need a head parameter to load
        kwargs = {}
        if model_path.startswith("mh-"):
            kwargs["head"] = "matpes_r2scan"  # default head for loading

        calc = mace_mp(
            model=model_path, device="cpu", default_dtype="float32", **kwargs
        )
        return calc.models[0]

    # Load from file
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _extract_config(model, state_dict: dict) -> dict:
    """Extract model configuration from PyTorch model."""
    import torch

    config = {}

    config["r_max"] = float(state_dict["r_max"].item())
    config["num_interactions"] = int(state_dict["num_interactions"].item())

    # Atomic energies: shape is (num_elements,) for single-head,
    # (num_heads, num_elements) for multi-head
    ae_tensor = state_dict["atomic_energies_fn.atomic_energies"]
    if ae_tensor.ndim == 2:
        config["num_elements"] = ae_tensor.shape[1]
    else:
        config["num_elements"] = len(ae_tensor)

    # Bessel basis
    config["num_bessel"] = len(state_dict["radial_embedding.bessel_fn.bessel_weights"])
    config["num_polynomial_cutoff"] = int(
        state_dict.get(
            "radial_embedding.cutoff_fn.p",
            torch.tensor(5),
        ).item()
    )

    # Atomic energies
    config["atomic_energies"] = ae_tensor.numpy().tolist()

    # Multi-head support: extract heads list
    if hasattr(model, "heads"):
        heads = model.heads
        if len(heads) > 1 or (len(heads) == 1 and heads[0] != "Default"):
            config["heads"] = heads

    # Hidden irreps: detect from the first product block's SymmetricContraction
    config["hidden_irreps"] = str(
        model.products[0].symmetric_contractions.irreps_out
    )

    # Detect interaction block class
    first_inter_cls = type(model.interactions[0]).__name__
    config["interaction_cls"] = first_inter_cls

    config["first_interaction_nonresidual"] = first_inter_cls in (
        "RealAgnosticInteractionBlock",
        "RealAgnosticDensityInteractionBlock",
    )

    # Detect Density interaction blocks (0b2/mpa-0 family).
    config["use_density_normalization"] = any(
        "Density" in type(inter).__name__
        and "NonLinear" not in type(inter).__name__
        for inter in model.interactions
    )

    # Edge irreps (NonLinear interaction blocks)
    if hasattr(model, "edge_irreps") and model.edge_irreps is not None:
        config["edge_irreps"] = str(model.edge_irreps)

    # Agnostic product (element-independent symmetric contraction)
    if hasattr(model, "use_agnostic_product") and model.use_agnostic_product:
        config["use_agnostic_product"] = True

    # Max ell from interaction conv_tp
    inter0 = model.interactions[0]
    sh_irreps = inter0.conv_tp.irreps_in2
    max_ell = max(ir.l for _, ir in sh_irreps)
    config["max_ell"] = int(max_ell)

    # Correlation from products
    correlations = []
    for prod in model.products:
        corr = prod.symmetric_contractions.contractions[0].correlation
        correlations.append(int(corr))
    if len(set(correlations)) == 1:
        config["correlation"] = correlations[0]
    else:
        config["correlation"] = correlations

    # avg_num_neighbors
    config["avg_num_neighbors"] = float(model.interactions[0].avg_num_neighbors)

    # z_table: atomic numbers supported by the model (for one-hot encoding)
    if hasattr(model, "atomic_numbers"):
        config["z_table"] = model.atomic_numbers.tolist()
    else:
        config["z_table"] = list(range(1, config["num_elements"] + 1))

    # Radial MLP dimensions — detect from conv_tp_weights structure
    inter0_tp_weights = model.interactions[0].conv_tp_weights
    if hasattr(inter0_tp_weights, "net"):
        # RadialMLP (mh-1 family): nn.Sequential with Linear, LayerNorm, SiLU
        mlp_layers = []
        for layer in inter0_tp_weights.net:
            if hasattr(layer, "out_features"):
                mlp_layers.append(int(layer.out_features))
        # radial_MLP = hidden dims (exclude output=weight_numel)
        config["radial_MLP"] = mlp_layers[:-1]
    else:
        # FullyConnectedNet (original models): _Layer modules
        mlp_layers = []
        for layer in inter0_tp_weights:
            if hasattr(layer, "h_out"):
                mlp_layers.append(int(layer.h_out))
        config["radial_MLP"] = mlp_layers[:-1]

    # Distance transform (Agnesi) — used by 0b/0b2/mpa-0 family
    if hasattr(model.radial_embedding, "distance_transform"):
        dt = model.radial_embedding.distance_transform
        config["distance_transform"] = {
            "q": float(dt.q.item()),
            "p": float(dt.p.item()),
            "a": float(dt.a.item()),
            "covalent_radii": dt.covalent_radii.tolist(),
        }

    # apply_cutoff flag — when False, cutoff is applied in the interaction
    # block to TP weights and density rather than in the radial embedding.
    if hasattr(model.radial_embedding, "apply_cutoff"):
        config["apply_cutoff"] = bool(model.radial_embedding.apply_cutoff)

    return config


def _map_weights(
    torch_model,
    state_dict: dict,
    mlx_model,
    config: dict,
) -> dict:
    """Map PyTorch MACE weights to MLX model parameter names."""
    import torch

    result = {}

    # 1. Atomic energies (frozen, but still needs to be loaded)
    result["atomic_energies_fn.atomic_energies"] = mx.array(
        state_dict["atomic_energies_fn.atomic_energies"].numpy()
    )

    # 2. Node embedding (e3nn Linear -> EquivariantLinear)
    _convert_e3nn_linear(
        state_dict,
        "node_embedding.linear",
        torch_model.node_embedding.linear,
        result,
        "node_embedding.linear",
    )

    # 3. Radial embedding (BesselBasis weights)
    result["radial_embedding.bessel_fn.bessel_weights"] = mx.array(
        state_dict["radial_embedding.bessel_fn.bessel_weights"].numpy()
    )

    # 4. Interaction blocks
    num_interactions = config["num_interactions"]
    for i in range(num_interactions):
        _convert_interaction(
            state_dict,
            torch_model.interactions[i],
            f"interactions.{i}",
            result,
            config,
            mlx_inter=mlx_model.interactions[i],
        )

    # 5. Product blocks
    for i in range(num_interactions):
        _convert_product(
            state_dict,
            torch_model.products[i],
            f"products.{i}",
            result,
            mlx_product=mlx_model.products[i],
        )

    # 6. Readout blocks
    for i in range(num_interactions):
        _convert_readout(
            state_dict,
            torch_model.readouts[i],
            f"readouts.{i}",
            result,
            is_last=(i == num_interactions - 1),
            mlx_readout=mlx_model.readouts[i],
        )

    return result


def _convert_e3nn_linear(
    state_dict: dict,
    torch_prefix: str,
    torch_linear,
    result: dict,
    mlx_prefix: str,
    mlx_linear=None,
):
    """Convert e3nn Linear (flat weight) to EquivariantLinear (per-instruction weights).

    e3nn stores all weights as a single flat tensor. Each instruction has
    path_shape = (mul_in, mul_out), stored contiguously in row-major order.
    Our EquivariantLinear stores weights as a list of (mul_in, mul_out) arrays.

    When the MLX model has unsimplified input irreps (e.g., after a TensorProduct
    whose output is not simplified), a single PyTorch instruction may correspond
    to multiple MLX instructions. In this case, the PyTorch weight is split along
    the mul_in dimension to match the MLX instruction layout.

    Args:
        mlx_linear: Optional MLX EquivariantLinear module. When provided, its
            instruction layout is used to split PyTorch weights correctly.
    """
    flat_weight = state_dict[f"{torch_prefix}.weight"].numpy()

    if mlx_linear is None or len(torch_linear.instructions) == len(mlx_linear.instructions):
        # Simple 1:1 mapping (same number of instructions)
        offset = 0
        for idx, inst in enumerate(torch_linear.instructions):
            numel = 1
            for s in inst.path_shape:
                numel *= s
            w_flat = flat_weight[offset : offset + numel]
            w = w_flat.reshape(inst.path_shape)
            result[f"{mlx_prefix}.weights.{idx}"] = mx.array(w)
            offset += numel
    else:
        # The PyTorch and MLX linears have different instruction counts.
        # This happens when output irreps are unsimplified in MLX (e.g., Gate
        # input has multiple 0e blocks) but simplified in PyTorch.
        #
        # Each torch instruction has one (i_in, i_out) mapping with a weight of
        # shape (mul_in, mul_out). When MLX has multiple output blocks that map
        # to the same torch output block, the torch weight must be split along
        # the column dimension. Input blocks are 1:1 matched (same i_in indexing
        # since both have unsimplified input irreps).

        # Extract all PyTorch weights keyed by (i_in, i_out)
        torch_weights = {}
        offset = 0
        for idx, inst in enumerate(torch_linear.instructions):
            numel = 1
            for s in inst.path_shape:
                numel *= s
            w_flat = flat_weight[offset : offset + numel]
            w = w_flat.reshape(inst.path_shape)  # (mul_in, mul_out)
            torch_weights[(inst.i_in, inst.i_out)] = w
            offset += numel

        torch_out_irreps = torch_linear.irreps_out
        torch_in_irreps = torch_linear.irreps_in

        # Build a mapping from MLX i_in -> torch i_in
        # Input irreps should have the same block count or be 1:1 matchable
        mlx_in_irreps = mlx_linear.irreps_in
        mlx_out_irreps = mlx_linear.irreps_out

        # Map MLX output blocks to torch output blocks by matching irrep type
        # and cumulating multiplicity within each type
        mlx_out_to_torch_out = {}  # mlx_i_out -> (torch_i_out, col_offset)
        for mlx_i_out, (mlx_mul, mlx_ir) in enumerate(mlx_out_irreps):
            # Find the torch output block with matching (l, p) and compute col offset
            col_offset = 0
            found = False
            for t_i_out, (t_mul, t_ir) in enumerate(torch_out_irreps):
                if t_ir.l == mlx_ir.l and t_ir.p == mlx_ir.p:
                    # Count how many MLX blocks before this one map to the same torch block
                    for prev_i_out in range(mlx_i_out):
                        prev_mul, prev_ir = mlx_out_irreps[prev_i_out]
                        if prev_ir.l == mlx_ir.l and prev_ir.p == mlx_ir.p:
                            col_offset += prev_mul
                    mlx_out_to_torch_out[mlx_i_out] = (t_i_out, col_offset)
                    found = True
                    break
            if not found:
                mlx_out_to_torch_out[mlx_i_out] = (None, 0)

        # Map MLX i_in -> torch i_in
        # If both have the same number of input blocks, use direct 1:1 mapping
        # Otherwise, match by type and accumulate offset
        mlx_in_to_torch_in = {}
        if len(mlx_in_irreps) == len(torch_in_irreps):
            # Direct 1:1 mapping (both unsimplified)
            for mlx_i_in in range(len(mlx_in_irreps)):
                mlx_in_to_torch_in[mlx_i_in] = (mlx_i_in, 0)
        else:
            # Map by type with row offset (input is simplified in torch)
            for mlx_i_in, (mlx_mul, mlx_ir) in enumerate(mlx_in_irreps):
                row_offset = 0
                found = False
                for t_i_in, (t_mul, t_ir) in enumerate(torch_in_irreps):
                    if t_ir.l == mlx_ir.l and t_ir.p == mlx_ir.p:
                        for prev_i_in in range(mlx_i_in):
                            prev_mul, prev_ir = mlx_in_irreps[prev_i_in]
                            if prev_ir.l == mlx_ir.l and prev_ir.p == mlx_ir.p:
                                row_offset += prev_mul
                        mlx_in_to_torch_in[mlx_i_in] = (t_i_in, row_offset)
                        found = True
                        break
                if not found:
                    mlx_in_to_torch_in[mlx_i_in] = (None, 0)

        for mlx_idx, mlx_inst in enumerate(mlx_linear.instructions):
            t_in_info = mlx_in_to_torch_in.get(mlx_inst.i_in)
            t_out_info = mlx_out_to_torch_out.get(mlx_inst.i_out)

            if t_in_info is None or t_out_info is None or \
               t_in_info[0] is None or t_out_info[0] is None:
                # No matching torch instruction
                w = np.zeros((mlx_inst.mul_in, mlx_inst.mul_out))
                result[f"{mlx_prefix}.weights.{mlx_idx}"] = mx.array(w)
                continue

            t_i_in, row_off = t_in_info
            t_i_out, col_off = t_out_info
            tkey = (t_i_in, t_i_out)

            if tkey not in torch_weights:
                w = np.zeros((mlx_inst.mul_in, mlx_inst.mul_out))
                result[f"{mlx_prefix}.weights.{mlx_idx}"] = mx.array(w)
                continue

            torch_w = torch_weights[tkey]
            sub_w = torch_w[
                row_off:row_off + mlx_inst.mul_in,
                col_off:col_off + mlx_inst.mul_out,
            ]
            result[f"{mlx_prefix}.weights.{mlx_idx}"] = mx.array(sub_w)


def _convert_interaction(
    state_dict: dict,
    torch_inter,
    prefix: str,
    result: dict,
    config: dict,
    mlx_inter=None,
):
    """Convert one interaction block's weights."""
    inter_cls = type(torch_inter).__name__

    if inter_cls == "RealAgnosticResidualNonLinearInteractionBlock":
        _convert_nonlinear_interaction(
            state_dict, torch_inter, prefix, result, config, mlx_inter
        )
        return

    # linear_up (e3nn Linear)
    _convert_e3nn_linear(
        state_dict,
        f"{prefix}.linear_up",
        torch_inter.linear_up,
        result,
        f"{prefix}.linear_up",
        mlx_linear=mlx_inter.linear_up if mlx_inter else None,
    )

    # conv_tp_weights (e3nn FullyConnectedNet -> nn.Sequential of nn.Linear)
    _convert_radial_mlp(
        state_dict, torch_inter.conv_tp_weights, prefix, result
    )

    # conv_tp has no internal weights (external from radial MLP)

    # linear (e3nn Linear) — may need weight splitting for L>0 models
    _convert_e3nn_linear(
        state_dict,
        f"{prefix}.linear",
        torch_inter.linear,
        result,
        f"{prefix}.linear",
        mlx_linear=mlx_inter.linear if mlx_inter else None,
    )

    # skip_tp (e3nn FullyConnectedTensorProduct or Linear)
    if hasattr(torch_inter.skip_tp, "weight") and not hasattr(torch_inter.skip_tp, "instructions"):
        # Plain e3nn Linear (no TP)
        _convert_e3nn_linear(
            state_dict,
            f"{prefix}.skip_tp",
            torch_inter.skip_tp,
            result,
            f"{prefix}.skip_tp",
            mlx_linear=mlx_inter.skip_tp if mlx_inter else None,
        )
    else:
        _convert_fctp(
            state_dict,
            f"{prefix}.skip_tp",
            torch_inter.skip_tp,
            result,
            f"{prefix}.skip_tp.tp",
        )

    # density_fn (Density interaction blocks only)
    if hasattr(torch_inter, "density_fn"):
        _convert_density_fn(state_dict, torch_inter.density_fn, prefix, result)


def _convert_radial_mlp(
    state_dict: dict,
    torch_mlp,
    inter_prefix: str,
    result: dict,
):
    """Convert e3nn FullyConnectedNet to nn.Sequential of nn.Linear.

    e3nn _Layer stores weight as (h_in, h_out) and computes:
        With activation:  act(x @ (w / sqrt(h_in * var_in))) * sqrt(var_out)
        Without activation: x @ (w / sqrt(h_in * var_in / var_out))
    where act = normalize2mom(silu) = silu(x) * _SILU_NORM_FACTOR

    Our nn.Linear stores weight as (h_out, h_in) and computes: x @ w.T

    We bake all normalization into the weights:
        w_eff_0 = w / sqrt(h_in)                        (first layer)
        w_eff_k = w / (sqrt(h_in) / _SILU_NORM_FACTOR)  (absorb prev normalize2mom)
        w_mlx = w_eff.T                                  (for nn.Linear)
    """
    layers = list(torch_mlp.children())

    for layer_idx, layer in enumerate(layers):
        w = state_dict[
            f"{inter_prefix}.conv_tp_weights.layer{layer_idx}.weight"
        ].numpy()
        h_in = layer.h_in
        var_in = layer.var_in
        var_out = layer.var_out
        has_act = layer.act is not None

        if has_act:
            # w_normalized = w / sqrt(h_in * var_in)
            # output = act(x @ w_normalized) * sqrt(var_out)
            # act = silu * _SILU_NORM_FACTOR
            # Combined: output = silu(x @ (w / sqrt(h_in * var_in))) * _SILU_NORM_FACTOR * sqrt(var_out)
            w_eff = w / math.sqrt(h_in * var_in)
        else:
            # w_normalized = w / sqrt(h_in * var_in / var_out)
            w_eff = w / math.sqrt(h_in * var_in / var_out)

        # If this is not the first layer, absorb previous layer's
        # _SILU_NORM_FACTOR * sqrt(var_out) into this layer's weight
        if layer_idx > 0:
            prev_layer = layers[layer_idx - 1]
            prev_has_act = prev_layer.act is not None
            if prev_has_act:
                scale_from_prev = _SILU_NORM_FACTOR * math.sqrt(prev_layer.var_out)
                w_eff = w_eff * scale_from_prev

        # Transpose for nn.Linear: (h_in, h_out) -> (h_out, h_in)
        w_mlx = w_eff.T

        # Map to our Sequential naming: conv_tp_weights.layers.{idx*2}.weight
        # Our make_radial_mlp creates: Linear, SiLU, Linear, SiLU, ..., Linear
        # nn.Sequential stores as layers.0, layers.1, ..., layers.N
        linear_idx = layer_idx * 2  # Linear layers at even indices
        result[
            f"{inter_prefix}.conv_tp_weights.layers.{linear_idx}.weight"
        ] = mx.array(w_mlx)

    # Handle the last layer's normalization factor if the last hidden layer has act
    # The last Linear layer (no activation) already has the correct factor baked in
    # from the loop above. But we need to also account for the fact that our nn.SiLU
    # is plain silu (not normalize2mom). So after the last silu layer, the output
    # scale differs by _SILU_NORM_FACTOR * sqrt(var_out).
    # Wait - we already absorbed that into the next layer's weight above.
    # So the last layer's weight already includes the scale from the previous silu.
    # Correct!


def _convert_density_fn(
    state_dict: dict,
    torch_density_fn,
    inter_prefix: str,
    result: dict,
):
    """Convert density_fn (e3nn FullyConnectedNet with single layer, no activation).

    The density_fn is FullyConnectedNet([num_radial, 1]) — a single layer with
    no activation (last layer). It computes: x @ (w / sqrt(h_in * var_in / var_out)).
    With var_in=var_out=1 (first and last layer), this simplifies to:
        x @ (w / sqrt(h_in))

    Our nn.Linear stores weight as (h_out, h_in) and computes: x @ w.T
    So: w_mlx = (w_torch / sqrt(h_in)).T  [shape: (1, num_radial)]
    """
    layer = torch_density_fn.layer0
    w = state_dict[f"{inter_prefix}.density_fn.layer0.weight"].numpy()
    h_in = layer.h_in
    var_in = layer.var_in
    var_out = layer.var_out
    # No activation on this layer (last/only layer)
    w_eff = w / math.sqrt(h_in * var_in / var_out)
    # Transpose: (h_in, h_out) -> (h_out, h_in) for nn.Linear
    w_mlx = w_eff.T
    result[f"{inter_prefix}.density_fn.weight"] = mx.array(w_mlx)


def _convert_nonlinear_interaction(
    state_dict: dict,
    torch_inter,
    prefix: str,
    result: dict,
    config: dict,
    mlx_inter=None,
):
    """Convert RealAgnosticResidualNonLinearInteractionBlock weights.

    This block uses RadialMLP (Linear+LayerNorm+SiLU) instead of FullyConnectedNet,
    and has additional components: source/target embeddings, density_fn, alpha/beta,
    linear_res, equivariant_nonlin (Gate), linear_1, linear_2.
    """
    # source_embedding (e3nn Linear -> EquivariantLinear)
    _convert_e3nn_linear(
        state_dict,
        f"{prefix}.source_embedding",
        torch_inter.source_embedding,
        result,
        f"{prefix}.source_embedding",
        mlx_linear=mlx_inter.source_embedding if mlx_inter else None,
    )

    # target_embedding (e3nn Linear -> EquivariantLinear)
    _convert_e3nn_linear(
        state_dict,
        f"{prefix}.target_embedding",
        torch_inter.target_embedding,
        result,
        f"{prefix}.target_embedding",
        mlx_linear=mlx_inter.target_embedding if mlx_inter else None,
    )

    # skip_tp (e3nn Linear -> EquivariantLinear, NOT a FullyConnectedTP)
    _convert_e3nn_linear(
        state_dict,
        f"{prefix}.skip_tp",
        torch_inter.skip_tp,
        result,
        f"{prefix}.skip_tp",
        mlx_linear=mlx_inter.skip_tp if mlx_inter else None,
    )

    # linear_up (e3nn Linear -> EquivariantLinear)
    _convert_e3nn_linear(
        state_dict,
        f"{prefix}.linear_up",
        torch_inter.linear_up,
        result,
        f"{prefix}.linear_up",
        mlx_linear=mlx_inter.linear_up if mlx_inter else None,
    )

    # conv_tp_weights: RadialMLP (Linear+LayerNorm+SiLU stack)
    _convert_torch_radial_mlp(
        state_dict, f"{prefix}.conv_tp_weights", result, f"{prefix}.conv_tp_weights"
    )

    # density_fn: RadialMLP (Linear+LayerNorm+SiLU stack)
    _convert_torch_radial_mlp(
        state_dict, f"{prefix}.density_fn", result, f"{prefix}.density_fn"
    )

    # alpha/beta (scalar parameters)
    result[f"{prefix}.alpha"] = mx.array(
        state_dict[f"{prefix}.alpha"].numpy()
    )
    result[f"{prefix}.beta"] = mx.array(
        state_dict[f"{prefix}.beta"].numpy()
    )

    # linear_res (e3nn Linear -> EquivariantLinear)
    _convert_e3nn_linear(
        state_dict,
        f"{prefix}.linear_res",
        torch_inter.linear_res,
        result,
        f"{prefix}.linear_res",
        mlx_linear=mlx_inter.linear_res if mlx_inter else None,
    )

    # linear_1 (e3nn Linear -> EquivariantLinear)
    _convert_e3nn_linear(
        state_dict,
        f"{prefix}.linear_1",
        torch_inter.linear_1,
        result,
        f"{prefix}.linear_1",
        mlx_linear=mlx_inter.linear_1 if mlx_inter else None,
    )

    # linear_2 (e3nn Linear -> EquivariantLinear)
    _convert_e3nn_linear(
        state_dict,
        f"{prefix}.linear_2",
        torch_inter.linear_2,
        result,
        f"{prefix}.linear_2",
        mlx_linear=mlx_inter.linear_2 if mlx_inter else None,
    )

    # equivariant_nonlin (e3nn Gate) — no trainable weights, just CG coefficients
    # Our Gate is parameter-free (activations applied directly)


def _convert_torch_radial_mlp(
    state_dict: dict,
    torch_prefix: str,
    result: dict,
    mlx_prefix: str,
):
    """Convert PyTorch RadialMLP (Linear+LayerNorm+SiLU) to MLX Sequential.

    PyTorch RadialMLP structure: net.0=Linear, net.1=LayerNorm, net.2=SiLU, ...
    MLX make_radial_mlp_with_layernorm: layers.0=Linear, layers.1=LayerNorm, layers.2=SiLU, ...

    Both have the same structure, just different naming:
    - PyTorch: {prefix}.net.{idx}.weight / .bias
    - MLX: {mlx_prefix}.layers.{idx}.weight / .bias

    SiLU layers have no parameters, so we scan all possible indices.
    """
    # Find the max index by scanning state_dict keys
    max_idx = -1
    prefix_dot = f"{torch_prefix}.net."
    for key in state_dict:
        if key.startswith(prefix_dot):
            parts = key[len(prefix_dot):].split(".")
            if parts[0].isdigit():
                max_idx = max(max_idx, int(parts[0]))

    for idx in range(max_idx + 1):
        w_key = f"{torch_prefix}.net.{idx}.weight"
        b_key = f"{torch_prefix}.net.{idx}.bias"

        if w_key in state_dict:
            w = state_dict[w_key].numpy()
            result[f"{mlx_prefix}.layers.{idx}.weight"] = mx.array(w)

        if b_key in state_dict:
            result[f"{mlx_prefix}.layers.{idx}.bias"] = mx.array(
                state_dict[b_key].numpy()
            )


def _convert_fctp(
    state_dict: dict,
    torch_prefix: str,
    torch_tp,
    result: dict,
    mlx_prefix: str,
):
    """Convert e3nn FullyConnectedTensorProduct flat weight to per-instruction weights.

    e3nn stores all TP weights as a single flat tensor, with each instruction's
    weights stored contiguously. path_shape for uvw mode = (mul1, mul2, mul_out).
    """
    flat_weight = state_dict[f"{torch_prefix}.weight"].numpy()

    offset = 0
    weight_idx = 0
    for inst in torch_tp.instructions:
        if not inst.has_weight:
            continue
        numel = 1
        for s in inst.path_shape:
            numel *= s
        w_flat = flat_weight[offset : offset + numel]
        w = w_flat.reshape(inst.path_shape)
        result[f"{mlx_prefix}.weights.{weight_idx}"] = mx.array(w)
        offset += numel
        weight_idx += 1


def _convert_product(
    state_dict: dict,
    torch_product,
    prefix: str,
    result: dict,
    mlx_product=None,
):
    """Convert product block weights (SymmetricContraction + linear)."""
    # SymmetricContraction weights
    sc = torch_product.symmetric_contractions
    for c_idx, contraction in enumerate(sc.contractions):
        c_prefix = f"{prefix}.symmetric_contractions.contractions.{c_idx}"

        # weights_max
        result[f"{c_prefix}.weights_max"] = mx.array(
            state_dict[f"{c_prefix}.weights_max"].numpy()
        )

        # weights (list, stored in reverse order: weights[0] = nu=correlation-1, etc.)
        for w_idx in range(len(contraction.weights)):
            result[f"{c_prefix}.weights.{w_idx}"] = mx.array(
                state_dict[f"{c_prefix}.weights.{w_idx}"].numpy()
            )

    # linear (e3nn Linear)
    _convert_e3nn_linear(
        state_dict,
        f"{prefix}.linear",
        torch_product.linear,
        result,
        f"{prefix}.linear",
        mlx_linear=mlx_product.linear if mlx_product else None,
    )


def _convert_readout(
    state_dict: dict,
    torch_readout,
    prefix: str,
    result: dict,
    is_last: bool,
    mlx_readout=None,
):
    """Convert readout block weights."""
    if not is_last:
        # LinearReadoutBlock
        _convert_e3nn_linear(
            state_dict,
            f"{prefix}.linear",
            torch_readout.linear,
            result,
            f"{prefix}.linear",
            mlx_linear=mlx_readout.linear if mlx_readout else None,
        )
    else:
        # NonLinearReadoutBlock: linear_1 + normalize2mom(silu) + linear_2
        # Our Gate uses plain silu, so we absorb normalize2mom factor
        # into linear_2's weights: w2_mlx = w2_e3nn * _SILU_NORM_FACTOR
        _convert_e3nn_linear(
            state_dict,
            f"{prefix}.linear_1",
            torch_readout.linear_1,
            result,
            f"{prefix}.linear_1",
            mlx_linear=mlx_readout.linear_1 if mlx_readout else None,
        )
        _convert_e3nn_linear(
            state_dict,
            f"{prefix}.linear_2",
            torch_readout.linear_2,
            result,
            f"{prefix}.linear_2",
            mlx_linear=mlx_readout.linear_2 if mlx_readout else None,
        )
        # Scale linear_2 weights to absorb normalize2mom factor
        for key in list(result.keys()):
            if key.startswith(f"{prefix}.linear_2.weights."):
                result[key] = result[key] * _SILU_NORM_FACTOR
