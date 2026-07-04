"""Probe: extract real dims of medium-mpa-0 for traffic estimation.

Read-only, no benchmark. Prints conv_tp path + dims per layer, SC
contraction dims (i, k, nrow, prefix), radial/SH config.
"""
import sys

sys.path.insert(0, "/Users/mastreina/Desktop/mace-mlx")

from mace_mlx.model import load_model

model = load_model("/Users/mastreina/.cache/mace_mlx/medium-mpa-0/v2")

print("=== interactions ===")
for li, inter in enumerate(model.interactions):
    tp = inter.conv_tp
    path = (
        "batched_uvu" if tp._batched_uvu_scalar
        else "batched_mul21" if tp._batched_mul21
        else "loop"
    )
    print(f"layer {li}: path={path} n_inst={len(tp._instructions)} "
          f"in1={tp.irreps_in1} in2={tp.irreps_in2} out={tp.irreps_out} "
          f"weight_numel={tp.weight_numel}")
    if tp._batched_mul21:
        print(f"  mul21: n={tp._bm21_n} mul={tp._bm21_mul} d1={tp._bm21_d1} "
              f"K0={tp._bm21_K0} K1={tp._bm21_K1} x2_dim={tp.irreps_in2.dim} "
              f"slots={tp._bm21_slots}")
    if tp._batched_uvu_scalar:
        print(f"  uvu: mul1={tp._batched_mul1} total_ir2={tp._batched_total_ir2} "
              f"ir_dims={tp._batched_ir_dims}")

print("=== products (SymmetricContraction) ===")
for pi, prod in enumerate(model.products):
    sc = prod.symmetric_contractions
    for ci, c in enumerate(sc.contractions):
        nrow = c._sp_u_rows.shape[0] if c._use_sparse_main else -1
        prefix = c._sp_u_rows.shape[1] if c._use_sparse_main else c._u_main_prefix_size
        print(f"prod {pi} contraction {ci}: ir_out={c.ir_out} corr={c.correlation} "
              f"i={c._u_main_i_dim} k={c._u_main_k_dim} nrow={nrow} "
              f"prefix={prefix} sparse={c._use_sparse_main} "
              f"nfeat={c.num_features}")

print("=== radial / SH ===")
re = model.radial_embedding
print(f"num_bessel={re.bessel_fn.num_basis} r_max={re.bessel_fn.r_max} "
      f"transform={re._has_transform} apply_cutoff={re._apply_cutoff}")
for li, inter in enumerate(model.interactions):
    mlp = inter.conv_tp_weights
    dims = [layer.weight.shape for layer in mlp.layers if hasattr(layer, "weight")]
    print(f"layer {li} radial MLP shapes: {dims}")
print(f"sh_irreps dim (via model): {getattr(model, '_sh_dim', None)}")
import mlx.core as mx
mx.eval(model.parameters())
print("ok")
