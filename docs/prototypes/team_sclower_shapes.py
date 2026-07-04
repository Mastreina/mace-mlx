"""Construction-time inspection of medium-mpa-0 SC lower-order structure.

Dumps (numpy only, no big-batch GPU work):
  - per-contraction shapes: prefix sizes, k dims, nrow of sparse main
  - U_lower sparsity: elementwise nnz, nonzero (i,k) pairs when viewed as
    bilinear U'(prefix_outer, i, k), nonzero rows/cols of the 2D (k, prefix)
  - U_rows3 = U_rows.reshape(nrow, prefix_outer, i): per-row i-support
    (feeds the "unroll main into iter0" feasibility check)
"""
from pathlib import Path

import numpy as np

from mace_mlx.model import load_model

CACHE = Path.home() / ".cache" / "mace_mlx"

model = load_model(str(CACHE / "medium-mpa-0" / "v2"))

for pi, prod in enumerate(model.products):
    sc = prod.symmetric_contractions
    print(f"\n=== products[{pi}] irreps_in={sc.irreps_in} irreps_out={sc.irreps_out} "
          f"num_elements={sc.num_elements} coupling_dim={sc._coupling_dim}")
    for ci, contr in enumerate(sc.contractions):
        corr = contr.correlation
        print(f"\n-- contraction[{ci}] ir_out={contr.ir_out} corr={corr} "
              f"c={contr.num_features} sparse_main={contr._use_sparse_main}")
        print(f"   main: prefix_shape={tuple(contr._u_main_prefix_shape)} "
              f"prefix={contr._u_main_prefix_size} i={contr._u_main_i_dim} "
              f"k={contr._u_main_k_dim}")
        if contr._use_sparse_main:
            nrow = contr._sp_u_rows.shape[0]
            print(f"   main sparse: nrow={nrow} "
                  f"({nrow}/{contr._u_main_i_dim * contr._u_main_k_dim} ik pairs)")
            # U_rows3: (nrow, prefix_outer, i_next) view for iter0 coupling term
            u_rows = np.array(contr._sp_u_rows)  # (nrow, prefix)
            i0 = contr._unrolled_lower_i_dims[0] if contr._unrolled_n_lower else None
            if i0:
                po0 = contr._u_main_prefix_size // i0
                ur3 = u_rows.reshape(nrow, po0, i0)
                # nonzero (r, i) pairs (max over prefix_outer)
                ri_nz = (np.abs(ur3).max(axis=1) > 1e-12)
                per_row_i = ri_nz.sum(axis=1)
                print(f"   U_rows3 (nrow,{po0},{i0}): nnz(r,i) pairs="
                      f"{int(ri_nz.sum())}  per-row i-support min/med/max="
                      f"{per_row_i.min()}/{int(np.median(per_row_i))}/{per_row_i.max()}")
                print(f"   elementwise nnz(U_rows)={int((np.abs(u_rows) > 1e-12).sum())} "
                      f"/ {u_rows.size} ({(np.abs(u_rows) > 1e-12).mean():.4%})")
        for idx in range(contr._unrolled_n_lower):
            nu = corr - 1 - idx
            U2t = np.array(contr._u_lower_2d_t[idx])  # (k, prefix)
            k_l, pref_l = U2t.shape
            i_d = contr._unrolled_lower_i_dims[idx]
            po = contr._unrolled_lower_prefix_sizes[idx] // i_d
            w_shape = tuple(contr.weights[idx].shape)
            nz = np.abs(U2t) > 1e-12
            # bilinear view: U'(k, po, i) -> nonzero (i, k) pairs (max over po)
            U3 = U2t.reshape(k_l, po, i_d)
            ik_nz = (np.abs(U3).max(axis=1) > 1e-12)  # (k, i)
            nz_rows_k = int((nz.any(axis=1)).sum())
            nz_cols_p = int((nz.any(axis=0)).sum())
            print(f"   lower[{idx}] nu={nu}: U2t=({k_l},{pref_l}) po={po} i={i_d} "
                  f"W={w_shape}")
            print(f"      elementwise nnz={int(nz.sum())}/{U2t.size} "
                  f"({nz.mean():.4%}); nz k-rows={nz_rows_k}/{k_l}; "
                  f"nz prefix-cols={nz_cols_p}/{pref_l}")
            print(f"      bilinear (i,k) pairs nnz={int(ik_nz.sum())}/{i_d * k_l} "
                  f"({ik_nz.mean():.2%})")

# entry assembly check: how often is the irreps->coupling reorder done
sc0 = model.products[0].symmetric_contractions
print(f"\n=== entry assembly: slices={list(sc0._slices)} ir_dims={sc0._ir_dims} "
      f"n_contractions={len(sc0.contractions)} (x built once per __call__)")
