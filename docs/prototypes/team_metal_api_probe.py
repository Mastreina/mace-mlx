"""API probe for mx.fast.metal_kernel (MLX 0.31.2, Apple Silicon).

Small arrays only — NOT a benchmark. Verifies:
  1. basic elementwise kernel, fp32 + fp16 template
  2. dual-gather+multiply kernel (the SC X-construction pattern)
  3. exact thread dispatch (grid not a multiple of threadgroup)
  4. mx.custom_function wrapping + handwritten VJP -> mx.grad works
  5. kernel inside mx.compile and mx.compile(mx.value_and_grad(...))
  6. multiple outputs from one kernel
  7. int template args + int32 index inputs
  8. CPU stream behavior (expected: error)
  9. dead-cotangent pruning with two separate VJP kernels
"""
import mlx.core as mx
import numpy as np

rng = np.random.default_rng(0)
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# ---------------------------------------------------------------- 1. basic
src_exp = """
    uint elem = thread_position_in_grid.x;
    T tmp = inp[elem];
    out[elem] = metal::exp(tmp);
"""
k_exp = mx.fast.metal_kernel(
    name="probe_exp", input_names=["inp"], output_names=["out"], source=src_exp
)
for dt, tol in [(mx.float32, 1e-6), (mx.float16, 1e-3)]:
    a = mx.array(rng.normal(size=(999,)).astype(np.float32)).astype(dt)
    out = k_exp(
        inputs=[a], template=[("T", dt)],
        grid=(a.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[a.shape], output_dtypes=[dt],
    )[0]
    err = float(mx.max(mx.abs(out - mx.exp(a))).item())
    check(f"basic exp {dt}", err < tol, f"err={err:.2e}")

# --------------------------------- 2+3. dual gather multiply, odd grid size
B, I, K, R = 37, 16, 23, 99  # odd B so grid=B*R not a multiple of 256
f = mx.array(rng.normal(size=(B, I)).astype(np.float32))
w = mx.array(rng.normal(size=(B, K)).astype(np.float32))
idx_i = mx.array(rng.integers(0, I, size=R).astype(np.uint32))
idx_k = mx.array(rng.integers(0, K, size=R).astype(np.uint32))
src_dg = """
    uint elem = thread_position_in_grid.x;
    uint r = elem % NROW;
    uint bc = elem / NROW;
    X[elem] = f[bc * IDIM + idx_i[r]] * w[bc * KDIM + idx_k[r]];
"""
k_dg = mx.fast.metal_kernel(
    name="probe_dualgather",
    input_names=["f", "w", "idx_i", "idx_k"],
    output_names=["X"],
    source=src_dg,
)
X = k_dg(
    inputs=[f, w, idx_i, idx_k],
    template=[("T", mx.float32), ("NROW", R), ("IDIM", I), ("KDIM", K)],
    grid=(B * R, 1, 1), threadgroup=(256, 1, 1),
    output_shapes=[(B, R)], output_dtypes=[mx.float32],
)[0]
ref = mx.take(f, idx_i.astype(mx.int32), axis=1) * mx.take(
    w, idx_k.astype(mx.int32), axis=1
)
err = float(mx.max(mx.abs(X - ref)).item())
check("dual-gather kernel + odd grid", err == 0.0, f"err={err:.2e}")

# ------------------------------------------- 4. custom_function + VJP + grad
# VJP kernels: df via CSR over i (no atomics)
i_np, k_np = np.array(idx_i), np.array(idx_k)
order_i = np.argsort(i_np, kind="stable")
rows_by_i = order_i.astype(np.uint32)
starts_i = np.searchsorted(i_np[order_i], np.arange(I + 1)).astype(np.uint32)
rows_by_i_mx = mx.array(rows_by_i)
starts_i_mx = mx.array(starts_i)
src_df = """
    uint elem = thread_position_in_grid.x;   // bc * IDIM + i
    uint i = elem % IDIM;
    uint bc = elem / IDIM;
    float acc = 0.0f;
    for (uint s = starts_i[i]; s < starts_i[i+1]; ++s) {
        uint r = rows_by_i[s];
        acc += (float)dX[bc * NROW + r] * (float)w[bc * KDIM + idx_k[r]];
    }
    df[elem] = (T)acc;
"""
k_df = mx.fast.metal_kernel(
    name="probe_df",
    input_names=["dX", "w", "idx_k", "rows_by_i", "starts_i"],
    output_names=["df"],
    source=src_df,
)

sel_i_np = np.zeros((I, R), dtype=np.float32)
sel_i_np[i_np, np.arange(R)] = 1.0
sel_k_np = np.zeros((K, R), dtype=np.float32)
sel_k_np[k_np, np.arange(R)] = 1.0
SelI = mx.array(sel_i_np)
SelK = mx.array(sel_k_np)
mx.eval(SelI, SelK, rows_by_i_mx, starts_i_mx, idx_i, idx_k)


@mx.custom_function
def fused_x(fa, wa):
    return k_dg(
        inputs=[fa, wa, idx_i, idx_k],
        template=[("T", fa.dtype), ("NROW", R), ("IDIM", I), ("KDIM", K)],
        grid=(fa.shape[0] * R, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(fa.shape[0], R)], output_dtypes=[fa.dtype],
    )[0]


@fused_x.vjp
def fused_x_vjp(primals, cotan, output):
    fa, wa = primals
    df = k_df(
        inputs=[cotan, wa, idx_k, rows_by_i_mx, starts_i_mx],
        template=[("T", fa.dtype), ("NROW", R), ("IDIM", I), ("KDIM", K)],
        grid=(fa.shape[0] * I, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[fa.shape], output_dtypes=[fa.dtype],
    )[0]
    # dW via plain MLX ops (checks mixing kernels + graph ops in one VJP)
    dw = ((cotan * (fa @ SelI)) @ SelK.T).astype(wa.dtype)
    return df, dw


def loss_fused(fa, wa):
    return (fused_x(fa, wa) * ref_r).sum()


def loss_ref(fa, wa):
    return (((fa @ SelI) * (wa @ SelK)) * ref_r).sum()


ref_r = mx.array(rng.normal(size=(B, R)).astype(np.float32))
mx.eval(ref_r)
g_fused = mx.grad(loss_fused, argnums=(0, 1))(f, w)
g_ref = mx.grad(loss_ref, argnums=(0, 1))(f, w)
err_df = float(mx.max(mx.abs(g_fused[0] - g_ref[0])).item())
err_dw = float(mx.max(mx.abs(g_fused[1] - g_ref[1])).item())
check("custom_function VJP df", err_df < 1e-5, f"err={err_df:.2e}")
check("custom_function VJP dw", err_dw < 1e-5, f"err={err_dw:.2e}")

# ------------------------------------------------- 5. inside mx.compile
comp = mx.compile(lambda fa, wa: fused_x(fa, wa) * 2.0)
out_c = comp(f, w)
err = float(mx.max(mx.abs(out_c - 2.0 * ref)).item())
check("kernel inside mx.compile fwd", err < 1e-6, f"err={err:.2e}")

vag = mx.compile(mx.value_and_grad(loss_fused, argnums=(0, 1)))
val_c, g_c = vag(f, w)
val_p = loss_fused(f, w)
err_v = abs(float(val_c.item()) - float(val_p.item()))
err_g = float(mx.max(mx.abs(g_c[0] - g_ref[0])).item())
check("compile(value_and_grad(kernel))", err_v < 1e-3 and err_g < 1e-5,
      f"dval={err_v:.2e} dgrad={err_g:.2e}")

# ------------------------------------------------- 6. multiple outputs
src_two = """
    uint elem = thread_position_in_grid.x;
    T v = inp[elem];
    a[elem] = v * v;
    b[elem] = v + v;
"""
k_two = mx.fast.metal_kernel(
    name="probe_two", input_names=["inp"], output_names=["a", "b"], source=src_two
)
aa, bb = k_two(
    inputs=[f], template=[("T", mx.float32)],
    grid=(f.size, 1, 1), threadgroup=(256, 1, 1),
    output_shapes=[f.shape, f.shape], output_dtypes=[mx.float32, mx.float32],
)
ok = (float(mx.max(mx.abs(aa - f * f)).item()) == 0.0
      and float(mx.max(mx.abs(bb - 2 * f)).item()) == 0.0)
check("multiple outputs", ok)

# ------------------------------------------------- 8. CPU stream
try:
    bad = k_exp(
        inputs=[f], template=[("T", mx.float32)],
        grid=(f.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[f.shape], output_dtypes=[mx.float32],
        stream=mx.cpu,
    )[0]
    mx.eval(bad)
    check("CPU stream raises", False, "no error raised!")
except Exception as e:
    check("CPU stream raises", True, f"{type(e).__name__}: {str(e)[:60]}")

# --------------------------- 9. dead-cotangent pruning (separate kernels)
# grad only w.r.t. f: the dw branch (plain-ops here) should be prunable.
g_only_f = mx.grad(loss_fused, argnums=0)(f, w)
err = float(mx.max(mx.abs(g_only_f - g_ref[0])).item())
check("grad wrt f only", err < 1e-5, f"err={err:.2e}")

# --------------------------- 10. shape flexibility: same kernel, new B
f2 = mx.array(rng.normal(size=(53, I)).astype(np.float32))
w2 = mx.array(rng.normal(size=(53, K)).astype(np.float32))
X2 = fused_x(f2, w2)
ref2 = (f2 @ SelI) * (w2 @ SelK)
err = float(mx.max(mx.abs(X2 - ref2)).item())
check("same kernel new batch size", err < 1e-6, f"err={err:.2e}")

# --------------------------- 11. atomic_outputs smoke (scatter-add pattern)
srcA = """
    uint elem = thread_position_in_grid.x;
    uint dst = idx[elem];
    atomic_fetch_add_explicit(&out[dst], src[elem], memory_order_relaxed);
"""
k_atomic = mx.fast.metal_kernel(
    name="probe_atomic", input_names=["src", "idx"], output_names=["out"],
    source=srcA, atomic_outputs=True,
)
N = 1000
srcv = mx.array(rng.normal(size=(N,)).astype(np.float32))
idxv_np = rng.integers(0, 10, size=N).astype(np.uint32)
idxv = mx.array(idxv_np)
outA = k_atomic(
    inputs=[srcv, idxv], template=[("T", mx.float32)],
    grid=(N, 1, 1), threadgroup=(256, 1, 1),
    output_shapes=[(10,)], output_dtypes=[mx.float32],
    init_value=0.0,
)[0]
refA = np.zeros(10, dtype=np.float64)
np.add.at(refA, idxv_np, np.array(srcv, dtype=np.float64))
errA = float(np.max(np.abs(np.array(outA, dtype=np.float64) - refA)))
check("atomic_outputs scatter-add", errA < 1e-4, f"err={errA:.2e}")

print()
nfail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results) - nfail}/{len(results)} passed")
