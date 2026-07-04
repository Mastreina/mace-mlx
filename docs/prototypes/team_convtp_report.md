# conv_tp `_batched_mul21` 手写 VJP：设计与原型验证报告

- 目标：teamA 报告 §5 的「进一步空间」——为第二层批量 TP 写 `mx.custom_function`
  手工 VJP，压缩反向的段归约与装配 VJP，整步再提 5-10%。
- 代码：`team_convtp_vjp.py`（常量 + 变体 + 数值验证）、`team_convtp_bench.py`
  （计时，单变体单进程）。本报告所有流量估算口径：medium-mpa-0
  interactions[1].conv_tp，E=46000，mul=128，fp32，字节数按 MB=1e6。
- 数值验证已完成（GPU，非计时）；计时留给主会话串行执行（见 §5）。

## 1. 现状 autograd 反向拆解（`_batched_mul21_forward`，即 B3）

前向图（10 条指令：scalar 组 4 条 K0=16，mul2_1 组 6 条 K1=24）：

```
w_t = w.reshape(E,10,128).T(0,2,1)          (E,128,10)
M1  = (x2 @ G1).reshape(E,3,24)             (E,3,24)
A   = x1m @ M1                              (E,128,24)   [autograd 保存]
W1  = w_t @ T1                              (E,128,24)   [保存]
out1 = A ⊙ W1                               (E,128,24)
xs  = x2 @ S                                (E,16)
W0  = w_t @ T0                              (E,128,16)   [保存]
P0  = x1s ⊙ W0                              (E,128,16)   [保存]
out0 = P0 ⊙ xs[:,None,:]                    (E,128,16)
out = concat(10 × src[:,:,o:o+d].reshape)   (E,5120)
```

autograd 反向逐算子流量（读+写，MB；关键形状 (E,128,24)=565、
(E,128,16)=377、(E,128,10)=235、(E,5120)=942）：

| 反向算子 | 物化形状 | 流量 MB |
|---|---|---:|
| concat-VJP：10 个 slice+reshape 拷贝 | 10×(E,mul,d) | 1 884 |
| **slice-VJP 重建 d_out1：6 个零 pad 到全尺寸** | 6×(E,128,24) | 3 956 |
| **d_out1 累加：5 次二元 add** | (E,128,24) | 8 477 |
| **slice-VJP 重建 d_out0：4 个零 pad** | 4×(E,128,16) | 1 884 |
| **d_out0 累加：3 次 add** | (E,128,16) | 3 391 |
| mul-VJP(out1)：dA=g1'⊙W1、dW1=g1'⊙A | 2×(E,128,24) | 3 390 |
| dx1m = dA @ M1ᵀ；dM1 = x1mᵀ @ dA | bmm×2 | 1 298 |
| dwt1 = dW1 @ T1ᵀ | (E,128,10) | 800 |
| mul-VJP(out0)：dP0、d(xs 广播)+axis1 归约 | 3×(E,128,16) | 2 268 |
| mul-VJP(P0)：dx1s 物化+归约、dW0 | 3×(E,128,16) | 2 310 |
| dwt0 = dW0 @ T0ᵀ | (E,128,10) | 612 |
| dw_t 累加 + transpose/reshape 成 dw | (E,1280) | 1 176 |
| dx1 两个 pad + add；dx2 小 GEMM | (E,512) | ~600 |
| **合计** | | **≈ 32 GB** |

按 M4 Pro 实测有效带宽 ~140-190 GB/s（用 B3 fwd 9.6 GB/69.7 ms 标定），
32 GB ≈ 170-230 ms，与实测反向 ≈ 180 ms（f+b3 249 − fwd 69.7）吻合。

**流量大头确认：装配 slice-VJP 的 pad+add 链 ≈ 19.6 GB，占反向 62%**。
它做的事只是「把 dout 的列重排回 d_out0/d_out1」——一个纯排列，却被
autograd 展开成 10 个全尺寸零 pad + 9 次全尺寸加法。次大头是乘法 VJP 与
广播归约的重复物化（~8 GB）。GEMM 部分只占 ~2.7 GB / 6.4 GFLOP，本就便宜。
这与 SC 稀疏化的教训一致：反向优化目标是消中间物化，不是减 FLOPs。

## 2. 手写 VJP 数学

记号：`wr=(E,n,mul)`、`w_t=wrᵀ`、`w1=w_t@T1`、`w0=w_t@T0`（T0/T1 为 0/1
选择阵，path_weight 已烘进 S/G1）。前向：

```
out1[e,u,κ] = w1[e,u,κ] · Σ_m x1m[e,u,m]·M1[e,m,κ]      M1=(x2@G1).reshape
out0[e,u,k] = w0[e,u,k] · x1s[e,u] · xs[e,k]             xs=x2@S
out = P(out0,out1)        P = 按 i_out 槽序的列排列
```

给定 dout=g，先做 **P⁻¹（regroup）**：g 的第 i 槽块
`g[:, cs_i:cs_i+mul·d_i].reshape(E,mul,d_i)`，按槽序分组 concat 得
g0 (E,mul,16)、g1 (E,mul,24)。构造期已断言两组段偏移在槽序下单调连续，
所以 concat 即对齐（10 slice + 2 concat，≤2 遍数据，替代 19.6 GB 的
pad+add）。之后：

```
mul2_1 组（G1W := g1⊙w1 物化一次，喂两个 bmm）:
  dx1m = G1W @ M1ᵀ                                  bmm (E,mul,3)
  dM1  = x1mᵀ @ G1W                                 bmm (E,3,24)
  dwt1 = (g1⊙A) @ T1ᵀ,  A=x1m@M1 重算               GEMM (E,mul,10)
scalar 组（G0W := g0⊙w0 物化一次，喂两个 bmm）:
  dx1s = G0W @ xs[:,:,None]        （k 归约 → bmm，不再物化+sum）
  dxs  = x1s[:,None,:] @ G0W       （u 归约 → bmm）
  dwt0 = ((g0⊙xs) @ T0ᵀ) ⊙ x1s[:,:,None]  （先 GEMM 缩到 n=10 列再乘）
汇总:
  dx2 = dM1.flat @ G1ᵀ + dxs @ Sᵀ           （两个小 GEMM）
  dw  = (dwt1+dwt0).T(0,2,1).reshape(E,1280)
  dx1 = concat([dx1s, dx1m.flat])           （x1 布局 [0e|1o] 相邻覆盖，免 pad）
```

要点：
- 段归约（dw 的 K→n）与装配逆排列各自压成 1 个 GEMM / 1 遍 concat；
- 中间只物化 4 个大 elementwise（G1W、g1⊙A、G0W、g0⊙xs），autograd 是 8 个
  再加 19.6 GB 装配链；
- 不保存任何前向残差，反向从 primals 重算 M1/A/w1/w0/xs（3 个大 GEMM
  ≈2.4 GB、6.4 GFLOP），换来前向可以用最快的 B2 形式且峰值内存下降；
- 转置常量（T0ᵀ 等）构造期从 numpy 直接建叶子数组并立即 `mx.eval`——
  规避 MLX 0.31.2 的 compile lazy-capture bug（闭包不捕获任何未 eval 派生数组）。

反向流量合计 ≈ **13-15 GB**（其中 regroup 1.9-3.8 GB，取决于 MLX 对
「行跨步 slice 的末轴拆分 reshape」是否免拷贝），为 autograd 的 0.42×。

## 3. 变体设计

| 变体 | 前向 | 反向 | 说明 |
|---|---|---|---|
| base | `_batched_mul21_forward`（B3） | autograd | 生产现状，f+b3 249 ms 基线 |
| b2 | fwd_split（B2 分段权重） | autograd | 对照组：fwd 最快但 autograd 反向差，用于分离「fwd 收益」与「VJP 收益」 |
| **v1** | fwd_split | 手写 split VJP（§2） | 主推。反向 6 bmm/GEMM + 4 大 elementwise |
| **v2** | fwd_unified（策略 C 型单 bmm：x1c (E,mul,4) @ Mf (E,4,40)） | 手写 unified VJP | kernel 更少（~28 vs ~36），字节略多（GW/GB 各 (E,128,40)）；两组合并成单链 |

前向选 B2 形式的原因：手写 VJP 后前向实现不再受 autograd 结构约束，B2 是
块级实测最快前向（45.9 vs B3 69.7 ms）；其「per-slot slice 权重」的 autograd
劣势恰好被手写 VJP 绕开。v2 是对 regroup/elementwise 做大核合并的对照方案，
v1/v2 谁胜由计时决定。

## 4. 数值验证结果（已完成，GPU 非计时）

fwd 与 dx1/dx2/dw 相对误差 = max|Δ|/max|ref|，ref = `_batched_mul21_forward`
的 autograd；损失用固定随机投影 `(out*R).sum()` 保证 cotangent 非平凡；
E=9936 另与 `_loop_forward` autograd 交叉验证。阈值 1e-5。

| 配置 | fwd | dx1 | dx2 | dw |
|---|---:|---:|---:|---:|
| E=9936 seed0 v1 / v1+compile | 5.3e-08 | 1.6e-07 | 1.2e-07 | 1.6e-07 |
| E=9936 seed0 v2 / v2+compile | 5.3e-08 | 1.6e-07 | 3.3e-07 | 1.0e-07 |
| E=46000 seed0 v1 / v1+compile | 1.0e-07 | 1.2e-07 | 1.4e-07 | 1.7e-07 |
| E=46000 seed0 v2 / v2+compile | 1.0e-07 | 1.2e-07 | 3.5e-07 | 1.7e-07 |
| E=9936 seed7（v1、v2 同量级） | ≤6.8e-08 | ≤1.8e-07 | ≤2.7e-07 | ≤1.4e-07 |
| 参考系自检：bm21 vs loop（E=9936 两种子） | ≤2.0e-07 | ≤2.0e-07 | ≤3.5e-07 | ≤1.9e-07 |

全部通过（比 fp32 求和重排的本底 ~2e-7 还小或同量级）。compile 组合下
无 nan/inf（lazy-capture 防御有效；custom_function + compile(value_and_grad)
+ 部分求导均已在冒烟测试确认路由正确）。

## 5. 待计时清单（主会话串行执行，GPU 独占）

```bash
cd /Users/mastreina/Desktop/mace-mlx
S=/private/tmp/claude-501/-Users-mastreina-Desktop-mace-mlx/25918f5b-8f0e-48d3-a2d3-5900e397f165/scratchpad
for E in 46000 9936; do
  for v in base b2 v1 v2; do
    .venv/bin/python $S/team_convtp_bench.py --variant $v --E $E
  done
done
for v in base v1 v2; do   # e2e 代表口径（calculator 整步是 compile 的）
  .venv/bin/python $S/team_convtp_bench.py --variant $v --E 46000 --compile
done
# 可选：--model large --E 46000（base/v1/v2），结构相同 mul=256
```

每次调用约 10-30 s；输出一行 `RESULT ...`（含 maxerr、fwd/f+b3 中位 ms、
峰值 MB）。计时口径与 teamA 一致（warmup 3 + 10 中位，loss=sum()）。

### 预期（E=46000，medium）

| 指标 | base 实测 | 预期 v1/v2 | 依据 |
|---|---:|---:|---|
| fwd | 69.7 ms | ~46 ms | = B2 前向实测值 |
| f+b3 | 249 ms | **140-180 ms（1.4-1.8×）** | 反向 32→13-15 GB，按 140-190 GB/s 折算 95-130 ms，加 fwd 46 ms |
| f+b3 峰值 | 8 972 MB | ~6 000-7 000 MB | 不保存 A/W1/W0/P0 残差（重算） |

不确定项：regroup 是 1 遍还是 2 遍（MLX reshape 对行跨步 slice 的处理）、
bmm 对转置视图是否免拷贝——两者合计影响 ±20-30 ms，计时会直接给出答案。
若 v1≈v2 且都 >200 ms，说明 regroup/小核开销超预期，下一步是把 regroup
的 slice+concat 换成跨步视图 concat 或减少槽级碎片。

### 整步预期

conv_tp 块级 f+b3 省 70-110 ms。当前整步基线（v0.4.0 稀疏 SC 已落地）
medium/Si1000 ≈ 402 ms，其中该块 ≈ 246-249 ms（compile 后同量级，teamA
实测 compile 对该块仅省 3 ms，装配 pad+add 不被融合）。故整步预期
**402 → 292-332 ms，即 1.21-1.38×（提速 17-27%）**；下限情形（只有装配链
收益兑现，省 ~60 ms）也有 ~15%。均超过任务的 5-10% 目标。原因：SC 稀疏化
落地后 conv_tp 反向在整步中的占比升到 ~60%，同样的绝对节省对应更大的相对
收益。Si216（E=9936）预期同比例（块级 53.1 ms 基线 → ~30-38 ms）。

## 6. 落地清单（计时确认后）

1. 在 `_setup_batched_mul21` 里并列构造 v1（或 v2）常量：转置副本一并建叶子
   数组并 `mx.eval`；`@mx.custom_function` 对象每 TP 实例建一次（函数对象
   身份稳定，compile 缓存友好）。
2. guard 增补：x1 的 [scal|m21] 相邻覆盖（medium/large 均满足；不满足退回
   pad 装配或现有 B3 路径）；无 scalar 组时裁掉 scalar 链。
3. dtype：常量按 x1.dtype 转换（fp16 路径与现有约定对齐）；批量维 flatten
   包在 custom_function 外层。
4. 测试：复用现有稀疏/稠密一致性测试模式——fwd、dx1/dx2/dw vs loop autograd，
   加 compile 组合与 fp16 用例。
5. 与 §4.2 SC 稀疏化正交，无接口冲突（只动 tensor_product.py 内部 dispatch）。
