# teamA：第二层 conv_tp 批量化——验证与实测报告

- 测量环境：Apple M4 Pro（14 核 CPU / 48 GB / ~273 GB/s），mlx 0.31.2，fp32。
- 代码版本：全部最终测量固定在 git HEAD `4d07c83` 的干净 checkout
  （`scratchpad/head-checkout`，经 `sys.path` 置顶 + `mace_mlx.__file__` 断言），
  不受主工作树并发重构影响。计时方法：warmup 3 + 10 次取中位，每次 `mx.eval`
  同步；GPU 串行独占。
- 原型脚本：`proto_convtp_batch.py`（块级）、`proto_convtp_e2e2.py`（整步，
  每变体独立进程——注意 `mx.compile` 以函数对象 id 做缓存键，同一 `vag` 对象
  compile 两次会复用第一次的 trace，同进程对照会互相污染）。

## 1. 结构确认（真实 mpa0-medium interactions[1].conv_tp）

`128x0e+128x1o ⊗ (0e+1o+2e+3o) → 10 槽输出`，weight_numel=1280，
`_batched_uvu_scalar=False`（走 `_loop_forward`）。10 条 instruction 全部
uvu、有权重、mul2=1、weight_shape=(128,1)、i_out 与指令序号一一对应（无累加）：

| idx | i_in1 | i_in2 | i_out | (d1,d2,do) | path_weight | 快路径 |
|---|---|---|---|---|---|---|
| 0 | 0(0e) | 0 | 0 | (1,1,1) | 1.0000 | scalar c=1.0000 |
| 1 | 1(1o) | 1 | 1 | (3,3,1) | 1.0000 | mul2_1 (3,3) |
| 2 | 0(0e) | 1 | 2 | (1,3,3) | 1.7321 | scalar c=0.5774 |
| 3 | 1(1o) | 0 | 3 | (3,1,3) | 1.7321 | mul2_1 (3,3) |
| 4 | 1(1o) | 2 | 4 | (3,5,3) | 1.7321 | mul2_1 (3,15) |
| 5 | 0(0e) | 2 | 5 | (1,5,5) | 2.2361 | scalar c=0.4472 |
| 6 | 1(1o) | 1 | 6 | (3,3,5) | 2.2361 | mul2_1 (3,15) |
| 7 | 1(1o) | 3 | 7 | (3,7,5) | 2.2361 | mul2_1 (3,35) |
| 8 | 0(0e) | 3 | 8 | (1,7,7) | 2.6458 | scalar c=0.3780 |
| 9 | 1(1o) | 2 | 9 | (3,5,7) | 2.6458 | mul2_1 (3,35) |

mp0-large 结构完全相同，仅 mul=256。关键量：6 条 mul2_1 的 Σ(d2·do)=106、
Σdo=24；4 条 scalar 的 Σdo=16。scalar 组的 c·path_weight 恰好全为 1
（e3nn 归一化相消），CG 均为标量恒等（无预旋转矩阵）。

## 2. FLOPs 推导：「5×」的修正

记 u=mul（128/256），逐边计数（乘加按 2 FLOPs）。

现状 `_loop_forward`（主导项为 6 条 mul2_1）：

- 6 次 matmul `x1_1o @ cg_i`：2·u·3·Σ(d2·do) = **636u**
- x2 广播收缩（乘 + 归约）：u·106 + u·82 = **188u**
- 权重/路径权重乘 + 输出累加：~150u
- 合计 ≈ **974u** ≈ 125k FLOPs/边（u=128）

**主张的做法（水平拼接 CG 为 (3,106) 单次 matmul，即策略 A）FLOPs 不变**：
`B @ [C1|C2|…]` 的代价恒等于逐个 `B @ Ci` 之和（同为 2·u·3·106），它省的只是
kernel 启动次数（6→1）与 x1 的重复读取（带宽），不是 FLOPs。
**「FLOPs 降约 5 倍」对该机制不成立，证伪**。

真正降 FLOPs 的是**换收缩顺序（x2-first，策略 B/B2）**：先把 x2（每边 16 维，
无 u 因子）与 CG 收缩成小矩阵，再对 x1 做一次 batched matmul——

- `M1 = (x2 @ G1).reshape(E,3,24)`，G1 为 (16, 3·24) 预计算常量
  （CG×path_weight 铺排）：2·16·72 ≈ 2.3k/边（与 u 无关）
- `out1 = x1_1o @ M1`：2·u·3·24 = **144u**（对比 636u+188u）
- scalar 组同理：`out0 = x1_0e ⊙ w ⊙ (x2@S)`，S 为 (16,16) 常量
- 逐段权重乘 + 拼接：~40u

合计 ≈ **187u**，即 **974u/187u ≈ 5.2×**。结论：**5× 这个数字量级正确，但
机制归因错了**——来自收缩顺序重排（利用 mul2=1、把 Σ(d2·do)=106 的中间维压到
Σdo=24），而非 CG 矩阵拼接。另注意该算子在 M4 Pro 上本就是带宽/启动开销瓶颈
而非 FLOPs 瓶颈（现状 fwd 6 GFLOP@46k 边理论 <1ms，实测 154ms），批量化的
实际收益同样主要来自中间张量流量 2.5 GB→0.6 GB 和 kernel 数 ~60→~25。

## 3. 块级实测（固定 HEAD 4d07c83，随机输入，maxerr 相对 `_loop_forward`）

变体说明：A=字面主张（拼 CG）；B=x2-first+take 权重；B2=x2-first+逐段权重
（前向最优）；B3=x2-first+0/1 选择矩阵 GEMM 权重（VJP 自动成为高效
segment-sum，替代 take 的 scatter-add，全梯度最优）；C=0e/1o 合并单次 K=4
bmm。fwd+bwd 对 x1 求导（任务规定）；fwd+bwd3 对 (x1,x2,w) 全求导——更接近
真实模型（力对 SH 与径向权重路径均要求梯度）。

### mpa0-medium，E=46000（Si1000 规模）

| 变体 | maxerr | fwd ms | f+b ms | f+b3 ms | f+b3 峰值MB | fwd / f+b3 加速 |
|---|---:|---:|---:|---:|---:|---|
| loop（现状） | — | 154.3 | 247.5 | 349.4 | 9673 | 1.00 / 1.00 |
| loop+compile | 1.1e-05 | 137.6 | 275.1 | 349.1 | 9673 | 1.12 / 1.00 |
| **A 拼CG（主张）** | 5.7e-06 | 156.3 | 535.8 | 666.2 | 23234 | **0.99 / 0.52** |
| B x2-first | 5.7e-06 | 63.8 | 201.2 | 254.1 | 9302 | 2.42 / 1.38 |
| **B2 逐段权重** | 5.7e-06 | **45.9** | 191.5 | 281.7 | 8996 | **3.36** / 1.24 |
| **B3 选择矩阵** | 5.7e-06 | 69.7 | 208.1 | **249.1** | 8972 | 2.21 / **1.40** |
| C 单bmm | 5.7e-06 | 69.0 | 279.7 | 324.8 | 11152 | 2.24 / 1.08 |
| B3+compile | 5.7e-06 | 62.2 | 206.1 | 246.1 | 8972 | 2.48 / 1.42 |

### mpa0-medium，E=9936（Si216）：loop 33.6/53.0/77.5 ms → B2 fwd 9.6（3.52×），B3 f+b3 53.1（1.46×）
### mp0-large，E=46000：loop 306.5/470.5/680.8 ms → B2 fwd 90.4（3.39×），B3 f+b3 477.2（1.43×）；A 的 f+b3 1867 ms（0.36×）、峰值 45.8 GB（2.6× 恶化）

要点：

- **A（字面主张）证伪**：fwd 0.97-1.01×（无收益），fwd+bwd 慢 ~2×，内存 2-2.6×
  （单块 (E,u,106) 大中间张量被 autograd 完整保留）。
- **B2 前向最优 3.4-3.5×**；**B3 全梯度最优 1.40-1.46×**。差别根源：`mx.take`
  的 VJP 是 scatter-add（慢），换成常量 0/1 选择矩阵 GEMM 后 VJP 自动变为转置
  GEMM（segment-sum），无需手写 custom VJP。
- 反向是收益上限的主因：批量化后反向里 `dW=dout⊙bmm_out` 的段归约、装配
  slice/concat 的 VJP 仍占大头；fwd 3.4× 被摊薄到 f+b3 1.4×。
- `mx.compile` 对该块基本无额外收益（matmul 不参与融合）。

## 4. 整步实测（mpa0-medium，monkeypatch interactions[1].conv_tp，vag=能量+对位置梯度）

每变体独立进程（规避 compile 缓存污染）；dE/dF 为与未 patch 基线的差。

| 体系 | 变体 | 整步 ms | 峰值 MB | 加速 | dE (eV) | dF max (eV/Å) |
|---|---|---:|---:|---:|---:|---:|
| Si216 (E=9936) | base | 137.9 | 4385 | 1.00 | — | — |
| | base+compile | 131.2 | 4755 | 1.05 | — | — |
| | B2 | 119.3 | 4047 | 1.16 | 0 | 2.9e-06 |
| | **B3** | **114.2** | 4474 | **1.21** | 0 | 2.3e-06 |
| | B3+compile | 112.2 | 4353 | 1.23 | — | — |
| Si1000 (E=46000) | base | 645.5 | 18673 | 1.00 | — | — |
| | base+compile | 613.8 | 19293 | 1.05 | — | — |
| | B2 | 568.1 | 16449 | 1.14 | 4.9e-04* | 2.7e-06 |
| | **B3** | **533.9** | 17504 | **1.21** | 0 | 2.9e-06 |
| | B3+compile | 525.6 | 18529 | 1.23 | — | — |

\* 总能 ~-5.4 keV 的 fp32 ±1 ulp（0.0005 meV/atom），求和重排所致，可忽略。

**结论：整步收益 1.21×（约 17%），落在审查报告预期 15-25% 区间内偏下限**；
上限被反向 VJP 结构压住。块级 fwd 154→46 ms 达成了报告「169→60-90 ms」的
前向预期。数值一致性满足 <1e-5 要求（块级 ≤7.6e-6，整步 dF≤2.9e-6）。

## 5. 落地实现要点

1. **采用 B3 作为唯一实现**（MD/弛豫总是要力，f+b3 是代表性路径；若未来有
   纯能量推理场景可再加 B2 开关，不建议现在做两套）。
2. **预计算放 `TensorProduct.__init__`**，与现有 `_batched_uvu_scalar` 检测
   并列（成本是几个 ≤(16,160) 的 numpy 矩阵，微秒级，无需惰性首调）。需要
   预计算并 `mx.stop_gradient` 的常量：
   - 分组：scalar 组（`_cg_scalars[i] is not None`）与 mul2_1 组
     （`_cg_mul2_1[i] is not None`）；各组的源块 slice 与段偏移表 `slot_segs`；
   - `S (x2_dim, K0)`：scalar 组 c·path_weight 铺排；
   - `G1 (x2_dim, d1·K1)`：mul2_1 组 CG×path_weight 铺排（M1=(x2@G1) 重塑）；
   - `T0 (n_inst,K0)`、`T1 (n_inst,K1)` 0/1 选择矩阵——**必须用 GEMM 而不是
     `mx.take` 做权重展开**，否则 take 的 VJP（scatter-add）吃掉反向收益。
3. **启用 guard**（不满足则回落 `_loop_forward`）：外部权重（
   `not internal_weights` 且调用时 `weight is not None`）；全部指令 uvu、
   has_weight、mul2==1、weight_shape==(mul,1) 且 mul 相同；i_out 两两不同且
   覆盖全部输出槽（否则补零槽，同 `_batched_forward` 的处理）；每组源块
   i_in1 唯一；mul2_1 组 ir1_dim 一致。三个已转换模型的第二层均满足；
   小模型/第一层（纯 0e）不满足 mul2_1 分支、继续走现有
   `_batched_uvu_scalar`，两条路径互斥共存（dispatch 顺序：
   `_batched_uvu_scalar` → 新路径 → `_loop_forward`），不必合并（新路径的
   scalar-组退化形式与 `_batched_forward` 等价，合并可留作后续清理）。
4. **权重重排**：`w_t = weight.reshape(E, n_inst, mul).transpose(0,2,1)` 一次
   完成，之后 `w_t @ T0/T1`；path_weight 全部烘进 S/G1，运行时无逐段标量乘。
5. **输出装配**：按 i_out 序对 out0/out1 分段 `reshape(E, mul·do)` 后一次
   `concatenate`，替代现状 zeros 初始化 + 逐条累加。
6. **dtype**：常量按 x1.dtype 转换（fp16 路径），与主工作树正在加的
   dtype 约定对齐。
7. 与并发重构的衔接：本报告基线是 HEAD 4d07c83；重构分支若已改变
   `_loop_forward`/dispatch，落地时以上常量与 guard 逻辑不变，只需接到新的
   dispatch 处。

## 6. 修正后的最终判断

- 批量化**值得落地**，但按修正后的机制（x2-first 收缩重排 + 选择矩阵 GEMM
  权重展开），而不是字面的 CG 水平拼接（后者实测为负优化）。
- 预期收益（M4 Pro，mpa0-medium）：conv_tp 块级 fwd 3.4×/全梯度 1.4×；
  **整步 vag 1.21×（Si216 与 Si1000 一致），峰值内存约 -6%（B3）**；
  mp0-large 块级同幅度，整步未测但结构相同、conv_tp 占比更高，预期 ≥1.2×。
- 审查报告的「整步 15-25%」修正为**实测 ~17-21%（1.21×）**，「FLOPs 5×」修正为
  「B2 形式 FLOPs ~5.2×、B3 形式 FLOPs 持平但流量/启动数大幅下降；瓶颈本质是
  带宽与反向 VJP 结构，非 FLOPs」。
- 进一步空间（未落地）：为整个批量 TP 写 `mx.custom_function` 手工 VJP，把
  反向的段归约与装配 VJP 也压成 2-3 个 GEMM，估计可再拿整步 5-10%；与 §4.2
  对称收缩稀疏化正交。
