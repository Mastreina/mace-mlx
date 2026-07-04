# MACE-MLX 全面优化审查报告

- **日期**：2026-07-04
- **对象**：mace-mlx v0.2.0（commit 4d07c83），MLX 0.31.2，Apple M4 Pro（14 核，48 GB）
- **方法**：全矩阵 profiling（3 模型 × 7 体系，含分阶段耗时与峰值内存）+ 6 组独立原型实测 + MLX 0.22→0.31.2 特性调研（6 个领域，逐条溯源到源码/PR）+ 多智能体四维度静态审查（59 条发现经对抗验证，另有 21 条经主审复核）。基线是当前 MLX 实现自身，不与 PyTorch 对比。
- **标注约定**：收益数字凡标 **[实测]** 为本机原型基准结果；标 **[估计]** 为基于 profiling 数据的推算，未经实现验证。

---

## 1. 执行摘要

当前实现在正确性（fp32 路径）和小体系性能上是扎实的，此前的优化（TP fast path、SC 的 matmul 分解、邻居表 skin 缓存）方向正确。本次审查发现的问题集中在四处：

1. **半精度路径完全损坏**：`default_dtype="float16"/"bfloat16"` 产生不可用的结果（fp16 误差 162 meV/atom、力误差 0.58 eV/Å；bf16 完全是垃圾值），且比 fp32 更慢。根因明确（§3.1），修复方案明确。
2. **内存是大体系的第一瓶颈**：mpa0-medium 在 2000 原子时峰值 36.7 GB，进入换页区间后单步从 2.4 s 恶化到 5.6 s。主要来源是 SymmetricContraction 的稠密 U 收缩（U 矩阵 99.5% 是零）与第二层 conv_tp 的逐边中间张量。
3. **时间热点在等变代数层**：medium 模型的第二层 interaction（10 条 instruction 的循环）占前向 50%，product（对称收缩）占 31%。逐块优化（skip_tp、compile）只能各拿 5-10%，大头需要 §4.1/§4.2 的结构性重构。
4. **drop-in 兼容承诺与现实有硬缺口**：`pip install mace-mlx` 后命名模型根本无法加载（转换运行时依赖 torch）、`mace_off()` 必然失败、默认模型与 mace-torch 不一致、mpa-0 家族的 ZBL 对斥势在转换时被静默丢弃。

**建议的动手顺序**：P0 正确性修复（§3，工作量小、风险为零）→ skip_tp guard 修复 + calculator 缓存 + 常量折叠（§4.3-4.5，一天内完成的确定性收益）→ mx.compile 整步化（§4.6）→ 等变层结构性重构（§4.1/4.2，最大收益、最大工作量）→ 路线图项（§7）。

---

## 1.5 实施结果（2026-07-04 更新）

审查后已实施 P0 全部、P1 的 §4.1（按修正方案）/§4.3/NPT 单遍化、P2 的 §4.4/§4.5/§4.6/pin 提升、P3 的死代码清理。**615 个测试全部通过**，对 torch float64 的力差保持 ~1e-5 eV/Å。整步 A/B（同机同模型同体系，vs 本报告 §2 基线）：

| 模型 | 体系 | 基线 e2e | 现在 e2e | 加速 |
|---|---|---:|---:|---:|
| mpa0-medium | water3 | 3.6 ms | 2.6 ms | **1.40×** |
| mpa0-medium | Si216 | 158.8 ms | 106.3 ms | **1.49×** |
| mpa0-medium | Si1000 | 677.6 ms | 502.8 ms | **1.35×** |
| mpa0-medium | Si2000 | 5598.5 ms | 1240.5 ms | **4.51×**（脱离换页区间） |
| mp0-large | Si512 | 479.1 ms | 388.3 ms | 1.23× |
| mp0-small | Si1000 | 172.6 ms | 165.4 ms | 1.04× |

正确性修复实测：fp16 误差 162 → **0.62 meV/atom**（力 0.58 → 0.013 eV/Å）、bf16 从完全损坏 → 3.6 meV/atom；ZBL 已实现（压缩构型 vs torch 0.026 meV/atom）；`mace_off` 可用；默认模型对齐 medium-mpa-0；持久转换缓存 `~/.cache/mace_mlx/`；零边体系（单原子）在 GPU+compile 下触发 MLX 0.31.2 段错误，已加回退（上游 bug）。

teammate 复核修正两处：§4.1 的「CG 水平拼接」机制**证伪**（FLOPs 不变且为负优化），实际落地的是「x2-first 收缩重排 + 0/1 选择矩阵 GEMM 权重展开」（块级前向 3.4×、全梯度 1.4×、整步 1.21× 单项实测，叠加其他项后达上表数字）；SC 的 one-hot→gather 在权重选择处**无感**（瓶颈在 WU matmul）。剩余最大机会：§4.2 对称收缩稀疏化 + custom VJP（未实施，小模型与内存的主要瓶颈仍在此）。

---

## 2. 性能画像（实测基线）

### 2.1 全矩阵结果（能量+力，每步中位数，NL 缓存命中）

| 模型 | 体系 | 原子 | 边数 | e2e (ms) | vag (ms) | 前向 (ms) | 构图 (ms) | 峰值内存 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mp0-small | water3 | 3 | 6 | 2.7 | 1.7 | 1.1 | 0.42 | 48 MB |
| mp0-small | Si64 | 64 | 2944 | 11.0 | 10.8 | 5.4 | 0.43 | 784 MB |
| mp0-small | Si216 | 216 | 9936 | 37.4 | 36.3 | 16.8 | 0.42 | 2.1 GB |
| mp0-small | Si1000 | 1000 | 46000 | 172.6 | 172.8 | 79.1 | 0.43 | 7.0 GB |
| mp0-small | Si2000 | 2000 | 92000 | 351.0 | 351.3 | 160.2 | 0.44 | 12.4 GB |
| mpa0-medium | water3 | 3 | 6 | 3.6 | 3.7 | 2.2 | 0.96 | 158 MB |
| mpa0-medium | Si216 | 216 | 9936 | 158.8 | 160.7 | 81.4 | 1.02 | 4.5 GB |
| mpa0-medium | Si1000 | 1000 | 46000 | 677.6 | 665.3 | 337.1 | 0.97 | 18.9 GB |
| mpa0-medium | Si2000 | 2000 | 92000 | **5598.5** | 2357.3 | 663.9 | 1.00 | **36.7 GB** |
| mp0-large | Si512 | 512 | 14336 | 479.1 | 473.2 | 249.0 | 0.84 | 16.5 GB |

关键读数：

- **e2e ≈ vag**：邻居表（<9 ms 且有 skin 缓存）、numpy↔MLX 搬运（<0.3 ms）都不是瓶颈。开销全在 MLX 计算图本身。
- **反向 ≈ 前向的 1.1 倍**（vag ≈ 2.1 × fwd），正常。
- **mpa0-medium Si2000 的 e2e(5.6s) >> vag(2.4s)**：36.7 GB 峰值触发内存压力/换页。**内存直接封死了 medium 模型在本机的可用规模（约 2000 原子）**。
- 小体系（<100 原子）单步 2-11 ms 中，Python 构图占 0.4-1.0 ms，加上每 kernel 的调度开销，属 overhead-bound。

### 2.2 前向分阶段耗时（Si1000，ms）

| 阶段 | mp0-small | 占比 | mpa0-medium | 占比 |
|---|---:|---:|---:|---:|
| edge_vec + SH + radial + embed | 1.8 | 2% | 2.2 | 1% |
| interaction 0 | 14.0 | 17% | 36.6 | 11% |
| product 0（对称收缩） | 25.3 | 31% | **104.5** | **31%** |
| interaction 1 | 14.1 | 17% | **169.3** | **50%** |
| product 1 | 25.4 | 31% | 26.1 | 8% |

结论：

- **mp0-small（L=0）**：product（SymmetricContraction）是最大热点，两层合计约 63%。
- **mpa0-medium（L=1）**：interaction 1（第二层 conv_tp，输入 128x0e+128x1o，10 条 instruction 走 `_loop_forward`）单独占前向一半；product 0 次之。
- SH、径向基、embedding 合计仅 1-2%——**不值得单独优化其算子本身**，价值只在减少小体系的 kernel 数（配合 compile）。

### 2.3 fast path 覆盖诊断（真实模型实测）

| 位置 | mp0-small | mpa0-medium | mp0-large |
|---|---|---|---|
| conv_tp 层 0 | batched_uvu ✓ | batched_uvu ✓ | batched_uvu ✓ |
| conv_tp 层 1 | batched_uvu ✓ | **loop_forward（10 inst）** | **loop_forward（10 inst）** |
| skip_tp 层 0 | scalar_fctp ✓ | **通用 einsum（最慢路径）** | scalar_in2 ✓ |
| fused kernel (`_can_fuse`) | **恒 False** | **恒 False** | **恒 False** |

---

## 3. P0：正确性与稳健性（必须修）

### 3.1 fp16/bfloat16 路径完全损坏 ⚠️

**[实测]** mpa0-medium，Si216：

| dtype | 能量误差 | 力误差上限 | 速度 |
|---|---:|---:|---:|
| float16 | 162 meV/atom | 0.58 eV/Å | 比 fp32 慢 6% |
| bfloat16 | ~4×10⁸ meV/atom | ~2×10⁷ eV/Å | 比 fp32 慢 4% |

**根因**（静态审查定位，与实测定量吻合）：

1. **位置先 astype 再求差**（[calculators.py:352](../mace_mlx/calculators.py)）：`pos.astype(compute_dtype)` 把绝对坐标（几十 Å）转到半精度后才计算 `positions[receiver] - positions[sender]`。fp16 在 32 Å 处分辨率约 0.03 Å，bf16 约 0.25 Å——边向量发生灾难性相消。这是主导误差源。**修法**：始终在 fp32 下计算边向量与长度（差值是小量），再把 vectors/lengths cast 到 compute dtype。mace-torch 的混合精度同样只对特征降精度。
2. **atomic_energies (E0) 被量化**：`set_dtype` 把逐元素基线能量（幅值几十 eV）转到半精度，产生随原子数线性放大的系统偏差。**修法**：E0、能量累加、scale/shift 全程锁 fp32。
3. **cell 体积在 fp16 计算**（stress 路径）：胞边长 ≥41 Å 时 det 溢出为 inf，stress 静默变 0。
4. **三套 dtype 转换机制并存**（`set_dtype` 重载、`_convert_private_arrays`、converter 的 npz dtype）互相不知情，直接调 `model.set_dtype` 或以 fp16 存 npz 再按 fp32 加载都会得到静默错误状态。

**建议**：短期在 `MACEMLXCalculator.__init__` 对 fp16/bf16 发 warning 或直接 raise；修复按上述 1+2 做完后，用「fp32 幅值敏感段 + 半精度特征段」的混合精度设计重新放开，并补测试（现状：**fp16/bf16 零测试覆盖**）。注意实测表明半精度在本负载没有速度收益（M 系列 GPU fp32 吞吐足够、瓶颈在带宽和调度），所以此项优先级是「防止用户拿到错误结果」而非提速。

### 3.2 ZBL 对斥势在转换时被静默丢弃

mpa-0/0b/0b2/0b3 家族的 PyTorch 检查点含 `pair_repulsion`（ZBL 短程对斥），converter 未转换该模块，也没有警告。平衡构型附近 ZBL 贡献≈0，所以现有「能量差 0」的测试全部通过；但高压、近距离碰撞、AIMD 淬火等场景会得到系统性错误的势能面（短距离处灾难性偏软）。**修法**：converter 检测到 `pair_repulsion` 时要么实现 ZBL（纯解析函数，工作量小），要么 raise/warn，绝不能静默。

### 3.3 drop-in 兼容的硬缺口

| 问题 | 现状 | 修法 |
|---|---|---|
| `pip install mace-mlx` 后 `mace_mp("small")` 直接失败 | 转换路径运行时 `import torch` + `from mace.calculators import ...`，两者都不在主依赖（[converter.py:39](../mace_mlx/converter.py)） | 见 §6.2 持久缓存 + 分发预转换权重；短期把报错改成给出明确指引的 ImportError |
| `mace_off()` 必然 FileNotFoundError | `"off-small"` 不在 `mace_mp_names`，`_resolve_model` 把它当目录路径 | 补 off 系列的名称路由与转换 |
| 默认模型不一致 | mace-torch 的 `mace_mp()` 默认 medium-mpa-0，本项目默认 small——drop-in 用户静默拿到不同势能面 | 对齐默认值为 `medium-mpa-0` |
| `default_dtype="float64"` 静默降级为 fp32 | mace-torch 默认 float64；MLX GPU 无 float64 | 至少发 warning，README 明示 |
| committee 模型列表被静默取第一个 | `model_paths=[...]` 只用 `[0]`（[calculators.py:82](../mace_mlx/calculators.py)） | raise NotImplementedError 或实现 ensemble 均值/方差 |
| 每进程重新转换权重到 tempdir | `tempfile.mkdtemp` + 类级缓存，进程退出即失效，且从不清理 | §6.2 |

### 3.4 其他稳健性项

- **重叠原子（零距离边）反向 NaN**：`sqrt(0)` 的梯度与 Agnesi 的负幂共三处；1e-20 floor 只保护前向。对 MD 是「坏构型本来就该炸」，但对结构优化的初猜值得防御（clamp lengths）。（主审复核：成立，低优先级）
- **能量求和顺序**：scatter(at[].add) 在大体系的累加顺序非确定，运行间能量在 1e-4 eV 量级波动（实测 compile 前后 dE 达 1.2e-4）。如需严格可复现，对 `num_graphs==1` 特判为 `mx.sum`（顺带也是性能微赢，见 §4.5）。
- **NPT/变胞 MD 每步跑两遍完整 forward+backward**（已对抗验证确认）：ASE 的 `get_property` 对缺失属性单独触发 `calculate`；`get_forces()` 后再 `get_stress()` 会完整重算一遍，而 stress 路径本来就能在同一次 vag 里同时给出 forces。**修法**：`atoms.cell.rank == 3` 且收到过 stress 请求后，统一走 `_compute_energy_forces_stress` 一次性填充所有结果。**NPT 每步耗时约减半 [估计，机制已验证]**。

---

## 4. 性能优化（按整步预期收益排序）

### 4.1 第二层 conv_tp 批量化（medium/large 的最大时间热点）

**现状**：L>0 模型第二层的 conv_tp（输入 `128x0e+128x1o`）有 10 条 instruction，走 `_loop_forward` 逐条执行：每条做切片、reshape、`cg_mul2_1` matmul 分解或 scalar 乘、权重切片、累加——Si1000 上占 mpa0-medium 前向的 **50%（169 ms）**。

**方案**（审查智能体提出，主审复核方向成立，未原型化）：

- 10 条 instruction 里 6 条 `mul2_1`（l₁⊗l₂→l₃，CG 分解为 matmul）按 `i_in1` 分组只有两个源块（0e 块与 1o 块），可将同源的 CG 矩阵水平拼接：`t = x1_block @ [cg_a | cg_b | cg_c]` 一次算完，再按段切分做 x2 收缩。matmul 数从 6 降到 2，FLOPs 降约 5 倍（消除重复的 x1 读取）。
- 4 条 scalar CG 指令已可用现有 `_batched_uvu_scalar` 的思路合并。
- `_loop_forward` 的输出槽 `mx.zeros` 初始化 + 逐条 add 在 i_out 全部唯一时纯属多余（两遍大张量内存流量），直接按槽拼接。

**预期**：inter1 前向从 169 ms 降到 60-90 ms 量级，整步（vag）收益 **15-25% [估计]**。这是所有未实施项中「工作量/收益比」最好的结构性优化。

### 4.2 SymmetricContraction 稀疏化 + custom VJP（时间第二热点、内存第一来源）

**事实**：U 矩阵极度稀疏——lout=0 nu=3 非零率 0.4%（353/94208），lout=1 nu=3 为 0.5%（2838/626688）。当前 weights-first 分解物化 `(b, 128, prefix·i)` 稠密中间张量（lout=1 时 6.3 MB/原子），是 18.9-36.7 GB 峰值内存的第一来源，也是 prod 阶段带宽瓶颈。

**[实测] 原型结果**（b=1000）：

| 变体 | lout=0 fwd | lout=0 fwd+bwd | lout=1 fwd | lout=1 fwd+bwd | 内存(lout=1 fwd) |
|---|---:|---:|---:|---:|---:|
| 现状（稠密） | 24.7 ms | 38.1 ms | 78.0 ms | 112.0 ms | 7.2 GB |
| 稀疏 S-matmul | **15.4 ms** | **30.7 ms** | 156.6 ms | 311.6 ms | 5.3 GB |
| 列压缩 + compile | 13.8 ms | 49.4 ms | **49.2 ms** | 164.6 ms | **3.1 GB** |
| mx.checkpoint | 无变化 | 更慢+更耗内存 | — | — | — |

结论与路线：

- **前向收益已证实**（lout=1: 1.6×，内存 2.3×），但 take/gather 链的 autograd VJP（逐个变成 scatter-add）让反向变慢，**净收益必须配 `mx.custom_function` 手写 VJP**。稀疏收缩的反向在数学上仍是同一稀疏结构的转置收缩（梯度对 features 的散射只落在 16 个 i 槽、对 W 的散射落在 k 槽），手写后反向预计与前向同量级。
- 完整实现后 prod 阶段时间预计 2-3×、**峰值内存降 3-5×**（medium Si2000 从 36.7 GB 回到 10-15 GB 区间，解除换页并放开可用体系规模）**[估计，前向部分已实测]**。
- **mx.checkpoint 无效**（实测反而更慢更耗内存），不要走重计算路线。
- 参考实现语义上就是 cuEquivariance/OpenEquivariance 的 segmented contraction，属于业界已验证的方向。

### 4.3 skip_tp 慢路径修复（默认模型，确定性收益）

**已对抗验证 + 独立 numpy 验证**：`wigner_3j(l,0,l)[:,0,:] = I/√(2l+1)` 对 l=0..3 精确成立，`_scalar_in2_fast` 的 guard（[tensor_product.py:699](../mace_mlx/tensor_product.py)）要求 `ir1_dim==1` 是过窄的，导致默认模型（mpa0-medium）第一层 skip_tp 落入通用 einsum，物化 `(N,128,89,2l+1)` 中间张量（0.68 MB/原子 × 反向）。

**[实测] 块级对比**（b=1000，四种实现）：

| 实现 | fwd | fwd+bwd | 峰值内存(fwd+bwd) |
|---|---:|---:|---:|
| 现状（通用 einsum） | 20.8 ms | 32.5 ms | 2.3 GB |
| guard 放宽（matmul 路径） | 4.0 ms | 6.0 ms | 415 MB |
| take + batched matmul | 3.7 ms | 5.6 ms | 391 MB |
| **mx.gather_mm** | **1.4 ms** | **3.3 ms** | **138 MB** |

数值一致性 3.6e-7。**整步收益 5-9% [实测：Si216 1.07×]**——块级 10× 但该块只占整步 5%。修法两档：

1. 保守：放宽 guard 为逐 instruction 检测 `CG[:,0,:]` 是否对角（一处小改动，走现成的 `_scalar_in2_forward`，它已正确处理 ir_dim>1）。
2. 激进：one-hot 语义改为 `mx.gather_mm(x1ᵀ, W_e, rhs_indices=z_idx)`（需 `mlx>=0.27.1`，其 VJP 该版本才修复）。

### 4.4 calculator 层的每步浪费（小而确定）

均经审查确认或主审复核：

- **one-hot 每步重建**（[calculators.py:262-273](../mace_mlx/calculators.py)）：两遍 Python 逐原子循环 + zeros/散布 + 上传。MD 中原子种类不变——把 `indices`、`node_attrs_mx` 挂到 NL 缓存同级的缓存上（`system_changes` 参数目前被完全忽略，正确的失效信号就在手边）。小体系每步省 0.1-0.5 ms。
- **edge_index/shifts 在 NL 缓存命中时仍每步重新 `mx.array` 上传**：同样挂缓存。
- **`node_energy`/`energies` 每步无条件转 numpy**：多数 MD 循环只读 energy+forces，改为惰性转换。
- **stress 的 Voigt 装配与 3×3 det 在 GPU 图内做 20+ 个标量算子**：det/装配移到 numpy 侧或用花式索引一次完成（已验证等价）。
- 位置无关量（node_e0、node_embedding、mh-1 的 source/target embedding）在 vag 的 trace 里每步重算——挂 NL/species 缓存。

合计对小体系（overhead-bound 区间）**约 10-20% [估计]**，大体系几乎无感。

### 4.5 常量折叠与微冗余（一批一次性小改动）

- `1/avg_num_neighbors` 与 ScaleShift 的 `scale` 可在加载时折叠进相邻 EquivariantLinear 权重（已验证数学等价、适用面为非 density 家族）。
- SH 的 e3nn 基底旋转 `edge_attrs @ self._sh_rotation`（每步一次 (E,16)×(16,16) matmul）**可完全折叠进 SH 递推的 l=1 种子系数**（审查智能体已数值验证到 1e-7；blocks.py 里「CG 预旋转」的注释描述的正是这个未做的事）。
- FCTP 每次前向重复 `transpose+reshape` 常量权重（`W_vuw`，5.8 MB/条/次）→ 预计算缓存。
- `_loop_forward`/`EquivariantLinear._accumulate_groups` 的 zeros 初始化 + add 在无累加冲突时移除。
- `num_graphs==1` 时能量聚合的 scatter_sum 特判为 `mx.sum`（顺带解决 §3.4 的非确定性）。
- Gate 单分支时跳过 concatenate；激活全同检查移到 `__init__`。
- `Contraction._ensure_weight_caches` 的 id() 检查移到加载后一次性完成。

单项都小，合计对整步 **3-8% [估计]**，且全部是零风险重构。

### 4.6 mx.compile 整步化

**[实测]**（`mx.compile(mx.value_and_grad(energy_fn))`，输入全部显式传参）：

| 体系 | mp0-small | mpa0-medium |
|---|---:|---:|
| water3 | **1.45×** | **1.24×** |
| Si64 | 1.04× | 1.09× |
| Si216 | 1.04× | 1.08× |
| Si1000 | 1.03× | 1.14× |

数值一致（dE ≤ 1.2e-4 eV 为求和重排所致，dF ≤ 6e-6 eV/Å）。内存不变——**compile 融合只覆盖 elementwise 链（fuse depth 11），matmul/scatter/gather 不融合**，与 MLX 源码级调研结论一致。工程要点（全部来自调研，有源码出处）：

- 必须 `compile(value_and_grad(f))` 这个顺序；反过来会**静默**失去全部收益。
- `captured` 字典副作用要改为 `(energy, node_energy)` aux 元组返回（value_and_grad 原生支持，已在 CPU 验证）。
- **shapeless=True 不可用**：Scatter/Slice 原语无 `output_shapes`，且反向图里 Gather 的 VJP 必然生成 Scatter。替代：num_edges 向上取整到桶（如 2 的幂或 4096 步长）+ mask padding，把形状集合收敛到少数几个，靠普通 compile 缓存吃满命中。
- 缓存按形状无淘汰、thread_local；重编译成本≈一步耗时（实测 12-170 ms），skin 缓存下每 10-50 步才变一次形状，可接受。
- Python 标量（`_head_idx`、`num_graphs`）经闭包读取会被烘焙，换 head 需失效缓存。

**定位**：确定性收益但非决定性；小体系和「其他优化做完后」价值上升（消掉的是固定开销）。建议在 4.1-4.5 之后统一做。

### 4.7 明确不建议做的（负结果，避免浪费时间）

- **mx.checkpoint 换内存**：实测更慢且峰值更高（§4.2）。
- **权重量化**：模型仅 4-16 M 参数、瓶颈在逐边算子与带宽，`nn.quantize` 覆盖面几乎为零，且量化误差直接进力的数值（已确认为 confirmed 结论：建议明确不做）。
- **SH 硬编码多项式**：SH 全链路只占前向 ~1%，CG 递推在配合 compile 后是正确取舍；只做 §4.5 的旋转折叠即可。
- **半精度提速**：实测无速度收益（§3.1），只作为正确性项处理。
- **scatter_sum 换排序 segment-reduce**：调研确认 MLX 0.22→0.31 对 Metal scatter 无任何官方优化，但 profiling 显示 scatter 不是当前主瓶颈；保留 kernels.py 的 atomic 思路，等 §4.1/4.2 做完后按新 profile 再决定。届时复活路径是 `mx.custom_function` 补 VJP（scatter-add 的 VJP 恰是 gather，0.29.0 起有快速路径）。

---

## 5. MLX 版本策略

**建议把 `mlx>=0.22.0` 提到 `mlx>=0.31.2`**。理由（全部溯源到 release notes/PR）：

| 需求 | 最低版本 | 说明 |
|---|---|---|
| gather_mm 的 VJP 正确 | 0.27.1 | §4.3 激进方案的前提（PR #2335） |
| module.update 默认 strict | 0.27.1 | 行为变化，权重加载需适配 |
| compile 融合正确性（broadcast 别名 bug） | 0.31.0 | PR #3166，§4.6 的前提 |
| value_and_grad 反复调用的内存泄漏修复 | 0.31.2 | PR #3290，**长 MD 稳定性** |
| Compiled kernel 缓存冲突修复 | 0.31.2 | PR #3427 |
| 多线程支持（NL 重建与 GPU 重叠） | 0.31.2 | PR #3281/#3423 |
| 顺带收益：contiguous gather 加速(0.29)、小 K GEMM 调优(0.30.x)、CPU-GPU 同步(0.24)、无拷贝上传(0.30.1) | — | 免费 |

升级需复核的行为变化：sort 对齐 NumPy（0.30.0）、sigmoid 精度变化影响 SiLU 数值（0.30.0，测试容差需按 fp32 设置）、仅支持 macOS ≥ 14（0.30.0）、OOM 从 abort 变为可捕获异常（0.30.1）。

**CUDA 后端**：0.31.2 起本项目所需算子集（含 gather_mm）在 CUDA 侧已齐，原则上 `pip install mlx[cuda]` 可跑 NVIDIA，但 scatter/小算子密集图无官方调优记录，两个 Metal kernel 需按 0.29.0 的 custom CUDA kernel 机制重写。定位为实验方向，不建议承诺支持。

---

## 6. 代码质量与 API

### 6.1 死代码与文档矛盾（清理）

- **fused kernel 全链路是生产死代码**（已实证）：`_can_fuse_scalar_tp` 要求 conv_tp 恰好 1 条 instruction，真实模型至少 4 条 → 恒 False；`set_fused_kernel(True)` 是空操作；四个交互块的 `if self._use_fused_kernel` 分支永不执行；kernels.py 两个 Metal kernel 未被生产使用；[test_kernels.py](../tests/test_kernels.py) 用 387 行测试死代码制造虚假安全感。文档还三处自相矛盾（「无 autograd 仅推理」vs「autograd 安全」）。**建议**：整体删除或明确降级为实验代码，Metal atomic kernel 的思路保留到 §4.7 的复活路径里。
- **model.py 有三到四份近似重复的 forward**：`__call__`、`_forward_from_vectors`、`_forward_from_vectors_with_node_energy`（MACE 与 ScaleShiftMACE 各一套），其中 `MACE._forward_from_vectors` 与 `energy_fn` 是无调用方的死方法。合并为单一核心 + 薄包装，消除漂移风险（docstring 已经漂移了）。
- **blocks.py 的「CG 预旋转」注释为假**：CG 未预旋转，基变换实际在 model.py 运行时执行——按 §4.5 真正把旋转折叠掉，注释就变真了。

### 6.2 命名模型的持久缓存（修 §3.3 的核心）

设计要点（审查确认）：转换产物写 `~/.cache/mace_mlx/<model_name>/<converter_version>/`，含 config.json + weights.npz + 完成标记文件；写入用「临时目录 + 原子 rename」保证并发安全；converter 版本号进路径实现失效。配合官方分发预转换权重（GitHub Release 或 HF），`pip install mace-mlx` 即可在无 torch 环境用命名模型——这是把 README 的「drop-in」承诺变真的唯一路径。

### 6.3 其他

- 构造 Calculator 时 `mx.set_default_device` 全局副作用——多实例/与其他 MLX 代码混用的进程互相干扰；改为在计算入口用 `mx.stream` 上下文。
- pyproject 用 PEP 639 字符串 license 但允许 `setuptools>=68`——旧 setuptools 源码构建直接报错，把下限提到 77。
- 报错质量：模型名 typo 会得到误导性的 FileNotFoundError（应列出可用名称）；`z_table` 类型注解与 None 矛盾。
- README 需要一节「与 mace-torch 的差异」：不支持项（committee、dispersion 需自配、float64、return_raw_model）、默认模型、torch 依赖现状。

---

## 7. 功能路线图（按价值/成本比排序）

1. **修复 §3.3 的兼容缺口 + 持久缓存**（小成本，直接影响所有新用户的第一次体验）。
2. **批量多构型推理**：模型层已支持 `num_graphs>1`，只差 calculator/API 层拼 batch——NEB、表面吸附扫描、主动学习采样都直接受益（中成本）。
3. **D3 色散**：不必移植到 MLX，用 ASE `SumCalculator` + CPU 端 torch-dftd/simple-dftd3 即可对齐 mace-torch 的 `dispersion=True`（小成本）。
4. **ZBL 对斥**（§3.2，正确性驱动，小成本）。
5. **committee/ensemble 推理**：多模型均值 + 不确定度输出，主动学习刚需（中成本）。
6. **LAMMPS/OpenMM 接入**：torch 侧靠 TorchScript 导出，MLX 无对应物；近期可行路径是 i-PI/LAMMPS `fix ipi` 的 socket 驱动（中成本）。远期可评估 `mx.export_function` 到 C++（0.30.1 起支持导出含 scatter 的图与自定义 kernel）。
7. **训练/微调**：MLX 优化器生态已齐，但当前前向硬编码 `stop_gradient`、fused 推理假设等三类阻断；全量训练不建议，可评估「冻结主干 + readout 微调」的小范围支持（大成本）。
8. **多线程流水线**：MLX 0.31.2 起线程安全，可把邻居表重建（CPU）与上一步 GPU 计算重叠；配合 `mx.async_eval` 隐藏取回延迟（中成本，收益限于 NL 重建步）。

---

## 8. 测试与基准的改进

- **纯 MLX 金标准回归**：当前所有数值正确性测试运行时依赖 torch/mace-torch + 模型下载。把小体系的能量/力金标准值固化为仓库内 npz，CI 无 torch 也能跑数值回归（顺带解决 3 个测试文件模块级 `import torch` 导致的收集错误——应为 skip 而非 error）。
- **性能回归防护**：现有阈值宽松 100-500 倍且无基线持久化，不构成防护。建议基准结果落盘 JSON + 相对上次基线 ±20% 报警。
- **fp16/bf16、多头 head 选择、`mace_mp`/`mace_off` 入口、conv_tp fast path（l>0 irreps）**：全部零或近零覆盖，是 §3 问题能存活到 v0.2.0 的直接原因。
- **MD 能量守恒测试强度**：10 步 × 0.5 fs、漂移阈值约为体系总动能的 2.5 倍——放宽到没有约束力。建议 NVE 500 步 + 漂移 < 1 meV/atom/ps 量级。
- 基准方法学：`benchmark_all_models.py` 只报 n=5 均值（应报 median/min + std）；MLX 与 torch 对比不对称（MLX 复用 NL 缓存、torch 每次重建）；`pytest.mark.timeout` 无 pytest-timeout 插件支撑。

---

## 9. 原型与数据存档

本报告全部实测数据与可复现脚本在 session scratchpad（`profile_matrix.py`、`proto_compile.py`、`proto_skiptp.py`、`proto_symcon.py`、`proto_sparse_sc.py`、`proto_sparse_sc2.py`、`proto_e2e_stack.py`、`profile_results.json`、`review_findings.json`）。审查智能体的完整发现清单（29 confirmed / 30 partial / 21 主审复核，含逐条代码证据与修法）在 `review_findings.json` 与 `unverified_findings.json`。

### 优先级速查表

| 优先级 | 项目 | 类型 | 预期收益 | 依据 |
|---|---|---|---|---|
| P0 | fp16/bf16 修复或禁用（§3.1） | 正确性 | 消除错误结果 | 实测 |
| P0 | ZBL 静默丢弃（§3.2） | 正确性 | 修正短程势能面 | 已验证 |
| P0 | mace_off/torch 依赖/默认模型（§3.3） | 兼容性 | 安装即用 | 已验证 |
| P1 | 第二层 conv_tp 批量化（§4.1） | 性能 | 整步 15-25% | 估计 |
| P1 | SC 稀疏化 + custom VJP（§4.2） | 性能+内存 | prod 2-3×、内存 3-5× | 前向已实测 |
| P1 | skip_tp guard/gather_mm（§4.3） | 性能+内存 | 整步 5-9% | 实测 |
| P1 | NPT 单遍化（§3.4） | 性能 | NPT 约 2× | 机制已验证 |
| P2 | calculator 缓存（§4.4） | 性能 | 小体系 10-20% | 估计 |
| P2 | 常量折叠系列（§4.5） | 性能 | 3-8% | 部分已验证 |
| P2 | mx.compile 整步（§4.6） | 性能 | 1.03-1.45× | 实测 |
| P2 | mlx pin ≥0.31.2（§5） | 基础 | 泄漏修复+免费加速 | 溯源 |
| P3 | 死代码清理/持久缓存/README（§6） | 质量 | 维护性 | 已验证 |
| P3 | 路线图 1-8（§7） | 功能 | — | — |
