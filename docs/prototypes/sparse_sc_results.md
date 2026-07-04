# SymmetricContraction 稀疏化：实施结果报告

- 任务定义与验收标准：`docs/HANDOFF_SC_SPARSE.md`。本文是其结果报告。
- 测量环境：Apple M4 Pro（48 GB），mlx 0.31.2，fp32，GPU 串行独占。
  计时 warmup 3 + 10 次取中位，每次 `mx.eval`；峰值内存
  `mx.reset_peak_memory()`/`mx.get_peak_memory()`，fwd 与 fwd+bwd 分别测。
  compile 对照每变体独立进程（规避 `mx.compile` 按函数对象 id 的缓存污染）。
- 块级基准 b=1000；`dense` 指 v0.3.0 现状（weights-first 稠密 GEMM）。

## 1. 最终方案：U 的 (i,k) 行压缩双线性 GEMM（无 custom VJP）

主收缩 `out[b,c,p] = Σ_{i,k} U[p,i,k]·W[b,c,k]·f[b,c,i]` 中，U 只有
nrow 个非零 (i,k) 对（lout=0: 99/368，lout=1: 233/816，≈30%；元素级
非零仅 0.4-0.5%）。只在这些行上求值：

    X = (f @ SelI) * (W_sel @ SelK)     # 0/1 选择矩阵 GEMM ×2 + 对齐乘
    out = X @ U_rows                    # (nrow, prefix) 有值 GEMM

全链只有 GEMM 和对齐 elementwise：autograd 的 VJP 自动是转置 GEMM
（无 gather/scatter），巨型 (b,c,prefix,i) 中间张量 WU 彻底消失，
force-only 反向下 dW 链是死节点（MLX lazy 求值自动剪枝，实测免费）。
**不需要 `mx.custom_function`**，与 `mx.compile` 天然兼容。

实现落点：`mace_mlx/symmetric_contraction.py` 的 `_setup_sparse_main`
（构造期 numpy 提取 + 常量物化）与 `_call_unrolled` 主收缩分支；
guard：`nrow == 0`（全零 U）或 `nrow > 0.5·i·k`（稀疏度不足，小玩具
irreps 会触发）时回落稠密路径。低阶（nu<corr）步骤保持现状。

## 2. 块级验收（medium products[0]，b=1000，力式反向 = 对 x 求导）

| 指标 | dense | 行压缩 | 改善 | 验收线 |
|---|---:|---:|---:|---|
| lout=0 fwd | 25.8 ms | 7.1 ms | 3.6× | — |
| lout=0 f+b | 39.7 ms | 14.5 ms | **2.74×** | ≥2× ✓ |
| lout=0 f+b 峰值 | 2579 MB | 653 MB | **3.95×** | ≥3× ✓ |
| lout=1 fwd | 81.1 ms | 23.5 ms | 3.5× | — |
| lout=1 f+b | 116.3 ms | 42.7 ms | **2.72×** | ≥2× ✓ |
| lout=1 f+b 峰值 | 7635 MB | 1723 MB | **4.43×** | ≥3× ✓ |

全梯度（x 与权重同时求导）f+b：lout=1 46.2 ms（dense 109+），dW 与
稠密 autograd 位级一致。compile 版性能相同（matmul 不参与融合）。
数值：块级 fwd ≤7.6e-06、df ≤3.8e-06（fp32 求和重排量级）。

## 3. 端到端 A/B（ASE calculator 整步含 compile，中位 10 次，同缓存模型）

| model/system | e2e dense→sparse | 加速 | 峰值 dense→sparse | 改善 |
|---|---:|---:|---:|---:|
| small/Si216 | 35.6→24.9 ms | 1.43× | 2164→1584 MB | 1.37× |
| small/Si1000 | 166.8→115.3 ms | **1.45×** | 6750→2894 MB | **2.33×** |
| medium/Si216 | 106.3→80.8 ms | 1.32× | 4512→2694 MB | 1.67× |
| medium/Si1000 | 504.3→379.0 ms | **1.33×** | 18878→10659 MB | 1.77× |
| medium/Si2000 | 1300.8→763.1 ms | **1.70×** | 37686→21249 MB | 1.77× |
| large/Si512 | 391.4→260.3 ms | **1.50×** | 16220→6857 MB | **2.37×** |

原始数据 `profile_sparse_sc_ab.json`（dense 口径与 v0.3.0 基线
`profile_after.json` 吻合，±2%）。medium/Si2000 峰值 21.2 GB，落在交接
文档修正后的 20-25 GB 预期；SC 的 WU（原 ~6.3 GB/千原子·contraction）
消失后峰值来源移到第二层 interaction 逐边中间量。large 的 lout=2 同样
激活并受益。Si2000 的 1.70× 高于其他体系，来自脱离换页压力。

数值验收（medium/Si216，rattle 0.05）：fp32 sparse vs dense
dE=0.0000 meV/atom、dF_max=4.5e-06 eV/Å（<1e-5 ✓）；fp16 vs fp32
sparse 0.990 / dense 0.938 meV/atom（恶化 1.06×，<2× ✓），力差
0.0139/0.0140 eV/Å 持平。回归 `pytest -m "not benchmark"` 640 passed
（基线 615 + 新增 25：稀疏/稠密两路径 fwd、df、dW 一致性 ×7 配置组合
（corr 2/3/4 × lout 0/1/2）、guard 行为、fp16 常量转换双机制）。

## 3.5 跨实现对比（mace-torch vs mace-mlx 三版本）

同一构型（rattle 0.05 种子 42）、同一计时口径（warmup 3 + 10 次中位，
清缓存 energy+forces），每配置独立进程。mace-torch 0.3.16；mlx 旧版本
用 git worktree（v0.2.0=4d07c83、v0.3.0=3c238cb）+ PYTHONPATH 隔离。
原始数据 `profile_cross_impl.json`。e2e 单位 ms：

| 配置 | small/Si1000 | medium/Si216 | medium/Si1000 | medium/Si2000 |
|---|---:|---:|---:|---:|
| torch cpu fp64（官方默认） | 568.2 | 458.0 | 2101.1 | 4293.2 |
| torch cpu fp32 | 322.0 | 281.0 | 1181.0 | 2354.9 |
| torch mps fp32（见注） | 361.8 | 244.5 | 1267.6 | 2914.0 |
| mace-mlx 0.2.0 | 185.3 | 150.1 | 701.9 | 2489.6 |
| mace-mlx 0.3.0 | 179.0 | 114.2 | 534.8 | 1543.3 |
| **mace-mlx sparse（本次）** | **123.3** | **86.3** | **402.0** | **799.9** |
| 本次 vs torch fp64 默认 | 4.6× | 5.3× | 5.2× | 5.4× |
| 本次 vs torch fp32 同精度 | 2.6× | 3.3× | 2.9× | 2.9× |
| 本次 vs 0.3.0 | 1.45× | 1.32× | 1.33× | 1.93× |
| 本次 vs 0.2.0 | 1.50× | 1.74× | 1.75× | 3.11× |

- **torch MPS 注**：mace-torch 官方不支持 MPS——fp64 checkpoint 无法
  map 到 MPS 设备，且 forward 硬编码 `.double()` 能量累加。表中数字
  是 workaround 参考值（CPU 加载转 fp32 后搬 MPS + `.double()` 在 MPS
  张量上降级 `.float()`；数值 vs CPU fp32 dF≤2e-05 eV/Å）。即便如此，
  MPS 只在小体系上略快于 CPU fp32，Si1000/Si2000 反而更慢。
- 各配置能量一致（Si2000 全部落在 -10739.39±0.02 eV，0.02 meV/atom
  精度带），力最大值一致。
- mlx 0.2.0 在 Si2000 上换页（2489.6 ms，甚至慢于 torch cpu fp32）；
  0.3.0 部分缓解（1543.3，仍在 38 GB 峰值的换页敏感区，波动大）；
  本次 21.2 GB 峰值彻底脱离，Si2000 步进 800 ms。
- xbench 与 §3 的 ab_bench 环境略有差异（±5-8%），版本间相对比较
  以本表（同环境同批）为准。

## 4. 探索路径与中间数据（勿重做）

行压缩不是第一个尝试。中间变体数据（medium，b=1000，f+b 为对 x 求导）：

| 变体 | lout=0 f+b(峰值MB) | lout=1 f+b(峰值MB) | 结论 |
|---|---:|---:|---|
| dense 现状 | 39.7 (2579) | 116.3 (7635) | 基线 |
| sela 列压缩+纯选择矩阵 | 23.7 (1025) | 134.1 (3454) | lout=1 聚合 GEMM (1371,768) FLOPs 超标，出局 |
| selb padded(P,width) 选择矩阵 | 29.8 (1662) | 83.6 (4883) | fwd 赢（47ms），f+b 1.39× 不达线 |
| vjpb selb+custom VJP | 29.6 (1680) | 83.1 (4904) | 与 selb 持平——MLX lazy 求值本来就剪掉了 dW 分支 |
| vjpb_take（custom 内 take 展开） | 30.8 (2073) | 86.6 (6083) | take 比选择矩阵 GEMM 慢且费内存 |
| **rowc 行压缩**（proto_sparse_sc5.py） | **14.5 (653)** | **42.7 (1723)** | 达标，落地 |

关键测量（`micro_sc_breakdown.py`，lout=1 单 kernel）：selb 的瓶颈
**不在 GEMM**（T GEMM 7.6 ms、df GEMM 6.6 ms，~4 TFLOPS）而在两个
elementwise 广播/归约（`(T*F).sum(-1)` 22.8 ms、`d4*T` 22.2 ms，物化
4D 中间的带宽）。行压缩把乘法收进 (b,c,nrow) 对齐形式，消掉全部 4D
物化，这是它赢的机制。custom VJP 对 force-only 反向无增益：MLX 反向
传播只沿依赖路径，vjpb 的 f+b(x) 与 autograd 的 selb 相同（83 ms）；
custom_function 的 vjp 返回的死节点 cotangent 也不会被求值。

## 5. 上游 bug 记录：mx.compile 捕获未求值闭包数组（MLX 0.31.2）

原型阶段 `proto_sparse_sc4.py --variant vjpb_c` 观察到 compile 图 fwd
输出 nan/inf/随机错误值。二分定位（`repro_compile_lazy_capture.py`）：
**与 custom_function 无关**。触发条件是被 compile 的函数通过闭包引用
在 trace 之前构建、但从未 `mx.eval` 的**派生**数组（有待执行计算图的
非叶子数组，此处为 `_ensure_weight_caches` 的 transpose/reshape 结果
链），compiled 图执行读到未初始化数据；错误值跨运行不稳定，且可污染
同一 `mx.eval` 批次的无关数组。从 numpy 直接构造的叶子数组不触发；
trace 内部构建的缓存被烘进图，也不触发（calculator 生产路径属此类，
安全）。防御：`_setup_sparse_main` 构造常量后立即 `mx.eval`。

## 6. 负结果清单（新增，接续交接文档 §6）

- 纯 (ncol, prefix) 选择矩阵聚合（sela）：lout=1 上 FLOPs 超过稠密基线，
  前向即输。列压缩后仍是 nnz→prefix 稠密 GEMM 的老问题。
- padded (P,width) 选择矩阵 + custom VJP（vjpb）：数学正确、lazy 剪枝
  有效，但对 force-only 反向零增益——autograd 已经不算 dW 分支；瓶颈
  在 4D elementwise 物化，重写 VJP 不解决。
- custom_function 内用 mx.take 做前向展开（vjpb_take）：比选择矩阵
  GEMM 慢 ~8%、内存 +36%（(b,c,P·w) gather 写放大）。
