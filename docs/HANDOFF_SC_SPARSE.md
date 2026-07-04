# 交接：SymmetricContraction 稀疏化 + custom VJP

> 写给接手本任务的工程师/智能体。本文档自包含：读完即可开工，不需要本仓库
> 之外的上下文。前置成果全部在仓库内，引用路径均为仓库相对路径。

## 0. 环境与工作方式

- 仓库：`/Users/mastreina/Desktop/mace-mlx`，Python 环境 `.venv/bin/python`
  （mlx 0.31.2、torch、mace-torch、e3nn、pytest 已装）。
- 机器：Apple M4 Pro，48 GB 统一内存。GPU 基准必须串行（不要并行跑两个基准）。
- 已转换模型缓存：`~/.cache/mace_mlx/<name>/v2/`（`small`、`medium-mpa-0` 已有；
  用 `mace_mlx.model.load_model(dir)` 加载）。
- 回归测试：`.venv/bin/python -m pytest tests -q -m "not benchmark"`
  （当前 615 passed，改动后必须保持全过）。
- 计时方法约定：warmup 3 + 10 次取中位，每次 `mx.eval` 同步；
  峰值内存用 `mx.reset_peak_memory()` / `mx.get_peak_memory()`。

## 1. 任务定义

把 `mace_mlx/symmetric_contraction.py` 的主收缩从稠密 GEMM 改为稀疏收缩，
并用 `mx.custom_function` 手写 VJP，使**前向与反向都**避开巨型中间张量。

验收标准：

1. 数值：与现实现输出差 < 1e-5（随机输入，逐 correlation/lout 覆盖
   corr=3 × lout∈{0,1}，以及至少一个真实模型端到端 dF < 1e-5 eV/Å）。
2. 性能：product 阶段（fwd+bwd）≥ 2×，峰值内存 ≥ 3×改善
   （medium/Si1000 的 prod0 现状约 104 ms fwd / 见 §3 数据）。
3. `pytest tests -m "not benchmark"` 全过；fp16 路径可用（见 §6 dtype 约定）；
   与 `mx.compile` 兼容（calculator 默认 compile 整步，custom_function 在
   compile 内的行为需实测确认，不兼容则在 calculator 层对该块禁用 compile
   并记录）。
4. 端到端 A/B：用 `docs/prototypes/profile_after.json` 作为本次起点基线，
   同体系矩阵复测（脚本参考 `docs/prototypes/` 同名 py 的计时写法）。

## 2. 问题本质（为什么值得做）

- SymmetricContraction 是 MACE 的多体乘积块。当前实现
  （`symmetric_contraction.py` 的 `_call_unrolled`）把按元素选好的权重
  `W_sel (b, c, k)` 与展平 U 矩阵 `_u_main_wf (k, prefix*i)` 做稠密 GEMM，
  物化中间张量 `WU (b, c, prefix, i)`：
  - lout=0：prefix=256, i=16, k=23 → 每原子 2.1 MB（c=128）
  - lout=1：prefix=768, i=16, k=51 → 每原子 6.3 MB
- **U 张量 99.5% 是零**（CG 系数结构）：lout=0 nu=3 非零 353/94208（0.4%），
  lout=1 nu=3 非零 2838/626688（0.5%）。稠密计算的 FLOPs 和带宽几乎全部
  浪费在零上。
- 后果（实测，profile_results.json）：prod 是 mp0-small 的第一热点
  （两层约 63% 前向）、mpa0-medium 的第二热点（31%）；medium 峰值内存
  19-38 GB 的第一来源，2000 原子触发换页。
- 低阶（nu<correlation）步骤和特征收缩步骤都不是问题——**只有主收缩
  （nu=correlation）的 WU 需要稀疏化**，其余保持现状。

## 3. 已验证的原型与数据（不要重做这些实验）

原型脚本与原始数据在 `docs/prototypes/`：

- `proto_symcon.py`：基线测量 + U 稀疏度统计 + **两个负结果**：
  - `mx.checkpoint` 无效（fwd+bwd 更慢且峰值更高，勿再试）；
  - one-hot→gather 换权重选择无感（瓶颈不在权重选择，在 WU GEMM）。
- `proto_sparse_sc.py`：稀疏 v1，三种聚合方式（b=1000，fwd+bwd 为对 x1 求导）：

  | 变体 | lout=0 fwd | lout=0 f+b | lout=1 fwd | lout=1 f+b | lout=1 fwd 内存 |
  |---|---:|---:|---:|---:|---:|
  | 稠密现状 | 24.7 ms | 38.1 ms | 78.0 ms | 112.0 ms | 7.2 GB |
  | 稀疏 S-matmul | 15.4 | **30.7** | 156.6 | 311.6 | 5.3 GB |
  | 稀疏 at[].add | 16.5 | 31.7 | 110.0 | 204.2 | 5.3 GB |
  | 稀疏 padded-CSR | 22.5 | 43.7 | 167.4 | 300.3 | 7.1 GB |

- `proto_sparse_sc2.py`：稀疏 v2「列压缩」——U_wf 只有约 10% 的**列**非零
  （lout=0: 353/4096；lout=1: 1371/12288）。先 GEMM 到非零列 (b,c,ncol)，
  再宽度 ≤3 的 take+add 循环聚合到 prefix：

  | 变体 | lout=1 fwd | lout=1 f+b | 内存(fwd) |
  |---|---:|---:|---:|
  | 稠密现状 | 78.0 ms | 112.0 | 7.2 GB |
  | 列压缩 | 66.6 | 165.5 | 4.2 GB |
  | 列压缩+compile | **49.2** | 164.6 | **3.1 GB** |

**核心结论**：前向已经能赢（1.6×，内存 2.3×），但所有 take/gather 基的
变体在**反向**都输——MLX 对 gather 自动生成的 VJP 是 scatter-add，把前向
省下的全部吃回。这就是必须手写 VJP 的原因。

## 4. 关键技巧：0/1 选择矩阵 GEMM（teamA 已在 conv_tp 验证）

读 `docs/prototypes/teamA_convtp_batch.md` 和 `proto_convtp_batch.py` 的
`forward_B3`：在 conv_tp 批量化中，用**常量 0/1 选择矩阵的 GEMM** 代替
`mx.take` 做索引展开后，autograd 生成的 VJP 自动变成转置 GEMM
（高效的 segment-sum），完全不用手写 custom VJP，全梯度 1.40×。
这一实现已落在 `mace_mlx/tensor_product.py` 的 `_batched_mul21_forward`，
可直接对照。

对 SC 的启示（建议的探索顺序）：

1. **先试纯选择矩阵方案**：把 v2 列压缩的「take+add 聚合」换成
   (ncol, prefix) 的 0/1 稀疏模式矩阵 GEMM 或其分块形式。若 autograd 的
   反向已经足够快（对照 §3 验收线），可能根本不需要 custom_function——
   这是最低成本路径，先测它。注意 lout=1 的 (ncol=1371, prefix=768)
   模式矩阵 GEMM FLOPs 是否可接受（v1 的 S-matmul 在 lout=1 上输就输在
   nnz→prefix 的稠密 GEMM 太大，列压缩后 ncol 比 nnz 小一半，重新测）。
2. **不够再上 custom VJP**：`mx.custom_function` 装饰前向，`.vjp` 手写。
   数学：主收缩 out[b,c,p] = Σ_{(p,i,k)∈nnz} U_val · W[b,c,k] · f[b,c,i]，
   反向是同一张 nnz 表的转置收缩——
   - dW[b,c,k] = Σ_{(p,i)|k} U_val · dout[b,c,p] · f[b,c,i]（散射目标只有
     k≤51 个槽）
   - df[b,c,i] = Σ_{(p,k)|i} U_val · dout[b,c,p] · W[b,c,k]（散射目标只有
     i=16 个槽）
   两者都可以按 §4.1 的选择矩阵思路组织成小 GEMM。
   MLX custom_function 的 API 现状（本仓库审查时确认，mlx 0.31.2）：
   `@mx.custom_function` + `f.vjp(fn)`；与 compile 的组合行为未实测，
   落地时验证。
3. 低阶步骤（`W_sel_i @ u_lower_t` 与批 matvec）保持现状——它们又小又快。

## 5. 代码落点

- 主战场：`mace_mlx/symmetric_contraction.py` 的 `Contraction` 类。
  `_call_unrolled` 是生产路径（corr≤4 全走它）。建议模式与仓库现有
  fast path 一致：`__init__` 里做结构检测与常量预计算（numpy 构造 →
  `mx.stop_gradient(mx.array(...))` 存私有属性），forward 处 dispatch，
  guard 不满足回落现路径。
- 参考 dispatch 风格：`tensor_product.py` 的 `_batched_mul21`
  （检测 + `_setup_*` + forward + 回落）。
- U 矩阵来源：`Contraction.__init__` 里 `U_matrix_real(...)`
  （`clebsch_gordan.py`），nnz 结构在构造期用 numpy 提取。
- SymmetricContraction 入口 reshape（irreps-major → feature-major）不变。

## 6. 必须遵守的约定（这轮重构定下的）

- **dtype**：几何前端 fp32、特征路径 compute dtype（见 `model.py` 模块
  docstring）。SC 属于特征路径：常量按需转 compute dtype（fp16 加载时
  `_convert_private_arrays` 会转私有 mx.array 属性，确认新常量被覆盖且
  数值可接受）；任何 `mx.zeros` 显式带 `dtype=`（fp16 隐式回退教训）。
- **fp16 验收**：`default_dtype="float16"` 端到端仍 ~1 meV/atom 量级
  （现状 0.62，别恶化超过 2×）。
- **compile**：calculator 默认 `mx.compile` 整步。已知上游 bug：零边输入
  + GPU + compile 段错误（calculators.py 已有零边回退，别删）。
  custom_function 进 compile 图的行为要实测；不行就文档化 + 局部禁用。
- 计时对照与端到端 A/B 都要在**同一份模型文件**上做（`~/.cache/mace_mlx`）。
- 负结果清单（勿重试）：mx.checkpoint 换内存、权重量化、SH 硬编码多项式、
  CG 水平拼接（FLOPs 不变，teamA 证伪）、SC 权重选择 one-hot→gather。

## 7. 背景资料索引

- 全面审查报告：`docs/OPTIMIZATION_REVIEW.md`（§4.2 是本任务的审查条目；
  §1.5 是已实施项与当前基线）。
- 性能基线：`docs/prototypes/profile_results.json`（v0.2.0 审查基线）、
  `docs/prototypes/profile_after.json`（当前 v0.3.0 基线，**以此为起点**）。
- 当前整步水平（v0.3.0，M4 Pro）：medium/Si1000 e2e ≈ 503 ms（其中 prod
  仍占前向约 1/3）、small/Si1000 ≈ 165 ms（prod 占 ~63%，本任务的最大
  受益者）、medium/Si2000 峰值内存 ~38 GB。
- 端到端内存的正确预期（2026-07-04 修正）：峰值是 max-concurrent 而非
  求和——SC 的 WU 消掉后，峰值大概率被第二层 interaction 的逐边中间
  张量接管，Si2000 落点估计 20-25 GB。端到端验收为「SC 不再是峰值
  来源 + Si2000 峰值显著下降」，不以 15 GB 为硬线；SC 块级 ≥3× 的
  验收（§1）不变。
