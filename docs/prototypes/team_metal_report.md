# Metal kernel 融合探索：结果报告

> agent 被会话限额打断，原型与脚本已完成；数值验证与全部计时由主会话
> 串行补齐（GPU 独占）。数据 team_metal_bench_results.json。

## 1. API 结论（team_metal_api_probe.py，12/12 通过）

`mx.fast.metal_kernel`（MLX 0.31.2）：fp32/fp16 模板、非对齐 grid、
多输出、原子输出全部可用；`mx.custom_function` 手写 VJP、`mx.compile`、
`compile(value_and_grad(...))` 组合全部正确；死分支剪枝对分离的 VJP
kernel 生效（force-only 时 dW kernel 不执行）；CPU stream 明确报错
（GPU-only，需在 dispatch 前 guard）。**API 层面无障碍。**

## 2. 原型与块级结果（medium-mpa-0 真实常量，fp32，warmup 3 + 10 中位）

### 2a. scx：SC 主收缩 X 构造融合（team_metal_proto_scx.py）

X[b,c,r] = f[b,c,i_r]·W[b,c,k_r] 单 kernel（双 gather+乘），反向
custom_function：df/dW 各一个 CSR 分组 kernel（无原子操作，dW kernel
在 force-only 反向被剪枝）。数值：fwd 位级一致，梯度相对误差 ≤2.5e-7，
compile(vag) 正常，fp16 冒烟过。

| b=1000 | f+b ref | f+b fused | 峰值 |
|---|---:|---:|---|
| lout=0 | 13.4 ms | 11.4 ms | 622→571 MB |
| lout=1 | 41.3 ms | 36.5 ms | 1687→1568 MB |

温和收益（3 contraction 合计 ~9 ms）。

### 2b. mul21：第二层 conv_tp 整块单 kernel（team_metal_proto_mul21.py）

前向一个 kernel（threadgroup=边，thread=通道）直接产出最终槽布局的
mji (E,5120)：w_t 转置、M1 广播、选择矩阵展开、乘法、10 槽装配全部
在寄存器/threadgroup 内存完成——production 路径的 ~3.4 GB 中间物化
（含 942 MB concat 写读）整体消失。x2 仍经两个小 GEMM（M1=x2@G1、
xs=x2@S）进入。反向一个 4 输出 kernel（dx1/dM1/dxs/dw，kernel 内重算，
simdgroup 归约）+ dx2 两个小 GEMM。数值：fwd 位级一致，dx1/dx2/dw
相对误差 ≤2.5e-7，compile(vag) 正常。

| E=46000 | ref | fused | 比 |
|---|---:|---:|---:|
| fwd | 66.9 ms | **7.2 ms** | 9.3× |
| f+b(x1,x2,w) | 266.7 ms | **37.2 ms** | 7.2× |
| f+b 峰值 | 11.2 GB | 3.5 GB | 3.2× |

**同块对决**：convtp-vjp 组的手写 VJP（GEMM 重组，v1+compile）同口径
f+b3 = 143.5 ms——单 kernel 融合快 3.9×，胜出。GEMM 重组仍受「每个
GEMM 输入输出必须物化」约束，kernel 融合直接消掉物化，这是代差。

## 3. 端到端（ASE calculator 整步含 compile，medium-mpa-0）

| 体系 | ref (v0.4.0) | scx | mul21 | both | 峰值 both |
|---|---:|---:|---:|---:|---|
| Si1000 | 381.5 ms | 374.7 | 153.1 | **145.7 ms（2.62×）** | 10.7→5.3 GB |
| Si2000 | 772.5 ms | — | — | **293.9 ms（2.63×）** | 21.2→**8.2 GB** |

## 4. 可行性判断

**值得落地，且是当前最大的单项机会。** 收益 2.6× 远超估计（1.3-1.8×）。
成本与风险：
- Metal source 以字符串形式内嵌 Python，调试面窄；布局假设
  （uvu、mul2=1、槽覆盖）需与 `_setup_batched_mul21` 同款 guard，
  不满足回落现路径；
- GPU-only：CPU stream 需回落（explicit guard 已验证会报错而非静默错）；
- threadgroup 尺寸=mul（128/256）依赖设备上限（M 系列 1024，安全）；
- fp16：kernel 模板已验证，端到端 fp16+metal 组合未计时（下一步）；
- 维护：每个新模型结构变体都要过 guard 或扩 kernel。

落地清单：kernel 源与常量进 `tensor_product.py`（`_setup_metal_mul21`），
dispatch 顺序 metal → batched_mul21 → loop；scx 同理进
`symmetric_contraction.py`；全测试回归 + 真实模型 dF 验收 + fp16 组合。
