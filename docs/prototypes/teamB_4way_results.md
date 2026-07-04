# teamB：v0.4.0 后四方向评估——结果索引

> 2026-07-04，四个并行 agent + 主会话串行计时（GPU 独占，warmup 3 +
> 10 次中位）。基线 = v0.4.0（sparse SC，commit 734fccd），medium-mpa-0
> / Si1000 端到端 381.5 ms / 峰值 10.7 GB。各方向详细报告见对应
> `team_*_report.md`，原型与计时脚本同目录。

## 结论一览

| 方向 | 块级结果 | 端到端（medium/Si1000） | 判定 |
|---|---|---|---|
| ① fp16 特征路径 | 误差评估（见下） | 1.45-1.52×（v0.4.0 上实测） | 分场景可用，默认保持 fp32 |
| ② conv_tp 手写 VJP（GEMM 重组） | f+b3 237→143.5 ms（1.65×） | 预期 ~309 ms | 达标但被 ③ 同块淘汰 |
| ③ **Metal kernel 融合** | mul21 f+b 266.7→**37.2 ms（7.2×）**；scx 省 ~9 ms | **381.5→145.7 ms（2.62×）**；Si2000 293.9 ms / 峰值 8.2 GB | **落地主线** |
| ④ SC lower 链 distrib | lout=1 contraction f+b 45.3→34.3 ms | 3 contraction 约省 14-16 ms | 达标，与 ③ 正交可叠加 |

组合潜力：③both（146 ms）+ ④（约 -14 ms）+ ①fp16（÷~1.45）≈
**90-100 ms / Si1000**（未联测，逐项落地时验证）。

## ① fp16（team_fp16_report.md）

「误差是否强烈」分场景两真两假：MD-NVT/NPT 与粗弛豫（fmax≥0.01）
误差不强烈（力相对 RMS ≤1%、NVT 统计与 fp32 一致、BFGS 步数无差）；
高应变构型绝对能量（应变 SiO2 实测 -10.4 meV/atom 系统偏移）与
二阶导数类（h=0.005 有限差分力常数误差 12.4%，声子不可用）担心成立。
fp16 弛豫实用下限 fmax≈0.005（力噪声地板 ~0.007 eV/Å）。

## ② conv_tp 手写 VJP（team_convtp_report.md）

反向拆解：autograd 反向 ~32 GB 流量，62% 是装配 slice-VJP 链
（pad+add 展开）。手写 VJP 重组到 13-15 GB，块级 1.65×，数值 ≤3.5e-7。
**被 ③ 淘汰**（同块 143.5 vs 37.2 ms）：GEMM 重组仍受「每个 GEMM 的
输入输出必须物化」约束，kernel 融合直接消掉物化，代差。其反向流量
拆解是 ③ 设计的直接依据，分析价值保留。

## ③ Metal kernel（team_metal_report.md）

`mx.fast.metal_kernel` API 探针 12/12 通过（fp16 模板、custom_function
手写 VJP、compile(value_and_grad)、死分支剪枝、多输出、原子输出）。
conv_tp 整块单 kernel（前向 1 kernel 直接产出最终槽布局、反向 1 个
4 输出 kernel + dx2 两个小 GEMM）：块级 f+b 7.2×、峰值 11.2→3.5 GB，
数值 ≤2.5e-7。端到端 both（mul21+scx）：Si1000 145.7 ms、
Si2000 293.9 ms / 8.2 GB。落地需要：guard（uvu/mul2=1/槽覆盖布局）+
CPU stream 回落 + 全测试回归 + fp16 组合验证。

## ④ SC lower 链 distrib（team_sclower_report.md）

分配律 `(W@U + out)_4d @ f = RC(W,f) + out_4d @ f` + lower U 行压缩
（双线性 (k,i) 非零对仅 24/16/3/1 个）：c_tensor 393 MB 物化与
prefix=768 上 1180 MB 的 add 整体消失。lout=1 f+b 45.3→34.3 ms、
lower_only 口径 16.1→10.6 ms（带宽下限 ~6.5 ms），数值 3.8e-06。
`mx.addmm` 低风险对照（位级一致）也有效但收益较小（45.3→36.8）。

## 跨实现基准脚本

`xbench.py`（生成 `profile_cross_impl.json` 的脚本，此前漏归档）：
mace-torch cpu fp64/fp32、mps fp32（含官方不支持 MPS 的 workaround
说明）、mace-mlx 0.2.0/0.3.0/0.4.0 六配置对比。
