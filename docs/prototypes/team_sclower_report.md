# SC lower 链（nu<correlation）与入口装配：反向结构分析与原型报告

口径：medium-mpa-0，b=1000，c=128，fp32；带宽 224 GB/s、GEMM 4 TFLOPS
（沿用 sparse_sc_results.md §4 的 M4 Pro 实测口径）。force 反向 = 对 x
求导，dW 分支被 MLX lazy 求值剪枝（已验证的既有结论）。

实测形状（team_sclower_shapes.py，构造期 numpy）：

| contraction | main prefix | main nrow | lower0 (nu=2) | lower1 (nu=1) |
|---|---|---|---|---|
| p0 lout=0 | 256 | 99 | U2t=(k=4, 256), po=16, i=16 | (k=1, 16), po=1 |
| p0 lout=1 | 768 | 233 | U2t=(k=6, 768), po=48, i=16 | (k=1, 48), po=3 |
| p1 lout=0 | 256 | 99 | 同 p0 lout=0 | 同 |

关键事实：**lower 的 k 维极小（6/4/1），GEMM FLOPs 全部可忽略
（lout=1 iter0 仅 1.18 GFLOP = 0.3 ms），lower 链是纯带宽问题**。

## 1. 反向图纸面拆解（lout=1 contraction，b=1000）

张量大小：(b,c,768)=393.2 MB，(b,c,48)=24.6 MB，(b,c,16)=8.2 MB。

前向（现状 `_call_unrolled` lower 循环）：

| # | op | 输出 | 流量 MB（读+写） | est ms |
|---|---|---|---:|---:|
| L1 | oh @ W_ck0 | (b,c,6) | 4 | 0.02 |
| L2 | W_sel0 @ U0t (6,768) | (b,c,768) | 3 + **393 写** | 1.77 |
| L3 | c0 + out0（add） | (b,c,768) | **786 读 + 393 写** | 5.27 |
| L4 | c_4d @ feat_col | (b,c,48) | 401 读 + 25 写 | 1.90 |
| L5-L8 | iter1 全部（k=1, po=3） | (b,c,3) | ~134 | 0.60 |
| | **fwd 合计** | | **~2140** | **9.6** |

反向（dout2 (b,c,3) 起；add 的 VJP 是 passthrough，零成本）：

| # | op VJP | 物化 | 流量 MB | est ms |
|---|---|---|---:|---:|
| B8 | iter1 外积 dc1 + dfeat1 | (b,c,48)+(b,c,16) | ~68 | 0.30 |
| B4a | dc0_4d = dout1 ⊗ f（外积） | **(b,c,48,16) = 393 MB 写** | 426 | 1.90 |
| B4b | dfeat0 = c_4d^T @ dout1 | 读 c_4d **393** | 426 | 1.90 |
| B3 | add passthrough → dout0 = dc0_4d | 0（同一数组） | 0 | 0 |
| Bacc | dfeat 三路累加（main + iter0 + iter1） | 2×add (b,c,16) | 49 | 0.22 |
| | **bwd 合计** | | **~970** | **4.3** |

- 跨迭代串行依赖：out0→L3→L4→out1→L7→L8，反向同链倒序，无并行余地。
- feat_col 三路复用：dfeat = df(main) + dfeat0 + dfeat1，MLX autograd
  自动累加，仅 2 个 (b,c,16) add，便宜。
- **dout0（B4a 的 393 MB 外积物化）是主收缩反向 dX = dout0 @ U_rows^T 的
  输入，任何方案都必须物化**——记账上归主收缩接口，不可省。
- lower 链 f+b 纸面 ≈ 3.11 GB ≈ 13.9 ms；与遗留实测 16.4 ms
  （lower_only 口径）的差 = kernel launch/图开销（~15 kernel）+ 口径差异：
  lower_only 里 L3 照付（+零常量的 add 不被优化掉）但 B4a 是死节点被剪
  （dout0 撞 stop_gradient），两相抵后纸面 ~12.1 ms，吻合。
- lout=0 同法：fwd ≈ 3.1 ms + bwd ≈ 1.3 ms ≈ 4.4 ms/contraction。
- **3 contraction lower 链纸面合计 ≈ 22.7 ms**（实际含 launch ~28 ms），
  是 SC 整块（60-80 ms）主收缩之外的第二大项。

## 2. 结构机会盘点

**a. U_lower 稀疏度**（实测，team_sclower_shapes.py）：

| | 元素级 nnz | 双线性 (k,i) 对 nnz |
|---|---|---|
| lout=1 lower0 | 70/4608 (1.5%) | **24**/96 |
| lout=1 lower1 | 3/48 | **3**/16 |
| lout=0 lower0 | 16/1024 (1.6%) | **16**/64 |
| lout=0 lower1 | 1/16 | **1**/16 |

U_lower 视为 U'(k, po, i) 的双线性 (k,i) 非零对只有 24/16/3/1 个——比主
收缩的 233/99 行还小一个量级，行压缩双线性（与主收缩同款）完全可行且
选择 GEMM 极薄。但注意：**压缩本身省不了 L2 的 393 MB 输出写**（输出
稠密），它的价值在与分配律联用（见 b）。

**b. (c_4d @ feat_col) 的 VJP 与分配律重排**：
- dc_4d 物化 (1000,128,48,16) 393 MB 本身不可避免——它就是 dout0，主收
  缩反向的必经输入（满秩：dout0[po,i] = dout1[po]·f[i] 覆盖全部 768 维）。
- 但 **c_tensor = W@U + out 的 768 维物化和 add 可以整体绕开**（分配律）：

      out1 = (W0@U0t + out0)_4d @ f = RC(W0, f) + out0_4d @ f

  RC 是 U_lower 非零 (k,i) 对上的行压缩双线性（24 对 → X_l (b,c,24) 仅
  12.3 MB），add 从 prefix=768（1180 MB 流量）降到 po=48（74 MB）。
  L2 的 393 MB 写、L3 的 1180 MB add 全部消失；coup 项读 out0 与原 L4
  读 c_4d 等量；反向新增 RC 小链 ~0.5 ms。
- 完全展开成 f 的多线性形式（把主收缩耦合项也展开）**不可行**：ν=3 耦
  合项的 (r,i1,i2) 组合数 = U_rows 元素级 nnz = 2838 → X''' (b,c,2838)
  = 1.45 GB，组合爆炸（与 §6 sela 教训同型）。逐级收缩 out0 是正确结构；
  ν=2 项一级化（nnz≤70 直接出 q）仅再省 ~0.2 ms，不值得复杂化。
- 备选低风险版：**mx.addmm**（mlx 0.31.2 存在，batched+VJP 齐全，数值位
  级一致）把 L2+L3 融成单 kernel：省 add 的独立读写（lout=1 省 787 MB
  = 3.5 ms），结构零改动。收益约为分配律的 6 成。

**c. 入口装配**：代码确认 `SymmetricContraction.__call__` 的 blocks
slice/reshape/concat 只做一次（x 在循环外构造，2 个 contraction 共享），
无重复。VJP 链 = concat←slice + slice←pad-scatter，总流量 ~66 MB ≈
0.3 ms。**无优化空间，不动**。

**d. 跨 contraction CSE**：同一 SC 内 2 个 contraction 独立算
feat@SelI（可拼列）、oh@W_ck（3 GEMM/contraction，可拼成 1）；输出总量
不变，省的只是 ~6-8 次 kernel launch 和 f 的重复读 8 MB，**上限 ~0.5
ms/整步，不值得单独做**（products[0]/[1] 输入不同，无跨 SC 的 CSE）。

## 3. 上限量化（带宽下限）

lower 链的结构必要流量（给定主收缩接口：out0 已物化、必须产出 dout0
和 dfeat；fwd/bwd 是两个 pass 无法共享读）：

| 必要项 | lout=1 | lout=0 |
|---|---:|---:|
| fwd 读 out0 一次（收缩进 out1） | 393 | 131 |
| bwd 写 dout0 一次（外积，主收缩输入） | 426 | 139 |
| bwd 再读 out0 一次（dfeat 的 coup 路） | 401 | 139 |
| RC/小张量/累加 | ~250 | ~90 |
| **下限合计** | **1.47 GB ≈ 6.5 ms** | **~0.50 GB ≈ 2.2 ms** |

对照：

| lower 链 f+b | lout=1 | lout=0 | 3 contraction 合计 |
|---|---:|---:|---:|
| 现状纸面 | 13.9 | 4.4 | 22.7 |
| addmm 纸面 | 10.2 | 2.5 | 15.2（省 7.4） |
| **distrib 纸面** | **7.9** | **2.6** | **13.1（省 9.6）** |
| 带宽下限 | 6.5 | 2.2 | 10.9 |

结论：**值得做，且已做**。lout=1 单 contraction 空间 6.0 ms ≥ 5 ms 门
槛；distrib 已把 lower 链推到距结构下限 ~2 ms 处（剩余 = launch 开销 +
RC 链非最简），这个方向到 distrib 为止，再往下不值得。预期端到端
medium/Si1000 省 ~8-10 ms（402 → ~393，约 2.4%）；lower_only 16.4 ms
口径预期降到 ~8-9 ms。

## 4. 原型与数值验证

脚本 `team_sclower_proto.py`（自包含，不改库文件——避免污染并行 agent
的测量）。三变体：

- `baseline`：库现状路径（sparse main + dense lower）。
- `addmm`：`c_t = mx.addmm(out, W_sel_i, U_t)`，其余不动。
- `distrib`：每个 lower 迭代改写为
  `out = (x@SelI_l * W_sel@SelK_l) @ U_l_rows + (out_4d @ feat_col)`，
  构造期 numpy 提取 U_lower 非零 (k,i) 对（`build_lower_rc_consts`，
  含 mx.eval 防 lazy-capture bug）。

数值验证（--check，b=64，随机 cotangent 加权和，fwd 与 dfeat 对
baseline，3 个 contraction 全过）：

| 变体 | fwd max abs (rel) | dfeat max abs (rel) |
|---|---|---|
| addmm | 0.0（位级一致） | 0.0（位级一致） |
| distrib | 3.8e-06 (≤1.4e-07) | 3.8e-06 (≤1.2e-07) |

均远低于 1e-5 验收线（distrib 的差异 = fp32 求和顺序重排，与主收缩稀疏
化同量级）。addmm 位级一致同时证明其 VJP 注册正确。

落地建议（验证计时后）：`symmetric_contraction.py` 增加
`_setup_sparse_lower`（每迭代 SelI/SelK/U_rows 常量 + mx.eval），
`_call_unrolled` 的 lower 循环换成 distrib 形式；guard：nnz==0（全零
U_lower）时退化为纯 coup 项，nnz > 0.5·i·k 时回落现状路径（与主收缩
guard 同型）。corr=2/4（small/large）结构相同，自动受益。

## 5. 待计时清单（主会话串行跑，GPU 独占时）

    P=/private/tmp/claude-501/-Users-mastreina-Desktop-mace-mlx/25918f5b-8f0e-48d3-a2d3-5900e397f165/scratchpad/team_sclower_proto.py
    PY=.venv/bin/python   # 仓库根目录下

    # 1) 单 contraction f+b（对 x 求导，warmup 3 + 10 中位，含峰值内存）
    for v in baseline addmm distrib; do
      $PY $P --bench --contraction 1 --variant $v          # lout=1（主战场）
      $PY $P --bench --contraction 0 --variant $v          # lout=0
    done

    # 2) lower_only 口径（对齐遗留 16.4 ms 数据点）
    $PY $P --bench --contraction 1 --variant baseline --mode lower_only
    $PY $P --bench --contraction 1 --variant addmm    --mode lower_only
    $PY $P --bench --contraction 1 --variant distrib  --mode lower_only

    # 3) 整 SC（含入口装配，products[0] 两个 contraction）
    $PY $P --bench --scope sc --contraction 0 --variant baseline
    $PY $P --bench --scope sc --contraction 0 --variant distrib

验收参考线（纸面预测的 2/3 计）：
- distrib lout=1 f+b 相对 baseline 省 ≥4 ms（纸面 6.0）；lower_only
  16.4 → ≤11 ms。
- addmm lout=1 省 ≥2.5 ms（纸面 3.7）。若 addmm 与 baseline 持平，说明
  mx.addmm 在该 batched 形状下未走融合 kernel，弃 addmm 只保 distrib。
- 峰值内存：distrib 不物化 c_tensor，f+b 峰值预计持平或略降（buffer
  复用影响，以实测为准）。
- 若达线：按 §4 落地建议改库 + 跑既有稀疏/稠密一致性测试 + 端到端 A/B。
