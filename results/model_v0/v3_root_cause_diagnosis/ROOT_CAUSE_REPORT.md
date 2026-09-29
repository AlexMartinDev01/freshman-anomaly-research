# V3 catastrophic degradation —— 根因结案报告

> 诊断对象：`model_v3_implementation_frozen` = `3049e1c3ec0708fa405ed2f0fca53d87e5a214db`
> 诊断方式：只读复算 + 固定位置 2×2 因果实验
> **本报告写定后，禁止根据后续 capsule 性能继续修改根因解释。**

---

## 0. 结论摘要

冻结实现 `3049e1c` 的 catastrophic degradation 由**两个独立因素叠加**造成：

```
根因 1  crop-local z-score 破坏绝对异常幅值        (score-definition 实现不匹配)
根因 2  crop-DINO 丢失 full-image ViT 上下文       (local-refinement 设计缺陷)
```

**两者都不是 selector 的问题，不是高分辨率方向的问题，也不是求值的问题。**
且修正两者后，该方法在诊断 cell 上**超过 448 baseline**。

旧 324-cell 运行的状态：

```
model_v3_implementation_frozen 3049e1c
  -> INVALIDATED BY FORENSIC AUDIT
```

**不是"成绩不好所以不跑了"**，而是已确认存在协议错误与实现定义错误，
即使跑完也不能作为最终 confirmatory result。它的用途改为 failure analysis /
forensic evidence。

---

## 1. 最终定性表

| 项目 | 最终定性 | 证据 |
|---|---|---|
| aggregation / evaluator | **已排除** | Stage 1 独立复算，三个 cell 全部 `MATCH`（差异 <1e-6） |
| selector | **已排除为 catastrophic collapse 的根因** | a5 (oracle) 比 a3 更差；位置、budget、bank 在四个版本中完全相同，唯一变量是特征来源与归一化 |
| crop-local z-score | **已确认造成真实损伤** | Stage 4：单独修正恢复 +34.03 AUPRO；Stage 4B 去掉 affine 后 +35.08，结论不变 |
| crop / global context loss | **已确认造成真实损伤** | Stage 4：单独修正恢复 +28.34 AUPRO；Stage 4B 去掉 affine 后 +28.62，结论不变 |
| calibration-domain mismatch | **客观存在，但非第三个实质因素** | Stage 4B：在该诊断 cell 的附加影响 ≤1.1 AUPRO；**不可外推到全部类别** |
| `SPLIT=0` 硬编码 | **独立 protocol bug**（与本次 collapse 无关） | 源码 + `06_split_audit.csv` |
| 仿射校准 domain mismatch | *（见下方修正条目）* | — |
| 672 高分辨率方向 | **未失败** | a2 = 87.95 > a0 = 85.19（同 cell） |
| selective refinement 整体思想 | **单 cell 因果探针中被救活**，尚未完成全数据正式验证 | C11 = 87.17 > A0 = 85.19 |

---

## 1b. Stage 4B：calibration 不是第三个实质因素（封口实验）

Stage 4 的四个版本**全部继续套用旧 frozen affine**（`a=0.97478, b=0.06219`），
因此 `C01−C00` 曾把"归一化口径变化"与"旧 affine 在新 domain 上的行为"混在一起。
Stage 4B 在同一代码路径上令 `a=1, b=0`（`RC_NO_AFFINE=1`），其余全部不变。

| 版本 | 带 affine (Stage 4) | 不带 affine (Stage 4B) | 差 |
|---|---|---|---|
| C00 | 27.24 | 26.17 | −1.07 |
| C10 | 55.58 | 54.80 | −0.78 |
| C01 | 61.27 | 61.26 | −0.01 |
| C11 | 87.17 | 87.16 | −0.01 |
| **context (C10−C00)** | **+28.34** | **+28.62** | +0.28 |
| **norm (C01−C00)** | **+34.03** | **+35.08** | +1.05 |
| **both (C11−C00)** | **+59.93** | **+60.98** | +1.05 |

Factorial interaction：带 affine `−2.44`，不带 `−2.72`——**近似可加的判断稳定**。

**结论**：两个核心效应在有无 affine 时几乎不动。且

```
C11_no_affine = 87.16 > A0 = 85.19
```

**即使完全拿掉旧 calibration，"修正 context + 修正 normalization 后恢复并超过 baseline"
这一结论仍然成立。**

**措辞限定（不得扩大）**：calibration-domain mismatch 是客观存在的定义不一致，
但在 `capsule/k1/s0` 这一个诊断 cell 中其附加影响不超过约 **1.1 AUPRO**。
**不能据此宣称它在所有类别上都可忽略。**

> `C01`/`C11` 几乎不受 affine 影响，是因为 A2 一致的全局 z 本就在 448 尺度上，
> affine≈恒等；`C00`/`C10` 的 crop-local z 处于另一 domain，故 affine 带来约 +1 AUPRO
> 的边角效应，但改变不了任何结论。

---

## 2. Stage 4 固定结果（`mvtec/capsule/k1/s0`）

固定不变：selector、positions、`m'`、`n`、budget、support bank、672→448
physical-overlap pooling、affine、writeback。唯一变量是两个正交因素。

| 版本 | 特征来源 | 归一化 | img_AUROC | px_AUROC | AUPRO |
|---|---|---|---|---|---|
| **C00** | crop-DINO | crop 局部 z（当前实现） | 64.22 | 83.72 | **27.24** |
| **C10** | full-672 同位置 | crop 局部 z | 61.59 | 82.46 | **55.58** |
| **C01** | crop-DINO | A2 一致的全局 z | 87.87 | 96.19 | **61.27** |
| **C11** | full-672 同位置 | A2 一致的全局 z | 82.81 | 98.32 | **87.17** |
| *参考* | — | — | — | — | *A0 = 85.19* |

### 效应分解（相对 A0 的总损失 L = 85.19 − 27.24 = **57.95**）

```
去掉 crop-context 问题   C10−C00 = +28.34   恢复 48.9%
去掉 crop-local norm     C01−C00 = +34.03   恢复 58.7%
两者都去掉               C11−C00 = +59.93   恢复 103.4%
                         且 C11 − A0 = +1.98  (超过 baseline)
```

Factorial interaction：

```
C11 − C10 − C01 + C00 = 87.17 − 55.58 − 61.27 + 27.24 = −2.44
```

**近似可加**（|−2.44| 相对 +59.93 约 4%）。两条损伤链都很大，
**没有任何一条单独就能解释全部崩塌**。

---

## 3. Stage 3 机制证据（三个 split-0 cell）

被细化图上 defect / normal 的**中位数间隔**：

```
a0   +4.349      <- 基线有清晰的异常/正常分离
a3   +1.026      <- V3 压到约 1/4
a5   +0.873      <- oracle 压得更狠

a3 − a0  gap 变化: mean −3.323
a5 − a0  gap 变化: mean −3.475
```

**oracle 比 selector 更差**是关键：若问题在"选哪里"，GT 应该救回来；它反而抹得更狠，
说明破坏来自**选完之后对那个区域做了什么**。

### 剂量-反应关系

```
bottle  k1/s0   几乎不触发（触发率 0.036%）  a3−a0 =  −0.91   基本无伤
cable   k8/s0   中等                          a3−a0 = −43.40
capsule k1/s0   大量触发                      a3−a0 = −57.94
```

破坏程度与该 cell 的细化量严格对应。

### Stage 1：求值无错

```
capsule/k1/s0   recheck vs CSV: MATCH
cable/k8/s0     recheck vs CSV: MATCH
bottle/k1/s0    recheck vs CSV: MATCH
```

---

## 4. Split 协议违例（独立问题）

源码事实：

```python
v3_check_affine_real.py:42   SPLIT = 0                       # 模块级常量
v3_check_affine_real.py:63   def run_case(obj, shot):        # 无 split 参数
v3_check_affine_real.py:70   idx = draw_images(..., shot, SPLIT, obj)
v3_cellrunner.py:187         fit = aff.run_case(obj, shot)["fit"]
```

`06_split_audit.csv` 的实测（4 个对象 × 4 个 shot，共 16 组）：

```
mvtec/bottle  k=1  s0 [26,140]  s1 [43,183]  s2 [153,167]     s1≠s0, s2≠s0
mvtec/cable   k=8  s0 [47,117,...]  s1 [36,52,...]  s2 [19,60,...]
每组的 s1、s2 support 都与 s0 不同
```

**结论：全部 split≠0 的 cell，其 affine `a, b` 是用 split 0 的 support 拟合的**，
而那些图不在 `S_k` 内。违反补充 2 A2.5 与 §2.2.2 的 support-only 原则。

**影响范围：216 / 324 cells（2/3）。**

**但它不是本次 catastrophic collapse 的原因** —— 三个诊断 cell 全是 split=0，
其 affine 恰好是正确的。

### 另一处独立缺陷：calibration-domain mismatch

```
拟合域   support 图整体归一化的 672 -> 448
应用域   crop 局部归一化的 672
```

`a, b` 只能改 scale/offset，无法恢复被 crop-local z-score 抹掉的绝对异常水平。
**这属于 score-definition 的实现不匹配，与 `SPLIT=0` 是两个独立问题。**

---

## 5. 必须写明的一条限定

```
C11 (87.17) > A0 (85.19)  不证明 G3 selector claim 已成立
```

它证明的是：

> 用 A3 **已选定的位置**，在修正 score/context 之后，这个 cell 的 selective refinement 能超过 baseline。

它**没有**比较修正后的 `A3_corrected` vs `A4_corrected`。
因此"selector 优于同预算随机"**仍然必须等新的 324-cell 的 G3**。

**`C11 = 87.17` 是诊断上界，不是最终模型** —— 它直接使用 full-672 全图特征；
若最终方法也必须先跑整张 672，G2 的效率优势就不存在了。

---

## 6. 三类问题的定性区分（不得混称）

```
SPLIT support hard-code          = implementation / protocol bug
crop-local z-score               = score-definition implementation mismatch
crop-context loss                = local-refinement design flaw
```

第三项不是"写错变量"那类 bug：若 V3 原本就设计成把局部 crop 单独送进 ViT，
那么丢失全局 attention 是这种设计产生的**结构性缺陷**。

---

## 7. 未做的事（明确记录）

```
Stage 5  crop-DINO ↔ full-672 token 的定量相似度（按 context size 分组）  —— 已完成（Stage 5B，见 §9）
Stage 2  writeback identity test                                          —— 部分（02_writeback_audit.csv）
Stage 6  C11 仍差于 A0 时的 affine 前后对比                                —— 未触发（C11 > A0）
```

Stage 5 的目的不是"证明 context 有问题"（Stage 4 已证明，值 +28.34 AUPRO），
而是量化**"需要多少上下文才能让局部 token 接近 full-image token"** ——
这决定 V3.1 的 halo 规则，**且 defect/test 数据不得用于挑选 halo 大小**。

---

## 8. 证据文件

```
01_metric_recheck.csv      独立复算 vs aggregation CSV（3 cells × 14 arms）
02_writeback_audit.csv     A3 改动点是否落在 covered set 内
03_score_chain.csv         162 张细化图的 defect/normal 分数链
04_stage4_2x2.csv          2×2 因果实验结果
06_split_audit.csv         split support 对照表（16 组）
```

旧 324-cell 的产物（`results/model_v0/v3/`，324/324 cells）与 aggregation 的部分结果
（`v3_agg/`，183/324 cells）**全部保留不删**，作为 forensic evidence。

---

## 9. Stage 5B：纯 halo 机制在 50% 预算内不成立

> 本节为**追加证据**。§0–§6 的根因结论一字未改，仍以 2×2 因果实验为准。
> 本节不使用任何 defect 性能指标（无 AUPRO / AUROC / GT / A5）。

**问题**：不用整张 672 前向时，局部 crop 的中心 5×5 token 要带多大的 halo，
才能与 full-672 在同物理位置上的 token 对齐？覆盖 27 类、324 cell。

**Positive control 先过**：全图 re-forward 逐位复现 `cache_m1_dino672`
（4 对象 cos=1.000000，maxdiff=0.00000），排除了预处理/缓存不一致这一整类替代解释。

### 9.1 保真度与预算是同时恶化的

| n_ctx | cos_median | nn_spearman | K_max | 触发图被 cap | 保留的触发 |
|---|---|---|---|---|---|
| 5 | 0.399 | 0.132 | 46 | 2.4% | 92.1% |
| 15 | 0.686 | 0.169 | 5 | 42.8% | 53.5% |
| 25 | 0.752 | 0.208 | 1 | 77.1% | 20.6% |
| 33 | 0.831 | **0.293** | **1** | **81.5%** | **15.9%** |

- 曲线在 33 处**仍在上升、未饱和** —— 33 不是"够大"，只是扫描区间的上界。
- **`final` 层 cos 已达 0.921 时 nn_spearman 只有 0.454**：余弦相似 ≠ 近邻序不变，
  而探测器消费的正是 1-NN 距离。只看 cosine 会严重高估 halo 的可用性。
- 存在**负相关**对象：`macaroni1` −0.282、`capsule` −0.178 —— halo 的近邻序与
  full-672 反向，比不提供上下文更差。

**不存在一个 n_ctx 同时满足"保真度可用"与"50% 预算装得下"。**

### 9.2 定性（措辞不得扩大）

被证伪的是**具体机制**——用局部 halo 重建全局上下文——而**不是**
"selective refinement 这个方向"。§2 的 `context (C10−C00) = +28.34 AUPRO` 恰恰说明
"把上下文补回来"本身是对的，只是不能靠放大裁剪窗口来补。

本节**不涉及 selector**，因此 §5 的限定依然成立：G3 仍需新的 324-cell 判定。

### 9.3 顺带确认的实现缺陷

Stage 5 的结论方向正确，但证据链当时无效（self-match 使 `nn_rel_error` 爆到 10⁵，
Spearman 是对近似常向量算的）。此外预算表的 `B` 曾由 `ceil(1.5 × 448 grid)` 推出，
而 448 缓存无 `grids` 键、VisA 的 672 grid 第二维不是 3 的倍数，
**12 个 VisA 对象的 `B` 被低估最多 33%**。两者均已在 Stage 5B 修正。

证据：`stage5b_package.zip` / `README_STAGE5B.md`。

---

## 10. Stage 5C：封口实验 —— pure-halo 路线关闭

> 追加证据。§0–§6 的根因结论一字未改。本节不含任何 defect 性能指标。

Stage 5B 的结论方向正确，但有两处实验设计未收紧：(a) crop 起点被裁剪，导致**被比较的
5×5 core 随 n 漂移**（48 行网格上 n=5 顶行 11 → n=33 顶行 14，漂移恰好落在曲线仍在上升
的一段）；(b) positive control 只验证了 `final` 一层，而 `mid`/`midlate` 恰是受质疑的层。
Stage 5C 固定 core（`cr ∈ [16, gh6-17]`，任何 n 都不裁剪）并对三层全部做 positive control
（12/12 逐位复现）。同时显式报告 detector-consistent 的 `agg3` 保真度。

### 10.1 预注册判据

存在 `n ≤ 33` 使 `median(agg3 Spearman) ≥ 0.50` 且 `trigger-mass retention ≥ 0.50`，
否则停止 pure-halo 路线。判据在运行前固定。

### 10.2 结果

```
n_ctx   agg3_pooled   retention(trigger-mass)
   5       0.133            0.921
  15       0.275            0.535
  21       0.364            0.314
  27       0.428            0.192
  33       0.467            0.159
```

**没有任何 n 同时满足两条。** 稳健性：`agg3` 的 5 种聚合口径（pooled + 4 个 anchor 序号）
× retention 的 2 种读法 = **10/10 全部 NONE**。最接近的 n 也差很远（trigger-mass 读法下
n=13 合计缺口 0.212）。`agg3` 单独越线最早出现在 n=29，而那时 retention 已跌到 0.182。

### 10.3 对 §9 的两处修正

1. **固定 core 后 fidelity 显著上升**：n=33 的 `agg3` 从 0.300 → 0.467，
   cos 0.831 → 0.903。同一段 agg3 代码跑 5B 向量得 0.300，证明差异来自数据而非方法。
2. **§9 的层级结论撤回**：5B 显示 `mid`/`midlate` 的 ρ 远低于 `final`（0.191/0.247 vs 0.454），
   据此推测"中层 global-context 依赖特别强"。Stage 5C 固定 core 后为 0.443/0.379 vs 0.491，
   **层间差异大幅收窄，该推测不成立。**

### 10.4 未分离的因素（明确记录）

Stage 5C 同时改变了 (a) core 不再随 n 漂移、(b) core 位置不同且最大 crop 不再贴图像边缘。
**二者未被分离。** 最小分离实验是把 core 中心取在下界 `cr = 16`（此时 n=33 的 crop 恰好贴边），
**未运行**。

### 10.5 定性

```
pure-halo（用局部 halo 重建全局上下文）  ->  关闭
selective refinement 方向本身            ->  未关闭（§2 的 +28.34 仍成立）
```

证据：`README_STAGE5C.md`、`05c_halo_decision.csv`、`05c_context_fidelity_fixed_core.csv`。

---

## 11. Stage 5D-A：adaptive halo 也不成立 —— pure-halo family 关闭

> 追加证据。§0–§6 一字未改。全离线：不跑 DINO、不用 GPU、不读缺陷标签。

Stage 5C 只关闭了**固定** n。冻结的 V3 用的是 addendum 1 的**按图分配**
（`m' = min(m, floor(B/25))`，`n` = `[5,33]` 内最大奇数且 `m'·n² ≤ B`），
其 Pareto 曲线与 fixed-n 不同，因此必须单独裁决。

审计读取 324/324 cells、46644 张图的冻结 `diagnostics.json`，**逐张复现 `(m2, n)` 并与冻结
记录比对（0 处不符，arm 间也 0 处不符）**。

### 11.1 结果

| 量 | mvtec | visa | ALL |
|---|---|---|---|
| `P_mass(n ≤ 15)` | 0.942 | 0.859 | **0.893** |
| `P_mass(n ≥ 23)` | 0.031 | 0.055 | **0.045** |
| trigger-mass retention | 0.849 | 0.965 | **0.917** |
| weighted fidelity proxy（中位） | 0.207 | 0.135 | **0.184** |

**自适应保住了 retention（0.917），却几乎没换来 context。** 原因是结构性的：
`m'·n² ≤ B` 把 n 与 m 反向绑定 —— **n=5 的图平均 m=52.4（占 mass 33.8%），
n=33 的图平均 m=1.0（仅占 1.4%）**。自适应把最大的 halo 给了证据最少的图。

### 11.2 与 fixed-n 的对照

```
adaptive        0.184   retention 0.917
fixed n=15      0.275   retention 0.535
fixed n=33      0.467   retention 0.159
```

**自适应比固定 n=15 还差**：用 retention 0.917 vs 0.535 的优势，换来 fidelity 从
0.275 掉到 0.184 —— 而 0.275 本身已不可用。

**optimistic proxy upper envelope**（每个 n 同时取最有利的 agg3 convention *与* 27 类中
最好的类别，再按 mass 加权）为 **0.445**，仍低于 0.50。**注意措辞：这是"上包络"
（upper envelope），不是"上界"（bound）** —— 它本身仍是代理量，只是把两处乐观假设叠满，
因此**不能**被引用为数学上界，也不能用来宣称"proxy 不可能更高"。

### 11.3 顺带发现的冻结实现缺陷

`v3_run.py:100` 用 `ceil(1.5 × 448 grid)` **推导** 672 grid 而非读真实 grid。
resize 取整导致**双向**偏离：`candle` B 高估 1.9%，`cashew` 低估 1.9%。
后果：**111 / 46644 张图（0.24%）的 `m'·n²` 超出真实 50% 预算**，最大约 **1.9%**。
与 §4 的 `SPLIT=0` 同类：**implementation bug，非方法问题**。

```
本 bug 不改变 §11.4 的 halo 结论 —— 受影响的 111 张图占 0.24%，
既不足以把 P_mass(n≤15) 从 0.893 拉下来，也不足以把 proxy 从 0.184 推上去。
```

**V3.1 必须直接读取真实 672 grid**（从 672 cache 的 `grids` 读，不得由 448 grid 推导）。
该缺陷此前已在 Stage 5B 的预算辅助代码中被发现并修掉，属 "derive instead of read" 家族；
证据：`05d_budget_derivation_audit.csv`。

### 11.4 裁决

按运行前固定的分岔（`P_mass(n≤15)` 占多数且 proxy 明显低于 0.40 → 关闭）：
**pure-halo family（固定 n 与自适应 n）正式关闭。** Stage 5D-B（真实 mixed-n fidelity
测量）按该分岔不跑 —— 它只在 proxy 处于 0.40–0.50 灰区以上才值得花 GPU。

**边界**：被关闭的是 **halo 机制**；**selective refinement 方向仍未关闭**（§2 的
`+28.34 AUPRO` 仍成立）。本结论为**筛选级**证据，唯一能推翻它的是被跳过的 5D-B 实测量。

证据：`README_STAGE5D.md`、`stage5d_adaptive_audit.zip`。
