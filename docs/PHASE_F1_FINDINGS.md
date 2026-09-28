# Phase F1：Downstream Failure Characterization（共同失败机制审计）

> 冻结于 tag `phase_f1`。
> **核心结论：两个结构完全不同的 detector 对「局部特征对比度」表现出几乎重合的
> 单调依赖；而空间形态学（碎片化、形状）与失败基本无关。**
> ⚠️ **但预注册判据 A 存在一处符号错误（本文 §4 完整披露），
> 字面判决是 CASE C。实质结论与字面判决不一致，需明确记录而非就地改写。**

---

## 1. 问题

`tile / cable / transistor / toothbrush` 在
**multi-layer kNN memory bank** 与 **PCA normal-subspace residual**
两个结构完全不同的 detector 下都难。同 backbone、同分辨率、同 shots。

要回答的不是"哪个 detector 更好"，而是：

> **为什么它们同时失败？**

## 2. 测量设计

对**全部 15 类、全部 1258 张异常测试图**（× shots{1,4} × 3 splits = **7548 行**），
每行记录：

| 量 | 定义 |
|---|---|
| `area_frac` | \|GT\|/\|image\|，原图 mask 分辨率 |
| `n_cc`, `largest_cc_frac` | 连通域数量、最大连通域占缺陷面积比 |
| `perimeter2_over_area`, `elongation` | 周长²/面积、最大连通域最小外接矩形长宽比 |
| **`C_feat`** | **raw DINO 末层原始距离**：`median d(defect→bank) − median d(nearby normal→bank)`，nearby = 32×32 网格上切比雪夫距离 ≤3 的非 GT patch |
| `M` | `median(score in GT) − P95(score outside GT)`，与 AUPRO 同一条 evaluator 路径 |
| `TPR` | GT 内像素超过**该图自身** GT 外分数 P95 的比例 |

两个 detector 按**同一 (object, shot, split)** 配对：kNN `agg_all3`
与 SubspaceAD **aug=0**（与 kNN 的无增强一致）。

### 两个定义上的判断（必须记录）

1. **`TPR@FPR=0.05` 无法逐图严格定义**（FPR 需要负样本）。
   本处用"该图自身 GT 外 P95"作阈值。代价：**GT 占比大时阈值会被缺陷像素污染**。
   故逐行记录 `gt_covers_p95`；全量 456/7548 行 GT 超过 15%。
   **所有结论都做了排除这些行的稳健性复核（§3）**。
2. **`M` 为主判据，`TPR` 为其二值化版本**。两者结论一致（§3），无不一致项。

## 3. 结果

### 3.1 因子与低 FPR 定位能力（within-object，去掉"难对象"解释后）

| factor | detector | global rho | **within rho** | per-object 一致率 |
|---|---|---|---|---|
| **C_feat** | kNN | +0.615 | **+0.526** | **15/15** |
| **C_feat** | SubspaceAD | +0.525 | **+0.424** | 14/15 |
| area_frac | kNN | −0.460 | −0.362 | 15/15 |
| area_frac | SubspaceAD | −0.485 | −0.348 | 14/15 |
| n_cc | kNN | −0.037 | −0.015 | — |
| n_cc | SubspaceAD | −0.079 | −0.040 | — |
| largest_cc_frac | kNN | +0.031 | −0.012 | — |
| largest_cc_frac | SubspaceAD | +0.065 | +0.012 | — |
| perimeter2/area | kNN | −0.079 | −0.084 | — |
| perimeter2/area | SubspaceAD | −0.123 | −0.096 | — |

### 3.2 控制 object 与面积的回归（object fixed effects，within-object 去均值）

```
TPR ~ C_feat + log10(area_frac) + fragmentation + elongation
```

| detector | C_feat | log_area | fragmentation | elongation |
|---|---|---|---|---|
| kNN | **+0.6615 (t=+44.3)** | −0.0640 (t=−18.4) | +0.033 (t=+3.5) | −0.002 (t=−4.6) |
| SubspaceAD | **+0.5664 (t=+32.2)** | −0.0591 (t=−14.5) | +0.038 (t=+3.5) | −0.002 (t=−4.3) |

（fragmentation 用 `largest_cc_frac`；换 `n_cc` 或 `perimeter²/area` 结论不变。）

### 3.3 效应量：按 within-object C_feat 四分位

| C_feat 分位 | TPR kNN | TPR SA | M kNN | M SA |
|---|---|---|---|---|
| Q1（最弱对比） | 0.793 | 0.798 | 1.295 | 2.292 |
| Q2 | 0.892 | 0.910 | 1.827 | 3.539 |
| Q3 | 0.956 | 0.953 | 2.619 | 5.251 |
| Q4（最强对比） | **0.980** | **0.978** | 3.391 | 7.844 |

**两个 detector 的曲线几乎重合**，且四段全部单调。

### 3.4 稳健性

| 检验 | kNN | SubspaceAD |
|---|---|---|
| within rho(C_feat, M) | **+0.715** | **+0.705** |
| 排除 GT>15% 的行后 rho(C_feat, TPR) | +0.492 | +0.379 |
| 排除后 rho(area, TPR) | −0.303 | −0.278 |
| 排除后 rho(n_cc, TPR) | −0.032 | −0.058 |
| 排除后 rho(largest_cc_frac, TPR) | +0.008 | +0.037 |

### 3.5 统计修正（事后补充，**不是判据变更**）

§3.2 的 t 值是**朴素标准误**，把 7548 行当作独立观测——但每 100 张图共享一个
category、一个 k-shot bank 和一片场景，这会产生**伪精度**。
按 category 聚类（G=15）重算：

| 系数 | 朴素 t（§3.2） | **clustered SE 的 t** | p | 95% CI |
|---|---|---|---|---|
| C_feat（kNN） | +44.3 | **+4.77** | 3.0e-4 | [+0.390, +0.933] |
| C_feat（SA） | +32.2 | **+3.64** | 2.7e-3 | [+0.262, +0.871] |
| log_area（kNN） | −18.4 | −3.97 | 1.4e-3 | [−0.096, −0.032] |
| log_area（SA） | −14.5 | −2.68 | 1.8e-2 | [−0.102, −0.016] |
| largest_cc_frac（kNN） | +3.5 | **+1.01（不显著）** | 0.33 | [−0.030, +0.094] |
| n_cc（kNN） | −2.3 | **−0.08（不显著）** | 0.94 | [−0.005, +0.004] |

**结论：朴素 t 被高估约 9 倍；但 C_feat 效应存活（t≈+4.8 / +3.6，CI 明显不含 0），
而形态学在正确标准误下进一步确认为零。**
`C_feat` 的 β 本身不变（+0.66 / +0.57），改变的只有不确定性。

## 4. 预注册判据与符号错误（完整披露）

运行前冻结：

```
A  within rho(C_feat, TPR) <= -0.4  两个 detector 都成立、过 object FE
B  within rho(area, TPR) >= +0.4 或 rho(frag, TPR) <= -0.4，同上
C  都不成立 -> 停止造模型
```

字面判决：

```
A  kNN +0.526, SA +0.424   -> FAILS（符号与判据相反）
B  FAILS（area 为负、fragmentation 近零）
-> CASE C
```

**A 的失败源于我自己的符号错误，且与同一文件三行之上的假设自相矛盾：**

- `C_feat` 定义为 `d(defect) − d(nearby normal)`，
  越大 = 缺陷离局部正常越远 = **越容易**；
- 因此它应与 TPR **正相关**，判据却写成 `<= −0.4`。

**这个错误不是"结果不好所以改门槛"，而是判据与它自己要检验的假设不一致。**
但仍必须记录为**判据缺陷**，不得就地宣布通过。

**B 是被真正否定的**，与符号无关：

- `n_cc` / `largest_cc_frac` 的近零相关在两个 detector、两种统计量下都成立；
- `area_frac` 虽显著，但符号为**负** —— 不是"小缺陷更难"，
  而是在固定图内 FPR 下，**大 GT 的逐像素命中率更低**（大缺陷边缘占比高），
  与 Case B 的叙事方向相反。

### 结论的两种读法（并列记录，不合并）

| 读法 | 内容 |
|---|---|
| **字面** | **CASE C**：预注册判据未被满足。此结论**保留在案，不被覆盖**。 |
| **实质** | 数据对 **Case A** 提供很强支持：跨两个 detector 一致、控制 object 与面积后仍显著（t=+44/+32）、四分位单调、稳健性复核通过。 |

### 本数据集的正式地位（冻结表述）

> **preregistered inequality contained a sign error; therefore this dataset is
> treated as hypothesis-generating / supportive rather than an independent
> confirmatory validation.**

理由，同时也是不能就地改判的根据：

1. 判据的不等式与同一文档三行之上的变量定义**直接矛盾**，属**判据缺陷**，
   不是结果不理想后的放宽；
2. 但该错误是在**看到结果之后**才发现的。**无论判据怎么写，
   同一批 MVTec 数据都不可能再构成独立 confirmatory test。**
3. 因此：**F1 = exploratory-confirmatory evidence。** 它强支持机制，
   但**不计为无瑕疵的 confirmatory test**。
   独立确认必须在**从未看过的第二数据集**上、用**重新冻结的正确判据**完成（Phase F2）。

历史保留：`phase_f1` 记录中的 CASE C 判决、原始判据文本、以及本节披露，
**全部保留，不删除、不覆盖**。

## 5. 对下一步模型问题的意义

若采纳实质读法：

> 两个 detector 共同失败的主因是
> **缺陷在特征空间里离其局部正常上下文太近**，
> 而不是空间形态（碎片化 / 形状），也不是尺度。

对应的问题就是：

$$\boxed{\text{Low-contrast contextual anomaly localization}}$$

新模型应做**局部上下文 / 关系建模**（比较缺陷与其**邻域**的正常结构），
而不是继续改距离度量、layer 权重或 normal bank。

若采纳字面读法，则按 CASE C 处理：**数据不足以识别机制，不建新模型。**

## 6. 限制

- `TPR` 的逐图阈值定义在 GT 占比大时受污染（456/7548 行 GT>15%），
  已做排除复核；`M` 不受此影响且给出同向更强结论。
- kNN 与 SubspaceAD 的 map 量纲不同（z-score 距离 vs 重建残差），
  但 `M` / `TPR` 都是图内相对量，且结论在两个 detector 上独立复现。
- 相关性证据，非因果。`C_feat` 与 `area_frac` 存在共变（回归中已同时纳入）。

## 7. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `phase_f1_failure.py`（逐图测量）、`phase_f1_analysis.py`（统计与判决） |
| 结果 | `metrics/phase_f1_failure.csv`（7548 行 × 1258 张缺陷图） |
| 日志 | `shards/f1_{1..4}.log` |
