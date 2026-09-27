# Gate 17A 结果总账（冻结）：covariance 分支关闭

> 冻结于 tag `gate17a_stop`。
> **判决：STOP。covariance 对下游异常检测没有任何可利用的增量信息。**
> **这是本分支的最后一刀，不再有 Gate 17B / 18。**
>
> ⚠️ **本文档的 pixel 两列已在 `PHASE_R_BASELINE_AUDIT.md` 中修正。**
> 初版报出的 `px_AUROC 81.47 / AUPRO 14.39` 来自本脚本的一个 reshape bug
> （把已展平的 `gt` 的 `shape[1:]` 当网格用 → 存成一维 map →
> `cv2.resize` 把它当列向量处理 → 空间结构被打乱）。
> **修正后为 `px_AUROC 96.27 / AUPRO 73.45`，oracle gain 分别只有 +0.05 / +0.02。**
> **image 级数字与 STOP 判决不受影响**（`mean_top1p` 只依赖取值分布，与 shape 无关）。

---

## 最终结论（严谨表述）

> **用 target label 做 oracle 融合（$S_\alpha=(1-\alpha)z(S_{raw})+\alpha z(S_{cov})$，
> 逐 object 取最优 α），image AUROC 平均只提升 **+0.19**（预注册门槛是 +1.0）；
> pixel AUROC **+0.05**、AUPRO **+0.02**（修正后；见文首说明）；
> image 层面 **131/180**、pixel 层面 24/30、AUPRO 27/30 个格子的最优 α 为 0。**
> **即使允许用真实异常标签选择融合权重，covariance 也没有带来可用信息。**

---

## 1. 结果

完整扫描：15 objects × 4 shots（1/2/4/8）× 3 splits × 11 个 α。

| metric | mean raw | mean best-α | gain | 最优 α = 0 的格子 |
|---|---|---|---|---|
| **img_AUROC** | 95.65 | 95.84 | **+0.19** | **131/180** |
| px_AUROC（已修正） | **96.27** | 96.32 | +0.05 | 24/30 |
| AUPRO（已修正） | **73.45** | 73.47 | **+0.02** | 27/30 |

逐 object 的 image AUROC 增益：

```
bottle +0.1  cable +0.5  capsule +0.3  carpet +0.0  grid +0.0
hazelnut +0.0  leather +0.0  metal_nut +0.0  pill +0.4  screw +0.0
tile +0.0  toothbrush +0.0  transistor +1.3  wood +0.2  zipper +0.0
```

**9/15 个 object 的增益四舍五入后是 0.0**，最大也只有 transistor 的 +1.3。
（toothbrush 只有单一缺陷类型，那些 +3.2 之类的像素增益是退化的。）

**修正后的 pixel 指标**：oracle 融合只值 +0.05 px_AUROC / +0.02 AUPRO，
27/30 个格子的最优 α 仍是 0.0 —— 加入任何 covariance 成分都不改善定位。

## 2. 预注册判决

```
STOP  若最佳 oracle fusion 相对 raw < +0.5，或改善 < 10/15 objects
GO    若 ≥ +1.0 且 ≥ 10/15 objects
```

**实测 +0.19（< +1.0）→ STOP。**

按预注册，STOP 意味着**以下全部不做**：

- tail calibration
- adaptive weighting
- density correction
- gating
- score-fusion neural network

**理由**：连用真实异常标签挑 α 的 oracle 都没有 headroom，
任何无标签方法都不可能凭空造出 headroom。

## 3. 与 Model V1 失败的关系

两者是一致的，且互相印证：

| 问题 | 答案 |
|---|---|
| covariance 替代 raw score？ | ❌ −6.45 AUROC / −31 AUPRO |
| covariance 作为 **补充** 信息？ | ❌ oracle 也没用（img +0.19 / px +0.05 / AUPRO +0.02） |
| covariance 编码了 defect subtype 几何？ | ✅ retrieval +1.9，shuffled 对照有效 |

**合起来是一个完整而罕见的结论：**

> **一个表示可以真实地编码「缺陷亚型几何」（有跨 object 一致性、
> 有 valid 的 shuffled 对照），同时对「正常 vs 异常」这个真正的任务
> 既不能替代、也不能补充。**

## 4. 机制回顾（已在 Model V1 中实测）

| | good 均值 | bad 均值 | gap |
|---|---|---|---|
| raw | 0.196 | 0.402 | **+0.206** |
| whitened | 0.609 | 0.725 | **+0.116** |

$\Sigma^{-1/2}$ 放大低方差方向；图像分数是 patch 距离的 **top-1% 尾部统计**。
放大后正常波动先落进尾部 → gap 压缩 44%。

**加上 shot 越多伤害越小（1-shot −5.4 → 8-shot −2.1）**，
指向的机制是：

> **few-shot 下协方差估计不稳 + 极端尾部评分 = 把「参考集中看起来稳定、
> 但在新 good 图上其实存在自然变化」的方向放大成假异常。**

不是「协方差理论上有害」，而是**这个组合在 few-shot + tail scoring 下极其脆弱**。

## 5. 项目当前状态（如实）

| 路线 | 结果 |
|---|---|
| handcrafted geometry / retrieval / selector / generator | ❌ |
| anomaly projection / shared residual / conditioned residual | ❌ |
| local covariance / multi-layer | ❌ |
| **target covariance whitening** | retrieval ✅ / 下游 ❌ / 补充信息 ❌ |

**没有任何一条经过下游验证的方法。**

唯一经下游验证的结论是负面的：
**在 frozen DINO 上，target-normal 二阶校正改善缺陷亚型检索，
但因放大正常尾部而破坏异常检测，且即使 oracle 融合也无增量。**

## 6. 必须永久冻结的方法纪律

1. **代理指标的响应方向必须与目标指标核对。**
   本次 retrieval +1.9 → AUROC −6.45 → AUPRO −31。
   **任何代理指标上的正结果，在真实指标之前不得表述为方法有效。**
2. **新想法必须先过 cheap downstream smoke test**（3 objects × 1/4-shot），
   再做机制分析。**顺序不能反。** 本次先跑了 5 个 representation gate 才做下游，
   是最大的流程错误。
3. **用 oracle 融合做「是否值得继续」的终审。**
   连 oracle 都没 headroom 的分支，不要做任何无标签补丁。
4. **尾部聚合对低方差方向放大极敏感**；任何线性校正都要先查 good/bad gap。
5. 以及此前的：零初始化在乘积结构中恒为鞍点、反事实必须同源同脚本、
   跨脚本数字必须重算、测试台必须先验证 headroom、
   matrix log 不能当 elementwise log、中间产物不得用于结论统计。

## 7. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `gate17a_fusion_oracle.py`、`model_v1_eval.py`、`gate16_multilayer.py` |
| 结果 | `metrics/gate17a_fusion.csv`、`model_v1_downstream.csv` |
| 中间产物 | `results/model_v0/maps_v1*/`（raw / V1_shot / V1_full / fused 的 anomaly maps） |

---

## 下一步的两条路（需决策）

**A. 分析型工作。**
现有材料足以支撑一个罕见而清晰的机制性故事：
「更好的缺陷表示为什么会伤害异常检测」——
含跨 15 object 的 retrieval/detection 反向证据、normal-tail 机制实测、
以及 oracle 融合无增量的终审。需要补：第二个数据集、不同 backbone、
不同 aggregation 的系统性相关性分析。

**B. 换优化目标重开。**
既然 raw AnomalyDINO 在 v1 上 image AUROC 已有 95.5–97.0，
问题可能本就不在表示层。应直接面向 **normal-tail-aware anomaly scoring** 设计，
并强制新想法先过 downstream smoke test。
