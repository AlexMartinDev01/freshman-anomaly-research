# Model V2 smoke：context-conditioned normal matching（NO-GO）

> 冻结于 tag `model_v2_context_smoke`。
> **判决：三条预注册判据全部失败。停止，不调参。**

---

## 1. 动机

`PHASE_F2_FINDINGS.md` 在 MVTec 与 VisA 上、用事先冻结的判据确认：

> 两个结构完全不同的 detector 的低 FPR 定位能力，
> 由**缺陷与其局部正常结构的特征可分性** `C_feat` 支配。

自然的模型回应是：不再问"这个 patch 离**任意**正常 patch 多远"，
而是问"离**处于相似局部上下文**的正常 patch 多远"，即建模
$p(z\mid\text{context})$ 而非 $p(z)$：

$$s_i=\min_{j\in\mathcal N(c_i)} d(z_i,z_j)$$

- $c_i$ = 以 patch $i$ 为中心的 3×3 邻域（去掉中心）特征均值
- $\mathcal N(c_i)$ = context 与 $c_i$ 最接近的 K 个 bank patch
- 其余全部与 `agg_all3` 相同：同特征、同 bank、同 draws、同 z-score 后平均、同 evaluator

**无训练、无异常标签。**

## 2. 预注册判据（跑之前冻结）

对象按 **F1 实测 `C_feat`** 选取，不是凭直觉：

- 困难组（最低 4）：`transistor 0.087`、`screw 0.111`、`zipper 0.154`、`cable 0.165`
- 对照组（最高 2）：`grid 0.312`、`carpet 0.388`

```
M1  4 个困难对象 AUPRO 平均 >= +2
M2  其中任一对象下降不超过 3
M3  整体 image AUROC 不比 agg_all3 差 0.3 以上
K=64 为唯一判决配置；K∈{16,256} 仅作敏感性表，不参与判决
```

## 3. 结果

```
hard-object AUPRO gain (K=64): cable=-1.02  screw=-0.69  transistor=-2.14  zipper=-3.54
M1 mean gain -1.85 (need >= +2.0)      -> FAIL
M2 worst object -3.54 (need >= -3.0)   -> FAIL
M3 mean image AUROC -1.90 (need >= -0.3) -> FAIL
-> NO-GO: stop
```

| object | AUPRO `agg_all3` | AUPRO `ctx_K64` | Δ | img Δ | px Δ |
|---|---|---|---|---|---|
| transistor | 57.78 | 55.64 | **−2.14** | −2.51 | −0.29 |
| zipper | 67.82 | 64.28 | **−3.54** | −7.39 | −0.31 |
| cable | 72.33 | 71.31 | **−1.02** | −1.76 | −0.07 |
| screw | 52.12 | 51.43 | **−0.69** | +0.29 | −0.21 |
| grid（对照） | 87.88 | 87.43 | −0.45 | −0.01 | −0.03 |
| carpet（对照） | 93.66 | 93.63 | −0.03 | −0.01 | −0.01 |

**每一个困难对象都是负的；对照组接近零。**
K 的敏感性方向一致（K=16 更差，K=256 与 K=64 几乎相同）——
**不是 K 选得不好，是这个做法本身没有信息可加。**

## 4. 机制解释（标为假设）

DINOv2 特征在空间上是**平滑**的：一个 patch 的邻域均值几乎可由它自身特征预测。
因此 $c_i$ 相对 $z_i$ 的**新增信息极少**，用 context 去筛 bank 近似等价于不筛。

而限制 bank 到 $\mathcal N(c_i)$ 只会把 $s_i=\min_{j\in\mathcal N}d$ **抬高**
（子集的最小值 ≥ 全集的最小值），却对"哪些 patch 被抬高得更多"没有带来正确的排序
——于是 z-score 之后总体是**噪声**，并因移除了原本有效的近邻而**净变差**。

> 若该解释成立：要让"上下文条件化"有效，必须先有一个**上下文信息不被中心特征吸收**
> 的表示。这是一个表示层面的问题，不是邻域半径或描述子的超参问题。

## 5. 我没有做什么（重要）

**没有**去调：邻域半径（3×3→5×5）、context 描述子（均值→拼接）、K 值、
或换回逐层加权的其它形式。那正是 `PHASE_R3_AUG_AND_RELIABILITY.md` 里
被关闭的两条线（augmentation predictor、layer reliability）的失败模式：
**在同一个机制上反复换参数直到好看。**

判据是预注册的，失败了就是失败了。

## 6. 对项目状态的影响

| | 状态 |
|---|---|
| **机制**（低特征可分性是两个 detector 共同失败的主因） | ✅ **MVTec + VisA 双数据集、预注册判据确认**（F1 为生成性，F2 为确证性） |
| **第一个模型回应**（context-conditioned matching） | ❌ NO-GO |

这不是矛盾：**诊断成立不等于某个具体处方成立。**
F2 告诉我们"困难在哪里"，Model V2 告诉我们"把上下文塞进最近邻搜索不足以解决它"。

**下一步要基于这个否定结果重新想，而不是在它上面继续调。**

## 7. 过程中发现的存储缺陷（不影响本次指标）

`evaluate()` 原本把 map 写到 `maps_model_v2_ctx/<config>/<obj>/`，
**路径不含 (shot, split)**，因此每个 cell 覆盖上一个 cell 的 map，
磁盘上只留下最后一个 cell。

- **本次所有指标不受影响**：`pixel_metrics_binned` 在同一个 `evaluate()` 调用内、
  写完立刻算完，覆盖发生在之后；且分片处理的对象互不重叠。
- **已修复**：路径改为 `<config>/<obj>/<shot>shot_s<split>/`。
- 记录原因：这是本项目第二次被"静默产物丢失/覆盖"咬到
  （第一次是 SubspaceAD dump 的类别名冲突），**同类问题必须留档**。

## 8. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `model_v2_context.py`、`model_v2_merge.py` |
| 结果 | `metrics/model_v2_ctx_s{1..6}.csv`、`metrics/model_v2_context_smoke_merged.csv` |
| 日志 | `shards/v2_{1..6}_*.log` |
| 冻结判据 | 见本文 §2 与 `model_v2_context.py` 的 docstring |
