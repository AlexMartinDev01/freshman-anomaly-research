# Model V1 终局总账：representation 收口 + 下游判决

> 冻结于 tag `model_v1_closed`。
> **Gate 16：multi-layer 未通过，保留 final layer（预注册判决）。**
> **下游判决：Model V1 在真正的 anomaly detection 上失败，且差距巨大。**
> **retrieval 的 +1.9 完全没有转化，符号翻转。**

---

## 最终结论（严谨表述）

> **十六轮 gate 唯一稳定的正结果 ——「target normal 全局协方差 + 分数阶白化
> 使缺陷类型 retrieval 从 84.7 提升到 86.6，且有 valid 的 shuffled 对照
> （own − shuffled = +1.94, 11/15）」—— 在真正的 anomaly detection 上
> **不仅没有转化，反而显著变差**：
> image AUROC **−6.45**（180 个格子里只有 8 个为正）、
> pixel AUROC **−5.1 ~ −7.9**、**AUPRO −30 ~ −35**。**
>
> **因此：本项目目前没有任何经过下游验证的方法。**
> **并且必须记录：retrieval Recall@1 是一个**误导性的代理指标**。**

---

## 1. Gate 16：最后一个 representation gate（预注册有界范围）

三个层级（block 5 / 8 / 11）× 两种接口（raw / Σ^{-0.5}）+ 等权融合：

| config | mean Recall@1 | vs final_cov |
|---|---|---|
| **final_cov** | **86.6** | — |
| final_raw | 84.7 | −1.87 |
| fuse_cov | 83.5 | −3.11 |
| midlate_cov | 83.2 | −3.42 |
| mid_cov | 80.3 | −6.31 |
| midlate_raw | 76.1 | −10.44 |
| mid_raw | 71.7 | −14.89 |

预注册判据（≥+1.0 且 ≥10/15）**未通过** → **KEEP final layer only。**

**final layer 明显最好；中层特征更差，融合更差。**

### 修正一个我自己写错的对照

Gate 16 初次实现里的 `final_cov_shuffled` 是**对白化矩阵按行置换**——
只有一个簇时这不是「用别的 object 的协方差」，是无效对照（给出 +0.00）。
按 Gate 12 的口径重做（用**另一个 object 的 Σ**，同一协议）：

| | mean |
|---|---|
| own covariance | **86.6** |
| shuffled covariance（其余 14 个 object 的均值） | 84.6 |
| **own − shuffled** | **+1.94（11/15）** |

与 Gate 12 的 +2.6（12/15）一致。**「target-specific 协方差」这件事本身是真的。**

## 2. Model V1 冻结

```
Frozen DINOv2 ViT-S/14 (final layer, 448px)
  → target normal patches
  → 全局协方差 Σ_t（eigenvalue floor）
  → M_t = (Σ_t + εI)^{-0.5}, γ = 0.5
  → 1-NN anomaly scoring
```
无训练、无 target 异常标签、无 source 异常数据。

## 3. 下游判决（15 objects × 4 shots × 3 splits × 3 configs）

| metric | shot | raw | V1_shot | V1_full | gain (shot/full) |
|---|---|---|---|---|---|
| **img_AUROC** | 1 | 95.5 | 88.2 | 87.6 | **−5.9 / −11.4** |
| | 2 | 96.0 | 89.9 | 90.3 | −4.2 / −6.1 |
| | 4 | 96.7 | 90.5 | 91.2 | −4.7 / −5.4 |
| | 8 | 97.0 | 91.9 | 91.8 | −3.0 / −3.0 |
| **px_AUROC** | 1 | — | — | — | −7.2 / −7.9 |
| | 8 | — | — | — | −5.1 / −5.2 |
| **AUPRO@0.05** | 1 | 72.0 | 41.8 | 36.9 | **−30.3 / −35.1** |
| | 8 | 75.3 | 43.6 | 43.9 | −31.7 / −31.4 |

**V1_full 相对 raw 在 180 个 (object, shot, split) 格子里只有 8 个为正，
平均 image AUROC −6.45。**

**AUPRO 是灾难性的：−30 ~ −35。**（AUPRO 是像素级定位指标，
说明白化不仅破坏图像级排序，更严重破坏缺陷定位。）

**shot 越多差距越小（−11.4 → −3.0）**，即白化的伤害在正常参考越少时越大。

## 4. 机制：白化放大的是正常尾部

用保存的 anomaly map 重算 4-shot 下的图像分数分布：

| config | good 均值 | good p95 | bad 均值 | **gap** |
|---|---|---|---|---|
| raw | 0.196 | 0.308 | 0.402 | **+0.206** |
| V1_full | 0.609 | 0.744 | 0.725 | **+0.116** |

> **白化把正常图像的分数抬高了 3.1×（0.196→0.609），
> 而缺陷只抬高 1.8×（0.402→0.725）。
> good–bad 间隔被压缩 44%。**

原因直接：白化按 Σ^{-1/2} 放大**低方差方向**，
而图像分数取**top-1% patch 距离的均值**——
低方差方向上的正常波动被放大后，正好落在这个尾部统计里。

**这正是本项目最初的 Phase 3 诊断（normal tail 主导）以新的形式再次出现。**

## 5. 最重要的方法论教训

> **retrieval Recall@1（同缺陷类型最近邻）是一个误导性的代理指标。**

它在本次中给出 **+1.9** 的正信号，有 valid 的 shuffled 对照支持，
单调、可复现、跨 15 个 object ——**然后下游 AUROC 是 −6.45，AUPRO 是 −31。**

两者对白化的响应方向**相反**：
- retrieval 在**缺陷 patch 之间**排序 → 放大低方差方向有助于区分缺陷亚型；
- detection 在**缺陷 vs 正常 bank**之间排序，且用**尾部统计** →
  放大低方差方向首先抬高的是正常尾部。

**这是本项目最贵的一课，应写进方法纪律最前面。**

## 6. 项目当前状态（如实）

| 路线 | 结果 |
|---|---|
| handcrafted geometry | ❌ |
| geometry retrieval | ❌ |
| learned selector | ❌ |
| conditional generator | ❌ |
| anomaly projection (binary / structured) | ❌ |
| shared residual | ❌ |
| covariance-conditioned residual | ❌ |
| local covariance mixture | ❌ |
| multi-layer statistics | ❌ |
| **target covariance whitening** | **retrieval ✅ / downstream ❌** |

**没有任何一条经过下游验证的方法。**

唯一留下的、**经过下游验证的**事实是一个负面的机制性结论：

> **在 frozen DINO 特征上，target normal 的全局二阶校正能改善缺陷类型检索，
> 但会通过放大正常尾部而破坏异常检测。**

## 7. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `cache_multilayer.py`、`gate16_multilayer.py`、`model_v1_eval.py` |
| 结果 | `metrics/gate16_multilayer.csv`、`model_v1_downstream.csv` |
| 缓存 | `results/model_v0/cache_ml/{mid,midlate,final}/`（15 类 × 3 层） |

---

## 继承的强制方法纪律（最终版）

1. **代理指标的响应方向必须与目标指标核对。**
   本次 retrieval +1.9 / AUROC −6.45 / AUPRO −31。
   **任何代理指标上的正结果，在跑真实指标之前不得表述为方法有效。**
2. **尾部统计对低方差方向放大极其敏感。** 任何线性校正（白化、归一化）
   在 top-k% 聚合的评分下都要先检查 good/bad gap 是否被压缩。
3. **反事实必须与主配置同源同脚本**（Gate 16 的 row-permutation 错误）。
4. **零初始化在乘积结构中一律是鞍点**（三次）。
5. **中间/未收敛产物不得用于结论性统计量**（Gate 14A 的 100-step residual）。
6. **跨脚本数字比较必须同源重算**（Gate 12B 的撤回）。
7. **测试台 headroom 必须先验证**（Gate 9B v1 的无效检验）。
8. **一个 Gate 前先做 1-object sanity test**：loss/embedding finite、
   grad norm > 0、update norm > 0、20–50 步后 loss 变化。
9. **矩阵函数必须确认语义**（`torch.log` vs 矩阵对数）。
