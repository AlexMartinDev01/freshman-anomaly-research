# Phase R — Baseline Audit（审计报告）

> 冻结于 tag `phase_r_audit`。
> **结论：raw downstream baseline 与 evaluation pipeline 都可信；
> 之前报出的「pixel AUROC 81.47 / AUPRO 14.39」是一个真实 bug，
> 位于 Gate 17A 的 map reshape，不在评估代码，也不在 detector。**
> **修正后：raw = image 95.65 / pixel 96.3 / AUPRO@0.05 73.9。**

---

## 0. 为什么要审计

Gate 17A 之后呈现出三个数字：

| | 值 | 是否合理 |
|---|---|---|
| image AUROC | 95.65 | ✅ 与 AnomalyDINO one-shot ≈96.6 同量级 |
| pixel AUROC | **81.47** | ❌ 偏低（MVTec-AD 上应 ~96-98） |
| AUPRO@0.05 | **14.39** | ❌ **完全不合理** |

AUPRO 十几个点是"定位几乎等于随机"的水平，而同一 detector 的 image AUROC 却有 95.65。
**这种组合在物理上说不通** —— 因此必须先在继续任何研究之前审计评估链路。

## 1. 三项 sanity check（全部通过）

| 检查 | bottle | carpet | grid | screw | tile | 期望 |
|---|---|---|---|---|---|---|
| GT 当 anomaly map → px_AUROC | 99.98 | 99.96 | 99.85 | 99.97 | 99.93 | ~100 |
| GT 当 anomaly map → AUPRO | 95.87 | 97.78 | 92.99 | 99.14 | 93.50 | ~100 |
| 随机 map → px_AUROC | 49.19 | 52.37 | 50.76 | 52.99 | 49.38 | ~50 |
| 随机 map → AUPRO | 2.19 | 1.93 | 2.89 | 1.86 | 1.86 | ~2.5 (=pro_limit/2) |
| 对齐（GT 内外均值比） | 83.2 | 155.6 | 106.5 | 337.9 | 73.3 | ≫1 |

- **GT 当分数** 逼近理论上限（AUROC ≈100，AUPRO 93–99）；
- **随机分数** 落在理论随机水平（AUROC ≈50；AUPRO ≈ pro_limit/2 = 2.5）；
- **对齐**：把 GT 下采样到 patch grid、再经与 detector 完全相同的
  `dists2map`（cv2.INTER_LINEAR → gaussian sigma=4）上采样回原图后，
  GT 内部的均值是外部的 **73–338 倍**。

**→ 评估代码（`pixel_metrics_binned`）、resize 约定、GT 加载、patch grid 映射
全部正确。**

## 2. 真正的 bug：Gate 17A 的 map reshape

`gate17a_fusion_oracle.py` 里：

```python
gt = te["gt_frac"].reshape(len(y), -1)      # -> (n, 1024)  已展平
...
np.save(p, F[i].reshape(gt.shape[1:]))      # -> (1024,)  一维！
```

`cv2.resize` 收到**一维数组**时会把它当成**列向量**再 resize 到 (w, h)，
**空间结构被彻底打乱**。

- 这把 pixel AUROC 从正确的 **96.3** 拉到 **81.5**，
  把 AUPRO 从 **73.9** 拉到 **14.4**。
- **image 级指标不受影响**（`mean_top1p` 只依赖展平后的取值分布），
  所以 Gate 17A 的 STOP 判决本身仍然成立。

**修正**：保留二维网格 `G = te["gt_frac"].reshape(len(y), *shape[1:])`，
再 `np.save(p, F[i].reshape(G.shape[1:]))`。

### 修正后的 Gate 17A（30 格校验 + 全量）

| metric | raw | best-α | gain | α*=0 的格子 |
|---|---|---|---|---|
| img_AUROC | 95.65 | 95.84 | **+0.19** | 131/180 |
| px_AUROC | **96.27** | 96.32 | **+0.05** | 24/30 |
| AUPRO | **73.45** | 73.47 | **+0.02** | 27/30 |

**STOP 判决不变，且更干净**：连 oracle 融合在 pixel 上也只值 +0.05 / +0.02。

## 3. 剩余的「与文献差距」其实只是 pro_limit

MVTec-AD 上的 PRO 通常报 **FPR=0.30**，而本项目的 `AUPRO` 用的是 **0.05**。

| object | pl=0.05 | pl=0.10 | **pl=0.30** |
|---|---|---|---|
| bottle | 83.0 | 90.6 | **96.7** |
| carpet | 87.4 | 93.4 | **97.7** |
| grid | 82.2 | 90.5 | **96.3** |
| screw | 61.1 | 69.1 | 82.3 |
| tile | 51.3 | 68.1 | 86.9 |
| transistor | 49.0 | 58.2 | 70.7 |

**在 pl=0.30 下落在 70–98，与文献区间一致。**
即 baseline 并无异常，只是本项目一直使用更严格的 0.05 口径
（该口径下所有方法都会显著偏低，比较仍然公平）。

## 4. raw baseline 的最终确认数值

| metric | 1-shot | 4-shot | 8-shot |
|---|---|---|---|
| img_AUROC | 95.5 | 96.7 | 97.0 |
| px_AUROC | ~95.5 | 96.7 | 97.0 |
| AUPRO@0.05 | 72.0 | 74.7 | 75.3 |

**与 AnomalyDINO（frozen DINOv2 + patch NN + training-free）同量级，
baseline 复现可信。**

## 5. 审计判据（用户给定）的执行结果

> 若 image AUROC 差 >1–2 点，或 pixel AUROC 差 >3–5 点 → 先修 baseline

| 指标 | 本项目 raw | 文献参考 | 差 |
|---|---|---|---|
| image AUROC | 95.65 | ~96.6 | ~1.0 ✅ 在容差内 |
| pixel AUROC | 96.3 | ~96.5 | ~0.2 ✅ |
| AUPRO | 73.9 @0.05 / ~88 @0.30 | ~90 @0.30 | 口径不同，校正常后 ✅ |

**→ 通过。不需要修 baseline。**

## 6. 对既有结论的影响

| 结论 | 是否受影响 |
|---|---|
| **Model V1 下游失败**（img −6.45，AUPRO −31） | ❌ **不受影响**。`model_v1_eval.py` 的 reshape 本来就是二维的；raw AUPRO 72.0→V1_full 36.9 与本次审计的 73.9 一致 |
| Gate 17A 的 STOP 判决 | ❌ 不受影响（image 级指标与 reshape 无关） |
| Gate 17A 的 pixel 数字 | ✅ **已修正**（81.5→96.3，14.4→73.5） |
| 各 representation gate（13–16） | ❌ 不受影响（只用 retrieval，不涉及 pixel map） |

**唯一被修正的是 Gate 17A 的 pixel 两列。**

## 7. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `audit_pixel_pipeline.py`、`gate17a_fusion_oracle.py`（已修） |
| 结果 | `metrics/gate17a_fusion.csv` |

---

## 继承的方法纪律

1. **任何 anomaly map 落盘前必须断言其 shape 是二维 patch grid。**
   一维 map 进入 `cv2.resize` 不会报错，只会静默打乱空间结构。
2. **像素级指标必须带 sanity check 才能上报**：
   GT 当分数应≈上限、随机分数应≈随机水平、GT 内外比值应≫1。
   本次靠第二项才发现异常。
3. **报告的 AUPRO 必须注明 pro_limit。** 0.05 与 0.30 差 15–20 个点，
   不注明口径的数字无法与文献比较。
4. **image 级指标与 pixel 级指标要分别做合理性检查。**
   本次 image 95.65 是合理的、pixel 14.39 不合理——
   **只有把两者放在一起看才发现矛盾。**
