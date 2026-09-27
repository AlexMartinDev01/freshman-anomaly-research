# Phase R 步骤 2：downstream-only layer 重测（冻结）

> 冻结于 tag `phase_r2_layer`。
> **判决：中层+末层聚合（agg_all3）在真实下游上明显优于只用末层，
> 且图像级几乎无代价。Gate 16 基于 Recall@1 的 layer 结论作废。**

---

## 最终结论（严谨表述）

> **用真实下游指标（1/4-shot，15 objects）重测 layer：
> `agg_all3` = z(mid) + z(midlate) + z(final) 相对只用 final，
> AUPRO **75.92 → 80.88（+4.96，13/15 个 object 改善，27/30 个格子改善）**，
> pixel AUROC **96.66 → 97.69（+1.03，23/30）**，
> image AUROC **+0.05（基本不变）**。**
> **即：layer aggregation 用几乎为零的图像级代价，换来了大幅的定位提升。**

---

## 1. 方法

| config | 说明 |
|---|---|
| `final_raw` | block 11 patch 距离（= 此前所有实验的 baseline） |
| `mid_raw` | block 5 |
| `midlate_raw` | block 8 |
| `mid_mean` | z(mid) + z(midlate)，平均 |
| **`agg_all3`** | z(mid) + z(midlate) + z(final)，平均 |

- 无 covariance、无训练、无新模块；
- 每层先按 object 的测试集做 z-score 再平均，避免被尺度大的层主导；
- **bank 用固定 patch 预算**（shot × 1024 个 patch，各层相同）——
  `cache_ml` 的 train 是展平的，无法按图取 shot；
  同时这也是正确的控制（Gate 7B-4 证明 patch 数是强杠杆，固定它才隔离出 layer 效应）。

## 2. 三对象 smoke（carpet / bottle / transistor）

| config | img_AUROC | px_AUROC | AUPRO |
|---|---|---|---|
| mid_raw | −2.88 | +0.22 | +3.93 |
| midlate_raw | −3.59 | +0.75 | +5.43 |
| mid_mean | −1.75 | +1.07 | +7.24 |
| **agg_all3** | **−0.51** | **+1.43** | **+7.08** |

单中层**大幅伤害 image AUROC**（−2.9 ~ −3.6）但**提升定位**（+4 ~ +5.4）；
聚合把图像级代价压到 −0.51，同时保留定位收益。

## 3. 十五类验证（1/4-shot）

| metric | final_raw | agg_all3 | gain | 更好的格子 |
|---|---|---|---|---|
| img_AUROC | 97.31 | 97.36 | **+0.05** | 12/30 |
| px_AUROC | 96.66 | **97.69** | **+1.03** | 23/30 |
| **AUPRO** | 75.92 | **80.88** | **+4.96** | **27/30** |

逐 object AUPRO：

| object | final_raw | agg_all3 | gain |
|---|---|---|---|
| tile | 50.5 | 64.2 | **+13.7** |
| transistor | 48.7 | 61.3 | **+12.6** |
| leather | 85.8 | 95.3 | +9.6 |
| wood | 77.6 | 84.1 | +6.5 |
| grid | 82.9 | 88.4 | +5.5 |
| carpet | 88.3 | 93.6 | +5.3 |
| cable | 71.8 | 76.8 | +5.0 |
| screw | 70.1 | 75.0 | +4.8 |
| zipper | 63.9 | 67.8 | +4.0 |
| metal_nut | 77.0 | 80.8 | +3.7 |
| bottle | 83.2 | 86.5 | +3.3 |
| capsule | 84.9 | 86.7 | +1.8 |
| pill | 86.9 | 88.2 | +1.4 |
| hazelnut | 87.5 | 86.9 | −0.6 |
| toothbrush | 79.8 | 77.5 | −2.3 |

**13/15 为正，且收益集中在最难的 object 上（tile、transistor 的 baseline AUPRO 只有 50 左右）。**

image AUROC 逐 object：7/15 为正，mean +0.05，最差 −1.7——**实质不变**。

## 4. 判决与行动

**Gate 16 的 layer 结论作废。** 它是用 Recall@1 选的 final layer，
而 Gate 17A 已证明 Recall@1 是误导性代理。用真实下游重测后：

> **应当使用 `agg_all3`（mid + midlate + final 的 z-score 平均），
> 而不是只用 final layer。**

这是本项目第一条**在真实下游上经过验证的正向方法改进**。

## 5. 限制

- bank 用固定 patch 预算而非 shot 图像（`cache_ml` train 是展平的）。
  要换成真正的 K-shot 图像 bank，需重缓存 mid/midlate 的 per-image train 特征。
- 只测了 1/4-shot、split 0。扩展 shot 需要更长时间。
- 层位置（5/8/11）是固定选的，没有扫 block——**这是有意的**，
  避免又回到「扫十几个超参」的模式。
- mid/midlate 的特征是**新缓存的**（`cache_ml`），
  与之前各 gate 用的 final-layer 缓存（`cache_v1`）来自同一次前向，数值一致。

## 6. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `gate_r1_layer_smoke.py` |
| 结果 | `metrics/gate_r1_layer_smoke.csv`（3 objects）、`gate_r1_layer_full.csv`（15 objects） |
| 缓存 | `results/model_v0/cache_ml/{mid,midlate,final}/` |
| maps | `results/model_v0/maps_layer_smoke/` |

---

## 方法纪律（新增）

1. **代理指标选出的超参必须用真实指标复检。**
   本次 layer 选择是第二次（第一次是 Model V1 整体路线）。
2. **超参扫描要保持有界**：固定 3 个层、固定 5 个配置、只跑一次 smoke，
   有信号才扩到全量。
3. **聚合前先做尺度归一化**（本次 z-score），否则聚合等于被尺度最大的那层主导。
