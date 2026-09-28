# Phase R 步骤 3：真 k-shot bank 上的多层确认（未通过预注册判决）

> 冻结于 tag `phase_r3_layer_confirm`。
> **判决：NOT CONFIRMED。按预注册标准，`agg_all3` 不予冻结。**
> 但失败**完全来自单一 object（screw）**，且定位指标三项全部通过。
> 这是一个"部分成立、且失败点可定位"的结果，不是一个笼统的否定。

---

## 1. 为什么要重做

`gate_r1_layer_smoke.py`（PHASE_R2）报出 `agg_all3` 相对 final-only：
AUPRO **+4.96**、px_AUROC **+1.03**、img_AUROC +0.05。但它有两个 confound：

1. **bank 不是真 k-shot。** `cache_ml` 的 train 是**展平**的 patch 池，
   所以 bank 只能是"从全部训练图里抽 shot×1024 个 patch"，
   而不是"抽 shot 张图"。8-shot 时预算还被 6000 的池上限截断。
2. **只有 split 0、只有 1/4-shot。**

本次把两者都修掉：新建 `cache_ml_img`（per-image 缓存），
bank = 真的抽 `shot` **张图**，覆盖 1/2/4/8-shot × 3 次独立抽样 × 15 objects。
其余全部与 PHASE_R2 保持一致：test 特征取自同一个 `cache_ml` npz，
同一套 `pixel_metrics_binned(pro_limit=0.05)`，同样的 z-score 后平均。

## 2. 预注册判决（运行前冻结）

```
px_AUROC 平均增益 >= +0.5
AUPRO    平均增益 >= +2.0
AUPRO    多数 (object, shot, split) 格子为正
img_AUROC 平均增益 >= -0.3      <- 不允许图像级系统性退化
```

## 3. 结果

15 objects × 4 shots × 3 splits = 180 格 × 2 config = 360 行。

| metric | final_raw | agg_all3 | gain | 更好的格子 |
|---|---|---|---|---|
| img_AUROC | 95.84 | 95.41 | **−0.43** | 79/180 |
| px_AUROC | 96.46 | 97.21 | **+0.74** | 127/180 |
| AUPRO | 73.98 | 77.38 | **+3.40** | **135/180** |

按 shot 的增益：

| metric | 1-shot | 2-shot | 4-shot | 8-shot |
|---|---|---|---|---|
| img_AUROC | −0.51 | −0.64 | −0.30 | −0.29 |
| px_AUROC | +0.63 | +0.82 | +0.75 | +0.77 |
| AUPRO | +2.23 | +2.95 | +3.83 | **+4.59** |

```
px_AUROC  gain +0.74   (need >= +0.5)   OK
AUPRO     gain +3.40   (need >= +2.0)   OK
AUPRO     better 135/180 cells          OK
img_AUROC gain -0.43   (need >= -0.3)   FAIL
-> NOT CONFIRMED: do not freeze
```

**定位三项全部通过，且 AUPRO 增益随 shot 单调上升（+2.23 → +4.59），
这是真实效应而非噪声。图像级守卫失败，差 0.13。**

## 4. 失败点的完整分解

| | img_AUROC 平均增益 |
|---|---|
| 全部 15 objects | **−0.433** |
| 去掉 screw | **+0.303**（会通过 −0.3 门槛） |
| screw 单独对均值的贡献 | **−0.716** |

**screw 在 12/12 个格子上为负**（−6.2 ~ −17.5），不是抽样噪声。

逐 object 平均增益（img_AUROC 排序）：

```
screw     -10.73   px +1.83   AUPRO  -5.85
zipper     -1.39   px +2.50   AUPRO  +4.37
tile       -0.41   px +1.89   AUPRO +12.57
capsule    -0.25   px +0.00   AUPRO  +1.51
metal_nut  -0.19   px -1.48   AUPRO  -0.99
toothbrush -0.12   px -0.14   AUPRO  -2.23
leather     0.00   px +0.56   AUPRO +11.71
carpet      0.00   px +0.40   AUPRO  +5.20
grid       +0.13   px +0.24   AUPRO  +4.87
bottle     +0.17   px +0.08   AUPRO  +4.22
wood       +0.50   px +1.07   AUPRO  +6.20
cable      +0.73   px +2.09   AUPRO  +2.62
transistor +1.26   px +2.96   AUPRO +10.71
hazelnut   +1.76   px -0.28   AUPRO  -4.20
pill       +2.06   px -0.58   AUPRO  +0.31
```

**14/15 个 object 的图像级增益 ≥ −0.41；唯一的 −10.73 拉垮了整个均值。**

## 5. 机制（screw 逐层实测）

screw split 0，各层单独评估（同 bank，同 evaluator）：

| config | 1-shot img / px / AUPRO | 4-shot img / px / AUPRO |
|---|---|---|
| `final_raw` | 69.7 / 89.4 / **52.7** | 78.6 / 92.4 / **55.8** |
| `mid_raw` | 53.2 / 88.9 / **18.0** | 56.1 / 91.2 / **20.7** |
| `midlate_raw` | 57.3 / 91.6 / **25.5** | 62.5 / 92.9 / **31.9** |
| `agg_all3` | 63.5 / **92.8** / 42.4 | 69.9 / **93.8** / 48.6 |

对照 tile / carpet（同一 bank 构造）：

```
tile  1-shot:  final 100.0 / 95.1 / 52.7   midlate 99.8 / 96.6 / 65.2   agg 99.5 / 96.9 / 63.9
tile  4-shot:  final 100.0 / 95.4 / 51.0   midlate 99.9 / 97.1 / 66.0   agg 99.9 / 97.2 / 64.0
carpet1-shot:  final 100.0 / 99.1 / 89.0   midlate 100.0/ 99.4 / 93.1   agg 100.0/ 99.5 / 93.6
```

**在 screw 上，中层本身就极差**（AUPRO 18.0 / 25.5，而 final 是 52.7），
等权平均把 2/3 的权重交给了两个坏层，于是 42.4。
在 tile / carpet 上中层本身就更好，聚合因此受益。

**统一机制（"空间弥散"特征）：**

> 中层距离图更"弥散"。弥散对 **pixel AUROC 有利**（全局排序仍然成立），
> 但**对 AUPRO 有害**（AUPRO 是低 FPR 下的逐连通域召回，
> 弥散会把异常摊到多个连通域上）；同时抬高大量正常 patch 的 top-1% 均值，
> **图像级分离度变差**。

screw 正是最敏感的 object：小而亮、低对比度的金属件。

**所以问题不是"该不该聚合"，而是"该不该等权"。**
等权平均无法区分"这层在这张图上可信"与"这层在这张图上不可信"。

## 6. 方法纪律：本次自查出的一个错误

复核时我曾用 `load_test('final', obj)['feats']` 去配 mid / midlate 的 bank，
即**用末层测试特征去查中层 bank**，得到过一组错误的逐层数字
（曾据以推测 mid AUPRO 23.9 之类）。该结果是**无效的，已撤回**。

用正确的逐层测试特征重算后，确认运行**逐比特可复现**：

```
bottle  fresh agg img_AUROC  99.84  matches disk: True
carpet  fresh agg img_AUROC 100.00  matches disk: True
screw   fresh agg img_AUROC  63.48  matches disk: True
tile    fresh agg img_AUROC  99.53  matches disk: True
```

`final_raw` 对原始末层距离图是仿射等价（R² = 1.000000），
即末层 bank 完全一致。

**教训：验证脚本本身必须先证明它能复现已知产物，再用来下结论。**
本次是靠"同一个格子两次算出不同数"才发现——如果当时不去追这个矛盾，
就会用一组假数字去解释 screw。

## 7. 限制

- 只有 3 次独立 k-shot 抽样。screw 的 12/12 全负已足以排除抽样噪声，
  但 14/15 个 object 的 +0.303 仍应视为估计而非定论。
- 层位置（block 5/8/11）固定，未扫 block——这是有意的，
  避免退回"扫十几个超参"的模式。
- 聚合权重来自 object 测试集统计量。这是**无标签但 transductive** 的。
  该 z-score 是全局仿射、对所有 map 相同，**不可能改变任何排秩指标**
  （img/px AUROC、AUPRO 全部对它有不变性），它只决定层间权重。
  因此"权重是 transductive"是成立的批评，"指标被抬高"不成立。
  **独立的权重稳健性检验已完成（`PHASE_R3_WEIGHTS.md`）：**
  完全去掉权重（三张原始距离图直接平均）时 AUPRO 仍 +3.05、px_AUROC +0.67，
  用纯训练集估计的权重时 AUPRO +3.18。
  **该 transductive 质疑已被证伪：增益不来自权重选择。**
  同时图像级退化在三种方案下都存在（−0.36 / −0.41 / −0.55），
  也**不是**权重假象。

## 8. 对原计划的影响

原计划第 3 步是"若多层 px/AUPRO 改善成立，冻结 `agg_all3` 为新 baseline"。

- px/AUPRO 改善**确实成立**（三项定位判据全过，且 AUPRO 随 shot 单调增长）。
- 但**预注册判据捆绑的 img_AUROC 守卫失败了**，因此**不冻结**。
- 判据**不得事后修改**。若要改变结论，只能：
  (a) 提出新的、事前冻结的判据并重跑；或
  (b) 明确承认这是对原判据的修正，并说明修正理由。

**本节的"下一步"曾写为"层权重是否可以从无标签信息中估计出来"。
该建议在权重稳健性检验（`PHASE_R3_WEIGHTS.md`）之后已作废：**
既然完全不加权也拿到 AUPRO +3.05，**权重估计不是瓶颈**，
继续做它只会得到一个已在手的增益的更精细版本。

SubspaceAD 对照（`PHASE_R3_SUBSPACEAD.md`）给出了真正有依据的方向：

> 两个方法失败的对象集合几乎不重叠。**真正值得立项的是两边一起失败的对象**：
> `tile`（knn 63.7 / SA 44.5）、`cable`（74.3 / 60.5）、
> `transistor`（57.9 / 45.5）、`toothbrush`（73.5 / 66.0）。

而 screw 给出了本阶段最强的单一线索：**同一个 backbone、同一批 k 张图，
只把每张图做 30 次随机旋转扩增，SubspaceAD 在 screw 上从 40.9 涨到 65.7 AUPRO。**
即 screw 上失败的不是"kNN vs 子空间"的选择，而是**"只有 k 张图"**本身。

## 9. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `cache_multilayer_img.py`（per-image 缓存）、`gate_r2_layer_confirm.py`、`gate_r2_confirm_merge.py` |
| 结果 | `metrics/gate_r2_layer_confirm.csv`（360 行）、`metrics/gate_r2_confirm_s{1..4}.csv` |
| 缓存 | `results/model_v0/cache_ml_img/{mid,midlate,final}/` |
| maps | `results/model_v0/maps_layer_confirm/`（15 objects × 12 cells × 2 config） |
| 日志 | `shards/s{1..4}.log`、`cache_ml_img.log` |
