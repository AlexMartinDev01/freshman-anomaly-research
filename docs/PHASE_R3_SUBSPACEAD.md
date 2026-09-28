# Phase R 步骤 4/5：kNN bank vs PCA normal subspace（SubspaceAD-B）

> 冻结于 tag `phase_r3_subspacead`。
> **判决：在匹配 backbone / 分辨率 / shots / evaluator 的前提下，
> 官方 SubspaceAD（ViT-S/14 @448）整体上不如我们的 kNN `agg_all3`。**
> **但两者是真正互补的——差异集中在 object 层面，不是全局高下。**

---

## 1. 实验设计（这是本次最关键的部分）

要回答的问题是：

    给定同样的冻结特征、同样的 k-shot bank、同样的 splits、同样的 evaluator，
    用 PCA 重建残差打分，能不能打赢用 1-NN 距离打分？

直接跑官方 CLI 再和我们的数字比是**不成立**的，因为至少四处会混进来：

| 差异 | 处理 |
|---|---|
| 它 resize 成 448×448（破坏长宽比）；我们保持长宽比 | MVTec v1 全是正方形 → 两者网格都是 32×32，**无影响** |
| `--layers` 索引 HF `hidden_states`，返回**未过 LayerNorm** 的块输出；我们缓存的是 LN 之后的 | `-7,-4,-1` 对应 block 6/9/12 = 我们的 5/8/11，**深度对齐但 LN 差异无法从外部消除**（已记录，见 §6） |
| 它自己的 AU-PRO 全局池化负样本、map 在 448×448；我们用原始 patch 网格在原图尺寸上评 | 在 `main.py` 加了一个 **env-var 控制的 dump 钩子**，在 `post_process_map` 之前把**原始 32×32 网格**落盘，再用**我们的** `pixel_metrics_binned(pro_limit=0.05)` 重评 |
| 它的 k-shot 抽样用另一个 RNG，无法让我们的 RNG 复现 | **从它自己的 `run.log` 里读回实际抽中的文件名**，再用**完全相同的图片**构造我们的 kNN bank → 逐格配对 |

`--aug_count 30`（随机旋转增强）**是它方法的一部分，不是 k-shot 协议的一部分**，
所以单独作为一个轴跑，而不是折进 baseline。

复现口径：`facebook/dinov2-small`（= DINOv2 ViT-S/14，无 register token，
12 层，hidden 384），`--image_res 448`，`--pca_ev 0.99`，`--agg_method mean`，
shots {1,2,4,8} × seeds {0,1,2} × aug {0,30} = 24 次完整调用，
每次 15 类共 1725 张测试图，**24 个 dump 目录每个都恰好 1725 个文件**。

## 2. 总体结果（配对）

| metric | kNN `agg_all3` | SubspaceAD | delta | SA 更好的格子 |
|---|---|---|---|---|
| **aug=0** | | | | |
| img_AUROC | 94.76 | 93.32 | **−1.45** | 36/180 |
| px_AUROC | 97.18 | 96.76 | **−0.41** | 75/180 |
| AUPRO | 77.25 | 69.58 | **−7.66** | 41/180 |
| **aug=30（它的 canonical few-shot 配方）** | | | | |
| img_AUROC | 94.76 | 94.65 | **−0.11** | 43/180 |
| px_AUROC | 97.18 | 96.78 | **−0.39** | 75/180 |
| AUPRO | 77.25 | 71.11 | **−6.13** | 39/180 |

kNN 那一列在 aug=0 与 aug=30 下**逐位相同**（max diff 0.000000，0 个 shot_files 不一致），
验证了"增强不改变 k-shot 抽取"。

**增强确实帮到了 SubspaceAD**（AUPRO −7.66 → −6.13，图像级 −1.45 → −0.11），
**但没有改变总体结论。**

## 3. 误差矩阵：差异是 object 级的，不是全局的

`delta = SubspaceAD − agg_all3_kNN`，按 object 归入四类
（AUPRO 弱 = 两者中较小者 < 70.0；此门槛在看数之前固定）。

### aug=0

| case | objects |
|---|---|
| `KNN_wins`（6） | grid, bottle, leather, wood, capsule, hazelnut |
| `KNN_HARD_LOC`（3） | cable, transistor, toothbrush |
| `COMPLEMENT`（3） | pill, metal_nut, tile |
| `SA_both`（2） | screw, zipper |
| `TIE`（1） | carpet |

### aug=30

| case | objects |
|---|---|
| `KNN_wins`（5） | bottle, grid, leather, capsule, wood |
| `KNN_HARD_LOC`（3） | cable, transistor, toothbrush |
| `COMPLEMENT`（4） | pill, zipper, hazelnut, tile |
| `SA_both`（2） | **metal_nut, screw** |
| `TIE`（1） | carpet |

### 关键读法

**我们的 kNN 在困难定位类上明显更强**，而且强的正是两边都弱的对象：

```
tile        knn 63.7  SA 44.5   (-19.2)
cable       knn 74.3  SA 60.5   (-13.8)
transistor  knn 57.9  SA 45.5   (-12.5)
toothbrush  knn 73.5  SA 66.0   ( -7.5)
zipper      knn 67.5  SA 68.3   ( +0.8)
```

**但 `screw` 是反过来的，而且幅度极大**（aug=30）：

| shot | kNN AUPRO | SubspaceAD AUPRO | delta |
|---|---|---|---|
| 1 | 42.8 | 65.7 | **+23.0** |
| 2 | 43.6 | 71.6 | **+27.9** |
| 4 | 47.1 | 75.0 | **+27.9** |
| 8 | 62.5 | 75.0 | **+12.5** |

图像级更极端：screw 平均 `knn 61.28 → SA 80.63`（**+19.34**）。

## 4. 最重要的单一发现：增强把 screw 救回来了

```
screw AUPRO, SubspaceAD:   1-shot  40.9 → 65.7     4-shot  48.1 → 75.0
                (kNN 不动:  1-shot  42.8          4-shot  47.1)
```

**同一批 k-shot 图片、同一个 backbone**，只把每张图做 30 次随机旋转扩增，
SubspaceAD 在 screw 上从"和 kNN 打平"变成"领先 20+ AUPRO"。

这和 `PHASE_R3_LAYER_CONFIRM.md` 的 screw 结论**直接互补**：

> screw 是 `agg_all3` 唯一严重退化的 object（img −10.73，12/12 格为负），
> 它同时也正是"少样本下正常样本多样性不足"最严重的 object——
> 而旋转扩增恰好治的就是这个病。

**即：screw 上失败的不是"kNN vs 子空间"这个选择，而是"只有 k 张图"这件事。**

## 5. 对下一步模型问题的意义

原计划第 6 步是"根据真实 downstream error matrix 定义下一步模型问题"。矩阵给出的答案是：

**两个方法各自失败的 object 集合几乎不重叠：**

- kNN `agg_all3` 输在：`screw`（−23 AUPRO）、`metal_nut`（低 shot）
- SubspaceAD 输在：`tile`（−19）、`cable`（−14）、`transistor`（−12）、`toothbrush`（−7.5）

而且**两边一起失败**的对象是稳定的：

```
tile 63.7/44.5   cable 74.3/60.5   transistor 57.9/45.5   toothbrush 73.5/66.0
```

> **这四个对象才是真正值得立项的 gap。**
> 它们的共同点不是"纹理/物体"（tile 是纹理，transistor/toothbrush 是物体），
> 而是**缺陷与正常区域对比度低、且空间弥散**——
> 与 §PHASE_R3_LAYER_CONFIRM 里识别出的"中层弥散伤害低 FPR 区域召回"是同一个机制。

这比之前任何一条代理路线（缺陷检索 / covariance / generator）都更有依据：
**它是真实下游指标直接指出来的，而且在两个独立方法上都复现。**

## 6. 限制与必须记录的差异

1. **LN 差异（无法消除）。** 官方代码用 HF `hidden_states`（块输出，未过末层
   LayerNorm），我们的缓存是 LN 之后的。深度对齐了，归一化没对齐。
   这属于"用它的代码路径就必然带进来的差异"，不能从外部去掉。
2. **`--no_aug_categories` 默认 `["transistor"]`。** 即使 `aug_count=30`，
   transistor 也**不带增强**跑。这是它官方的默认行为，本次保留。
3. **图像分数的语义略有不同。** 官方在 448×448 后处理图上取 top-1%，
   我们在原始 32×32 网格上取 top-1%。两者都已用**我们的** evaluator 重评，
   所以配对是公平的；但与论文表格里的绝对数字不可直接比。
4. **e3（metal_nut/pill/screw）只有 3 个 object**，其 "SA 全面获胜" 的
   结论权重应低于其它 shard。真正稳的是 screw 单对象上的
   **跨 4 个 shot × 3 个 seed 一致**的大幅领先。
5. 我们**没有**跑 canonical 配置（giant @672）：本机 8 GB 显存跑不动。
   canonical 复现留到需要与论文主表对齐时再租卡。

## 7. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `subspacead_run.py`、`gate_r2_subspacead_eval.py`、`gate_r2_error_matrix.py` |
| 结果 | `metrics/gate_r2_subspacead_b.csv`（360 行） |
| dumps | `results/subspacead_b/dumps/`（24 × 1725 张原始网格 map） |
| 日志 | `results/subspacead_b/run_aug{0,30}*.log`、`shards/e{1..4}.log` |
| 上游改动 | `third_party/SubspaceAD/main.py`：env-var 控制的 dump 钩子、`np.trapz`→`np.trapezoid` |
