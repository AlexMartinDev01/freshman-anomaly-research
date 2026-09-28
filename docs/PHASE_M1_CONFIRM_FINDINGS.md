# Phase M1-confirm 结果（全量 27 类）

> 判据在跑之前冻结于 `docs/PHASE_M1_CONFIRM_PREREGISTRATION.md`，未作任何修改。
> 本文档只记录数字与读法，不下"GO/STOP"的结论——§5 出口条件存在一处
> 文本歧义，需要项目负责人裁定（见最后一节）。

---

## 0. 运行

| | |
|---|---|
| 表示 | R0 = DINOv2-S/14 @448，R1 = DINOv2-S/14 @672（同三层 blocks 5/8/11，同 `agg_all3` 配方） |
| 数据 | MVTec 全 15 类 + VisA 全 12 类 |
| 设计 | shots {1,2,4,8} × splits 3 × 2 表示 = 648 行 / 324 cells |
| evaluator | `pixel_metrics_binned(pro_limit=0.05)`，全测试图入 `jobs`（good 图 gt=None） |

四个分片全部正常收尾（168 + 192 + 144 + 144 = 648 行，27 对象 × 24）。

**两项前置对齐验证（都通过）：**

1. `verify_m1_align.py` —— R0 与 R1 的 train 列表**逐项相同**（所以
   `draw_images` 抽到的是同一批 k-shot 图像），test 列表也逐项相同
   （所以 r0[i] 与 r1[i] 是同一张图）。27/27 对象通过；网格严格按
   ×1.5 缩放（448 的 32×1.4 → 672 的 48×14，宽度 35→52 等）。
   若这一步不成立，本阶段所有 Δ 都无意义。
2. `verify_m1_maps.py` —— 从落盘的 map 重建的全局 px_AUROC / AUPRO
   与 `m1confirm_all.csv` **逐位一致**（差异 ≤1.4e-14），确认机制脚本
   读的是产生 headline 数字的那批 map。

---

## 1. 判据结果（全部通过）

```
C1  整体为正         mean dAUPRO = +3.330          288/324 cells = 88.9%   PASS
                     [另一读法] 25/27 对象为正 = 92.6%
C2  无系统性退化     mean d_img_AUROC = +0.306     mean d_px_AUROC = +0.276 PASS
C3  异质性（核心）   within-object Fisher-z rho = -0.162 (21/27 为负)
                     FE beta = -0.1702, clustered SE 0.0488, t = -3.49, p = 1.7e-03  PASS
C4  剂量响应         HARD: C_feat 0.0981 -> 0.1049 -> 0.1087  MONOTONE UP
                           sep_auc 0.7881 -> 0.8068 -> 0.8167  MONOTONE UP
                           AUPRO  57.124 -> 61.466 -> 62.287   MONOTONE UP  PASS
```

聚类稳健 SE（按 object，G=27）下的描述性结果：

```
AUPRO      mean +3.330   naive SE 0.177   clustered SE 0.561   t_clust +5.93
px_AUROC   mean +0.276   naive SE 0.033   clustered SE 0.110   t_clust +2.52
img_AUROC  mean +0.306   naive SE 0.110   clustered SE 0.301   t_clust +1.02
sep_auc    mean +0.007   naive SE 0.001   clustered SE 0.004   t_clust +1.88
```

朴素的 per-cell SE 把 AUPRO 的 t 抬到 18.8，按对象聚类后是 **+5.93**——
方向不变，但"精度"缩水 3.2 倍。这与 F1 的教训同源（那里膨胀约 9 倍）。

**C3 在预注册允许的两个指标上都成立**（`PHASE_M1_CONFIRM_PREREGISTRATION.md` §4
允许 "ΔAUPRO 或 ΔTPR@5%"），且各自用两种估计量：

| 机制量 | Fisher-z rho | 负向对象 | FE beta | t (clustered) |
|---|---|---|---|---|
| `C_feat_448` vs ΔAUPRO（**预注册主判据**） | −0.162 | 21/27 | −0.1702 | −3.49 |
| `sep_auc_448` vs ΔAUPRO（尺度无关） | −0.220 | 23/27 | −0.2508 | −5.57 |
| `C_feat_448` vs ΔTPR@5% | −0.184 | 22/27 | −0.1833 | −3.74 |

四个组合方向一致、都显著——C3 不是单点结果。

---

## 2. 必须并列报告的负面/异质结果

**逐对象 ΔAUPRO 的两处为负**：`carpet −0.975`（0/12 cells 为正）、`macaroni2 −0.294`。

**image AUROC 退化最大的四类**：`macaroni2 −3.66`、`screw −2.91`、
`toothbrush −1.51`、`fryum −1.24`。C2 判的是**均值**（+0.306，安全），
但 `screw` 的 image 级退化是真实的、且随分辨率**单调恶化**
（img_AUROC 66.97 → 65.30 → 63.76 @ 448/560/672）。

**screw 是一个持续的反例**：它在 Phase A 拉穿了 `agg_all3`，
在这里 AUPRO 又不单调（52.12 → 53.44 → **51.11**，672 反而低于 448）。
672 没有解决 screw。

**机制在部分对象上反向**：within-object rho 为正的对象是
`carpet +0.563`、`bottle +0.341`、`tile +0.325`——
即在这三类里，**baseline 可分性越高**的缺陷反而**收益越大**。
`carpet` 恰是 baseline 可分性最高、总体 ΔAUPRO 为负的对照对象。
所以 C3 是一个**总体成立、但并非普遍成立**的规律。

## 3. M1-mechanism §4 的第二个方向是零结果

预注册 §4 列出两个期望方向。第一个强成立（C3）。**第二个不成立**：

```
gt_patch_n vs dAUPRO   Fisher-z -0.059   18/27    FE beta -0.0000  t -0.04  p 0.97
log_area   vs dAUPRO   Fisher-z -0.032   16/27    FE beta -0.0069  t -1.42  p 0.17
```

联合回归（同时含两者）里 `C_feat_448` 系数为 −0.1816（t = −3.77），
控制住缺陷面积后**依然成立**；而 `log_area` 只有 t = −1.94（p = 0.063）、
`gt_patch_n` t = +0.19。

**读法**：驱动收益的是**低可分性**，不是**尺度**。
"分辨率高 → 小缺陷不再被 patch 平均掉 → 收益大"这个直觉版本
**没有被数据支持**。可支持的表述是更窄的那一句：

> higher resolution is not universally beneficial; it specifically rescues
> anomalies that are **poorly separable** at coarse patch granularity.

注意这两者在数据里相关（小缺陷往往也低可分）但不重合——
所以这个区分是实证的，不只是措辞。

## 4. C4 的剂量响应强烈饱和

```
HARD subset AUPRO:  448 -> 560 : +4.34
                    560 -> 672 : +0.82
```

**448→560 一步就拿到 448→672 总增益的 84%。**
这对 V3 的设计直接相关：如果要做 coarse-to-fine，
目标应该是"把 token 密度提到 ~560 等效水平"，
而不是特别地追求 672——后者的边际收益已经很小。

逐对象看，6 个困难对象里 4 个 AUPRO 三点单调；`screw` 在 672 反降，
`pcb2` 在 672 基本持平（65.21 → 64.87）。
预注册 §3 明确"不要求每个对象严格单调"，总体剂量响应成立。

## 5. 口径与局限（事先已声明，此处复述）

1. **逐图像 AUPRO ≠ headline AUPRO。** 机制分析用的逐缺陷 AUPRO 在**单张图内
   重新归一化 FPR**，而官方 AUPRO 的 FPR 轴跨整个 cell 的所有正常像素，
   因此官方值**不可按图像相加**。逐图像版本是合法的 per-defect 效应量，
   也是唯一能在对象内做相关的量，但**不得**被引用为"那个 AUPRO"。
2. **`C_feat` 的跨分辨率比较**是预注册 §4 自己声明的尺度混淆。C4 按要求
   字面计算了它；本次 `C_feat` 与尺度无关的 `sep_auc` 在困难子集上**方向一致**，
   所以该风险本次未兑现。含对照的 ALL7 上 `C_feat` 不单调（由 carpet 贡献），
   但 `sep_auc` 与 AUPRO 仍单调。
3. **机制与剂量都在 shots {1,4} 上测的**（与 M1 smoke 及预注册 §3 一致），
   不是全部四档 shot。C1/C2 用了全部四档。
4. **`C_feat` / `sep_auc` 使用 GT mask，是 oracle 诊断量，不得进入最终方法。**
5. **已知列名错标（不影响任何指标）**：`phase_m1_rescue.py` 中
   `"n_bad": len(jobs)`，而 `jobs` 是全部测试图（含 good 图）。
   该列没有任何分析使用；运行期间未修改冻结的测量脚本，仅在此记录。

## 6. 出口条件的文本歧义（需要裁定，不由本文档决定）

`PHASE_M1_CONFIRM_PREREGISTRATION.md` §5：

```
三步（confirm / dose / mechanism）全部成立 -> GO：Model V3
任一步不成立 -> STOP，不进入 V3
```

- **confirm**（C1/C2/C3/C4）：**四條全部通过**。
- **dose**（C4）：通过。
- **mechanism**：§4 列了两个期望方向，**第一个强成立、第二个是零结果**。

所以"三步全部成立"是否要求 §4 的两个方向都成立，是冻结文本本身的歧义：

- **读法 A（宽）**：mechanism 步的要求已由 C3 承载（§2 把 C3 定义为
  "本阶段真正的科学主张"），§4 的尺度方向是一个附加的**探索性**方向，
  零结果不阻断 GO。→ 三步全部成立，**GO V3**。
- **读法 B（严）**：§4 的两个方向都是 mechanism 步的组成部分，
  其一为零结果 → mechanism 步**部分成立** → 按"任一步不成立"**STOP**。

本文档不替这个判断做选择。需要指出的是：**读法 A 会让 §4 的第二个方向
事实上失去约束力**——如果它是探索性的，就不该和第一个方向并列写在同一段
期望里。这一点在选择时应当被明确考虑到。

## 7. 本阶段**没有**建立的结论

- 没有证明"瓶颈就是 patch 分辨率，不是表示族"。WRN50 的失败（M1 smoke）只排除
  "换成这个 CNN 就能普遍解决"，**不排除** ConvNeXt / SAM / RADIO / CLIP。
- 没有证明机制在**所有**对象上成立（carpet / bottle / tile 上反向）。
- 没有证明 672 在 image 级检测上更好（mean Δimg +0.306 但 t_clust 仅 +1.02；
  screw 显著退化）。
- `C_feat` / `sep_auc` 全程为 oracle 诊断量，**未**据此做任何超参调优。

## 8. 复现命令

```bash
# 对齐与正确性验证
python experiments/model_v0/verify_m1_align.py
python experiments/model_v0/verify_m1_maps.py --cells 5

# C1 / C2
python experiments/model_v0/phase_m1_confirm_analysis.py

# C3 + M1-mechanism §4
python experiments/model_v0/phase_m1_mechanism.py --objects <shards> --shots 1,4 --splits 3 --tag mech_a
python experiments/model_v0/phase_m1_mechanism_analysis.py --tags mech_a,mech_b,mech_c

# C4
python experiments/model_v0/cache_m1_repr.py --repr dino672 --edge 560
python experiments/model_v0/phase_m1_dose.py --measure --shots 1,4 --splits 3
python experiments/model_v0/phase_m1_dose.py --analyze --shots 1,4 --splits 3
```

原始输出留存：`results/model_v0/{confirm,mechanism,dose}_analysis.txt`。
