# Phase M1-confirm 预注册（在跑全量之前冻结）

> 冻结时间：在生成任何**全量** DINO@672 特征之前。
> 本文档写定后不得修改；若需变动，另开新文档并说明理由。

---

## 0. 已成立的前提（M1 smoke，7 对象）

冻结判据下，**R1 = DINOv2-S/14 @672 通过全部四条**：

```
M1-1 sep_auc hard mean +0.029, 6/6 困难对象为正   OK
M1-2 AUPRO hard mean +5.16                       OK
M1-3 AUPRO 正向 5/7                              OK
M1-4 img_AUROC mean +0.87                        OK
```

而 **R2 = WRN50 layer2+3 未通过**（M1-3 只有 4/7）。

**严谨表述（本阶段沿用，不得加强）：**

> **在当前测试范围内，提高同一 DINOv2 的空间采样分辨率，是第一个同时稳定
> 提高 feature separability 与真实 anomaly localization 的干预；
> 而 WRN50 局部特征没有表现出同样的一致性。**
>
> ——不能表述为"瓶颈就是 patch 分辨率，不是表示族"。
> WRN50 的失败只排除"换成这个 CNN 就能普遍解决"，
> **不排除所有其他 representation family**（ConvNeXt / SAM / RADIO / CLIP 均未测）。

## 1. 本阶段只做一件事

把 R0 与 R1 在**完整数据集**上跑完，不再试任何其它 backbone：

| | |
|---|---|
| R0 | DINOv2-S/14 **@448** |
| R1 | DINOv2-S/14 **@672** |
| 数据 | MVTec **全 15 类** + VisA **全 12 类** |
| shots | 1 / 2 / 4 / 8 |
| splits | 3 |
| 其它条件 | 完全相同（同 bank draws、同 z-score 后平均、同 evaluator `pro_limit=0.05`） |

## 2. 判据（**必须全部成立**）

```
C1  整体为正：mean ΔAUPRO > 0，且 >= 60% 的 object/cell 为正向
C2  无系统性退化：mean Δimg_AUROC >= -0.3 且 mean Δpx_AUROC >= -0.3
C3  异质性（核心机制）：
    within-object 的 Spearman( C_feat_448 , ΔAUPRO ) < 0
    即 baseline 可分性越差的对象/缺陷，672 的收益越大
C4  剂量响应（M1-dose，448 → 560 → 672）：
    mean C_feat 与 mean AUPRO 在三点上单调递增
```

**C3 是本阶段真正的科学主张。** "672 普遍更强"本身不新
（AnomalyDINO 自身从 448 提到 672 也有提升；SubspaceAD 的消融显示 448–672 接近饱和）——
新的是**它对低可分性困难缺陷特别有效**。

## 3. M1-dose（分辨率响应曲线）

只加一个中间点，**不再扫更多**：

```
448 (32x32)  →  560 (40x40)  →  672 (48x48)
```

在固定的困难 subset 上检验三点单调。不要求每个对象严格单调，
但**总体上必须出现剂量响应**——否则 672 的好处可能只是某次
resize / interpolation 的偶然。

## 4. M1-mechanism（机制证据）

把每个 defect instance 的 `ΔAUPRO`（或 `ΔTPR@5%`）与两个量联系起来：

1. **baseline 可分性** `C_feat^448`
2. **缺陷在 token grid 中的有效尺度**（GT 面积 / 每个 patch 的像素数，或覆盖的 patch 数）

期望的方向：

```
C_feat^448 越低  =>  Δ_672 越大
缺陷在 448 网格上越小  =>  Δ_672 越大
```

若成立，论文故事是：

> **higher resolution is not universally beneficial; it specifically rescues
> anomalies that are poorly separable at coarse patch granularity.**

## 5. 出口（跑之前定好）

```
三步（confirm / dose / mechanism）全部成立 -> GO：Model V3
    Coarse-to-Fine Resolution-Adaptive Anomaly Localization
      global 448 -> candidate regions -> high-resolution local crops -> refined map
    注意：贡献不能是"高分辨率有用"，必须是
    "针对 few-shot frozen foundation features 的 low-separability bottleneck，
     以及如何高效、选择性地恢复空间细节"
任一步不成立 -> STOP，不进入 V3
```

## 6. 限制（事先声明）

1. M1 smoke 只测了 7 个对象，其中 VisA 三个全为 PCB，类别同质性高。
   全量阶段覆盖全部 27 类，可检验这一点是否影响结论。
2. `C_feat` / `sep_auc` 使用 GT mask，是 **oracle 诊断量**，不得进入最终方法。
3. 高分辨率会使评测成本上升（672 的 patch 数为 448 的 2.25 倍）；
   本阶段只关心**是否**有效，成本问题留给 V3 设计。
4. 不训练任何东西；两个 representation 全部冻结。

## 7. 资产（待生成）

| 类型 | 内容 |
|---|---|
| 缓存 | `cache_m1_dino672/`（全 27 对象）、`cache_m1_dino560/`（dose 用） |
| 脚本 | `cache_m1_repr.py`（扩展全量 + 560）、`phase_m1_rescue.py`、`phase_m1_analysis.py` |
| 本文件 | `docs/PHASE_M1_CONFIRM_PREREGISTRATION.md`（冻结，不得修改） |
