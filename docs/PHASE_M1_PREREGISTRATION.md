# Phase M1 预注册：Feature Separability Rescue（在跑任何实验之前冻结）

> 冻结时间：在生成任何 R1（DINO@672）或 R2（CNN）特征之前。
> 本文档写定后不得修改；若需变动，另开新文档并说明理由。

---

## 0. 问题

F1+F2 已确证（跨 MVTec/VisA、跨 kNN/SubspaceAD、预注册判据）：

> **限制低 FPR 定位的是缺陷与局部正常结构的特征可分性 `C_feat`。**

Model V2 又证明：**仅仅重组现有 DINO 邻域信息不能制造新的可分性**
（context-conditioned matching：M1 −1.85 / M2 −3.54 / M3 −1.90，三条全 FAIL）。

所以问题从"怎么更聪明地读 DINO"变成：

> **怎么让低对比异常在 feature space 里真正离正常更远？**

## 1. 三个 representation

| | 配置 | 回答的问题 |
|---|---|---|
| **R0** | DINOv2-S/14 **@448** | 现有基线（结果已存在，直接复用） |
| **R1** | DINOv2-S/14 **@672** | 是不是 14×14 patch 在 448 下把低对比、小尺度变化平均掉了？ |
| **R2** | **WideResNet50-2, layer2 + layer3**（PatchCore 式，ImageNet 预训练，冻结） | 是不是 DINO 这一**表示族**本身缺少局部纹理信息？ |

R2 只用**一个** CNN probe，不试 CLIP / SAM / ConvNeXt 等其它 backbone。

## 2. 对象（按**实测** `C_feat` 选取，不凭直觉）

- **MVTec 困难组（最低 3）**：`transistor 0.087`、`screw 0.111`、`zipper 0.154`
- **MVTec 对照**：`carpet 0.388`（最高）
- **VisA 困难组（最低 3）**：`pcb4 0.060`、`pcb2 0.081`、`pcb3 0.095`

共 **7 类**。shots {1, 4} × 3 splits。

**注**：VisA 最低三个都是 PCB 变体，类别同质性高，会削弱"非单类偶然"的论证力，
作为已知局限记录（见 §6）。

## 3. 测量

- 真实 downstream：image AUROC / pixel AUROC / AUPRO，evaluator 与全程一致
  （`pixel_metrics_binned(pro_limit=0.05)`）
- 机制量：`C_feat`（与 F1/F2 同定义）**以及** `sep_auc`（见 §4）
- 每个 representation 都用**同一套 bank draws**（`draw_images` 按对象名播种，
  与 representation 无关）

## 4. 关于跨 representation 比较 `C_feat` 的一个必要说明（**事先声明**）

`C_feat = median d(defect→bank) − median d(nearby normal→bank)` 的**量纲**
依赖 representation 自身的距离尺度。R1/R2 与 R0 的尺度不同，
**直接比较 `C_feat` 的绝对大小是尺度混淆的**。

因此机制判据用一个**尺度无关**的量：

> `sep_auc`：在**同一张图内**，用原始距离 `d(·→bank)` 区分
> 「缺陷 patch」与「邻近正常 patch」的 rank-AUC（1 张图 1 个值，取平均）。
> 1.0 = 完全可分，0.5 = 不可分。跨 representation 可直接比较。

`C_feat` 仍按原定义一并报告，用于与 F1/F2 衔接；**判决用 `sep_auc`**。

## 5. 判据（**四条同时考虑**）

设对某个 representation R 相对 R0 的增量：

```
M1-1  机制：sep_auc 在困难对象上提升
       mean(sep_auc_R - sep_auc_R0) over hard objects > 0
       且至少 4/6 个困难对象为正（不是单点）
M1-2  定位：AUPRO 在困难对象上平均 >= +2.0
M1-3  一致性：>= 5/7 个测试类别 AUPRO 方向为正
M1-4  守卫：image AUROC 平均 >= -0.3（不允许系统性退化）
```

对照组 `carpet` 报告但不计入 M1-1/M1-2。

## 6. 已知局限（事先声明）

1. VisA 三个困难类同为 PCB，类别同质性高。
2. R2 的 patch 网格与 DINO 不同（stride 8/16 vs 14），
   `sep_auc` 在两者上定义的"缺陷 patch / 邻近正常 patch"粒度不同；
   这是该指标可比较性的**残余弱点**，已知且不因此放宽判据。
3. 只跑 1/4-shot，不跑 8-shot。
4. 不训练任何东西；三个 representation 全部冻结。

## 7. 三个出口（跑之前定好）

| 结果 | 下一步方向 |
|---|---|
| **A**：R1（高分辨率 DINO）同时提升 `sep_auc` 与 AUPRO | **High-resolution / coarse-to-fine DINO anomaly localization** —— 不换表示族 |
| **B**：R2（CNN 局部）赢而 R1 不明显 | **DINO 语义分支 + 局部纹理分支**的 hybrid（先固定融合，不训练） |
| **C**：两者都救不了 | **停止模型开发**。整理成分析型论文，或另开需要训练/工业预训练的新项目。**不再试第 4 个 backbone。** |

## 8. 资产（待生成）

| 类型 | 内容 |
|---|---|
| 缓存 | `cache_m1_dino672/`、`cache_m1_wrn50/` |
| 脚本 | `cache_m1_repr.py`、`phase_m1_rescue.py`、`phase_m1_analysis.py` |
| 结果 | `metrics/phase_m1_*.csv` |
| 本文件 | `docs/PHASE_M1_PREREGISTRATION.md`（冻结，不得修改） |
