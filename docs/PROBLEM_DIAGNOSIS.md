# Problem Diagnosis Report（草稿 v0.9，跨方法表格待 stress test 完成后定稿）

> 课题：少量正常样本条件下，面向真实工业环境分布变化的鲁棒视觉异常检测
> 基线：AnomalyDINO（frozen DINOv2 ViT-S/14, 448, 1NN cosine, top-1% mean 图像分数）
> 数据：MVTec AD 2 TEST_pub（本地 GT 可用）

## 1. 基线迁移结果（AD2 TEST_pub，8 类 × 1/2/4-shot × 3 seeds）

| Shot | img AUROC | img F1@μ+3σ | px AUROC | AU-PRO@0.05 |
|---:|---:|---:|---:|---:|
| 1 | 66.6 ± 17.3 | 28.7 | 86.9 | 35.3 |
| 2 | 68.9 ± 17.1 | 36.9 | 87.9 | 37.0 |
| 4 | 70.5 ± 17.4 | 41.9 | 88.3 | 38.6 |

（注：AD2 论文 full-shot 基线为不同方法/不同设置，不作直接同条件对照。）

## 2. 证据链

### E1. Aggregation sweep（实验②）
max / top0.1% / top1% / top5% / p99 / p99.9 之间 image AUROC 相差 ≤3 点。
**结论：图像级崩坏不是聚合问题。**

### E2. Scene vs Lighting 分解（实验③，good 图分数双因素 ANOVA）
| 类型 | 类别 | scene 方差占比 | 备注 |
|---|---|---:|---|
| scene 主导 | can 98.3%, fruit_jelly 93.4%, vial 92.4%, rice 78.6%, fabric 60.3% | | 正常变化是主噪声源 |
| light 主导 | walnuts 68.3%, sheet_metal 66.4%, wallplugs 52.0% | | 光照 shift 敏感 |
| 崩坏类 | can / wallplugs | | Cohen's d（缺陷效应量）为负（−0.43 / −0.68） |

### E3. 扩展 shot curve（实验①，can/wallplugs/vial/sheet_metal，1→full）
- **vial（覆盖型）**：img AUROC 67.3→90.5，AU-PRO 62.4→75.8，随 shot 完全恢复
- **can/wallplugs（表示型）**：full-shot 也只有 52.3 / 42.2 AUROC——加参考图救不回来
- **sheet_metal（饱和型）**：1-shot 即 82.9，全程平坦

### E4. Patch-level GT contrast（证据闭环）
| 类别 | P(干净区最大响应 > 缺陷区最大响应) | 缺陷中位大小(patch) |
|---|---:|---:|
| can | **1.000** | 1 |
| wallplugs | **0.944** | 2.5 |
| fabric | 0.648 | 1 |
| rice | 0.589 | 3 |
| 健康四类 | 0.21~0.33 | 14~36 |

**机制**：像素级排序有效（缺陷 patch 均分 > 干净 patch 均分），但缺陷只有 1~2 个 patch，
图像分数取 top-1%（~22 patch）→ 正常变化的噪声尖峰淹没单个真缺陷 patch。
448 分辨率下 can 有 48/90 张异常图的缺陷 < 一个 patch 的 10% 面积。

### E5. 跨方法 stress test（进行中）
SubspaceAD（PCA 子空间残差，完全不同打分）在 can 1-shot：img AUROC 36.1 —— 同样崩坏。
**初步结论：这是 DINOv2 特征 + few-shot 范式的共性问题，不是 AnomalyDINO 特例。**

## 3. 问题定义（收敛）

> **在少量正常样本下，如何让异常分数区分"场景级正常变化"与"真实缺陷"——
> 尤其是在缺陷只有 1~2 个 patch、而正常变化产生更高响应尖峰的情况下。**

## 4. 模型设计空间（按证据排序）

1. **正常变化抑制 / 前景聚焦**（针对 can/wallplugs 型：背景与印刷变化是噪声源）
2. **参考库表示方式**（针对 vial 型：prototype/coverage）
3. **Score calibration 与阈值稳健性**（针对光照敏感类 + TEST_priv,mix）
4. ~~Aggregation 创新~~（已排除）
5. **有效分辨率**（can 的缺陷 <1 patch：multi-scale / 更高输入分辨率——注意官方指出分辨率-性能-成本权衡）

## 5. 待办

- [ ] SubspaceAD 4 类 × 3 shots × 3 seeds 完整表（进行中）
- [ ] feature 层 PCA/UMAP 可视化（scene/light/defect 着色）
- [ ] TEST_priv / TEST_priv,mix 出图 + benchmark server 提交（需账号）
- [ ] 报告定稿
