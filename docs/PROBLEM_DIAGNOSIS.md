# Problem Diagnosis Report（v1.0 · 问题定义封版）

> 课题：少量正常样本条件下，面向真实工业环境分布变化的鲁棒视觉异常检测
> 基线：AnomalyDINO（frozen DINOv2 ViT-S/14, 448, 1NN cosine, top-1% mean 图像分数）
> 数据：MVTec AD 2 TEST_pub（本地 GT 可用）
> 状态：**问题发现阶段已封版**。E1–E6 证据链闭环，进入模型设计阶段。

---

## 0. 一句话结论

> 在少量正常样本下，**正常变化在特征空间中产生的响应，其尾部已经进入真实缺陷的响应区间**：
> 正常分布 p99 与缺陷响应同量级甚至更高（wallplugs 上 99.1% 的真实缺陷 patch
> 比正常 p99 还安静）；而图像分数取 top-1% 的尾部统计，恰好采样在这个混淆区上。
>
> 关键在于——**崩坏的不是缺陷信号，是正常分布的尾巴**：缺陷响应幅度跨六个类别
> 几乎恒定（1.38×），正常尾部却变化 3.14×，且 `mean_defect` 与性能完全不相关（p=0.79）。
> 因此设计目标是**压正常尾部**，而非放大缺陷响应。
>
> 该失效在两种完全不同的 DINOv2-based 范式下均可复现（E5），
> 并非 AnomalyDINO 的实现特例。支撑证据：E1–E6。

---

## 1. 基线迁移结果（AD2 TEST_pub，8 类 × 1/2/4-shot × 3 seeds）

| Shot | img AUROC | img F1@μ+3σ | px AUROC | AU-PRO@0.05 |
|---:|---:|---:|---:|---:|
| 1 | 66.6 ± 17.3 | 28.7 | 86.9 | 35.3 |
| 2 | 68.9 ± 17.1 | 36.9 | 87.9 | 37.0 |
| 4 | 70.5 ± 17.4 | 41.9 | 88.3 | 38.6 |

（注：AD2 论文 full-shot 基线为不同方法/不同设置，不作直接同条件对照。）

**分类别崩坏程度**（1-shot img AUROC）：`walnuts` / `fruit_jelly` / `rice` / `fabric` 尚可，
`can` / `wallplugs` 低于随机水平，`vial` 随 shot 恢复但 1-shot 不足。

---

## 2. 证据链

### E1. Aggregation sweep（实验②）—— 排除聚合

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

### E4. Patch-level GT contrast（证据闭环，8 类）

对每张异常图，把 GT 映射到模型实际使用的 patch 网格，比较缺陷 patch 与干净 patch 的分数：

| 类别 | P(干净区最大响应 > 缺陷区最大响应) | 缺陷中位大小(patch) | patch AUROC | 缺陷均分 | 干净均分 |
|---|---:|---:|---:|---:|---:|
| can | **1.000** | 1.0 | 0.832 | 0.287 | 0.150 |
| wallplugs | **0.944** | 2.5 | 0.655 | 0.180 | 0.152 |
| fabric | 0.648 | 1.0 | 0.840 | 0.256 | 0.124 |
| rice | 0.589 | 3.0 | 0.806 | 0.168 | 0.072 |
| walnuts | 0.333 | 33.0 | 0.905 | 0.302 | 0.135 |
| fruit_jelly | 0.321 | 14.5 | 0.908 | 0.316 | 0.082 |
| vial | 0.267 | 22.0 | 0.952 | 0.308 | 0.080 |
| sheet_metal | 0.211 | 36.0 | 0.908 | 0.213 | 0.090 |

**机制**：像素级排序在**图内**有效（patch AUROC 0.66–0.95，缺陷 patch 均分 > 干净 patch 均分），
但缺陷只有 1~2 个 patch，图像分数取 top-1%（~22 patch）→ 正常变化的噪声尖峰淹没单个真缺陷 patch。
448 分辨率下 can 有 48/90 张异常图的缺陷 < 一个 patch 的 10% 面积。

### E5. 跨方法 stress test（封版）—— 排除实现特例

SubspaceAD（PCA 子空间残差，完全不同的打分函数；DINOv2-**giant**，7 层特征）
在同样 4 类 × 3 shots × 3 seeds 上复现。**36/36 组合全部完成**（`image_res=448`，与 AnomalyDINO 匹配）。

**img AUROC（3 seeds 均值）**

| shot | 方法 | can | wallplugs | vial | sheet_metal | 4类均值 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | AnomalyDINO | 44.7 | 38.0 | 67.3 | 82.9 | 58.2 |
| 1 | SubspaceAD | 44.1 | 45.6 | 74.6 | 79.4 | 60.9 |
| 2 | AnomalyDINO | 50.8 | 39.8 | 75.8 | 83.9 | 62.6 |
| 2 | SubspaceAD | 41.1 | 44.3 | 74.9 | 79.7 | 60.0 |
| 4 | AnomalyDINO | 50.4 | 42.4 | 85.6 | 83.6 | 65.5 |
| 4 | SubspaceAD | 47.4 | 47.3 | 84.1 | 80.1 | 64.8 |

**AU-PRO@0.05**（像素级定位）

| shot | 方法 | can | wallplugs | vial | sheet_metal |
|---:|---|---:|---:|---:|---:|
| 1 | AnomalyDINO | 9.9 | 2.2 | 62.4 | 48.7 |
| 1 | SubspaceAD | 11.9 | 3.9 | 62.3 | 34.6 |
| 4 | AnomalyDINO | 12.3 | 3.8 | 66.4 | 51.0 |
| 4 | SubspaceAD | 11.2 | 3.9 | 71.1 | 34.0 |

**结论（三条独立证据）**

1. **逐类失败模式一致**：两种方法给出相同的类别排序与相同的 shot 响应形状
   —— `vial` 随 shot 恢复、`can`/`wallplugs` 始终 <50、`sheet_metal` 全程平坦。
2. **分辨率不敏感**：早期 `res672` 跑在 shot 1/2 上同样有 3 seeds
   —— can 44.3 / 41.0，wallplugs 46.0 / 48.0，vial 88.6 / 88.0，sheet_metal 80.4 / 81.2，
   与 `res448` 几乎一致。**（`res672` 的 shot-4 只有 1 个 seed，不引用。）**
   → 不是有效分辨率问题，是表示问题。
3. **backbone 规模不救**：ViT-g/14 + 7 层特征 与 ViT-S/14 单层 同样崩坏。

> **判定：这是 DINOv2 特征 + few-shot 范式的共性问题，不是 AnomalyDINO 的实现问题。**

### E6. Feature-level 机制诊断（封版）

在同一套 frozen DINOv2-S/14 特征空间、同一个 1-shot memory bank（`train/good[0]`, seed 0）下，
测量三类扰动的每-patch 1NN 余弦距离（**即检测器实际阈值的那个量**）：

- `D_scene`：不同场景、同光照
- `D_light`：同场景、不同光照
- `D_defect`：异常图的缺陷 patch（GT>10%）
- 参照组 `floor`：**同场景 + 同光照**的 good 图（= 纯拍摄噪声地板），以及 clean patch

> 注：scene id 在 train/ 与 test_public/ 之间一一对应已验证（同一物理场景），
> 但两边是**不同次拍摄**，所以 floor 组不是 ~0，而是"只换一次拍摄"的真实代价。

在同一套 frozen DINOv2-S/14 特征空间、同一个 1-shot memory bank（`train/good[0]`, seed 0）下，
测量每-patch 1NN 余弦距离（**即检测器实际阈值的那个量**）在六类对象上的分布：

- `D_scene`：不同场景、同光照 ｜ `D_light`：同场景、不同光照
- `D_defect`：异常图的缺陷 patch（GT>10%）
- `floor`：**同场景 + 同光照**的 good 图 = 纯拍摄噪声地板
- `正常 p99`：所有 good 图 patch 距离的 99 分位

（距离为该组全部 patch 的均值；`缺陷 < 正常 p99` = 真实缺陷 patch 中低于**全部正常 patch**
99 分位的比例，即按正常分布 p99 设阈值时的漏检比例。）

| 类别 | 类型 | floor | D_scene | D_light | D_defect | 正常 p99 | **缺陷 < 正常 p99** | 1-shot img AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| wallplugs | 崩坏 | 0.152 | 0.140 | 0.183 | 0.192 | 0.362 | **99.1%** | 38.0 |
| can | 崩坏 | 0.202 | 0.166 | 0.209 | 0.265 | 0.461 | **79.6%** | 44.7 |
| vial | 覆盖型 | 0.084 | 0.064 | 0.101 | 0.237 | 0.317 | **77.5%** | 67.3 |
| fabric | 健康 | 0.125 | 0.117 | 0.127 | 0.255 | 0.257 | **59.4%** | 59.5 |
| sheet_metal | 饱和型 | 0.078 | 0.072 | 0.092 | 0.213 | 0.173 | **47.6%** | 82.9 |
| rice | 健康 | 0.069 | 0.068 | 0.072 | 0.229 | 0.147 | **40.6%** | 79.0 |

**三条结论**

1. **机制成立**：`缺陷 < 正常 p99` 与 1-shot img AUROC 强负相关（Spearman **−0.886**, p=0.019）。
   wallplugs 上 **99.1% 的真实缺陷 patch 比正常分布的 p99 还安静** ——
   阈值落在哪里都救不回来。
2. **崩坏的不是缺陷信号，是正常分布的尾巴**：
   缺陷响应幅度在六类上几乎恒定（0.192–0.265，仅 **1.38×** 差异），
   而正常尾部 p99 跨类别变了 **3.14×**（0.147 rice → 0.461 can）。
   `mean_defect` 与性能**不相关**（Spearman −0.143, p=0.79），
   而 `floor` / `mean_light` / `正常 p99` 都强相关（−0.886 / −0.886 / −0.829）。
   → **要压的是正常分布的尾部，不是放大缺陷响应。**
3. **图像分数的 top-1% 恰好采样在混淆区**：`can` 的图内 patch 排序其实有效
   （within-image AUROC 0.832，缺陷 patch 均分 > 干净 patch 均分），
   但图像分数取 top-1%（~22 patch），正常变化的噪声尖峰把 1~2 个真缺陷 patch 淹没
   —— 与 E4 的 `P(干净区最大响应 > 缺陷区最大响应)=1.000` 完全一致。

**方法学校验**：仅用特征层每-patch 距离重算图像级 AUROC，与检测器实测值**排序完全一致**
（Spearman **+1.000**），说明该诊断忠实复现了检测器行为，不是另一个独立指标的巧合。

---

## 3. 问题定义（封版）

> **在少量正常样本条件下，正常变化（场景、光照、乃至同场景重拍的拍摄噪声）
> 在视觉基础模型特征空间中产生的响应，其尾部已经进入真实缺陷的响应区间——
> 正常分布的 p99 与缺陷响应同量级甚至更高。由于图像分数取 top-1% 的尾部统计，
> 恰好采样在这个混淆区上，导致正常样本被误判为缺陷、真实缺陷被淹没。
> 混淆的严重程度由正常分布尾部的长度决定，与缺陷信号的强弱无关。**

该定义由 E1–E6 六条证据共同支撑：

- **E5** 证明失效跨方法复现（不是实现问题）
- **E6** 给出特征空间层面的直接机制：缺陷响应跨类别几乎恒定（1.38×），
  而正常尾部变化 3.14×；`mean_defect` 与性能不相关（p=0.79）

### 3.1 正常尾部的两个来源（决定能否靠加样本解决）

E6 显示两类崩坏**归约为同一个量**——正常分布的尾部长度，区别只在尾部的**成因**：

| 尾部来源 | 代表类别 | 证据 | 加参考图能否缓解 |
|---|---|---|---|
| **场景 / 光照 / 拍摄变化**，不可被参考库覆盖 | can, wallplugs | E3：full-shot 仍只有 52.3 / 42.2；E6：floor 0.152–0.202，D_light ≈ D_defect | ❌ 无效 |
| **参考库未覆盖的外观模式** | vial | E3：67.3 → 90.5（full-shot）；E6：floor 低（0.084）但正常 p99 高（0.317） | ✅ 完全恢复 |

**统一表述**：两者都是"正常分布尾部进入缺陷响应区间"，因此
**统一的设计目标是压缩正常尾部**；而尾部成因决定了是否需要额外的表示层手段
（原型/多模态覆盖）还是纯粹的特征层去噪。

把这两类混为一谈会导致选错方法，模型设计阶段必须分别对待。

---

## 4. 模型设计空间（按证据排序）

**统一目标：压缩正常分布的尾部**（E6：缺陷信号跨类别恒定 1.38×，正常尾部变化 3.14×，
且 `mean_defect` 与性能不相关）。在此目标下按可操作性排序：

1. **正常变化抑制 / 前景聚焦**（针对"场景/光照/拍摄"来源的尾部）
   —— 背景、印刷、光照、拍摄噪声都是噪声源；这是 can/wallplugs 的唯一出路
2. **参考库表示方式**（针对"覆盖不足"来源的尾部：prototype / 多模态覆盖 / 子空间）
   —— E3 证明这一类靠增加参考样本即可完全恢复（vial 67.3 → 90.5）
3. **Score calibration 与阈值稳健性**（针对光照敏感类 + TEST_priv,mix）
4. **特征层解耦 / adapter**：在 frozen backbone 上学习 defect-sensitive 方向，
   而非直接使用原始 patch 距离

### 已排除的方向

- ~~Aggregation 创新~~（E1 证明无差异）
- ~~"few-shot 样本太少"作为唯一解释~~（E3：can/wallplugs full-shot 仍崩坏）
- ~~"1NN 匹配有问题"~~（E5：完全不同的 PCA 子空间残差同样崩坏）
- ~~AnomalyDINO 实现特例~~（E5）
- ~~有效分辨率不足~~（E5：448 → 672 的 can 为 44.1 → 44.3，几乎无变化；
  E6 亦显示瓶颈是正常尾部而非缺陷可分辨性。
  —— 注意这不否定 E4 的"can 有 48/90 张缺陷 <1 patch"，只是说明
  单纯提高输入分辨率不解决该问题）

---

## 5. 待办（**非阻塞**，不进入关键路径）

- [ ] `TEST_priv` / `TEST_priv,mix` 出图 + benchmark server 提交（需账号）
- [ ] 第三个 baseline（可选，用于加强 E5 的外部效度）
- [ ] 把 `feature_patchdist_*.csv` 接入后续模型实验的特征分析工具链
- [ ] 报告定稿为论文 related-work / problem-statement 章节

---

## 6. 产物索引

| 产物 | 路径 |
|---|---|
| 跨方法表 | `results/mvtec_ad2/metrics/cross_method_res448.csv` |
| SubspaceAD 全量结果 | `results/mvtec_ad2/metrics/stress_subspacead_{runs,summary}.csv` |
| 特征层诊断 | `results/mvtec_ad2/metrics/feature_diagnostics.csv` |
| 每图每组距离 | `results/mvtec_ad2/metrics/feature_patchdist_<obj>.csv` |
| Patch GT 对比 | `results/mvtec_ad2/metrics/patch_contrast.csv` |
| 图 | `results/mvtec_ad2/figures/feature_{box,pca}_<obj>.png` |

> `results/` 与 `third_party/` 默认不入 git（体积原因）。
> 报告引用的**小体积汇总表**需用 `git add -f` 显式纳入。

### 复现命令

```powershell
$PY = "E:\work\freshman\.venv\Scripts\python.exe"

# E5：SubspaceAD 跨方法 stress test（36 组合，逐 (shot,seed,object) 子进程，含重试）
& $PY experiments\mvtec_ad2\run_stress_subspacead.py
& $PY experiments\mvtec_ad2\aggregate_stress_subspacead.py     # -> 跨方法表

# E6：特征层机制诊断（6 类，含 fabric/rice 健康对照；~10 min on RTX 5060）
& $PY experiments\mvtec_ad2\feature_diagnostics.py can wallplugs vial sheet_metal fabric rice

# 基线汇总（命中缓存，快）
& $PY experiments\mvtec_ad2\eval_ad2_public.py
```
