# V3.1-F0 —— resolution-increment surrogate 的 support-normal 可行性

> 27 类、157464 个探针样本。生成于 2026-09-29。
> 脚本 `experiments/v3_1_f0_resolution_increment/f0_resolution_increment.py`。
> **support-normal only**：无 test anomaly、无 GT、无 AUPRO/AUROC、无 A5、无 selector、无 324-cell。

---

## 0. 判决

```
NO-GO —— 三个尺度全部未通过预注册 gate。
按预注册规则：停止 V3 architecture exploration。
不扩展 n，不增加复杂模型，不查看 defect performance。
```

| n | median B0 | median B1 | median B2 | gain(B2−B1) | mvtec | visa |
|---|---|---|---|---|---|---|
| 9 | 0.246 | **0.690** | 0.384 | **−0.302** | −0.208 | −0.365 |
| 13 | 0.234 | **0.690** | 0.435 | **−0.272** | −0.161 | −0.334 |
| 15 | 0.253 | **0.690** | 0.473 | **−0.233** | −0.136 | −0.295 |

```
G1 median rho(B2) >= 0.50        FAIL   (0.384 / 0.435 / 0.473)
G2 median [rho(B2)-rho(B1)] >= 0.05  FAIL (gain 为负)
G3 MVTec 与 VisA 的 B2-B1 均 > 0   FAIL   (两边都为负)
G4 B2 > B0                        PASS
```

**加进 resolution increment 之后，B2 反而比只用 global448 的 B1 差 0.23–0.30。**

---

## 1. 正确性

```
positive control  12/12 通过（2 对象 x 2 域 x 3 层）
  448 与 672 两个域都逐位复现各自的 cache
  cos_mean = 1.000000, max_abs_diff = 0.00000
selftest  local_weights == cal.pool_weights（40 个随机 448 patch，逐权重对齐）
```

448 侧的 resize 是本脚本自行重建的（`cs.build()` 只提供 672 的
`transforms.Resize`）。若它与构建 M1 448 cache 时的预处理不一致，**所有 448 crop
距离都会被污染，而那种污染看起来恰好会像"448 局部裁剪无用"** —— 正是本实验要检验的
假设的方向。positive control 排除了这一整类解释。

---

## 2. 设计

同一物理位置、每层：

```
d_G448  full448 raw 1NN （全局上下文，448 分辨率）
d_L448  crop448 raw 1NN （局部上下文，448 分辨率）
d_L672  crop672 raw 1NN （局部上下文，672 分辨率）
d_G672  full672 raw 1NN （support-normal 参考）
```

```
Delta_l     = d_L672_l - phi_l(d_L448_l)
dhat_G672_l = psi_l(d_G448_l) + Delta_l
```

φ、ψ 为**仿射**，只用 support-normal 拟合，方向固定（把 448 域量映射到 672 域），
**不反向**以迁就指标。无 MLP / attention / adapter / 非线性搜索。

**B0** = d_L672 ｜ **B1** = ψ(d_G448) ｜ **B2** = ψ(d_G448) + Δ

**bank**：leave-one-image-out（探针所在图不进自己的 bank），无 self-match。

**交叉拟合**：φ/ψ 按 support image 奇偶做 2 折（在一半 support 上拟合、另一半上评分），
避免"同批数据既拟合又评分"让 B1/B2 相对 B0 获得不对称优势。

**探针**：3×3 的 448 patch（= 5×5 的 672 token），每图 3×3 个位置，全部无裁剪。
672→448 的重叠权重走冻结的 `cal._span` / `cal._overlap`，并与 `cal.pool_weights` 逐权重对齐。

### 一处必须说明的取舍：物理尺度无法精确相等

448 token 与 672 token 不可通约（1 个 448 token = 1.5 个 672 token），两个 crop 的物理
尺度**无法相等**。448 侧取"能对称容纳共享探针的最近奇数窗"：

| n (672) | N448 | 物理尺度 448 vs 672 | 比 |
|---|---|---|---|
| 9 | 7 | 98px vs 126px | 1.167 |
| 13 | 9 | 126px vs 182px | 1.038 |
| 15 | 11 | 154px vs 210px | 1.100 |

每行样本都记录 `physical_px_448` / `physical_px_672`，因此这个不匹配是**可审计的**，
不是被抹掉的。这是 Δ 与目标无关的原因之一（见 §3）。

---

## 3. 失败机制（本节比判决本身更有用）

Δ 在三个尺度上都没能提供增量。逐层分解（n=15）：

| layer | std(target) | std(B1) | std(Δ) | std(Δ)/std(target) | **corr(Δ, target)** | corr(B1, target) |
|---|---|---|---|---|---|---|
| mid | 0.0339 | 0.0310 | 0.0267 | 0.787 | **−0.003** | 0.914 |
| midlate | 0.0342 | 0.0297 | 0.0317 | 0.928 | **+0.026** | 0.867 |
| final | 0.0308 | 0.0264 | 0.0355 | **1.153** | **+0.178** | 0.856 |

**Δ 与它要帮助预测的目标几乎不相关（corr ≈ 0），而它的量级却和目标相当甚至更大
（0.79–1.15×）。** 把一个"和目标一样大、但和目标无关"的项加到 B1 上，必然摧毁排序 ——
这就是 B2 从 0.69 掉到 0.47 的全部原因。

**为什么 Δ 没有信息？** 因为两个 crop 给出的距离本来就高度相似：

```
corr(d_L448, d_L672)      = 0.833
median |d_L672 - d_L448|  = 0.031    vs    median d_L672 = 0.167
```

这正是 diD 的**前提**（两个 crop 共享同一种 context bias），但它同时意味着**二者之差
几乎只剩噪声**。前提成立，推论却因此不成立：**共享 bias 让差值失去了分辨率信号。**

φ 的斜率 0.70–0.79（不是 1），说明 crop672 距离确实相对 crop448 有所压缩，但仿射之后
剩下的残差与 full672 无关。

---

## 4. 一条必须谨慎对待的观察：B1 = 0.690

**B1 在所有三个尺度上都是 0.690**，因为 B1 只用 `d_G448` 与 `d_G672`，而这两者**不依赖 n**
（它们是整图距离）。分数据集：MVTec 0.760，VisA 0.686；27 类跨度 0.288–0.935。

**但我不能把它写成"发现"**，原因有三：

1. **B1 不是本实验要检验的假设。** 预注册的 gate 针对的是 B2（带 resolution increment
   的代理）。B1 未过 G2/G3（gain 为负），GO 判定与它无关。
2. **ψ 是单调仿射，在单个交叉拟合 fold 内无法改变 Spearman。** 实测
   `max|B1 − raw Spearman(d_G448,d_G672)| = 0.078`（非 0，因为两个 fold 的系数不同、
   跨 fold 池化会改变排名）。所以 B1 ≈ 0.69 基本就是 **global448 与 global672 两个表示的
   原始排序一致度**，不是一个被"拟合出来"的量。
3. **它是两个表示之间的排序一致度，不是性能指标**，也没有触及任何缺陷标签。
   它不说明 448 分辨率"够用"，只说明两个分辨率下的近邻几何**高度重合**。

若这条线索值得追，它需要**自己的一套 pre-registration**，不能用 F0 的结论外推。

---

## 5. 边界（不得扩大）

- 被判 NO-GO 的是 **resolution-increment 这一具体构造**（`Δ = d_L672 − φ(d_L448)`，
  再 `ψ(d_G448) + Δ`）。不是"全局上下文 + 局部增量"这个想法的全部可能实现。
- 但按预注册规则，**本轮到此停止**：不扩展 n、不加复杂模型、不看 defect performance。
- 本包**不含任何 defect 性能指标**，不含 selector 信息。

---

## 6. 文件

| 文件 | 内容 |
|---|---|
| `f0_positive_control.csv` | 12 行：2 对象 × 2 域 × 3 层，cos / maxdiff |
| `f0_probe_distances.csv` | 157464 行：每个探针的 `d_G448 / d_L448 / d_L672 / d_G672`，含 `physical_px_448/672`、`fold` |
| `f0_per_layer_metrics.csv` | 729 行 = 27 × 3 尺度 × 3 层 × 3 arms，**原始尺度**的 Spearman / MAE / rel_error |
| `f0_agg3_metrics.csv` | 243 行 = 27 × 3 尺度 × 3 arms，**agg3（z 尺度）** |
| `f0_gate_decision.csv` | 3 行：每个尺度的四条 gate 与 GO |

注意 MAE 的量纲：`f0_agg3_metrics.csv` 的 MAE 在 **z 尺度**上（B0 因尺度完全未对齐，
MAE≈8），`f0_per_layer_metrics.csv` 的 MAE 在**原始 672 距离尺度**上（B0≈0.13, B1≈0.01,
B2≈0.02）。两处不可混读。

---

## 7. 复现

```bash
OMP_NUM_THREADS=2 python experiments/v3_1_f0_resolution_increment/f0_resolution_increment.py
```

约 20 分钟（含两次 positive control）。脚本在 positive control 失败或探针越界时直接
`SystemExit`，不会产出无法解释的结果。
