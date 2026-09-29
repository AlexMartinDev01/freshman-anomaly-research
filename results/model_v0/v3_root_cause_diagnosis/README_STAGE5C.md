# Stage 5C —— halo 路线的封口实验

> 27 类（15 MVTec + 12 VisA），38880 行。生成于 2026-09-29。
> 脚本 `experiments/model_v0/v3_stage5c_context.py`。这是**封口实验，不是继续探索**。

---

## 0. 边界（不得扩大）

**本包不含任何 defect 性能指标**（无 AUPRO / AUROC / GT / A5）。halo 尺寸只能由
normal-only 的特征保真度 + 预算算术决定。

`n_ctx` **未扩展到 33 以上**。selector 与 G1–G5 未被触碰。

---

## 1. Positive control：三层全部通过

`05c_full_cache_identity_all_layers.csv` —— 12 行（4 对象 × 3 层）

```
mid      (n=5)   cos_mean=1.000000  max_abs_diff=0.00000  PASS
midlate  (n=8)   cos_mean=1.000000  max_abs_diff=0.00000  PASS
final    (n=11)  cos_mean=1.000000  max_abs_diff=0.00000  PASS
```

Stage 5B 只验证了 `final` 一层，而 `mid`/`midlate` 恰恰是保真度受质疑的层。
现在全图 re-forward 对三层都逐位复现缓存，这一整类替代解释被排除。
脚本在此项失败时直接 `SystemExit`。

---

## 2. 文件

| 文件 | 粒度 | 关键列 |
|---|---|---|
| `05c_full_cache_identity_all_layers.csv` | 4 对象 × 3 层 | `cos_mean`, `max_abs_diff`, `pass_` |
| `05c_context_fidelity_fixed_core.csv` | 38880 行 | `cos_median`, `nn_spearman`, `d_full_median`, `d_halo_median`, `core_r`, `core_c` |
| `05c_context_fidelity_by_category.csv` | 27 类 × 15 | 按对象聚合 |
| `05c_context_fidelity_by_layer.csv` | 2 数据集 × 3 层 × 15 | 每层曲线 |
| `05c_agg3_fidelity.csv` | 27 类 × 15 = 405 | `agg3_spearman`（detector-consistent） |
| `05c_halo_decision.csv` | 15 | 判据表，含 5 种 agg3 口径与 2 种 retention 读法 |
| `05c_budget_decision_input.csv` | 1296 | retention 独立重算（未从任何 CSV 回读） |
| `05c_fidelity_vectors.npz` | 38880 × 25 | 原始 `dH`/`dF`，可离线重算任何统计量 |
| `V3_TRIGGER_PROFILE_K8S0.csv` | 27 | 只有 k8/s0 一个 cohort，**文件名如实说明** |

---

## 3. 相对 Stage 5B 改了什么

### A. 固定物理 core（主修正）

Stage 5/5B 里 crop 起点被裁剪：`r0 = min(max(0, ar - h), gh6 - n)`。因为 core 是
crop 的中心，**被比较的 5×5 core 会随 n 漂移**：

```
5B，48 行网格，第一个 anchor:  n=5 -> core 顶行 11
                               n=25 -> 11
                               n=29 -> 12
                               n=31 -> 13
                               n=33 -> 14      <- 漂移恰好落在曲线仍在上升的一段
```

Stage 5C 先按 `n_max=33` 选定 core 中心，使**任何 n 都不会发生裁剪**：

```
cr >= (n_max-1)//2 = 16          cr <= gh6 - (n_max+1)//2 = gh6 - 17
```

48 行网格上 core 中心取 `[21, 26]`，于是 15 个 halo 尺寸比较的是**逐位相同的 5×5 core**
（脚本内有 `assert r0 + k0 == cr - CORE // 2`，几何性质已用 10 种 grid 形状独立验证，
零违规）。

### B. 三层 positive control（见 §1）

### C. detector-consistent `agg3`（见 §5）

### 效果（n_ctx=33，中位数）

| 量 | Stage 5B（裁剪，混入漂移） | Stage 5C（固定 core） |
|---|---|---|
| cos_median | 0.831 | **0.903** |
| nn_spearman（per-layer 中位） | 0.293 | **0.454** |
| agg3 Spearman（pooled，27 类中位） | 0.300 | **0.467** |
| `d_full` / `d_halo` 中位 | 0.042 / 0.105 | 0.042 / 0.083 |

**同一个 agg3 代码跑 5B 向量得到 0.300，跑 5C 数据得到 0.467** —— 所以差异来自数据
（core 修复），不是统计方法。

---

## 4. 保真度 vs halo 尺寸

| n_ctx | 5 | 9 | 15 | 21 | 27 | 33 |
|---|---|---|---|---|---|---|
| cos_median | 0.412 | 0.646 | 0.798 | 0.852 | 0.882 | **0.903** |
| nn_spearman | 0.078 | 0.186 | 0.276 | 0.348 | 0.406 | **0.454** |

n=33 分层：`final` cos 0.936 / ρ 0.491；`mid` 0.890 / 0.443；`midlate` 0.894 / 0.379。

**曲线在 33 处仍未饱和。**

---

## 5. `agg3` 的定义

探测器消费 `agg_all3`：逐层 1-NN 距离 → 逐层 z-score → 三层取平均。保真度按同样方式打分：

```
每层用 FULL-REFERENCE 统计量 (mu_l, sd_l) 对 d_halo 与 d_full 做同一套 z-score
-> 三层取平均得到 agg_halo / agg_full
-> 二者的 Spearman
```

只报逐层 ρ 会留下"聚合之后会不会刚好恢复"的疑问，所以聚合被显式报告。

---

## 6. 预注册判据与结果

判据（在 5C 运行**之前**固定）：

```
存在 n <= 33 使得   median(agg3 Spearman) >= 0.50   且   trigger-mass retention >= 0.50
否则：停止 pure-halo 路线，不再扩大 halo，不查看 defect performance
```

结果：

```
n_ctx  agg3_pooled  retention(trigger-mass)
   5      0.133          0.921
  15      0.275          0.535
  21      0.364          0.314
  27      0.428          0.192
  33      0.467          0.159

=> 没有任何 n 同时满足两条。STOP。
```

### 稳健性：10 / 10 组合全部 NONE

`agg3` 的聚合口径与 retention 的读法都曾是活的歧义，所以两者都枚举：

```
agg3 口径:  pooled / anchor00 / anchor01 / anchor10 / anchor11
retention:  trigger-mass (Σmin(m,K)/Σm)  /  naive (每图保留比例的平均)
=> 5 × 2 = 10 种组合，全部 -> NONE
```

**最接近的 n 也差得很远**：trigger-mass 读法下最好的是 n=13（agg3 0.288，retention 0.603），
两条合计缺口 0.212；naive 读法下最好是 n=23（agg3 0.427，retention 0.505），缺口 0.073。
`agg3` 单独越线最早出现在 n=29（最好 anchor 0.511），而那时 retention 已跌到 0.182。

**结构是复合失败，不是"再调大一点就行"。**

### 关于 anchor 口径的一处修正（重要）

per-anchor 稳健性**必须按 anchor 序号**而非绝对坐标。VisA 各对象的有效 core 坐标不同
（`[22,28] / [23,30] / [29,42] / [31,46] …`），按坐标筛选会**只覆盖拥有该坐标的对象子集**
—— 例如 `(21,21)…(26,26)` 实际只有 MVTec 15 个对象。按坐标做会得到"per-anchor 超过 0.50、
判据翻转"的假结论；按序号重算后 per-anchor 在 n=33 是 0.437–0.528，与 pooled 的 0.467 同水平，
**没有任何 convention 翻过阈值**。

---

## 7. 必须写明的边界与对 Stage 5B 的修正

1. 被证伪的是**机制**（用局部 halo 重建全局上下文），**不是** selective refinement 方向。
   Stage 4 的 `context (C10−C00) = +28.34 AUPRO` 说明"补回上下文"本身是对的。

2. **Stage 5B"中层 global-context 依赖特别强"的说法要撤回。** 固定 core 后：

   | 层 | ρ@n=33 (5B) | ρ@n=33 (5C) | Δ |
   |---|---|---|---|
   | mid | 0.191 | 0.443 | +0.252 |
   | midlate | 0.247 | 0.379 | +0.132 |
   | final | 0.454 | 0.491 | +0.038 |

   层间差异大幅收窄，5B 的层间故事主要是 confound 造成的。

3. **但本实验不能把两个因素分开**：5C 同时改变了 (a) core 不再随 n 漂移、
   (b) core 位置不同且最大 crop 不再贴图像边缘（5B 的 n=33 crop 从第 0 行开始，
   5C 的是第 5–37 行）。哪个是主因**尚未分离**。若需要分离，最小实验是把 core 中心
   取在允许范围下界 `cr = 16`（此时 n=33 的 crop 恰好贴边）—— 但那属于扩大 Stage 5，
   **未运行**。

4. 本包**不含 selector 信息**：`A3_corrected` vs `A4_corrected` 的 G3 依然未决。

---

## 8. 复现

```bash
# 全量（~13 分钟，27 类）
OMP_NUM_THREADS=2 python experiments/model_v0/v3_stage5c_context.py

# 只重算判据表（秒级，读 npz + diagnostics，不需要模型）
S5C_DECIDE_ONLY=1 python experiments/model_v0/v3_stage5c_context.py
```

`05c_budget_decision_input.csv` 的 retention 是**从冻结的 `diagnostics.json` 独立重算**的，
没有从任何已有 CSV 回读。存储列 `mean_retained_triggered` 经 1296 个 (cell, n_ctx) 组合核对，
确认就是 `Σmin(m,K)/Σm`（最大偏差 1.1e-16，浮点噪声级），**不是**每图保留比例的平均
—— 两者在阈值附近差别很大（n=15：0.535 vs 0.743）。
