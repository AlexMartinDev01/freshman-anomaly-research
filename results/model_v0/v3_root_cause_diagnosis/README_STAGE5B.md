# Stage 5B —— support-normal context fidelity + token-budget feasibility

> 覆盖 27 个类别（15 MVTec + 12 VisA），全部 324 个 cell 的预算算术。
> 生成时间 2026-09-29。脚本 `experiments/model_v0/v3_stage5b_context.py`。

---

## 0. 这个包回答什么，不回答什么

**回答**：不使用整张 672 前向时，局部 crop 的中心 5×5 token 要带多大的 halo，
其表征才能与 full-672 在同物理位置上的 token 对齐？

**不回答**：这个方向能不能 work。

**本包内不含任何 defect 性能指标**（没有 AUPRO / px_AUROC / img_AUROC，也不含
test anomaly、GT mask、A5）。这是刻意的：**halo 大小必须由 normal-only 的
特征保真度加预算算术决定，绝不允许用缺陷性能来挑**。A0/A2/A3/A4/A5 的对比属于
V3.1 的新 324-cell，不在本包范围内。

---

## 1. Positive control（先看这个）

`05b_full_cache_identity.csv`

```
mvtec/bottle    cos_mean=1.000000  max_abs_diff=0.00000  PASS
mvtec/capsule   cos_mean=1.000000  max_abs_diff=0.00000  PASS
visa/pcb2       cos_mean=1.000000  max_abs_diff=0.00000  PASS
visa/macaroni1  cos_mean=1.000000  max_abs_diff=0.00000  PASS
```

全图重新 forward 逐位复现 `cache_m1_dino672`。**这一步成立，下面所有 fidelity
数字才可解释** —— 它排除了 preprocessing / resize / checkpoint / 缓存不一致这一
整类替代解释。脚本在此项失败时直接 `SystemExit`，不会写出误导性的 fidelity 表。

---

## 2. 文件怎么读

| 文件 | 粒度 | 关键列 |
|---|---|---|
| `05b_full_cache_identity.csv` | 4 对象 | `cos_mean`, `max_abs_diff`, `pass_` |
| `05b_context_fidelity_corrected.csv` | 逐行：27 类 × 8 图 × 4 anchor × 15 n_ctx × 3 层 = **38880 行** | `cos_median`, `cos_p10`, `feature_rel_error`, `d_full_median`, `d_halo_median`, `nn_spearman` |
| `05b_context_fidelity_by_category.csv` | 27 类 × 15 n_ctx = 405 行 | 同上，按对象聚合 |
| `05b_context_fidelity_by_layer.csv` | 2 数据集 × 3 层 × 15 n_ctx = 90 行 | 每层单独的保真度曲线 |
| `05b_budget_all324.csv` | 324 cell × 15 n_ctx = 4860 行 | `B`, `gh672`, `gw672`, `Kmax`, `pct_capped_triggered`, `mean_retained_triggered` |
| `V3_TRIGGER_PROFILE_ALL324.csv` | 324 cell | `n_triggered`, `m_mean`, `m_max` |
| `05b_fidelity_vectors.npz` | 38880 × 25 | 原始 `dH` / `dF` 向量，可在不重跑 sweep 的前提下重算任何统计量 |

`nn_spearman` = 在每个 (dataset, object, n_ctx, layer) 分组内，把该组全部
support 图 × anchor × core token 的 `d_halo` 与 `d_full` 池化后算的 Spearman。

---

## 3. 与 Stage 5 的差别：修掉的四个 bug

Stage 5（`05_context_fidelity_support.csv`）的结论**方向正确但证据链无效**，
本包把它修好后重跑：

1. **self-match**。Stage 5 的 bank 就是那 k 张 support 图，查询的 token 又来自
   其中一张，1-NN 直接找到它自己 → `d_full ≈ 0` → 相对误差爆到 10⁵–10⁶，
   Spearman 是对一个近似常向量算的，无意义。
   Stage 5B 让 halo 与 full **共用同一个** eligible bank，并禁掉查询图自己的
   3×3 物理邻域（addendum 2 已冻结的规则）。
   **中途还修了第二处**：`base` 一度传成 0（而不是 `si * n_tok`），那样只有
   第 0 张图被正确排除、其余 7/8 的样本 self-match 依旧 —— 已定点验证
   query token 必在禁选集内、且禁选列全部落在查询图自己的 bank 块内。
2. **无 positive control**。见 §1。
3. **Spearman 按 25-token core 单独算**，现在按 (dataset, object, n_ctx, layer) 池化。
4. **只有 `all3`，没有分层**；trigger profile 被重复 15 次。均已修正。

**另外修掉一个预算表的静默错误**：`B = 0.5·gh672·gw672` 原先由
`ceil(1.5 × 448 grid)` 推出，但 448 缓存里根本没有 `grids`（永远走 `(32,32)`
兜底），而 VisA 的 672 test grid 是 `(48,52)…(48,78)`，第二维不是 3 的倍数，
**根本不是某个整数的 1.5 倍**。12 个 VisA 对象的 `B` 因此被低估最多 33%
（`K_max` 偏小、cap rate 偏高）。现在直接读 672 缓存的 `grids`。
`05b_budget_all324.csv` 的 `B` 落在 **1152（MVTec）– 1872（VisA pcb3）**。

---

## 4. 结果

### 4.1 保真度 vs halo 尺寸（全 38880 行取中位数）

| n_ctx | cos_median | cos_p10 | feature_rel_error | nn_spearman |
|---|---|---|---|---|
| 5 | 0.399 | 0.247 | 1.006 | 0.132 |
| 7 | 0.505 | 0.355 | 0.924 | 0.196 |
| 11 | 0.610 | 0.459 | 0.841 | 0.181 |
| 15 | 0.686 | 0.545 | 0.761 | 0.169 |
| 21 | 0.732 | 0.586 | 0.709 | 0.195 |
| 27 | 0.759 | 0.631 | 0.675 | 0.213 |
| 33 | **0.831** | 0.726 | 0.570 | **0.293** |

曲线在整个扫描区间内**仍在上升、未见饱和** —— 也就是说 33 不是"够大了"的点，
而是"这个区间里最大的点"。要收敛到 full-672，需要的 halo 比 33 更大。

### 4.2 按层

```
layer       n=5     n=15    n=25    n=33      (cos_median / nn_spearman)
final      .358/0.02 .782/0.18 .854/0.30 .921/0.45
midlate    .385/0.13 .646/0.16 .682/0.14 .771/0.25
mid        .422/0.15 .607/0.21 .623/0.16 .716/0.19
```

**最深的 `final` 层最依赖全局上下文**，也最难由 halo 还原。

**关键**：`final` 在 n=33 时 cos 已达 **0.921**，但 nn_spearman 只有 **0.454**。
**余弦相似 ≠ 近邻序不变** —— 而探测器消费的正是 1-NN 距离。只看 cosine 会严重
高估 halo 的可用性。

### 4.3 有对象是负相关

n=33 时最差：`visa/macaroni1` **−0.282**、`mvtec/capsule` **−0.178**。
对这些图，halo 的近邻序与 full-672 **反向** —— 比不提供上下文更差。
最好的是 `mvtec/metal_nut` 0.712、`mvtec/tile` 0.521。

### 4.4 预算（324 cell 全覆盖）

| n_ctx | K_max(中位) | 触发图被 cap | 保留的触发 |
|---|---|---|---|
| 5 | 46 | 2.4% | 92.1% |
| 15 | 5 | 42.8% | 53.5% |
| 25 | 1 | 77.1% | 20.6% |
| 33 | **1** | **81.5%** | **15.9%** |

---

## 5. 判读

保真度与预算**在两个方向上同时恶化**：

```
halo 小  ->  预算宽裕，但 nn_spearman 0.13–0.17，近邻序基本无关
halo 大  ->  nn_spearman 仍只有 0.29，而 K_max 塌到 1，
             81% 的触发图被截断，只剩 16% 的触发点被细化
```

不存在一个 n_ctx 同时满足"保真度可用"和"50% 预算装得下"。
**"纯 halo 保存全局上下文"这一机制在 50% token 预算内不成立** —— 这正是
预先定义的 STOP 条件。

**必须写明的边界**：

- 这**不是**"选择性细化这个方向失败了"。被证伪的是**具体机制**（用局部
  halo 重建全局上下文），不是目标。Stage 4 已证明 context 损失值 +28.34
  AUPRO，说明"补回上下文"本身是对的 —— 只是不能靠放大裁剪窗口来补。
- 本包**没有**任何 selector 相关信息。`A3_corrected` vs `A4_corrected` 的
  G3 问题依然未决，仍需新的 324-cell。
- 结论建立在 **support-normal 特征保真度 + 预算算术**上，未触碰任何缺陷标签。

---

## 6. 复现

```bash
# 全量 sweep（~13 分钟，27 类）
OMP_NUM_THREADS=2 python experiments/model_v0/v3_stage5b_context.py

# 只重算预算表（秒级，不需要 sweep）
S5B_BUDGET_ONLY=1 python experiments/model_v0/v3_stage5b_context.py
```

危险写法（本次踩过的坑，已在源码注释中标注）：
`np.concatenate(<pandas Series>)` 在 groupby 内会抛 `KeyError: 0` —— numpy 按
**位置**遍历（`seq[0]`），而 `Series.__getitem__(int)` 是**标签**语义；只有当该
group 的 index 恰好含标签 0 时才"正常"。本脚本已改用 plain dict 累积。
