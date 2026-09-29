# P0-B —— Feature-Level Parameter-Free Global→Local Retrieval Audit

> **状态：FROZEN。** 本协议在任何结果产生之前写定。运行后不得根据结果修改。
> 若发现真实 implementation bug：保存旧结果 → 标记 `INVALIDATED` → 写 addendum →
> 修复 → 完整重跑。**不得静默覆盖。**

---

## 0. 实验定位（先写死）

P0-B 是 **feature-level parameter-free Global→Local retrieval audit**。

```
只使用 normal support 数据。
不使用缺陷图像、不使用 GT、不计算 AUPRO/AUROC/A5。
不训练 Transformer、不训练 Q/K/V、不训练 MLP、不搜索超参数。
P0-B 只决定是否值得进入后续轻量 Cross-Attention T0。
```

**研究假设 H**：`F^{Local672}` 虽然在 P0-A 的**标量 1NN distance** 中互补信息较弱，
但其**高维表示**可能仍含有被标量距离压缩掉的信息。若让 Local672 token 去查询
`F^{Global448}` 的整图 token memory，得到的 `F^H` 应比单独 Global448 与单独 Local672
**更接近** `F^{Global672}`。

**P0-B 不是 Transformer**，也不为 Transformer 做超参搜索。它的唯一产物是一个准入判断。

---

## 1. 数据范围

```
SHOT  = 8
SPLIT = 0
MVTec = 15 categories
VisA  = 12 categories
总计  = 27 categories
```

只读取 `train/good` normal support。**严禁**加载 test anomaly / GT mask / AUPRO /
pixel AUROC / image AUROC / A5。

n 只允许 `{9, 13, 15}`；层只允许 `mid / midlate / final`。

---

## 2. Geometry manifest：直接复用 F0，不重新采样

读取 `results/model_v0/v3_1_f0_resolution_increment/f0_probe_distances.csv`
作为 P0-B 的 **geometry manifest**。

**不重新生成 anchor、不重新随机采点、不改变类别。** 必须完全沿用其
`dataset / object / support_id / anchor_r / anchor_c / patch_r / patch_c / n / layer`。

理论行数：

```
27 categories x 8 support normals x 9 anchor crops x 9 probes per anchor
             x 3 scales (n=9,13,15) x 3 layers
= 157464
```

**raw CSV 不等于 157464 直接 FAIL implementation，不允许分析结果。**

---

## 3. Cache 审计（主实验之前）

必须核验 448 与 672 cache 的：image name、offsets、token grid、layer、feature dim、
dtype、preprocessing、DINO checkpoint。

必须断言 **support identity**：

```
support_names_448 == support_names_672        # 逐 name 比较
```

**不允许只检查长度**，不允许 `len(pos) == m` 这类假 provenance check。

---

## 4. Positive control

对象 `mvtec/bottle` 与 `visa/pcb2`，两个 resolution × 三层 = **12 项**。
要求 `cos = 1.000000`、`max_abs_diff <= 1e-5`。**失败即 STOP，不得继续 P0-B。**

---

## 5. 四个核心 feature

对每个 F0 probe `(dataset, obj, support_image, anchor, n, patch_r, patch_c, layer)`：

| 符号 | 定义 |
|---|---|
| **B** | `F^{G448}` 在物理对应位置的 token；L2 normalize |
| **T** | **Full672 reference**：把该 448 token 的物理 footprint 用 **area-overlap pooling**（`cal.pool_weights` 或完全相同的 frozen arithmetic）映射到 Full672 grid；L2 normalize。**不得使用 `round(1.5*r)`** |
| **Q** | **Local672 query**：对同一个 F0 crop，取 Local672 crop feature，用**同一套 overlap weights**对**完全相同物理 footprint** pooling；L2 normalize。**必须调用 F0 原本的 crop forward 函数**，不得重新实现裁剪逻辑 |
| **K = V** | 同一 support image 的整张 Full448 token grid，全部 L2 normalize。**必须直接读取真实 grid**，禁止假定 32×32 |

**T 与 Q 对应同一个原图物理区域**，只是分别来自 full672 与 crop672。

---

## 6. Parameter-free retrieval（固定，不训练）

```
s_j = sqrt(D) * cos(Q, K_j)
a_j = exp(s_j) / sum_k exp(s_k)
C   = normalize( sum_j a_j V_j )
H   = normalize( Q + C )          # alpha 固定为 1
```

**禁止**：`nn.MultiheadAttention`、Linear Q/K/V、MLP、alpha sweep、temperature sweep、
top-k sweep、attention heads。

---

## 7. 四个 arm + shuffled negative control

同时保存 **B**（Global448）、**Q**（Local672-only）、**C**（global retrieval context）、
**H = normalize(Q+C)**（global-conditioned local），目标 **T = Full672**。

**Shuffled control**：对 support image `i`，真实版本 `Q_i → G448_i`；shuffled 版本
`Q_i → G448_{i+1}`（八张 support image 固定 cyclic shift `0→1→…→7→0`），得到
`C_shuf` 与 `H_shuf = normalize(Q + C_shuf)`。**固定 cyclic shuffle 一次，不做 10 seeds。**

它回答：若随便给 Local672 一张同类别 normal 图的 global memory 也能获得同样效果，
就不能说是正确的 image-specific global context 在起作用。

---

## 8. NN bank：严格 leave-one-support-image-out

当前 query 来自 support image `#i` 时，所有 NN bank **完全删除 image #i 的全部 token**，
只用另外 7 张。**不采用**"删同位置 3×3 但保留同图远处"的做法。

**主 bank = `Bank672`**（另外 7 张的 Full672 全部 token）。
`d_B / d_Q / d_H / d_Hshuf / d_T` **全部查询同一个 Bank672**，以避免 bank-domain confound。
可附加记录 `d_B^{native} = NN(B, Bank448)`，但**不得作为主 gate**。

---

## 9. 三层聚合：image-level cross-fitted

**禁止 crop-local mean/std**，**禁止**用当前 query image 自己拟合 normalization。

对每个 held-out support image：用**另外 7 张** support images 的 probes 计算每层
`mu_l, sigma_l`，然后用**同一套 layer calibration** `z_l(d) = (d - mu_l)/sigma_l` 变换
`d_B, d_Q, d_H, d_Hshuf, d_T`，再 `agg3 = (z_mid + z_midlate + z_final)/3`。

---

## 10. Feature headroom recovery

在 category 内先取 median（**不逐 probe 相除**，因为 `1-cos(B,T)` 可能很小而爆）：

```
c_B = median[cos(B,T)], c_Q = median[cos(Q,T)], c_H = median[cos(H,T)]
c_base = max(c_B, c_Q)
R_feat = (c_H - c_base) / (1 - c_base)
```

---

## 11. Correctness gates（任一失败 → 不分析结果，先修 implementation）

| 编号 | 内容 | 判据 |
|---|---|---|
| PC1 | 448/672 cache reproduction | 12/12 PASS，`cos≈1`，`maxdiff <= 1e-5` |
| PC2 | 40 个随机 probe 的 local pooling weights vs `cal.pool_weights` | 逐 index、逐 weight 一致 |
| PC3 | 1000 个随机 query 的 held-out support image 在 bank 中出现次数 | **0** |
| PC4 | attention 权重和 | `|sum_j a_j - 1| < 1e-6` |
| PC5 | B/Q/C/H/T 全部 finite | 无 NaN/Inf |
| PC6 | normalize 后 `\|\|B\|\|,\|\|Q\|\|,\|\|C\|\|,\|\|H\|\|,\|\|T\|\|` | `≈ 1` |
| PC7 | raw rows | `= 157464` |
| PC8 | 每个 `(category, n, layer)` 的行数 | `= 648` |

---

## 12. Smoke run（只验证实现）

只跑 `mvtec/bottle` 与 `visa/pcb2`，`n = 9,13,15`，三层全跑。目的仅检查 shape、memory、
NaN、bank exclusion、crop mapping、output schema、resume。

**不得**根据结果修改 temperature、alpha、n 或 attention formulation。

---

## 13. 完整运行

27 categories，支持 resume。中间缓存 `results/model_v0/p0b_feature_level_audit/cache/`，
按 `dataset/object/support_id/anchor/n/` 保存（最贵的是 Local672 DINO forward，
机器中断后不得全部重算）。每完成一个 category 写 `DONE`；聚合脚本只读 DONE categories。

---

## 14. 冻结的 GO gate

```
G1  median_27cats( rho_H - rho_B   ) >= 0.03
G2  median_MVTec( rho_H - rho_B ) > 0  AND  median_VisA( rho_H - rho_B ) > 0
G3  median_27cats( rho_H - rho_Q   ) > 0
G4  median_27cats( R_feat ) >= 0.10
G5  median_27cats( rho_H - rho_shuf ) > 0

GO  =  G1 ∧ G2 ∧ G3 ∧ G4 ∧ G5
```

其中 `rho_X = Spearman(d_X, d_T)`，按 `(category, n, layer)` 计算后进入 agg3 口径。

**若多个 n PASS，选择最小的 n**，不得选择效果最高的 n。

---

## 15. Bootstrap（只做 uncertainty，不改 gate）

以 category 为统计单位，MVTec 15 / VisA 12，做 **10000 次 dataset-stratified
category bootstrap**，输出 `median ΔB / ΔQ / Δshuf / R_feat` 的 95% CI。
**bootstrap 不修改已冻结的 GO gate。**

---

## 16. 结果的三个解释（只允许这三种）

1. **全部 PASS → P0-B GO**：可以说"高维 Local672 feature 中确实存在 score-level P0-A
   没捕获的互补信息，且 parameter-free global context retrieval 已能利用其中一部分"。
   **下一步才允许进入 T0：轻量 Cross-Attention。**
2. **feature cosine 提高但 `rho_H` 没提高 → NO-GO**：representation 看起来更像 full672，
   但没有恢复 anomaly NN geometry。**不得因 cosine 好看就继续 Transformer。**
3. **G1–G5 任一失败 → P0-B NO-GO → Transformer branch 正式关闭**：不允许继续用
   2 layers / 4 heads / MLP / temperature tuning / positional bias tuning / n=17,21 去救。

无论 GO 或 NO-GO，**都不得直接跑 defect performance 或 324 cells**。

---

## 17. 交付物

目录 `results/model_v0/p0b_feature_level_audit/`：

```
P0B_PROTOCOL.md   P0B_FREEZE.txt   README_P0B.md   run.log
p0b_manifest.json p0b_positive_control.csv  p0b_selftest.csv
p0b_probe_features.csv  p0b_per_layer.csv  p0b_agg3_by_category.csv
p0b_summary.csv   p0b_bootstrap.csv   p0b_decision.csv
feature_cache_manifest.csv
p0b_feature_level_audit_package.zip
```

`README_P0B.md` 顶部直接给 `FINAL DECISION: GO / NO-GO`，随后是
`| n | ΔH-B | ΔH-Q | MVTec Δ | VisA Δ | Rfeat | H-shuf | Decision |`。

---

## 18. Git 流程

从 `v3-rootcause-closed` 创建 `p0b-feature-audit`。**先只提交协议**，记录 commit SHA 到
`P0B_FREEZE.txt`，再开始写实验代码。**运行后禁止根据结果改 protocol。**
