# Phase 6/7 结果总账（冻结）

> 冻结于 tag `phase7_defect_support_closed`。
> **共享 Adapter 路线正式结束；PCGrad 不进入主线。**
> 下一阶段（Phase 8）转向 external defect support acquisition。
> 本文档是路线判决依据，不是过程记录；过程见 `MODEL_V0.md`。

---

## 最终结论（严谨表述）

> **在冻结 DINOv2 特征下，目标类缺陷 support 的「量」是真实有效的杠杆；
> 融合必须用 raw（`S_n − S_d`，β=1），任何逐分支标准化都会破坏它。
> 但 support 的「质」——即用无标签几何手段挑选更好的 support——至今没有任何一条有效。
> support 的规模本身就同时包含实例多样性与 patch 密度两个成分，二者不可再分且都有效。**

---

## 排除链

```
共享 Adapter（K4）提升来自 support overfitting，不是学到的不变性   ❌ Gate 6A
        ↓
冻结双分支 oracle 在严格划分下只 +4（且是我用错了标准化）          ⚠️ Gate 7A
        ↓
改用 raw fusion，oracle 提升巨大（wallplugs 41.8 → 77.0）          ✅ Gate 7B-2
        ↓
该提升随 support 规模单调上升                                       ✅ Gate 7B-3R
        ↓
但「选得更好」（k-center / coverage）完全无效                        ❌ Gate 7B-3R
        ↓
规模的两个成分（实例数 / patch 密度）都有效，无法互相替代            ✅ Gate 7B-4
        ↓
结论：Generator 必须同时提供模式多样性 + 高密度支持，
      但「如何挑选」不是可解的子问题
```

---

## 1. Gate 6A：Adapter 提升的真实来源

K4 adapter 在训练集正常样本尾部上保持平坦（cosine p99 ×0.98–1.01），
但 TEST_pub 正常样本 p99 从 0.104 升到 0.333。

逐图检验：把「该图到 support 的冻结距离」对「adapter 抬升量」做 Spearman 相关，
**正相关**——即 adapter 按未见正常样本「本来离 support 多远」成比例地抬高其分数。
这是 **support overfitting** 的签名，不是学到了不变性。

**判决：共享 Adapter 路线正式结束。**

## 2. Gate 6C 泄漏（方法错误，已修正）

6C 的 defect support 由 defect images 的**监督那一半**构建，而评估覆盖**全部** defect images，
bank 的一部分是它自己的测试集。6C 的 +48 不可用。

## 3. Gate 7A：干净 oracle

严格 50/50 划分 defect images 为 support / evaluation，重复 3 个 split。
所有机制统计量（normal_p99、defect_mean、`frac_defect_below_p99`）来自**同一个** fused score。

**但 7A 用了 median/MAD 标准化，这后来被证明是我的错误**（见 §5）。
在 MAD 下 β=1 的结果是：

| object | img_AUROC | 冻结 AnomalyDINO 参考 |
|---|---|---|
| wallplugs | 45.9 ± 1.1 | 41.7 |
| sheet_metal | 88.2 ± 2.0 | 83.6 |
| vial | 95.3 ± 1.1 | 91.7 |
| can | 52.5 ± 3.2 | 51.5 |

即「一个真实但微弱的 +4」。**这个 +4 是标准化选错造成的假上限。**

## 4. Gate 7B-1：分支信息量（rescue / damage）

| object | AUROC(S_n) | AUROC(S_d) | rescue | damage |
|---|---|---|---|---|
| wallplugs | 41.8 | 80.6 | 0.935 | 0.374 |
| sheet_metal | 84.3 | 86.9 | 0.709 | 0.100 |
| vial | 90.6 | 99.4 | 1.000 | 0.006 |
| can | 52.6 | 56.2 | 0.557 | 0.434 |

`S_d` **单独**在 wallplugs 上就有 80.6，在 vial 上 99.4。
缺陷分支携带的是排序信息，不是全局平移（常量平移不改变 AUROC）。

## 5. Gate 7B-2：无标签融合 —— 我的 MAD 错误已撤回

三种融合，全部只用目标类**正常** support 标定，不使用任何目标异常标签：

| fusion | can | sheet_metal | vial | wallplugs |
|---|---|---|---|---|
| raw (`S_n − S_d`) | **56.8** | **91.3** | **100.0** | **77.0** |
| mad | 52.5 | 88.2 | 95.3 | 45.9 |
| cdf | 49.6 | 80.6 | 99.7 | 62.6 |

**raw 在所有四个对象上都最好。** wallplugs 从 45.9 → 77.0（+31.1）。

> 我在 7A 阶段把 MAD 当作「正确的标定」并据此宣布「+4 是真实上限」，
> 这是错的。7B-2 证明 raw 才是对的。**该结论已撤回。**

## 6. Gate 7B-3R：support 规模曲线

按 defect **image 数**索引（之前的 patch 索引把同图 patch 当成独立样本，已修正）：

| n images | wallplugs | vial | sheet_metal | can |
|---|---|---|---|---|
| 1 | 55.7 | 71.0 | 73.6 | 57.8 |
| 2 | 69.9 | 72.2 | 76.5 | 48.0 |
| 4 | 60.3 | 71.7 | 77.1 | 50.0 |
| 8 | 66.6 | 78.3 | 85.6 | 51.7 |
| 16 | 72.6 | 89.2 | 90.7 | 53.6 |
| all | ~77 | 100.0 | 91.3 | ~57 |

wallplugs +21、vial +29、sheet_metal +18。**can 在所有 support 规模下持平（~50–57）。**

### 两个负子结果

**(a) 多样性（k-center vs random）在匹配 budget 下基本无差别：**

| budget | wallplugs random / kcenter | sheet_metal random / kcenter | vial random / kcenter |
|---|---|---|---|
| 20 | 71.3 / 71.6 | 84.4 / 74.5 | 69.0 / 83.2 |
| 100 | 76.5 / 74.9 | 85.7 / 84.9 | 89.6 / 99.8 |

严格表述：**「由 DINO feature distance 定义的 diversity」没有帮助**，
不能推广为「diversity 不重要」。

**(b) coverage distance 不预测失败**：按每个 held-out defect image 到 support 的距离中位数分组，
「覆盖好」与「覆盖差」的融合分数几乎相同（wallplugs 0.035 vs 0.034；
sheet_metal 0.144 vs 0.145；vial 0.205 vs 0.187）。

**判决：support 的「质」——即用无标签几何手段选更好的 support——不是可解的子问题。**

## 7. Gate 7B-4：实例数 vs patch 数的解耦（最后一个机制实验）

### 先说一个设计失败（如实记录）

用户设计的原始方案——**固定总 patch 数 100，image 数取 1/2/4/8/16**——**在 MVTec AD 2 上跑不了**。
单张图的缺陷 patch 数（`gt_frac > 0.10`）：

| object | 中位数 | 最大 | 全类总数 |
|---|---|---|---|
| wallplugs | 2 | 28 | 528 |
| vial | 22 | 208 | 5258 |
| sheet_metal | 36 | 197 | 4257 |
| can | 1 | 3 | 54 |

没有任何一张图能提供 100 个 patch。所以 n=1 时「固定预算」静默失效
（wallplugs 实际只拿到 3 个 patch 而不是 100），该曲线测的仍然是 patch 数。
**这个配置已作为诊断记录在案，不从中得出任何结论。**

### 可行的设计

把预算降到每张图都供得起。预算 B 摊到 n 张图，每张需 ⌈B/n⌉ 个，故**最多 B 张图参与**。

- B = 4，n ∈ {1, 2, 4}
- B = 8，n ∈ {1, 2, 4, 8}

n = B 时每张图恰好出 1 个 patch（该预算下最大多样性）；n = 1 时单图出全部 B 个（最大密度）。
预算在每一格都严格等于 B（运行时 `assert`）。每格 30 次重复抽样 × 3 split = 90 个样本。

### 控制实验：合格图像集必须固定

第 (2) 节的天然缺陷：n=1 时只允许「含 ≥B 个 patch」的图入选，n=8 时任何含 ≥1 个的图都行，
**合格图像集随 n 变宽**，增益可能是选择效应。第 (2b) 节把每格都限制在**同一集合**
（含 ≥B 个 patch 的图）中抽样：

**B = 8（预算严格 = 8）**

| n images | sheet_metal | vial | wallplugs |
|---|---|---|---|
| 1 | 73.3 ± 8.6 | 63.0 ± 18.7 | 60.4 ± 10.0 |
| 2 | 78.0 ± 7.3 | 69.5 ± 13.1 | 65.8 ± 9.5 |
| 4 | 78.5 ± 7.8 | 70.0 ± 13.6 | 66.6 ± 9.0 |
| 8 | 80.3 ± 7.0 | 77.7 ± 9.9 | 67.5 ± 8.4 |
| **Δ (1→8)** | **+7.0** (t=6.0) | **+14.7** (t=6.6) | **+7.1** (t=5.2) |

**B = 4**

| n images | sheet_metal | vial | wallplugs |
|---|---|---|---|
| 1 | 72.6 ± 10.7 | 66.9 ± 17.7 | 59.7 ± 8.7 |
| 2 | 74.5 ± 9.6 | 72.0 ± 13.3 | 62.4 ± 10.7 |
| 4 | 74.5 ± 9.3 | 73.9 ± 10.4 | 63.7 ± 10.5 |
| **Δ (1→4)** | +1.9 (t=1.3, n.s.) | **+7.0** (t=3.2) | **+4.0** (t=2.8) |

**结论：固定 patch 预算下，image 数越多仍显著更好。选择效应被排除。**

### 另一根轴：固定 image 数，变 patch 数（第 3 节）

固定 8 张图，取 8 → 64 个 patch：

| object | 8 patches | 64 patches | Δ | t |
|---|---|---|---|---|
| sheet_metal | 76.0 ± 8.1 | 82.5 ± 4.8 | +6.5 | 6.5 |
| vial | 68.8 ± 10.6 | 78.7 ± 6.7 | +9.9 | 7.5 |
| wallplugs | 61.0 ± 5.3 | 65.4 ± 5.0 | +4.4 | 5.7 |

**两根轴都显著上升。**

### 归一化对比（各跨 3 个 doubling）

| object | 每 doubling：实例多样性 | 每 doubling：patch 密度 |
|---|---|---|
| sheet_metal | +2.3 | +2.2 |
| vial | +4.9 | +3.3 |
| wallplugs | +2.4 | +1.5 |

**判决：两个成分都有效，不可互相替代。**
实例多样性在 vial 上略占优，其余两者相当。

### 限制（必须一起声明）

- 这些绝对 AUROC（60–80）是在**极小预算（4–8 patch）**下的水平，
  方差很大；结论是**方向性**的，不是绝对性能。
- can 在所有预算下都不可行（全类仅 54 个 defect patch，单图中位数 1）——
  与 7B-3R 一致，can 属于**非 support-limited regime**，不参与本实验的判决。

## 8. can 的处理（定案）

can 在 7B-3R 与 7B-4 中均无响应。**将其定义为独立的、非 support-limited 的失败 regime**，
不再用 support 侧手段解释它，也不为其调参。

## 9. 本阶段资产

| 类型 | 内容 |
|---|---|
| 冻结机制 | `S_n − S_d`，raw，β=1，无 adapter，无训练 |
| 脚本 | `gate6a_6c.py`、`gate7a.py`、`gate7b.py`、`gate7b3r.py`、`gate7b4.py` |
| 数据 | `results/model_v0/metrics/gate7*.csv` |
| 性能修复 | `load_cache` 改为 `materialize`（npz 惰性解压曾导致 sheet_metal 假死）；`make_dist` 双侧分块 |

---

## 继承到 Phase 8 的强制方法纪律

1. **预算是设计约束，不是事后描述**——运行时 `assert` 实际预算，不满足就标为不可行。
2. **合格集合随自变量变化 = 选择效应**，必须固定合格集重跑一次才能下结论。
3. **任何「更好」都要有匹配规模的对照**（k-center 必须对 random@同 budget）。
4. **无标签几何手段至今 0 命中**——不要再用几何接近性论证 support 质量。
5. **报告统计量时给 t 值**；单 split 的差异不下结论。
6. **跑不出设计就说跑不出**，不要用打了折的实现冒充原设计。
