# Model V3 预注册 · 补充 2：normal-only 仿射校准的 self-exclusion

> `docs/MODEL_V3_PREREGISTRATION.md`（tag `model_v3_preregistered`）与
> `docs/MODEL_V3_PREREGISTRATION_ADDENDUM_1.md`（tag `..._addendum1`）已冻结。
> 本文档补上 §2.4 仿射校准中一个**会导致校准分数退化为 0 的实现定义缺口**。
> 依据原文自身的规定"若需变动，另开新文档并说明理由"。
>
> **冻结时间：在任何 V3 主实验运行之前。主实验尚未开始。**

---

## A2.1 缺口

预注册 §2.4 要求：

> 校准用仿射映射 `a·x + b`，使 refined 图在**正常参考 patch** 上的中位数与
> 四分位距分别匹配 448 图。校准只用正常数据。

但"正常参考 patch"取 §2.2.2 的 support 图时有一个退化：

> **support 图本身就在 bank 里**，所以 support patch 到 bank 的 1-NN 距离
> 恒为 ~0（最近邻就是它自己）。用它拟合的 `a, b` 是把一个退化分布映射到
> 448 分布，`a` 会爆掉或被噪声支配。

## A2.2 exclusion 必须按**物理空间**对齐，不能按 token 数量对齐

448 与 672 的 token 在原图上覆盖的尺度不同，**同一个"3×3"在两个分辨率下
不是同一块物理区域**：

```
448 token  = 14 px（448 尺度下的缩放图）  = 21 px（672 尺度）  = 1.5 个 672 token
=> 448 的 3x3 token 邻域在物理上 = 4.5 x 4.5 个 672 token
```

因此排除集定义为**物理区域**：

```
对 query patch p，其 448 位置为 token (r, c)：
  物理范围 F(p) = 448 尺度像素 [14(r-1), 14(r+2)) x [14(c-1), 14(c+2))
                = 672 尺度像素 [21(r-1), 21(r+2)) x [21(c-1), 21(c+2))
  排除集 E(p)   = bank 中所有【footprint 与 F(p) 有交集】的 patch
```

**不是**在两个分辨率上机械地各写"3×3 token"。

## A2.3 两端用**同一条原则**

448 与 672 的校准分数必须由**同一套 pseudo-query / self-exclusion 逻辑**
产生。若只对 672 做 exclusion 而 448 不做（或反之），拟合出的 `a, b` 会混入
**校准协议差异**，而不是分辨率之间的尺度差。因此：

```
s448_cal(p) = min_{q in B448, q not in E(p)}  d(z448(p), z448(q))
s672_cal(p) = min_{q in B672, q not in E(p)}  d(z672(p), z672(q))
```

**同一 E(p)**，同一个 query 集（同一批 support patch）。

## A2.4 方向（沿用 §2.4，不得临时更换）

```
s448_cal  ~=  a * s672_cal + b
```

即把 672 侧校准到 448 侧。拟合量为 `a, b`（median + IQR 匹配，见 §2.4）。

## A2.5 support-only 约束（冻结）

`a, b` **只**由当前 `(object, k, split)` 抽中的 `S_k` 估计：

- 不使用任何额外 `train/good` 图；
- 不使用任何 test 图；
- 不使用 GT / anomaly label。

## A2.6 健康检查：记录但**不改算法**

拟合后必须记录：

```
a, b, R^2（或至少 Pearson/Spearman 相关）
a_min, a_max, b_min, b_max（跨 (object, k, split)）
```

- `R^2` 低 **只作为 failure diagnostic**，**不得**据此改用非线性校准。
- 出现以下任一情况**直接 fail-fast**，**不得**自动 fallback 到别的校准：

```
a < 0
|a| 极端大（实现中冻结为 |a| > 100）
a 或 b 为 NaN / Inf
```

## A2.7 `m = 0` 的图（冻结，同时适用于 §10-F）

```
m = 0  =>  A3 = A4 = A5 = A0
```

即触发数为 0 的图**不做任何细化**，A4/A5 **不得**自行强行细化一个区域。
否则 same-budget 不再成立。

## A2.8 与原文的关系

| 项 | 来源 |
|---|---|
| 仿射校准的方向、median/IQR 匹配、只用正常数据 | 预注册 §2.4（不变） |
| 排除集 `E(p)` 按物理区域定义、两端同原则 | **本补充 A2.2 / A2.3（新增冻结）** |
| 健康检查与 fail-fast 阈值 | **本补充 A2.6（新增冻结）** |
| `m = 0` 的退化处理 | **本补充 A2.7（新增冻结）** |
| 所有判据、阈值、Gate | 预注册 §2.2.4 / §4（**不变**） |
