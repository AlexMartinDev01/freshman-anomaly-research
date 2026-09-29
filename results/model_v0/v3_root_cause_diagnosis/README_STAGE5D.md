# Stage 5D-A —— Adaptive-Halo Budget Audit（全离线）

> 读取 324 / 324 cells、46644 张 test image 的冻结 `diagnostics.json`。
> 生成于 2026-09-29。脚本 `experiments/model_v0/v3_stage5d_adaptive_audit.py`。
> **不跑 DINO、不用 GPU、不读缺陷标签、不碰 selector。**

---

## 0. 这个包回答什么

Stage 5C 证明**固定** halo 在 `n ≤ 33` 内没有同时满足 `agg3 Spearman ≥ 0.50` 与
`trigger-mass retention ≥ 0.50` 的点。但冻结的 V3 用的**不是固定 n**：addendum 1 按图分配

```
m' = min(m, K_max),   K_max = floor(B/25)
n  = [5,33] 内最大的奇数 n，满足 m'·n² ≤ B
```

所以 **fixed-n 的 Pareto 曲线不等于 adaptive 的 Pareto 曲线**，fixed-n STOP 不能直接关闭
adaptive。本包用真实触发分布回答这一步。

**本包不含任何 defect 性能指标。**

---

## 1. 正确性：复现冻结实现

```
images audited                              46644
cells covered                               324 / 324
(m2, n) 复算 vs 冻结 a3 记录的不符数            0
arm 间 (m2, n) 不一致数                        0
B != 0.5*ceil(1.5*gh448)*ceil(1.5*gw448)     0   <- 复现的是冻结公式
```

`n` 与 `m2` 是**逐张比对**冻结记录，不是各算各的。同时验证 `m'·n² ≤ B` 在**真实** 672
grid 下不越界（见 §2 的例外）。

---

## 2. 审计中发现的冻结实现缺陷：`B` 是推导来的，不是读来的

```python
v3_run.py:100-101
    grid672 = (int(np.ceil(gh4 * 1.5)), int(np.ceil(gw4 * 1.5)))
    B = sel.budget(grid672)
```

冻结代码用 `ceil(1.5 × 448 grid)` **推导** 672 grid，而不是读真实的 672 grid。
由于 resize 取整，二者会**双向**偏离：

| 对象 | 448 grid | 推导 672 | 真实 672 | B_frozen | B_true | 比 |
|---|---|---|---|---|---|---|
| visa/candle | 32×35 | 48×53 | 48×**52** | 1272 | 1248 | **1.019** |
| visa/cashew | 32×34 | 48×51 | 48×**52** | 1224 | 1248 | 0.981 |

**后果**：

```
真正超出 50% token 预算的图    111 / 46644  (0.24%)，最大超出 ~1.9%
若改用真实 B，n 会变的图       113 / 46644
```

即 "budget holds by construction"（G2）在 0.24% 的图上不成立。数字很小，不影响任何已有
结论，但 **V3.1 必须直接读 672 grid**。

> 注：同一缺陷此前在 Stage 5B 的预算辅助代码里也被发现并修掉过 —— 属于
> "derive instead of read" 家族。证据见 `05d_budget_derivation_audit.csv`。

---

## 3. 真实自适应分配下的 n 分布（按 trigger mass 加权，主结果）

| n | 图数 | 图占比 | trigger mass | **mass 占比** | 平均 m |
|---:|---:|---:|---:|---:|---:|
| 5 | 1466 | 6.3% | 76754 | **33.8%** | 52.4 |
| 7 | 1831 | 7.9% | 37930 | 16.7% | 20.7 |
| 9 | 2253 | 9.7% | 30960 | 13.6% | 13.7 |
| 11 | 2793 | 12.0% | 26621 | 11.7% | 9.5 |
| 13 | 2174 | 9.4% | 16554 | 7.3% | 7.6 |
| 15 | 2617 | 11.3% | 14024 | 6.2% | 5.4 |
| 17–21 | 3764 | 16.2% | 14021 | 6.2% | — |
| 23–29 | 3128 | 13.5% | 7036 | 3.1% | — |
| **33** | **3159** | **13.6%** | **3159** | **1.4%** | **1.0** |

**结构性事实**：`m'·n² ≤ B` 把 n 与 m 反向绑定，于是

> **自适应把最大的 halo 给了证据最少的图（m=1），把最小的 halo 给了证据最多的图（m=52）。**

可达的 n 也比扫描网格粗：MVTec 只能取 `{5,7,9,11,13,15,19,23,33}`（25/27/31 不可达），
VisA 为 `{5,7,9,11,13,15,17,19,21,23,29,33}`。

---

## 4. 四个头号数字

| 量 | mvtec | visa | **ALL** |
|---|---|---|---|
| `P_mass(n ≤ 15)` | 0.942 | 0.859 | **0.893** |
| `P_mass(n ≥ 23)` | 0.031 | 0.055 | **0.045** |
| overall trigger-mass retention | 0.849 | 0.965 | **0.917** |
| mass-weighted mean n | 8.25 | 10.60 | **9.62** |
| **weighted fidelity proxy（中位）** | 0.207 | 0.135 | **0.184** |

**自适应确实保住了 retention（0.917），但几乎没有换来 context** —— 89.3% 的 trigger mass
落在 `n ≤ 15`，即 Stage 5C 中 `agg3 Spearman ≈ 0.08–0.28` 的区间。

---

## 5. Fidelity proxy 的定义、caveat 与稳健性

```
rho_proxy(c) = Σ_n  w_c(n) · rho_c(n)
```

`w_c(n)` = 类别 c 在真实自适应分配下的 **trigger-mass** 占比（本包 §3）；
`rho_c(n)` = Stage 5C 该类别在 n 处的 detector-consistent `agg3` Spearman。

### 这**不是**测得的 mixed-n Spearman，只是筛选代理

(a) n **不是随机的** —— 拿到大 n 的恰好是低 m 图，代理假设 fidelity 主要由 n 决定；
(b) Stage 5C 的 fidelity 只在 **k=8/split=0** 的 support 上测过，而这里的分配横跨全部
shot/split。它只用来决定"值不值得花 GPU 做 Stage 5D-B"，**不能当作结果引用**。

### 稳健性

| 口径 | median_cat | max_cat |
|---|---|---|
| 按报告口径（pooled per-category） | **0.184** | 0.422 |
| 每个 n 取最有利的 agg3 convention | 0.231 | 0.308 |
| **optimistic proxy upper envelope**（再叠加每个 n 取最好的类别） | 0.445 | 0.498 |

**措辞：这是 "upper envelope"，不是 "upper bound"。** 它把两处乐观假设叠满
（每个 n 都用最有利的 convention，且每个 n 都假设达到 27 类里最好的类别），
但**本身仍是代理量**，因此不能引用为数学上界，也不能据此宣称"proxy 不可能更高"。

即便按这个上包络，中位也只有 0.445，低于 0.50 的 gate。

### 与 fixed-n 的对照

```
adaptive (mass-weighted)   0.184     retention 0.917
fixed n=15                 0.275     retention 0.535
fixed n=33                 0.467     retention 0.159
```

**自适应比固定 n=15 还差。** 它用 0.917 vs 0.535 的 retention 优势，换来 fidelity 从
0.275 掉到 0.184 —— 而 0.275 本身已经不可用。覆盖更多位置没有意义，如果那些位置的
score geometry 是错的。

---

## 6. 裁决

预设分岔（在 Stage 5D-A 运行前固定）：

```
若大部分 trigger mass 落在 n<=15 且 weighted proxy 明显低于 0.40  -> 关闭 halo family
```

实际：

```
P_mass(n<=15) = 0.893        (大部分)
weighted proxy = 0.184       (远低于 0.40)
```

**=> 关闭 pure-halo family（含固定 n 与自适应 n）。** 不再试 35/37/41，不用 defect
performance 去救；因此 Stage 5D-B（真实 mixed-n fidelity 测量）**不跑** —— 按照预定的
分岔规则，它只在 proxy 处于 0.40–0.50 灰区或更高时才值得花 GPU。

**边界（不得扩大）**：

- 被关闭的是 **halo 机制**（用局部裁剪重建全局上下文）。**selective refinement 方向未关闭** ——
  Stage 4 的 `context (C10−C00) = +28.34 AUPRO` 说明"补回上下文"本身是对的。
- 本包**不含 selector 信息**，G3 仍未决。
- proxy 的 caveat（§5）意味着这个关闭结论建立在**筛选级**证据上。若将来有人质疑
  "会不会实际测量更好"，唯一能推翻它的正是被跳过的 Stage 5D-B —— 但按预注册分岔，
  0.184 与 0.50 的距离远大于代理可能的方法误差（最乐观口径 0.231）。

---

## 7. 文件

| 文件 | 内容 |
|---|---|
| `05d_adaptive_per_image.csv` | 46644 行，每张图 `m / m_prime / n_dynamic / n_under_true_B / tokens_used / retained_trigger_mass` |
| `05d_n_distribution_by_trigger_mass.csv` | 按 n 分布，同时给 image 加权与 **trigger-mass 加权**（含 `share_mass_n_ge` / `share_mass_n_le`） |
| `05d_adaptive_by_category.csv` | 27 类：retention、mass-weighted mean n、`P_mass(n≤15)`、`P_mass(n≥23)` |
| `05d_adaptive_by_dataset.csv` | 同上按数据集 |
| `05d_adaptive_fidelity_proxy.csv` | 长表：每个 (类别, n) 的 `mass_share`、`rho_c_n`、`contribution` 与 `rho_proxy` 合计 |
| `05d_budget_derivation_audit.csv` | §2 的 B 推导偏差逐 cell 记录 |

---

## 8. 复现

```bash
OMP_NUM_THREADS=2 python experiments/model_v0/v3_stage5d_adaptive_audit.py
```

秒级完成，无 GPU。脚本在 `(m2, n)` 与冻结记录不符时直接 `SystemExit`，
不会在审计偏离实现的情况下输出任何分布。
