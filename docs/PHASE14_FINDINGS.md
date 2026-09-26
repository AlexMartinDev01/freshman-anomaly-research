# Gate 14 结果总账（冻结）：Residual Oracle + Covariance-Conditioned Metric

> 冻结于 tag `gate14_closed`。
> **Gate 14A：residual 模型族可行（rank-16 关闭 39% headroom），
> 且 covariance 显著预测 residual 几何（r=+0.435, p<0.001）。**
> **Gate 14B：但把它学出来 FAIL —— F1 与 F3（常数 context）逐位相同。**

---

## 最终结论（严谨表述）

> **剩余 headroom 确实存在且可被低秩 residual 表达（rank-16 关闭 39%）；
> target covariance 也确实与所需 residual 的几何显著相关。
> 但以此为条件训练出的 metric，
> 相比常数 context **没有任何作用**（F1−F3 = −0.00），
> 且整体低于协方差 baseline（−4.67）。**
> **即：二阶协方差能描述「base metric 该怎样」，
> 但不足以预测「defect-specific residual 该是什么」。**

---

## 1. Gate 14A：三个子问题

### (1) residual 模型族能不能表达 headroom？

在 γ=0.5 base metric 上，用 **target 自己的 defect label** 拟合
$M_t=\Sigma_t^{-0.5}+U_tU_t^\top$：

| config | mean | vs E1 | better | 关闭 gap |
|---|---|---|---|---|
| E1 covariance | 86.7 | — | — | — |
| E1+U r=4 | 87.4 | +0.71 | 7/15 | 13.9% |
| E1+U r=8 | 87.9 | +1.20 | 8/15 | 23.4% |
| **E1+U r=16** | **88.7** | **+2.00** | **10/15** | **39.0%** |
| E3 full oracle | 91.8 | | | 100% |

**随 rank 单调上升（13.9% → 23.4% → 39.0%）。模型族可行，但需要容量。**

### (2) covariance 相似 → residual 相似吗？

$\log\Sigma$ 的 log-Euclidean 距离 vs residual 子空间的主角距离，
**按 object 做置换检验**（105 个 pair 不独立，不能直接看 pearson）：

| rank | observed r | null (n=1000) | p |
|---|---|---|---|
| 4 | +0.250 | +0.005 ± 0.118 | 0.035 |
| 8 | +0.401 | +0.005 ± 0.120 | **<0.001** |
| 16 | **+0.435** | +0.006 ± 0.129 | **<0.001** |

**显著。协方差相似的 object，需要的 residual 也相似。**

> ⚠️ 注意：早期用 100-step（未训练）residual 算得 r=+0.145~0.213，
> **不显著**。用训练充分的 residual 才得到上述结果。
> **这是一个「中间产物不能拿来下结论」的实例。**

### (3) covariance-NN residual 直接借用

取 covariance 最近的 source 的 residual 直接用（nn1/nn3/nn5）：

| | r=4 | r=8 | r=16 |
|---|---|---|---|
| nn1 | 74.0 | 69.9 | 73.2 |
| nn3 | 75.1 | 71.5 | 74.7 |
| nn5 | 76.3 | 72.6 | 73.0 |

（E1 = 86.7）**全部低 10~17 分。**

这不否定 (2)：整体借用一个 residual 与「covariance 含预测信息」是两回事。
但它说明**朴素迁移不可用**。

## 2. Gate 14B：学出来的 conditioner FAIL

$M_t=\Sigma_t^{-0.5}+\sum_k\alpha_k(C_t)B_kB_k^\top$，
$C_t=P^\top\Sigma_tP$（P 为 source normal 上的固定基，kp=32），
$\alpha$ 由 $\log(C_t)$ 经小 MLP 给出，K=4，rank=16，**LOCO**。

| config | mean |
|---|---|
| F0 covariance only | 86.7 |
| F1 correct covariance | **82.0** |
| F2 shuffled covariance | 81.3 |
| F3 constant context | **82.0** |

| 判据 | 结果 |
|---|---|
| F1 − F0 ≥ +1.0 且 ≥10/15 | ❌ **−4.67，1/15** |
| F1 > F2 | +0.71（6/15）—— 弱 |
| **F1 > F3** | ❌ **−0.00（5/15）** |

**F1 与 F3 逐位相同。** conditioner 输出的 metric 与常数 context 无法区分。

**health 正常**：update_norm 19.39（min 17.86）、active 0.918。
**确实训练了，确实失败了。**

### 一个值得记录的细节

逐 object 看，$\alpha$ **数值上差异很大**
（bottle `[0.003,2.44,1.55,0.004]` vs grid `[0,4.0,0.001,0]`），
**但 F1 ≈ F3**。

> 即：**conditioner 输出的权重确实随协方差变化，但这个变化对检索结果没有影响。**
> 「条件化在数值上生效」≠「条件化在功能上生效」——
> 只看 α 会得出完全相反的结论。

## 3. 判决与下一支

预注册分岔：

| 分支 | 判定 |
|---|---|
| F1 − F0 ≥ +1.0，≥10/15，且 F1 > F2、F1 > F3 | ❌ FAIL |
| **14A PASS 但 14B FAIL** | ✅ **命中** |

命中该支的结论：

> **target normal 的**二阶协方差**能改善 base metric，
> 但不足以预测 defect-specific residual。
> 不能再继续换更大的 MLP —— 已经回答了 $\Sigma_t \nRightarrow \Delta M_t$。**

下一阶段应转为**从 target normal 分布的更丰富结构取 context**，
而不是只使用二阶协方差：

- local prototype distribution；
- mixture components（GMM / k-means 原型）；
- normal patch graph / 局部邻域结构；
- multi-layer feature statistics（浅层+深层）。

## 4. 一个贯穿 13B/14B 的模式

| gate | base metric | 加上 shared/conditioned residual |
|---|---|---|
| 13B | E1 86.7 | E2 84.8–85.8（shared，三种 rank） |
| 14B | F0 86.7 | F1 82.0（conditioned） |

**在 γ=0.5 协方差 metric 之上叠加任何学到的 defect-space residual，
都使结果变差。** 这本身是一个一致的负结果，
说明「在已经校准好的 metric 上再学一个 defect 方向」这条路在现有数据量下不成立。

## 5. 本次踩的三个坑

1. **`torch.log` 是逐元素的。** $P^\top\Sigma P$ 是 PSD 但**有负的非对角元**，
   逐元素 log 直接产出 **NaN** α。NaN 又经 `0 * NaN = NaN` 穿过残差门，
   让所有 target 都在垃圾上打分（recall ≈ 23，且 F1/F2/F3 全同）。
   **修正：用真正的矩阵对数（eigh → log → 重建）。**
2. **用可学习标量把残差门控到 0，又造了一个鞍点。**
   $f=\mathrm{cat}([w,\ \mathrm{scale}\cdot\mathrm{extra}])$，
   $\partial L/\partial B = \mathrm{scale}\cdot(\dots)=0$。
   scale 停在 0.0、update_norm 0.004、300 步没动。
   **这是本项目第三次撞上「零初始化放在乘积结构里」**（Gate 12 D3、Gate 13B 一度、
   Gate 14B）。**修正：不对残差门控，而是把 B 初始化得足够小。**
3. **未训练的中间产物不能拿来计算统计量。**
   14A 的相关系数用 100-step residual 时不显著，用 1500-step 时 p<0.001。

## 6. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `gate14a_residual_oracle.py`、`gate14b_conditioner.py` |
| 结果 | `metrics/gate14a_residual_oracle.csv`、`gate14a_transfer.csv`、`gate14b_conditioner.csv` |
| 权重 | `checkpoints_g14a/{target}_r{4,8,16}.pt`、`checkpoints_g14b/{target}.pt` |

---

## 继承到下一阶段的强制方法纪律

1. **「条件化在数值上生效」必须与「条件化在功能上生效」分开检验。**
   本次 α 逐 object 差异巨大，但 F1 ≡ F3。
   **任何 conditioning 方法都必须带 shuffled / constant 反事实，且以反事实为准。**
2. **零初始化在乘积结构中一律是鞍点。** 已经三次。
   **规则：任何新模块先验证 $\|\theta_{final}-\theta_{init}\| \gg 0$ 再解读结果。**
3. **中间/未收敛产物不得用于计算结论性统计量。**
4. **矩阵运算先确认是逐元素还是矩阵语义**（`torch.log` vs 矩阵对数）。
   NaN 会静默穿过所有后续计算，只在最终指标上表现为「全都一样」。
