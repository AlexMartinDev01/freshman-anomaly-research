# Gate 12 结果总账（冻结）：Target-normal-conditioned metric

> 冻结于 tag `gate12_conditional_closed`。
> **判决：预注册判据（D3）FAIL；但它要回答的那个问题被 D1.5 正面回答了。**
> **关键机制发现：metric 相关的 target context 就是协方差本身，而我把它摘要掉了。**

---

## 最终结论（严谨表述）

> **目标正常样本确实决定了一个更好的 defect metric —— 这一点成立，
> 而且有干净的 shuffled-context 对照支持（own − shuffled = +2.6，12/15）。**
> **但成功的实例是闭式 $\Sigma_t^{-1/2}$ whitening（无训练，+2.4，11/15，
> 捕获 oracle gap 约 31%），不是我设计的可学习条件网络。**
> **D2/D3 失败的原因是结构性的：它们的假设类装不下 $\Sigma_t^{-1}$。**

---

## 1. Gate 12 第 1 步：15×15 transfer matrix

先回答「是否某些 source metric 对某些 target 有用」：

| 指标 | 值 |
|---|---|
| off-diagonal 格子 | 210 |
| **delta > 0** | **1**（carpet→wood，+0.67，噪声级） |
| delta < −2 | 190 |
| mean delta | **−13.51** |
| 每个 source 的 mean delta | −6.5 ~ −21.3，**全部为负** |
| 每个 target 的 mean delta | −5.4 ~ −24.4，**全部为负** |

**没有任何 source 的 metric 能帮到任何 target。**
即「选择现有 metric」这条路被彻底关闭，metric 只能从目标自身**生成**。

## 2. Gate 12 主结果

| config | 说明 | mean Recall@1 | vs D0 | better | recovery | 判定 |
|---|---|---|---|---|---|---|
| D0 | raw DINO | 84.7 | — | — | — | — |
| **D1.5** | **target-normal whitening（无训练）** | **87.1** | **+2.4** | **11/15** | **+69.5%** | ✅ **VALID PASS** |
| D2 | 条件对角 metric | 84.3 | −0.4 | 3/15 | −23.3% | 假设类受限，不下结论 |
| D3 | 条件 low-rank adapter | 84.7 | −0.0 | 2/15 | +2.6% | ❌ **INVALID（零梯度）** |
| D3-shuf | 错误 object 的 context | 84.7 | −0.0 | 2/15 | +2.6% | 随 D3 一并作废 |
| D3-const | 平均 context | 84.6 | −0.1 | 0/15 | −1.0% | 随 D3 一并作废 |
| D4 | target-supervised oracle | 92.5 | +7.8 | — | 100% | — |

### ⚠️ D3 必须记为 INVALID，不是 FAIL

D3 与 D3-shuf 逐位相同，**看起来**像是「没用 context」。
但实测最后一层权重**恒等于 0.0**——$z'=z+UV^\top z$ 中 U、V 同时零初始化，
$\partial o/\partial U\propto V=0$、$\partial o/\partial V\propto U=0$，
**参数从未更新**。

所以 D3 只能证明「训练实现无效」，**不能**证明「context 没用」。
**这条区分是方法纪律，必须保住。**

预注册 Basic PASS（D3 > D0 于 ≥11/15）：**2/15，FAIL** ——
但由于 D3 是无效运行，该 FAIL 同样不构成对假设的证据。

## 3. D1.5 的 shuffled-context 对照（关键）

对 whitening 做与 D3 相同的反事实检验：

| | mean gain vs raw | better |
|---|---|---|
| D1.5（自己的 normal 统计） | **+2.4** | 11/15 |
| D1.5（别的高相似 object 的 normal 统计） | −0.3 | 7/15 |
| **own − shuffled** | **+2.6** | **12/15** |

**收益确实来自 target conditioning，不是「任何归一化都行」。**

逐 object：

| object | raw | white(own) | white(shuffled) |
|---|---|---|---|
| grid | 82.7 | **90.7** | 80.4 |
| pill | 78.7 | **85.3** | 80.2 |
| carpet | 82.7 | **88.0** | 85.6 |
| metal_nut | 95.3 | **98.7** | 95.8 |
| transistor | 91.3 | 92.0 | 84.2 |
| wood | 75.3 | 74.0 | 80.2 ← 反向 |
| leather | 96.0 | 95.3 | 97.3 ← 反向 |

（grid/pill/carpet 上是大幅正收益；wood/leather 上 whitening 反而有害。）

## 4. 机制：为什么可学习的 D2/D3 会输给闭式 whitening

**context 的设计出了问题。**

我用的上下文是 $c_t=[\mu_t,\sigma_t]$（768 维）——
它只有**逐维**的均值与标准差，**不含任何跨维相关性**。

而最优 metric $M=\Sigma_t^{-1}$ 一般**不是对角的**。

由此：

- **D2** 只能表达对角 metric $M=\mathrm{diag}(w)$，
  **结构上就无法表示 $\Sigma_t^{-1}$**——不是没训好，是假设类装不下。
- **D3** rank-4 只能低秩逼近全协方差逆。
- **whitening** 用的是完整 384×384 协方差，且**每个 target 在测试时现场估计**，
  根本不需要跨 object 泛化。

**所以正确表述是：**

> **metric 相关的 target context 就是协方差；把 context 摘要成 $[\mu,\sigma]$
> 恰好丢掉了唯一有用的那部分信息。**

这也解释了为什么 D3 的反事实对照完全相同：
context 里没有它需要的东西，网络退化成另一个 universal model。

## 5. 判决与下一支

| 预注册分支 | 判定 |
|---|---|
| Basic PASS：D3 > D0 于 ≥11/15 | ❌ FAIL（2/15） |
| Strong PASS：D3 > D3-shuffle | ❌ FAIL（1/15，因 D3 未用 context） |
| **FAIL 分支条件**：D2/D3 ≤ D0 而 D4 高 | ✅ 命中 |

但预注册 FAIL 分支的**理由**是「normal statistics 无法预测 target metric」——
**这一点被 D1.5 证伪了**：normal statistics 完全能给出一个 +2.4 / recovery 69.5% 的 metric。

所以正确的下一步**不是**转向 multi-level representation，而是：

> **把 conditioning 换回它本该有的形式：以 target 协方差为输入/基底的条件化 metric，
> 而不是 $[\mu,\sigma]$。**
>
> 具体地：whitening 已经拿到 +2.4 / 31%（按均值比），
> 下一步是学一个**在 whitening 之上的修正项**（残差 metric），
> 而不是从零学一个 metric。

## 6. 限制

- D1.5 的 recovery 有两种算法：逐 object 比的均值 **69.5%**，
  与均值之比 **+2.4/+7.8 = 31%**。前者被少数 $(D4-D0)$ 很小的 object 抬高，
  **应以 31% 为准**，69.5% 仅作参考。
- whitening 的特征值下限取均值 5%，这是人为选择，未做敏感性扫描。
- 15 个 object 是硬上限；条件网络只有 14 个训练 object，容量本就受限。

## 7. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `gate12_matrix.py`、`gate12_conditional.py` |
| 结果 | `metrics/gate12_matrix.csv`、`gate12_conditional.csv`、`gate12_whiten_ctrl.csv` |
| 权重 | `checkpoints_g12/{source}.pt`、`checkpoints_g12c/{target}_{D2,D3}.pt` |

## 8. 本次踩的两个坑（都是「看起来在训练、实际零梯度」）

1. **双线性鞍点。** $z'=z+UV^\top z$ 里把 $U,V$ **同时**零初始化，
   则 $\partial o/\partial U\propto V=0$、$\partial o/\partial V\propto U=0$，
   **两个因子梯度都恒为 0**，adapter 永远停在 identity
   （实测最后一层权重绝对值恒等于 0.0）。
   **零初始化残差技巧只适用于加法项，不适用于乘积因子。**
2. **铰链饱和。** D3 初始即 identity，而 raw DINO 本来就满足
   $d_{ap}+0.2<d_{an}$，于是 `relu(...)` 恒为 0、梯度恒为 0。
   改用 softplus 才有信号。**这与 Gate 9B 的 rank_loss 是同一个陷阱，第二次踩到。**

---

## 继承到下一阶段的强制方法纪律

1. **条件化必须把「metric 所需的信息」放进去。** 目标是学 metric 却只喂 $[\mu,\sigma]$，
   等于结构上排除了正解；**先确认假设类装得下已知解**（本例中即 $\Sigma^{-1}$）。
2. **每个可学习模块都要先验证它能否表达一个已知的平凡解。**
   D2 连 whitening 都表示不了，训练再久也没用。
3. **零初始化在乘法结构中是鞍点**，不是「安全的起点」。
4. **recovery 必须同时报告逐 object 均值与均值之比**，
   两者差异大时说明结论被少数 object 主导。
