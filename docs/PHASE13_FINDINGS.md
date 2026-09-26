# Gate 13 结果总账（冻结）：Partial Covariance Correction + Shared Residual

> 冻结于 tag `gate13_closed`。
> **Gate 13A PASS：γ=0.5 是干净 Pareto 点，baseline 与 oracle 同时上升。**
> **Gate 13B FAIL：在协方差 metric 之上叠加 shared structured residual 无效。**
> 顺带**撤回 Gate 12B 的一个结论**（见 §4）。

---

## 最终结论（严谨表述）

> **部分协方差校正（γ=0.5）是目前的表示接口：
> baseline 84.7→86.7（12/15），oracle 89.8→91.8，两条曲线同时上升。**
> **但在此之上叠加一个 source 训练的 shared structured residual，
> 在 r=4/8/16 三档全部低于 E1（−0.84 / −1.91 / −1.29）。
> 即：剩余的 defect 校正本身是 object-dependent 的，
> 不能用一个跨对象共享的低秩残差替代。**

---

## 1. Gate 13A：fractional whitening 的 Pareto 扫描

$M_t(\gamma)=(\Sigma_t+\epsilon I)^{-\gamma}$，γ=0 为 raw，γ=1 为 full whitening。

| γ | baseline | oracle | base−raw | oracle−raw | eff_rank | baseline 更好的 object |
|---|---|---|---|---|---|---|
| 0.00 | 84.7 | 89.8 | +0.00 | +0.00 | 40.0 | — |
| 0.25 | 85.6 | 89.8 | +0.93 | +0.00 | 57.1 | 8/15 |
| **0.50** | **86.7** | **91.8** | **+1.96** | **+2.00** | 70.5 | **12/15** |
| 0.75 | 87.1 | 89.7 | +2.36 | −0.13 | 77.0 | 11/15 |
| 1.00 | 87.1 | 90.8 | +2.36 | +1.02 | 78.8 | 11/15 |

**γ=0.5 是 balanced operating point**（Pareto knee），
不是「唯一 Pareto 点」——严格按双目标 Pareto 定义，γ=0.5 与 γ=1
**互不支配**（前者 oracle 高，后者 baseline 高），两者都在 frontier 上。

选 γ=0.5 的理由是它同时满足：baseline +1.96、oracle +2.00、
object 一致性最好（12/15）、且比 full whitening 留下更多 oracle headroom。

eff_rank 随 γ 单调上升（40→79），说明协方差校正是在**展开**表示，
而不是压缩。

> **不需要训练任何模型，表示接口就已经确定在 γ=0.5。**

## 2. Gate 13B：shared residual 失败

$M_t=\Sigma_t^{-\gamma^*}+UU^\top$，$U$ 由 14 个 source object 训练。

| config | mean | vs E1 | better | vs E0 | better |
|---|---|---|---|---|---|
| E0 raw | 84.7 | — | — | — | — |
| **E1 covariance γ=0.5** | **86.7** | — | — | +1.96 | 12/15 |
| E2 r=4 | 85.8 | **−0.84** | 5/15 | +1.11 | 9/15 |
| E2 r=8 | 84.8 | **−1.91** | 2/15 | +0.04 | 6/15 |
| E2 r=16 | 85.4 | **−1.29** | 4/15 | +0.67 | 7/15 |
| E3 = γ* oracle | 91.8 | +5.12 | — | +7.13 | — |

**三个 rank 全部低于 E1。预注册判据 FAIL。**

**health 正常**：update_norm 12.54（min 9.92）、|grad| 0.044、active-triplet 0.895，
**0/45 无效运行**。所以这是真实的科学失败，不是零梯度。

## 3. 判决与下一支

预注册分岔：

| 分支 | 判定 |
|---|---|
| E2 > E1 且多数 object 正向 | ❌ FAIL |
| **E2 ≤ E1 而 E3 ≫ E1** | ✅ **命中**（−0.84 ~ −1.91，而 E3 高出 5.12） |

命中第二支的结论：

> **协方差校正解决了 object nuisance；
> 剩下的 defect 校正仍然是 object-dependent 的，
> 无法用一个跨对象共享的低秩残差补上。**

因此下一步是 **Gate 14：covariance-conditioned residual**：
让残差本身依赖 $\Sigma_t$，即

$$A_t = I + R_\theta(\Sigma_t)$$

（而不是继续寻找 universal residual）。

**但不要把整个 $\Sigma_t$ 直接塞进 MLP。** 建议先把协方差投影到一个
共享 global basis $U_g\in\mathbb{R}^{D\times k}$（k=16/32，从所有 source normal 上取），
用 $C_t=U_g^\top\Sigma_tU_g\in\mathbb{R}^{k\times k}$ 作为条件输入，
输出一个小残差 $M_t=\Sigma_t^{-\gamma^*}+U_gB_tU_g^\top$，
其中 $B_t=h_\theta(C_t)$。

## 4. 撤回 Gate 12B 的一个结论

初版 Gate 12B 写道：

> 「raw oracle 92.6 → whitened oracle 90.8，说明 whitening 删掉了一部分真正有用的结构。」

**该比较无效。** 92.6 来自 `gate10b`（5 负例采样器 + hinge），
90.8 来自 `gate12b`（1 负例采样器 + softplus）——**两个不同脚本**。

用**同一采样器**重测（本文 §1）：

| γ | oracle |
|---|---|
| 0.0 raw | 89.8 |
| 0.5 | **91.8** |
| 1.0 whitening | 90.8 |

**whitening 并没有降低上界，γ=0.5 反而把它抬高 +2.0。**

所以 **「whitening 损失 defect geometry」这一说法撤回**。
Gate 12B 的 W3 崩坏（63.8 vs 87.1）仍然成立——
但它的解释应改为「whitened 表示不利于跨对象**共享** metric」，
而不是「whitening 丢掉了信息」。

## 5. 本次踩的两个坑（都是影响结论的静默错误）

1. **RNG 流交叉污染。** `recall_one` 在两次 `rng.choice` 之间夹了
   `rng.integers`，导致「同一个」采样规则在不同脚本里抽到**不同的 query patch**，
   单独一项就让 E1 差了约 5 AUROC。
   **修正：所有 query 一次性抽完。**
2. **未归一化特征送入点积排序器。** E1 最初把 norm≈45 的原始 whitened 特征
   直接传给 `recall_one`，点积由模长主导而非方向。
   **修正：`l2norm` 后再排序。**
   两者都让 E1 从 81.7 回到正确的 86.7。

> 这两个都不是「方法失败」，而是**跨脚本数值不一致**。
> 一旦用它们做跨实验比较，就会产生虚假的科学结论。

## 6. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `gate13a_fractional.py`、`gate13b_shared_residual.py` |
| 结果 | `metrics/gate13a_fractional.csv`、`gate13a_health.csv`、`gate13b_shared_residual.csv` |

---

## 继承到下一阶段的强制方法纪律

1. **跨脚本比较必须同源。** 采样器/损失/归一化只要有一项不同，
   数字就不能直接比。本次 12B 的错误结论正是这样产生的。
   **新规矩：任何跨 gate 的数字比较，必须在同一进程内用同一函数重算一次。**
2. **随机采样要在循环外一次性抽完**，否则流交叉会让"同种子"失效。
3. **排序/检索类函数入口处统一做 l2norm**，不给调用方留未归一化的可能。
4. **health 三件套（update_norm / grad_norm / active-ratio）继续强制**——
   本次 13B 正是靠它确认「确实训练了但失败了」，而不是又一次零梯度。
