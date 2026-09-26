# Gate 12B 结果总账（冻结）：Whitening 消融 + Whitened Structured Transfer

> 冻结于 tag `gate12b_closed`。
> **判决：whitening 的收益被完全归因到 cross-dimensional covariance（消融成立）；
> 但 whitening 不能让跨对象 metric 迁移变得有用 —— 反而更差（W3 63.8 vs W1 87.1）。**

---

## 最终结论（严谨表述）

> **target normal 的完整协方差确实带来全部收益：
> centering(−0.58) 与 diagonal z-score(−0.22) 都无效，
> full whitening 有效(+2.36)，且 A3−A2 = +2.58（12/15），
> 在 eigenvalue floor 与 shrinkage 之间稳定。**
> **但把每个 object 各自 whitening 之后再训练共享 metric，
> 迁移会**崩坏**（−23.3，0/15）。
> 机制：whitening 是一个**逐 object 的线性映射**，它在去掉 object nuisance 的同时
> **摧毁了 raw DINO 提供的共享坐标系**。**

---

## 1. 消融：收益来自哪里

cross-image same-defect Recall@1，相对 raw 的增益：

| config | mean | median | better |
|---|---|---|---|
| A1 centering `z − μ` | **−0.58** | +0.00 | 6/15 |
| A2 diagonal z-score `(z−μ)/σ` | **−0.22** | +0.67 | 8/15 |
| **A3 full whitening `Σ^{-1/2}(z−μ)`** | **+2.36** | +2.67 | **11/15** |
| **A3 − A2** | **+2.58** | — | **12/15** |

**A1/A2 完全没有效果。全部收益来自 cross-dimensional covariance。**

数值稳定性（`floor.05` 之外的设定）：

| 正则 | mean gain |
|---|---|
| floor 0.01 / 0.05 / 0.10 / 0.20 | +2.84 / +2.36 / +1.91 / +1.82 |
| shrink 0.01 / 0.10 / 0.50 | +2.44 / +2.49 / +1.20 |
| pca 64 / 128 / 256 | −2.93 / +0.18 / +2.18 |

eigenvalue floor 与 shrinkage 之间稳定在 +1.2 ~ +2.8，
**不是数值偶然**。PCA 截断则是另一回事（丢掉方向会伤害 1-NN），不参与该结论。

## 2. Gate 12B：whitened transfer 失败

| config | 说明 | mean |
|---|---|---|
| W0 | raw DINO | 84.7 |
| **W1** | **target whitening（无训练）** | **87.1** |
| W3 | 各 object 先 whitening，再训练共享 metric | **63.8** |
| W4 | whitened 空间下 target 监督 oracle | 90.8 |

逐 object（W1 vs W3）：

| target | W1 | W3 | | target | W1 | W3 |
|---|---|---|---|---|---|---|
| bottle | 88.7 | 60.7 | | metal_nut | 98.7 | 87.3 |
| grid | 90.7 | 44.0 | | tile | 99.3 | 66.7 |
| screw | 88.7 | 36.7 | | wood | 74.0 | 69.3 |
| transistor | 92.0 | 84.0 | | zipper | 62.0 | 40.0 |

**W3 − W1 = −23.29，0/15 更好。预注册判据 FAIL。**

**W3 的 effective rank 只有 1.9–4.6**（W1 是 36–129），
即 whitened 空间里的共享 metric 同样发生了塌缩。

## 3. 排除替代解释

**(a) 不是零梯度。** update_norm = 26.3（min 26.1），grad_norm 0.30，
active-triplet 0.96。**训练是有效的**（这条纪律现在已强制记录）。

**(b) 不是负例池失衡。** `oth` 含 normal patch（数量约为 defect 的 10 倍），
改为只用 defect 负例后：

| target | W1 | W3(全部负例) | W3(仅 defect 负例) |
|---|---|---|---|
| bottle | 88.7 | 60.7 | 65.3 |
| grid | 90.7 | 44.0 | 50.7 |
| screw | 88.7 | 36.7 | 37.3 |
| tile | 99.3 | 66.7 | 74.7 |

有改善但**仍然远低于 W1**，rank 仍塌缩在 4.5–5.6。**失衡是次要因素，不是主因。**

## 4. 机制：whitening 摧毁了共享坐标系

W1 与 W4 的关系给出了关键线索：

- **raw 空间的 oracle（Gate 10B B2）= 92.6**
- **whitened 空间的 oracle（W4）= 90.8**

（两者采样器不同，非严格可比，但方向清楚：**whitening 降低了可学到的上界**。）

于是形成一个干净的张力：

| | 共享坐标系 | object nuisance |
|---|---|---|
| **raw frame** | ✅ 所有 object 共用一套坐标 | ❌ 污染严重（W0=84.7） |
| **whitened frame** | ❌ 每个 object 被各自线性变换，坐标不再对齐 | ✅ 已去除（W1=87.1） |

> **whitening 是逐 object 的线性映射：它去掉 nuisance 的同时，
> 把 raw DINO 里本来共享的方向在每个 object 中旋转成了不同方向，
> 于是跨对象对齐被破坏，共享 metric 再也无法迁移。**

这也解释了为什么 W3 会塌缩到接近二分类的低秩：
在失去共享坐标系之后，唯一还能跨对象成立的对比只剩「defect vs normal」。

## 5. 判决与下一支

按用户预注册的分岔：

| 分支 | 判定 |
|---|---|
| W3 > W1 且 ≥10/15 | ❌ FAIL（0/15） |
| W3 ≤ W1 而 W4 ≫ W1 | ✅ 命中（63.8 ≤ 87.1，90.8 > 87.1） |

但该分支的**原表述**（「协方差归一化解决部分 nuisance，剩余 metric 仍 object-dependent」）
需要修正为更准确的版本：

> **协方差归一化确实解决了 nuisance，但它同时破坏了跨对象对齐。
> 问题不是「剩余 metric 仍然 object-dependent」，
> 而是「逐 object 归一化与跨对象共享坐标系在结构上冲突」。**

因此下一步**不是**在 whitened 空间上叠加 conditional residual，
而是在 **raw 共享坐标系下**做 covariance-conditioned residual：

$$A_t = I + R_\theta(\Sigma_t)$$

即：**保持 raw frame 的对齐性，只学一个由 target 协方差驱动的修正项**。
这样既利用协方差信息，又不破坏共享坐标。

## 6. 限制

- W4 与 Gate 10B 的 B2 使用了不同的三元组采样器，两者差值的解释需谨慎。
- `whiten_all` 用 20000 个 normal patch 估计 384×384 协方差，
  同图 patch 非独立，实际自由度低于名义值。
- W3 的负例池失衡（normal 约占 10 倍）未在正式运行中修正；
  补救实验显示影响约 +5，不改变结论。

## 7. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `gate12b_whiten_ablation.py`、`gate12b_transfer.py` |
| 结果 | `metrics/gate12b_ablation.csv`、`gate12b_transfer.csv` |

---

## 继承到下一阶段的强制方法纪律

1. **零梯度与「方法无效」必须分开记账。** D3 已改记为 INVALID；
   本次 W3 通过 update_norm/grad_norm/active-triplet 三项确认「确实训练了」，
   其失败才是科学意义上的失败。
2. **任何一个「有效」的归一化/变换，必须先做消融确认收益来源。**
   本次 centering 与 z-score 双双无效，才把结论锁定在协方差上；
   只看 full whitening 会误以为「任何归一化都行」。
3. **寻找替代解释要主动构造**（本次：负例池失衡、数值正则敏感性）。
   两者都排除后，结论才成立。
