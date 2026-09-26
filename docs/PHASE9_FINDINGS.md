# Gate 9A 结果总账（冻结）：Learned Target-Conditioned Defect Selector

> 冻结于 tag `gate9a_selector_closed`。
> **判决：学习式 Selector 失败。命中用户预先写死的第二支 → 进入 Gate 9B。**
> 关键点：失败不是「判据没选好」，而是 **target × bank 的交互项本身没有可迁移信号**。

---

## 预注册的判决标准（用户给定，未事后修改）

```
baseline < learned selector < oracle external class     → 继续 Retrieval / Reweighting
learned selector ≈ R1 / random                          → 进入 Conditional Feature Generator
```

---

## 最终结论（严谨表述）

> **在单个 target 内部，「哪个外部 bank 有用」是稳定且收益可观的
> （wallplugs 上 fabric 在 89% 的 episode 中都是最优，相对 baseline +15.2）。
> 但这个对应关系无法从目标的正态外观学到，也无法从其他 target 迁移过来：
> 交互项在同一数据集内 LOCO 与跨数据集迁移下都不显著。**
> **「选择已有 exemplar」这一操作本身不足。**

---

## 1. 实验设计

冻结 DINOv2 / 正常分支 / raw fusion β=1，**唯一新增组件是 selector**。

```
z_t = proj(h_target)         目标 normal 外观摘要
z_b = proj(h_bank)           候选外部 defect bank 摘要
s   = MLP([z_t, z_b, z_t*z_b, scalars])    预测 utility
```

- `utility` = 用该 bank 做 `S_n − S_d` 在目标全部测试图上的 AUROC。
- 训练用 pairwise ranking loss（决策是「哪个 bank 排第一」，不是拟合 AUROC 数值）。
- **训练池 = MVTec AD v1（15 类，210 个有序对）**；**测试 = MVTec AD 2（8 类）**。
  这样 **AD2 的任何类别都不参与训练**（连作为 bank 也不参与）。
- 三个变体，消融才是重点：

| 变体 | 含义 |
|---|---|
| `target_only` | 只用 h_target —— 不看 bank，无法做选择 |
| `bank_only` | 只用 h_bank —— 「某个 bank 全局都好」 |
| `interaction` | 两者 + 逐元素乘积 —— **本次要检验的核心假设** |

## 2. v1 内部 LOCO：交互项输了

15 类留一，per-episode utility：

| 变体 | learned − random | 关闭 oracle gap | beat baseline |
|---|---|---|---|
| `bank_only` | **+2.90**（15/15 类更好） | 56.4% | 2/15 |
| `interaction` | +1.14（14/15） | 38.3% | 2/15 |
| `target_only` | +0.64（8/15） | 11.9% | 1/15 |

**`bank_only` 明显优于 `interaction`。** 即：能排序 bank 的那部分信号是
**bank 身份本身**，不是「这个 bank 对这个 target 有用」。交互项没有增加任何东西。

同时 v1 本身几乎无 headroom：baseline 96.2 → oracle 97.1（**+0.9**）。
v1 的 utility 排序实质是「哪个 bank 破坏最小」，不是「哪个 bank 能帮上忙」。

## 3. 迁移到 AD2：全部失效

训练用 v1 全部 15 类，测试 AD2 全部 8 类（每类 4 shot × 3 seed × 3 size）：

| 变体 | learned − random | t | learned > baseline |
|---|---|---|---|
| `target_only` | +3.31 | 1.34 | 40% |
| `interaction` | −0.09 | −0.09 | 26% |
| `bank_only` | −1.06 | −0.68 | 22% |

**全部不显著，且两个真正依赖 bank 的变体低于随机。**

⚠️ `target_only` 的 +3.31 是**假象**：它无法对 bank 排序，argmax 退化成常数，
8 个 target 中有 7 个永远选 `can`。它测的不是选择能力。

## 4. 同数据集 LOCO（排除「只是 v1 太简单」）

在 AD2 内部做 leave-one-object-out，让训练与测试处于**同一 regime**：

| 变体 | learned − random | t | episode > random | episode > baseline |
|---|---|---|---|---|
| `target_only` | +3.31 | 1.34 | 66% | 40% |
| `bank_only` | +0.97 | 0.95 | 56% | 28% |
| `interaction` | +1.10 | 0.51 | 52% | 35% |

**仍然全部不显著。** 所以失败不是 regime 错配造成的。

## 5. 最强 target-independent 策略也失败

如果 selector 完全放弃 target 条件化，退化成「挑在其他 target 上平均最好的 bank」：

| target | LOO 选中的 bank | 该策略 | random | oracle | baseline |
|---|---|---|---|---|---|
| can | fabric | 47.1 | 44.0 | 48.8 | 43.6 |
| fabric | can | 50.8 | 52.4 | 65.1 | 53.9 |
| fruit_jelly | can | 64.2 | 69.8 | 78.0 | 83.2 |
| rice | can | 71.2 | 67.0 | 74.4 | 79.3 |
| sheet_metal | can | 81.4 | 80.4 | 84.2 | 83.7 |
| vial | can | 78.2 | 70.8 | 83.8 | 83.1 |
| wallplugs | can | 42.8 | 42.4 | 55.3 | 40.1 |
| walnuts | fabric | 66.7 | 69.6 | 87.5 | 77.2 |

**LOO-global 只在 2/8 上超过 baseline，平均 Δ = −5.2**（oracle 为 +4.1）。

注意它反复选 `can`——因为 can 只有 54 个 defect patch，bank 越小破坏越小。
**这个策略学到的是「谁破坏最小」，不是「谁能帮上忙」。**

## 6. 但「选择本身有价值」这件事是真的

这是 Gate 9A 留下的、必须带进下一阶段的正面事实。

**每个 target 的最优 bank 在内部高度稳定**（8 个 target 中最优 bank 的 episode 占比）：

| target | 最优 bank | 占比 |
|---|---|---|
| wallplugs | fabric | **89%** |
| walnuts | can | **89%** |
| sheet_metal | fruit_jelly | 72% |
| fabric | fruit_jelly | 69% |
| can | fabric | 53% |
| vial | rice | 50% |

**收益也是真的**（oracle − baseline）：wallplugs **+15.2**、fabric +11.2、walnuts +10.3、can +5.2。

**所以问题不是「没有可选项」，而是「选不出来」。**
（这也与 Gate 8 一致：fabric/walnuts 对 wallplugs 有 +20 的真实、稳定、缺陷特异的增益，
但任何几何判据都识别不出来。）

## 7. 判决

按预注册标准：

```
learned selector (bank_only +0.97, interaction +1.10, 均不显著, >baseline 仅 28-35%)
        ≈ random
        ≪ oracle
```

**Gate 9A 失败 → 进入 Gate 9B：Conditional Feature Generator。**

### 必须同时声明的限制

- **样本量的硬上限**：AD2 只有 8 个 target、v1 只有 15 个，
  不同的 (target, bank) 关系总共约 100–200 个。
  「从这个数量学不出来」**不等于**「原理上学不出来」。
- 所以 Gate 9B 的合理性**不是**「selector 试过了没用」，
  而是一个有明确理由的**归纳偏置差异**：

  > selector 要学的是「(target, bank) → 标量」的**离散选择映射**，
  > 对未见 target 没有共享结构可迁移；
  > generator 学的是「(z_ext, h_target) → ẑ」的**共享变换**，
  > 同一个变换可以作用到任何 target 上。

  后者参数共享方式不同，可能在同样的样本量下可学。**这是假设，不是保证。**

## 8. 资产

| 类型 | 内容 |
|---|---|
| 缓存 | `results/model_v0/cache_v1/`（MVTec AD v1 全部 15 类 train+test） |
| 表 | `metrics/selector_{v1,ad2}.npz`（v1: 7560 行 / 210 对；AD2: 2016 行 / 56 对） |
| 脚本 | `cache_features_v1.py`、`gate9a_dataset.py`、`gate9a_selector.py`、`gate9a_transfer.py` |
| 结果 | `metrics/gate9a_loco_{v1,ad2}.csv`、`gate9a_transfer_ad2.csv` |

## 9. 本次踩的坑

1. **`meta.size` 静默返回行数**，不是名为 `size` 的列。pandas 属性遮蔽列名，
   报错信息（`'int' object has no attribute 'astype'`）完全不指向真因。
   已改用 `meta["size"]`。
2. **`spread_of` 最初用 `make_dist(t)(t)`**，那是「每个 patch 到自己的距离」≈0，什么都没测。
   改为真正的成对 1-NN（排除自身）。
3. **ranking loss 每 epoch 重建 pair 集合**，是运行时瓶颈；pair 只依赖标签，已预计算一次。

---

## 继承到 Gate 9B 的强制方法纪律

1. **预注册的判决标准不得事后修改。** 本次标准在跑之前写死，结果命中第二支即执行。
2. **变体消融必须包含「不能条件化的退化版」**（`target_only`），
   否则会把 argmax 退化当成能力（本次 +3.31 差点被误读为成功）。
3. **同一数据集内 LOCO 与跨数据集迁移都要跑**，否则无法区分「学不会」与「迁移不了」。
4. **必须报告「最强 target-independent 策略」**（LOO-global），
   它是任何选择类方法的上界代理，也是判断「选择是否足够」的基准。
5. **负结果必须附带正面事实**（§6：最优 bank 稳定且收益真实），
   否则下一阶段会失去问题定义。
