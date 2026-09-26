# Gate 10A 结果总账（冻结）：Cross-Object Anomaly Semantic Projection

> 冻结于 tag `gate10a_projection_closed`。
> **判决：失败，且这一次的失败与 testbed headroom 无关。**
> 依据是强制 monitor（identity 去除代价）与 λ 扫描，两者都不依赖下游 headroom。

---

## 最终结论（严谨表述）

> **在冻结 DINO 特征上，用 source anomaly supervision 学到的 384→256→128 投影
> 确实能让外部缺陷分支变得更有用（相对 raw 提升 +3.19，t=3.30，13/15，
> 而 raw external 本身是有害的）。**
> **但「主动去除 object identity」这一机制被证伪：**
> **加上 gradient reversal 后，identity 下降的同时 anomaly 判别力同步崩塌，
> 且 λ 扫描显示不存在任何能落在「identity 降 / anomaly 保」象限的 λ。**
> **因此不是超参数问题，是两者在特征空间中纠缠。**

---

## 1. 设置

正常分支**完全不动**（Gate 6A 已证明动它会导致 support overfitting）。只投影缺陷分支：

```
z --P_theta--> h          384 -> 256 -> 128
L_anom = BCE(anom_head(h), is_defect)
L_obj  = CE(obj_head(GRL(h)), object_id)
L = L_anom + λ · L_obj
```

配置：`R0` raw DINO｜`R1` 仅 `L_anom`｜`R2` `L_anom + GRL`。
oracle **在该 representation 下**计算。15 类 LOCO，2 seeds，1500 steps。

## 2. 强制 monitor：identity 去除的代价

| config | object-ID acc | anomaly acc |
|---|---|---|
| raw DINO | 1.000 | 0.947 |
| **R1**（仅 anomaly） | 1.000 → **0.930** | 0.947 → **0.997** |
| **R2**（+ GRL，λ=1.0） | 1.000 → **0.794** | 0.947 → **0.810** |

逐 fold（R2）：obj 下降 0.19–0.25，anom 下降 0.11–0.15。
**15/15 个 fold 都是「identity 降、anomaly 同时降」**，无一例外。

### λ 扫描：不存在可用的 λ

| λ | object-ID | anomaly |
|---|---|---|
| 0.00 | 0.939 | **0.996** |
| 0.05 | 0.930 | **0.867** |
| 0.10 | 0.927 | 0.838 |
| 0.30 | 0.902 | 0.815 |
| 1.00 | 0.872 | 0.813 |
| 3.00 | 0.862 | 0.811 |

**在 λ=0.05（最轻的对抗强度）就已经用 0.129 的 anomaly 精度换 0.009 的 identity 下降。**
λ 加到 3.0 也只把 identity 压到 0.862，而 anomaly 早已停在 0.81。
**整条曲线在对角线附近滑动 —— 这正是 Gate 4 whitening 的失败形态。**

> 按预注册规则：「如果 object identity ↓ 而 anomaly information 也一起 ↓，
> 那就重复了 Gate 4 whitening 的问题……这种情况也应该直接判失败。」
> **该条件已被满足。**

## 3. 下游结果（v1 LOCO，paired over 15 folds）

| 对比 | mean | t | 更好 fold 数 |
|---|---|---|---|
| **R2 − R1** | **−0.43** | −0.89 | 6/15 ❌ |
| **R2 − baseline** | **−0.00** | −0.73 | 7/15 ❌ |
| R1 − R0 | **+3.19** | **3.30** | 13/15 ✅ |
| R1 − baseline | +0.42 | 0.89 | 8/15 |

**预注册判据：**
- R2 > R1 ❌ FAIL
- R2 > baseline 多数类 ❌ FAIL（7/15）

## 4. 一个必须记录的正面结果

**R1 − R0 = +3.19（t=3.30，13/15）**：anomaly supervision 学出的投影，
让外部缺陷分支显著优于 raw DINO。

但 **R1 − baseline = +0.42（t=0.89，不显著）**。结合
`R0`（raw external）在多数类上**低于** baseline（cable 82.4 vs 91.5、
transistor 81.8 vs 92.4、toothbrush 95.4 vs 99.3），
这与 Gate 9B 的模式完全一致：

> **学到的是「把 raw external 的伤害修掉」，不是「注入了可迁移的缺陷知识」。**

## 5. 另一个负面发现：投影摧毁了 target-defect oracle

oracle 在**各自 representation 下**计算（按预注册要求）：

| config | oracle headroom 中位数 | >2 的 fold 数 |
|---|---|---|
| R0 raw | **+0.78** | 5/15 |
| R1 | **+0.01** | 3/15 |
| R2 | **+0.00** | 0/15 |

**投影之后，连「用真实目标缺陷」都不再带来任何提升。**
合理解释：二分类目标只需要「是不是缺陷」，Pθ 会把所有缺陷 patch 压到同一区域，
丢失了 1-NN 所需的类内结构。

所以投影不是「免费」地改善外部迁移——它同时破坏了缺陷分支本身。

## 6. 判决

```
R2 > R1          ❌ −0.43 (t=−0.89)
monitor          ❌ identity↓ 时 anomaly↓，15/15 fold；λ 扫描无可用点
oracle 退化      ❌ +0.78 → +0.00
```

**Gate 10A 失败。按预注册规则不跑 AD2。**

### 关于 headroom 的说明

v1 下游这次**仍然是 range 不足**的（R0 oracle headroom 中位数 +0.78）。
但**本次判决不依赖它**：决定性证据是 monitor 与 λ 扫描，
两者都是 representation 内部的可解码性度量，与下游 headroom 无关。
所以不需要像 Gate 9B 那样偏离预注册去重跑 AD2。

### 限制

- 二分类目标本身可能就是问题：它只需要「是不是缺陷」，
  而我们最终需要的是「是**哪种**、在什么坐标下的缺陷」。更强的 objective
  （带 margin 的度量学习、多正例对比）未被排除。
- probe 是**线性**探针；若 Pθ 把 identity 变成非线性可解码，线性探针会低估其残留。
- 训练存在不稳定（部分 fold 的 obj loss 冲到 91），可能放大了 R2 的劣势。
  但 λ=0.05 的崩塌发生在训练稳定区，不是这个原因。

## 7. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `gate10a_projection.py`（v1/ad2 两模式）、`gate10a_lambda.py` |
| 结果 | `metrics/gate10a_v1.csv`、`gate10a_v1_probes.csv`、`gate10a_lambda.csv` |

## 8. 本次踩的坑

1. **`rng.choice(n, size=k, False)`** —— 关键字参数后再跟位置参数，语法错误。
2. **object head 只喂了 defect patch**，normal patch 没有 object label，
   导致 batch 512 vs target 256 的错位。object 监督必须覆盖全部 patch。
3. **probe 在 `P(Z)` 上做了 400 次 backward**，第二次就报
   "backward through the graph a second time"。probe 必须在 `.detach()` 上训练。
4. **`pd.DataFrame(list_of_dfs)`** 不能构造宽表；要 `pd.concat`。
5. **merge 时把 `baseline` 重复并入**导致 `MergeError`；只让第一个 config 携带它。

---

## 继承到下一阶段的强制方法纪律

1. **representation 类实验必须同时报告「该 representation 下的 oracle headroom」。**
   本次正是它暴露了「投影摧毁 target-defect 分支」这一独立问题——
   只看下游 AUROC 会完全看不到。
2. **对抗式去除某属性时，必须同时监控该属性与被保留属性。**
   只看 identity 下降会把「表征崩塌」误读成「解耦成功」。
3. **超参数扫描用于区分「方法失败」与「调参失败」。**
   本次 λ 扫描把「R2 没调好」升级为「该机制不可用」。
4. **任何「比 raw 好」的结论都要再对一次 baseline。**
   R1 比 raw 好 +3.19，但对 baseline 只有 +0.42——
   前者是「修复伤害」，后者才是「带来信息」。
