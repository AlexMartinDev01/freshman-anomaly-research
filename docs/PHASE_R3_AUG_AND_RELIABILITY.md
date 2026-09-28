# Phase R 步骤 6：augmentation 与 normal-only reliability（两条自动修正路线）

> 冻结于 tag `phase_r3_aug_reliability`。
> **两条路线都得到否定结论，但性质不同：**
> - **augmentation**：机制成立（screw 大幅恢复），但**该判据无法无标签决定是否该用**；
> - **layer reliability**：**当前这个统计量不能用于 layer gating**（P2 12%，0/15 对象）。

---

## 0. 出发点

`PHASE_R3_LAYER_CONFIRM.md` 确认 `agg_all3` 的定位增益真实（AUPRO +3.40，135/180），
但 screw 一个对象把 image-level 守卫拉穿。
`PHASE_R3_SUBSPACEAD.md` 给出关键自然实验：同 backbone、同 k 张图，
只把每张图旋转 30 次，SubspaceAD 在 screw 上 AUPRO **40.9 → 65.7**。

所以先验假设是：

> **screw 的病不是 scoring family，而是 reference normal diversity 不足。**

按"简单解释先验证"的顺序，先做 augmentation，再做 reliability gating。

---

## 1. augmentation 加进 kNN bank（4-object smoke）

只改 reference bank：`k 张图 → 各 30 次随机旋转 → patches → kNN bank`。
层、scoring、evaluator、shots 全不变。旋转与 SubspaceAD 完全一致
（`RandomRotation` on [0,345)，fill 0，原图保留）。
角度来自每 `(object, image, aug)` 的独立 SeedSequence，
用 `functional.rotate` 施加，不走 torchvision 全局 RNG。

### 预注册判据（运行前冻结）

```
S1  screw 恢复：agg_all3+aug vs agg_all3，img >= +5 且 AUPRO >= +10
S2  无附带损伤：非 screw 对象 AUPRO 下降不超过 5
S3  守卫成立：agg_all3+aug 的 img 均值不比 final_raw 差 0.3 以上
```

### 结果

| object | `agg+aug − agg` AUPRO | `agg+aug − agg` img |
|---|---|---|
| screw | **+23.11** | +19.64 |
| hazelnut | **+6.11** | +1.07 |
| tile | +0.43 | +0.33 |
| transistor | **−5.16** | −0.37 |

```
S1 screw recovers  img +19.64 (need >=+5)   AUPRO +23.11 (need >=+10)   OK
S2 no collateral   worst non-screw AUPRO -5.16 (transistor) (need >=-5) FAIL
S3 guard holds     mean(agg_all3_aug - final_raw) +3.14  (need >=-0.3)  OK
-> NO-GO: do not scale up
```

screw 逐 shot 的 AUPRO（kNN 固定不动）：

| shot | kNN | SubspaceAD aug0 | SubspaceAD aug30 |
|---|---|---|---|
| 1 | 42.8 | 40.9 | 65.7 |
| 2 | 43.6 | 44.8 | 71.6 |
| 4 | 47.1 | 48.1 | 75.0 |
| 8 | 62.5 | 65.3 | 75.0 |

**augmentation 不只是"恢复"，是反超**：`agg_all3+aug` 在 screw 上
（img 85.5 / px 97.9 / AUPRO 75.6）全面超过 `final_raw`（76.6 / 92.6 / 58.3）。
即"中层拖累"这个问题消失了，而不是被掩盖。

### transistor 的独立旁证

**SubspaceAD 官方默认 `--no_aug_categories ["transistor"]`** ——
即使 `aug_count=30`，transistor 也**不带增强**跑。
两个方法、两套代码、同一个对象撞到同一堵墙。

### 判据失败的方式，必须如实说

S2 只超了 0.16。它被设计来抓的失败模式（"单个巨大提升掩盖大面积下滑"）
**并没有发生**——4 个对象里 3 个上升。**判据的字面被违反，意图没有被违反。**

但按纪律**不改门槛**。结论是：**固定 augmentation 配方形状不对**，
缺的不是"更强的增强"，而是**决定"这个对象该不该增强"的依据**。

---

## 2. 那个依据存在吗？——normal-tail 判据 H1

用**留出的训练正常图**（不在 bank 内，不用任何 label、不碰测试集）在 bank 上打分：

```
T_base  = Q0.99(该图 patch 到「无增强 bank」的距离)
T_aug   = Q0.99(该图 patch 到「有增强 bank」的距离)
dT      = T_aug - T_base
```

### 冻结假设

> **H1**：`dT < 0`（增强降低了正常尾部）⟺ 增强减少正常误报 ⟺ 该对象受益。
> 因此 `dT` 应把 4 个对象排成与实测一致：`screw/hazelnut < tile < transistor`。

### 结果

| object | dT_mean | 实测 AUPRO 增益 |
|---|---|---|
| screw | −0.203 | +23.11 |
| hazelnut | −0.093 | +6.11 |
| transistor | −0.052 | **−5.16** |
| tile | −0.036 | +0.43 |

```
predicted order: screw, hazelnut, transistor, tile
measured  order: transistor, tile, hazelnut, screw
-> DISAGREES        pearson r = -0.963
```

**H1 被证伪。** 相关性 −0.963 看着很好，但预注册的是**排序检验**，
而换位的正是决定符号的位置：transistor 的 dT 说"增强对正常尾部无害"，
实测却是唯一被伤害的对象。

### 机理解释（可诊断，非玄学）

transistor 的缺陷类型是 `bent_lead / cut_lead / misplaced` —— **方向性结构异常**。
旋转增强把正常引脚的旋转版本灌进 bank，于是**被旋转过的缺陷看起来就像正常**。

> **dT 只测量增强对"正常图"的影响，它对"增强 bank 吸收旋转缺陷"这个代价
> 是结构性失明的。任何只用正常数据的判据都会有同样的盲区。**

### 严谨表述（重要）

本结果证明的是：**这个 dT 判据无法无标签决定 augmentation 是否有益。**
**不能**进一步推出"任何可能的 normal-only 判据都绝对不可能"——
后者是一个关于所有可能统计量的全称命题，本实验没有、也不可能证明它。
从**项目管理**角度，结论是：**不要再换统计量重试**（那是 covariance 分支的死循环）。

---

## 3. normal-only layer reliability（P1/P2）

动机：screw 的 `mid` AUPRO 只有 18.0、`midlate` 25.5，而 `final` 52.7。
等权平均把两个坏层硬塞进去。合理的问题不是"什么权重最好"，
而是"**能否只用目标正常数据判断某层在这个对象上是否可靠**"。

统计量：

```
T_l = Q0.99(留出正常图 patch 到 bank 在层 l 的距离)
w_l ∝ 1/(T_l + ε)     或   softmax(-β T_l)
```

不训练，不用异常标签，不用测试集。

### 冻结判据

```
P1  Spearman rho(T_l, AUPRO_l) <= -0.5 且 p < 0.01   （45+ 点：object × layer）
P2  每个 (object, shot, split) 内，T_l 最大的层也是 AUPRO 最低的层，比例 >= 60%
```

### 结果（15 objects × 6 cells × 3 layers = 270 点）

```
P1  rho = -0.526   p = 1.35e-20   -> OK
P2  11/90 cells = 12%  (need >= 60%)  -> FAIL
    per-object agreement: 0/15
-> NO-GO: close this line
```

### 为什么 P1 通过而 P2 惨败——这正是 P2 存在的意义

| | T 最大的层 | AUPRO 最低的层 |
|---|---|---|
| 分布 | **midlate 14/15** | mid 9/15、final 6/15，**从不是 midlate** |

**两个集合几乎不相交（重合 11/90）。** `T` 几乎在每个对象上都是 midlate 最高，
而真正最差的层是 mid 或 final。screw 上 `T_mid=0.43`、`T_midlate=0.44` ——
几乎打平，统计量恰好挑了错的那个。

> **P1 的相关性来自跨对象混杂**：难对象（screw / hazelnut / metal_nut）
> 在**所有层**上 T 都高、AUPRO 都低。统计量实际在说"这个对象难"，
> 而不是"该信哪一层"。P2 把这两种解释分开，成功地挡下了一个假信号。

### 严谨表述（重要）

本结果证明的是：**当前这个 T_l 不能用于 layer gating。**
**不能**推出"所有 normal-only reliability signal 都不存在"。

---

## 4. 两条路线关闭后的项目状态

| 方向 | 结论 | 性质 |
|---|---|---|
| normal augmentation | screw −5.85 → 全面反超；transistor −5.16 | **机制成立，但无法无标签自动化** |
| normal-only layer reliability | P2 12%（需 60%），0/15 对象一致 | **当前统计量不成立，关闭** |

**明确停止的清单**（不再重试）：
augmentation predictor、normal-only layer gating、covariance、layer 权重调参。

第二条是干脆的否定。第一条要说得准确：

> **augmentation 不是"方法"，它是一个有效的个体化修正。**
> 病因（reference diversity 不足 + 中层对正常图高分）已被两个独立方法证实；
> 但"该不该对某个对象用"无法用纯正常数据判断，
> 除非接受一张人工例外表（如 SubspaceAD 的 `no_aug_categories`）。

不得粉饰为"方法有效"。

---

## 5. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `gate_r2_aug_bank.py`、`gate_r2_normal_tail_predictor.py`、`gate_r2_layer_reliability.py` |
| 结果 | `metrics/gate_r2_aug_bank_smoke.csv`、`metrics/gate_r2_normal_tail_predictor.csv`、`metrics/gate_r2_layer_reliability.csv`（270 行） |
| maps | `results/model_v0/maps_layer_aug/`、`maps_layer_reliability/` |
| 日志 | `aug_smoke.log`、`normal_tail.log`、`shards/{lr,f1}_*.log` |

## 6. 方法纪律（本轮新增）

1. **判据的符号必须与它自己要检验的假设逐字核对。**
   本轮 F1 就出现了一处符号写反（见 `PHASE_F1_FINDINGS.md` §4）。
2. **"单个对象巨大提升 + 其余轻微下降"必须被判据显式挡住**（S2 的设计），
   即使实际的失败方式与设计意图不同，也不得事后放宽。
3. **相关性通过、排序检验失败 = 统计量没有区分力**，不是"稍微弱一点"。
   P2 是本节最有价值的一条设计。
