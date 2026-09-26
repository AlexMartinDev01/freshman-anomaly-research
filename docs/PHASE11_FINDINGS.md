# Gate 11 结果总账（冻结）：Leave-One-Object-Out Structured Metric Transfer

> 冻结于 tag `gate11_transfer_closed`。
> **判决：FAIL。且失败形态正是预注册里写好的那一支。**
> **defect structure 存在（C3 >> C0），但它是 object-conditional 的，
> 不能用统一 metric 跨 object 迁移（C2 < C0）。**

---

## 最终结论（严谨表述）

> **在目标类 defect label 不可得、只用其余 14 个 object 的 defect 监督训练时，
> 学到的 metric 在 held-out object 上**劣于**什么都不做（raw DINO）：
> C2a −5.1（1/15）、C2b −6.0（3/15）。**
> **换用跨 object 共享的粗糙 morphology 词表（7 类）也救不回来：
> 同 morphology 的 recall 同样低于 raw（C2a −4.3, 2/15；C2b −5.0, 2/15）。**
> **而目标类监督的 C3 远高于 raw（+7.8 平均，15/15）。**
> **→ defect geometry 是 object-conditional 的。**

---

## 1. 设置

| config | 训练监督 | 备注 |
|---|---|---|
| C0 | 无 | raw DINO，floor |
| C1 | source binary anomaly | 负对照 |
| C2a | source defect **type** | 数据集原始类别名 |
| C2b | source **morphology**（7 组） | 跨 object 共享词表 |
| C3 | **target** defect type | Gate 10B 的 B2，oracle 上界 |

LOCO：15 个 object 轮流作 held-out target，投影只在其余 14 个上训练。
retrieval 的 gallery 排除 query 自己的 image。

**morphology 词表**（7 组）：
`crack_like`（crack/broken/cut/scratch…）、`hole_missing`、`contamination`、
`deformation`（bent/flip/squeeze…）、`surface_texture`（rough/thread/print…）、
`color_change`、`other`。

## 2. 主结果：Recall@1（cross-image，同 defect type）

| target | C0 raw | C1 | C2a | C2b | **C3 oracle** |
|---|---|---|---|---|---|
| bottle | 86.0 | 78.7 | 78.0 | 72.7 | **93.3** |
| cable | 70.0 | 64.0 | 62.7 | 54.7 | **82.7** |
| capsule | 86.0 | 80.0 | 76.7 | 84.0 | **92.0** |
| carpet | 82.7 | 79.3 | 80.7 | 80.7 | **95.3** |
| grid | 82.7 | 74.7 | 75.3 | 69.3 | **98.0** |
| hazelnut | 82.7 | 80.7 | 84.0 | 84.0 | **96.7** |
| leather | 96.0 | 84.0 | 85.3 | 84.0 | **97.3** |
| metal_nut | 95.3 | 92.7 | 91.3 | 93.3 | **96.0** |
| pill | 78.7 | 74.7 | 76.7 | 79.3 | **85.3** |
| screw | 84.7 | 82.7 | 84.0 | 75.3 | **87.3** |
| tile | 98.0 | 92.7 | 96.0 | 93.3 | **100.0** |
| toothbrush | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| transistor | 91.3 | 86.7 | 85.3 | 85.3 | **96.7** |
| wood | 75.3 | 62.7 | 70.0 | 78.0 | **95.3** |
| zipper | 61.3 | 52.7 | 48.7 | 46.0 | **72.0** |
| **mean** | **84.7** | **79.1** | **79.6** | **78.7** | **92.5** |

## 3. 预注册判据

| 判据 | 结果 |
|---|---|
| **C2a > C0，≥12/15** | ❌ **FAIL**（1/15，mean −5.1） |
| oracle recovery ≥30% 多数 | ❌ **FAIL**（0/14 达到，mean −156%） |

Oracle Recovery Ratio `(C − C0)/(C3 − C0)`：

| config | mean gain | better | recovery |
|---|---|---|---|
| C1 | −5.6 | 0/15 | −162% |
| C2a | −5.1 | 1/15 | −156% |
| C2b | −6.0 | 3/15 | −175% |

## 4. 关键补充：换 label 粒度也救不回来

如果「数据集缺陷类别名不可跨 object 对齐」是原因，
那么用**跨 object 共享的 morphology 词表**应该有救。实测：

| 监督粒度 | C2a − C0 | C2b − C0 |
|---|---|---|
| 同 fine defect type | −5.1（1/15） | −6.0（3/15） |
| 同 **morphology** | −4.3（2/15） | −5.0（2/15） |

**两个粒度都低于 raw。**

而且注意：**raw DINO 的同 morphology recall 本来就已经很高**
（bottle 96.7 vs fine 86.0；transistor 95.3 vs 91.3），
说明 morphology 这一层信息 raw 本来就抓到了，不需要学。

**所以失败原因不是「标签语义对不齐」，是 defect geometry 本身 object-conditional。**

## 5. 不是表征塌缩 —— 这一点很重要

| config | effective rank 范围 | d_ap < d_an |
|---|---|---|
| C0 raw | 27.6 – 50.7 | — |
| C1 source binary | **4.0 – 15.7** | 75/75 ✅ |
| C2a source type | 11.6 – 19.2 | 75/75 ✅ |
| C2b source morph | 11.6 – 19.2 | 75/75 ✅ |
| C3 target type | 4.6 – 20.0 | ✅ |

**health check 75/75 全部通过**（无一行 d_ap ≥ d_an），
rank 也和 C3 同量级。**源训练的 metric 确实学到了一个健康的、有判别力的几何 ——
只是那个几何对应的是 source object 的缺陷结构，不是 target 的。**

这排除了「又是 triplet bug」「表征塌缩」这类解释：
**失败是真实的泛化失败。**

## 6. 判决与下一分支

预注册里写好的分岔：

> 如果 C2 ≈ C0 甚至低于 raw，但 C3 >> C0
> → defect structure 存在，但主要是 object-conditional，不能直接跨 object 用统一 metric 迁移
> → 转向 **Conditional metric**

**观测到的正是这一支**（C2 低于 raw，C3 高 +7.8）。

因此下一步不该再做 universal defect metric，而应做：

> **object-conditioned metric**：同一个 defect morphology 在不同 object context 下需要不同的度量。

形式上是 `d(z_i, z_j | object context)`，或由一个 target-side 信号
`M(x_object)` 生成 Mahalanobis metric / low-rank adapter。

**注意这与之前被否决的 conditional generator 不同**：
不是重新生成 defect feature，而是**只调整度量本身**。

## 7. 限制

- morphology 分组是我人工定义的 7 类。分组方式可能仍有改进空间，
  但「fine 和 coarse 两个粒度都失败」这一事实说明粒度不是主因。
- C3 用 target label 训练，与 Gate 10B 的 B2 是同一个 checkpoint，
  因此它是**上界而非方法**。
- 8 个 object 的 defect type 名在不同 object 间可能语义重叠有限，
  但我已用 morphology 词表显式处理了这一点。

## 8. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `gate11_transfer.py` |
| 结果 | `metrics/gate11_retrieval.csv`、`gate11_health.csv`、`gate11_recovery.csv`、`gate11_morph.csv` |
| 权重 | `results/model_v0/checkpoints_g11/{target}_{C1,C2a,C2b}.pt` |

## 9. 本次踩的坑

1. **单 object 内不存在「其他 object」这个负例池**，
   导致 `oth` 恒为空、health 全部返回空 dict。已改为
   「同 object、不同 defect type」——这也正是 retrieval 真正关心的对比。
2. `mean_dist` 最初在**去中心化之后**的特征上计算，得到恒等于 1.0 的伪值；
   去中心化只应用于求 effective rank。

---

## 继承到下一阶段的强制方法纪律

1. **健康检查要在正确的对照上定义。** 单 object 场景下「跨 object 负例」不存在，
   照搬会导致静默失败（返回空而不是报错）。
2. **负结果要主动寻找「换一个粒度是否仍然失败」的反驳。**
   本次主动做了 fine vs morphology 两档，把「词表没对齐」这个替代解释提前排除。
3. **区分「塌缩」与「学错了」。** effective rank 与 d_ap/d_an 同时健康，
   说明失败是泛化失败而非优化失败——这个区分决定了下一步是调模型还是换问题。
