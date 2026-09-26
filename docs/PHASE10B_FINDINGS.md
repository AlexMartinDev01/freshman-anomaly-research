# Gate 10B-O 结果总账：Structured Defect Oracle Diagnostic

> 冻结于 tag `gate10b_structured_pass`。
> **判决：PASS。这是本项目第一条正面结果。**
> 关键结论：**DINO 里有缺陷类型结构，是 binary readout 把它毁掉的。**

---

## 最终结论（严谨表述）

> **raw DINO 已经携带相当强的 defect-type 结构（cross-image 同类型 Recall@1
> chance-adjusted 67.3%）。**
> **binary「是不是缺陷」目标会把表征压塌（effective rank 1.7），并摧毁这份结构
> （chance-adjusted 降到 52.5%）——这就是 Gate 10A 中 oracle headroom
> +0.78 → +0.01 的机制。**
> **换成 structured defect-type metric 后，retrieval 升到 83.9%，且
> target-defect oracle headroom 恢复到 raw 之上（mean +3.08 vs raw +1.10）。**

---

## 1. 设置

| config | feature | objective |
|---|---|---|
| B0 | raw DINO | 原始 1-NN |
| B1 | 384→256→128 | binary anomaly（Gate 10A 的目标） |
| B2 | 384→256→128 | structured triplet on **defect type** |
| B3 | 384+XY→256→128 | structured triplet + 位置编码 |

triplet 的正例 = **同 object、同 defect type、不同 image**；
负例 = ①同 object 不同 type ②同 object normal ③不同 object 的 defect。

**retrieval 的 gallery 排除 query 自己所在的 image**，否则学到的只是「图像实例」。

## 2. 主结果

cross-image 同类型 Recall@1（15 objects）：

| config | Recall@1 | Recall@5 | Recall@10 | chance-adjusted |
|---|---|---|---|---|
| B0 raw | 84.6 | 83.6 | 82.5 | 67.3 |
| B1 binary | 78.2 | 76.2 | 74.9 | 52.5 |
| **B2 structured** | **92.6** | **92.8** | **92.7** | **83.9** |
| **B3 structured+XY** | **93.0** | 92.4 | 92.4 | **85.3** |

**chance-adjusted** 指 `(recall − majority)/(100 − majority)`；
majority = 该 object 占比最大的 defect type 的比例（均值 51.0%），
即「完全塌缩的表征」会拿到的分数。

逐 object（排除只有单一 defect type、因而 recall 恒为 100% 的 toothbrush）：

| | B2 − B0 | B3 − B0 |
|---|---|---|
| 平均 | **+8.5** | +8.9 |
| 更好的 object 数 | **14/14** | 13/14 |

## 3. 预注册判据

| 判据 | 结果 |
|---|---|
| B2 > B0 on Recall@1，≥12/15 objects | ✅ **PASS**（14/15） |
| oracle headroom(B2) 明显为正且高于 raw | ✅ **PASS**（见下） |

target-defect oracle headroom（在各自 representation 下）：

| config | median | mean | n_positive |
|---|---|---|---|
| B0 raw | +0.56 | +1.10 | 9/15 |
| B1 binary | +0.00 | +0.78 | 6/15 |
| **B2** | **+0.57** | **+3.08** | 10/15 |
| B3 | +0.57 | +3.01 | 10/15 |

**B2 在每一个 object 上都 ≥ raw，其中 8 个严格更好**
（screw +14.6 vs +2.3、transistor +9.5 vs +5.0、
cable +7.4 vs +4.6、capsule +6.6 vs −1.2）。
中位数打平（+0.57 vs +0.56），mean 约为 raw 的 2.8 倍。

## 4. 机制：binary objective 把表征压塌了

采样 8000 patch，测 effective rank 与平均 pairwise cosine distance：

| config | effective rank | mean pairwise cos-dist |
|---|---|---|
| B0 raw | 88.8 | 0.837 |
| **B1 binary** | **1.7** | 0.972 |
| B2 structured | 24.8 | 0.944 |
| B3 | 23.6 | 0.958 |

**B1 的 effective rank 只有 1.7——binary 目标把 384 维压成约 2 个方向。**
这直接解释了 Gate 10A：

> 二分类只需要「是不是缺陷」，于是所有 defect patch 被压到同一条轴上；
> 1-NN 赖以工作的类内结构随之消失，oracle headroom 从 +0.78 掉到 +0.01。

B2/B3 的 rank 比 raw 低（24.8 vs 88.8）但平均距离**更大**（0.944 vs 0.837）
——这正是 metric learning 应有的形态：丢掉 nuisance 维度，把有意义的区分拉开。

## 5. 位置信息：B3 ≈ B2

| | B2 | B3 |
|---|---|---|
| Recall@1 | 92.6 | 93.0 |
| oracle headroom mean | +3.08 | +3.01 |

**加不加 (x, y) 几乎没有区别。**
按预注册规则：`B2 PASS 且 B3 不显著 > B2`
→ **defect morphology 信息存在，coordinate 不是主要问题**，
下一阶段不需要做成 structure-aware 的相似度。

## 6. 判据回答

用户的问题：

> raw DINO 中有没有 defect-level structured information，
> 只是 binary anomaly objective 把它破坏了？

**答案：有，而且是 binary objective 破坏的。**

| 假设 | 判定 |
|---|---|
| DINO 缺少 defect semantics | ❌ **证伪**（raw 已有 67.3% chance-adjusted） |
| binary objective 摧毁了它 | ✅ **证实**（52.5%，rank 1.7） |
| structured objective 能恢复 | ✅ **证实**（83.9%，oracle headroom 反超 raw） |
| 还需要 coordinate | ❌ 不需要（B3 ≈ B2） |

## 7. 限制（必须一起声明）

- **这是 oracle 诊断，不是可用方法。** B2 用**目标类的 defect-type 标签**训练，
  且训练所用的相似性定义与评测指标是同一个。
  所以 92.6 是「若已知正确的相似性，能做到多好」的**上界**，
  **不是泛化结果**。
- **v1 的 oracle headroom 本身很小**（median +0.57），
  所以判据后半部分用的是弱仪器；mean +3.08 主要由少数 object 拉动
  （screw、transistor、cable、capsule）。
- **必须记录一个实现 bug**：第一版 triplet loss 把负例项写成
  `relu(margin + d_an)`，即**把负例也往一起拉**，
  结果所有 patch 塌到一点（平均 pairwise cos-dist = 0.0000），
  B2 在 14/14 个 object 上**劣于** raw —— 一个完全错误的负结果。
  正确形式是 `relu(d_ap − d_an + margin)`。
  **是「测 effective rank / 平均距离」这一步把 bug 抓出来的**：
  只看 AUROC 或 recall 会直接把它当成「structured metric 没用」。

## 8. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `gate10b_structured.py` |
| 结果 | `metrics/gate10b_retrieval.csv`、`gate10b_oracle.csv`、`gate10b_per_object.csv` |
| 权重 | `results/model_v0/checkpoints_g10b/{B1,B2,B3}.pt` |

## 9. 本次踩的坑

1. **triplet 负例项符号写反**（见 §7），产生了一个方向完全相反的结论。
   **教训：metric learning 必须检查 embedding 的退化指标**（effective rank、
   平均 pairwise 距离），不能只看下游指标。
2. **`sample_batch` 每一步重建 O(N) 索引**，单 config 30 分钟跑不完；
   改为在 `Index.__init__` 里预计算后降到 19 秒。
3. `np.array([(obj,type),...])` 会生成 2-D array，取出的 key 是 ndarray，
   **不可哈希**，不能索引 dict；要存成 list of tuple。
4. `pd.DataFrame(dict_of_dict)` 的嵌套方向与预期相反，要转 long-form。

---

## 继承到 Gate 11 的强制方法纪律

1. **metric learning 必须报告 embedding 退化指标**（effective rank、平均 pairwise
   距离）。本次靠它抓出一个会颠倒结论的 bug。
2. **oracle 诊断的结论要写明「这是上界，不是泛化」。**
   训练相似性 == 评测相似性时尤其如此。
3. **判据两半都要如实执行**：retrieval 过了、oracle headroom 只在 mean 上过、
   median 打平——必须照实写，不能只说「PASS」。
4. **消融要能回答「还需不需要下一个复杂度」**：B3 ≈ B2 直接砍掉了
   structure-aware similarity 这条支线。
