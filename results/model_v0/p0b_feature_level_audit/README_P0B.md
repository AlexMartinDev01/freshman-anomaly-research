# P0-B —— Feature-Level Parameter-Free Global→Local Retrieval Audit

```
FINAL DECISION: NO-GO   (all three scales)
Transformer branch CLOSED
```

| n | ΔH-B | ΔH-Q | MVTec Δ | VisA Δ | R_feat | H-shuf | Decision |
|---:|---:|---:|---:|---:|---:|---:|---|
| 9 | **−0.7023** | +0.0061 | −0.6632 | −0.7168 | **−12.4534** | +0.0066 | NO-GO |
| 13 | **−0.7300** | +0.0045 | −0.6858 | −0.7670 | **−10.9264** | +0.0197 | NO-GO |
| 15 | **−0.7331** | +0.0125 | −0.7147 | −0.7392 | **−10.9013** | +0.0187 | NO-GO |

按协议 §16，这是第三种解释：**G1 / G2 / G4 失败 → P0-B NO-GO → Transformer branch 正式关闭**。
不扩展 n，不加层、heads、MLP、temperature、position bias 去救，**不查看 defect performance**。

---

## 1. 判决不是勉强不过，而是方向性错误

绝对量（27 类中位）：

| arm | agg3 ρ (n=9 / 13 / 15) | cos(·, T) |
|---|---|---|
| **B** Global448 | **0.8153 / 0.8153 / 0.8153** | **0.9578** |
| Q Local672 | 0.1031 / 0.0508 / 0.0718 | 0.403 / 0.449 / 0.460 |
| C retrieval context | 0.0737 / 0.0879 / 0.1044 | 0.392 / 0.455 / 0.457 |
| H = normalize(Q+C) | 0.0988 / 0.0865 / 0.0779 | 0.424 / 0.476 / 0.483 |
| H_shuf | 0.0862 / 0.0545 / 0.0438 | 0.410 / 0.465 / 0.471 |

**`rho_B` 在三个尺度上完全相同（0.8153）** —— B 只用 Full448 与 Full672，二者都不依赖 n。
**`delta_H_B > 0` 的类别数：0 / 27，三个尺度全部为零。** 检索在任何一类上都没有带来增益。

**Global448 本身就是最好的**：特征上 cos 0.958 直接匹配 pooled Full672，几何上 ρ 0.815。
其余所有 arm 都在 0.05–0.10 一线。

## 2. 机制：softmax 平均产生的是"通用全局均值"，不是位置特异的上下文

`cos(C, T) ≈ 0.39–0.46` —— 对整张 448 token grid 做 softmax 加权平均得到的 C，
在特征空间里离 T 很远，与 Q 自身（0.40–0.46）几乎同一水平。原因不是加权方式选错：
**对上千个 token 求加权平均，本身就抹掉了位置特异的内容**，无论权重取什么。

于是 `H = normalize(Q + C)` 基本由 Q 主导：`cos(H,T)` 0.42–0.48，只比 Q 高一点点，
远低于 B 的 0.958。`R_feat = (c_H − max(c_B, c_Q)) / (1 − max(c_B,c_Q))` 因此是
**−11 到 −12.5**：不是"恢复了 10% 的剩余 headroom"，而是**相对于 B 损失了约 11 倍的 headroom**。

## 3. G3 与 G5 的"通过"必须标注为无实质意义

```
G3 (median ΔH-Q > 0)     通过：+0.005 ~ +0.013，但 bootstrap 95% CI 全部跨 0
G5 (median ΔH-shuf > 0)  通过：+0.007 ~ +0.020，CI 在 n=13,15 勉强不跨 0
```

两项的量级都接近 0。G5 的含义仅仅是"H 比打乱后的 global memory 略好"——
打乱版本更接近通用均值，所以更差是必然的，这**不构成 image-specific context 有效的证据**。
协议里 G1/G2/G4 才是实质门槛，它们失败得很彻底。

## 4. 对假设 H 的回答

> H：Local672 的高维表示可能含有被 P0-A 标量 1NN distance 压缩掉的互补信息。

**在 parameter-free retrieval 下，没有得到支持。** 需要注意措辞边界：

- 被否证的是**这一具体构造**：`sqrt(D)·cos` 打分 + softmax 加权平均 + `α=1` 固定残差。
- 高维 Q **本身**与 T 的 cos 只有 0.40–0.46 —— 也就是说 Local672 crop 特征
  **在特征空间里离 Full672 就很远**，这不像是"被标量距离压掉了信息"，
  更像是**crop 表征本身就不携带 full672 的位置特异信息**。
- 因此协议 §21 的另一种解释（"cosine 提高但 ρ 没提高"）在这里**不适用**：
  cosine 根本没有提高，是下降的。

## 5. 实现修复记录（透明性要求，协议 §18 addendum 规则）

三次崩溃与一处内存缺陷都在 **smoke 阶段、任何有效结果产生之前**发现并修复，
因此**没有结果需要标记 INVALIDATED**。协议本身一字未改。

| # | 问题 | 性质 |
|---|---|---|
| 1 | `SMOKE_OBJECTS` 常量在重写时丢失 | 崩溃 |
| 2 | `o6[j]` 把 `{layer: offsets}` 当图像索引 | 崩溃 |
| 3 | `bank448` 外提改了一半（`bank448.T` 指向 dict） | 崩溃 |
| 4 | **门禁段为全部 27 类加载 cache → 约 51 GB，必然 OOM** | **致命，且 smoke 不可见** |

第 4 项特别值得记录：两个对象的 smoke 跑得完全正常（2 × 1.9 GB），
而这个缺陷只会在 27 类跑到一半时炸掉。

修复后 smoke 重跑，`probes.csv` 与修复前**逐字节相同**（SHA256 一致），
证明第 3 项的外提是数值中性的。

**同时被抓住的一次假阳性**：第一次等价性检查显示 "IDENTICAL"，
但那是因为 `/usr/bin/time` 不存在、命令根本没执行（`PY_EXIT=127`），
被比较的是尚未重写的旧文件。**检查通过 ≠ 被检查的事情发生了。**

## 6. 正确性门禁（全部通过）

```
PC1  positive control            12/12 PASS (cos>0.999999, maxdiff<=1e-5)
PC2  pooling weights             == cal.pool_weights, 40 随机 patch，index 与 weight 精确一致
PC3  bank exclusion              1000 次随机检查，held-out 图在自己 bank 中出现 0 行（按位比较）
PC4  attention 权重和            max |sum a - 1| = 8.88e-16
PC5  全部 finite                 通过
PC6  normalize 后范数            Q 1.000000  H 1.000000  T 1.000000
PC7  raw rows                    157464  == 期望
PC8  每 (category,n,layer)        648      == 期望（243 格全部）
```

PC3 是这次最容易泄漏的地方，按协议用**逐位**比对而非索引记账：
把 held-out 图的所有 token 行与 bank 的所有行做 `intersect1d`，结果为空。

## 7. 边界（不得扩大）

- 本包**不含任何 defect 性能指标**：无 test anomaly、无 GT、无 AUPRO/AUROC/A5。
  判决完全建立在 normal support 上。
- 被关闭的是 **Transformer branch**，依据是预注册的 G1–G5。
- `rho_B = 0.815` 是"Global448 与 Full672 两个表示的近邻几何一致度"，
  **不是性能指标**，也不说明 448 分辨率"够用"。

## 8. 文件

```
P0B_PROTOCOL.md              冻结协议副本（commit 5f86b1d）
P0B_FREEZE.txt               冻结记录
p0b_manifest.json            运行元数据与判决
p0b_positive_control.csv     PC1，12 行
p0b_selftest.csv             PC1–PC8 记录
p0b_cache_audit.csv          support identity 按真实 image name 比对
p0b_probe_features.csv       157464 行：B/Q/C/H/Hshuf/T 的特征余弦与 1NN 距离
p0b_per_layer.csv            729 行：逐层 Spearman（原始尺度）
p0b_agg3_by_category.csv     243 行 = 27 × 3 尺度
p0b_summary.csv              §0 的表
p0b_bootstrap.csv            10000 次 dataset-stratified category bootstrap
p0b_decision.csv             冻结 gate 的三行判定
feature_cache_manifest.csv   243 个缓存文件的 sha256
run.log                      两个分片的完整日志
```

## 9. 复现

```bash
P0B_SMOKE=1 python experiments/p0b_feature_level_audit/p0b_feature_audit.py   # 门禁 + 2 对象
python experiments/p0b_feature_level_audit/p0b_feature_audit.py               # 27 类，可 resume
P0B_AGG_ONLY=1 python experiments/p0b_feature_level_audit/p0b_feature_audit.py
```

27 类实测约 30 分钟（两个分片并行）；聚合阶段秒级。
