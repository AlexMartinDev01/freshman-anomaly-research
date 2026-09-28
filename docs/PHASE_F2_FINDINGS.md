# Phase F2：在 VisA 上的独立确认（预注册判据全部通过）

> 冻结于 tag `phase_f2`。
> **判决：REPLICATED。** F1 在 MVTec 上发现的机制，在**从未接触过的 VisA** 上、
> 用**在看到任何 VisA 结果之前冻结的判据**复现。
> **这是本项目第一个真正意义上的 confirmatory result。**

---

## 1. 为什么 F2 是必要的

`PHASE_F1_FINDINGS.md` 的预注册不等式含有一处**符号错误**，且该错误是在看完
MVTec 结果之后才发现的。按 F1 文档的冻结表述：

> **preregistered inequality contained a sign error; therefore this dataset is
> treated as hypothesis-generating / supportive rather than an independent
> confirmatory validation.**

**同一批数据无法再构成独立确认。** F2 用 VisA 补上这一步。

## 2. 预注册（见 `PHASE_F2_PREREGISTRATION.md`）

先声明两件事，以证明"未见数据"是真的：

- 判据写入时，**VisA 的下载仍在进行中**；
- 当时**没有任何 VisA 特征、map 或指标被生成或查看**。

```
F2-1  两个 detector 的 within-category rho(C_feat, TPR) 均 > 0 且 >= +0.4
F2-2  各自至少 80% 的 category 方向为正
F2-3  控制 area/morphology 后 C_feat 系数仍显著为正，用 category-clustered SE
F2-4  M（连续 margin）上结论一致，rho >= +0.4
```

## 3. 数据与配置

| | |
|---|---|
| 数据集 | VisA（`1cls`），12 categories，**本项目此前从未使用** |
| 规模 | 1200 张缺陷图 × 6 cells = **7200 行**（shots {1,4} × 3 splits） |
| 两个 detector | kNN `agg_all3`（与全程一致）与 SubspaceAD（官方，ViT-S/14 @448，**aug=0**） |
| evaluator | 与 MVTec 完全相同的 `pixel_metrics_binned(pro_limit=0.05)` 路径 |

## 4. 结果

```
F2-1 rho >= +0.4 for both detectors   (kNN +0.499, SA +0.405)  -> OK
F2-2 >= 80% categories positive       (kNN 100%,  SA 100%)     -> OK
F2-3 C_feat > 0, p < 0.05, clustered SE                        -> OK
F2-4 M-based rho >= +0.4              (kNN +0.730, SA +0.666)  -> OK
-> REPLICATED
```

F2-3 明细（object FE + **category-clustered SE, G=12**）：

| 系数 | kNN | SubspaceAD |
|---|---|---|
| **C_feat** | β **+0.827**, t **+6.44**, p 4.8e-5, CI [+0.576, +1.079] | β **+0.643**, t **+4.83**, p 5.3e-4, CI [+0.382, +0.903] |
| log_area | −0.040 (t=−3.02) | −0.008 (t=−0.87, **不显著**) |
| largest_cc_frac | +0.125 (t=+2.13, p=0.057) | +0.150 (t=+2.74, p=0.019) |
| n_cc | −0.001 (t=−0.64) | +0.000 (t=+0.11) |
| elongation | ~0 (t=+1.31) | ~0 (t=+1.14) |

**两个 detector 在 12/12 个 category 上方向全部为正。**

### 4.1 估计量稳健性（独立复核）

F2-1 用「within-category z-score 后 Spearman」估计。用两种**独立实现**复核：

| 估计量 | kNN | SubspaceAD |
|---|---|---|
| 预注册（z-score 后 Spearman） | +0.499 | +0.405 |
| rank-pooled within-category（先秩变换再池化） | **+0.528** | **+0.436** |
| 逐 category Spearman 的均值 | **+0.532** | **+0.444** |
| 方向为正的 category 比例 | 12/12 | 12/12 |

**三种估计量全部超过 +0.4 门槛，且预注册所用的那个是最保守的。**
结论不依赖估计量的选择。

## 5. 与 MVTec 的一致性（跨数据集、跨图像几何）

| 统计量 | MVTec (F1) | VisA (F2) |
|---|---|---|
| within rho(C_feat, TPR), kNN | +0.526 | **+0.499** |
| within rho(C_feat, TPR), SA | +0.424 | **+0.405** |
| within rho(C_feat, M), kNN | +0.715 | **+0.730** |
| within rho(C_feat, M), SA | +0.705 | **+0.666** |
| 方向一致率 | 15/15、14/15 | **12/12、12/12** |

**在完全不同的数据集、不同的缺陷类型、不同的图像几何下，数值几乎重合。**
这不是"又找到一个相关量"，而是同一个机制的稳定复现。

### 而且是在不利条件下复现的

VisA 图像**全部非正方形**（宽高比 1.08–1.63，类内固定），而 SubspaceAD
把图 resize 成精确 448×448，**它看到的图是被拉伸的**；我们的 kNN 则保持宽高比
（网格如 (32,48)、(32,52)，而 SubspaceAD 恒为 (32,32)）。

> 这个差异是 SubspaceAD 自身在非正方形输入上的行为，它对它是**不利**的。
> **机制仍然复现**——说明结论不是靠某个 detector 的特殊优势撑起来的。

（几何上无对齐错误：`dists2map` 把各自的网格 resize 回原图 (W,H)，
这正是各自 map 在原坐标系下的正确解释。）

## 6. 必须诚实记录的差异

1. **`largest_cc_frac` 在 VisA 上不再完全为零**（t=+2.13 / +2.74，p=0.057/0.019），
   而 MVTec 上是 +1.01（不显著）。即**碎片化在 VisA 上有一个小而真实的正效应**。
   但量级远小于 C_feat（β 0.13–0.15 vs 0.64–0.83），
   **不改变"可分性是主因"的结论**，但它说明形态学不是严格为零。
2. **`log_area` 只在 kNN 上显著**（t=−3.02），SubspaceAD 上不显著（t=−0.87）。
   面积是次要且 detector 相关的因素。
3. VisA 只有 `good`/`bad` 二分类，无 defect subtype，因此**只能做 category 级**。
4. `gt_covers_p95` 中位数 0.002、最大 0.320；未做排除复核（F2 判据未要求），
   但 `M` 不受该污染影响且给出同向更强结论。

## 7. 这个结果**不**意味着什么

- **`C_feat` 使用 GT defect mask，是 oracle 诊断量，不能进入最终方法。**
  它证明的是"机制存在于哪里"，不是"如何无标签地检测它"。
- 相关性 ≠ 因果。但两个结构不同的 detector、两个数据集、
  控制 object 与面积后均成立，因果解释（局部可分性不足）现在是
  **最有依据的工作假设**，而非唯一被证明的机制。

## 8. 资产

| 类型 | 内容 |
|---|---|
| 预注册 | `docs/PHASE_F2_PREREGISTRATION.md`（冻结，未修改） |
| 脚本 | `cache_visa.py`、`phase_f2_visa.py`、`phase_f2_analysis.py` |
| 数据 | `data/VisA_20220922/`、`data/VisA_pytorch/1cls/` |
| 缓存 | `results/model_v0/cache_visa/`（2162 测试图，断言全过） |
| dumps | `results/subspacead_visa/dumps/`（6 × 2162） |
| 结果 | `metrics/phase_f2_visa.csv`（7200 行） |
| 日志 | `shards/f2_{1..4}.log` |

## 9. 下一步（已由判据预先指定）

F2 通过 → 按预注册计划进入 **Model V2 的 downstream smoke test**：

> **context-conditioned normal matching**（training-free）：
> 每个测试 patch 除自身特征 $z_i$ 外，再构造邻域 context descriptor $c_i$；
> 在 target normal bank 中先找 context 与 $c_i$ 相近的正常 patch，
> 再比较中心特征 $s_i=\min_{j\in\mathcal N(c_i)} d(z_i,z_j)$，
> 即从 $p(z)$ 转向近似建模 $p(z\mid\text{context})$。

smoke 判据（在跑之前冻结）：3–4 个已知低 $C_{\text{feat}}$ 的困难对象
+ 1–2 个对照，只跑 1/4-shot，**困难对象 AUPRO 平均 >= +2 且
image AUROC 不系统性下降超过 0.3**；过了才跑全量，不过立即停止。
