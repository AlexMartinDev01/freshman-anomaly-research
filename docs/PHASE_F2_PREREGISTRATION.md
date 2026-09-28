# Phase F2 预注册（在看到任何 VisA 结果之前冻结）

> 冻结时间：VisA 数据开始下载时，**尚未运行任何 VisA 特征提取、map 生成或统计**。
> 本文档写定后不得修改；若需变动，另开新文档并说明理由。

---

## 0. 为什么要做 F2

`PHASE_F1_FINDINGS.md` 在 MVTec AD v1 上发现：

> 两个结构完全不同的 detector（multi-layer kNN bank 与 PCA normal-subspace
> residual）对 **局部特征对比度 `C_feat`** 表现出几乎重合的单调依赖；
> 空间形态学（碎片化、形状）基本无关。

但 F1 的预注册判据含有一处**符号错误**，且该错误是在看到结果之后才发现的。
因此按 F1 文档的冻结表述：

> **preregistered inequality contained a sign error; therefore this dataset is
> treated as hypothesis-generating / supportive rather than an independent
> confirmatory validation.**

**同一批 MVTec 数据无法再构成独立确认。** F2 的任务就是：
在**从未看过的数据集**上，用**重新冻结的正确判据**做一次真正的独立确认。

## 1. 数据集

**VisA**（`VisA_20220922` → `tools/prepare_visa.py` → `1cls` 布局）。

- 12 个 category，与 MVTec 无重叠；
- 布局经 SubspaceAD 的 `VisADataset` 处理为
  `category/train/good/*.JPG`、`category/test/{good,bad}/*.JPG`、
  `category/ground_truth/bad/*.png`；
- **本项目此前从未在该数据集上运行任何实验。**

F2 开始前已确认：本预注册写入时，VisA 的下载仍在进行中，
**没有任何 VisA 的特征、map、或指标被生成或查看**。

## 2. 两个 detector（与 F1 一致）

| detector | 配置 |
|---|---|
| **kNN `agg_all3`** | 冻结 DINOv2 ViT-S/14 @448，block 5/8/11 的距离图 z-score 后平均，bank = k 张训练正常图 |
| **SubspaceAD** | 官方实现，ViT-S/14 @448，`pca_ev 0.99`，`agg_method mean`，**aug=0** |

配对方式与 F1 一致：同一 `(category, shot, split)`，同一 evaluator
（`pixel_metrics_binned(pro_limit=0.05)`）。shots {1,4} × 3 splits。

## 3. 逐 defect image 的测量（与 F1 相同的定义）

- `C_feat = median d(defect→bank) − median d(nearby normal→bank)`，raw DINO 末层原始距离；
  nearby = 32×32 网格上切比雪夫距离 ≤3 的非 GT patch。
  **注意：`C_feat` 使用 GT mask，是 oracle 诊断量，只能用于诊断，不得进入最终方法。**
- `M = median(score in GT) − P95(score outside GT)`（主判据）
- `TPR` = GT 内像素超过**该图自身** GT 外 P95 的比例
- 形态学：`area_frac`、`n_cc`、`largest_cc_frac`、`perimeter2_over_area`、`elongation`
- 逐行记录 `gt_covers_p95` 以监控阈值污染

## 4. 判据（**全部必须成立**）

设 `rho_c` 为 category `c` 内 `C_feat` 与 `TPR` 的 Spearman 相关（within-category）。

```
F2-1  两个 detector 的「跨 category 汇总 within rho(C_feat, TPR)」均为正，
      且 rho >= +0.4
F2-2  方向一致率：两个 detector 各自至少有 80% 的 category 满足 rho_c > 0
F2-3  C_feat 系数在控制 defect area / morphology 后仍显著为正，
      且标准误按 **category 聚类**（category-clustered SE）——
      不得使用把 1 万+ 张缺陷图当作独立观测的朴素 t 值
F2-4  M（连续 margin）上的结论与 TPR 一致（同为中等强度正相关）
```

**F2-1..F2-4 全过 → 机制跨数据集复现，进入 Model V2 smoke test。**
**任一条不过 → 机制未复现，停止，不建模型。**

### 明确不作数的检验

- **不**使用跨全部点的 pooled correlation 作为主判据
  （F1 的 `rho=-0.526` 就是被对象间混杂撑起来的）。
- **不**在 F2 之后调整上述任何阈值。

## 5. 已知的、预先声明的风险

1. **`C_feat` 与 detector 无关，但与 GT 有关。** VisA 的 GT 质量、mask 分辨率
   与 MVTec 不同，可能改变 `C_feat` 的分布。这**不是**允许放宽判据的理由；
   若因此不通过，结论就是"未复现"。
2. **VisA 只有 `good`/`bad` 二分类**，没有 MVTec 那样的 defect subtype。
   因此 F2 只能做 category 级，做不到 defect-type 级细分。
3. **`TPR` 的逐图阈值在 GT 占比大时受污染**，与 F1 同样处理：
   记录 `gt_covers_p95`，并做排除 GT>15% 的稳健性复核（作为复核，不替代主判据）。
4. VisA 图像分辨率与宽高比与 MVTec 差异较大，`448` 短边缩放后 patch 网格可能不是 32×32。
   若网格变化，`C_feat` 的邻域半径仍取切比雪夫 ≤3 patch，保持定义一致。

## 6. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `cache_visa.py`（待写）、`phase_f2_visa.py`（测量）、复用 `phase_f1_analysis.py`（统计） |
| 数据 | `data/visa_raw/`、`data/VisA_pytorch/` |
| 本文件 | `docs/PHASE_F2_PREREGISTRATION.md`（冻结，不得修改） |
