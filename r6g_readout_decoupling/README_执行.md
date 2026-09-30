# R6-G 完整执行说明

这是 R6 的最后一个根因实验。

## 目标

不是继续找新的传播现象。

只回答最后一个因果桥：

**传播增强时，defect cue 是真的在深层消失了，还是 cue 仍然可线性读取，但 normal-only 1NN 越来越读不出来？**

---

## 前置依赖

你的本地项目里必须已经存在：

1. `r6a1e_preencoding\`
2. `r6a1f_kit\`
3. `results\model_v0\metrics\r6a1e_preencoding\`
4. `results\model_v0\metrics\r6a1f_layerwise\`
5. `results\model_v0\metrics\r6a_smoke\`

这正是你当前已经完成的实验环境。

脚本不会重抽 fold，不会重抽 support。

---

## 放置

将整个：

`r6g_readout_decoupling`

放入：

`E:\work\freshman\r6g_readout_decoupling\`

---

## 运行

```powershell
cd E:\work\freshman
Set-ExecutionPolicy -Scope Process Bypass
.\r6g_readout_decoupling\RUN_R6G.ps1
```

---

## 第一阶段会强制 STOP 检查

它会验证：

- R6-A smoke folds 原样存在
- support IDs 与 smoke 逐字一致
- A1F propagation curve 原样存在
- A1E test path manifest 原样存在
- dense block11 与 final cache parity
- support train image 路径可以唯一解析
- test image 名称与 frozen folds 完全一致

任何一个不一致：

**直接 STOP。不要绕过。**

---

## 计算量

每个对象需要：

- 1 次 dense pass：收集固定 sampled training patches
- 1 次 dense pass：做所有 held-out OOF evaluation
- 4 个 support images 的 dense extraction

所以不是 12 层分别跑 12 次 DINO。

一次 forward 同时取 12 blocks。

临时 sampled feature 用 float16 memmap。
默认对象跑完自动删除临时文件。

---

## 指标

逐层输出：

### Normal-only 1NN
- Patch AUROC
- Patch AP
- Image AUROC
- Precision@Top1%
- Recall@Top1%

### Supervised oracle
两套都跑，不择优：

- Logistic Regression（PRIMARY）
- Linear SVM（robustness）

同样输出全部五项。

---

## 为什么 Patch AP 是 primary

你之前已经实验证明：

macaroni2 可以出现：

- Patch AUROC ≈ 0.97
- 但 AP ≈ 0.31
- Image AUC 又明显崩掉

所以只看 AUROC 会掩盖极少量 extreme clean false positives。

R6-G 用 AP gap 作为 readout distortion 的 primary bridge。

---

## 关键输出

结果目录：

`E:\work\freshman\results\model_v0\metrics\r6g_readout_decoupling\`

请完整上传。

尤其：

- `parity_report.csv`
- `support_path_manifest.csv`
- `fold_metrics.csv`
- `per_image_metrics.csv`
- `object_layer_metrics.csv`
- `layerwise_readout_gaps.csv`
- `propagation_gap_relations.csv`
- `probe_states.csv`
- `R6G_FROZEN_OBJECT_VERDICTS.csv`
- `R6G_REPORT.md`
- `R6G_INPUT_SHA256.txt`

---

## 最终四种判决

### READOUT_DISTORTION_SUPPORTED

说明：

- propagation 已经被 A1F 证明
- supervised cue 到 final 仍保留
- 但 normal-only gap 随 propagation 增强

这时 R7 走：

**Context-Stable / Propagation-Aware Normality Modeling**

---

### CUE_FADING_SUPPORTED

说明：

supervised defect separability 本身随深度明显下降。

R7 走：

**defect-cue-preserving / intermediate-layer representation**

---

### PROPAGATION_DECOUPLED_FROM_READOUT

说明：

传播是真现象，但不是导致 1NN readout failure 的主因。

关闭 propagation-as-root-cause。

R7 转：

**D-vs-N_H directional identifiability / normality geometry**

---

### MIXED_INDETERMINATE

不再继续新增 R6 传播实验。

进入 R7 时按 object regime 拆分验证。

---

## 研究纪律

这一轮之后：

**R6 结束。**

无论结果漂亮还是不漂亮，都不要再无限追加 A1G/A1H。

后面必须进入 R7 方法设计。
