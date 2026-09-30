# R6-A1E 完整执行说明

## 这一步到底测什么
R6-A1D 是“DINO 已经看完整张图以后，再把 defect 位置的 score 删掉”。

R6-A1E 是：

**先把 defect + context 在 DINO 输入端替换成正常像素，再重新跑 DINO。**

所以它第一次能区分：

1. 远场像素本身就带 bad/good 差异；
2. defect 信息通过 ViT self-attention 被传播进了远场 token。

---

## 文件夹放置

解压后将整个：

`r6a1e_preencoding`

放到：

`E:\work\freshman\r6a1e_preencoding\`

最终应有：

- `E:\work\freshman\r6a1e_preencoding\R6A1E_CONFIG.json`
- `E:\work\freshman\r6a1e_preencoding\R6A1E_SELECTION.csv`
- `E:\work\freshman\r6a1e_preencoding\R6A1E_PREREGISTRATION_FROZEN.md`
- `E:\work\freshman\r6a1e_preencoding\r6a1e_utils.py`
- `E:\work\freshman\r6a1e_preencoding\r6a1e_precheck.py`
- `E:\work\freshman\r6a1e_preencoding\r6a1e_preencoding_counterfactual.py`
- `E:\work\freshman\r6a1e_preencoding\analyze_r6a1e.py`
- `E:\work\freshman\r6a1e_preencoding\RUN_R6A1E.ps1`

---

## 数据集路径

脚本会自动寻找常见位置。

MVTec 默认候选：
- `data\mvtec_anomaly_detection`
- `data\MVTec`
- `datasets\mvtec_anomaly_detection`

VisA 默认候选：
- `data\VisA_pytorch\1cls`
- `data\VisA\1cls`
- `datasets\VisA_pytorch\1cls`

如果你的路径不在这些位置：

编辑：

`R6A1E_CONFIG.json`

把：

```json
"mvtec_root": "",
"visa_root": ""
```

改成真实绝对路径即可。

不要改其他实验参数。

---

## 运行

PowerShell：

```powershell
cd E:\work\freshman
Set-ExecutionPolicy -Scope Process Bypass
.\r6a1e_preencoding\RUN_R6A1E.ps1
```

---

## 第一步会先做 Parity Gate

这是强制的。

程序会从原始图片重新走：

`AnomalyDINO prepare_image -> DINOv2 feature`

再与现有 `load_obj("r0")` 的 final cache 比较。

冻结要求：

- grid 完全一致
- feature shape 完全一致
- median cosine >= 0.9995
- p01 cosine >= 0.995

只要任意一个不满足：

**程序立即 STOP。**

这时不要自己改阈值，不要绕过检查，把 `parity_report_FAILED.csv` 发回来。

---

## Normal donor

每个对象固定使用：

**2 张不属于 4-shot support bank 的 train/good 图片**

作为正常像素 donor。

如果找不到 2 张可解析且非 support 的 donor：

程序 STOP。

不会偷偷改成 support donor 或 test donor。

---

## Primary analysis set

为了修复 A1D 的样本缩水问题：

先根据 r=16 几何 mask 冻结：

`valid patches >= 10`

的 bad images。

然后 r=0/2/4/8/16 始终使用同一批 bad images。

所以主轨迹不再混入 attrition confound。

---

## 输出目录

`E:\work\freshman\results\model_v0\metrics\r6a1e_preencoding\`

关键文件：

### 前置审计
- `parity_report.csv`
- `resolved_test_paths.csv`
- `donor_manifest.csv`
- `pair_manifest.csv`
- `survivor_manifest.csv`

### 原始结果
- `per_pair.csv`
- `per_bad_summary.csv`
- `ring_profile.csv`
- `raw_full_good_baseline.csv`

### 聚合结果
- `object_donor_summary.csv`
- `donor_mean_summary.csv`
- `ring_bad_minus_good.csv`
- `FROZEN_OBJECT_VERDICTS.csv`

### 视觉检查
- `examples\`

---

## 最重要的结果怎么看

固定 headline：

`alpha = 1%`

只裁决：

`r=8` 与 `r=16`

### PROPAGATION_SUPPORTED
需要：

- raw matched PWR >= 0.60
- counterfactual 后至少掉 0.05
- 两个 donor 同方向下降
- bad far-field feature drift > good control
- paired Wilcoxon p < 0.05

含义：

**A1D 的部分远场信息其实来自被替换 defect/context 区域，经 DINO representation mixing 传播到远处 token。**

### FAR_FIELD_PERSISTENT
需要 r=8 和 r=16 同时：

- raw PWR >= 0.60
- counterfactual PWR >= 0.60
- 变化绝对值 < 0.03

含义：

**即便 DINO 从输入端没看到原 defect/context 像素，远场区域仍保持判别力。**

下一步不能直接利用它，而要审计：

- global material shift
- acquisition covariate
- unannotated anomaly
- dataset shortcut

### MIXED_INDETERMINATE
其它结果全部归这一类。

不能硬解释。

---

## 严禁做的事情

本轮不允许：

- 改 alpha
- 改 radius
- 改 GT threshold
- 换 donor 数
- 换 matched good 数
- 事后挑表现最好的 donor
- 因 macaroni2 特殊而删除它
- 扩 27 类
- 开始设计最终 detector

先把因果机制判清楚。
