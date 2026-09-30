# R5 Raw-Map Causal Intervention Kit

## 你现在只跑 MAIN
把整个 `r5_rawmap_causal_interventions` 文件夹放到：

`E:\work\freshman\r5_rawmap_causal_interventions\`

使用原来跑 formal TSN 的同一个 Python/conda 环境。

PowerShell：

```powershell
cd E:\work\freshman
Set-ExecutionPolicy -Scope Process Bypass
.\r5_rawmap_causal_interventions\RUN_R5_MAIN.ps1
```

## MAIN 做什么
不是新模型，也不调参。

- 27 个对象全部纳入
- 固定 4-shot
- support split 0/1/2 全部纳入
- 共 81 个真实 cells
- 直接从原 DINO cache 重建 final-layer raw NN-distance maps

R5-A:
只提高真实 GT defect patches 的 score，位置/数量/clean map 不变。
检验 top1 和 q 对“真正 defect strength 增强”的单调性。

R5-B:
在真实 normal map 上植入固定强度、不同 patch 数的 synthetic anomaly。
强度固定，只改变 extent。
这是对 q 是否存在“异常越多反而分数越低”的直接因果反例。

R5-C:
只看 top-tail patch 的空间位置。
对每张 map 做 100 次 histogram-preserving spatial shuffle。
如果 original coherence 能稳定超过 shuffle null，说明 histogram-only image score 丢掉了独立空间证据。

## 为什么不用之前挑出来的 42 cells
因为这次不再做 selection-based mechanism discovery。
MAIN 用全部 27 objects，避免只在 screw/macaroni/bottle 等已知 regime 上验证。

## 为什么只用 4-shot
为了冻结一个中间 support regime并控制计算量。
shot=2/8 的 split=2 已经写入 LOCKED replication 文件，MAIN 结论冻结前不要运行。

## 输出
`E:\work\freshman\results\model_v0\metrics\r5_rawmap\`

重点文件：
- `R5_cells.csv`
- `R5A_strength_per_image.csv`
- `R5A_strength_auc.csv`
- `R5A_object_summary.csv`
- `R5B_extent_normal_background.csv`
- `R5B_object_summary.csv`
- `R5C_coherence_original.csv`
- `R5C_coherence_shuffle.csv`
- `R5C_vs_shuffle.csv`
- `R5_MAIN_REPORT.md`

跑完请把整个 `r5_rawmap` 文件夹打包上传。

## 严格规则
1. MAIN 结果出来之前不改 dose、不改 alpha、不改对象。
2. 不按结果选择“最好 alpha”当结论。
3. headline 统计单位是 object，不是 image/cell。
4. R5-C 如果方向跨对象翻转，就否定“单一 coherence scalar”，不能事后按类选正负号。
5. MAIN verdict 写死以后，才允许运行 `RUN_R5_LOCKED_REPLICATION.ps1`。
