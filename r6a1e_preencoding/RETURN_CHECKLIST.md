# 结果返回时请一并上传

请上传整个：
`E:\work\freshman\results\model_v0\metrics\r6a1e_preencoding\`

不要只发终端截图。

我会按以下顺序复核：

1. parity_report.csv
2. donor_manifest.csv — donor 是否真的不在 support
3. survivor_manifest.csv — 是否全 radius 同一批 bad
4. raw_full_good_baseline.csv — 与 R6-A1D survivor-matched raw 轨迹是否对齐
5. object_donor_summary.csv — 两 donor 是否同方向
6. per_bad_summary.csv — 结果是否被少数图主导
7. ring_bad_minus_good.csv — propagation 是否随距离衰减
8. FROZEN_OBJECT_VERDICTS.csv — 最后才看自动判决

如果前 4 项任一失败，后面的机制判决全部作废。
