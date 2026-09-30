# 返回结果检查顺序

上传整个：
`results\model_v0\metrics\r6g_readout_decoupling\`

复核顺序固定：

1. `R6G_INPUT_SHA256.txt`
2. `parity_report.csv`
3. `support_path_manifest.csv`
4. `fold_metrics.csv`
5. `object_layer_metrics.csv`
6. `layerwise_readout_gaps.csv`
7. `propagation_gap_relations.csv`
8. `probe_states.csv`
9. `R6G_FROZEN_OBJECT_VERDICTS.csv`
10. `R6G_REPORT.md`

前 3 项任意失败，后续结果全部不解释。

特别检查：
- block11 1NN final 指标应与既有 R6-A 同方向/近似量级；
- block11 supervised probe final 指标应复现 R6-A smoke；
- 不允许择优 LogReg/SVM；
- 不允许择优 layer。
