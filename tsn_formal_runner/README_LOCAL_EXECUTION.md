# TSN 本机正式过关实验执行说明

## 为什么需要在原项目环境执行
正式 TSN 不是只读取已经导出的 image-level CSV。它必须重新访问：
1. train support 的 raw DINO patch features；
2. test raw DINO patch features；
3. 现有 split/support selection；
4. 原始 1NN scoring；
5. MVTec 与 VisA 的完整 cache。

因此可在任何有这些资源的机器执行；当前最自然的是 E:\work\freshman 本机环境。

## 目录放置
把整个 `tsn_formal_runner` 文件夹复制到：

E:\work\freshman\tsn_formal_runner\

不要覆盖任何旧代码、旧结果或 frozen V3 输出。

## 科学协议（禁止改）
shots = 2,4,8
splits = 0,1,2
27 categories = 15 MVTec + 12 VisA
total = 243 cells

q = log(max(raw-final-1NN) / mean_top1%(raw-final-1NN))

support calibration:
对每个 support image，用其余 k-1 张 support 建 normal bank 给它打分；
q_ref = 所有 support q 的 median。

test:
用完整 k-shot bank 给 test image 打分；
TSN = abs(q_test - q_ref)。

No defect labels.
No test-normal calibration.
No post-result threshold tuning.

## Frozen Gates（每个 dataset 单独判）
G1 mean delta vs agg_all3 > 0
G2 >=70% cells have delta >=0
G3 bottom baseline quartile mean delta >= +2.0 AUROC points
G4 mean TSN >= mean agg_all3
MVTec AND VisA 都必须 ALL_PASS。

## 执行顺序
1. PowerShell 进入 E:\work\freshman。
2. 如使用 venv/conda，先激活原来能跑模型实验的环境。
3. 运行 RUN_01_PREFLIGHT.ps1。
4. PREFLIGHT 必须显示：
   - CUDA available True
   - MVTec cached objects = 15
   - VisA cached objects = 12
   - TOTAL = 27
   - 所有 CACHE OK
5. 跑 RUN_02_SMOKE.ps1。
6. 检查 smoke:
   - 无 import / cache / CUDA / OOM 错误
   - baseline AUROC 数值范围合理
   - q_ref / q_support 非 NaN/Inf
7. 正式跑 RUN_03_FORMAL_243.ps1。
8. 中途若电脑重启/程序中断，重新运行同一个 ps1；`--resume` 会跳过已完成 cell。
9. 不要开第二个 GPU runner。
10. 正式结束后保留：
   - tsn_formal_v2.csv
   - tsn_formal_v2_gate.csv
   - tsn_formal_v2.log
   - 当前 git HEAD
   - pip freeze 或 conda env export
   - nvidia-smi 信息

## 正式结果不通过时怎么处理
不要改 gate，不要先调公式。
先分类：
A. MVTec PASS, VisA FAIL -> 跨数据集泛化问题。
B. mean positive but G2 FAIL -> 少数类别收益过大、普适性不足。
C. G3 FAIL -> 没有真正修 hardest regime，机制故事不成立。
D. baseline replay 与历史结果不一致 -> 实现/环境问题，先停止科学解释。
E. 只有某 shot FAIL -> shot-dependent calibration问题，单独研究，不得合并宣称。

## 一个必须增加的 baseline replay sanity check
正式结果出来后，应把 runner 重算的 `baseline_agg_img_AUROC` 与历史 frozen baseline 的同 cell 指标对齐。
如果明显不一致，先排查：
- support selection 是否一致；
- cache 版本是否一致；
- `load_obj("r0")` 是否对应 frozen baseline；
- `mean_top1p` 版本；
- layer 名称和 z-score population；
- split seed。
在 baseline replay 未对齐前，TSN 的正式 PASS/FAIL 都不能作为论文结论。

## Patch-count stress
当前导出 AD2 每个 object 内 patch count 固定，无法直接验证 q 是否受 patch 数影响。
正式 raw map 可用后，对同一 test map做固定内容随机 patch subsampling：
100%,75%,50%,25%，每级重复100次。
同时比较 formal top1% 与 TSN AUROC。
若 TSN 随 patch count 大幅漂移，则需要重新设计 order-statistic normalization。

## TSN + 560
先单独通过 TSN 243-cell。
然后才允许把 image head=TSN 与 localization head=560 组合。
组合阶段必须验证：
- image AUROC 不低于正式 TSN；
- AUPRO/pxAUROC 达到或超过冻结 560；
- 总 feature/scoring overhead；
- 不能重新引入 crop-local normalization。
