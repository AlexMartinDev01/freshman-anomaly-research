# R6 本机实验总计划（冻结版）

## 总原则
本阶段不设计最终算法，不追求“先涨点”，只解决四个科学问题：

1. R5 MAIN 的结论是否跨 shot 复现？
2. Frozen-DINO patch feature 本身是否包含足够的 defect information？
3. defect cue 是否随网络层级衰减？
4. 若 feature 中有信息，为什么 normal-only readout 识别不到？空间结构是否提供独立补充信息？

实验顺序不得改变：

R5 Locked Replication
→ R6-A Supervised Frozen-Feature Upper Bound
→ R6-B Layer-wise Signal Localization
→ R6-C Defect Direction vs Normal Manifold
→ R6-D Spatial Conditional Complementarity
→ 冻结 Root-Cause Verdict
→ 才允许设计 R7 方法

任何一步 FAIL 都按 Stop Rule 转向，不为了救某条路线改阈值。

---

# Phase 0 — R5 Locked Replication

## 目的
确认 R5 MAIN 在 4-shot 得到的机制结论不是 shot-specific。

## 已锁定
- shot = 2, 8
- split = 2
- 27 objects
- 共 54 cells
- 不改 dose / alpha / GT threshold / statistic

## 必须复现的结论
### R5-A
Top1 对 strength intervention 单调：
- object-level mean rho_top1 > 0
- MVTec / VisA 同方向

不要求 q 在 strength 上统一反向；MAIN 已经否定“q universal strength pathology”。

### R5-B
Extent-only：
- 真实 normal background
- 固定 implant strength
- 仅增大 affected patch count
- Top1 必须 non-decreasing
- q 应出现结构性下降

### R5-C
Spatial：
- histogram-preserving shuffle
- original adjacency/LCC 与 shuffle null 比较
- 重点验证空间信息跨 shot 是否仍真实存在

## Verdict
若 MAIN 与 replication 同方向：
R5 root-cause conclusions CLOSED/FROZEN。

若 replication 明显翻转：
不得进入 R6 method inference，先调查 shot-specific mechanism。

---

# R6-A — Frozen-DINO Supervised Upper Bound

## 核心问题
Frozen DINO feature 里面到底有没有 defect information？

这是诊断实验，不是最终方法。
允许使用异常 GT label，但结果只能作为 representation upper bound。

## 特征
对每个 object 分别读取：
- block5
- block8
- block11
- final

保持原始 frozen feature，不 finetune backbone。

## Patch label
Primary：
- defect patch: gt_frac > 0.10
- clean patch: gt_frac == 0
- 0 < gt_frac <= 0.10：排除，不训练、不测试

## 防泄漏
绝对禁止 patch-level random split。
必须按 image group 划分；有 scene ID 时优先按 scene group。

## Probe
1. L2 Logistic Regression
2. Linear SVM

只允许线性 probe。

## 训练采样
- 每张 image defect ≤512 patches
- clean ≤512 patches
- class_weight = balanced
- test 使用全部合法 patches

## CV
Primary：5-fold Group CV；不足时 3-fold。
所有 layer/probe 共用完全相同 folds。
Secondary：重复 3 次 grouped split 做 stability。

## Metrics
Patch-level：AUROC、AP。
Image-level diagnostic：supervised defect probability 的 top1% mean → image AUROC。

## 与 normal-only baseline 对比
在同一 held-out images 上计算原始 final-layer 1NN patch distance 的 patch AUROC / image top1 AUROC。

## R6-A Gate
Representation-sufficient 证据：
- hard objects supervised patch AUROC >= 0.90；
- 相对 1NN patch AUROC 平均 >= +10 points；
- >=70% hard objects 成立；
- MVTec / VisA 均为正方向。

或 defect-type-held-out：
- mean gain >= +5 points；
- >=60% eligible objects positive。

Representation-bottleneck：
若所有层在 hard objects supervised patch AUROC < 0.80 且相对 normal-only gain < +5，则优先 representation branch。

---

# R6-A2 — Defect-Type Held-Out

对有 >=2 defect types 的 object：每次留一 defect type 做 test，其余 defect types + normal 做 train；normal image 也必须 held out。

必须报告 seen-type / unseen-type AUROC 及 drop。

---

# R6-B — Layer-wise Signal Localization

完全复用 R6-A 的 folds、labels、probe、采样。
层：block5 / block8 / block11 / final。

Primary：每 object 计算 supervised patch AUROC 随 depth 的 Spearman / slope，并报告 final-block5、final-block8。

Cue-fading Gate：
- >=70% hard objects depth slope/rho < 0；
- object-level exact sign p < .05；
- MVTec / VisA 同方向；
- early/mid 对 final 平均优势 >= +3 AUROC points。

不满足则不得宣称“深层丢失 defect cue”。

---

# R6-C — Defect Direction vs Normal Manifold

仅在 R6-A 表明 representation sufficient 时运行。

每 fold：
1. supervised train patches fit Logistic Regression → w；
2. ORIGINAL normal train/support feature 建 PCA/SVD；
3. primary normal subspace = 95% variance；90/99% 仅 sensitivity；
4. w_parallel = U U^T w；w_perp = (I-U U^T) w；
5. held-out patch 上分别算 score_full / score_parallel / score_perp AUROC。

Random-direction null：每 fold 1000 random unit vectors，得到 normal-subspace energy null。

判定：
- AUC_perp 接近 full（<=3 points）、perp >> parallel、eta_perp > null q95、>=70% hard objects同方向 → outside-manifold branch。
- AUC_parallel 接近 full、eta_parallel 显著高于 null、而 normal-only readout仍差 → in-manifold identifiability bottleneck。
- 两者都保留明显信息 → mixed geometry，不强行归类。

---

# R6-D — Spatial Conditional Complementarity

R5-C 已证明 spatial arrangement 是真实信号，但还不知道是否在 Top1 之外提供独立信息。

冻结 spatial scales：0.5%, 1%, 2%, 5%。禁止事后挑 best alpha。

Baseline diagnostic model：[Top1]
Spatial diagnostic model：[Top1 + 4个 adjacency + 4个 LCC]
模型只允许 L2 Logistic Regression。

CV：image/scene grouped CV。

Gate：
- mean object ΔAUROC > 0；
- >=70% objects Δ>=0；
- hard quartile mean gain >= +2 points；
- MVTec / VisA dataset mean都不下降；
- object-level Wilcoxon p<.05。

否则：Spatial = real but redundant，关闭 fusion。

---

# 决策树

1. Supervised all layers weak → Representation bottleneck → R7 研究 defect-cue-preserving / multi-layer representation。
2. Supervised strong + perp dominates → normal-manifold/non-conformity readout bottleneck。
3. Supervised strong + parallel dominates → normal-only identifiability bottleneck → query-conditioned / relational / structural evidence。
4. Spatial conditional PASS → spatial 作为 complementary branch。
5. Spatial conditional FAIL → spatial 不进最终方法。

---

# 严格执行顺序

1. R5 Locked Replication
2. 冻结 R5 verdict
3. R6-A final-layer smoke（6 objects：MVTec screw/bottle/cable；VisA macaroni2/pcb2/chewinggum）
4. smoke replay 正常后扩 27 objects
5. R6-A2 defect-type-held-out
6. R6-B all layers
7. 若 representation sufficient 才运行 R6-C
8. 独立运行 R6-D
9. 冻结 ROOT_CAUSE_VERDICT
10. 才允许设计 R7

---

# Provenance 必存
- git rev-parse HEAD
- git status --short
- Python / torch / CUDA / GPU
- dataset roots
- cache paths
- selection CSV
- fold assignment CSV
- random seeds
- command line
- start/end time

所有 folds 必须落盘，layer/probe 共用，不可重抽。

---

# 输出目录
results/model_v0/metrics/r5_replication/
results/model_v0/metrics/r6/r6a_supervised_upper_bound/
results/model_v0/metrics/r6/r6a2_defect_type_heldout/
results/model_v0/metrics/r6/r6b_layer_audit/
results/model_v0/metrics/r6/r6c_normal_manifold/
results/model_v0/metrics/r6/r6d_spatial_complementarity/
results/model_v0/metrics/r6/ROOT_CAUSE_VERDICT.md

---

# Stop Rules
1. 不因为某 layer 好就改变主 Gate。
2. 不因为某 alpha 好就挑最好 spatial scale。
3. 禁止 patch random split。
4. 禁止用 test labels 设计最终 router。
5. supervised probe 只能作为 upper bound。
6. headline statistical unit = object。
7. MVTec / VisA 分开报告。
8. 新机制必须先过 small true-protocol，再跑 full formal。
