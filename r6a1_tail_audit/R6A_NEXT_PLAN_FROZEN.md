# R6-A Smoke 审计结论与下一步冻结计划

## Smoke primary verdict
原预注册 smoke Gate：FAIL。
hard objects 中 “probe AUROC>=0.90 且相对 1NN gain>=+10 points” 仅 screw 满足，1/3。
不得事后修改此 Gate，也不得把 smoke 重新宣布为 PASS。

## 但已建立的 secondary evidence
1. final-layer frozen DINO 在 6/6 对象上都存在很强的 label-informed linear signal；
   logreg patch AUROC 范围约 0.978–0.997，三个 hard objects 均 >=0.981。
2. supervised oracle 与 normal-only 1NN 之间存在明显 information-extraction gap，
   但这尚不能区分“可修复 readout”与“normal-only identifiability fundamentally difficult”。
3. AUROC 对稀疏缺陷会掩盖 extreme-tail failure。hard cases 上 AP 差距远大于 AUROC 差距。
4. macaroni2 是关键机制对象：GT defect 极稀疏，median 2 patches / image，
   top1% 约 16 patches，100% 有可见 defect 的 bad images 都满足 defect_count < top1 width。
   同时 1NN median within-image patch AUROC≈0.999，而 image AUROC≈0.654。
   因而“defect 在本图内部不可见”不是主要解释；需要分离 fixed-tail dilution 与 cross-image nuisance-tail calibration。
5. pooled OOF image AUC 与 fold-wise AUC 均值几乎一致，当前 image 结论不是 fold-score calibration artefact。
6. LogReg 与 LinearSVM 的 patch AUROC很接近，但部分对象 image AUC差异明显，
   证明 top-tail mean 对 score scale/shape 敏感；不能把 probe-vs-1NN image gap直接归咎于聚合算子本身。

# 下一步：R6-A1 Tail Geometry & Calibration Audit（先 6 对象，不扩 27）

这一步仍然不是新方法，只是根因分解。

## A1. Tail occupancy / purity
固定 alpha = {0.1%, 0.25%, 0.5%, 1%, 2%, 5%}。
对 1NN 与 probe 分别记录：
- top-alpha GT precision
- top-alpha GT recall
- selected clean-patch fraction
- defect_count / selected_count
- bad-image clean-tail score
- good-image tail score

核心问题：
真实 defect 是否进入 tail，但被大量 clean extreme patches 稀释？

## A2. Oracle extent-matched diagnostic
对每张 bad image，令 m = GT defect patch count。
与每张 good image比较时，对两张图都使用同一个 m：
T_m(x)=mean top-m score。
形成 bad-good pairwise oracle AUC。

这是 diagnostic oracle，不可部署。

如果 sparse hard objects 的 oracle-extent AUC 相比 fixed top1% 明显恢复，
说明 fixed 1% support mismatch 是真实 contributor。

## A3. Cross-image calibration diagnostic
至少做两个预冻结 arm：

1. raw 1NN
2. per-image robust relative evidence:
   z_p = (d_p - median(d)) / (1.4826*MAD(d)+eps)

并保留相同 alpha grid。
这不改变同图 patch ordering，但改变跨图 scale。

若 within-image ranking 已强的对象（尤其 macaroni2）在 robust-relative arm 显著恢复，
支持 cross-image nuisance/scale shift。

## A4. Normal-only monotonic calibration control
构造 support-normal LOO score distribution F_N。
将测试 1NN distance 做固定的 monotonic normal-CDF calibration：
u_p = F_N(d_p)
或 Gaussianized z = Phi^{-1}(clip(u_p))。

重要性质：
- patch ranking/AUROC理论上保持不变；
- image top-tail mean可能改变。

如果 patch AUROC不变而 image AUROC明显改变，
则直接证明 image aggregation 对 score calibration/shape 敏感。

不得使用 test label 拟合 calibration。

## A5. Decomposition
对每个 bad image 分解 fixed top1%：
- 真 defect 被选中的数量
- 被选中的 clean 数量
- defect selected mean
- clean selected mean
- final top1 mean

这可以定量回答：
image collapse 是“缺陷没进 tail”，还是“进了但被 clean tail 稀释”。

# A1 Frozen interpretation rules

不把单一阈值当最终 paper Gate；6 对象仍是 mechanism-discovery set。
只允许形成待 27-object confirm 的 hypothesis。

支持 “sparse fixed-tail dilution”：
- sparse hard objects 的 top1 GT precision低；
- oracle extent-matched pairwise AUC相对 fixed 1% 明显恢复；
- dense easy controls恢复很小。

支持 “cross-image calibration”：
- within-image defect ranking高；
- raw image AUC低；
- normal-only monotonic / robust calibration在不改善 patch ranking的情况下恢复 image AUC。

两者可以同时成立，不强行二选一。

# 然后才做 R6-A Full-27 confirmatory

Full-27 必须在看到 A1 后重新预注册，不沿用 smoke 的 +10 gain Gate。
Primary 应包含：
- fold-wise patch AUROC
- fold-wise patch AP
- absolute supervised upper bound
- 1NN vs probe AP gap
- patch→image collapse
- defect/top1 occupancy
- dataset-separated statistics

Full-27 之后：
- 若 final feature普遍有强 linear signal -> R6-A2 defect-type-heldout -> R6-C identifiability/manifold
- 若一批对象 final probe仍弱 -> 对这些对象进入 R6-B layer audit
- R6-D spatial conditional complementarity仍独立保留，暂不融合。
