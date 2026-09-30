
# Frozen Protocol

Primary questions:
Q1: defect-region counterfactual intervention 是否造成远场 token representation change?
Q2: propagation 是否随着 ViT layer 增深增强?
Q3: propagation strength 是否预测 image-level anomaly collapse?

Primary metrics:
- layer-wise far-field cosine drift
- bad-good drift difference
- radius profile
- score impact

必须同时报告:
- raw input difference
- feature drift
- good matched control

禁止:
- 挑 layer
- 挑 radius
- 挑 alpha
