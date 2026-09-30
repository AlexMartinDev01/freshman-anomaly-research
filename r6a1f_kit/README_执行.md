
# R6-A1F Layer-wise Causal Propagation Trace

目的：
验证 R6-A1E 发现的 encoder-mediated contextual propagation：
1. 从哪一层开始产生远端 token drift；
2. propagation 是否随网络深度增强；
3. 是否与 final anomaly score failure 对应；
4. 排除单纯 input replacement magnitude confound。

实验对象：
positive:
- bottle
- chewinggum
- cable

negative/control:
- macaroni2
- screw

不要修改冻结参数。
