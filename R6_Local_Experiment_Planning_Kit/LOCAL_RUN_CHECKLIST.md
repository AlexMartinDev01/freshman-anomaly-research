# 本机执行检查清单

## 开始前
- [ ] 使用与 formal TSN 相同 conda/python 环境
- [ ] 不更新 torch / sklearn / numpy
- [ ] git head/status 已记录
- [ ] raw cache / GT mask 可读
- [ ] CUDA 正常
- [ ] R5 MAIN 已备份

## R5 Replication
- [ ] 运行 locked 2/8-shot
- [ ] 不改参数
- [ ] 上传完整 replication 包
- [ ] 冻结 verdict 后再进入 R6

## R6-A
- [ ] 先 smoke 6 objects
- [ ] grouped folds 落盘
- [ ] Logistic/SVM 共用 folds
- [ ] 禁止 patch random split
- [ ] smoke replay 后才全 27 objects

## R6-B
- [ ] 共用 R6-A folds
- [ ] block5/8/11/final 全部跑

## R6-C
- [ ] 仅 representation sufficient 后运行
- [ ] normal PCA 只用 normal train/support
- [ ] 95% primary；90/99% sensitivity
- [ ] random-direction null 1000

## R6-D
- [ ] alpha 固定 0.5/1/2/5%
- [ ] 不挑 best alpha
- [ ] grouped CV
- [ ] baseline 与 spatial 使用同 folds

## 收尾
- [ ] per-object / per-dataset CSV 保存
- [ ] ROOT_CAUSE_VERDICT.md
- [ ] 整个 r6 目录打包上传
