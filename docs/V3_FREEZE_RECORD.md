# Model V3 实现冻结记录

> 本文件在正式确认性实验**开始之前**写定。
> 从 `model_v3_implementation_frozen` tag 起，模型数值逻辑不再修改。

---

## 1. 科学预注册

| 文档 | commit | tag |
|---|---|---|
| 主预注册 | `742126a` | `model_v3_preregistered` |
| 补充 1（crop 尺寸与预算分配） | `92dbb27` | `model_v3_preregistered_addendum1` |
| 补充 2（仿射校准 self-exclusion） | `6dd0a26` | `model_v3_preregistered_addendum2` |
| 补充 3（pooling 与 exclusion 几何） | `697fb36` | `model_v3_preregistered_addendum3` |

判据、阈值、Gate 全程未修改。

## 2. 实现提交

| 内容 | commit |
|---|---|
| A3 端到端 | `b9d5e70` |
| A4 + A5 单一 refinement 路径 | `536b96c` |
| engineering smoke（transistor + pcb2） | `f0605c8` |
| 确定性 A4 seed 派生 | `c665ba3` |
| manifest + shard planner + dry-run | `6d0ce92` |
| cell runner + 事务式输出 | `3f7020d` |
| runner 两处静默 bug 修复 | `e3c4edf` |
| gate 3 + 诊断索引修复 | `ab5fd7c` |
| 预 V3 实验记录补录 | `7b59321` |
| **冻结提交** | 见 `model_v3_implementation_frozen` tag |

## 3. Manifest

```
路径      results/model_v0/v3/manifest.json
SHA256    d41aba198f3e3f33d1d7924aa4c7670c3880187fd54eab5ef452370848c92160
cells     324   (MVTec 180 + VisA 144)
A4 seeds  3240
```

分片（category 级贪心装箱，MVTec 成本 1.0 / VisA 3.0）：
2-worker 成本 26.0 / 25.0（失衡 1.020x）；3-worker 17.0 / 17.0 / 17.0。
**默认 2 worker**；3 路需并发探针的实测吞吐证明，不因分片漂亮而启用。

## 4. Seed 派生

```
seed(dataset, object, shot, split, seed_id, arm) =
    int.from_bytes(SHA256(f"{dataset}|{object}|{shot}|{split}|{seed_id}|{arm}")
                   .digest()[:8], "big")

A4: arm = "A4", seed_id = 0..9
锚点  seed("mvtec","bottle",1,0,3) = 2959745414427513892
```

## 5. 外部依赖

| 依赖 | 来源 | 状态 |
|---|---|---|
| AnomalyDINO | `dammsi/AnomalyDINO` @ `b9d1c26` | 无本地改动；13 GB，**不进库**，但是 V3 **运行时依赖**，必须在磁盘上 |
| SubspaceAD | `CLendering/SubspaceAD` @ `ef56d5c` | patch `feadd999b1759a99c7d8dae70ca98055737d98db2e1de135aa8c7532b498c077`；非 V3 运行时依赖 |

详见 `third_party_patches/README.md`。

## 6. 环境

```
OS          Windows-11-10.0.26200
Python      3.14.4
numpy       2.4.4
scipy       1.17.1
scikit-learn 1.8.0
torch       2.11.0+cu128
torchvision 0.26.0+cu128
CUDA rt     12.8
cuDNN       91900
GPU         NVIDIA GeForce RTX 5060
VRAM        8.0 GB          <-- 见 §8 说明
```

## 7. 已完成的验收

**合成 / 单元**

```
尺度几何（672-equivalent）        v3_check_scale.py    C1 几何精确、C2a 逐位等于全局切片、负对照有判别力
bank-level NN exclusion          v3_calibration.py    nearest-自身被禁->返回次近、两近被禁->第三、
                                                       k=1 远端可用、k>1 他图不受影响、角点裁剪、
                                                       全禁 fail-fast、tie 可复现、672 映射
672->448 pooling                 v3_calibration.py    线性性、质量守恒、常数场复现、精确覆盖
已知真值仿射                      v3_calibration.py    a=2/b=0.3 恢复、方向敏感、|a|>100 fail-fast
support-only 隔离                 v3_refine.py         shot=1/4 只触及抽中的 k 张
分辨率 bank 隔离                   v3_refine.py         448 cache 被拒绝
写回几何                          v3_refine.py         位置、形状、overlap=mean、仿射、无转置
seed 派生                        v3_refine.py         跨进程锚点一致、各分量改变 seed、0 碰撞
resume 校验                      v3_cellrunner.py     8 个 case 全判对
```

**真实数据 / 端到端**

```
真实 affine 健康检查              a∈[0.80,0.93]，全 finite、a>0、|a|≤100，support_ids 与 S_k 一致，
                                 N_pairs = k x n448（含非方形 pcb2 32x41）
A3 端到端                        bottle 81/83 与 transistor 47/100 的 m=0 图逐位等于 A0
A4/A5 + 编排层 §10-F             12 个臂 (m',n,T) 逐图相等；m=0 在每个臂都逐位等于 A0
engineering smoke                transistor + pcb2，无断言失败、无 NaN/Inf
direct-vs-runner 逐位一致        12 臂 + a0 + a2 np.array_equal；诊断量逐字段一致
resume / hash 验收               第二次运行 skipped 0s，全部文件 hash 不变
非方形序列化验收                  pcb2 448(200,32,41) / 672(200,48,62)，pos 往返仍为列表
```

## 8. 已知诊断与非阻塞观察（**均未改变任何预注册阈值**）

1. **support-only 校准分布不迁移**：校准集上 `P(hot≥tau_hot)` 按构造恒为 1%，
   测试图上实测 0.3–4.4%。根因是 k=1 时 `d_cal` 量的是同图自相似距离，
   与跨图匹配不同分布。按冻结声明，触发率是结果不是调参目标。
2. **selector 判别力因对象而异**：transistor 上 bad/good 触发率比达 20–400×，
   screw 在 8-shot 近乎不判别（0.133% vs 0.114%），screw 1-shot 触发为 0。
   这意味着 G3 的检验力集中在 selector 真正开火的对象上。
3. **G3 反事实比例更正**：transistor k=1 s0 正确值为 **53/53**；
   此前报告的 29/53、30/53 因诊断索引错位（按 refined 个数迭代却按全量下标索引）
   而作废。该更正不改任何 map（详见 `MODEL_V3_IMPLEMENTATION_NOTES.md` I8）。
4. **affine R² 范围 0.35–0.79**，非方形 VisA 在 k 增大时下降。
   按冻结约定**仅作诊断**，不得据此改用 isotonic / quantile / 分段 / log 校准。
5. **触发率并非普遍低**：`n=33`（单触发→最大上下文）在 transistor 与 pcb2 上
   都出现，`K_max` 逐图不同（transistor 46 / pcb2 59），确认预算按每图网格计算。
6. **8 GB 显存是本机瓶颈**。此前多次 OOM 均由重任务并发导致；
   正式主实验前必须做 2-worker 并发探针，且不得因分片均衡而默认升到 3 worker。

## 9. 正式确认性结果根目录

```
results/model_v0/v3/
```

**初始状态：已完成 cell = 0 / 324，A4 随机臂 = 0 / 3240。**

预冻结期的两个验收 cell 已移出至
`results/model_v0/v3_validation_pre_freeze/`，**不得混入确认性结果**。
completeness audit 将要求全部 324 个 `metadata.json` 的 `git_commit`
等于本 freeze tag 指向的 commit。

## 10. 裁决顺序（冻结，不得更改）

```
G1 QUALITY      A3 vs A0，dataset-stratified category bootstrap，
                mean>0 且 95% CI 不跨 0，且 Δ_MVTec>0 且 Δ_VisA>0
G2 EFFICIENCY   token ≤ A2 的 50%（按构造成立）；wall-clock ≤70% 须按 §6.2
                单独实测（单 GPU、单进程、warmup、5 次 median），
                绝不用并发主实验的日志耗时
G3 SELECTIVITY  A3 vs A4（同预算随机，10 seed 均值）—— 论文核心主张
G4 NO-HARM      mean Δimg_AUROC ≥ -0.3
--------------------------------------------------
GO = G1 ∧ G2 ∧ G3 ∧ G4
G5 ORACLE       A3 vs A5，仅 diagnostic，不是 GO 必要条件
```
