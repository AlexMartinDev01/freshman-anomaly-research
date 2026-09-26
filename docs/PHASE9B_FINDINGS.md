# Gate 9B-v0 结果总账：Target-Conditioned Defect Feature Generator

> 状态：**预注册判据未通过；但测试台 range 不足，该负结果不具诊断力。**
> 结论待定（见 §5）。**尚未运行 AD2 主实验。**

---

## 1. 实验设置（按预注册执行）

```text
z_hat = normalize( z_ext + G(z_ext, h_source_normal, h_target_normal, eps) )
```

- Set encoder：mean + std pooling + 2 层 MLP（刻意不用 attention）。
- Generator：`Linear(d_in, 512) → ReLU → Linear(512, 384)`，residual 形式。
- Loss：`L_rank`（softplus 版，beta=0.05）+ `λ·L_SWD`（λ=0.1）+ `γ·L_resid`（γ=1e-3）。
- Episodic LOCO：MVTec AD v1 15 类留一，每 episode 随机 1/2/4-shot target normals，
  source 取另一类，query 用 target 的 held-out normal + defect。
- 300 steps × 3 episodes，AdamW lr=1e-3。
- **AD2 全程未参与训练，也未参与评估。**

### 一个必须记录的实现修正

`rank_loss` 最初用 `relu(margin - d)`。冒烟测试显示 bottle 在第 0 步 loss 就是 0.0014
——**铰链已经饱和，梯度恒为 0，生成器拿不到任何学习信号**。改为
`softplus(-d/beta)*beta`（beta=0.05，匹配 fused score 的量级差 ~0.1），
这也是本项目 `soft_separation_loss` 用过的同一手法。

另修正：评估里的 oracle bank 最初直接取 defect image 的**全部 patch**，
而缺陷图大部分区域是**正常**区域，等于造了一个「几乎是正常 bank」的东西。
必须先过 `gt_frac > 0.10`。修正前 13/15 类的 oracle **低于** baseline。

## 2. v1 LOCO 结果

per held-out category（14 sources × 3 shots × 3 reps 平均）：

| heldout | baseline | G0 identity | G1 bank_only | G2 target_only | **G3 interaction** | oracle |
|---|---|---|---|---|---|---|
| bottle | 99.3 | 98.6 | 99.7 | 99.8 | 99.7 | 99.8 |
| cable | 91.4 | 80.9 | 91.3 | 93.4 | 91.8 | 93.8 |
| capsule | 88.6 | 87.6 | 93.2 | 91.8 | 93.6 | 92.5 |
| carpet | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| grid | 99.7 | 99.8 | 99.7 | 99.9 | 99.8 | 100.0 |
| hazelnut | 98.4 | 93.9 | 99.7 | 99.6 | 99.8 | 100.0 |
| leather | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| metal_nut | 99.7 | 99.7 | 99.7 | 99.9 | 100.0 | 98.8 |
| pill | 93.5 | 90.8 | 96.0 | 95.8 | 95.1 | 93.0 |
| screw | 71.1 | 74.8 | 76.8 | 78.2 | 76.3 | 75.3 |
| tile | 100.0 | 99.3 | 100.0 | 100.0 | 99.9 | 100.0 |
| toothbrush | 99.3 | 94.3 | 97.7 | 99.1 | 96.7 | 98.0 |
| transistor | 91.4 | 79.4 | 89.2 | 91.8 | 90.1 | 95.5 |
| wood | 97.4 | 93.9 | 99.5 | 99.0 | 99.3 | 99.8 |
| zipper | 97.9 | 89.5 | 98.4 | 99.1 | 98.6 | 98.5 |

## 3. 预注册判据的判决

| 判据 | 结果 |
|---|---|
| **G3 > G1 bank_only** | ❌ **FAIL**（−0.02，t=−0.17，仅 9/15 更好） |
| G3 > raw external G0 | ✅ PASS（+3.87，t=3.80，12/15） |
| G3 > baseline 多数类 | ✅ PASS（+0.86，t=1.63，10/15） |
| closure > 0 于 ≥2/3 eligible | ✅ PASS（+68.5%，8/9） |

**决定性判据 G3 > G1 失败。** 按预注册规则：不跑 AD2，Gate 9B 判失败。

### 最重要的诊断：G3 − G2 = −0.44

| 对比 | mean | t | 更好类数 |
|---|---|---|---|
| G3 − G1 | −0.02 | −0.17 | 9/15 |
| G3 − G0 | +3.87 | 3.80 | 12/15 |
| **G3 − G2** | **−0.44** | **−1.61** | **4/15** |
| G3 − baseline | +0.86 | 1.63 | 10/15 |

**G2 完全不接触任何外部异常**（纯噪声 + target normal），却比完整的 G3 还好。

按 closure 排序同样如此：

```
G2 target_only   closure +93.3%   (9/9 positive)   ← 最好，且不用外部数据
G3 interaction   closure +68.5%   (8/9)
G1 bank_only     closure +59.0%   (6/9)
G0 identity      closure -292.5%  (2/9)
```

**可以确定的结论**：raw external 是有害的（G1 − G0 = +3.89，t=4.02），
所有训练过的生成器都能**把这份伤害抵消回 baseline 附近**，但**没有一个能超过 baseline**。
而「抵消伤害」这件事 G2 就能做到——**外部异常内容没有贡献任何东西。**

## 4. 但这个测试台 range 不足 —— 负结果不具诊断力

同一份 oracle 构造代码原样搬到 AD2：

| object | shot | baseline | oracle | headroom |
|---|---|---|---|---|
| wallplugs | 4 | 45.1 | 80.3 | **+35.2** |
| vial | 4 | 75.1 | 98.8 | **+23.7** |
| can | 4 | 33.1 | 48.7 | **+15.6** |
| sheet_metal | 4 | 85.2 | 93.1 | **+7.9** |

而在 v1 上：

| 指标 | v1 | AD2 |
|---|---|---|
| oracle headroom 中位数 | **+0.49** | +7.9 ~ +35.2 |
| headroom > 2 的类数 | **5/15** | 4/4 |
| G3 超过 oracle 的格子 | **32/135** | — |

**在 v1 上，连「用真实目标缺陷」都只能换来 ~+1 AUROC，且生成器在 135 个格子里有 32 个
直接超过了 oracle——oracle 在这里根本不是上界。**

原因是 v1 的 few-shot baseline 已经在 71–100 的天花板区，
而 AD2 的 baseline 是 33–85，正是本项目全部证据（Phase 8 的 +4~+35）所在的地方。

> 这正对应本项目继承的方法纪律第 2 条：
> **「不能从 range 不足的 sweep 下结论。」**
> 所以 Gate 9B-v0 的失败**不能**被写成
> 「跨类别异常知识中不存在可学的条件变换结构」。

## 5. 待决问题

预注册规则写的是「G3 连 source-LOCO 都过不了 → 不跑 AD2」。
但该规则的前提是**source-LOCO 是一次有效检验**，而事后证据表明它 range 不足
（连 oracle 都只有 +0.49）。

**因此需要决定：**

1. **把 v1 的结果判定为「测试无效」并在 AD2 上重跑同一套 Gate 9B-v0**
   （LOCO over AD2 的 8 个对象；headroom +8~+35；约 40 分钟）。
   代价：偏离预注册流程，需明确记录理由。
2. **接受判据失败，就此收束生成器路线**，把「外部异常知识无法转化为
   target-specific defect support」作为强负结果写入（并附上 range 不足的限定）。
3. 其他。

## 6. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `gate9b_generator.py`、`gate9b_verdict.py` |
| 结果 | `metrics/gate9b_eval.csv`（540 行）、`metrics/gate9b_<fold>_<variant>.csv` |
| 权重 | `results/model_v0/checkpoints_g9b/`（15 folds × 3 trained variants） |

## 7. 本次踩的坑

1. **`relu` 铰链在容易类上第 0 步就饱和**，梯度恒 0，生成器完全没学到东西。
   换 softplus 后才有信号。**这是「看起来在训练、实际没梯度」的典型陷阱。**
2. **oracle bank 忘了过 `gt_frac` 阈值**，把缺陷图的正常区域也收了进去，
   导致 oracle 低于 baseline。修正后 12/15 类恢复为 oracle ≥ baseline。
3. `make_dist` 结尾是 `.cpu().numpy()`，**会静默切断计算图**。
   需要反传时必须用 `make_dist_t`（本文件内实现）。
4. 评估 CSV 每行只带自己 variant 的列，做配对比较前必须先 pivot，
   否则得到全 NaN。
