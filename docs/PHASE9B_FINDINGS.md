# Gate 9B-v0 结果总账（冻结）：Target-Conditioned Defect Feature Generator

> 冻结于 tag `gate9b_generator_closed`。
> **判决：生成器路线在 AD2 上判失败 —— 且这次不是 range 不足。**
> v1 的失败先被判定为测试无效（headroom +0.49），重跑 AD2 后依然失败。

---

## 最终结论（严谨表述）

> **在 AD2 的全部 8 个类别上，真实目标缺陷 support 相对 normal-only baseline 平均值
> +18.4 AUROC（合计 +147.4）；而条件生成器 G3 净获得 −1.0（合计 −8.3）。**
> **生成器既没有超过「只用 source context」的 G1（−1.10, t=−1.77），
> 也没有超过「完全不接触外部异常」的 G2（−0.82, 3/8）。**
> **跨类别异常知识中，没有可学的条件变换结构可以仅凭 target normal 推断出
> target-specific defect support。**

---

## 1. 为什么跑了两次

预注册规则：「G3 连 source-LOCO 都过不了 → 不跑 AD2」。

v1 上判据确实失败，但事后证据表明 **v1 的检验本身无效**：

| 指标 | v1 | AD2 |
|---|---|---|
| oracle headroom 中位数 | **+0.49** | +18.4 |
| headroom > 2 的类数 | **5/15** | **8/8** |
| G3 超过 oracle 的格子 | **32/135** | 0/288 |

同一份 oracle 构造代码原样搬到 AD2 立刻给出 +35.2 / +23.7 / +15.6 / +7.9。
即 v1 上连「用真实目标缺陷」都只值 ~+1 AUROC，oracle 在那里根本不是上界。

**判定 v1 测试无效，理由记录在案，在 AD2 上重跑同一套协议。** 下面是 AD2 的结果。

## 2. AD2 LOCO 结果（leave-one-object-out，8 类）

per held-out object（7 sources × 3 shots × 3 reps 平均）：

| heldout | baseline | G0 identity | G1 bank_only | G2 target_only | **G3 interaction** | oracle |
|---|---|---|---|---|---|---|
| can | 47.4 | 48.5 | 48.1 | 47.3 | 47.8 | 50.8 |
| fabric | 55.0 | 62.2 | 63.1 | 61.2 | **64.8** | 87.1 |
| fruit_jelly | 81.2 | 72.6 | 79.2 | 77.5 | 76.3 | 93.9 |
| rice | 82.3 | 71.2 | 75.8 | 76.8 | 73.0 | 97.8 |
| sheet_metal | 82.7 | 78.7 | 84.7 | 84.5 | **85.0** | 88.9 |
| vial | 75.9 | 64.2 | 74.7 | 75.4 | 74.8 | 97.8 |
| wallplugs | 39.7 | 42.4 | 40.8 | 39.1 | 38.0 | 73.2 |
| walnuts | 76.0 | 70.0 | 74.3 | 76.6 | 72.3 | 98.2 |

## 3. 预注册判据

| 判据 | 结果 |
|---|---|
| **G3 > G1 bank_only** | ❌ **FAIL**（−1.10, t=−1.77, 3/8） |
| **G3 > baseline 多数类** | ❌ **FAIL**（−1.04, t=−0.52, 3/8） |
| **closure > 0 于 ≥2/3 eligible** | ❌ **FAIL**（−5.8%, 3/8；eligible = 8/8） |
| G3 > raw external G0 | ✅ PASS（+2.75, t=1.75, 6/8） |

### 核心数字：可获得的 vs 实际拿到的

| | 合计 |
|---|---|
| oracle 相对 baseline 的可获得 headroom | **+147.4 AUROC** |
| G3 实际获得 | **−8.3** |

逐类：

| heldout | headroom | G3 获得 | 捕获率 |
|---|---|---|---|
| fabric | +32.1 | +9.8 | 31% |
| wallplugs | +33.5 | −1.7 | −5% |
| walnuts | +22.2 | −3.7 | −17% |
| vial | +21.8 | −1.1 | −5% |
| rice | +15.5 | −9.2 | −60% |
| fruit_jelly | +12.7 | −5.0 | −39% |
| sheet_metal | +6.2 | +2.2 | 36% |
| can | +3.4 | +0.4 | 12% |

**fabric 与 sheet_metal 上有正捕获（31% / 36%），但剩下的类净为负，合计为负。**

## 4. 反复出现的模式：G2 又赢了

| 对比 | mean | t | 更好类数 |
|---|---|---|---|
| G3 − G1 | −1.10 | −1.77 | 3/8 |
| **G3 − G2** | **−0.82** | −0.92 | **3/8** |
| G3 − baseline | −1.04 | −0.52 | 3/8 |
| G1 − G0 | **+3.85** | **2.71** | 6/8 |

- **G2（纯噪声 + target normal，完全不接触外部异常）再次不劣于 G3。**
- G1 − G0 = +3.85 (t=2.71)：**raw external 有害，这一点稳定成立**；
  任何训练过的变换都能把这份伤害抵消掉。
- 但**抵消伤害不需要外部异常内容** —— G2 就能做到。

closure：`G1 +1.2% | G2 −2.5% | G3 −5.8% | G0 −27.8%`。全部 ≈ 0 或负。

**这与 Gate 9A 的结论完全一致**：能学的只有 donor identity 那一层，
target conditioning 那一层学不到。

## 5. 判决

```
G3 < G1        预注册判据失败
G3 ≈ G2        外部异常内容零贡献
closure −5.8%  在 +147.4 的 headroom 上净获得为负
```

**Gate 9B 失败。不进入 Gate 9B-v1。**

### 限制（必须一起声明）

- **8 个类别**是硬上限。paired t 在错误方向且 |t| < 2，
  严格说「未证明有效」而非「证明无效」；但 closure 为负 + G2 不劣于 G3 是同一个方向的两条独立证据。
- 生成器是**小型 MLP，300 步**。更强的生成器（diffusion / flow / 大规模预训练先验）
  未在本实验中排除；结论只覆盖「小型条件 MLP 从 ~10^2 量级的类别级关系里学习」这一设定。
- 结论**只针对 feature-space defect support 生成**，不排除 VLM / 大模型外部先验的路线。

## 6. 资产

| 类型 | 内容 |
|---|---|
| 脚本 | `gate9b_generator.py`（v1 + ad2 两模式）、`gate9b_verdict.py` |
| 结果 | `metrics/gate9b_eval.csv`（v1, 540 行）、`metrics/gate9b_ad2_eval.csv`（288 行） |
| 权重 | `results/model_v0/checkpoints_g9b/{v1,ad2}_<fold>_<variant>.pt` |

## 7. 本次踩的坑（都是「看起来在工作、实际没有」型）

1. **`relu` 铰链在容易类上第 0 步就饱和**（loss 0.0014），梯度恒为 0，
   生成器完全没学到东西却显示在训练。改 `softplus(-d/beta)*beta` 后才有信号。
2. **oracle bank 忘了过 `gt_frac` 阈值**，把缺陷图的正常区域也收进 bank，
   导致 13/15 类的 oracle 低于 baseline。修正后恢复正常。
3. **`gate7b3r.make_dist` 结尾是 `.cpu().numpy()`，静默切断计算图。**
   需要反传必须用本文件内的 `make_dist_t`。
4. **评估 CSV 每行只带自己 variant 的列**，配对比较前必须先 pivot，否则全 NaN。
5. `for a, b in ...` 覆盖了 argparse 的 `args` 变量名。

---

## 继承到下一阶段的强制方法纪律

1. **测试台的 headroom 必须先验证，再解读结果。** 本次靠「把 oracle 构造搬到 AD2」
   这一步才发现 v1 的检验无效。**任何 AUROC 类实验，先报告 oracle 的 headroom。**
2. **必须包含「不能条件化的退化版」与「不用外部数据的退化版」**
   （G1 / G2）。否则会把「抵消伤害」误读成「学到条件变换」——本次 G2 两次拆穿了它。
3. **报告可获得总量与实际获得量**（+147.4 vs −8.3），而不是只看平均 AUROC 的微小差异。
4. **预注册标准不得事后修改**；但如果**检验前提本身失效**，应显式记录理由后重跑，
   而不是拿无效检验的结果当结论。
