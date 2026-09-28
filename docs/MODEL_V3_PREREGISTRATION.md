# Model V3 预注册：Coarse-to-Fine Selective Resolution Refinement

> 冻结时间：在生成任何 V3 结果之前。
> 本文档写定后不得修改；若需变动，另开新文档并说明理由。
>
> **状态：待签字（v3）。签字前不跑任何 V3 实验。**
>
> **修订记录**
> - v1 → v2：按 7 点意见修订 §2.2 / §2.3 / §2.4 / §4。方向（hot + low-gap
>   trigger、A4 同预算随机对照）保留。
> - v2 → v3：**§2.2.2 由"全部正常图 LOO"改为 support-only jackknife**——
>   v2 的写法在 k-shot 协议下泄漏（1-shot bank + many-shot normal calibration）。
>   §4.1 改为 dataset-stratified bootstrap 并加两数据集方向守卫；
>   §6.2 写死 wall-clock benchmark protocol；新增 §8 第三数据集锁定式外部验证。
>   其余（3×3 window、q=1/3、0.99、low-gap 1/3、A4 十个 seed、HARD6、
>   G1/G3 分离、G4 −0.3、G5 diagnostic、10k bootstrap）按建议**不动**。

---

## 0. V3 要证的不是什么

M1-confirm 已确立（全 27 类，四条冻结判据全过）：提高同一 DINOv2 的空间采样
分辨率，是第一个同时稳定提高 feature separability 与真实 anomaly localization
的干预。

因此 **"高分辨率有用"不是 V3 的贡献**。M1 另给出四条约束设计的结果：

| M1 发现 | 对 V3 的约束 |
|---|---|
| 驱动收益的是**可分性**，不是尺度（`gt_patch_n` p=0.97，`log_area` p=0.17；控制面积后 `C_feat` 仍 t=−3.77） | 触发判据必须是**可分性代理**，不能是缺陷大小 |
| 机制在 `carpet +0.563` / `bottle +0.341` / `tile +0.325` 上**反向**；`screw` AUPRO 在 672 反降、img_AUROC 单调恶化 | 全局提高分辨率**会伤害一部分对象**——"选择性"本身是要验证的东西 |
| `C_feat` / `sep_auc` 用 GT mask，是 **oracle 诊断量** | 触发判据必须**无标签** |
| 448→560 拿到 448→672 总增益的 84%（+4.34 / +5.16） | 560 是**饱和点**，但**未**被证明是"最低有效剂量"（560→672 仍有 +0.82） |

## 1. 主张

> 用**无标签的可分性代理**，只对**低可分区域**做高分辨率细化，
> 能以显著低于全局高分辨率的代价拿到其定位质量，
> **并且不复制全局高分辨率对 `carpet` / `screw` 那类对象的伤害**。

三个**分别验证、不可互相替代**的卖点：**质量**、**效率**、**选择性**。

---

## 2. 方法

### 2.1 记号

- `M0` : 448 全局图（现有 `agg_all3` 配方：三层各自 z-score 后平均），patch 网格 `G0`
- `bank_k` : k-shot 正常 bank（**只用正常图**，`draw_images` 同现有约定）
- `d(·)` : 到 bank 的 1-NN 余弦距离

### 2.2 Stage 2 — 无标签触发（**全部用 normal-reference 校准，不用测试图分位数**）

#### 2.2.1 为什么不使用测试图内分位数

在每张测试图内取"分数最高的 10%"会让**一张完全正常的图也必然产生 top 10%**，
于是模型几乎总会被迫细化一部分区域，"按需计算"的语义被削弱，且触发量
随测试集构成漂移。因此所有阈值一律从**正常参考分布**读出。

#### 2.2.2 校准集：support-only jackknife（冻结）

**协议约束（本节的全部要点）**：在 shot `k` 下，V3 使用的正常数据必须**恰好是
这个 split 抽中的那 `k` 张**。若阈值用该对象 `train/good` 的**全部**图像估计，
那么即使 detector 的 bank 只有 1 张，方法实际利用的仍是
`1-shot bank + many-shot normal calibration`——"1-shot" 这个协议标签就站不住，
审稿人会直接质疑。

因此把 v1 的"全部正常图 LOO"换成 **support-only**：

对每个 `(object, k, split)`，记该 split 抽中的 `k` 张正常图为 `S_k`，
bank `B_k = S_k`（与测试时**同一个 bank**）。对 `S_k` 中每张图 `i` 的每个 patch `p`：

```
d_cal(p) = min_{q in B_k,  q NOT in N_3x3(p)} d(p, q)
```

`N_3x3(p)` = **同一张图 `i` 内**、与 `p` 的网格位置切比雪夫距离 ≤ 1 的 9 个 patch。
即**禁止匹配 `p` 自身及其 3×3 空间邻域**（相邻 patch 高度相关，允许匹配会
人为把距离压到接近 0），但**允许匹配同一张图中任何非局部 patch**，也允许匹配
其他 support 图的所有 patch。

`k = 1` 时 bank 只剩这一张图，但非局部 patch 仍然可匹配，**不会出现空 bank**。

边界情形（理论防护）：若网格小于 5×5 导致 `N_3x3(p)` 把 `B_k` 抽空，
退化为只排除 `p` 自身并断言打印。本项目的网格均 ≥32×32（448 下），实际不会触发。

每个 support 图的 `d_cal` 图按 §2.2.3 的同一套 3×3 窗口规则算出 `hot(W)` / `gap(W)`，
`S_k` 全部图的窗口汇总成该 `(object, k, split)` 的**两个参考分布**。

**为何按 `(object, k, split)` 分别校准**：bank 越大距离越小，分数尺度随 `k` 变化；
不同 split 抽到的图不同，阈值也应随之不同。测试时使用与该校准**同一个 bank**。

#### 2.2.3 窗口统计（冻结，不再调）

设 `S` 为**当前图的分数图**：测试时 `S = M0`（448 全局图），
校准时 `S = d_cal`（§2.2.2 的 pseudo-normal 图）。**窗口规则对两者完全相同**。

```
窗口      3 x 3 patch，stride 1
边界      只取完全落在网格内的窗口；部分越界的窗口直接丢弃（不 padding）
cand      W 内 S 分数最高的 3 个 patch（= 9 个中的 1/3）
ctx       W \ cand
gap(W)    = median(S[cand]) - median(S[ctx])          可分性代理
hot(W)    = max(S[W])
```

`gap` 是 `C_feat` 的**无标签镜像**：`C_feat` 用 GT 划分缺陷/邻近正常；
`gap` 用 `M0` 的**排序**在同一窗口内做同样的划分，不假设知道哪些是缺陷。

#### 2.2.4 阈值（冻结）

"正常参考窗口" **严格指 §2.2.2 的 support-only `d_cal` 图上的窗口**——
即只用该 `(object, k, split)` 抽中的那 `k` 张图，不用任何其它正常图。

```
tau_hot = {hot(W) : 全部 d_cal 窗口} 的 0.99 分位数
tau_gap = {gap(W) : d_cal 窗口中 hot(W) >= tau_hot 的那部分} 的 1/3 分位数
```

**触发条件（冻结）**：

```
W 被选中  <=>  hot(W) >= tau_hot  且  gap(W) <= tau_gap
```

在 **calibration distribution 上**的名义触发比例约为 `1% × 1/3 ≈ 0.33%` 的窗口。

这只是**名义**比例，不是对测试图的期望值：窗口之间高度相关，且 `d_cal` 的
邻域排除规则使校准分布与真实测试图的正常分数分布**并不完全同分布**。
**真实 test trigger rate 未必是 0.33%**，它作为**结果**逐对象报告，不作调参依据。

> **冻结声明**：`q=1/3`、`0.99`、`1/3`、窗口 3×3、stride 1、越界丢弃、
> **一旦冻结，看任何 V3 结果之后不得再动**。触发率是**结果**而不是**调参目标**；
> 若实际触发率过低导致 G1 失败，那是方法的结论，不是改阈值理由。

### 2.3 Stage 3 — 高分辨率细化（与 M1 已验证的干预一致）

**细化目标 = 672-equivalent token density**，不是 560。

理由：M1 的正式 winner 是 672（四条冻結判据全过）。剂量实验证明 560 是
**饱和点**，但**没有**证明它是"最低有效剂量"（560→672 仍有 +0.82）。
若 V3 失败，使用 560 将无法区分是**选择性策略失败**还是**细化分辨率本身不够**。
这个混淆不值得为省算力去冒。

具体约束（冻结）：

1. **尺度匹配**：crop 按 `r = 672 / min(H, W)` 的**同一线性因子**缩放，
   使每个 token 覆盖的**地面面积**与全局 672 的 token 相同。
2. **bank 同分辨率**：细化阶段的 bank 必须是**同一批 k-shot 正常图在 672 下的特征**
   （复用已缓存的 `cache_m1_dino672`，同三层、同 `agg_all3` 配方）。
   **禁止**用 672 crop 特征去匹配 448 bank——那会引入新的 resolution-domain mismatch。
3. **实现前提（跑之前必须验证）**：DINO 预处理**不得**对已缩放的 crop 二次 resize，
   否则尺度匹配被破坏。需在实现阶段确认如何绕过全局 resize，并写一个断言检查
   "crop token 的地面面积 == 全局 672 token 的地面面积"。

### 2.4 Stage 4 — 写回规则（冻结）

```
1. crop 得到的 672 密度 token 网格
2. 每个 token 按中心落入哪个 448 patch 归属；对每个 448 patch 取其覆盖 token 的 mean
   -> 得到该 patch 的 refined 值
3. 被 >=1 个选中 crop 覆盖的 patch：用 refined 值 REPLACE 原 448 值
   被 >1 个 crop 覆盖的 patch：取这些 crop refined 值的 MEAN
4. 未被覆盖的 patch：保留原 448 值
```

**分数尺度校准（normal-only，冻结）**：448 与 672 的分数尺度存在系统性偏移，
直接替换会产生接缝。校准用**仿射映射** `a·x + b`，使 refined 图在
**正常参考 patch**（2.2.2 的 LOO 正常图，同时算 448 与 672 两版）上的
**中位数与四分位距**分别匹配 448 图。

**校准只用正常数据**：不看任何 test anomaly，不用 GT，不在测试时调参。
校准参数按 (object, shot) 冻结并缓存。

**不做 feather blending**（冻结）。替换发生在 **patch 网格层**，
之后与全程一致地套用标准 `dists2map`（双线性 + `gaussian_filter(sigma=4)`），
该平滑本身即提供接缝过渡。本阶段不引入额外的边界超参。

---

## 3. 对照臂（缺一不可）

| 臂 | 说明 | 隔离出的东西 |
|---|---|---|
| **A0** | global-448 | 现有 baseline |
| **A1** | global-560 | 中间剂量 |
| **A2** | global-672 | 上界质量 / 高代价参照 |
| **A3** | V3（selective） | — |
| **A4** | **same-budget random-select** | **关键对照** |
| **A5** | oracle-select（GT 选窗） | **diagnostic only** |
| **A6** | **与 A3 完全相同的 selector / crop / 写回 / 校准，仅把细化密度由 672-equivalent 换成 560-equivalent** | 描述性：效率/质量权衡（**非判据**，不参与 GO/STOP） |

**A4 必须是对每张测试图严格等预算的反事实**（冻结）：

- 与 A3 使用**完全相同的** refined patch/window 数量、crop 大小、分辨率、bank、
  写回规则、校准参数
- **唯一区别**是"选哪里"：在满足 `hot(W) >= tau_hot` 的窗口中**均匀随机**抽取
  与 A3 数量相同的窗口
- 使用 **10 个预先固定的随机种子**（seed = 按对象名 + `0..9` 派生，不用 `hash()`），
  A4 的分数取 10 次的**均值**

若不使用 10 个种子而只随机一次，A3−A4 很容易被单次抽样噪声左右。

**A5 同样使用完全相同的预算**，只允许 GT 决定选区；它是
**diagnostic upper bound，不参与模型选择，也不是 GO 的必要条件**。

---

## 4. 冻结判据（GO / STOP 层级）

在 **MVTec 15 + VisA 12 全量**、shots {1,2,4,8} × splits 3 上评。
**所有类别全部报告**；`HARD6 = {transistor, screw, zipper, pcb4, pcb2, pcb3}`
（M1 已冻结、由 **baseline `C_feat`** 选出，与 Δ 无关）作为
**预注册 subgroup 单独强调**，但**不独自承担泛化结论**。

### 4.1 统计单元与重采样（冻结）

**统计单元 = category/object（27 个），不是 (object, shot, split) cell。**
F1 已证明按 cell 计数会把 t 抬高约 9 倍（+44 → +4.8）——几千张图带来的
"巨大 t 值"而真正独立的实验单位只有十几个类别，是伪精度。

对每个类别 `c` 定义配对差：

```
Delta_c = AUPRO_c(A3) - AUPRO_c(A4)        (G3 用)
Delta_c = AUPRO_c(A3) - AUPRO_c(A0)        (G1 用)
每个 AUPRO_c = 该类别在所有 (shot, split) cell 上的均值
```

**主检验（冻结）：dataset-stratified category bootstrap。**
27 个类别来自两个数据集（MVTec 15 + VisA 12）。**不得**把它们当作同分布的
27 个样本混合重抽——否则某一个数据集的极端类别可以单独把 pooled 结果扛起来。

每次重采样：**在 MVTec 的 15 类中有放回抽 15 个，在 VisA 的 12 类中有放回抽 12 个**，
合并后计算总体 `mean Delta`。共 **10,000 次**，取 **percentile 95% CI**。

判定：**总体 mean Delta > 0 且 CI 不跨 0**。

**方向守卫（冻结）**：同时强制报告 `Delta_MVTec` 与 `Delta_VisA` 各自的 mean，
且 **G1 与 G3 都要求两者均 > 0**。不要求两个数据集各自的 CI 显著——
那会因单数据集样本量不足而非必要地收紧；只要求**方向一致**，
这已足以阻止"一个数据集扛起全部结论"。

`HARD6` 作为 subgroup 单独报告 **effect size 与方向一致率**（6 个里几个为正），
并与全体 27 类的方向做比较。

**不因 G1/G3 各有统计检验而做 Bonferroni 之类校正**：本预注册是
**intersection gate**（四条必须全部通过），不是"挑一个显著即宣布成功"，
多重比较的逻辑在这里不适用。

### 4.2 五个 Gate

```
[GO 必要条件 —— 四条全部成立才 GO]

G1 QUALITY       A3 相对 A0（448 coarse）有真实提升
                 Delta_c = AUPRO_c(A3) - AUPRO_c(A0)
                 判定：dataset-stratified bootstrap mean > 0 且 95% CI 不跨 0
                       且 Delta_MVTec > 0 且 Delta_VisA > 0（方向守卫）

G2 EFFICIENCY    A3 的高分辨率 token/patch 计算量 <= A2(full-672) 的 50%
                 且实测端到端 wall-clock <= A2 的 70%（测法见 §6，冻结）
                 并报告 recovery-per-compute（见 4.3）

G3 SELECTIVITY   A3 > A4（同预算随机，10 seed 均值）        <-- 论文主张
                 Delta_c = AUPRO_c(A3) - AUPRO_c(A4)
                 判定：dataset-stratified bootstrap mean > 0 且 95% CI 不跨 0
                       且 Delta_MVTec > 0 且 Delta_VisA > 0（方向守卫）
                 HARD6 subgroup：报告 effect size 与方向一致率

G4 NO-HARM       mean d_img_AUROC(A3 - A0) >= -0.3
                 并逐对象报告，退化最大的几类必须点名

[DIAGNOSTIC —— 只作答，不作 GO 条件]

G5 ORACLE HEADROOM   A3 vs A5：selector 还有多少空间
                     **不参与模型选择，不是 GO 必要条件**
```

**G1 与 G3 不可互相替代、不可合并**：

- `A3 > A0` 证明**方法有用**；
- `A3 > A4` 证明**你的选择机制有用**。

二者缺一不可。只有 `A3 > A0` 时，方法退化为"多加高分辨率像素"；
只有 `A3 > A4` 时，方法可能整体弱于 baseline。

### 4.3 recovery-per-compute（冻结定义）

```
recovery(X) = (AUPRO(AX) - AUPRO(A0)) / (AUPRO(A2) - AUPRO(A0))
compute(X)  = AX 额外的高分辨率 token 数 / A2 的高分辨率 token 数
report      = recovery(X) / compute(X)，X in {A1, A3, A4, A6}
```

即"每单位额外算力换回多少全局 672 的增益"。
**必须同时报告分子与分母**，不得只报比值。

---

## 5. 纪律（沿用全程，不得放宽）

1. **禁止任何目标类缺陷进入 bank。** bank 只由正常图构成；触发阈值只由正常参考校准。
2. **绝不使用目标异常标签做超参调优。** §2.2.4 / §2.3 / §2.4 的全部常数
   在看任何 V3 结果之前冻结，之后不再修改。
3. **`C_feat` / `sep_auc` / A5 是 oracle 诊断量，不得进入方法。**
4. **预注册判据不得事后修改。** 若发现判据本身有误，完整披露并降级为假设生成，
   不得静默修正（F1 的符号错误是前例）。
5. **不训练任何东西。** 所有表示冻结。
6. **必须并列报告负面对象**（A3 不如 A0/A1/A4 的类别），不得只报均值。
7. **不得把 HARD6 当作唯一主表。** 27 类全部报告；HARD6 单独强调。
   即使 V3 只在低 separability 类别有效，也应诚实写成 **targeted method**，
   而不是误写成 universal improvement。

---

## 6. 效率的计法（冻结，必须可复核）

### 6.1 硬计算量（不依赖运行环境）

- A3 实际前向的**高分辨率 token/patch 数** / A2 的对应值
- A0 的 448 patch 数单独报告，不混入分子分母

**G2 的 `token compute <= 50%` 用这一项判定**，因为它不受机器噪声影响。

### 6.2 wall-clock benchmark protocol（冻结）

`wall-clock <= 70%` 极易被运行环境噪声左右，因此测法写死：

```
同一 GPU、同一进程内测量，不跨机器、不跨会话
batch size 固定，并记录在结果中
先跑 >= 3 次 warmup，丢弃
正式重复 5 次，取 median（不取 mean，避免长尾）
计时段 = 端到端：
    feature extraction + selector + high-res crop + kNN scoring + writeback
排除：首次模型加载、首次磁盘读 cache、以及一切一次性校准
      （校准按 (object, shot, split) 缓存，不计入推理计时）
A0..A6 在**同一次会话内**背靠背测量，避免跨会话漂移
```

**若 wall-clock 比值远差于 patch 比，必须如实说明**——crop/resize/写回的开销
是真实的，**不得**只报理论 patch 比而隐去。

## 7. 出口（跑之前定好）

```
G1 ^ G2 ^ G3 ^ G4 全部成立
  -> 方法成立：写成 "low-separability bottleneck + selective, efficient
     restoration of spatial detail"；A6 作为效率/质量权衡的描述性附录
G3 不成立（A3 ~ A4）
  -> 选择性无效，退化回 "高分辨率有用"，STOP，不作为方法论文
G1 不成立（细化后质量没提升）
  -> 先查 §2.3 的实现前提（尺度匹配断言）与 §2.4 的校准，不得改判据；
     确认实现无误后仍不成立 -> STOP
G4 不成立（复制了伤害）
  -> 选择信号抓错目标，回到 §2.2 重新设计触发量，且**另开新文档**
G5 低（A3 接近 A5）
  -> selector 已接近 oracle 上限，这是正面信息，不改变 GO/STOP
```

## 8. 第三数据集：锁定式外部验证（现在写死，暂不运行）

**背景**：MVTec 与 VisA 已经深度参与问题发现（F1 / F2）、干预验证（M1）
与 V3 设计。因此即使 V3 在这 27 类上全部 GO，它属于很强的
**development / validation evidence**，但严格讲**不是**完全独立的
external confirmation。

**现在冻结（在看任何 V3 结果之前）：**

> **锁定第三数据集 = MPDD**（Magnetic Tile Defect Detection）。
> 作为 **one-time final evaluation** 保留。
> V3 冻结之后**只跑一次**；结果**不得**用于修改 V3 的任何设计、阈值、
> 超参或判据。该数据集**不参与** V3 的模型选择、超参冻结或任何中间决策。
>
> **不得更换。** 若 MPDD 上失败，如实报告为 external validation failure，
> 不得改跑 Real-IAD 或任何其它数据集来替换结论。

**为何锁 MPDD 而不是 Real-IAD**：本机 32 GB 内存，本轮会话中已多次触发
系统 OOM killer（重任务并发上限实测为 2–3）。Real-IAD 的规模（30 类、
高分辨率）在现有资源下无法可靠跑完。**一个能真正跑完的外部验证集，
胜过一个跑不完的更强选择**——后者只会变成又一个"无法交付"的承诺。

**现在不跑。** 本节只作冻结声明。

论文叙事因此是：

```
MVTec / VisA :  问题发现 + 方法开发      (development / validation)
第三数据集    :  locked external validation  (一次性)
```

**若第三数据集上失败**：如实报告为 external validation failure，
**不得**回头修改 V3，也**不得**换一个数据集重来。

## 9. 已知局限（事先声明）

1. crop 的 672 特征与全局 448 特征**尺度不同**，写回依赖 §2.4 的
   normal-only 仿射校准；校准误差的影响本阶段不量化，留给后续。
2. `gap` 是 `C_feat` 的**代理**不是同一个量：M1 已证两者相关但不等价
   （控制面积后 `C_feat` 仍显著，`log_area` 不显著）。
3. A5 说明的是"给定 GT 最多能选多好"，**不代表可实现上限**。
4. 触发率由正常参考分布决定，**不做跨对象平衡**。若某些对象触发率极低，
   该对象上 A3 ≈ A0，这会被如实报告为"该方法对该对象不适用"。
5. 本阶段不涉及任何 backbone 替换；`ConvNeXt / SAM / RADIO / CLIP` 仍未测。

## 10. 实现前必须完成的检查（跑之前）

1. §2.3 的**尺度匹配断言**：crop token 的地面面积 == 全局 672 token 的地面面积。
2. A4 **等预算核验**：**逐测试图**断言 `|selected(A4_seed_i)| == |selected(A3)|`，
   `i = 0..9`（A3 在某图触发 0 个则 A4 也为 0，触发 17 个则 A4 也随机取 17 个）；
   并断言 crop 大小、分辨率、bank 完全一致。
3. §2.4 **校准只用正常数据**的代码级断言：校准路径不得读取任何 test anomaly
   或 GT（A5 除外，A5 独立于方法路径）。
4. §2.2.2 **support-only 断言**：校准路径只能触及 `S_k` 中的 `k` 张图，
   代码级断言其读不到该对象 `train/good` 的其它图。
5. 触发率报告：逐对象、逐 shot 的 `tau_hot` / `tau_gap` / 实际触发窗口占比
   （作为**结果**报告，不作调参依据）。
