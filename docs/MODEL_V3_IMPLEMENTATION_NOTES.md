# Model V3 实现说明（预注册冻结之后）

> `docs/MODEL_V3_PREREGISTRATION.md` 已冻结并打 tag `model_v3_preregistered`
> （commit `742126ac32a9ace1218e0c43819a225a5c934049`），该文档**不得修改**。
> 本文档只记录**实现层**的决定与发现。任何影响判据的改动都不属于本文档的范畴。

---

## I1. crop 的取法（实现层，满足 §2.3 的不变式）

§2.3 要求 crop 的 token 与全局 672 的 token **覆盖相同的地面面积**。

`DINOv2Wrapper.transform` 用 `transforms.Resize(size=smaller_edge_size)`，
整数参数意味着**总是把较小边缩放到 672**。若把 crop 直接交给
`prepare_image`，它会被**二次缩放**，地面尺度随之改变——那样 A3 测的
就是另一个干预。

采用的做法：

```
整图 resize 到较小边 672  →  从"已缩放图"中取 token 对齐（14 的倍数）子窗口
                          →  不做任何二次 resize，直接送入 ViT
```

这样窗口的 token 与全局 pass 的 token **在网格位置上逐位对应**，
每个 token 覆盖 `14/r` 个原图像素，`r = 672 / min(H, W)`。

**证据**（`experiments/model_v0/v3_check_scale.py`）：

```
C1  几何       窗口 token 中心映射回原图坐标，与全局 token 差 0.000e+00
C2a 无二次缩放 窗口张量 == 全局张量的对应切片（torch.equal）：True
    负对照     同一窗口走 prepare_image：2.77 原图px/token vs 正确路径 18.75
               比值 0.148  ->  该检查确实能发现重采样 bug
```

## I2. 必须披露：crop token ≠ 全局 672 token（注意力范围）

DINOv2 是**全局注意力** ViT，任何 crop 都限制了注意力范围。
实测：一个 7×9 token 的窗口，其 token 与全局 pass 同位置 token 的
平均 L1 差为 **2.11，而特征本身的平均幅度是 1.88（相对差异 112.5%）**。

结论（必须写进结果披露，且**不是**实现错误）：

> A3 的细化区域处于 672 **地面尺度**，但其特征是**crop 内注意力**下的特征，
> 不是全局 672 的特征。**A3 不可能完全复现 A2**，这正是 A2 作为质量上界
> 参照、以及 G5 用 A5 而非 A2 作为 oracle 上界的原因。

## I3. 响应在 crop 内的空间抹开（窗口尺寸扫描）

对单 token 的异常（原图 19 px 方块 == 一个 672 token 的覆盖面积），
在窗口内的响应**不是局部化的**：

```
win    peak     median   peak/med   argmax      predicted
  7    51.76    32.65      1.59     (2,3)       (3,3)
 13    56.60    33.28      1.70     (5,7)       (6,6)
 21    67.03    33.68      1.99     (8,10)      (10,10)
 33    63.43    28.22      2.25     (13,16)     (16,16)
 48    61.45    22.81      2.69     (25,24)     (25,24)     <- 窗口=全图，精确相符
```

`n = 48` 一行是**恒等式检验**：窗口等于全图时 argmax 必须精确等于全局
argmax，它同时验证了整条计算链。（本检查初次编写时把窗口从**真实照片**
而非 square 图取出，导致差的其实是"照片 vs 空白"；正是这一行把它暴露出来。）

读数：抹开尺度约 **2–3 个 token**，且随窗口增大，峰值/中位数对比度从
1.59 升到 2.69。**crop 越小，响应越不局部。**

## I4. 实现修正记录（implementation corrections）

按预注册的纪律，实现 bug 可修，但必须记录；**判据门槛不得修改**。

| # | 位置 | 问题 | 处理 |
|---|---|---|---|
| 1 | `v3_check_scale.py` | C3 从真实照片取窗口，实际在比较"照片 vs 空白" | 改为从 square / blank 图取窗口；由 `n=48` 的恒等式发现 |

## I5. 待决：crop 的**尺寸**在预注册中未定义

预注册 §2.3 写了 crop"以其为中心"，§2.4 写了"被 >=1 个 crop 覆盖的 patch"
如何写回，并要求 A4/A5 **与 A3 使用相同的 crop 大小**——
但**没有规定 crop 究竟取多大**。

这是一个必须在跑 smoke 之前定死的参数：它同时决定 G1（质量，因为 I3 显示
窗口越大响应越局部）与 G2（效率，代价随 crop 面积平方增长）。

**在项目负责人确认之前，不跑 smoke test，不跑主实验。**

## I8. 更正：transistor 的 A3-vs-A4 反事实比例是 53/53，不是 29/53

**此前报告的 29/53 与 30/53 作废。**

`v3_run.main` 里那段诊断写成：

```python
ndiff = sum(1 for i in range(len(integ))          # 按 refined 的"个数"迭代
            if not np.array_equal(np.sort(arms["a3"][i]["pos"]),
                                  np.sort(arms["a4s0"][i]["pos"])))
```

它按 `len(integ)`（refined 图的个数）迭代，却用该下标去索引**完整的 100 张图**列表。
refined 的 53 张**散布**在 100 张里，所以它实际比的是**前 53 张**——
与真正被细化的那 53 张基本不重合，53 次比较全是错配的图像对。

正确写法（`v3_verify_runner.py` 用的）：

```python
ref_idx = [i for i, x in enumerate(arms["a3"]) if x["m2"] > 0]
```

**更正后：transistor k=1 s0，A3 与 A4(seed 0) 的位置在 `53/53` 张被细化的图上不同。**

该更正**不改变任何 A3/A4 的 map**（只改一处打印语句），因此不推翻模型实现，
也不作废任何已保存 cell。它改变的是 G3 的解释：A4 的反事实空间比此前报告的
更大，而不是只有约六成的图有区别。

## I7. A4 随机 seed 的派生（确定性实现）

补充 1 A1.3 只说"A4 使用 10 个预先固定的随机种子"，**没有规定 seed 如何派生**。
实现冻结如下（属实现层，不新增预注册 addendum——方法与随机对照的定义未变）：

```
seed(dataset, object, shot, split, seed_id, arm) =
    int.from_bytes(SHA256(f"{dataset}|{object}|{shot}|{split}|{seed_id}|{arm}")
                   .digest()[:8], "big")

A4: arm = "A4",  seed_id = 0..9
```

**为何把全部标识写进 key**：

- **`dataset`**：MVTec 与 VisA 未来可能出现同名 category；
- **`shot` / `split`**：最初的实现只用了 `(object, seed_id)`，导致**同一对象在各
  shot/split 上的 A4 随机画完全重复**。每个 cell 的反事实本身仍合法（预算相等
  断言照样通过），但 cell 之间的随机位置被人为耦合，Monte Carlo 覆盖不自然。
  这不是统计单元的问题——统计单元已冻结为 27 个 category，不是 cell——而是
  随机结构的问题，在任何性能结果出现之前修掉。
- **`arm` namespace**：防止未来第二个随机臂与 A4 碰撞。

**不用 `hash()`**（进程间随机化），**不用位置索引**（依赖有多少个对象排在前）。

**验证**（`v3_refine.py --selftest-seed`）：

```
✓ 同一 key 跨进程、跨运行得到完全相同的 seed
✓ dataset / object / shot / split / seed_id 任一改变都改变 seed
✓ namespace 改变 seed
✓ 单 cell 的 10 个 seed 互异
✓ 跨 27x4x3x10 规模的扫描无碰撞
锚点 seed("mvtec","bottle",1,0,3) = 2959745414427513892（两次独立进程一致）
```

## I6. 预运行发现：support-only 校准的触发率**不迁移**（2026-09-28）

`v3_selector.py` 按 §2.2 冻结的设计实现了 support-only 校准，并在
`--splits 1` 上实测触发率。**这不是判据，是 §10.5 要求作为结果报告的触发率。**

```
object      shot  trigger   bad      good     P(hot>=tau_hot) on test
bottle       1    0.036%   0.048%   0.000%        —
screw        1    0.000%   0.000%   0.000%        —
screw        8    0.128%   0.133%   0.114%      0.309%
transistor   1    1.230%   2.900%   0.117%      3.551%
transistor   8    0.973%   2.425%   0.006%      2.703%
zipper       1    0.604%   0.764%   0.010%      3.543%
pcb2         1    0.861%   1.063%   0.658%      2.931%
pcb2         8    0.212%   0.315%   0.109%      1.412%
pcb4         1    0.617%   0.779%   0.456%      4.445%
```

**两个问题，第二个更重要：**

1. **校准分布不迁移。** 校准集上 `P(hot>=tau_hot)` 按构造恒为 1%，
   测试图上却是 0.3–4.4%。根因：k=1 时 support-only bank 就是这一张图，
   `d_cal` 量的是"排除 3×3 邻域后**同图内**的自相似距离"，而同图自相似
   远强于跨图匹配 —— `d_cal` 偏小 → `tau_hot` 被压低 → 测试图过度触发。
   k 增大时 bank 变为跨图，迁移略有改善但不完全。
   （这正是 §2.2.4 事先声明的分布不匹配，此处给出实测幅度。）

2. **触发不判别对象（更严重）。** pcb2 1-shot：good 0.658% vs bad 1.063%；
   screw 8-shot：good 0.114% vs bad 0.133%。**触发的大多是正常区域。**
   而 transistor 上判别很好（bad 是 good 的 20–400 倍）。
   即 selector 的效果**高度依赖对象**，在 screw / pcb2 上近乎不判别。

**不因此修改任何阈值**（§2.2.4 冻结声明：触发率是结果，不是调参目标）。
本发现的意义是：G1/G3 的结果将主要由"哪些对象上 selector 能判别"决定，
必须逐对象报告，不能只看均值。
