# Phase 8 结果总账（冻结）：External Defect Support Acquisition

> 冻结于 tag `phase8_external_closed`。
> **「Retriever / Selector」分支关闭：判据在目标内部零预测力，且几何解释也失败。**
> 下一阶段进入 Conditional Feature Generator。
> 过程脚本见 `gate8_external.py` / `gate8b_perclass.py` / `build_external_pool.py`。

---

## 最终结论（严谨表述）

> **外部类的缺陷 patch 确实携带可迁移的信息——在匹配规模下，外部缺陷 bank 系统性地
> 优于同一类别的正常 bank（84 格中 14 格高于 baseline vs 4 格）。**
> **但「该选哪个外部类」既不能由目标的 normal manifold 几何决定（目标内 ρ=−0.04, p=0.71），
> 也不能由缺陷分布本身的几何决定（目标内 ρ=+0.29, p=0.13，且方向相反）。**
> **四个对象中只有 wallplugs 一个从中获益（fabric +20.6，跨 split 稳定）。**

---

## 1. 协议

- 冻结 DINOv2，**不训练 adapter**，raw fusion `S_n − S_d`，β=1。
- `S_n`：到**目标类**正常 bank（全部 train/good）的 1-NN 距离。
- `S_d`：到**外部 bank** 的 1-NN 距离。
- **任何目标类缺陷 patch 都不得进入 bank**（`load_external()` 硬性排除）。
- 评估集统一为：全部 good 测试图 + **留出的那一半**缺陷图。
  baseline、ORACLE、R0/R1/R2 全部在同一评估集上打分，因此可直接比较。
- ORACLE = 用另一半（support half）的目标自身缺陷，仅作上限参考，非可用方法。

**外部池**（8 类 AD2 × 各自 `test_public` 的缺陷 patch，`gt_frac > 0.10`）：

| class | defect patches |
|---|---|
| can | 54 |
| wallplugs | 528 |
| rice | 564 |
| fabric | 1512 |
| fruit_jelly | 1624 |
| walnuts | 2982 |
| sheet_metal | 4257 |
| vial | 5258 |

⚠️ **「外部」在这里只是数据集内的跨类别迁移**，不是来自独立语料。任何正面结论只能支持较弱的表述。

## 2. 池化 / 随机 external bank：全面有害

相对 baseline 的 Δ（3 splits）：

| config | can | sheet_metal | vial | wallplugs |
|---|---|---|---|---|
| baseline | 53.3 | 84.1 | 89.4 | 42.3 |
| **ORACLE（目标自身缺陷）** | 57.5 | 91.5 | 100.0 | 77.5 |
| R0 random external | −8.1 | −7.6 | −15.5 | −2.4 |
| R1 matched top1 | −8.4 | −6.5 | −12.4 | +10.8 |
| R1 matched top3 | −9.0 | −6.9 | −14.2 | −7.9 |
| R2 pooled (all) | −11.0 | −3.5 | −15.1 | −6.7 |

**机制**：`S_d` 单独当检测器时 AUROC < 50（wallplugs 低至 28–42）。
即目标异常**离**外部缺陷比目标正常样本**更远**，所以减去它必然伤。

## 3. 单类矩阵：14/84 格高于 baseline，且高度集中在 wallplugs

`gate8b_perclass.py`：对每个 (目标, 外部类, N) 建恰好 N 个 patch 的 bank。

| object | baseline | 高于 baseline 的格数 | best Δ | median Δ |
|---|---|---|---|---|
| can | 53.3 | 0/21 | −0.1 | −4.7 |
| sheet_metal | 84.1 | 2/21 | +0.6 | −2.9 |
| vial | 89.4 | 0/21 | −6.3 | −15.1 |
| **wallplugs** | 42.3 | **12/21** | **+20.8** | **+3.3** |

## 4. 关键对照：缺陷 bank vs 正常 bank（同类别、同规模）

这是本阶段**唯一抢救出来的正面结果**。

用同一外部类的 **正常** patch 建同样大小的 bank 作为对照：

| | 高于 baseline 的格数 |
|---|---|
| 外部**缺陷** bank | 14 / 84 |
| 外部**正常** bank | 4 / 84 |

逐格「缺陷 Δ − 正常 Δ」在 28 格中 22 格为正（只有 6 格 ≤0，且都在 ±3 以内）。

**判决：外部缺陷 patch 携带的信息，同一类别的正常 patch 没有。这不是「类别外观」效应，
是缺陷特异性效应。**

wallplugs 上最清楚（N=1000，baseline 42.3）：

| class | defect bank | normal bank | 差 |
|---|---|---|---|
| fabric | **62.9** | 35.7 | +27.2 |
| walnuts | **59.6** | 39.8 | +19.7 |
| rice | 52.7 | 52.0 | +0.7 ← 非缺陷特异 |
| 其余 4 类 | ≤ 45.6 | ≤ 33.3 | — |

fabric 逐 split：61.2 / 65.2 / 62.2；walnuts：58.0 / 60.6 / 60.1 —— **稳定**。

## 5. 判据验证：pooled 显著是假象（本阶段最重要的方法论发现）

R1 的判据是「用目标 normal manifold 匹配外部类」，必须验证它能否**预测**哪一类有用。

把 Spearman 拆成 pooled 与 within-target：

| 数据 | pooled ρ | p | **within-target ρ** | p |
|---|---|---|---|---|
| 外部缺陷 bank | −0.411 | <0.001 | **−0.041** | **0.709** |
| 外部正常 bank | −0.482 | <0.001 | −0.363 | 0.001 |

逐目标（缺陷 bank）：wallplugs 0.00、vial −0.26、can +0.24、sheet_metal +0.87 —— **符号都不一致**。

> pooled 的显著性完全来自**目标间**结构（各目标的平均 match 距离与平均 AUROC 不同），
> 不是判据在目标内部有区分力。**如果只看 pooled，这里会得出完全相反的结论。**

**判决：R1 判据无效。**

### 补充：换一个更 Oracle 的判据也失败

用「目标缺陷 patch 到各外部类缺陷 patch 的 1-NN 距离」作为 relevance（这需要目标标签，
只能作为上界参考）：

| N | defect_match pooled | defect_match **within** |
|---|---|---|
| 200 | −0.179 (p=0.36) | **+0.291** (p=0.13) |
| 500 | −0.159 (p=0.42) | **+0.296** (p=0.13) |
| 1000 | −0.174 (p=0.38) | **+0.263** (p=0.18) |

方向**相反**（越近 AUROC 越低）且不显著。

**判决：即使拥有目标缺陷标签，「哪个外部类有用」也无法由特征空间几何解释。**
这不是「判据选得不好」，是这条几何路线本身走不通。

## 6. 效果对比总结（N=1000，Δ vs baseline）

**外部缺陷 bank：**

| class | can | sheet_metal | vial | wallplugs |
|---|---|---|---|---|
| can | — | −2.9 | −6.3 | +3.3 |
| fabric | −0.1 | −5.1 | −26.5 | **+20.6** |
| fruit_jelly | −9.1 | 0.0 | −13.9 | −3.4 |
| rice | −8.9 | −4.2 | −13.1 | +10.4 |
| sheet_metal | −8.0 | — | −23.1 | −11.0 |
| vial | −4.2 | −2.7 | — | −2.2 |
| wallplugs | −4.6 | −2.4 | −23.2 | — |
| walnuts | −4.9 | −3.8 | −15.1 | **+17.3** |

**外部正常 bank（对照）：**

| class | can | sheet_metal | vial | wallplugs |
|---|---|---|---|---|
| can | — | −18.3 | −26.8 | −9.0 |
| fabric | −7.6 | −13.8 | −29.8 | −6.6 |
| fruit_jelly | −5.8 | −25.0 | −25.3 | −18.4 |
| rice | −7.3 | −14.7 | −35.0 | +9.7 |
| sheet_metal | −6.3 | — | −29.1 | −10.8 |
| vial | −8.4 | −23.9 | — | −17.6 |
| wallplugs | −3.3 | −18.6 | −32.4 | — |
| walnuts | −9.1 | −31.1 | −18.9 | −2.5 |

## 7. 分叉判决

用户的判据：

| 结果 | 分支 |
|---|---|
| retrieval 有效 | → 构建 Retriever / Selector |
| retrieval 无效 | → 构建 Conditional Feature Generator |

**命中第二支，且是强形式**：不只是「当前判据不好」，而是
**两条几何判据（normal manifold、defect manifold）在目标内部都无预测力**。

但要带着一个**未解释的正面事实**进入下一阶段：
**wallplugs 上外部缺陷 bank 有 +20.6 的真实、缺陷特异、跨 split 稳定的增益，而几何无法解释它。**
这说明可迁移的缺陷结构**存在**，只是不体现在「分布距离」上。
Conditional Feature Generator 要学的正是这个「几何之外的」条件化关系。

## 8. can / vial / sheet_metal

- **can**：所有 21 格都不高于 baseline。与 7B-3R/7B-4 一致，属非 support-limited regime。
- **sheet_metal / vial**：最好也只有 +0.6 / −6.3，即完全没有外部迁移。这两类 baseline 本身已经很高
  （84.1 / 89.4），外部 bank 只会引入噪声。

## 9. 本阶段资产

| 类型 | 内容 |
|---|---|
| 脚本 | `gate8_external.py`、`gate8b_perclass.py`、`build_external_pool.py` |
| 数据 | `results/model_v0/external/*_{defect,normal}.npy`（8 类），`metrics/gate8*.csv` |
| 特征缓存 | 新增 fabric / fruit_jelly / rice / walnuts 四类的 train+test 缓存 |

### 性能修复（再次踩坑，已记录）

- `gate8b` 曾在 sheet_metal 上卡死。原因是**两个进程同时跑**，各自约 2.5 GB 显存，
  超过 8 GB 后 **Windows 驱动把显存换页到系统内存**——不是 OOM，是静默地慢 40 倍。
  **此类 sweep 必须串行。**
- 分块尺寸是实测的：450k bank / 4096 query 下 `1024×8192` 为 0.24 s/img，
  `128×1024` 为 0.88 s，`64×512` 为 7.17 s。**不要往小改。**
- `S_n` 只依赖对象、不依赖 split，之前在每个 split 里重算（sheet_metal 上 ~3.7 s/图）。

---

## 继承到下一阶段的强制方法纪律

1. **相关系数必须拆 pooled / within-group。** pooled 显著而 within 为 0 = 组间结构假象。
   本阶段靠这一步避免了一个完全相反的结论。
2. **任何「外部知识有用」的声明，必须配同类别同规模的正常 bank 对照。** 否则无法区分
   缺陷特异性效应与类别外观效应。
3. **bank 规模必须匹配**（`min(N, len(pool), cap)`），并对小池（如 can 的 54 个）如实标注。
4. **两个 GPU 进程不要同跑**——显存超限在 Windows 上表现为静默变慢，不是报错。
5. **单对象上的正面结果不构成方法**，但也不能当作噪声丢掉：要记录为待解释事实。
