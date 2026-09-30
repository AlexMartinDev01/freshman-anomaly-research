# TSN Formal Failure Root-Cause Local Kit V2

## 目的
这不是新模型训练，也不是重新跑 243 cells。

它只对正式 243-cell 失败结果做 **真实协议因果诊断**，使用原项目的 raw DINO train/test feature cache。

目标只回答三件事：

1. `q=log(max/top1%)` 本身是否还有判别信息？
2. `|q-q_ref|` 是否因为方向性而失败？
3. 真正失败源是否是 `train-support calibration / bank-domain mismatch`？

---

## 一、放到哪里

把整个文件夹：

`tsn_rootcause_local_kit_v2`

复制到：

`E:\work\freshman\tsn_rootcause_local_kit_v2\`

最后目录类似：

E:\work\freshman\
- experiments\
- results\
- third_party\
- tsn_rootcause_local_kit_v2\

正式结果必须仍在：

`E:\work\freshman\results\model_v0\metrics\tsn_formal_v2.csv`

---

## 二、使用原来跑 DINO 实验的 Python 环境

如果你之前使用 conda：

```powershell
conda activate 你的原实验环境
```

不要临时重装 torch，也不要新建一个不同版本的实验环境。

---

## 三、先运行 Precheck

PowerShell:

```powershell
cd E:\work\freshman
python tsn_rootcause_local_kit_v2\precheck_rootcause.py
```

必须看到：

- CUDA = True
- GPU 名称正确
- 所有 selected objects 都 `[CACHE OK]`
- 最后 `PRECHECK PASS`

如果失败，不要继续跑因果实验，把完整报错发回 ChatGPT。

---

## 四、一键正式运行

如果 PowerShell 允许运行 ps1：

```powershell
cd E:\work\freshman
Set-ExecutionPolicy -Scope Process Bypass
.\tsn_rootcause_local_kit_v2\RUN_ROOTCAUSE_AUDIT_V2.ps1
```

它会自动：

1. precheck
2. 跑代表性真实 cell 的 causal audit
3. 自动分析结果

支持 `--resume`，中断后重新执行即可。

---

## 五、这次实际计算什么

### A. Formal replay
重新计算：
- raw-final baseline
- naive formal TSN

必须和原 243-cell 结果对得上。

若 replay 不一致，停止机制解释。

---

### B. Directionality
同一批 test image 计算：

- `AUC(q)`
- `AUC(-q)`
- `AUC(|q-q_ref|)`

如果 `q` 或 `-q` 明显优于 absolute deviation，
说明“取绝对值导致方向信息丢失”成立。

---

### C. Oracle full-k center
用 test-good label 只做诊断：

`q_good = median(q(test-good; full k bank))`

然后：

`|q_test - q_good|`

这不是可部署方法，只是因果 intervention。

如果它能把 AUROC 大幅恢复：
说明 tail-shape signal 仍然存在，但 train-support center 不对。

---

### D. Matched-bank support-only
对每个 support i：

`B_i = R \ {r_i}`

同时计算：

- `q_ref_i = q(r_i; B_i)`
- `q_test_i = q(test; B_i)`

再：

`median_i |q_test_i - q_ref_i|`

这保证 reference 与 test 使用同一个 bank / operator。

如果明显优于 naive：
说明 bank/scoring-domain mismatch 是重要原因。

---

### E. Matched oracle
在每个 B_i 下，用 test-good center：

`median_i |q_test_i - median(q_good; B_i)|`

如果它远高于 matched-support：
说明主要问题是 train-normal -> test-normal calibration / identifiability。

---

### F. Negative controls

1. cyclic wrong pairing
2. repeated shuffled pairing
3. same-size common-reference control

目的：排除“只是多算几次产生 ensemble 平滑”的解释。

Matched 必须明显优于 shuffled 才能把作用归因给 bank matching。

---

## 六、输出在哪里

运行结束后：

`E:\work\freshman\results\model_v0\metrics\tsn_failure_audit_v2\`

会有：

1. `causal_audit_cells.csv`
2. `causal_audit_per_image.csv`
3. `causal_audit_bank_level.csv`

把这三个文件打包上传给 ChatGPT。

---

## 七、结果怎么判

### 情况 1
oracle full 大幅恢复 + matched support 也恢复

=> `signal still exists`，主要是 bank-domain calibration mismatch。

### 情况 2
oracle full 恢复，但 matched support 不恢复；matched oracle 恢复

=> train-support 无法可靠估计 test-normal reference。
真正问题是 identifiability / train-test normal shift。

### 情况 3
q 或 -q 明显好于 oracle absolute

=> absolute-value / directionality 是主要问题。

### 情况 4
连 oracle absolute 都救不回来

=> tail-shape statistic 本身在 formal scoring 下失效。
TSN 线路应该停止，而不是继续救。

### 情况 5
matched ≈ shuffled

=> matched 的提升如果存在，也更可能是 ensemble 或 bank-size effect，
不能声称 bank identity matching 是因果机制。

---

## 八、这轮之后才允许设计新机制

本轮只做 root-cause identification。

不允许：
- 看结果后临时调 q 公式；
- 临时换 top-k 百分比；
- 改 Gate；
- 给某个对象单独设方向；
- 用 test labels 构造最终方法。

只有根因因果实验完成以后，才设计 small true-protocol mechanism。
