# 第三方依赖的外部记录

`third_party/` 下的检出**不进版本库**（AnomalyDINO 排除 `.git` 后仍有 13 GB）。
本目录记录每个依赖的上游来源、固定 commit、以及本地改动，使这些树可以重建。

---

## AnomalyDINO — 无本地改动，纯上游检出

```
upstream   https://github.com/dammsi/AnomalyDINO.git
commit     b9d1c2648e3a5247437d4d953d907a8f3d994457
local      git status = clean（无任何本地修改）
size       13 GB（不含 .git）
```

**它是 V3 pipeline 的运行时依赖**，必须存在于磁盘上：

- `experiments/model_v0/v3_refine.py`、`cache_m1_repr.py`、`v3_check_scale.py` 等
  通过 `sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")` 引入
- 实际使用的是 `src.backbones.get_model("dinov2_vits14", ...)`
  （其内部 `torch.hub.load('facebookresearch/dinov2', ...)` 还会拉取 DINOv2 本体，
  缓存于 `~/.cache/torch/hub/`）

重建：

```bash
git clone https://github.com/dammsi/AnomalyDINO.git third_party/AnomalyDINO
cd third_party/AnomalyDINO && git checkout b9d1c26
# 无需打任何 patch
```

---

## SubspaceAD — 有本地改动，以 patch 固化

```
upstream   https://github.com/CLendering/SubspaceAD.git
base       ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7
patch      subspacead.patch
sha256     feadd999b1759a99c7d8dae70ca98055737d98db2e1de135aa8c7532b498c077
files      main.py                               (26 lines changed)
           src/subspacead/core/extractor.py      (36 lines changed)
```

**不是 V3 主实验的运行时依赖**（V3 只用 DINOv2）。它服务于 V3 之前的
SubspaceAD 对照实验（`gate_r2_subspacead_eval.py`）。

重建：

```bash
git clone https://github.com/CLendering/SubspaceAD.git third_party/SubspaceAD
cd third_party/SubspaceAD && git checkout ef56d5c
git apply ../../third_party_patches/subspacead.patch
```

**已验证**：在干净检出 @ `ef56d5c` 上 `git apply --check` 通过；在当前工作树上
`git apply --reverse --check` 通过（即 patch 与工作树完全一致）。

### 三处改动，全部环境变量门控、默认惰性

| # | 文件 | 改动 | 是否改变默认行为 |
|---|---|---|---|
| 1 | `main.py` | `np.trapz` → `np.trapezoid` | **是**（但为必需：NumPy 2.0 删除了 `np.trapz`，原样运行会崩溃并中止整个 run。二者是同一函数的改名） |
| 2 | `main.py` | dump hook：`SUBSPACEAD_DUMP_DIR` 设置时，在 `post_process_map` **之前**保存原始 patch-grid map | 否（不设环境变量则完全无操作） |
| 3 | `extractor.py` | `SUBSPACEAD_NO_SALIENCY=1` 时以 `output_attentions=False` 运行 | 否（不设环境变量则与上游逐位一致） |

改动 1 的必要性：上游的 AU-PRO 只是交叉核对（我们的 map 会用自研 evaluator 重打分），
但该行崩溃会中止整个 run，所以在 NumPy 2.0 下必须修。

改动 2 的原因：外部 evaluator 需要 **patch grid**，不是被 `post_process_map`
重采样/模糊到 `image_res` 之后的图——后者会在 evaluator 自身的插值之上再叠加一次。
另：dump 文件名**必须带类别名**，因为该目录被 15 个类别共用，而缺陷类型名
跨类别重复（`color` 出现 5 次、`crack` 3 次、`good` 全部 15 次），
按 `<defect_type>_<stem>` 命名会静默覆盖约一半的 map（实测只 dump 出 908/1725）。

改动 3 的原因：ViT-giant @672 存储全部 40 层的注意力图每张图约需 14 GB，
在 8 GB 显卡上 OOM。注意图只被 `--bg_mask_method dino_saliency` 和可视化使用，
默认 pipeline 下跳过是安全的。

**`src/subspacead.egg-info/*` 的改动未包含在 patch 中**——那是
`pip install -e` 自动生成的构建产物，不是源码改动。

---

## MVTecAD2_code_utils

`third_party/MVTecAD2_code_utils/` 为空目录，未使用，同样不进版本库。
