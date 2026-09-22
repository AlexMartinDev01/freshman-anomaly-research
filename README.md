# robust_fewshot_iad — Phase 1: Baseline

> 少量正常样本条件下，面向真实工业环境分布变化的鲁棒视觉异常检测。
> Phase 1 目标：**不创新**，跑通 AnomalyDINO + DINOv2 + MVTec AD 基线并彻底理解数据流。

## 环境

| 项 | 版本 / 配置 |
|---|---|
| OS | Windows 11（原生，未用 WSL2） |
| Python | 3.10（`C:\Python310`，venv 在 `.venv/`） |
| PyTorch | 2.11.0+cu128（RTX 5060 Blackwell 需要 CUDA≥12.8） |
| numpy | 1.26.4（官方要求，勿升 2.x） |
| faiss | faiss-cpu（Windows pip 无 faiss-gpu，用 `--faiss_on_cpu`） |
| GPU | RTX 5060 8GB |

网络备注：机器有本地代理 `127.0.0.1:15732`。HuggingFace / fbaipublicfiles 直连极慢，
需走代理（设 `HTTP_PROXY`/`HTTPS_PROXY` 环境变量）；数据集走 ModelScope 国内镜像直连很快。

## 目录

```
freshman/
├── .venv/                        # Python 3.10 虚拟环境
├── third_party/
│   └── AnomalyDINO/              # 官方代码（尽量不改）
│       └── results_MVTec/        # 官方脚本输出（metrics JSON / 示例图 / anomaly maps）
├── data/
│   ├── mvtec_ad_raw/             # ModelScope 下载的原始 tar.xz
│   └── mvtec_anomaly_detection/  # 解压后的 MVTec AD（15 类）
├── experiments/
│   └── baseline/
│       ├── smoke_test.py         # 单类 smoke test（bottle, 1-shot）
│       └── aggregate_results.py  # 汇总 metrics_seed=*.json 为表格
├── results/
│   ├── metrics/                  # 汇总 CSV
│   ├── heatmaps/
│   └── smoke_test/               # smoke test 示例图
└── README.md
```

## 数据来源

- MVTec AD：ModelScope 镜像 `OpenDataLab/MVTecAD`（官方 CC BY-NC-SA 4.0，
  原官网下载需注册表单，HF 直连过慢不可用）
- DINOv2 权重：`torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`
  走本地代理下载，缓存在 `C:\Users\msi\.cache\torch\hub\`

## 复现命令

```powershell
# smoke test（bottle 单类，1-shot）
E:\work\freshman\.venv\Scripts\python.exe E:\work\freshman\experiments\baseline\smoke_test.py

# 完整 baseline（检测 + 图像级评估；Windows 安全封装，含 read_tiff 补丁）
E:\work\freshman\.venv\Scripts\python.exe E:\work\freshman\experiments\baseline\run_baseline.py

# 像素级指标（内存高效版，直接从 npy 计算；与官方 eval_segmentation 逐位一致，已在 bottle 上验证）
E:\work\freshman\.venv\Scripts\python.exe E:\work\freshman\experiments\baseline\eval_pixel.py

# 汇总结果
E:\work\freshman\.venv\Scripts\python.exe E:\work\freshman\experiments\baseline\aggregate_results.py
```

## Windows 适配说明（重要）

1. `read_tiff` bug：官方探测 `.tif/.tiff/.TIF/.TIFF` 四种扩展名，Windows 不区分大小写导致
   一个文件匹配多次 → 运行/评估脚本统一用 monkeypatch 修复（见 `run_baseline.py`），不改官方源码。
2. 官方 `--eval_segm` 的 PRO 计算会建 (n_img, H, W) 的 float64 数组（峰值 ~8GB），
   本机 32GB 内存 + 常驻软件下会被 OOM 杀掉；且 tiff 每 seed 约 30GB 临时磁盘。
   → 检测只存 32×32 patch 距离 `.npy`（8KB/张），像素指标由 `eval_pixel.py` 流式计算，
     已与官方实现逐位对齐验证（bottle: AU-PRO/px_AUROC/px_F1 差异 = 0.00000）。

## 数据流（smoke test 实测）

```
448x448 RGB → DINOv2 ViT-S/14 → 32x32=1024 patch tokens × 384 dim
→ 参考图 patch features 建 Memory Bank（FAISS IndexFlatL2，L2 normalize 后等价余弦距离）
→ 测试图每个 patch 查 1NN 距离 → 32x32 anomaly map
→ image score = top 1% patch 距离均值（mean_top1p）
```

Smoke test 结果（bottle, 1-shot, seed 0）：image AUROC 99.84%，正常/异常分数均值 0.196/0.430。
