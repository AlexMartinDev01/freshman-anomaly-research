# R6-A1D 执行说明

## 目的
不是调模型。
只验证：删掉真实 defect + 周围 halo 后，远场 patch 是否仍保留 bad-image 判别信息。

## 放置
把整个：
`r6a1d_defect_excision`

放到：
`E:\work\freshman\r6a1d_defect_excision\`

保持原项目和 cache 不动。

## 运行
```powershell
cd E:\work\freshman
Set-ExecutionPolicy -Scope Process Bypass
.\r6a1d_defect_excision\RUN_R6A1D.ps1
```

## 输出
`E:\work\freshman\results\model_v0\metrics\r6a1d_defect_excision\`

主要文件：
- object_radius_alpha_summary.csv
- per_bad_image.csv
- random_translation_null.csv
- alpha1pct_decision_table.csv

把整个目录打包上传。

## 严格规则
- 不改 radius
- 不改 alpha
- 不改 GT threshold
- 不挑 best alpha
- 不追加对象
- random translation null 固定 100 次
- 这一步只做机制诊断，不设计最终 detector
