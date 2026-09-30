# R6-A Smoke 执行

## 放置
把 `r6a_supervised_smoke` 放到：

`E:\work\freshman\r6a_supervised_smoke\`

## 环境
必须使用和 formal/R5 相同 Python/conda 环境。

## 运行
```powershell
cd E:\work\freshman
Set-ExecutionPolicy -Scope Process Bypass
.\r6a_supervised_smoke\RUN_R6A_SMOKE.ps1
```

## 结果
`E:\work\freshman\results\model_v0\metrics\r6a_smoke\`

上传整个目录。

## 这不是最终模型
它允许使用 test GT 做 grouped-CV supervised linear probe，仅用于判断 frozen DINO representation 中是否存在可线性恢复的 defect 信息。
不能把 supervised 结果当 few-shot anomaly detection 成绩。
