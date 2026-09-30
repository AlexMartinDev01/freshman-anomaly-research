# R6-A1C 修正版执行

把 `r6a1c_corrected` 放到：
`E:\work\freshman\r6a1c_corrected\`

使用与前面 R5/R6 完全相同环境。

运行：
```powershell
cd E:\work\freshman
Set-ExecutionPolicy -Scope Process Bypass
.\r6a1c_corrected\RUN_R6A1C_CORRECTED.ps1
```

结果：
`E:\work\freshman\results\model_v0\metrics\r6a1c_corrected\`

本次只是修复实现并复验原 R6-A1C，不允许改变科学结论阈值或追加对象。
