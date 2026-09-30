# 执行步骤

1. 把整个 `r5c_replication_amendment` 文件夹放到：
   `E:\work\freshman\r5c_replication_amendment\`

2. 保持原来的 `r5_rawmap_causal_interventions` 文件夹不动。

3. 使用和 R5 MAIN 完全相同的 Python/conda 环境。

4. PowerShell：
```powershell
cd E:\work\freshman
Set-ExecutionPolicy -Scope Process Bypass
.\r5c_replication_amendment\RUN_R5C_REPLICATION.ps1
```

5. 跑完上传：
`E:\work\freshman\results\model_v0\metrics\r5c_replication\`

注意：
- 这轮只用于补上 R5-C 跨 shot replication。
- A/B 即使脚本顺便输出，也不要据此改变已冻结的 A/B 结论。
- 不改 alpha，不改 shuffle 次数，不挑 best alpha。
