# 使用说明

建议分工：
- 你本机：执行命令、保留环境、上传结果。
- ChatGPT：生成正式脚本、冻结 Gate、复核结果、统计检验、决定下一分支。

第一步不要直接跑 R6。
先完成现有 `RUN_R5_LOCKED_REPLICATION.ps1`。
R5 replication 回传后，再生成 R6-A 正式可执行脚本，避免 replication 尚未封口前提前污染下一阶段。
