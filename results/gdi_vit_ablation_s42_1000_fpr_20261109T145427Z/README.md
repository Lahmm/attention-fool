# GDI ViT-B/16 完整消融实验

本轮 20 个配置按 `execution_manifest.json` 执行：A0、A1、固定 1–12 层的 A2、B1–B3、C1–C3。共同设置和精确执行命令见 manifest，样本及顺序见 `samples.json`，权重身份见 `weights.json`。

运行状态见 `run_status.json`。仅已完成并经逐样本预测核对的测量进入 `summary.csv` 和 `per_target.csv`；主要指标为排除同源 ViT-B/16 的 12 目标 Strict 平均 ASR，差值以相对 A0 的百分点表示。
