# GDI：另外三个源模型的逐层固定输入干预消融

计划执行 CaiT-S24 24 层、PiT-T 12 层、蒸馏 DeiT-B 12 层，共 48 个配置。每配置为同序 1000 图、seed 42，沿用对应源主结果的权重、增强、drop 比例及攻击 batch。每个配置先攻击，再做完整 FPR13 评估（batch 32、关闭 TF32、无 AMP）。

运行状态见 `run_status.json`；精确边界、原生网格、实际清零数和全部命令见 `execution_manifest.json`；样本与权重身份见 `samples.json`、`weights.json`。动态 GDI 的已执行主结果按原字节保存在 `baselines/`，保留原参数标签与执行身份。
