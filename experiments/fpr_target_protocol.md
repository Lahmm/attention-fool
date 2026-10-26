# PRD 的 FPR 目标评估协议

这是当前 PRD 迁移测试的主协议。新结果默认按以下 13 个 FPR 目标报告；project 的 14 目标协议保留作历史对照，不与 FPR 结果混合汇总。

使用 `transfer_eval.py --protocol fpr` 评估 13 个目标：ViT-B/16、CaiT-S24、PiT-B、Visformer-S、Swin-T、DeiT-T、CoaT-T，以及 ResNet-18、VGG-16、DenseNet-121、EfficientNet-B0、MobileNet-V3-Small、ResNeXt-50-32x4d。

七个 Transformer 使用 timm 预训练权重，六个 CNN 使用 torchvision DEFAULT 权重；图像直接 bicubic resize 到 224×224，转换到 [0,1] 后按各模型配置归一化。该预处理与 project 14 目标协议分别保留。

权重准备工具 `experiments/download_fpr_models.py` 保留，并准备额外的 PiT-T／DeiT-B distilled 源权重，下载到 `data/huggingface` 与 `data/torch`。攻击和评估直接通过 `main.py`、`transfer_eval.py` 调用，不再提供实验编排脚本。

当前论文源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled；PiT-B 额外保留。所有新攻击执行固定 seed 42。PiT-T 和 distilled DeiT-B 不在此目标集中，严格黑盒平均保留全部 13 个目标；其他源只排除与其完全同名的目标。

ASR 使用所有输入的对抗样本，不做 clean-correct 筛选。使用 `--no-tf32` 可关闭 TF32，默认不启用 AMP。评估自动保存逐目标 CSV、配置与汇总 JSON；`--predictions-output` 可额外保存逐样本预测。

相关依据：[FPR 论文](https://arxiv.org/html/2503.15404v1)、[官方模型加载](https://github.com/RYC-98/FPR/blob/main/transferattack/utils.py)。当前目标架构与预处理对齐，并不据此声称本地 Transformer 权重与官方二进制文件逐字节相同。

当前配置的 seed-42、1000 图像四源结果和 2×2 消融见 [FPR 完整实验记录](../results/prd_s42_1000_fpr_ablation.md)；Overall ASR 为 89.72%，严格黑盒 ASR 为 89.29%。全部单元的轻量参数／迁移原始记录保存在受 Git 管理的 `results/prd_s42_fpr_run_artifacts/`。
