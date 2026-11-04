# PRD 的 FPR13 评估协议

FPR13 是唯一保留的评估协议，`transfer_eval.py` 默认使用它，仍可显式指定 `--protocol fpr`。旧 project 协议及其模型加载分支已移除，全部模型权重缓存保留。

七个 Transformer 目标为 ViT-B/16、CaiT-S24、PiT-B、Visformer-S、Swin-T、DeiT-T、CoaT-T，使用 timm 预训练权重；六个 CNN 为 ResNet-18、VGG-16、DenseNet-121、EfficientNet-B0、MobileNet-V3-Small、ResNeXt-50-32x4d，使用 torchvision DEFAULT 权重。

全部图像直接 bicubic resize 到 224×224，转为 [0,1] 并按各模型配置归一化。权重准备工具 `download_fpr_models.py` 同时准备 PiT-T、distilled DeiT-B 两个源的额外权重，缓存到 `data/huggingface` 与 `data/torch`。PiT-B 是目标，旧固定三边界源入口已经移除。

四个攻击源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled，新攻击执行固定 seed 42。ASR 使用所有对抗样本的 `1 - adversarial accuracy`，不做 target-clean-correct 筛选。strict 仅排除完全相同源目标：ViT/CaiT 各 12 个，PiT-T/distilled DeiT-B 各 13 个。四源等权汇总。

```bash
python transfer_eval.py --image-dir outputs/attack/prd_vit_new --no-tf32
```

归档测量使用 batch 32、关闭 TF32、不启用 AMP。自动保存逐目标 CSV 和配置／汇总 JSON；`--predictions-output` 可保存逐图预测。当前千图 opponent 结果见 [结果索引](../results/README.md)。最终动态 i.i.d. 主表须使用对应实际运行，已有结果不改标签。

协议依据为 [FPR 论文](https://arxiv.org/html/2503.15404v1) 与 [官方模型加载](https://github.com/RYC-98/FPR/blob/main/transferattack/utils.py)。本地架构和预处理对齐，不据此声称 Transformer 权重与官方二进制文件逐字节相同。
