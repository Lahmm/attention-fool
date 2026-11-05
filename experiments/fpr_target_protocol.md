# GDI 的 FPR13 评估协议

更新日期：2026-11-05。`transfer_eval.py` 使用 FPR13，支持显式指定 `--protocol fpr`。

七个 Transformer 目标为 ViT-B/16、CaiT-S24、PiT-B、Visformer-S、Swin-T、DeiT-T、CoaT-T，使用 timm 预训练权重；六个 CNN 为 ResNet-18、VGG-16、DenseNet-121、EfficientNet-B0、MobileNet-V3-Small、ResNeXt-50-32x4d，使用 torchvision DEFAULT 权重。

全部图像直接 bicubic resize 到 224×224，转为 [0,1] 并按各模型配置归一化。`download_fpr_models.py` 准备目标权重及 PiT-T、distilled DeiT-B 两个额外源的权重，缓存到 `data/huggingface` 和 `data/torch`。

四源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled，新攻击执行固定 seed 42。ASR 为全部对抗样本的 `1 - adversarial accuracy`。strict 仅排除完全相同源目标：ViT/CaiT 各 12 个目标，PiT-T/distilled DeiT-B 各 13 个目标；四源等权汇总。结果分析使用 ASR 与直接 ASR 差值。

```bash
python transfer_eval.py --image-dir outputs/attack/gdi_vit_new --no-tf32
```

归档测量采用 batch 32、关闭 TF32、无 AMP，保存逐目标 CSV 和参数／汇总 JSON；`--predictions-output` 可保存逐图预测。当前 [i.i.d. 主结果](../results/gdi_iid_paired_s42_1000_fpr/README.md) 与历史配置由 [结果索引](../results/README.md) 分别标明实际执行设置。

协议依据为 [FPR 论文](https://arxiv.org/html/2503.15404v1) 与 [官方模型加载](https://github.com/RYC-98/FPR/blob/main/transferattack/utils.py)。本地目标、架构和预处理对应关系由 `transfer_eval.py` 及其测试记录。
