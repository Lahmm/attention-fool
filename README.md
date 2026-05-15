# Attention-fool

本项目用于在 ViT 类视觉模型上生成对抗样本，并评估这些样本在多个黑盒 Transformer 视觉模型上的迁移攻击效果。当前主攻击流程包含基础 MI-FGSM，以及一个基于 FFT stability 的前后景对比塌缩攻击 `fft-cc`。

## 环境安装

```powershell
pip install -r requirements.txt
```

默认数据路径：

```text
data/clean_resized_images
data/image_name_to_class_id_and_name.json
```

## 生成干净样本

只导出被当前白盒模型正确分类的干净样本：

```powershell
python main.py --mode clean --max-attacked-samples 20 --output-dir outputs/clean
```

输出文件会保留原始标注文件名，例如：

```text
clean_ILSVRC2012_val_00000001.JPEG
```

## 基础 MI-FGSM 攻击

```powershell
python main.py --mode attack --attack-type mifgsm --max-attacked-samples 20 --epsilon 0.0313725 --steps 10 --decay 1.0 --output-dir outputs/attack/mifgsm
```

常用攻击强度：

```text
8/255  = 0.0313725
16/255 = 0.062745
32/255 = 0.12549
```

如果不传 `--step-size`，代码会自动使用：

```text
step_size = epsilon / steps
```

## FFT-CC 攻击

`fft-cc` 是全图 L_inf 约束下的 MI-FGSM 风格攻击。它先用 clean tokens 的 channel-wise FFT stability 得到 foreground-like / background-like patch 的软划分，然后在多个 residual-stream 层上缩小两组 patch 与 CLS 的对齐差异。

攻击目标：

```text
loss = CE - lambda_contrast * mean_l |fg_align_l - bg_align_l|
```

默认层是：

```text
--layers=-4,-2,-1
```

运行示例：

```powershell
python main.py --mode attack --attack-type fft-cc --max-attacked-samples 20 --epsilon 0.062745 --steps 10 --decay 1.0 --layers=-4,-2,-1 --lambda-contrast 1.0 --fft-topk 1 --output-dir outputs/attack/fftcc
```

参数说明：

```text
--layers             参与 contrast loss 的 transformer block，支持任意数量，例如 -6,-4,-2,-1
--lambda-contrast    前后景 patch-CLS 对齐差异塌缩项权重
--fft-topk           channel-wise FFT stability selection 的 Top-K，默认 1
```

PowerShell 中带负数的 `--layers` 建议写成等号形式：

```powershell
--layers=-4,-2,-1
```

## Lazy Aggregation 攻击

`lazy-agg` 是单白盒 ViT 非定向迁移攻击。它用 clean CLS attention、patch norm、token FFT stability 和图像低频结构自动构造 foreground set 与 background anchor set，并在 MI-FGSM 更新中把 CLS 聚合从前景推向 anchor patch。

默认配置会在选择 `--attack-type lazy-agg` 时启用：

```text
epsilon=16/255, steps=20, decay=1.0, layers=-6,-5,-4,-3,-2,-1
anchor_top_ratio=0.25, fg_top_ratio=0.25, lambda_anchor=1.0
warmup_steps=3, grad_combine=anchor_modulate
```

运行示例：

```powershell
python main.py --mode attack --attack-type lazy-agg --max-attacked-samples 50 --decay 1.0 --output-dir outputs/attack/lazyagg
```

## 迁移攻击评估

对保存好的对抗样本做多黑盒模型评估：

```powershell
python transfer_eval.py --image-dir outputs/attack/fftcc --prefix adv_
```

迁移评估默认会批量推理，建议在 4090 上从下面的配置开始试：

```powershell
python transfer_eval.py --image-dir outputs/attack/fftcc --prefix adv_ --batch-size 128 --num-workers 8 --prefetch-factor 4
```

如果显存足够，可以继续提高 `--batch-size`；如果 CPU 图像解码仍然吃紧，可以把 `--num-workers` 调到 12 或 16。`--amp` 会启用 fp16 autocast，速度通常更快，但可能让极少数边界样本的预测发生变化。

默认黑盒模型包括：

```text
deit_base_patch16_224
beit_base_patch16_224
swin_tiny_patch4_window7_224
pvt_v2_b2
cait_s24_224
levit_256
pit_s_224
crossvit_15_240
```

也可以手动指定模型，多个模型用逗号分隔：

```powershell
python transfer_eval.py --image-dir outputs/attack/fftcc --prefix adv_ --model-name deit_base_patch16_224,swin_tiny_patch4_window7_224,pvt_v2_b2,cait_s24_224
```

评估结束后会输出每个模型的 `acc`、`ASR`，并额外给出字典形式的攻击成功率：

```text
ASR by model:
{'deit_base_patch16_224': 0.75, ...}
```

迁移评估结束后会自动记录结果。Excel 会写到 `outputs/excel`，文件名由对抗样本目录唯一确定，例如 `outputs_attack_fftcc.xlsx`。实验名默认使用对抗样本目录名，也可以用 `--exp-name` 指定：

```powershell
python transfer_eval.py --image-dir outputs/attack/fftcc --prefix adv_ --exp-name fftcc
```

## 可视化

可视化内容包括：

```text
Input
Attention Scores
A@V Scores
FFT Stability Selection
Patch Score Overlay
Mechanism Overlap
```

可视化干净样本：

```powershell
python visualize_attention_patchscores.py --image-dir outputs/clean --pattern "clean_*" --model-name deit_base_patch16_224 --output-dir outputs/vis_clean_deit_base_patch16_224
```

可视化对抗样本：

```powershell
python visualize_attention_patchscores.py --image-dir outputs/attack/fftcc --pattern "adv_*" --model-name deit_base_patch16_224 --output-dir outputs/vis_adv_deit_base_patch16_224
```

常用参数：

```text
--model-name          可视化使用的 timm 模型名
--block-index         可视化的 transformer block，默认 -1
--fft-topk            FFT stability selection 的 Top-K
--overlap-top-ratio   重叠诊断图中 high attention / high patch score / high FFT 的比例
```

## 模型设置

白盒攻击模型默认在 `nets.py` 中设置：

```python
DEFAULT_MODEL_NAME = "vit_base_patch16_224"
```

迁移评估中的黑盒模型由 `transfer_eval.py` 的 `--model-name` 控制，不会使用 `nets.py` 的 hook 包装。
这是ds
