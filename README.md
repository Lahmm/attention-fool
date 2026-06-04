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

`lazy-agg` 是单白盒 ViT 非定向迁移攻击。当前实现将攻击拆成前向增强、CE loss 和反向梯度处理三个可组合模块。

默认配置：

```text
epsilon=16/255, steps=20, layers=-6,-5,-4,-3,-2,-1
guide_aug=False, dim=False, si=False, eot=False
mi=False, ni=False, normalize_grad=False, ti_sigma=3.0
```

最简单的无增强 FGSM 示例：

```powershell
python main.py --mode attack --max-attacked-samples 50 --steps 1 --ti-sigma 0 --output-dir outputs/attack/lazyagg/fgsm_plain
```

注意力引导增强示例：

```powershell
python main.py --mode attack --max-attacked-samples 500 --layers=-6,-5,-4,-3,-2,-1 --guide-aug --guide-aug-area background --guide-aug-method dropout,jitter,freq --guide-aug-copies 3 --attention-guide-type postsoftmax_cls --attention-guide-build-method pixel --output-dir outputs/attack/lazyagg/bgaug_s02_lastsix_500
```

前向增强与反向梯度模块均由独立参数控制：

```powershell
--dim
--si --si-scales 2
--eot --eot-iter 2
--mi --mi-decay 1.0
--ni
--normalize-grad
--ti-sigma 0
--ti-sigma 3
```

`--guide-aug-area` 可选 `foreground`、`background`、`all`；`all` 不使用 attention guide map。`--guide-aug-method` 可传 `dropout,jitter,freq,lowpass_gauss,laplacian_low,fft_lowboost,illumination_low,band_noise,colored_noise,progressive_spectral_noise,wavelet_noise` 中的一个或多个，实际前向分支数为 `len(methods) * guide_aug_copies`。其中新增低频方法分别对应 Gaussian scale-space 低通、Laplacian pyramid 高频残差抑制、FFT2D 径向低频增益、低频 illumination field 调制；频谱/小波噪声方法分别对应带通频域噪声、有色频域噪声、渐进式频谱噪声和 Haar 小波子带噪声。它们先生成整图增强版本，再由 pixel/patch attention guide map 选择前景或背景区域混合。

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

迁移评估结束后会自动记录结果。CSV 会写到 `outputs/csv`，文件名由对抗样本目录唯一确定，例如 `outputs_attack_fftcc.csv`。实验名默认使用对抗样本目录名，也可以用 `--exp-name` 指定：

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

## 梯度迁移性因果分析

独立入口 `causal_analysis.py` 固定使用当前 DIM + low-attention background augmentation 基准，支持逐步追踪、空间频率干预、MI 开关和汇总报告：

```powershell
python causal_analysis.py trace --max-samples 100 --gradient-decomposition --output-dir outputs/causal/trace_s0
python causal_analysis.py frequency-intervention --component haar:LLH --region low --intervention drop --evaluate-targets --output-dir outputs/causal/drop_llh
python causal_analysis.py mi-switch --mi-switch reset --switch-step 10 --evaluate-targets --output-dir outputs/causal/mi_reset_10
python causal_analysis.py report --input-dir outputs/causal/drop_llh --output-dir outputs/causal/drop_llh
```

FFT 分量写作 `fft:BAND[:ORIENTATION]`，BAND 为 `0..7`，方向可选 `all/horizontal/vertical/diagonal`；三级 Haar 小波包分量写作 `haar:PATH`。无干预执行器逐像素复现现有攻击路径；FFT 和 Haar 投影有 Parseval 能量守恒及重建测试。

## 跨 ViT 梯度分量发现与因果确认

`cross_vit_components.py` 实现固定的 100 图、seeds `0,1,2`、8 个 ViT/Transformer 目标模型协议。它分析完整攻击在增强平均后、进入 MI 累积前的梯度，并使用两组正交分量：

- `fft:BAND:ORIENTATION`：8 个径向频带乘以 `horizontal/vertical/diagonal`，共 24 个全局分量。
- `haar:PATH:ROW:COL`：三级 Haar 小波包的 64 条路径乘以 `4x4` 系数区域，共 1024 个局部分量。区域在小波系数域选择，避免直接空间 mask 的频谱泄漏。

完整流程：

```powershell
python cross_vit_components.py screen --output-dir outputs/cross_vit_components
python cross_vit_components.py confirm-attacks --output-dir outputs/cross_vit_components --candidate-file outputs/cross_vit_components/selected_candidates.json
python cross_vit_components.py confirm-evaluate --output-dir outputs/cross_vit_components --candidate-file outputs/cross_vit_components/selected_candidates.json
python cross_vit_components.py report --output-dir outputs/cross_vit_components --candidate-file outputs/cross_vit_components/selected_candidates.json
```

也可以直接运行：

```powershell
bash scripts/run_cross_vit_component_experiment.sh outputs/cross_vit_components
```

快速协议筛选阶段使用前 15 张图和步骤 `1,10,20,40`，逐目标模型流式计算方向导数、能量归一化方向导数、跨 seed 相干度、目标模型方向一致性和分量能量比例。它从合格 FFT/Haar 分量中确认最高排名的 2 个候选。

确认阶段使用 seeds `0,1` 的 Full 攻击并严格校验图像索引。每个候选运行 `drop` 和 `keep`，最终报告使用后 35 张图做独立确认，对 2 个候选执行 3000 次按 seed、图像、目标模型分层的配对 bootstrap 和 Benjamini-Hochberg FDR。主要产物：

```text
screening_metrics.npz       # 1048 个候选的紧凑筛选观测
screening_report.json       # 候选排名、资格条件和筛选显著性
selected_candidates.json    # 进入因果确认的 2 个候选
final_report.json           # 后 35 张确认判定和全部 50 张效应量
```
## 5-6 小时双实验快速协议

严格串行运行 DIM/BG 机制实验与跨 ViT 分量确认：

```bash
bash scripts/run_dim_bg_then_cross_vit_quick.sh outputs/quick_serial
```

协议统一使用 seeds `0,1`。DIM/BG 阶段输出 `method_high_frequency_ranking.json` 和 `dim_bg_mechanism_report.json`；跨 ViT 阶段使用 15 张筛选、35 张独立确认、8 个目标模型、top-2 候选和 3000 次 bootstrap，输出 `cross_vit_quick/final_report.json`。总控脚本检查各阶段产物并支持断点续跑，最后生成 `combined_conclusion.md`。
