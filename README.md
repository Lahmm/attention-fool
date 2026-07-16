# Patch-Score Routing Attack

本仓库实现面向 ViT、CaiT、PiT 和 Visformer 白盒源模型的 patch-score 路由攻击。默认主线为：

```text
final-layer patch score
→ high-score candidate stochastic pixel patch dropout
→ original / phase-shifted view pair
→ kept-only feature noise at the initial RGB projection
→ raw 20-view gradient mean
→ Gaussian gradient residual
→ MI-FGSM update
```

## 安装与数据

```bash
pip install -r requirements.txt
```

默认使用 `data/clean_resized_images` 中的 1000 张 ImageNet 验证图像，以及
`data/image_name_to_class_id_and_name.json` 标签文件。模型权重默认从项目内的
`data/huggingface` 离线缓存加载。

## 默认主线

```bash
python main.py \
  --whitebox-model vit_base_patch16_224 \
  --seed 20260716 \
  --output-dir outputs/attack/vit_mainline
```

默认设置为 1000 样本、`epsilon=16/255`、10 steps、10 groups × 2 views、约 15%
实际 patch drop、kept-only opponent-projected RGB 噪声，以及：

```text
g' = g + 0.75 * GaussianBlur(g, sigma=4)
```

Gaussian residual 位于 20-view raw mean 之后、MI 累积之前。它保留完整原始梯度，
并不是用低通梯度替换原始方向。设置 `--gaussian-alpha 0` 可复现 raw-mean 路径。

### 两种主线噪声

主线只保留两种噪声，二者都固定注入到 initial RGB projection 输出，并且只作用于
kept tokens：

- `opponent_projected`（默认）：在 opponent-channel RGB 基上采样，再通过首层 RGB
  projection 映射到特征空间。
- `gaussian`：直接在初始投影特征上采样 IID Gaussian。

两者都按当前特征 RMS 缩放。切换到普通 Gaussian：

```bash
python main.py \
  --post-dropout-feature-noise-type gaussian \
  --output-dir outputs/attack/vit_feature_gaussian
```

## 保留的对照攻击

仓库继续支持 `none`、通用 pixel `patch_dropout` 和 ViT `token_patch_dropout`，以及
MI、NI、DIM、TI。它们用于与默认结构化主线进行直接对照。

```bash
# 通用 pixel patch dropout
python main.py \
  --attack-method patch_dropout \
  --guide-aug-copies 20 \
  --feature-layer -1 \
  --gaussian-alpha 0 \
  --output-dir outputs/attack/patch_dropout

# ViT token dropout
python main.py \
  --attack-method token_patch_dropout \
  --input-diversity-groups 20 \
  --input-diversity-views-per-group 1 \
  --gaussian-alpha 0 \
  --output-dir outputs/attack/token_patch_dropout

# 基础 MI + NI + DIM + TI
python main.py \
  --attack-method none \
  --dim --ni --ti-sigma 1.0 \
  --gaussian-alpha 0 \
  --output-dir outputs/attack/dim_ti_ni
```

默认 phase-pair 主线不与 DIM 叠加。所有 group/view 配置仍受每步最多 20 次实际
model view 的限制。

## 白盒模型

| 模型 | score 来源 | score 网格 | 默认实际 drop |
| --- | --- | --- | --- |
| `vit_base_patch16_224` | `blocks[11]` CLS/patch | 14×14 | 29/196 |
| `cait_s24_224` | `blocks[23]` + class-attention CLS | 14×14 | 29/196 |
| `pit_b_224` | `transformers[2].blocks[3]` | 8×8 | 10/64 |
| `visformer_small` | `stage3[3]` + GAP pseudo-CLS | 7×7 | 7/49 |

## 迁移评估

```bash
python transfer_eval.py \
  --image-dir outputs/attack/vit_mainline \
  --prefix adv_
```

迁移评估使用 7 个 Transformer 和 6 个 CNN，并将结果写入 `outputs/csv`。保留的
1000 样本历史结果与完整方法说明见
`experiments/mainline_data_aug_gaussian_story_s1000.md`。其中已有 CaiT/PiT/Visformer
结果是 raw mean；当前代码对四个白盒默认启用 Gaussian residual，但不将未运行的
全模型 Gaussian 配置描述为已有实验结论。
