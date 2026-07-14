# Patch-Score Routing Attack

本仓库实现单白盒 ViT 的 Patch-Score–Conditioned Stochastic Token Masking 攻击。
默认配置是当前最强主线：

```text
original L12 score/mask
→ pixel-space patch dropout
→ original/post-dropout phase pair
→ kept-only opponent-channel token noise
→ 20-view gradient mean
→ raw-scale MI-FGSM update
```

梯度后处理默认是 `mean`，保持上述主线行为。可选模式为
`view_l2_mean`、`sign_consensus` 和 `sign_consensus_transport`；transport 的权重由
`--gradient-consensus-lambda` 控制，默认 `0.2`。100 样本完整消融结果见
`experiments/gradient_postprocess_ablation_s100.md`。

主线不再执行旧的 `_normalize_grad`：20 个 view 聚合后的梯度以原始绝对幅值进入
MI 累积。更新仍使用 `sign()`，并继续执行 epsilon 投影，因此攻击约束和步长不变。

仓库仅保留以下攻击能力：

- 当前 `original_score_postdrop_phase_pair` 主线；
- 上一代 `token_patch_dropout`；
- 通用 `patch_dropout`；
- 基础 MI/NI、DIM 和 TI；
- `nets/` 中已有的全部白盒模型。

## 安装

```bash
pip install -r requirements.txt
```

默认数据路径：

```text
data/clean_resized_images
data/image_name_to_class_id_and_name.json
```

## 当前主线

默认参数直接对应 100-image 最强配置：ViT-B/16、`epsilon=16/255`、10 steps、
10 groups × 2 views、L12 high-score random mask、post-dropout phase pair、kept-only
opponent-channel noise。

```bash
python main.py \
  --max-attacked-samples 100 \
  --output-dir outputs/attack/patch_score_routing
```

主要 phase 参数：

```bash
--input-diversity-groups 10
--input-diversity-views-per-group 2
--input-diversity-phase-shift-set "4,4;8,8;12,12"
```

## 保留的对照攻击

旧 token dropout（20×1）：

```bash
python main.py \
  --attack-method token_patch_dropout \
  --input-diversity-groups 20 \
  --input-diversity-views-per-group 1 \
  --output-dir outputs/attack/token_patch_dropout
```

通用 pixel patch dropout：

```bash
python main.py \
  --attack-method patch_dropout \
  --guide-aug-copies 20 \
  --feature-layer -1 \
  --output-dir outputs/attack/patch_dropout
```

基础 MI/NI、DIM、TI 可以与 `none`、`patch_dropout` 或 `token_patch_dropout` 组合：

```bash
python main.py \
  --attack-method none \
  --dim --ni --ti-sigma 1.0 \
  --output-dir outputs/attack/dim_ti_ni
```

当前 post-dropout phase-pair 主线固定不叠加 DIM。

## 迁移评估

```bash
python transfer_eval.py \
  --image-dir outputs/attack/patch_score_routing \
  --prefix adv_
```

攻击目录会保存 `attack_params.json`，迁移评估结果记录到 `outputs/csv`。
