# Patch-Score Routing Attack

本仓库当前主线研究一种“语义路由 + 颜色结构随机扰动”的黑盒迁移攻击，重点不是继续堆叠攻击技巧，而是围绕两个核心机制建立清晰、可分析且自洽的文章方法：

1. **Patch-score 引导的 patch drop：扰动应该放在哪里？**
2. **RGB opponent-channel 随机噪声：保留信息应该如何被扰动？**

前者负责空间选择，后者负责扰动形态。核心假设是：patch-score 提供一个 label-free、gradient-independent 的 global/local 表示路由坐标；drop 该坐标中的候选位置并扰动剩余证据，可能重组源模型的梯度路径并增加跨模型共享成分。它不是单 patch 因果显著性定义。

默认主线为：

```text
final-layer global/local patch score on the current adversarial image
→ high-score-tail candidate stochastic pixel patch dropout per step/group
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

## 研究主线与方法动机

### Patch-score 路由：从随机删 patch 变成语义选择性删 patch

普通 patch dropout 只控制删多少，不回答删哪里。当前动态主线在每个攻击 step 的每个 group 上，对当前对抗图像提取 final-layer global representation 和 local patch tokens，用二者余弦相似度得到 patch-score。攻击从 high-score-tail 候选区随机抽取少量 patch，并把 mask 映射回像素空间。同一 group 的 original/phase 两条 view 共用这一个 mask；不同 group 和不同 step 重新计算 score 并重新采样 mask。

研究把它当作一个**语义路由器**来分析：相同 native token ratio 下，不同层的 score-guided drop 是否重组 kept-token 表示、多视图梯度和跨模型梯度？哪一层能产生更高的 held-out transfer ASR？最终层只是对照，不是预设答案。clean logit、单 patch occlusion 和 source feature change 只作为边界诊断，不能用于选层。

完整的冻结规则、Grad-CAM 对照和实验命令见 [候选层路由论文协议](experiments/patch_score_routing_layer_story_and_protocol.md)。

### RGB opponent-channel noise：从各向同性噪声变成颜色结构扰动

普通 feature-space IID Gaussian 只表达“有随机扰动”，没有表达 RGB 输入本身的颜色几何，也没有利用模型第一层如何把 RGB 混合成特征。当前默认噪声在亮度、红绿对抗、黄蓝对抗三个 opponent-channel 方向上采样，再通过真实首层 RGB projection 映射到初始特征空间，并按当前特征 RMS 缩放。

文章动机应明确为：颜色对抗方向提供了比独立 RGB/feature Gaussian 更有结构的随机扰动坐标，而首层 projection 让该坐标与模型的输入通道混合保持一致。噪声只注入 kept tokens，使 patch-score 负责“删掉哪些语义证据”，opponent noise 负责“扰动剩余证据的颜色表达”，形成清晰的职责分工。

后续重点分析三类颜色方向、噪声协方差和频谱、initial projection 注入位置、kept-only 与其他 mask 策略，以及 RMS matching 对有效扰动强度的影响。

### 其他组件的定位

phase-shifted view pair、20-view raw gradient mean、Gaussian gradient residual 和 MI-FGSM 是稳定梯度估计和完成优化的支撑组件，不应喧宾夺主。最终有效性指标是 target-clean-correct transfer ASR；直接机制证据包括跨模型 gradient cosine、sign agreement、held-out target one-step response 和多视图 effective rank。

## 默认主线

```bash
python main.py \
  --whitebox-model vit_base_patch16_224 \
  --seed 20260716 \
  --output-dir outputs/attack/vit_mainline
```

默认 production run 使用历史动态-mask的 `final/high` 行为。`--routing-config` 仍可用于冻结层实验，但不再定义默认生产主线。

默认设置为 1000 样本、`epsilon=16/255`、10 steps、10 groups × 2 views、约 15%
实际 patch drop、kept-only opponent-projected RGB 噪声，以及：

```text
g' = g + 0.75 * GaussianBlur(g, sigma=4)
```

Gaussian residual 位于 20-view raw mean 之后、MI 累积之前。它保留完整原始梯度，
并不是用低通梯度替换原始方向。设置 `--gaussian-alpha 0` 可复现 raw-mean 路径。

### 两种保留的噪声实现

代码保留两种噪声实现，二者都固定注入到 initial RGB projection 输出，并且只作用于
kept tokens。文章主线优先研究 `opponent_projected`；`gaussian` 主要作为结构噪声对照：

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
MI、NI、DIM、TI。它们用于隔离两个核心机制，而不是作为后续主线继续堆叠的方向。

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

## 白盒模型与候选路由层

| 模型 | 预注册候选层 | global 表示 |
| --- | --- | --- |
| `vit_base_patch16_224` | block 3/6/9/12 | CLS |
| `cait_s24_224` | block 6/12/18/24 GAP；block24 class | GAP / class-attention CLS |
| `pit_b_224` | stage1 b3；stage2 b3/b6；stage3 b2/b4 | CLS |
| `visformer_small` | stage1 b4/b7；stage2 b4；stage3 b2/b4 | GAP |

跨层公平性固定 native token ratio：候选集为一半 tokens，实际 drop 约 15%。128图校准和500图selector suite属于已完成的固定-mask历史研究协议，不再覆盖当前动态-mask生产主线。

## 迁移评估

```bash
python transfer_eval.py \
  --image-dir outputs/attack/vit_mainline \
  --prefix adv_
```

迁移评估使用 7 个 Transformer 和 6 个 CNN，并将结果写入 `outputs/csv`。ASR 用于
检验迁移性，不作为后续工作的唯一目标。保留的
1000 样本历史结果与完整方法说明见
`experiments/mainline_data_aug_gaussian_story_s1000.md`。其中已有 CaiT/PiT/Visformer
结果是 raw mean；当前代码对四个白盒默认启用 Gaussian residual，但不将未运行的
全模型 Gaussian 配置描述为已有实验结论。
