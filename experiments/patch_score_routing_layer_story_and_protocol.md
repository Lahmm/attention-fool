# Patch-score 路由选层：论文故事与完整实验协议

## 1. 论文主张

本工作的起点不是证明高层 patch-score 会迁移到背景，也不是把 patch-score 解释为单 patch 因果显著性。需要检验的核心故事是：

> 不同架构在不同深度形成可用于 global/local 语义路由的表示。基于该表示构造 patch-score，并在统一 score 极性下 drop 一部分候选 patch，可以重组源模型的证据使用路径；RGB opponent-channel noise 随后扰动剩余证据的表达。如果该重组减少源模型特异成分、保留更多跨模型共享成分，它应提高跨模型梯度一致性和最终迁移 ASR。

因此，层是架构相关的路由坐标，极性是方法级定义：

- 每个源模型可以选择不同候选层；
- 所有源模型必须使用同一个 `high` 或 `low` 极性；
- 层与极性只在独立校准集上选择一次，然后冻结；
- 语义成熟度、clean logit drop 和单 patch occlusion 都不能替代跨模型迁移选层。

参考工作中高层语义位置迁移的现象只提供启发。当前实现的 patch-score 明确定义为

```text
s_i = cosine(local_token_i, global_representation)
```

其中 global representation 可以是 CLS、CaiT class-attention CLS 或 GAP。若参考工作的 patch-score 公式、global token 构造或模型结构与此不同，不能把其观察当作当前公式的直接证据。

## 2. 可证伪假设

### H1：架构相关的语义路由层

有池化、分阶段或 class-attention 结构的模型不必都在最后一层获得最可迁移的路由。每个架构至少存在一个候选层，其 production-style patch drop 在独立校准集上优于该架构的其他候选层；该层是否为 final 是实验结果，不是前提。

语义成熟度实验只回答该层的 global/local 表示是什么、是否稳定以及与最终表示的关系。真正支持 H1 的证据是校准集及冻结后的测试集上的跨模型梯度和 ASR。

### H2：drop 引起路由与梯度重组

在固定主线管线下，冻结路由相对 random/no-drop 应改变 kept-token 表示、drop 前后 score map、多视图梯度结构和源/目标梯度一致性。它不需要造成最大的 source clean-logit drop。

### H3：重组带来迁移收益

冻结路由在 500 张最终测试图上的 transfer ASR 应优于 random，且跨模型梯度 cosine、sign agreement 或 held-out target one-step response 至少给出方向一致的机制证据。若冻结路由 ASR 不优于 random，不能用 clean-output 或特征变化把主张圆回来。

### H4：patch-score 与 Grad-CAM 是不同选择准则

Patch-score 是 label-free、gradient-independent 的 global/local 表示关系；Grad-CAM 是 true-class conditioned 的梯度敏感度。二者可以使用同一局部激活坐标，但不等于使用相同显著性标准。

需要同时检验：

1. 在同一激活上的 rank、top-half IoU 和实际 drop-budget IoU，澄清二者选择区域是否高度相似；
2. 将主线 selector 完整替换为 true-class Grad-CAM，并保持所有其他配置一致，比较 transfer ASR 和迁移梯度。

不预设 patch-score 必须胜过 Grad-CAM。预注册的辅助判据是：patch-score 相对 Grad-CAM 的 paired macro ASR 置信区间下界高于 -1 个百分点时，记为在该 margin 下非劣。

## 3. 候选层与公平预算

| 模型 | 预注册候选层 | global representation |
| --- | --- | --- |
| ViT-B/16 | block 3、6、9、12 | 各层 CLS |
| CaiT-S24 | block 6、12、18、24，及 block 24 + class attention | 前四个 GAP；最后一个 class-attention CLS |
| PiT-B | stage1 block3；stage2 block3/6；stage3 block2/4 | 各 checkpoint CLS |
| Visformer-S | stage1 block4/7；stage2 block4；stage3 block2/4 | 各 checkpoint GAP |

不同层的 native grid 不同，因此不强行固定 patch 数。统一固定：候选集合为 native tokens 的一半，实际 drop 为 native tokens 的约 15%。输出必须记录 token count、grid、实际 drop count 和 ratio。

## 4. 数据划分与冻结规则

| 用途 | 样本 | offset | 是否参与选择 |
| --- | ---: | ---: | --- |
| routing calibration | 128 | 500 | 是，只选全局极性和各模型层 |
| final selector suite | 500 | 0 | 否，只报告冻结结果 |

两个集合使用确定性图像排序和固定 seed；每个 attack job 的 `replay_manifest.json` 保存完整图像 ID，并用冻结配置中的 `image_ids_sha256` 记录校准集合校验值。校准过程为：

1. 对每个 source、每个 candidate layer、`high/low` 两种极性，运行完全相同的主线攻击；
2. 在另外三个注册白盒架构上计算 off-diagonal transfer ASR；
3. 对每个极性，分别为四个 source 选取该 source 的最佳层；
4. 比较两种极性在四个 source 上的 macro transfer ASR，只冻结一个全局极性；
5. 冻结该极性下每个 source 的最佳层。精确并列时使用预注册的 deepest/final tie-break；极性精确并列时选 `high`。

这是“层可不同、极性必须相同”的唯一生产配置。历史 architecture-adaptive polarity 结果只保留为边界分析，不能回流到主方法。

## 5. 固定的 production attack

除 selector/layer/polarity 对照项外，所有实验固定：

```text
epsilon = 16/255
steps = 10
MI decay = 1
10 groups × 2 original/phase views = 20 views/step
opponent_projected noise = 0.2 RMS，initial RGB projection，kept-only
Gaussian gradient residual: sigma = 4, alpha = 0.75
same image IDs, seed, phase choices, mask replay and optimizer
```

主线的两个核心机制保持职责分离：patch-score 决定在哪里重组语义证据，opponent-channel noise 决定如何扰动保留下来的证据。phase pair、多视图聚合、Gaussian residual 和 MI-FGSM 均为固定支撑组件。

## 6. 实验矩阵

### E1：候选层的语义成熟度（描述性，不选层）

对每个模型、每个候选层报告：

- global 表示的 within/between-class cosine margin 与确定性 split 1-NN；
- 对最终 global 表示的 linear CKA；
- 水平翻转与 reflect-padded phase shift 下的 global cosine；
- 对齐后的 patch-score Spearman；
- 固定温度 score entropy、top-15% probability mass 与 score standard deviation；
- global mode、source module、native grid 和 token count。

若抽样中某类别没有至少两个样本，class-within 和 split-1NN 对该项记为缺失，不伪造类别成熟度结果。该实验用于解释 pooling transition、CLS/GAP/class-attention 的形成过程，不用于决定攻击层。

### E2：独立校准并冻结 layer/polarity

唯一选择指标是三目标 off-diagonal transfer ASR。不得使用 clean logit drop、source feature change 或 source masked-gradient cosine 进行 tie-break。

### E3：路由和跨模型梯度机制

在冻结层上比较 `selected/opposite/deviation/random/no_drop`，并可用 `--layers all` 扫描所有候选层。报告：

- clean/masked global cosine；
- kept-token clean/masked cosine；
- drop 前后 score-map Spearman；
- route gradient 与 source clean gradient cosine；
- 20 个实际 view gradients 对 raw mean 的 cosine、sign agreement、effective rank；
- raw/processed source route gradient 与每个 held-out target clean gradient 的 cosine/sign agreement；
- 沿 route gradient 做一步更新后 target true-logit drop、CE increase 和 prediction change。

kept-token 与 score-map 变化是通用 token-interaction proxy。若后续增加 exact attention rollout，它是补充项，不能替代跨模型梯度与 ASR。

### E4：Grad-CAM Protocol A，同一激活比较

在每个冻结层的同一 logit-connected local activation 上构造：

- patch-score：同激活的 global/local cosine；
- Grad-CAM：true-class gradient weighted token activation + ReLU。

报告 Spearman、top-half IoU、top-drop IoU、true-class 与 alternate-class Grad-CAM 的变化，以及同激活 patch-score 与 production patch-score 的一致性。ViT/PiT final checkpoint 若模型 forward 的最终 block output 与 logit 断开，必须显式报告使用 final-block-input 的 logit-connected fallback；不得静默产生全零 Grad-CAM。

### E5：500 图 selector-only 主实验

七个条件全部使用冻结 production attack：

```text
selected polarity + selected layer
opposite polarity + selected layer
deviation + selected layer
random + selected layer
no drop
selected polarity + final layer
true-class Grad-CAM + selected layer
```

`final_layer` 直接检验“是否必须最后一层”。Grad-CAM 条件只替换 selector，不改变 candidate ratio、drop ratio、opponent noise、phase pairs、20-view aggregation、Gaussian residual、MI-FGSM、epsilon、steps、seed 或图像。

ASR 分母只包含各 target 在 clean image 上预测正确的样本。报告每 source-target ASR、每 source 的 per-image target macro，以及 selected 对 Grad-CAM 的 paired bootstrap 95% CI。

### E6：opponent-channel 的正交消融

选层与 selector 冻结后，再比较 opponent projected、RGB Gaussian 与 feature IID Gaussian，三方向单独注入、注入位置、kept-only/other masks 和 RMS matching。该实验不与 E2 的选层混合，避免把颜色机制的超参数吸收到 routing calibration 中。

## 7. 执行顺序

### 7.1 生成校准任务

```bash
python experiments/patch_score_routing_calibration.py \
  --write-template outputs/research/routing_calibration/results.csv
```

命令同时生成 `results.csv.manifest.json`。执行其中 114 个固定 attack/eval job，把 off-diagonal ASR 填回模板后冻结配置：

```bash
python experiments/patch_score_routing_calibration.py \
  --results outputs/research/routing_calibration/results.csv \
  --image-ids-sha256 <calibration-image-id-sha256> \
  --output-config outputs/research/routing_calibration/frozen_routing.json \
  --output-summary outputs/research/routing_calibration/summary.json
```

### 7.2 运行语义成熟度与机制分析

```bash
python experiments/patch_score_layer_semantic_maturity_experiment.py \
  --models all --samples 128 --sample-offset 500

for source in vit_base_patch16_224 cait_s24_224 pit_b_224 visformer_small; do
  python experiments/patch_score_routing_gradient_experiment.py \
    --source "$source" \
    --routing-config outputs/research/routing_calibration/frozen_routing.json \
    --layers frozen --samples 64 --sample-offset 500
done
```

需要解释选层时，将 `--layers frozen` 改为 `--layers all`；该扫描仍然只作机制分析，不重新选层。

### 7.3 运行 Grad-CAM Protocol A

```bash
python experiments/patch_score_gradcam_selected_layer_experiment.py \
  --routing-config outputs/research/routing_calibration/frozen_routing.json \
  --samples 128 --sample-offset 500
```

### 7.4 生成并执行最终 selector suite

```bash
python experiments/patch_score_selector_suite.py \
  --routing-config outputs/research/routing_calibration/frozen_routing.json \
  --write-manifest outputs/research/selector_suite/manifest.json \
  --samples 500 --sample-offset 0
```

依次或经调度器执行 manifest 中 28 个 attack command。全部对抗样本生成后进行统一 paired evaluation：

```bash
python experiments/patch_score_selector_transfer_eval.py \
  --manifest outputs/research/selector_suite/manifest.json \
  --output-dir outputs/research/selector_suite
```

## 8. 结论判定

- `selected > random` 且跨模型 gradient alignment/one-step response 方向一致：支持语义路由主线；clean logit drop 可小。
- `selected > random` 但直接梯度诊断不一致：只主张经验迁移收益，机制措辞降级。
- `selected <= random`：H2/H3 不成立，不能用 clean-mask 结果补救。
- selected layer 不是 final：支持架构相关选层；若四个模型均为 final，只能报告该数据与协议下 final 胜出，不能推广为普遍规律。
- patch-score 与 Grad-CAM 区域部分重叠但 label dependence/gradient dependence 不同：支持“同坐标、不同准则”。不能把低 overlap 当作必须成立的结论。
- patch-score 对 Grad-CAM 非劣或更高：支持用 label-free routing 替代 class-conditioned selector；若明显更低，应收缩方法优势并重新审视 routing motivation。

历史 clean-mask polarity、single-patch occlusion 和 source-only route response 均保留在附录，职责是说明 patch-score 不等于因果显著性，而不是选择生产极性或证明迁移性。
