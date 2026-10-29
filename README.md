# Progressive Route Disruption

本项目研究面向黑盒迁移的 **Progressive Route Disruption（PRD）**。我们观察到：同一个局部 token 与全局表示的余弦相似度会随网络深度发生明显变化，高 patch-score 的区域也随之迁移与更替。局部证据参与全局表示形成的过程是动态的。当前执行版先对每张干净图像定位 patch-score 层内方差最大的 block，再在该 block 输入处随机清零局部 token；RGB 对手通道噪声继续扰动全部初始局部特征。

2026-10-29 起，四个主要源模型默认按样本自身的方差峰值选择一个 checkpoint。Patch score 用于选层；该层内的 drop 位置仍从全部局部 token 中均匀随机抽取。

## 方法主线

PRD 只有两个核心机制：

1. **逐样本方差定位与随机 token drop**：攻击开始前，用无 drop、无特征噪声的干净前向采集各完整 block 的输出，计算原始 patch-score 的总体方差；在方差最大 block 的输入处 hard-zero 随机局部 token，并继续同一次前向传播；
2. **RGB opponent-channel random noise**：在亮度、红绿和黄蓝方向采样，经模型真实初始 RGB projection 映射并在特征空间 RMS matching，注入全部初始局部特征，包括之后会被 drop 的位置；CLS/prefix 不加噪声。

每张图只在攻击开始前定位一次，全部攻击步共享该图的 checkpoint；同一 batch 中各图可以使用不同边界。每个 attack step、每个 augmentation group 都生成新的随机 mask。Original view 使用该 mask，phase view 使用其空间变换后的对应 mask。默认 10 steps × 10 groups × 2 views，每张图对应 100 条 schedule 和 100 次 mask selection。

```text
clean image → per-image maximum raw patch-score variance → fixed per-image boundary
current adversarial pixels
→ fresh uniformly random token mask at that boundary
→ original schedule / spatially transformed phase schedule
→ RGB opponent-channel noise on all initial local features
→ traversal to the selected block input and local-token hard zeroing
→ raw 20-view gradient mean
→ MI projected update
```

## 当前配置

论文当前四源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled；PiT-B 额外保留源支持。Visformer 仅作为评估目标和历史结果中的模型保留。

| Source | Checkpoint 选择 | 总 drop 比例 | Opponent strength |
| --- | --- | ---: | ---: |
| ViT-B/16 | 12 个 block 中逐图选取方差峰前边界 | 20/196 ≈ 10.2041% | 0.2 |
| CaiT-S24 | 24 个局部 block 中逐图选取方差峰前边界 | 30/196 ≈ 15.3061% | 0.2 |
| PiT-T | 全部 stage 的 2+6+4 个 block 中逐图选择 | 30/196 ≈ 15.3061% | 0.4 |
| DeiT-B distilled | 12 个 block 中逐图选取方差峰前边界 | 20/196 ≈ 10.2041% | 0.2 |
| PiT-B（额外保留） | 固定 `stage2_block1,stage3_block2,stage3_block3` | 沿用 5/2/6 的原生配置 | 0.4 |

ViT、DeiT 使用最终 `model.norm` 后的分类 CLS；DeiT 的 distillation token 不作为参照或局部 token。CaiT 使用 class-attention 模块及最终归一化之后的 CLS。PiT 每个 stage 使用该 stage 末端的原始 CLS，最后一个 stage 使用最终归一化后的 CLS。局部 token 均取完整 block 输出，不额外应用最终 `model.norm`。先用 float32 计算原始余弦分数，再用 float64 计算层内总体方差；不做 min–max，不跨样本平均。方差并列时选择前向顺序最早的 block。

`block0` 表示第一个 block 的输入。PiT 的 `stageN_block0` 表示该 stage 完成 pooling 和 stage norm 后、第一个 block 之前的输入。PiT-T 的局部网格为 27×27、14×14、7×7；按所选网格乘总比例取整。总比例统一保留 12 位小数，drop 数为 `max(1, round(N*r))`，实际比例随整数化记录。每个 batch 内按边界分组前向，再按原样本顺序合回梯度；损失保持原完整 batch 的平均尺度。

所有新执行统一使用 seed **42**。`--drop-ratios` 对四个动态源表示各值之和，即单图总比例；省略时使用上表预算。PiT-B 仍按每个固定 checkpoint 分别设置。显式 `--vit-analysis-checkpoints` 仅用于固定边界的 ViT 对照，并关闭逐样本定位。

```bash
python main.py --whitebox-model vit_base_patch16_224 --output-dir outputs/attack/prd_vit
python main.py --whitebox-model pit_ti_224 --output-dir outputs/attack/prd_pit_t
python main.py --whitebox-model deit_base_distilled_patch16_224 --output-dir outputs/attack/prd_deit_b
```

`--input-diversity-views-per-group 1` 使用原始视图，`2` 使用原始／平移配对（默认）；`--input-diversity-groups` 控制每步组数，每步总视图不超过 20。初始 opponent noise 的特征 RMS 和投影权重固定停止梯度。像素梯度平均后直接进行系数为 1 的动量累积与投影更新。

`main.py` 在运行开始时统一设置 PyTorch seed 42，mask、噪声和 phase 直接使用 PyTorch 随机函数；固定 batch 和执行流程时可重复采样。入口保存对抗 PNG 与 `attack_params.json`。动态执行记录每张图的图像名、方差峰值层、峰前 checkpoint、峰值方差、原生网格、实际 drop 数和比例，以及所有样本的 checkpoint 频次。现存实验运行脚本已移除，使用上述入口执行方法。

## 1000-image transfer results

当前评估协议为 **FPR13**。下表是已完成的固定 checkpoint 版本：seed 42、1000 张图像，2026-10-23 更新 PiT-T 和 distilled DeiT-B 的 drop/full 消融单元。逐样本动态 checkpoint 版本尚未完成同规模评估，下表数值属于固定版本。

| Source | Overall ASR | Strict black-box |
| --- | ---: | ---: |
| ViT-B/16 | 88.02% | 87.06% |
| CaiT-S24 | 90.24% | 89.50% |
| PiT-T（14/2/2） | 85.81% | 85.81% |
| DeiT-B distilled（block2/block9） | 94.80% | 94.80% |
| **Four-source mean** | **89.72%** | **89.29%** |

完整逐目标矩阵、2×2 消融和 ASR 差值见 [FPR13 实验记录](results/prd_s42_1000_fpr_ablation.md)。[results 索引](results/README.md) 和 [轻量原始参数／迁移记录](results/prd_s42_fpr_run_artifacts/) 随 Git 提交，可在其他机器查看。历史 randomness.py seed 42 与当前 PyTorch seed 42 按用户约定作为统一报告标准。

方法执行顺序见 [PRD 方法记录](experiments/prd_method_contract.md)。论文叙事、主实验记录与建议写作结构见 [PRD 论文叙事文档](experiments/prd_paper_story.md)。

异地写作所需的当前主实验参数、原始记录和服务器材料位置见 [写作交接与文件清单](experiments/server_handoff_inventory.md)。

论文以 ASR 呈现迁移效果。写作结构、图表计划和表达约定见 [CVPR 2027 写作蓝图](experiments/cvpr2027_writing_blueprint.md)。

## 评估与验证

图像位于 `data/clean_resized_images`，标签为 `data/image_name_to_class_id_and_name.json`，源与 Transformer 目标从 `data/huggingface` 离线缓存读取。

```bash
python transfer_eval.py --image-dir outputs/attack/prd_vit --protocol fpr --no-tf32
python -m unittest discover -s tests
RUN_REAL_PRD_ADAPTERS=1 python -m unittest tests.test_progressive_adapters
```

当前迁移测试以 FPR 的 13 目标协议为主，使用 FPR 对齐的模型加载和预处理；`results/` 仅保留 FPR13 的结果与实验记录。目标集的选择独立于源支持。结果默认写入图像目录的 `transfer.csv` 与 `transfer.json`，可用 `--predictions-output` 保存逐样本预测。ASR 使用全部输入样本；严格黑盒排除与源模型完全相同的目标。协议定义见 [FPR 目标评估协议](experiments/fpr_target_protocol.md)。

权重准备工具 `experiments/download_fpr_models.py` 保留，负责缓存 FPR 目标和额外的 PiT-T／DeiT-B distilled 源权重，不执行攻击或评估。

## Figure 2 固定 checkpoint 版本的首次迭代素材

```bash
python figure2.py
```

使用与 Figure 1 相同排序的前 50 个有效样本，输出到 `outputs/figure2/<样本名>/`。固定预训练 ViT-B/16、seed 42、block3/block10 各清零 10 个 token，保持默认 opponent noise 和 10 groups×2 views。仅执行第一步，显式使用原 10 步设置的步长 `epsilon/10`，保存第一个 group 的两个分支；全部 20 个视图参与该步像素更新。

每个样本目录包含 `x_t.png`、`x_t_grid.png`、`x_t_plus_1.png`、`trace.pt`、`drop_positions.csv` 和 `metadata.json`。`group00/original/` 与 `group00/translated/` 分别包含真实视图及网格图、实际采样的 RGB 噪声显示图、等效加噪输入预览、初始特征加噪前后及噪声本身的逐 token RMS 图，以及每个 checkpoint 的 mask、mask 叠加图和清零前后的 RMS 图。网格按实际 14×14 空间位置绘制；RMS 图共享状态尺度，两视图 RGB 噪声显示共享零中心尺度，具体映射记录于元数据。

噪声直接截取实际随机采样和实际返回的投影噪声，不重新抽样。脚本验证等效 RGB 输入投影后的初始特征与真正的加噪特征一致；`noisy_equivalent_clipped.png` 是截断到显示范围的预览，未截断数据保存在 `trace.pt`。`*_mask_on_view.png` 是实际特征清零位置的空间示意，模型前向仍执行特征 hard-zero。各 checkpoint 单独显示自己的 mask。实际像素更新结果为 `x_t_plus_1.png`，精确浮点像素同时保存在 `trace.pt`。

采集类、局部随机采样拦截和临时实例包装全部位于 `figure2.py`，普通攻击入口不受影响；包装在异常退出时也会恢复。可用 `--batch-size`、`--device`、`--image-dir`、`--annotations-path` 配置执行。批次布局会影响随机采样，实际 batch size 和样本索引记录在元数据中。运行需要已安装的依赖及 `data/huggingface` 内对应预训练权重；再次运行更新相同样本的素材。

## Figure 1 跨层观察元素

```bash
python figure1.py
```

该入口使用预训练 ViT-B/16，对按标注图像名称排序的前 50 个有效样本进行干净前向观察，采集全部 12 个 block 完整输出处、最终 `model.norm` 之前的局部 token，并统一与完整编码器经过最终 `model.norm` 后的 CLS token 计算余弦相似度。每张图像的 12 层共用同一个最终 CLS 参照。图像使用现有的 224×224 输入，以模型自身的 mean/std 归一化；固定 seed 42。运行前需安装依赖，并在 `data/huggingface` 准备好对应预训练权重。

所有文件写入 `outputs/figure1/<图像名称去除扩展名>/`，共 50 个样本子目录。每个目录包含：

- `input.png`：与模型输入空间对齐的 RGB 原图。
- `heatmap_block01.png` 至 `heatmap_block12.png`：平滑 patch-score 热力图与原图的叠加图，不绘制网格线或边框。默认使用 45% 热图与 55% 原图混合。每张图像的每层独立对 196 个原始余弦分数做 min–max 归一化，再将 14×14 的归一化分数按双线性插值平滑放大至 224×224（`align_corners=False`），最后映射到连续蓝→青→绿→黄→橙→红的彩虹渐变，相邻颜色控制点之间按 RGB 线性插值。配图色条应标注为 `Normalized patch score`，左端 0 为蓝色，右端 1 为红色。颜色表示各层内部的相对高低；整层分数相同时统一映射到 0.5（绿色）。平滑插值仅用于显示。
- `patch_scores.csv`：全部 2352 个真实分数，列为 `layer,patch_index,row,col,score`。层编号为 1–12，patch 索引和行列坐标从 0 开始，`patch_index = row * 14 + col`。自行选定 `(row,col)` 后，按 `layer` 排序即可得到 12 个曲线数据点。
- `metadata.json`：样本顺序、输入图像 SHA-256、模型及预处理配置、观察位置、坐标约定和热图色标。

脚本保留全部位置的数据，不选择固定位置、不生成曲线或完整拼图。首个 batch 会核对采集路径与模型原生前向的 logits 一致。可用 `--batch-size`、`--device cpu/cuda/mps` 调整执行设置，`--overlay-alpha` 调整热图叠加权重，`--image-dir` 和 `--annotations-path` 可指定输入。再次运行会更新相同样本目录中的导出文件；若目录内包含其他样本子目录，则报错并保留已有文件。

```bash
pip install -r requirements.txt
```
