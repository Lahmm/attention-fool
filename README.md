# PRD：方向方差比 × 正收缩比例选层与随机 token drop

更新日期：2026-11-04。PRD 为仓库工作名称，正式论文方法名待定。用户确认方向方差比乘正收缩比例为唯一主线选层规则。

## 唯一主线选层

每张干净图像在攻击前只做一次无 drop、无特征增强的完整前向，保存各原生 block 的局部输入和输出。将每个局部向量及对应全局参考分别在 float64 中做 L2 归一化（模型特征为 float32）。输入和输出分别减去各自的局部 token 中心，平均平方距离分别记为输入、输出分散度。

输出特征沿全局单位参考的投影总体方差，除以“输出分散度／原生特征维数”，得到方向方差比。正收缩比例为 `max(0, 1 - 输出分散度 / 输入分散度)`。两者乘积是唯一主线选层分数。输入或输出分散度为零时分数为零；不人为截断正的小分母。非有限值或零范数特征报错。全图没有正分数时也按原生顺序选择最早的 block。

每图选择最高分 block，并列选择原生前向顺序最早者；在该 block 的输入处 drop。该边界在全部攻击步与视图中固定。统计量仅依赖当前图像，无固定层标签、跨图校准、拟合系数或深度权重。原 float32 余弦分数的总体方差只作为所选层的记录字段，不再直接决定选层。

ViT/DeiT 使用最终 `model.norm` 后的分类 CLS；DeiT 排除 CLS 和 distillation 两个 prefix。CaiT 使用 class-attention 后、最终 LayerNorm 前 CLS。PiT 使用各 stage 末端的原始 CLS，最后 stage 使用最终归一化 CLS。各 block 的输入和输出都不额外施加最终 LayerNorm。PiT 的输入位于原生 pooling 和 stage norm 之后，收缩不包含 pooling。

`block0` 是第一个 block 输入；`stageN_block0` 是该 stage pooling 和 stage norm 后的第一个 block 输入。架构约定见 [执行契约](experiments/prd_method_contract.md)。

## 随机干预与优化

每个 step/group 在所选输入边界的全部局部 token 中均匀、无放回抽样，清零全部通道，保留 prefix、网格和 token 数。分数只决定层，不决定被丢弃的位置。同组原始／反射平移视图使用空间对应的 mask，drop 数保持一致。同批样本可以选择不同边界，沿原生顺序遍历边界并集时只干预对应样本。

默认 seed 42、10 steps、10 groups × 2 views、epsilon 16/255、步长 epsilon/10、动量系数 1；每图 100 次原始 mask 抽样。像素梯度平均后执行动量和投影更新。关闭 progressive drop 时跳过选层。保留 `--input-diversity-views-per-group 1/2` 消融。

| Source | 候选 block | 总 drop 比例 | 增强强度 β |
| --- | --- | ---: | ---: |
| ViT-B/16 | 12 | 20/196 | 0.2 |
| CaiT-S24 | 24 | 30/196 | 0.2 |
| PiT-T | 全部 2+6+4 | 30/196 | 0.4 |
| DeiT-B distilled | 12 | 20/196 | 0.2 |
| PiT-B（额外源） | 单独保留固定 stage2_block1、stage3_block2、stage3_block3 | 原固定比例 | 0.4 |

动态四源的 `--drop-ratios` 各值求和为单图比例，保留 12 位小数，数量为 `max(1, round(N*r))`。PiT-T 原生网格为 27×27、14×14、7×7。PiT-B 按各固定边界比例执行。显式 `--vit-analysis-checkpoints` 保留为固定边界分析入口。

每个可微视图在全部初始局部位置注入一次经真实 RGB 投影并 RMS 匹配的随机特征增强，prefix 不变。代码默认 `opponent`；`--noise-structure isotropic_rgb` 显式启用投影前 i.i.d. RGB 采样。当前已完成的新方法测量使用 opponent。论文的 i.i.d. 共同增强规划放在补充设置，须用实际配置对应的结果，不将 opponent 表改标签。噪声不单列为贡献。

```bash
python main.py --whitebox-model vit_base_patch16_224 --output-dir outputs/attack/prd_vit
python main.py --whitebox-model cait_s24_224 --output-dir outputs/attack/prd_cait
python main.py --whitebox-model pit_ti_224 --output-dir outputs/attack/prd_pit_t
python main.py --whitebox-model deit_base_distilled_patch16_224 --output-dir outputs/attack/prd_deit_b
```

`attack_params.json` 记录唯一选层公式与归一化、逐图最高分层和输入 checkpoint、原始余弦方差 `peak_variance`、方向比、`peak_contraction_score`、正收缩比例、输入／输出分散度、原生网格和实际 drop 数。`peak_variance` 是所选层的诊断量，并非选层分数。模型 API 为 `select_contraction_checkpoints`，统计内核为 `select_contraction_peaks`；无旧最大方差策略或兼容别名。

## 已完成的 1000 图 FPR13 结果

下表对应 seed 42、10 steps、10 groups × 2 views、opponent 增强。新方法最初在仓库外独立快照执行，2026-11-04 被确认为唯一主线；原始归档的实验标签、参数、源码差异及哈希保持真实执行状态。

| Source | Overall ASR | Strict ASR | 固定单层 Strict ASR | 差值（pp） |
| --- | ---: | ---: | ---: | ---: |
| vit | 87.95% | 86.95% | 87.53% | -0.58 |
| cait | 89.14% | 88.37% | 89.05% | -0.68 |
| pit_t | 84.28% | 84.28% | 85.08% | -0.81 |
| deit_b | 95.30% | 95.30% | 94.78% | +0.52 |
| macro | 89.17% | 88.72% | 89.11% | -0.39 |

结果、全部候选 counts、逐目标 ASR、固定单层和历史动态方法的直接差值见 [当前方法结果](results/prd_direction_contraction_s42_1000_fpr/README.md) 和 [汇总 CSV](results/prd_direction_contraction_s42_1000_fpr/summary.csv)。固定层命中数依次为 848、843、277、184。

旧最大方差与二阶矩、标准差加均值等是历史对照。固定多 checkpoint/opponent 的 89.72% Overall、89.29% strict 及 2×2 消融仍是原配置证据，见 [历史记录](results/prd_s42_1000_fpr_ablation.md)。[故事](experiments/prd_paper_story.md)、[蓝图](experiments/cvpr2027_writing_blueprint.md) 和 [交接](experiments/server_handoff_inventory.md) 按当前选层定义维护。

## 评估与验证

```bash
python transfer_eval.py --image-dir outputs/attack/prd_vit --protocol fpr --no-tf32
python -m unittest discover -s tests
RUN_REAL_PRD_ADAPTERS=1 python -m unittest tests.test_progressive_adapters
```

输入为 `data/clean_resized_images` 和标签 JSON，离线权重位于 `data/huggingface`。FPR13 使用全部对抗样本上的 `1 - adversarial accuracy`；strict 排除完全相同的源目标。协议见 [FPR13 定义](experiments/fpr_target_protocol.md)。`results/` 仅保存 FPR13 证据和来源；历史 seed-42 随机实现与 PyTorch 全局 seed 42 按用户约定统一报告。

保留 `experiments/download_fpr_models.py` 下载工具，不在仓库恢复已完成的实验运行脚本。Figure 1 是跨层原始分数观察素材，Figure 2 是历史固定双 checkpoint 素材，均不冒充当前选层可视化。

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
