# PRD：动态单边界选层与随机 token drop

更新日期：2026-11-04。仓库只提供一种攻击：每张干净图像按方向方差比 × 正收缩比例选择一个原生 block，在该 block 输入处随机 hard-zero 局部 token。PRD 为工作名称。旧固定双／多 checkpoint 攻击、固定层覆盖入口及其结果和消融已经移除。

## 方法

每图在攻击前进行一次无 drop、无特征增强的前向，采集各 block 的局部输入、输出与架构对应的全局参考。从 float32 特征转为 float64，逐 token 与参考分别 L2 归一化。用输出沿全局方向的总体方差除以输出局部分散度／原生维数，乘 `max(0, 1 - 输出分散度 / 输入分散度)`。选择最高分 block，并列或全零时选原生顺序最早者，在其输入处干预；输入或输出分散度为零时分数为零，非有限或零范数特征报错。

选定边界在全部步与视图中保持。每个 step/group 从该边界全部局部位置均匀、无放回抽样，清零全部通道，保留 prefix、token 数和原生网格。原始／反射平移视图使用空间对应且数量一致的 mask。同批不同边界是不同图像选择的并集，每图仍只有一个边界。

| 源模型 | CLI 名称 | 候选 block | 总 drop 比例 | β | 已归档攻击 batch |
| --- | --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | `vit_base_patch16_224` | 12 | 20/196 | 0.2 | 48 |
| CaiT-S24 | `cait_s24_224` | 24 | 30/196 | 0.2 | 16 |
| PiT-T | `pit_ti_224` | 2+6+4 | 30/196 | 0.4 | 32 |
| DeiT-B distilled | `deit_base_distilled_patch16_224` | 12 | 20/196 | 0.2 | 48 |

PiT-B、Visformer 作为 FPR13 目标保留。全部现有模型权重保留。架构参考、边界与统计见 [方法契约](experiments/prd_method_contract.md)。

## 环境与运行

`requirements.txt` 对齐当前实验环境的 torch 2.10.0、torchvision 0.25.0 和 timm 1.0.24。当前服务器 Python 为 `/root/miniconda3/envs/att-atk/bin/python`。入口离线使用 `data/huggingface` 与 `data/torch`；需要补充权重时运行保留的 `experiments/download_fpr_models.py`。

```bash
python main.py --whitebox-model vit_base_patch16_224 --batch-size 48 --output-dir outputs/attack/prd_vit_new
python main.py --whitebox-model cait_s24_224 --batch-size 16 --output-dir outputs/attack/prd_cait_new
python main.py --whitebox-model pit_ti_224 --batch-size 32 --output-dir outputs/attack/prd_pit_t_new
python main.py --whitebox-model deit_base_distilled_patch16_224 --batch-size 48 --output-dir outputs/attack/prd_deit_b_new
python transfer_eval.py --image-dir outputs/attack/prd_vit_new --no-tf32
```

输出必须是 `outputs/attack/` 下的子目录；入口在生成前清空该子目录，请为新运行指定新目录。输入默认为 `data/clean_resized_images` 与 `data/image_name_to_class_id_and_name.json`。1000 张输入图与标签 JSON 实际受 Git 管理；模型缓存和 `outputs/` 被忽略。

默认固定 seed 42、1000 图、10 steps、10 groups × 2 views、epsilon=16/255、步长 epsilon/steps、动量 1。攻击 batch 默认 96；上表为已有测量的实际 batch。各视图像素梯度直接平均，再进行动量、sign 更新和投影。

| 参数 | 作用 |
| --- | --- |
| `--drop-ratio` | 单图单边界比例；默认有效值 ViT/DeiT 为 0.102040816326，CaiT/PiT-T 为 0.15306122449，沿用已执行设置 |
| `--input-diversity-groups` | 默认 10；每步总视图数不超过 20 |
| `--input-diversity-views-per-group` | 保留 1/2 视图消融，默认 2 |
| `--input-diversity-phase-shift-set` | 默认 `4,4;8,8;12,12` |
| `--noise-structure` | 默认 `opponent`；显式 `isotropic_rgb` 为投影前 i.i.d. RGB 采样 |
| `--opponent-noise-strength` | 两种增强模式的 β，0 关闭增强 |
| `--no-progressive-drop` | 无 drop 消融，跳过干净选层 |
| `--epsilon`、`--step-size`、`--steps` | 扰动预算与更新设置 |
| `--image-dir`、`--annotations-path`、`--max-attacked-samples` | 数据与样本数 |

每个可微视图在全部初始局部位置注入一次经真实 RGB 初始投影和 RMS 匹配的随机特征增强，prefix 不变。论文规划使用显式 i.i.d. 配置；已完成主结果仍为 opponent，保留实际标签。增强构造放入补充实现设置。

## 结果与评估

| Source | Overall ASR | Strict ASR |
| --- | ---: | ---: |
| ViT-B/16 | 87.95% | 86.95% |
| CaiT-S24 | 89.14% | 88.37% |
| PiT-T | 84.28% | 84.28% |
| DeiT-B distilled | 95.30% | 95.30% |
| 四源等权平均 | **89.1673%** | **88.7234%** |

这是 seed 42、每源 1000 图、opponent、10 steps、10 groups × 2 views 的实际 FPR13 测量。相对历史最大方差选层，Overall 和 strict 分别高 1.8346、1.8543 个百分点。完整 [结果索引](results/README.md)、[当前归档](results/prd_direction_contraction_s42_1000_fpr/README.md) 保留逐目标 ASR、逐图选层、实际参数、来源和哈希。

FPR13 是唯一评估协议，`--protocol fpr` 可显式指定。ASR 为全部对抗样本上的 `1 - adversarial accuracy`，不做 target-clean-correct 筛选。strict 排除完全相同源目标，四源等权平均。评估保存 `transfer.csv/json`；`--predictions-output` 可保存逐样本预测。协议见 [FPR13](experiments/fpr_target_protocol.md)。

`attack_params.json` 记录总比例、唯一选层公式、逐图边界、方向比、正收缩比例、乘积分数、原始余弦方差诊断、网格和实际 drop 数。PiT-T 在 27×27、14×14、7×7 网格分别丢弃 112、30、8 个位置。`peak_variance` 是诊断量，选层 API 为 `select_contraction_checkpoints` / `select_contraction_peaks`。

## 验证与论文材料

```bash
python -m unittest discover -s tests
RUN_REAL_PRD_ADAPTERS=1 python -m unittest tests.test_progressive_adapters
python figure1.py
```

`figure1.py` 保留真实样本跨层原始分数和热图导出；完整方向比／收缩曲线及当前方法 Fig. 2 待准备。旧 Fig. 2 脚本和独立 token 实验 runner 已移除；token 实验 [历史证据](experiments/prd_token_selection_evidence.md) 与原始记录仍保留。

论文规划见 [story](experiments/prd_paper_story.md)、[blueprint](experiments/cvpr2027_writing_blueprint.md)，文件和服务器原件见 [handoff](experiments/server_handoff_inventory.md)。历史选层数据仅作证据，仓库没有其他可执行选层策略或固定 checkpoint 攻击。
