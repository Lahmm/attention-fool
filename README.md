# GDI：定位有效干预层，增强迁移攻击

更新日期：2026-11-05。GDI（Geometry-Guided Depth Intervention）通过逐样本选择有效干预层，在其输入处随机清零局部 token，优化可迁移的对抗扰动。

论文围绕两项创新展开：发现一些可能承担较重要信息汇总作用、且输入干扰具有较好迁移效果的层；提出模型无关、样本相关的计算方式，为每张图像定位有效干预层。故事与写作结构见 [论文故事](experiments/gdi_paper_story.md) 和 [blueprint](experiments/cvpr2027_writing_blueprint.md)。

## 方法与代码

`main.py` 调用 `attack.py` 中的 `GDIAttacker`。每张干净图像先进行一次无 drop、无特征增强的前向，采集各原生 block 的局部输入、输出及架构对应的全局参考。将 float32 特征转为 float64 后逐向量 L2 归一化，以输出全局方向总体方差除以输出局部分散度／原生维数，再乘 `max(0, 1 - 输出分散度 / 输入分散度)`。选择最高分 block，并在其输入处干预；并列或全零时选择原生顺序最早者。

所选边界在全部攻击步与视图中保持固定。每个 step/group 从该边界全部局部位置均匀、无放回抽样，清零所选 token 的全部通道，保留 prefix、token 数和原生网格。同批图像各自选择一个边界。完整定义见 [方法契约](experiments/gdi_method_contract.md)。

| 源模型 | CLI 名称 | 候选 block | drop 比例 | β | 已归档攻击 batch |
| --- | --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | `vit_base_patch16_224` | 12 | 20/196 | 0.2 | 48 |
| CaiT-S24 | `cait_s24_224` | 24 | 30/196 | 0.2 | 16 |
| PiT-T | `pit_ti_224` | 2+6+4 | 30/196 | 0.4 | 32 |
| DeiT-B distilled | `deit_base_distilled_patch16_224` | 12 | 20/196 | 0.2 | 48 |

PiT-B 和 Visformer 是 FPR13 目标。PiT-T 按所选 27×27、14×14 或 7×7 网格实际清零 112、30 或 8 个位置。

## 环境与运行

`requirements.txt` 对齐 torch 2.10.0、torchvision 0.25.0、timm 1.0.24；服务器解释器为 `/root/miniconda3/envs/att-atk/bin/python`。入口离线使用 `data/huggingface` 和 `data/torch` 权重缓存；补充权重使用 `experiments/download_fpr_models.py`。

```bash
python main.py --whitebox-model vit_base_patch16_224 --batch-size 48 --output-dir outputs/attack/gdi_vit_new
python main.py --whitebox-model cait_s24_224 --batch-size 16 --output-dir outputs/attack/gdi_cait_new
python main.py --whitebox-model pit_ti_224 --batch-size 32 --output-dir outputs/attack/gdi_pit_t_new
python main.py --whitebox-model deit_base_distilled_patch16_224 --batch-size 48 --output-dir outputs/attack/gdi_deit_b_new
python transfer_eval.py --image-dir outputs/attack/gdi_vit_new --no-tf32
```

输出为 `outputs/attack/` 下的子目录，入口生成前清空该子目录；新运行请指定新目录。默认输入为 `data/clean_resized_images` 和 `data/image_name_to_class_id_and_name.json`，其中 1000 张图和标签受 Git 管理。权重缓存与 `outputs/` 被忽略。

默认 seed 42、1000 图、10 steps、10 groups × 2 views、epsilon=16/255、步长 epsilon/steps、动量系数 1。基础设置包括初始局部特征的投影 i.i.d. RGB 高斯增强、原始／反射平移视图与空间对应 mask。直接平均视图像素梯度，再累积动量并做 sign 投影更新。详细构造与参数集中在方法契约。

| 参数 | 作用 |
| --- | --- |
| `--drop-ratio` | 单图所选边界比例；ViT/DeiT 默认 0.102040816326，CaiT/PiT-T 默认 0.15306122449 |
| `--input-diversity-groups` | 默认 10，每步总视图数最多 20 |
| `--input-diversity-views-per-group` | 保留 1/2 视图消融，默认 2 |
| `--input-diversity-phase-shift-set` | 默认 `4,4;8,8;12,12` |
| `--noise-strength` | 基础特征增强强度 β，0 关闭，默认使用适配器设置 |
| `--no-gdi-drop` | 无 drop 消融，跳过干净选层 |
| `--fixed-block K` | 四源固定选层消融：CaiT 为 1–24，其他源为 1–12；按原生 block 顺序编号，与 `--no-gdi-drop` 互斥 |
| `--epsilon`、`--step-size`、`--steps` | 扰动预算、步长和步数 |
| `--image-dir`、`--annotations-path`、`--max-attacked-samples` | 输入、标签和样本数 |

攻击 batch 默认 96，上表列出已有测量的实际 batch。`attack_params.json` 保存当前参数、逐图选择、几何统计、实际网格与清零数量。

四源固定选层消融使用 `--fixed-block K`：ViT/DeiT 为 1–12，CaiT 为 1–24（不含 class-attention），对应内部 `block{K-1}`；PiT-T 为跨 stage 顺序的 1–12，依次覆盖 stage1 的 2 个、stage2 的 6 个和 stage3 的 4 个 block，边界名为 `stageS_blockB`（B 从 0 开始），stage 入口位于原生 pooling 和 norm 之后。全部样本在指定 block 输入处干预，跳过干净前向选层；每个 step/group 仍重新抽样 token 位置，并使用空间对应的 phase mask。省略该参数使用逐图 GDI 选层。固定模式记录 `checkpoint_selection_mode=fixed`、`fixed_block`、`fixed_checkpoint`、逐图边界和实际清零数量，不记录几何分数。示例为第 11 个 block（内部 `block10`）：

```bash
python main.py --whitebox-model vit_base_patch16_224 --fixed-block 11 --batch-size 48 --noise-strength 0.2 --drop-ratio 0.102040816326 --output-dir outputs/attack/gdi_vit_fixed11_new
```

## FPR13 结果

已完成 seed 42、每源 1000 图、i.i.d. 增强及配对视图的四源测量：

| Source | Overall ASR | Strict ASR |
| --- | ---: | ---: |
| ViT-B/16 | 87.41% | 86.36% |
| CaiT-S24 | 89.75% | 88.98% |
| PiT-T | 84.85% | 84.85% |
| DeiT-B distilled | 95.37% | 95.37% |
| 四源等权平均 | **89.3423%** | **88.8893%** |

参数、逐目标结果和来源见 [GDI i.i.d. 归档](results/gdi_iid_paired_s42_1000_fpr/README.md)。历史选择器与实际增强配置由 [结果索引](results/README.md) 管理，原始运行位置见 [交接记录](experiments/server_handoff_inventory.md)。

评估使用 FPR13 的 7 Transformer＋6 CNN。ASR 为全部对抗样本的 `1 - adversarial accuracy`；strict 排除完全相同源目标，四源等权平均。`transfer_eval.py` 保存逐目标 CSV 和配置／汇总 JSON，`--predictions-output` 保存逐图预测。完整协议见 [FPR13](experiments/fpr_target_protocol.md)。

## 验证与观察材料

```bash
python -m unittest discover -s tests
RUN_REAL_GDI_ADAPTERS=1 python -m unittest discover -s tests -p test_gdi_adapters.py
python figure1.py
```

`figure1.py` 导出前 50 张真实 ViT 样本的跨层原始余弦分数和热图。几何曲线、观察与干预 ASR 的对应编排，以及 GDI 流程图按 blueprint 准备。内部维护的历史验证见 [精简记录](experiments/gdi_internal_simplification.md)。
