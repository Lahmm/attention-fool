# GDI 文件与实验材料交接

更新日期：2026-11-05。方法为 GDI（Geometry-Guided Depth Intervention），当前执行采用方向方差比 × 正收缩比例逐图定位，在所选 block 输入处随机清零局部 token。

## 当前文件

| 内容 | 路径 |
| --- | --- |
| 攻击入口、优化与数据 | `main.py`、`attack.py`、`utils.py` |
| 四源适配与几何统计 | `nets/__init__.py`、`base.py`、`vit.py`、`cait.py`、`pit.py` |
| FPR13 与权重下载 | `transfer_eval.py`、`experiments/download_fpr_models.py` |
| 验证 | `tests/test_dynamic_checkpoints.py`、`test_attack.py`、`test_gdi_adapters.py`、`test_main.py`、`test_transfer_eval.py` |
| 真实样本观察 | `figure1.py`、`tests/test_figure1.py` |
| 方法与协议 | [方法契约](gdi_method_contract.md)、[FPR13](fpr_target_protocol.md) |
| 论文规划 | [故事](gdi_paper_story.md)、[blueprint](cvpr2027_writing_blueprint.md) |
| 结果及来源 | [结果索引](../results/README.md) |
| 输入与标注 | `data/clean_resized_images/` 的 1000 张 PNG、`data/image_name_to_class_id_and_name.json` |

输入图与标签受 Git 管理。模型缓存及 `outputs/` 由服务器保存。当前环境为 Python 3.10.19、torch 2.10.0、torchvision 0.25.0、timm 1.0.24，解释器为 `/root/miniconda3/envs/att-atk/bin/python`。

## 当前 i.i.d. 四源主结果

[GDI i.i.d. 归档](../results/gdi_iid_paired_s42_1000_fpr/README.md) 保存逐源参数、原始迁移表、52 项逐目标结果、选层频次和四源汇总。设置为 seed 42、每源 1000 图、10 steps、10 groups × 2 views、i.i.d. RGB 增强、原始／phase 空间配对 mask。Overall 为 89.3423%、strict 为 88.8893%。

攻击 batch 按 vit/cait/pit_t/deit_b 为 48/16/32/48；评估 batch 32、关闭 TF32、无 AMP。每源的 1000 张对抗图、逐样本预测和日志保留于原始服务器目录，精确位置由归档 `provenance.json` 的 `source_runs` 与 `execution_manifest.json` 提供。

本次归档从已经完成的运行提取测量文件；来源 manifest 保存测量时的命令、commit、源码哈希及参数名称。当前命令和接口见 README 与方法契约。

## 历史结果

| GDI 归档 | 实际配置与用途 |
| --- | --- |
| [direction_contraction](../results/gdi_direction_contraction_s42_1000_fpr/README.md) | 当前乘积选择器，历史 opponent 设置；Overall 89.1673%、strict 88.7234%，用于同配置选层比较 |
| [direction_variance_ratio](../results/gdi_direction_variance_ratio_s42_1000_fpr/README.md) | 仅方向方差比的历史对照 |
| [second_raw_moment](../results/gdi_second_raw_moment_s42_1000_fpr/README.md) | 二阶原点矩的历史对照 |
| [std_abs_mean](../results/gdi_std_abs_mean_s42_1000_fpr/README.md) | 标准差加绝对均值的历史对照 |
| [normalized_std_abs_mean](../results/gdi_normalized_std_abs_mean_s42_1000_fpr/README.md) | 按图层间最大值归一化后相加的历史对照 |

各归档的 `runs/`、`variance_references/` 及其他 reference 子目录保留真实参数与迁移表；`execution.json`、`provenance.json` 保存原始运行路径和源码身份。CaiT 最大方差主对照采用 class-attention 后、最终 LayerNorm 前 CLS 的后续测量。

固定 block10 的两组 token 策略实验见 [证据说明](gdi_token_selection_evidence.md)。原始图像、预测及 selection traces 仍在相应服务器目录。历史频次、比例、增强和视图预算按原记录解释。

## 命名与归档维护

当前文档、结果目录、报告标题和内部链接统一使用 GDI。原始测量 JSON、CSV、来源 manifest 及源码差异按原字节保存，其执行标识可用于定位服务器原件。归档目录迁移与 README 更新记录在各归档的 `provenance.json`，包含维护前文件哈希和刷新后的完整文件清单。

当前代码接口为 `GDIAttacker`、`GDIAdapter`、`GDIState` 与 `begin_gdi_forward` / `advance_gdi_state` / `finish_gdi_forward`；关闭 drop 使用 `--no-gdi-drop`。

## 论文证据准备

按原有章节结构推进，以有效干预层的迁移收益和逐样本定位能力为中心。优先准备 Fig. 1 的几何曲线及对应干预 ASR、最终 i.i.d. 设置的选层对照矩阵、GDI 方法图和同协议外部比较。具体任务及材料状态见 blueprint。

`data/huggingface/`、`data/torch/` 全部权重保留，包含 PiT-B、Visformer 等 FPR13 目标。历史强固定层基线、固定双／多边界路径及完成的实验 runner 按已有维护决定保留清理状态。
