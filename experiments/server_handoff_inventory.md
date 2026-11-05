# 当前动态 PRD 的文件与服务器材料

更新日期：2026-11-05，分支 `prd`。可执行攻击只有方向方差比 × 正收缩比例逐图选层与单边界随机 token drop；旧固定 checkpoint 攻击及相关结果已清理。

## 受 Git 管理

| 内容 | 路径 |
| --- | --- |
| 攻击入口与优化 | `main.py`、`progressive_attack.py`、`utils.py` |
| 四源适配与几何统计 | `nets/__init__.py`、`base.py`、`vit.py`、`cait.py`、`pit.py` |
| FPR13 与权重下载 | `transfer_eval.py`、`experiments/download_fpr_models.py` |
| 验证 | `tests/test_{dynamic_checkpoints,progressive_attack,progressive_adapters,main,transfer_eval}.py` |
| 观察素材 | `figure1.py`、`tests/test_figure1.py` |
| 方法、协议、写作 | `prd_method_contract.md`、`fpr_target_protocol.md`、`prd_paper_story.md`、`cvpr2027_writing_blueprint.md` |
| 当前 ASR 与历史选层证据 | `results/README.md` 索引的五个归档 |
| 输入与标注 | `data/clean_resized_images/` 的 1000 张 PNG、`data/image_name_to_class_id_and_name.json` |

虽然 `.gitignore` 包含 `data/*`，上述输入图与标签实际上已经被 Git 跟踪。权重缓存及 `outputs/` 未跟踪。

## 当前执行流程

2026-11-05 起，每步默认 20 个独立视图，每图每视图 50% 原始、50% 随机 phase。每个视图独立抽噪声和 native-grid mask，每图十步 200 次 mask 抽样。选层与更新不变。新流程尚无千图 ASR。

## 当前测量原件

四个 `outputs/attack/prd_direction_contraction_s42_20261104T044138Z_{vit,cait,pit_t,deit_b}/` 各保留 1000 张对抗图、`attack_params.json`、`transfer.csv/json` 与 `predictions.json`。对应旧配对视图 Overall 89.1673%、strict 88.7234%，实际增强是 opponent；不能重标为新独立视图结果。攻击 batch 分别为 48、16、32、48；评估 batch 32、关闭 TF32、无 AMP。

原始来源为 `/root/autodl-tmp/prd_jobs/direction_contraction_s42_20261104T044138Z/`，保留样本 manifest、执行源码快照、验证、日志及来源。仓库中的 `source_changes.patch`、执行 commit 和源码哈希描述测量当时状态，不是额外执行入口。

## 历史选层原件

`outputs/attack/` 保留：

- `prd_dynamic_variance_s42_20261029T031203Z_{vit,cait,pit_t,deit_b}/`；CaiT 主对照使用后续 `prd_cait_preln_variance_s42_20261029T070204Z/`。
- `prd_second_raw_moment_s42_20261101T151302Z_{vit,cait,pit_t,deit_b}/`。
- `prd_std_abs_mean_s42_20261102T081807Z_{vit,cait,pit_t,deit_b}/`。
- `prd_normalized_std_abs_mean_s42_20261102T142706Z_{vit,cait,pit_t,deit_b}/`。
- `prd_direction_variance_ratio_s42_20261103T090754Z_{vit,cait,pit_t,deit_b}/`。
- 两组 `vit_drop_selection_*` 的 token 策略记录，见 [证据说明](prd_token_selection_evidence.md)。

这些都是已执行配置的证据，不恢复历史攻击选项。

## 权重、环境与待准备内容

`data/huggingface/` 与 `data/torch/` 全部保留，包括旧目标权重；PiT-B 与 Visformer 是 FPR13 目标。当前环境为 Python 3.10.19、torch 2.10.0、torchvision 0.25.0、timm 1.0.24，服务器解释器 `/root/miniconda3/envs/att-atk/bin/python`。

最终动态 i.i.d. 主表、匹配无 drop／随机层／历史方差／当前选择器的对照仍需实际配置支持。`figure1.py` 保留跨层原始分数导出；完整方向比／收缩曲线与当前方法 Fig. 2 待准备。旧固定双边界 Fig. 2 代码已删除。

已完成的无 phase/i.i.d. ViT 对照位于 `outputs/attack/prd_vit_isotropic_g20v1_s42_1000_20261105T084552Z/`，实际为 20×1、1000 图，Overall 81.3231%、strict 79.7667%；它不是新 identity/phase 混合采样的测量。
