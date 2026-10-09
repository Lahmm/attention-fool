# PRD 写作交接与服务器文件清单

服务器文件核对日期：2026-10-08，当时分支为 `patch-score-routing-attack`；本地论文写作分支为 `prd`。此文档用于在另一台机器上写作；相对路径均以仓库根目录为起点。服务器上未纳入 Git 的逐文件索引见 [`results/server_untracked_files.tsv.gz`](../results/server_untracked_files.tsv.gz)：143110 个文件或符号链接，列为路径、类型、逻辑大小和用途类别。图像文件数量大，以下按同质目录说明；索引保留了每一个实际文件名。

## 论文故事与材料分工

**观察 → 方法 → 验证**：同一个局部 token 与全局表示的余弦相似度会随网络深度发生明显变化，高 patch-score 的位置随之迁移与更替。样本可视化呈现这一动态发现。PRD 沿着表示演化的过程，在多个 checkpoint 随机 hard-zero 局部 token，并用 RGB opponent-channel noise 扰动保留证据。每个 step/group 重新抽取 schedule，每个 checkpoint 从当前原生 local-token 网格的全部位置独立均匀抽样，允许跨 checkpoint 重复。相位视图使用原始 schedule 的空间变换。噪声在初始 RGB 特征处注入并按特征 RMS 匹配，随后顺序执行 checkpoint 干预。20 视图平均、Gaussian residual、动量和 `L∞` 投影更新组成完整运行流程；迁移效果通过 ASR 呈现。

动机的原始统计文件已提取为 [`results/prd_run_artifacts/patch_rank_observation_summary.json`](../results/prd_run_artifacts/patch_rank_observation_summary.json)：64 张图、offset 628、统一 7×7 网格，ViT/CaiT/PiT/Visformer 的早晚层 Spearman 约为 −0.071/0.182/0.063/0.12，作为动态发现的辅助统计。原数据来自 Git 历史 `d1ac809:outputs/research/patch_score_promotion_e1_e2/summary.json`。写作时优先准备真实样本的多层 patch-score 图和 token 分数变化曲线。

## 正式实验的可复核记录

四个正式运行均使用 seed `20260907`、1000 张相同输入、`epsilon=16/255`、10 steps、step size `epsilon/10`、动量衰减 1、每步 10 groups × 2 views、相位平移集合 `(4,4),(8,8),(12,12)`、Gaussian `sigma=4, alpha=0.75`。生成命令逐字见 [`run_prd_1000s.sh`](run_prd_1000s.sh)；实际记录的全量参数见 `results/prd_run_artifacts/{vit,cait,pit,visformer}/attack_params.json`。四次原始 transfer CSV 的 `git_head` 均为 `98cb7cf1`，执行时间和评估参数在相应 CSV 中。此处版本号描述的是**实际运行时提交**，论文实现说明须同时参照现仓库代码。

| Source | Checkpoints | 原生网格 | 每站 drop 数 | Opponent strength | 攻击 batch size |
| --- | --- | --- | --- | ---: | ---: |
| ViT-B/16 | `block3,block10` | 14×14, 14×14 | 10, 10 | 0.2 | 96 |
| CaiT-S24 | `block17,block23` | 14×14, 14×14 | 2, 28 | 0.2 | 48 |
| PiT-B | `stage2_block1,stage3_block2,stage3_block3` | 16×16, 8×8, 8×8 | 5, 2, 6 | 0.4 | 96 |
| Visformer-S | `stage2_block1,stage3_block1` | 14×14, 7×7 | 41, 10 | 0.4 | 48 |

默认每张图像生成 100 条新 schedule；两个 checkpoint 的 source 进行 200 次 mask selection，PiT 进行 300 次。四份 `replay_manifest.json.gz` 为原始 JSON 的无损压缩，含 1000 个同序 sample ID、随机事件摘要及 100000 个 phase events；各 manifest 的事件数为 ViT/CaiT/Visformer 500000、PiT 600000。`gradient_diagnostics.json` 保留了有效秩及其他梯度诊断原值。四个对抗图像目录各有 1000 张 `adv_*.png` 和 3 个 metadata 文件；图像的逐文件 SHA256 见 [`results/prd1000_image_sha256.tsv.gz`](../results/prd1000_image_sha256.tsv.gz)。

迁移目标为 8 个 Transformer 和 6 个 CNN；`transfer_eval.py` 依据各目标模型自己的 timm transform 评估。ASR = `1 - correct/total`，`total` 是全部对抗图像，不以目标模型干净预测正确为前提。56 组 source-target 结果各为 `total=1000, skipped=0`。原始记录是 `outputs/csv/outputs_attack_prd1000_{vit,cait,pit,visformer}_seed20260907.csv`；逐目标规范化表是 [`results/prd_cross_arch_s1000.csv`](../results/prd_cross_arch_s1000.csv)；梯度摘要是 [`results/prd_gradient_diagnostics_s1000.csv`](../results/prd_gradient_diagnostics_s1000.csv)。总体均值为 ViT 85.20%、CaiT 86.58%、PiT 84.36%、Visformer 80.65%；四源均值 84.20%。排除 source 同架构 target 后，四源严格黑盒均值为 83.10%。各目标结果见 [`progressive_cross_arch_mainline_s1000.md`](progressive_cross_arch_mainline_s1000.md)。

上述数字从 56 行规范化 CSV 重新计算，与四份原始 CSV 的 `avg`、`avg_vit`、`avg_cnn` 一致。四份 replay manifest 均有 1000 个唯一 sample ID，且 `sample_ids_sha256` 同为 `baf340386caf885fcb99298bb89c5c853622a7d149e7e79297437aefb015703c`。梯度诊断文件作为内部实验分析记录保存，论文的迁移性表征统一使用 ASR。

## Git 内文件用途

| 文件或同质文件组 | 作用 |
| --- | --- |
| `.gitignore` | Git 排除缓存、权重及大批实验产物的规则；本次小型证据文件显式加入。 |
| `AGENTS.md` | 当前研究主线、保留范围及仓库操作约束。 |
| `README.md` | PRD 的快速入口、主要配置和结果概览。 |
| `requirements.txt` | Python 依赖；版本约束并非完整环境锁文件。 |
| `data/clean_resized_images/*.png`（1000 个） | 正式攻击输入图像；全部已由 Git 跟踪。 |
| `data/image_name_to_class_id_and_name.json` | 图像名到 ImageNet 类别 ID 和名称的标注映射。 |
| `main.py` | 攻击 CLI、数据与模型加载、对抗图像和 metadata 保存。 |
| `progressive_attack.py` | PRD schedule、顺序 mask、相位配对、opponent noise、梯度处理及投影更新。 |
| `gradient_replay.py` | seed 派生的随机事件与 replay manifest 记录。 |
| `utils.py` | 输入数据集、标准化、对抗图像保存和运行设备。 |
| `nets/__init__.py`, `nets/base.py` | adapter 导出、共同的 PRD traversal 契约与 RGB projection 信息。 |
| `nets/vit.py`, `nets/cait.py`, `nets/pit.py`, `nets/visformer.py` | 四个 source 架构各自的原生 checkpoint 遍历与 local-token hard-zero。 |
| `transfer_eval.py` | 14 个目标模型的对抗准确率/ASR 评估及逐目标 transform。 |
| `record_experiment.py` | 将迁移结果和实际参数写为原始 CSV。 |
| `experiments/run_prd_1000s.sh` | 四次正式生成与迁移评估的完整命令。 |
| `experiments/prd_paper_story.md` | 论文叙事、写作结构和表达约定。 |
| `experiments/cvpr2027_writing_blueprint.md` | 论文整体纲要、图表计划和写作顺序。 |
| `experiments/progressive_cross_arch_mainline_s1000.md` | 正式主实验协议和 4×14 逐目标表。 |
| `experiments/server_handoff_inventory.md` | 本文件：服务器证据核对和异地写作交接。 |
| `outputs/csv/outputs_attack_prd1000_*.csv`（4 个） | 每个 source 的原始迁移结果、运行时间、实际参数和目标 ASR。 |
| `results/README.md` | 结果文件入口说明。 |
| `results/prd_cross_arch_s1000.csv` | 56 条正式 source-target 结果及评估数量。 |
| `results/prd_gradient_diagnostics_s1000.csv` | 内部实验分析记录及 schedule 数。 |
| `results/prd_run_artifacts/{vit,cait,pit,visformer}/attack_params.json`（4 个） | 正式运行的完整攻击参数。 |
| `results/prd_run_artifacts/{vit,cait,pit,visformer}/gradient_diagnostics.json`（4 个） | 内部实验分析使用的梯度诊断原值。 |
| `results/prd_run_artifacts/{vit,cait,pit,visformer}/replay_manifest.json.gz`（4 个） | 完整的随机事件 replay manifest，无损压缩。 |
| `results/prd_run_artifacts/patch_rank_observation_summary.json` | 动机观察的历史统计快照，不参与攻击。 |
| `results/prd1000_image_sha256.tsv.gz` | 4000 张正式对抗图像的路径、字节数、SHA256。 |
| `results/server_untracked_files.tsv.gz` | 服务器未入 Git 文件的逐项路径和用途索引。 |
| `tests/test_main.py`, `tests/test_progressive_attack.py`, `tests/test_progressive_vit.py`, `tests/test_progressive_adapters.py`, `tests/test_gradient_replay_manifest.py`, `tests/test_transfer_eval.py` | 分别验证 CLI、PRD 核心、ViT schedule、四架构 adapter、replay 和迁移 ASR。 |

## 服务器未入 Git 的文件：下载清单

在另一台机器上**只写论文**，先 clone 本分支即可取得输入图像、实现、正式参数、manifest、统计和原始 CSV。若要展示/复核具体对抗图像，另下载下面第一项。若要在本机精确重跑/复评，还需要第二项。其余为探索或历史产物，只有计划复查对应实验时才下载。路径可用 `rsync -a` 或打包工具按目录传输；压缩索引可用 `gzip -dc results/server_untracked_files.tsv.gz` 查看完整文件名。

| 路径 | 数量及大小（逻辑字节） | 用途与下载建议 |
| --- | ---: | --- |
| `outputs/attack/prd1000_{vit,cait,pit,visformer}_seed20260907/` | 4×1000 图，含 metadata 共 4012 文件、516165960 B | **优先下载**：正式对抗图像原件。三类 metadata 已无损复制到 Git；图像可按 SHA256 清单验收。 |
| `data/huggingface/` | 79 文件/链接，实际文件约 4652717582 B | **复现时下载**：source 和 target 的离线 timm/HF 权重及 cache；保留符号链接结构。仅写作可暂不下载。 |
| `outputs/attack/` 中除四个 `prd1000_*` 外的目录 | 325 个实验目录、138482 文件；约 17.69 GB | 历史探索的图像与 metadata，包含旧 patch-score/selector 路径，供回查研究过程。按逐文件清单择需打包。 |
| `outputs/csv/` 中除四个 `prd1000_*` 外的 CSV | 294 个 | 探索实验结果；正式四个 CSV 已由 Git 跟踪。按逐文件清单择需下载。 |
| `outputs/logs/` | 210 个 | 探索运行日志及控制器输出；按逐文件清单择需下载。 |
| `outputs/reports/` | 4 个 | 旧 selector 研究报告及 smoke 预测文件，非 PRD 主线。 |
| `outputs/prd1000_background.log` | 1 个，67855 B | 四次正式运行的终端进度和逐目标结果；数字已有 CSV，若需运行过程日志可下载。 |
| `__pycache__/`, `experiments/__pycache__/`, `nets/__pycache__/`, `tests/__pycache__/` | 28 个 | Python 编译缓存，无需下载。 |

如需全部服务器实验历史，建议整体打包 `outputs/`。Git 的现有代码已清理至 PRD 主线，历史实验的回查依据其对应版本和记录。服务器的 `data/huggingface/` 是模型缓存；权重的许可与来源由对应 timm/HF 模型负责。正式对抗图像原件保存在服务器，可按上述下载清单取得。

验收下载的正式图像示例：

```bash
gzip -dc results/prd1000_image_sha256.tsv.gz | awk -F '\t' 'NR>1 {print $3 "  " $1}' | sha256sum --check
```

在仓库根目录执行；该命令需要四个 `outputs/attack/prd1000_*` 目录均已放回原相对路径。解读压缩的 replay manifest：`gzip -dc results/prd_run_artifacts/vit/replay_manifest.json.gz`。
