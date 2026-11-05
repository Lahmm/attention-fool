# GDI 材料：历史 ViT token 选择证据

更新日期：2026-11-05。该材料用于 §4.4 的 token 干预设置分析。六组测量固定在 ViT-B/16 的 block10 边界，每图 20/196 个局部 token，seed 42、10 steps、epsilon=16/255、opponent β=0.2，原始／反射平移两个视图，用于固定边界条件下的层内 token 策略比较。

high/low 的分数来自干净图像该边界局部表示与最终归一化 CLS 的带符号余弦。high-random/low-random 在各自 98 个位置的半池无放回抽样，random 在全部位置抽样；fix 表示原始 mask 全部步／组复用，其他 case 按 step/group 重采样。相位 mask 始终按空间对应变换，保持数量。

| Case | 10 groups strict ASR | 1 group strict ASR |
| --- | ---: | ---: |
| high-fix | 86.40% | 75.07% |
| high-random | 87.28% | 75.12% |
| low-fix | 87.06% | 74.93% |
| low-random | 87.19% | 75.23% |
| random-fix | 87.11% | 75.35% |
| random | 87.24% | 75.23% |

原件位于 `outputs/attack/vit_drop_selection_s42_20261029T125456Z/` 和 `outputs/attack/vit_drop_selection_g1v2_s42_20261101T131251Z/`。各目录保留 `experiment.json`、`asr_comparison.csv`、`asr_per_target.csv`，各 case 保存攻击参数、FPR13 结果、对抗图与 `selection_traces/`。六组使用相同 batch/RNG 起点和随机消耗形状，按原记录解释匹配设置。

random 与 random-fix 是同一候选池内的重采样对照；high-fix 与 high-random 同时改变极端 top-k 与半池采样覆盖。直接差值以原始 CSV 为准。该材料按固定边界、实际增强和视图预算组织 ASR 比较。
