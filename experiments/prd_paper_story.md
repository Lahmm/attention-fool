# 论文故事与 outline：方向方差比与正收缩比例指导选层和随机 token drop

更新日期：2026-11-04，依据用户确认的写作重心调整。PRD 暂作为仓库工作名称；论文方法名与标题待讨论确定。本文档是写作规划，论文自然段仍仅在对话中交付。

## 1. 中心叙事

局部—全局关联随深度变化，block 前后的局部方向分布也会改变。当前唯一主线用输出方向方差比乘正收缩比例确定每图的干预边界，再在该 block 输入处均匀随机清零局部 token。几何聚拢是观测量，不直接等同于信息丢失、语义整合或分类决定性；有效性通过实际迁移 ASR 检验。

核心链条：**表示随深度变化 → 干预位置问题 → 全局相关差异与正收缩的乘积分数 → 输入边界随机 drop → ASR 验证**。旧最大方差规则成为历史对照。共同随机增强和优化设置保留真实执行标签。

## 2. 五个论证节点

| 节点 | 读者需要理解的内容 | 证据或方法内容 |
| --- | --- | --- |
| 观察表示演化 | 同一位置的局部 token 与全局表示的关联沿深度变化，高分区域随之更替 | 实际样本的跨层 patch-score 图，固定位置曲线可作辅助 |
| 提出干预深度问题 | 各层具有不同的局部关联分布，需要确定随机干预的位置 | 将跨层观察连接到方向方差比与输入／输出分散度 |
| 定义统计选层 | 方向方差比乘正收缩比例是唯一定位规则 | 归一化几何、方向比、正收缩比例、最高分 block 与输入边界 |
| 实施随机干预 | 在定位边界从全部局部位置均匀抽样并 hard-zero，再继续前向 | 每个 step/group 重采样位置，空间配对共享对应 mask |
| 验证设计 | 选层规则、层内策略和完整攻击分别取得什么 ASR | 匹配设置的选层表、token 选择表、四源 FPR13 主表 |

同一 token 跨层变化和同一层不同 token 的分数差异是两个不同的观察维度。前者建立动态背景，后者是方向比的组成部分；当前定位还使用 block 输入／输出分散度。方差表示局部—全局关联的分散程度；关于选层效果的结论由 ASR 对照给出。

## 3. 方法事实与算法骨架

1. 每张干净图像在攻击前做一次无 drop、无增强的完整前向，采集各 block 的局部输入和输出。
2. 逐 token 与参考分别 L2 归一化。用输出全局方向总体方差除以输出分散度／原生维数，再乘正收缩比例 `max(0, 1 - 输出分散度 / 输入分散度)`。输入或输出分散度为零时分数为零；并列选原生最早者。
3. 在得分最高 block 的输入处干预。每张图只选层一次；全部攻击步与视图保持边界固定，PiT pooling 不计入 block 收缩。
4. 每个 step/group 从选定边界的全部局部位置均匀、无放回抽取指定数量，清零所选 token 的全部通道。token 数量与原生网格保持不变，模型继续同一次前向。
5. 默认原始／反射平移两个视图；平移视图使用原始 mask 的空间对应变换，按原生网格 occupancy top-k 保持 drop 数量。
6. 平均各视图的像素梯度，直接累积动量，执行 sign 更新、相对干净输入的 L∞ 投影与像素截断。

“动态”指根据输入计算边界；位置在攻击期间重新采样。默认 seed 42、10 steps、10 groups×2 views，每张图有 100 次原始 mask 抽样，每个视图只在该图选定的一个边界 drop。同批样本可以选取不同边界，整批遍历它们的并集时仅干预对应样本。

当前四源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled。总 drop 比例分别为 20/196、30/196、30/196、20/196；多尺度模型在所选原生网格计算实际数量。PiT-B 与 Visformer 保留为 FPR13 目标；可执行源只有这四个动态适配器。架构参照、取整和执行细节见 [方法契约](prd_method_contract.md)。

## 4. 增强设置与贡献组织

采用 i.i.d. RGB 采样经初始投影映射并 RMS 匹配的随机特征增强。i.i.d. 指投影前 RGB 采样；投影后的特征可以相关。噪声注入覆盖全部初始局部位置，prefix 保持不变。具体构造、强度与注入位置放入补充材料的共同实现设置。

摘要、引言、贡献、结论、核心流程图不突出噪声；方法章不设置独立噪声小节。主文围绕选层与随机 drop 组织，补充材料如实记录共同增强。增强的实际标签保留在参数和来源记录中。

共同增强的论文规划与选层迁移分别记录。命令行和攻击类仅支持直接 i.i.d. RGB 标准高斯采样；opponent 已退出执行与消融。已有 opponent 测量保留原标签；四源 i.i.d. 配对测量已完成，原件见交接记录。

贡献建议为两点：

1. 提出方向方差比与正收缩比例指导的选层与随机 token 干预：从一次干净前向确定输入对应边界，在该边界均匀随机 hard-zero 局部 token，用于迁移攻击优化。
2. 通过四源 FPR13 迁移 ASR、选层对照、token 选择与预算分析，验证完整方法和关键设计。

## 5. 正文章节与图表

正文采用 Introduction、Related Work、Method、Experiments、Conclusion。方法章为 3.1 Preliminaries、3.2 Depth-wise Local–Global Alignment、3.3 Direction Variance and Contraction-Guided Layer Selection and Random Token Drop、3.4 Attack Optimization。

实验章为 4.1 Experimental Setup、4.2 Main Transfer Results、4.3 Layer Selection、4.4 Token Dropping and Additional Analysis。完整章节任务、算法、图表和写作顺序见 [写作蓝图](cvpr2027_writing_blueprint.md)。

- Fig. 1：真实样本的跨层 patch-score 图、原始分数方差诊断、方向比与收缩及乘积分数曲线、最高分 block 与输入边界。热图层内归一化只用于显示，方差使用原始分数。
- Fig. 2：一次干净选层与迭代攻击双支路；突出边界固定、随机 mask、空间配对、单边界 hard-zero 与梯度更新。
- Table 1：最终方法的四源迁移主结果与同协议比较。
- Table 2：匹配增强、drop 比例和优化预算的选层对照。
- Table 3：层内 token 选择与固定／重采样策略对照。
- Fig. 3（可选）：drop 比例或 ASR—计算预算关系。架构全配置与选层频次放入补充材料。仓库仅提供当前动态单边界方法。

## 6. 现有证据与待准备内容

同一选择器在历史独立快照中完成的 opponent/1000 图 FPR13 结果如下，2026-11-04 迁入主线；不重标为 i.i.d.。参数与源码来源保留原执行状态，见 [归档](../results/prd_direction_contraction_s42_1000_fpr/README.md)。完整选层频次保留在 selection_counts.csv。

| Source | Overall ASR | Strict ASR |
| --- | ---: | ---: |
| vit | 87.95% | 86.95% |
| cait | 89.14% | 88.37% |
| pit_t | 84.28% | 84.28% |
| deit_b | 95.30% | 95.30% |
| macro | 89.17% | 88.72% |


以下旧最大方差对照已在本地核实四源动态选层、每源 1000 图像、FPR13 的 13 目标评估；这些运行仍使用 opponent 增强。采用当前参照定义的记录为：

| Source | Overall ASR | Strict black-box ASR | 本地原始运行目录（`outputs/attack/` 下） |
| --- | ---: | ---: | --- |
| ViT-B/16 | 88.13% | 87.15% | `prd_dynamic_variance_s42_20261029T031203Z_vit/` |
| CaiT-S24 | 83.22% | 82.34% | `prd_cait_preln_variance_s42_20261029T070204Z/` |
| PiT-T | 83.60% | 83.60% | `prd_dynamic_variance_s42_20261029T031203Z_pit_t/` |
| DeiT-B distilled | 94.38% | 94.38% | `prd_dynamic_variance_s42_20261029T031203Z_deit_b/` |

各目录包含 `attack_params.json`、`transfer.csv`、`transfer.json`。CaiT 使用 class-attention 后、最终 LayerNorm 前 CLS 的后续运行；早期 CaiT 参照不同的运行另行保留。原始目录不随 Git 分发。

历史最大方差 ViT 记录中 996/1000 张图选择 `block10`；选层分布按真实频次展示。十组视图下，六组 token 选择实验的 high-random/low-random/random strict ASR 为 87.28%/87.19%/87.24%，random-fix 为 87.11%；一组视图下 random/random-fix 为 75.23%/75.35%。这些固定 `block10` 的实验支持简洁随机策略的使用，重采样的作用按直接差值呈现。原始表位于 `outputs/attack/vit_drop_selection_{s42_20261029T125456Z,g1v2_s42_20261101T131251Z}/asr_comparison.csv`。

优先准备最终 i.i.d. 配置的四源结果、匹配设置的随机选层／历史方差选层／当前乘积选层／无 drop 对照，以及 Fig. 1 的原始方差曲线。当前单边界 Fig. 2 需另行准备。旧固定双／多 checkpoint 执行路径、结果与消融以及强固定单层基线已按用户要求删除；其他历史选层数据保留。

ASR 定义为全部评估对抗样本上的 `1 - adversarial accuracy`。严格黑盒排除完全相同的源目标；四源等权汇总。所有机制结论使用 ASR 与直接 ASR 差值，梯度诊断留在内部分析。

## 7. 表达与写作约定

围绕观察、选层、随机干预和 ASR 写作。正文不再沿用“双核心机制”或“多 checkpoint 顺序中断”的方法描述。保持直接、鲜明、肯定的语言，以真实样本和数值推进论证。

每次论文交付仍为一个英文自然段及对应中文翻译，讨论确认后继续。图表占位、公式呈现与本地文件操作遵循 `AGENTS.md`。
