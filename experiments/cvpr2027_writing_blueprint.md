# PRD：CVPR 2027 论文写作蓝图

核对及表达约定更新日期：2026-10-09。当前本地分支：`prd`；最初阅读基准：`2f2213d`。四次正式运行记录的实现版本：`98cb7cf1`。后续提交归档正式结果与写作材料。

## 1. 项目目标与本轮工作内容

项目的主要任务是：依据本地 PRD 分支的实现与实验材料，以已有四源、1000 图像、14 目标结果作为主实验，撰写面向 CVPR 2027 投稿的论文。

本对话承担写作前的整体理解、论证组织、章节设计与写作顺序规划。已有主实验是论文的事实基础；必要的补充比较和机制验证围绕这一基础开展。方法保持两个核心机制：checkpoint-wise progressive random token drop 与 RGB opponent-channel random noise。

建议工作标题：**Progressive Route Disruption for Transferable Adversarial Attacks**。

写作约定：直接讲发现、设计与结果，用鲜明的动态过程吸引读者。动机以同一样本的 token 分数变化和高 patch-score 区域的迁移为中心；ASR 呈现迁移效果。梯度诊断保留为内部分析工具。省去针对未提出主张的反复澄清，让每一段推进论文主线。长期约定已记录在 `AGENTS.md` 的 Paper-writing objective and voice。

## 2. 已阅读材料与证据层级

| 材料 | 用途 | 写作地位 |
| --- | --- | --- |
| `AGENTS.md`、`README.md` | 确认研究范围、当前配置与术语 | 当前主线约束 |
| `progressive_attack.py`、`main.py`、`nets/*.py` | 核对 schedule、前向遍历、噪声、更新与架构适配 | 方法事实依据 |
| `utils.py`、`transfer_eval.py`、`record_experiment.py`、`gradient_replay.py` | 核对数据、预处理、结果记录、ASR 与随机重放 | 实验协议依据 |
| `tests/*.py` | 阅读已有实现验证及 optional real-model 验证条件 | 行为契约；本轮未重新执行测试 |
| `experiments/prd_paper_story.md` | 论文故事与表达约定 | 叙事起点 |
| `experiments/progressive_cross_arch_mainline_s1000.md` | 正式配置、4×14 主表与验证记录 | 主实验说明 |
| 四份 `outputs/csv/outputs_attack_prd1000_*.csv` | 原始模型结果、实际参数、实现版本和时间戳 | 主实验原始记录 |
| `results/prd_cross_arch_s1000.csv` | 56 项 source-target 结果 | 制作主表的数值依据 |
| `results/prd_run_artifacts/*/attack_params.json`、`replay_manifest.json.gz` | 实际参数、同序样本与随机事件 | 可复核性依据 |
| `results/prd_run_artifacts/*/gradient_diagnostics.json` | 实验过程中的梯度诊断 | 内部分析记录 |
| `results/prd_run_artifacts/patch_rank_observation_summary.json` | 64 图像深度排名重组观察 | 动机证据 |
| `experiments/server_handoff_inventory.md`、两份压缩文件索引 | 区分本地已归档与仅在服务器上的材料 | 文件可用性依据 |
| Git 历史中的动机文档、排序观察程序、checkpoint/drop 扫描、192 图像 follow-up、旧主实验报告 | 理解研究如何收敛到当前 PRD | 研究过程背景 |

本轮独立复核了：56 项结果的均值；各项 ASR 与原始 CSV 一致；归档的完整参数与原始 CSV 中的参数一致；四份 manifest 含相同、同序的 1000 个唯一 sample ID；phase events 和随机事件数量符合配置。

本机已有 1000 张干净输入，但没有正式对抗图像目录和模型缓存。因此本轮核对的是代码与归档证据，未重新生成对抗样本、复评模型或测量已保存 PNG 的扰动。

交接文档记录的 `patch-score-routing-attack` 是服务器当时的分支名称；当前本地分支为 `prd`。主实验统一采用四份正式 `prd1000` 运行记录。

## 3. 研究故事：观察、设计问题、方法与验证

### 3.1 起点：高分 patch 随深度迁移，局部证据持续演化

随着同一张图像向网络深处传播，同一个局部 token 与全局表示的余弦相似度会发生明显变化，高 patch-score 的 patch 位置也会随层数变化而迁移与更替。局部证据参与全局表示形成的过程是动态的，这就是故事的出发点。

用同一样本的多层 patch-score 图展示高分区域的更替，再用若干固定位置 token 的分数曲线呈现其与全局表示关联的变化。先让读者看到这一过程，再提供辅助统计。

观察记录以 local/global 表示的余弦相似度定义 patch-score，global 表示依架构使用 CLS 或 GAP。已有统计覆盖 64 张图像和统一 7×7 网格；早晚层平均 Spearman 为 ViT −0.0708、CaiT 0.1817、PiT 0.0632、Visformer 0.1168。统计作为可视化发现的辅助材料，具体观察层对随图说明。

### 3.2 设计问题：如何干预持续演化的局部证据？

推荐的叙事连接：既然局部证据在前向传播中持续演化，攻击也可以沿着这一过程展开，在不同深度反复干预正在形成的表示。

PRD 将这一想法落实为逐 checkpoint 的随机局部中断：每一次中断都改变接下来的传播状态，后续中断继续作用于已经演化的表示。重新采样让局部干预在不同 step、group 和深度上展开。

### 3.3 核心机制一：逐 checkpoint 的随机局部中断

在每个攻击 step、每个 augmentation group 上重新抽取 schedule。每个 checkpoint 从其当前原生网格的全部 local-token 位置中，均匀、无放回抽取指定数量。不同 checkpoint 独立采样，允许跨 checkpoint 重复位置。

每次到达 checkpoint 后立即将选中 token 的所有通道 hard-zero，然后沿已改变的状态继续同一次前向传播。token 数量和网格保持不变；此前归零的位置在后续计算中可能重新产生非零表示。

Route 指局部表示沿后续网络计算传播的轨迹。Progressive 强调每一次中断之后，模型继续在已经改变的状态上前向传播。

### 3.4 核心机制二：扰动保留证据的 RGB 对手通道投影噪声

已选 schedule 决定各 checkpoint 的空间 drop 区域并集。噪声从亮度、红绿、黄蓝三个 RGB 方向采样，通过源模型真实的初始 RGB 卷积权重投影，并根据初始特征 RMS 匹配尺度；注入符合保留区域规则的初始局部特征。

三个方向的采样方差为 0.5、1.25、1.25，为扰动赋予不同的亮度与色度能量。初始 RGB 投影把这一颜色结构映射到源模型的特征空间，RMS 匹配控制扰动强度。

两项机制形成清楚的分工：progressive drop 中断局部证据的传播，opponent noise 扰动保留证据的表达。消融实验通过 ASR 呈现各项机制与组合的效果。

### 3.5 可用于后续引言的中心论点

随着图像向网络深处传播，同一个局部 token 与全局表示的关联会发生明显变化，高 patch-score 的区域也不断更替。PRD 沿着这一演化过程，在多个 checkpoint 反复随机中断局部表示的传播，同时用 RGB 对手通道噪声扰动保留证据。四种源架构到 14 个目标模型的 ASR 结果呈现完整方法的迁移效果。

## 4. 方法必须与实现一致

### 4.1 完整流程

1. 从当前对抗像素开始，为本 step/group 构建 fresh schedule；构建过程遍历 checkpoint 并立即应用 mask，不计算 patch-score。
2. 由原始 schedule 得到各 checkpoint mask 的图像空间并集；原始视图使用此 schedule。
3. 用反射填充的像素平移构造 phase view，把同一 schedule 的 mask 同步变换；映回各 checkpoint 原生网格时按 occupancy top-k 保持 drop 数量。
4. 对每个视图，在初始 RGB 特征处注入该视图独立抽取的 opponent noise；然后顺序到达 checkpoint、hard-zero、继续前向并计算真实标签 CE loss。
5. 对两个视图的损失分别求关于当前像素的梯度；10 groups 提供 20 个梯度，直接取平均。
6. 添加 Gaussian 平滑梯度残差；进行原始梯度的动量累积、sign 更新、相对干净输入的 L∞ 投影与 [0,1] 截断。
7. 重复 10 steps，保存最终像素对抗图像；目标模型评估使用各自 timm transform。

流程图按实际执行顺序呈现：先采样 checkpoint mask，在可微前向的初始特征处注入噪声，随后顺序执行 progressive traversal。phase view 使用轻微像素平移。

### 4.2 推荐的最小公式集合

无目标攻击使用真实标签交叉熵，目标约束是 `||x_adv − x||∞ ≤ 16/255`。

对于 step t、group g、checkpoint c，设原生网格有 N_c 个位置，drop 数量为 k_c：

`S_(t,g,c) ~ Uniform({S ⊂ [N_c] : |S| = k_c})`。

checkpoint 干预写为：

`H_c^+ = (1 − M_(t,g,c)) ⊙ H_c^-`，其中 mask 沿通道广播，后续网络从 `H_c^+` 继续。

对手通道基为：

`q_L=(1,1,1)/√3`，`q_RG=(1,−1,0)/√2`，`q_YB=(1,1,−2)/√6`。

每个初始 token 的卷积核位置采样独立标准 Gaussian 系数，用 `√0.5, √1.25, √1.25` 缩放后组合为 RGB 噪声，再由卷积权重 W 投影。写为：

`η = W ξ_RGB`，`η_hat = β · RMS(H_0) / RMS(η) · η`。

RMS 在单样本的全部初始 token 和通道上计算，并在保留区域门控之前匹配；实现中对特征 RMS 与投影权重使用 detach。

像素梯度与更新写为：

`g_t = (1/20) Σ_(g=1..10) Σ_(v=0..1) ∇_x CE(f_(schedule,noise,view)(x_t), y)`；

`g_t' = g_t + 0.75 · G_(σ=4) * g_t`；

`m_(t+1) = m_t + g_t'`；

`x_(t+1) = Clip_[0,1](x + Clip_[-ε,ε](x_t + (ε/10) sign(m_(t+1)) − x))`。

动量直接累积处理后的梯度，更新过程按上述递推式描述。

### 4.3 跨尺度保留区域的精确定义

`_image_mask_to_projection_drop_mask` 根据初始 RGB 卷积的 kernel、stride、padding、dilation 计算每个感受野的被 drop 区域覆盖比例，比例严格大于 0.5 才标为 initial drop。噪声施加在其补集。

在不重叠且对齐的 ViT patch 投影上，这对应 drop 并集之外的初始 token。重叠投影和跨尺度网格按上述感受野覆盖比例确定保留位置。

本节及前述 detach、卷积几何等细节作为实现备忘保存。正文以解释两项机制所需的定义和公式为主，完整实现细节按复现需要安排。

### 4.4 当前唯一主实验配置

| Source | Checkpoints | 原生网格 | Drop 数量 | Opponent β | 生成 batch size |
| --- | --- | --- | --- | ---: | ---: |
| ViT-B/16 | block3, block10 | 14×14, 14×14 | 10, 10 | 0.2 | 96 |
| CaiT-S24 | block17, block23 | 14×14, 14×14 | 2, 28 | 0.2 | 48 |
| PiT-B | stage2_block1, stage3_block2, stage3_block3 | 16×16, 8×8, 8×8 | 5, 2, 6 | 0.4 | 96 |
| Visformer-S | stage2_block1, stage3_block1 | 14×14, 7×7 | 41, 10 | 0.4 | 48 |

ViT 各 checkpoint 的 drop ratio 均为 `0.051020408163`。四种架构按原生网格和阶段设置 checkpoint 与 drop 数量。

共同配置：seed 20260907、10 steps、ε=16/255、步长 ε/10、动量系数 1、10 groups×2 views、平移集合 (4,4)/(8,8)/(12,12)、Gaussian σ=4/α=0.75。100 条 schedule/图像；ViT/CaiT/Visformer 200 次随机 checkpoint mask selection，PiT 300 次。phase masks 由共享 schedule 空间变换得到。

计算量记录包含 schedule 构建遍历和视图前向/反向；效率分析使用实测运行时间或完整计算量。

## 5. 现有结果与贡献组织

### 5.1 主实验结果

| Source | 14 目标 Overall | Transformer（8） | CNN（6） | 严格黑盒（13） |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 85.20% | 90.45% | 78.20% | 84.29% |
| CaiT-S24 | 86.58% | 90.06% | 81.93% | 85.66% |
| PiT-B | 84.36% | 89.91% | 76.95% | 83.27% |
| Visformer-S | 80.65% | 83.19% | 77.27% | 79.18% |
| 四源宏平均 | 84.20% | 88.40% | 78.59% | 83.10% |

推荐摘要与引言优先报告 **83.10% 的四源平均严格黑盒 ASR**。84.20% 的 Overall 含 source-matched target，保留在主表并明确标注。

Transformer 的 88.40% 同样含各源的同架构目标；若叙述“跨模型 Transformer 迁移”，使用去除 source-matched target 后的四源平均 **86.97%**。CNN 的 78.59% 均为跨架构迁移。

ASR = `1 − correct/total`，分母是全部评估的 1000 张对抗图像，不筛选 target-clean-correct 子集。因此 ASR 包含目标模型原本的分类错误；补报相同样本的 clean accuracy 有助于解释攻击增加了多少错误，但保持现有主指标定义。

四个 source 分别独立攻击相同的 1000 张输入，共生成 4000 张对抗图像；56 项评估对应 56000 次样本预测。strict black-box 指排除 source-matched target 后对其余 13 个目标取平均。

主表覆盖标准模型与两个对抗训练 CNN，按目标类型呈现迁移表现与差异。

### 5.2 论文评价与内部分析的分工

论文统一用 ASR 呈现迁移性。主实验、方法比较、机制消融和参数分析都围绕 ASR 组织。effective rank 等梯度诊断保存在内部实验记录，用于研究过程中的检查和分析；正文与补充材料的迁移论证均采用 ASR。

### 5.3 建议的贡献组织

1. **逐深度干预的攻击设计。** PRD 通过同一前向轨迹上的 fresh、uniform、checkpoint-wise hard-zero，干预演化中的局部证据；执行不依赖 patch-score。
2. **保留证据的结构化扰动。** RGB 对手方向的非各向同性噪声经真实初始 RGB 投影并匹配特征 RMS，与 progressive drop 形成职责分工。
3. **跨架构实现与实证。** 在固定网格、池化层级和卷积式前端上保留同一遍历原则；四源×14 目标的现有主实验提供可复核的迁移结果。

贡献段将动态发现、逐深度干预设计和 ASR 验证连接起来。四架构实例化呈现方法适用性。Related Work 围绕 PatchOut、深层 self-ensemble、token gradient regularization、surrogate/model augmentation 梳理具体差异。

## 6. 正文章节：每一节要解决什么问题

下面是适合本工作的建议叙事结构。

| 章节 | 目的与作用 | 本文应写的内容与产物 |
| --- | --- | --- |
| Title | 让读者立即知道方法与任务 | PRD + transferable adversarial attacks |
| Abstract | 用最短篇幅交代问题、方法、验证和意义 | 深度证据演化→两机制→四源/1000 图/14 目标→83.10% 严格黑盒；比较增益须等同协议结果 |
| 1. Introduction | 建立值得解决的问题，解释设计选择与贡献 | 迁移攻击场景；动态局部证据观察；干预演化轨迹的设计问题；两机制；贡献三点；开篇概念图 |
| 2. Related Work | 把创新放入最接近的已有研究 | 迁移优化/输入变换；token/patch attack；self-ensemble/model augmentation；特征噪声。每类说明与 PRD 的具体关系 |
| 3. Depth-wise Evidence Reorganization | 让读者看到动态发现，并自然进入方法 | 同一样本的多层 patch-score 图、固定 token 分数变化曲线、高分区域更替；附 score 定义与辅助统计 |
| 4. Progressive Route Disruption | 给出能复现、能比较的方法 | 4.1 threat model/notation；4.2 progressive random schedule；4.3 opponent noise；4.4 phase pairing/gradient/update；4.5 architecture-specific instantiation |
| 5. Experiments | 呈现完整方法与各机制的迁移效果 | 5.1 setup；5.2 现有主实验及方法比较；5.3 两机制的 ASR 消融；5.4 ASR 参数分析、效率与定性样本 |
| 6. Conclusion | 以发现、方法和结果收束全文 | 动态局部证据→progressive 干预与保留证据扰动→主要 ASR 结果 |
| Limitations / responsible use 段落 | 简洁交代方法适用范围与安全研究用途 | 参数选择、计算预算与目标范围；按篇幅及正式投稿要求安排 |
| References | 提供研究来源，明确方法定位 | 引用实际使用的相关文献，核对题名、版本与结论 |

实验子节应完整回答：

- **5.1 Experimental Setup：** 哪些图像、标签和权重，攻击预算、source/target、预处理、seed、metric 与参数选择过程是什么？
- **5.2 Main Transfer Results：** 在现有同协议主表中表现如何？去除 source-matched target 后是否仍有迁移？与可比方法相比如何？
- **5.3 Component Ablations：** progressive 与 opponent 各自贡献什么？组合是否优于各单项？收益能否由已有支撑流程解释？
- **5.4 Analysis：** checkpoint/drop budget/噪声强度怎样影响 ASR，计算代价如何，成功与失败样本呈现什么特点？

页数紧张时，将 §3 压缩为 Introduction 的观察段与 Method 的动机子节，或将 §5.4 的较长机制分析单列。必须保留“观察→方法→验证”的逻辑关系。

## 7. 补充材料及图表计划

### 7.1 补充材料目录

| 部分 | 作用 |
| --- | --- |
| A. Complete implementation and algorithm | 完整伪代码、mask 变换、RMS/stop-gradient、动量、投影与 PNG 保存细节 |
| B. Architecture adapters | 四源原生阶段、初始投影几何、CLS/GAP/卷积前端、跨尺度 mask 映射 |
| C. Reproducibility and evaluation protocol | 样本 ID、权重版本、环境、seed、运行参数、原始表、指标定义与参数选择过程 |
| D. Extended motivating observation | score 定义、层对、空间对齐、更多真实样本的分数图与变化曲线 |
| E. Extended results and sensitivity | 逐目标结果、补充 seed/参数扫描/基线与消融，标注其与当前配置的对应关系 |
| F. Qualitative cases and computational cost | 成功/失败案例、可视化与实测计算代价 |

正文保留主要 ASR 比较和机制消融，补充材料展开复现细节与扩展结果。

### 7.2 图表与证据准备

| 图表 | 要回答的问题 | 当前材料状态 |
| --- | --- | --- |
| Fig. 1 概念与方法总览 | 为什么多深度中断？两个机制在哪发生？ | 可依据代码绘制；注意初始噪声注入位置 |
| Fig. 2 动态 patch-score | 同一 token 分数如何变化，高分区域如何随深度更替？ | 统计已归档；从真实样本的层输出准备多层图与分数曲线 |
| Algorithm 1 PRD | 怎样从 x 生成 x_adv？ | 可直接从当前实现写出 |
| Table 1 实验配置 | 四架构如何实例化共同机制？ | 参数已完整归档；可主文简表、补充全表 |
| Table 2 主实验与比较 | 对哪些目标可迁移，较基线改善多少？ | PRD 4×14 已有；公平外部比较尚未在当前归档中提供 |
| Table 3 核心消融 | 两机制是否各有作用、是否互补？ | 当前 PRD 同协议完整因子结果尚未归档 |
| Fig. 3 ASR 参数分析 | checkpoint、drop budget 和噪声强度怎样影响迁移效果？ | 按当前随机 PRD 配置准备对应 ASR 对照 |
| Supplement qualitative cases | 对抗样本外观、成功与失败如何？ | 正式对抗图像在服务器；本地仅有哈希清单 |

## 8. 写作材料准备：围绕发现与 ASR 展开

### 8.1 可以现在写的部分

方法定义、实际算法、四架构配置、已有 ASR 主表，以及动态 patch-score 发现的叙事。

### 8.2 接下来准备的材料

1. **动态发现的可视化。** 选取真实样本，排列不同深度的 patch-score 图，标出高分区域的更替；以固定位置 token 的分数曲线展示其沿深度的变化。
2. **同协议 ASR 比较。** 使用相同 1000 样本、source/target 权重及预处理、ε=16/255 和 ASR 分母；共同报告计算预算。
3. **两机制的 ASR 消融。** 配对 seed、固定支撑流程，比较共同优化底座、仅 progressive、仅 opponent 与完整 PRD。对照中的 schedule 和保留区域规则随实验设置说明。
4. **Progressive 参数分析。** 在当前随机 PRD 配置下，用 ASR 对照呈现 checkpoint 位置、数量和 drop budget 的效果；按原生网格说明各项预算。
5. **Opponent 参数分析。** 在相同注入位置与 feature RMS 下，围绕噪声结构、强度和区域准备 ASR 对照。
6. **实验设置和过程记录。** 简洁记录参数选择：历史筛选使用过主实验图像的子集和目标 ASR，当前参数承接这些选择。正文设置或复现材料按实际过程说明。
7. **结果说明与计算代价。** 根据写作需要准备 clean accuracy、重复 seed、逐样本统计、生成时间和显存记录，配合已有 ASR 主表呈现实验结果。

已有主表固定为论文的主实验。写作先推进方法与结果初稿，后续材料围绕动态发现、两项机制和 ASR 评价补充。

## 9. 相关工作定位的起始清单

以下近邻文献的题名及摘要已核验，作为精读和方法定位的起点。

- [Towards Transferable Adversarial Attacks on Vision Transformers（PNA/PatchOut，AAAI 2022）](https://ojs.aaai.org/index.php/AAAI/article/view/20169)：需要核对 patch 随机化的位置、粒度与梯度使用方式。
- [On Improving Adversarial Transferability of Vision Transformers（Self-Ensemble/Token Refinement，ICLR 2022）](https://arxiv.org/abs/2106.04169)：需要区分多深度判别分支与同一轨迹上的顺序状态中断。
- [Transferable Adversarial Attacks on Vision Transformers With Token Gradient Regularization（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Transferable_Adversarial_Attacks_on_Vision_Transformers_With_Token_Gradient_Regularization_CVPR_2023_paper.html)：需要区分 token 梯度处理与 PRD 的前向干预。
- [Learning Transferable Adversarial Examples via Ghost Networks](https://arxiv.org/abs/1812.03413)：需要核对随机 surrogate 变体和梯度聚合的已有设计。
- [ViT-EnsembleAttack（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Cao_ViT-EnsembleAttack_Augmenting_Ensemble_Models_for_Stronger_Adversarial_Transferability_in_Vision_ICCV_2025_paper.html)：核对 model augmentation 与单 source PRD 的关系，比较时对齐 source 与计算预算。

Related Work 说明已有方法如何处理 patch/token 和模型计算，以及 PRD 在干预位置、顺序传播与保留证据扰动上的设计特点。

## 10. 写作顺序与每一步完成标准

推荐顺序：**发现与图表 → 方法 → 实验设置与已有 ASR 结果 → 动机观察 → 相关工作 → 引言 → ASR 消融与分析 → 结论 → 摘要与标题 → 补充材料和全文核对**。

文献精读与补充证据的准备从第一步开始，并行于写作；这里的顺序指章节初稿的产出顺序。

| 次序 | 工作 | 为什么此时写 | 完成标准 |
| --- | --- | --- | --- |
| 0 | 固定主表、协议、术语和核心故事 | 让发现、方法与结果形成清楚的叙事主线 | 动态发现鲜明；主实验数字与设置对应 |
| 1 | Method、Algorithm 1、流程图、四源配置表 | 实现是当前最确定的部分，也是比较/消融的前提 | 噪声位置、mask 独立性、网格变换、RMS、更新可从文本复现 |
| 2 | Experimental Setup 与现有 Main Results | 呈现完整方法的 ASR 表现和实验设置 | 主表由原始数据产生；指标、样本数与参数选择过程清楚 |
| 3 | Observation and Motivation | 让读者看到局部证据的动态演化，再进入 progressive 干预 | 实际样本分数图与曲线清楚；发现到方法的连接自然 |
| 4 | Related Work 初稿与差异表 | 先确认近邻方法，再确定创新措辞 | 每项差异有原论文/实现依据；公平基线方案明确 |
| 5 | Introduction 初稿 | 此时方法、证据、近邻工作均已清楚 | 问题→观察→设计问题→两机制→验证→贡献，保持一条主线 |
| 6 | Ablation/Analysis 与适用范围说明定稿 | 围绕两机制的效果、参数与成本展开 | 消融和参数分析以 ASR 呈现；过程细节简洁说明 |
| 7 | Conclusion | 收束经过核验的结论 | 回答引言问题，没有新证据或超范围结论 |
| 8 | Abstract 与 Title 定稿 | 准确压缩最终论文 | 首要指标用 83.10% strict；外部增益只在已有公平比较时写入 |
| 9 | Supplement 与全文核对 | 将可复现细节补齐，并检查各节一致性 | 术语、配置、指标、引用、图注、算法与代码对应一致 |

第一轮写作的具体交付物应是：Method 草稿、Algorithm 1、流程图、Experimental Setup 草稿、主表及其说明。随后再产出观察段和 Introduction；无需等待所有补充实验才开始写这些已有事实。

## 11. 篇幅与投稿安排

截至核对日期，CVPR 2027 的 [Call for Papers](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) 与 [Dates](https://cvpr.thecvf.com/Conferences/2027/Dates) 可访问：注册截止 2026-11-10 AoE、论文截止 2026-11-16 AoE、补充材料截止 2026-11-23 AoE。相应的北京时间截止为 11 月 11 日、17 日、24 日 19:59:59。

2027 CFP 链接的 Author Guidelines 页面在最初访问时返回 404。篇幅暂按 [CVPR 2026 Author Guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines) 的 8 页正文（含图表）+参考文献规划，最终核对 2027 正式指南。

临时篇幅预算：Abstract 0.25 页、Introduction 1.25 页、Related Work 0.6 页、Observation 0.6 页、Method 2.0 页、Experiments/Analysis 3.0 页、Conclusion/Limitations 0.3 页，共 8 页。图表计入所属章节；这只是排版预算，不是固定比例。

全文让读者先看到高分 patch 随深度迁移的动态发现，再理解 PRD 如何沿这一过程施加中断和扰动，最后通过 ASR 看到完整方法及两项机制的迁移效果。
