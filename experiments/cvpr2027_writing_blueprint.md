# PRD：CVPR 2027 论文写作蓝图

材料统一日期：2026-10-12。当前本地分支：`prd`。主实验采用 2026-10-12 完成的四源、1000 图像、14 目标 PRD 运行，实际实现提交为 `bf092246`。方法执行契约见 [PRD 方法记录](prd_method_contract.md)，数据唯一入口为 `results/prd_cross_arch_s1000.csv` 和 `results/prd_summary_s1000.csv`。

## 1. 项目目标与本文档工作内容

项目的主要任务是：依据本地 PRD 分支的实现与实验材料，以已有四源、1000 图像、14 目标结果作为主实验，撰写面向 CVPR 2027 投稿的论文。

本文档承担写作前的整体理解、论证组织、章节设计与写作顺序规划。已有主实验是论文的事实基础；必要的补充比较和机制验证围绕这一基础开展。方法保持两个核心机制：checkpoint-wise progressive random token drop 与 RGB opponent-channel random noise。

全文采用以 PRD 为中心的方法论文结构：Introduction、Related Work、Progressive Route Disruption、Experiments、Conclusion。Preliminaries 放在方法章开头；动态观察在引言中呈现，并在 progressive drop 小节中直接连接到方法设计。正文的章节编号与逐节任务见本文档 §6。

建议工作标题：**Progressive Route Disruption for Transferable Adversarial Attacks**。

写作约定：直接讲发现、设计与结果，用鲜明的动态过程吸引读者。动机以同一样本的 token 分数变化和高 patch-score 区域的迁移为中心；ASR 呈现迁移效果。梯度诊断保留为内部分析工具。省去针对未提出主张的反复澄清，让每一段推进论文主线。长期约定已记录在 `AGENTS.md` 的 Paper-writing objective and voice。

## 2. 材料与证据层级

| 材料 | 用途 | 写作地位 |
| --- | --- | --- |
| `AGENTS.md`、`README.md`、`experiments/prd_method_contract.md` | 确认单一方法、执行顺序、配置与术语 | 当前主线约束 |
| `progressive_attack.py`、`main.py`、`nets/*.py` | 核对 schedule、噪声、遍历与更新 | 方法事实依据 |
| `utils.py`、`transfer_eval.py`、`record_experiment.py`、`gradient_replay.py` | 核对数据、预处理、ASR 与随机记录 | 实验协议依据 |
| `tests/*.py` | 当前执行行为与 optional real-model 验证 | 行为契约 |
| `experiments/prd_paper_story.md` | 论文故事与表达约定 | 叙事起点 |
| `experiments/progressive_cross_arch_mainline_s1000.md` | 当前正式配置、汇总和 4×14 逐目标表 | 主实验说明 |
| `results/prd_run_artifacts/{vit,cait,pit,visformer}/transfer.csv` | 原始结果、实际参数、实现版本与时间戳 | 主实验原始记录 |
| `results/prd_cross_arch_s1000.csv`、`results/prd_summary_s1000.csv` | 56 项 source-target ASR 与四源汇总 | 制作主表的数值依据 |
| `results/prd_run_artifacts/{vit,cait,pit,visformer}/attack_params.json`、`provenance.json` | 攻击参数、环境、源配置与目标集 | 可复核性依据 |
| `results/patch_rank_observation_summary.json` | 64 图像深度关系变化观察 | 动机证据 |
| `experiments/server_handoff_inventory.md` | 本地材料与服务器当前运行目录 | 文件可用性依据 |

规范化表与四份原始 transfer CSV 的逐目标数值及实际参数一致；汇总从 56 条记录按目标类型和 source-match 标记计算。四次主实验使用相同的 1000 图像输入集、seed 20260907、全部样本 ASR 分母，记录的 skipped 为零。

当前主实验的对抗图像、replay manifests、逐样本预测和日志保存在服务器。本地保存完整结果 CSV、参数与环境记录；样本可视化和进一步逐样本分析从当前运行目录取得原件。

## 3. 研究故事：观察、设计问题、方法与验证

### 3.1 起点：高分 patch 随深度迁移，局部证据持续演化

随着同一张图像向网络深处传播，同一个局部 token 与全局表示的余弦相似度会发生明显变化，高 patch-score 的 patch 位置也会随层数变化而迁移与更替。局部证据参与全局表示形成的过程是动态的，这就是故事的出发点。

用同一样本的多层 patch-score 图展示高分区域的更替，再用若干固定位置 token 的分数曲线呈现其与全局表示关联的变化。先让读者看到这一过程，再提供辅助统计。

正文中，Introduction 用代表性样本与 Fig. 1 引出这一发现；§3.2 Progressive Random Token Drop 简洁定义 patch-score、展开动机，并紧接着介绍沿深度的随机中断。更多样本、观察层对和统计放入补充材料 D。

观察记录以 local/global 表示的余弦相似度定义 patch-score，global 表示依架构使用 CLS 或 GAP。已有统计覆盖 64 张图像和统一 7×7 网格；早晚层平均 Spearman 为 ViT −0.0708、CaiT 0.1817、PiT 0.0632、Visformer 0.1168。统计作为可视化发现的辅助材料，具体观察层对随图说明。

### 3.2 设计问题：如何干预持续演化的局部证据？

推荐的叙事连接：既然局部证据在前向传播中持续演化，攻击也可以沿着这一过程展开，在不同深度反复干预正在形成的表示。

PRD 将这一想法落实为逐 checkpoint 的随机局部中断：每一次中断都改变接下来的传播状态，后续中断继续作用于已经演化的表示。重新采样让局部干预在不同 step、group 和深度上展开。

### 3.3 核心机制一：逐 checkpoint 的随机局部中断

在每个攻击 step、每个 augmentation group 上重新抽取 schedule。每个 checkpoint 从其当前原生网格的全部 local-token 位置中，均匀、无放回抽取指定数量。不同 checkpoint 独立采样，允许跨 checkpoint 重复位置。

每次到达 checkpoint 后立即将选中 token 的所有通道 hard-zero，然后沿已改变的状态继续同一次前向传播。token 数量和网格保持不变；此前归零的位置在后续计算中可能重新产生非零表示。

Route 指局部表示沿后续网络计算传播的轨迹。Progressive 强调每一次中断之后，模型继续在已经改变的状态上前向传播。

### 3.4 核心机制二：扰动初始局部表示的 RGB 对手通道投影噪声

噪声从亮度、红绿、黄蓝三个 RGB 方向采样，通过源模型真实的初始 RGB 卷积权重投影，并根据初始特征 RMS 匹配尺度；在 checkpoint 遍历前注入全部初始局部特征，包括未来被 drop 的位置。CLS/prefix 保持原值。

三个方向的采样方差为 0.5、1.25、1.25，为扰动赋予不同的亮度与色度能量。初始 RGB 投影把这一颜色结构映射到源模型的特征空间，RMS 匹配控制扰动强度。

两项机制形成清楚的分工：progressive drop 中断局部证据的传播，opponent noise 扰动初始局部表示。消融实验通过 ASR 呈现各项机制与组合的效果。

### 3.5 可用于后续引言的中心论点

随着图像向网络深处传播，同一个局部 token 与全局表示的关联会发生明显变化，高 patch-score 的区域也不断更替。PRD 沿着这一演化过程，在多个 checkpoint 反复随机中断局部表示的传播，同时用 RGB 对手通道噪声扰动初始局部表示。四种源架构到 14 个目标模型的主实验取得 83.49% 的平均严格黑盒 ASR，呈现完整方法的迁移效果。

## 4. 方法实现备忘：用于正文第 3 章

本节整理完整执行事实、公式与配置。写作时按正文 §3.1–§3.4 的任务分配内容；四架构 checkpoint 与预算表安排在 Experimental Setup，adapter 细节安排在补充材料 B。

### 4.1 完整流程

1. 从当前对抗像素开始，为本 step/group 构建 fresh schedule；构建过程遍历 checkpoint 并立即应用 mask，不计算 patch-score。
2. 保存各 checkpoint 的原生网格与 mask；原始视图使用此 schedule。
3. 用反射填充的像素平移构造 phase view，把同一 schedule 的 mask 同步变换；映回各 checkpoint 原生网格时按 occupancy top-k 保持 drop 数量。
4. 对每个视图，在全部初始 RGB 局部特征处注入该视图独立抽取的 opponent noise；然后顺序到达 checkpoint、hard-zero、继续前向并计算真实标签 CE loss。
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

RMS 在单样本的全部初始 token 和通道上计算；实现中对特征 RMS 与投影权重使用 detach。初始注入写为 `H_0^noise = H_0 + η_hat`，覆盖全部局部位置。

像素梯度与更新写为：

`g_t = (1/20) Σ_(g=1..10) Σ_(v=0..1) ∇_x CE(f_(schedule,noise,view)(x_t), y)`；

`g_t' = g_t + 0.75 · G_(σ=4) * g_t`；

`m_(t+1) = m_t + g_t'`；

`x_(t+1) = Clip_[0,1](x + Clip_[-ε,ε](x_t + (ε/10) sign(m_(t+1)) − x))`。

动量直接累积处理后的梯度，更新过程按上述递推式描述。

### 4.3 全部初始局部特征的噪声注入

噪声范围覆盖当前架构的全部初始局部特征，独立于后续 checkpoint mask。ViT/CaiT 在初始 patch 特征处注入；PiT/Visformer 在各自 adapter 定义的初始 RGB 特征处注入。随后按各 checkpoint 原生网格执行 hard-zero。

每个可微视图在初始位置注入一次噪声，随后随特征传播。到达 checkpoint 时，按随机 mask 清零局部 token；相位 mask 按图像空间平移与原生网格 occupancy 保持 drop 数量。

完整执行顺序见 [PRD 方法记录](prd_method_contract.md)。

### 4.4 当前主实验配置

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
| ViT-B/16 | 85.06% | 89.96% | 78.52% | 84.18% |
| CaiT-S24 | 87.64% | 90.89% | 83.32% | 86.79% |
| PiT-B | 84.34% | 89.55% | 77.38% | 83.24% |
| Visformer-S | 81.18% | 83.16% | 78.53% | 79.75% |
| 四源宏平均 | 84.55% | 88.39% | 79.44% | 83.49% |

推荐摘要与引言优先报告 **83.49% 的四源平均严格黑盒 ASR**。84.55% 的 Overall 含 source-matched target，保留在主表并明确标注。

Transformer 的 88.39% 同样含各源的同架构目标；若叙述“跨模型 Transformer 迁移”，使用去除 source-matched target 后的四源平均 **86.97%**。CNN 的 79.44% 均为跨架构迁移。

ASR = `1 − correct/total`，分母是全部评估的 1000 张对抗图像，不筛选 target-clean-correct 子集。因此 ASR 包含目标模型原本的分类错误；补报相同样本的 clean accuracy 有助于解释攻击增加了多少错误，但保持现有主指标定义。

四个 source 分别独立攻击相同的 1000 张输入，共生成 4000 张对抗图像；56 项评估对应 56000 次样本预测。strict black-box 指排除 source-matched target 后对其余 13 个目标取平均。

主表覆盖标准模型与两个对抗训练 CNN，按目标类型呈现迁移表现与差异。

### 5.2 论文评价与内部分析的分工

论文统一用 ASR 呈现迁移性。主实验、方法比较、机制消融和参数分析都围绕 ASR 组织。effective rank 等梯度诊断保存在内部实验记录，用于研究过程中的检查和分析；正文与补充材料的迁移论证均采用 ASR。

### 5.3 建议的贡献组织

1. **提出 Progressive Route Disruption。** 从局部证据随深度演化的观察出发，在同一前向轨迹的多个 checkpoint 以 fresh、uniform random masks 反复中断局部 token 的传播。
2. **构造初始局部表示的结构化扰动。** 从 RGB 对手方向采样噪声，经源模型真实初始 RGB 投影映射并匹配特征 RMS，施加于全部初始局部特征，与 progressive disruption 共同组成 PRD。
3. **验证跨模型与跨架构迁移效果。** 在四种源架构、每源 1000 图像和 14 个目标模型上评估完整方法，报告 ASR，并通过组件消融与参数分析展开各机制的作用。

贡献段将动态发现、逐深度干预设计和 ASR 验证连接起来。四架构实例化呈现方法适用性。Related Work 围绕 PatchOut、深层 self-ensemble、token gradient regularization、surrogate/model augmentation 梳理具体差异。

## 6. 正文章节：每一节要解决什么问题

全文以提出和验证 PRD 为中心，采用下面的五章结构。以下编号指论文正文；本蓝图 §3 的研究故事与 §4 的实现备忘是写作材料的组织方式。

| 章节 | 目的与作用 | 本文应写的内容与产物 |
| --- | --- | --- |
| Title | 让读者立即知道方法与任务 | PRD + transferable adversarial attacks |
| Abstract | 用最短篇幅交代问题、方法、验证和意义 | 深度证据演化→两机制→四源/1000 图/14 目标→83.49% 严格黑盒；比较增益须等同协议结果 |
| 1. Introduction | 建立值得解决的问题，解释设计选择与贡献 | 迁移攻击场景；动态局部证据观察；干预演化轨迹的设计问题；两机制；贡献三点；开篇概念图 |
| 2. Related Work | 把创新放入最接近的已有研究 | 迁移优化/输入变换；token/patch attack；self-ensemble/model augmentation；特征噪声。每类说明与 PRD 的具体关系 |
| 3. Progressive Route Disruption | 从动态观察进入设计，完整提出 PRD | 3.1 Preliminaries；3.2 Progressive Random Token Drop（含观察与动机）；3.3 RGB Opponent-Channel Noise；3.4 Overall Attack Algorithm |
| 4. Experiments | 呈现完整方法与各机制的迁移效果 | 4.1 Experimental Setup；4.2 Main Transfer Results；4.3 Component Ablations；4.4 Analysis |
| 5. Conclusion | 以方法与验证结果收束全文 | 动态局部证据启发 PRD→两机制组成完整攻击→主要 ASR 结果 |
| Limitations / responsible use 段落 | 简洁交代方法适用范围与安全研究用途 | 参数选择、计算预算与目标范围；按篇幅及正式投稿要求安排 |
| References | 提供研究来源，明确方法定位 | 引用实际使用的相关文献，核对题名、版本与结论 |

### 6.1 方法章：从必要定义进入两个机制

| 小节 | 目的与内容 | 公式、图表与篇幅安排 |
| --- | --- | --- |
| **3.1 Preliminaries** | 简短交代无目标迁移攻击：源模型的参数与梯度用于样本生成，黑盒目标模型用于迁移评估；生成过程不使用目标模型的参数、梯度或查询反馈。定义图像、标签、扰动约束和后文所需的局部表示、层与 checkpoint 符号 | 给出攻击目标与 L∞ 约束；符号随需要引入。保持简洁，让读者尽快进入 PRD |
| **3.2 Progressive Random Token Drop** | 从同一 token 的分数变化与高分 patch 的位置更替出发，说明干预随表示演化沿深度展开。随后定义每个 step/group 的 fresh schedule、各 checkpoint 的独立均匀抽样、hard-zero 和继续传播 | patch-score 定义用于解释观察；配合 Fig. 1，接着给出随机集合与 checkpoint 更新公式。主体篇幅用于 progressive 设计 |
| **3.3 RGB Opponent-Channel Noise** | 解释初始局部表示的结构化扰动。定义亮度、红绿、黄蓝方向，经真实 RGB 投影映射、feature RMS 匹配，在全部初始局部特征处注入一次，再随网络传播 | 给出对手通道基、投影噪声与 RMS 匹配公式，说明其在 Fig. 2 中的注入位置 |
| **3.4 Overall Attack Algorithm** | 将两机制组织成完整攻击迭代：生成 schedule，原始/相位视图独立注入噪声，按各自 schedule 顺序 drop，求梯度并进行投影更新 | Fig. 2、Algorithm 1 与整体更新公式。相位配对、梯度平均、Gaussian residual 和动量作为执行流程简洁交代 |

Preliminaries 提供理解 PRD 所需的背景与符号。方法章的核心是“动态观察 → progressive 设计 → opponent noise → 完整算法”。四源的具体 checkpoint、drop 数量和噪声强度放在正文 §4.1 配置表；各架构的 traversal 与投影几何在补充材料 B 展开。

### 6.2 实验章：用 ASR 组织方法验证

实验子节应完整回答：

- **4.1 Experimental Setup：** 哪些图像、标签和权重，攻击预算、source/target、预处理、seed 和 ASR 定义是什么？给出四架构配置表。
- **4.2 Main Transfer Results：** 在 ViT 与 CNN 目标上分别表现如何？报告 Overall 与严格黑盒 ASR，并在取得同协议结果后展开外部方法比较。
- **4.3 Component Ablations：** progressive drop、opponent noise 及其组合分别取得什么 ASR？两项机制怎样共同贡献完整方法的效果？
- **4.4 Analysis：** checkpoint、drop budget、颜色方向与噪声强度怎样影响 ASR？计算代价如何，定性样本呈现什么特点？

Introduction 首先呈现问题与动态发现，再概括 PRD 及结果。§3.2 用观察直接引出机制定义；§4 用 ASR 完成验证。正文中的代表性发现、两个机制和主实验形成连续叙事，扩展观察与复现细节按 §7 安排。

## 7. 补充材料及图表计划

### 7.1 补充材料目录

| 部分 | 作用 |
| --- | --- |
| A. Complete implementation and algorithm | 完整伪代码、mask 变换、RMS/stop-gradient、动量、投影与 PNG 保存细节 |
| B. Architecture adapters | 四源原生阶段、初始投影几何、CLS/GAP/卷积前端、相位 mask 的跨尺度变换 |
| C. Reproducibility and evaluation protocol | 样本 ID、权重版本、环境、seed、运行参数、原始表、指标定义与参数选择过程 |
| D. Extended motivating observation | 扩展正文 §3.2 的动机：score 定义、层对、空间对齐、更多真实样本的分数图与变化曲线，及辅助统计 |
| E. Extended results and sensitivity | 逐目标结果、补充 seed/参数扫描/基线与消融，标注其与当前配置的对应关系 |
| F. Qualitative cases and computational cost | 成功/失败案例、可视化与实测计算代价 |

正文保留主要 ASR 比较和机制消融，补充材料展开复现细节与扩展结果。

### 7.2 图表与证据准备

| 图表 | 要回答的问题 | 当前材料状态 |
| --- | --- | --- |
| Fig. 1 动态发现与设计直觉 | 同一 token 分数如何变化，高分 patch 如何随深度更替，怎样引出沿深度的干预？ | 用于 Introduction 与 §3.2；统计已归档，代表性样本的多层图与分数曲线待从实际层输出准备 |
| Fig. 2 PRD 完整流程 | 初始噪声、checkpoint drop 和像素更新分别在哪里发生？ | 用于 §3.3–§3.4；按当前实现绘制，明确全部初始局部特征加噪声后顺序 drop |
| Algorithm 1 PRD | 怎样从 x 生成 x_adv？ | 放在 §3.4，可直接从当前实现写出 |
| Table 1 实验配置 | 四架构如何实例化共同机制？ | 放在 §4.1；参数已完整归档，主文简表、补充全表 |
| Table 2 主实验与比较 | 对哪些目标可迁移，与可比方法相比怎样？ | 放在 §4.2；PRD 4×14 已有，公平外部比较尚未在当前归档中提供 |
| Table 3 核心消融 | 两机制分别贡献什么，组合取得什么 ASR？ | 放在 §4.3；当前 PRD 同协议完整因子结果尚未归档 |
| Fig. 3 ASR 参数分析 | checkpoint、drop budget 和噪声强度怎样影响迁移效果？ | 放在 §4.4；按当前随机 PRD 配置准备对应 ASR 对照 |
| Supplement qualitative cases | 对抗样本外观、成功与失败如何？ | 当前主实验对抗图像在服务器；本地有原始 CSV、参数与环境记录 |

## 8. 写作材料准备：围绕发现与 ASR 展开

### 8.1 可以现在写的部分

§3 Method 的完整初稿（包括 §3.2 的动态观察与设计动机）、实际算法、四架构配置，以及 §4.1 Experimental Setup 和 §4.2 已有 ASR 主表的说明。

### 8.2 接下来准备的材料

1. **动态发现的可视化。** 选取真实样本，排列不同深度的 patch-score 图，标出高分区域的更替；以固定位置 token 的分数曲线展示其沿深度的变化。
2. **同协议 ASR 比较。** 使用相同 1000 样本、source/target 权重及预处理、ε=16/255 和 ASR 分母；共同报告计算预算。
3. **两机制的 ASR 消融。** 配对 seed、固定支撑流程，比较共同优化底座、仅 progressive、仅 opponent 与完整 PRD。消融使用当前方法定义，固定 schedule 与噪声构造的对应设置。
4. **Progressive 参数分析。** 在当前随机 PRD 配置下，用 ASR 对照呈现 checkpoint 位置、数量和 drop budget 的效果；按原生网格说明各项预算。
5. **Opponent 参数分析。** 在全部初始局部特征的统一注入位置与 feature RMS 定义下，围绕颜色方向和噪声强度准备 ASR 分析。
6. **实验设置和过程记录。** 简洁记录四架构 checkpoint、drop 数量、噪声强度、样本集、权重、环境和评估流程。
7. **结果说明与计算代价。** 根据写作需要准备 clean accuracy、重复 seed、逐样本统计、生成时间和显存记录，配合已有 ASR 主表呈现实验结果。

已有主表固定为论文的主实验。写作先推进方法与结果初稿，后续材料围绕动态发现、两项机制和 ASR 评价补充。

## 9. 相关工作定位的起始清单

### 9.1 方法论文的组织参考

正文采用 FPR 的整体组织方式，并参考 TGR 简洁进入核心设计的写法；GNS-HFA 提供在各机制中连接观察与设计的参考。

| 论文 | 已核对的组织方式 | 本文采用的写作安排 |
| --- | --- | --- |
| [TGR（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Transferable_Adversarial_Attacks_on_Vision_Transformers_With_Token_Gradient_Regularization_CVPR_2023_paper.pdf) | Method 开头定义符号与约束，随后介绍机制和实现 | Preliminaries 简洁，尽快进入 progressive 设计 |
| [FPR（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_Improving_Adversarial_Transferability_on_Vision_Transformers_via_Forward_Propagation_Refinement_CVPR_2025_paper.pdf) | Methodology 内的 Preliminary、两个机制和协同讨论 | 方法章内安排 Preliminaries、两个核心机制和完整算法 |
| [GNS-HFA（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/file/e25198b6a75f74277ee3a2bd4165d9ef-Paper-Conference.pdf) | 独立 Preliminaries，Method 按两个机制展开并结合各自观察 | §3.2 从动态发现直接进入 progressive drop；§3.3 展开初始结构化噪声 |

### 9.2 研究定位与比较对象

以下近邻文献的题名及摘要已核验，作为精读和方法定位的起点。

- [Towards Transferable Adversarial Attacks on Vision Transformers（PNA/PatchOut，AAAI 2022）](https://ojs.aaai.org/index.php/AAAI/article/view/20169)：需要核对 patch 随机化的位置、粒度与梯度使用方式。
- [On Improving Adversarial Transferability of Vision Transformers（Self-Ensemble/Token Refinement，ICLR 2022）](https://arxiv.org/abs/2106.04169)：需要区分多深度判别分支与同一轨迹上的顺序状态中断。
- [Transferable Adversarial Attacks on Vision Transformers With Token Gradient Regularization（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Transferable_Adversarial_Attacks_on_Vision_Transformers_With_Token_Gradient_Regularization_CVPR_2023_paper.html)：需要区分 token 梯度处理与 PRD 的前向干预。
- [FPR（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_Improving_Adversarial_Transferability_on_Vision_Transformers_via_Forward_Propagation_Refinement_CVPR_2025_paper.pdf)：精读前向 attention/token refinement 与 PRD 顺序 hard-zero、初始 opponent noise 的设计差异。
- [GNS-HFA（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/file/e25198b6a75f74277ee3a2bd4165d9ef-Paper-Conference.pdf)：精读梯度调整与频域变换，明确与 PRD 的干预位置和噪声构造的关系。
- [Learning Transferable Adversarial Examples via Ghost Networks](https://arxiv.org/abs/1812.03413)：需要核对随机 surrogate 变体和梯度聚合的已有设计。
- [ViT-EnsembleAttack（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Cao_ViT-EnsembleAttack_Augmenting_Ensemble_Models_for_Stronger_Adversarial_Transferability_in_Vision_ICCV_2025_paper.html)：核对 model augmentation 与单 source PRD 的关系，比较时对齐 source 与计算预算。

Related Work 说明已有方法如何处理 patch/token 和模型计算，以及 PRD 在干预位置、顺序传播与初始局部表示扰动上的设计特点。

## 10. 写作顺序与每一步完成标准

推荐顺序：**确定动态观察图与方法流程 → 写完整 Method（含观察与动机）→ 实验设置与已有 ASR 结果 → 相关工作 → 引言 → ASR 消融与分析 → 结论 → 摘要与标题 → 补充材料和全文核对**。

文献精读与补充证据的准备从第一步开始，并行于写作；这里的顺序指章节初稿的产出顺序。

| 次序 | 工作 | 为什么此时写 | 完成标准 |
| --- | --- | --- | --- |
| 0 | 固定主表、协议、术语，准备 Fig. 1 与 Fig. 2 | 确定方法论文的故事起点与完整流程 | 动态发现鲜明；图表内容与实际材料对应；主实验数字核对一致 |
| 1 | §3.1–§3.4 Method、Algorithm 1 | 将观察、progressive 设计、初始噪声和完整算法写成连续章节 | §3.2 的样本观察自然进入设计；公式、噪声位置、mask 与更新可复现 |
| 2 | §4.1 Experimental Setup 与 §4.2 Main Transfer Results | 呈现完整方法的 ASR 表现和实验设置 | 四源配置表与原始参数一致；主表由当前数据产生；指标与样本数清楚 |
| 3 | §2 Related Work 初稿与差异表 | 确认近邻方法，再确定创新措辞 | TGR、FPR、GNS-HFA 等的差异有原论文/实现依据；公平比较方案明确 |
| 4 | §1 Introduction 初稿 | 此时方法、证据、近邻工作均已清楚 | 问题→动态观察→PRD 设计→两机制→ASR 验证→贡献；Fig. 1 承接叙事 |
| 5 | §4.3 Component Ablations、§4.4 Analysis 与适用范围说明 | 围绕两机制的效果、参数与成本展开 | 消融和参数分析以 ASR 呈现；计算与过程细节简洁说明 |
| 6 | §5 Conclusion | 收束经过核验的结论 | 回答引言问题，以方法与 ASR 收束 |
| 7 | Abstract 与 Title 定稿 | 准确压缩最终论文 | 首要指标用 83.49% strict；外部增益只在已有公平比较时写入 |
| 8 | Supplement 与全文核对 | 补齐复现细节并检查各节一致性 | 术语、配置、指标、引用、图注、算法和章节编号对应一致 |

第一轮写作的具体交付物应是：含动态观察与动机的完整 Method 草稿、Algorithm 1、Fig. 1 与 Fig. 2 的内容方案、Experimental Setup 草稿、四源配置表，以及当前主表和说明。实际写作按 `AGENTS.md` 的逐自然段工作流推进：每次提供一个英文段落及对应中文翻译，讨论确认后再推进下一段。

## 11. 篇幅与投稿安排

截至核对日期，CVPR 2027 的 [Call for Papers](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) 与 [Dates](https://cvpr.thecvf.com/Conferences/2027/Dates) 可访问：注册截止 2026-11-10 AoE、论文截止 2026-11-16 AoE、补充材料截止 2026-11-23 AoE。相应的北京时间截止为 11 月 11 日、17 日、24 日 19:59:59。

2027 CFP 链接的 Author Guidelines 页面在最初访问时返回 404。篇幅暂按 [CVPR 2026 Author Guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines) 的 8 页正文（含图表）+参考文献规划，最终核对 2027 正式指南。

临时篇幅预算：Abstract 0.25 页、Introduction 1.25 页、Related Work 0.6 页、Method 2.6 页（含 §3.2 动态观察与动机）、Experiments/Analysis 3.0 页、Conclusion/Limitations 0.3 页，共 8 页。图表计入所属章节；这只是排版预算，不是固定比例。

全文让读者先看到高分 patch 随深度迁移的动态发现，再理解 PRD 如何沿这一过程施加中断和扰动，最后通过 ASR 看到完整方法及两项机制的迁移效果。
