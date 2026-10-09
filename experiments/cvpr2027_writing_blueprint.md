# PRD：CVPR 2027 论文写作蓝图

核对日期：2026-10-09。当前本地分支：`prd`；阅读基准：`2f2213d`。四次正式运行记录的实现版本：`98cb7cf1`。这两个版本之间没有攻击实现、adapter 或测试的变化，后续提交主要归档正式结果与写作证据。

## 1. 项目目标与本轮工作边界

项目的主要任务是：依据本地 PRD 分支的实现与实验材料，以已有四源、1000 图像、14 目标结果作为主实验，撰写面向 CVPR 2027 投稿的论文。

本对话承担写作前的整体理解、论证组织、章节设计与写作顺序规划。已有主实验是论文的事实基础；必要的补充比较和机制验证围绕这一基础开展。方法保持两个核心机制：checkpoint-wise progressive random token drop 与 RGB opponent-channel random noise。

建议工作标题：**Progressive Route Disruption for Transferable Adversarial Attacks**。标题暂不加入 optimal、state-of-the-art、universal 等尚未得到比较或泛化证据支持的修饰。

## 2. 已阅读材料与证据层级

| 材料 | 用途 | 写作地位 |
| --- | --- | --- |
| `AGENTS.md`、`README.md` | 确认研究范围、当前配置与术语 | 当前主线约束 |
| `progressive_attack.py`、`main.py`、`nets/*.py` | 核对 schedule、前向遍历、噪声、更新与架构适配 | 方法事实依据 |
| `utils.py`、`transfer_eval.py`、`record_experiment.py`、`gradient_replay.py` | 核对数据、预处理、结果记录、ASR 与随机重放 | 实验协议依据 |
| `tests/*.py` | 阅读已有实现验证及 optional real-model 验证条件 | 行为契约；本轮未重新执行测试 |
| `experiments/prd_paper_story.md` | 已有论文故事和结论边界 | 叙事起点 |
| `experiments/progressive_cross_arch_mainline_s1000.md` | 正式配置、4×14 主表与验证记录 | 主实验说明 |
| 四份 `outputs/csv/outputs_attack_prd1000_*.csv` | 原始模型结果、实际参数、实现版本和时间戳 | 主实验原始记录 |
| `results/prd_cross_arch_s1000.csv` | 56 项 source-target 结果 | 制作主表的数值依据 |
| `results/prd_run_artifacts/*/attack_params.json`、`replay_manifest.json.gz` | 实际参数、同序样本与随机事件 | 可复核性依据 |
| `results/prd_run_artifacts/*/gradient_diagnostics.json` | 多视图梯度诊断 | 描述性分析依据 |
| `results/prd_run_artifacts/patch_rank_observation_summary.json` | 64 图像深度排名重组观察 | 动机证据 |
| `experiments/server_handoff_inventory.md`、两份压缩文件索引 | 区分本地已归档与仅在服务器上的材料 | 文件可用性依据 |
| Git 历史中的动机文档、排序观察程序、checkpoint/drop 扫描、192 图像 follow-up、旧主实验报告 | 理解研究如何收敛到当前 PRD | 历史背景；不能直接作为当前方法的消融 |

本轮独立复核了：56 项结果的均值；各项 ASR 与原始 CSV 一致；归档的完整参数与原始 CSV 中的参数一致；四份 manifest 含相同、同序的 1000 个唯一 sample ID；phase events 和随机事件数量符合配置。

本机已有 1000 张干净输入，但没有正式对抗图像目录和模型缓存。因此本轮核对的是代码与归档证据，未重新生成对抗样本、复评模型或测量已保存 PNG 的扰动。

交接文档开头的 `patch-score-routing-attack` 是服务器当时的分支名称；当前本地分支实为 `prd`。历史报告中的 score selector、旧 checkpoint、旧统计口径与旧结果不得混入当前主实验。

## 3. 研究故事：观察、设计问题、方法与验证

### 3.1 起点：局部证据的表示关系随深度重组

历史观察用 local/global 表示余弦相似度作为 patch-score 代理量，在统一 7×7 空间网格上比较同一图像不同层的排序。global 表示的构造依架构而异，包含 CLS 与 GAP；它衡量表示关系，不能直接称为 patch 的因果重要性。

64 张图像的早晚层平均 Spearman：ViT −0.0708、CaiT 0.1817、PiT 0.0632、Visformer 0.1168。这支持“早晚层排序对应较弱、局部表示关系发生重组”的描述。

措辞边界：观察来自特定代理量、特定层对与小样本，不能推广为每个相邻层均剧烈变化，更不能据此断言语义因果路径已被识别。观察层与攻击 checkpoint 是两个不同设置，须分别给出。

### 3.2 设计问题：如何干预持续演化的局部证据？

推荐提出的问题：如果局部证据关系沿深度演化，是否可以通过同一前向轨迹上的多次局部中断，使攻击梯度覆盖不同的表示状态，而无需预先定位一个全程有效的高分区域？

随机采样与多 checkpoint 干预是对此问题的设计回应。排序重组提供启发，并不逻辑上证明随机采样最优，或证明 progressive 必然优于单点干预。这些比较需要实验。

### 3.3 核心机制一：逐 checkpoint 的随机局部中断

在每个攻击 step、每个 augmentation group 上重新抽取 schedule。每个 checkpoint 从其当前原生网格的全部 local-token 位置中，均匀、无放回抽取指定数量。不同 checkpoint 独立采样，允许跨 checkpoint 重复位置。

每次到达 checkpoint 后立即将选中 token 的所有通道 hard-zero，然后沿已改变的状态继续同一次前向传播。token 数量和网格保持不变；此前归零的位置在后续计算中可能重新产生非零表示。

Route 在本文中应操作性地定义为“局部表示沿后续网络计算传播的轨迹”。目前没有显式恢复一张语义因果路径图，也没有使用 routing graph。

### 3.4 核心机制二：扰动保留证据的 RGB 对手通道投影噪声

已选 schedule 决定各 checkpoint 的空间 drop 区域并集。噪声从亮度、红绿、黄蓝三个 RGB 方向采样，通过源模型真实的初始 RGB 卷积权重投影，并根据初始特征 RMS 匹配尺度；注入符合保留区域规则的初始局部特征。

当前代码的三个方向方差为 0.5、1.25、1.25。这是非各向同性的颜色结构；仅用一个正交 RGB 基替换等方差 Gaussian，并不会改变 Gaussian 分布。方法文字与公式应写明实际方差，不能只声称“换成 opponent 坐标后自然更有结构”。

设计上的职责互补是：progressive drop 改变局部证据的传播，opponent noise 改变保留证据的初始表达。组合是否产生实测独立增益或协同收益，仍需要受控消融和迁移梯度证据。

### 3.5 可用于后续引言的中心论点

局部表示关系会随模型深度重组。受此启发，PRD 在演化中的前向表示上施加重新采样的随机局部中断，并以初始 RGB 投影空间中的对手通道噪声扰动保留证据。现有评估表明，该完整方法可从四种源架构向多种 Transformer、CNN 及对抗训练目标迁移。

## 4. 方法必须与实现一致

### 4.1 完整流程

1. 从当前对抗像素开始，为本 step/group 构建 fresh schedule；构建过程遍历 checkpoint 并立即应用 mask，不计算 patch-score。
2. 由原始 schedule 得到各 checkpoint mask 的图像空间并集；原始视图使用此 schedule。
3. 用反射填充的像素平移构造 phase view，把同一 schedule 的 mask 同步变换；映回各 checkpoint 原生网格时按 occupancy top-k 保持 drop 数量。
4. 对每个视图，在初始 RGB 特征处注入该视图独立抽取的 opponent noise；然后顺序到达 checkpoint、hard-zero、继续前向并计算真实标签 CE loss。
5. 对两个视图的损失分别求关于当前像素的梯度；10 groups 提供 20 个梯度，直接取平均。
6. 添加 Gaussian 平滑梯度残差；进行原始梯度的动量累积、sign 更新、相对干净输入的 L∞ 投影与 [0,1] 截断。
7. 重复 10 steps，保存最终像素对抗图像；目标模型评估使用各自 timm transform。

重点：checkpoint mask 先被采样，噪声在可微前向的初始特征处注入，随后进行 progressive traversal。论文图不要把噪声画成最后一个 checkpoint 后的注入操作。phase 是轻微空间平移，不是 Fourier phase 扰动。

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

RMS 在单样本的全部初始 token 和通道上计算，并在保留区域门控之前匹配；代码对特征 RMS 与投影权重使用 detach。不能误写成每个 token 独立匹配，或只在 kept token 上计算 RMS。

像素梯度与更新写为：

`g_t = (1/20) Σ_(g=1..10) Σ_(v=0..1) ∇_x CE(f_(schedule,noise,view)(x_t), y)`；

`g_t' = g_t + 0.75 · G_(σ=4) * g_t`；

`m_(t+1) = m_t + g_t'`；

`x_(t+1) = Clip_[0,1](x + Clip_[-ε,ε](x_t + (ε/10) sign(m_(t+1)) − x))`。

代码直接累积梯度，没有标准 MI-FGSM 常见的逐样本 L1 梯度归一化。正文用实际递推式，避免把它概括成完全相同的标准 MI-FGSM 更新。

### 4.3 跨尺度保留区域的精确定义

`_image_mask_to_projection_drop_mask` 根据初始 RGB 卷积的 kernel、stride、padding、dilation 计算每个感受野的被 drop 区域覆盖比例，比例严格大于 0.5 才标为 initial drop。噪声施加在其补集。

在不重叠且对齐的 ViT patch 投影上，这对应 drop 并集之外的初始 token。对于重叠投影和跨尺度网格，应明确“感受野覆盖比例 >0.5 的投影位置被排除”这一实际规则；不应声称噪声的每个对应像素都严格不与 drop 区域相交。

### 4.4 当前唯一主实验配置

| Source | Checkpoints | 原生网格 | Drop 数量 | Opponent β | 生成 batch size |
| --- | --- | --- | --- | ---: | ---: |
| ViT-B/16 | block3, block10 | 14×14, 14×14 | 10, 10 | 0.2 | 96 |
| CaiT-S24 | block17, block23 | 14×14, 14×14 | 2, 28 | 0.2 | 48 |
| PiT-B | stage2_block1, stage3_block2, stage3_block3 | 16×16, 8×8, 8×8 | 5, 2, 6 | 0.4 | 96 |
| Visformer-S | stage2_block1, stage3_block1 | 14×14, 7×7 | 41, 10 | 0.4 | 48 |

ViT 各 checkpoint 的 drop ratio 均为 `0.051020408163`。不同架构没有统一相同的 drop count，也没有统一相同的 checkpoint 深度。

共同配置：seed 20260907、10 steps、ε=16/255、步长 ε/10、动量系数 1、10 groups×2 views、平移集合 (4,4)/(8,8)/(12,12)、Gaussian σ=4/α=0.75。100 条 schedule/图像；ViT/CaiT/Visformer 200 次随机 checkpoint mask selection，PiT 300 次。变换后的 phase masks 是共享 schedule 的派生结果，不是额外独立抽样。

这些 schedule 数量不等于全部前向计算量：还存在无梯度 schedule 构建遍历。讨论效率时须报告实际运行时间、前向/反向计算或 FLOPs，而不能只报 20 views。

## 5. 现有结果与可主张的贡献

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

四个 source 分别独立生成对抗样本，不是四源模型集成攻击。共有 4000 张生成图像，来自相同 1000 个输入；56 项评估对应 56000 次样本预测，不是 56000 张不同输入。strict black-box 在当前报告中仅表示排除 source-matched target，不能自动等同于“从未参与参数选择的 held-out target”。

主表覆盖标准模型与两个对抗训练 CNN。可讨论 CNN 迁移及防御目标迁移差异；不能把两个防御目标推广成对所有防御有效。

### 5.2 梯度诊断的正确用途

20-view effective rank：ViT 19.5131、CaiT 19.3441、PiT 19.4609、Visformer 18.3944。它由单样本的归一化视图梯度 Gram 矩阵特征值熵计算，接近上限 20，表明这些视图梯度并不高度重复。

这是源模型内部的描述性多样性证据，不测量源与目标梯度对齐，也不说明随机方向一定有利于迁移。现有 summary 对 batch/step 记录等权平均，尾批较小也占一项；如重新生成统计，须说明加权口径并保留原记录。

### 5.3 建议的贡献组织

1. **逐深度干预的攻击设计。** PRD 通过同一前向轨迹上的 fresh、uniform、checkpoint-wise hard-zero，干预演化中的局部证据；执行不依赖 patch-score。
2. **保留证据的结构化扰动。** RGB 对手方向的非各向同性噪声经真实初始 RGB 投影并匹配特征 RMS，与 progressive drop 形成职责分工。
3. **跨架构实现与实证。** 在固定网格、池化层级和卷积式前端上保留同一遍历原则；四源×14 目标的现有主实验提供可复核的迁移结果。

深度排序观察是设计动机，可以作为贡献段的启发性发现，但不应膨胀为独立的因果理论贡献。跨架构 adapter 是方法适用性支撑，不能仅凭接口设计宣称新理论。

这三点是对本项目贡献的组织，尚不等于已确认相对全部文献的首创。特别需要与 PatchOut、深层 self-ensemble、token gradient regularization、surrogate/model augmentation 对照。

## 6. 正文章节：每一节要解决什么问题

下面是适合本工作的建议叙事结构，不是 CVPR 官方规定的固定章节清单。

| 章节 | 目的与作用 | 本文应写的内容与产物 |
| --- | --- | --- |
| Title | 让读者立即知道方法与任务 | PRD + transferable adversarial attacks；避免无证据的优越性修饰 |
| Abstract | 用最短篇幅交代问题、方法、验证和意义 | 深度证据演化→两机制→四源/1000 图/14 目标→83.10% 严格黑盒；比较增益须等同协议结果 |
| 1. Introduction | 建立值得解决的问题，解释设计选择与贡献 | 迁移攻击场景；动态局部证据观察；干预演化轨迹的设计问题；两机制；贡献三点；开篇概念图 |
| 2. Related Work | 把创新放入最接近的已有研究 | 迁移优化/输入变换；token/patch attack；self-ensemble/model augmentation；特征噪声。每类说明与 PRD 的具体关系 |
| 3. Depth-wise Evidence Reorganization | 用可复核观察连接问题与方法 | score 代理量定义、CLS/GAP 区别、层对、64 图/7×7 协议、Spearman 与排序示意；明确观察的局限 |
| 4. Progressive Route Disruption | 给出能复现、能比较的方法 | 4.1 threat model/notation；4.2 progressive random schedule；4.3 opponent noise；4.4 phase pairing/gradient/update；4.5 architecture-specific instantiation |
| 5. Experiments | 检验完整方法效果、相对收益与机制 | 5.1 setup；5.2 现有主实验及公平比较；5.3 两机制消融；5.4 transferable-gradient/敏感性/效率与定性分析 |
| 6. Conclusion | 回答研究问题，收束适用边界 | 总结 progressive 干预与保留证据扰动及已支持的迁移结果；不引入新结果 |
| Limitations / responsible use 段落 | 给出结果适用范围与安全研究用途 | 小样本观察、参数选择、计算预算、目标范围、基线/消融证据边界；放在 §5 末或结论附近，独立成节取决于空间与正式要求 |
| References | 提供可查的研究来源 | 按实际使用核验引用、版本、作者与结论；不把旧项目故事替代文献差异分析 |

实验子节应完整回答：

- **5.1 Experimental Setup：** 哪些图像、标签和权重，攻击预算、source/target、预处理、seed、metric 与参数选择过程是什么？
- **5.2 Main Transfer Results：** 在现有同协议主表中表现如何？去除 source-matched target 后是否仍有迁移？与可比方法相比如何？
- **5.3 Component Ablations：** progressive 与 opponent 各自贡献什么？组合是否优于各单项？收益能否由已有支撑流程解释？
- **5.4 Analysis：** 随机中断如何影响梯度、为什么选择此 checkpoint/drop budget、噪声结构是否必要、计算代价如何、何处失败？

页数紧张时，将 §3 压缩为 Introduction 的观察段与 Method 的动机子节，或将 §5.4 的较长机制分析单列。必须保留“观察→方法→验证”的逻辑关系。

## 7. 补充材料及图表计划

### 7.1 补充材料目录

| 部分 | 作用 |
| --- | --- |
| A. Complete implementation and algorithm | 完整伪代码、mask 变换、RMS/stop-gradient、动量、投影与 PNG 保存细节 |
| B. Architecture adapters | 四源原生阶段、初始投影几何、CLS/GAP/卷积前端、跨尺度 mask 映射 |
| C. Reproducibility and evaluation protocol | 样本 ID、权重版本、环境、seed、运行参数、原始表、指标定义与参数选择过程 |
| D. Extended motivating observation | score 定义、层对、空间对齐、更多样本或 per-image 分布；仅放实际可提供的材料 |
| E. Extended results and sensitivity | 逐目标结果、补充 seed/参数扫描/基线与消融，标注其与当前配置的对应关系 |
| F. Gradient diagnostics and qualitative cases | 诊断公式、目标梯度分析、成功/失败案例、可视化与计算代价 |

主结论必需的比较和消融应在正文出现，不把决定贡献成立的证据全部放进补充材料。

### 7.2 图表与证据准备

| 图表 | 要回答的问题 | 当前材料状态 |
| --- | --- | --- |
| Fig. 1 概念与方法总览 | 为什么多深度中断？两个机制在哪发生？ | 可依据代码绘制；注意初始噪声注入位置 |
| Fig. 2 深度排序重组 | 同一图像的局部关系如何变化？ | 汇总统计已归档；真实热图需取历史逐样本材料或独立复测，不能凭 summary 编造 |
| Algorithm 1 PRD | 怎样从 x 生成 x_adv？ | 可直接从当前实现写出 |
| Table 1 实验配置 | 四架构如何实例化共同机制？ | 参数已完整归档；可主文简表、补充全表 |
| Table 2 主实验与比较 | 对哪些目标可迁移，较基线改善多少？ | PRD 4×14 已有；公平外部比较尚未在当前归档中提供 |
| Table 3 核心消融 | 两机制是否各有作用、是否互补？ | 当前 PRD 同协议完整因子结果尚未归档 |
| Fig. 3 梯度/参数/代价分析 | 多样性是否联系到迁移，选择是否稳定？ | effective rank 已有；目标梯度与随机主线敏感性需要核对/补充 |
| Supplement qualitative cases | 对抗样本外观、成功与失败如何？ | 正式对抗图像在服务器；本地仅有哈希清单 |

## 8. 证据缺口：保持主实验，补足关键论证

### 8.1 可以现在写的部分

方法定义、实际算法、四架构配置、已有主表、观察的描述性结论、现有梯度 effective rank 与局限。

### 8.2 应优先补足或限制措辞的部分

1. **同协议方法比较。** 使用相同 1000 样本、source/target 权重及预处理、ε=16/255、ASR 分母。补充实测结果之前，不写 SOTA 或领先多少个百分点。除相同步数外，另对实际计算预算作公平说明。
2. **两个机制的归因。** 最小因子对照包含共同优化底座、仅 progressive、仅 opponent、完整 PRD；配对 seed、固定支撑流程。用于验证的对照不意味着恢复被删除的历史生产攻击路径。原始 pair schedule 与 kept-region 规则在无 drop 的对照中需要明确规定。
3. **Progressive 的特定收益。** 如提出“多深度干预优于一次随机中断”，需在当前随机主线下比较，说明公平预算是 drop 次数、去重空间面积还是计算量。不能把旧 score-based K 扫描直接当作此证据，也不能只把跨尺度 drop 数简单相加当等预算。
4. **Opponent 的特定收益。** 如声称颜色结构、初始 RGB 投影或 kept-only 有必要，需在相同注入位置与 feature RMS 下比较结构/尺度/区域对照。正交方向的命名本身不能构成效果证据。
5. **Transferable-gradient 诊断。** 区分内部视图梯度多样性与源/目标对齐；可在独立分析中测量目标梯度 cosine、sign agreement 或沿攻击方向的目标 loss 变化，并和 transfer ASR 对应。目标梯度不用于生成攻击。现有 rank 不能替代此验证。
6. **参数选择与泛化边界。** 历史记录显示多轮筛选使用这 1000 张图的不同子集及目标 ASR；当前默认参数承接这些选择。因此不能宣称当前主表是完全未参与调参的 held-out 测试。正文或补充透明说明；若后续要主张未知目标泛化，可另加独立样本/held-out target 验证，同时保留现有主表。
7. **Clean accuracy、统计和成本。** 补报目标干净准确率；当前只有一个正式 seed，不写多 seed 稳定性或显著性。需要置信区间时取得逐样本预测并采用配对、按图像重采样的统计；56 项评估并不独立。报告实际生成时间/显存/计算量，未测量前不主张高效。

核心评估问题是比较依据和归因依据的完整程度，而不是主结果的绝对 ASR 是否足够高。写作可以现在推进，缺少证据的结论应保留为空位或使用准确的弱表述。

## 9. 相关工作定位的起始清单

以下是已核验题名及摘要的近邻文献，作为精读与公平比较的起点；本轮不是完整文献综述，也尚未核定所有实现差异。

- [Towards Transferable Adversarial Attacks on Vision Transformers（PNA/PatchOut，AAAI 2022）](https://ojs.aaai.org/index.php/AAAI/article/view/20169)：需要核对 patch 随机化的位置、粒度与梯度使用方式。
- [On Improving Adversarial Transferability of Vision Transformers（Self-Ensemble/Token Refinement，ICLR 2022）](https://arxiv.org/abs/2106.04169)：需要区分多深度判别分支与同一轨迹上的顺序状态中断。
- [Transferable Adversarial Attacks on Vision Transformers With Token Gradient Regularization（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Transferable_Adversarial_Attacks_on_Vision_Transformers_With_Token_Gradient_Regularization_CVPR_2023_paper.html)：需要区分 token 梯度处理与 PRD 的前向干预。
- [Learning Transferable Adversarial Examples via Ghost Networks](https://arxiv.org/abs/1812.03413)：需要核对随机 surrogate 变体和梯度聚合的已有设计。
- [ViT-EnsembleAttack（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Cao_ViT-EnsembleAttack_Augmenting_Ensemble_Models_for_Stronger_Adversarial_Transferability_in_Vision_ICCV_2025_paper.html)：需要核对 model augmentation 与单 source PRD 的关系；原文 ensemble 设置不能未经对齐直接比较。

Related Work 的最终任务是明确“什么已被做过、本文改变了哪一处干预、这一改变有什么证据”。不能把所有已有方法一概说成仅依赖最终层、静态 patch 或输入空间处理。

## 10. 写作顺序与每一步完成标准

推荐顺序：**证据与图表 → 方法 → 实验设置与已有结果 → 动机观察 → 相关工作 → 引言 → 分析与局限收口 → 结论 → 摘要与标题 → 补充材料和全文核对**。

文献精读与补充证据的准备从第一步开始，并行于写作；这里的顺序指章节初稿的产出顺序。

| 次序 | 工作 | 为什么此时写 | 完成标准 |
| --- | --- | --- | --- |
| 0 | 固定主表、协议、术语和 claim-evidence 对应关系 | 先确定有哪些事实，防止引言反向要求结果 | 每个数字有来源；区分观察、方法事实、效果和未验证假设 |
| 1 | Method、Algorithm 1、流程图、四源配置表 | 实现是当前最确定的部分，也是比较/消融的前提 | 噪声位置、mask 独立性、网格变换、RMS、更新可从文本复现 |
| 2 | Experimental Setup 与现有 Main Results | 明确论文能证明什么和比较口径 | 主表由原始数据产生；strict/overall、样本数、干净错误与调参边界清楚 |
| 3 | Observation and Motivation | 以真实观察建立到方法的合理桥梁 | score 定义及层对可查；不把相关性观察写成因果证明 |
| 4 | Related Work 初稿与差异表 | 先确认近邻方法，再确定创新措辞 | 每项差异有原论文/实现依据；公平基线方案明确 |
| 5 | Introduction 初稿 | 此时方法、证据、近邻工作均已清楚 | 问题→观察→设计问题→两机制→验证→贡献，保持一条主线 |
| 6 | Ablation/Analysis 与 Limitations 定稿 | 根据已经取得的证据收紧归因和创新强度 | 对照可比；未完成结果不写成事实；有效秩和迁移对齐分开 |
| 7 | Conclusion | 收束经过核验的结论 | 回答引言问题，没有新证据或超范围结论 |
| 8 | Abstract 与 Title 定稿 | 准确压缩最终论文 | 首要指标用 83.10% strict；外部增益只在已有公平比较时写入 |
| 9 | Supplement 与全文核对 | 将可复现细节补齐，并检查各节一致性 | 术语、配置、指标、引用、图注、算法与代码对应一致 |

第一轮写作的具体交付物应是：Method 草稿、Algorithm 1、流程图、Experimental Setup 草稿、主表及其说明。随后再产出观察段和 Introduction；无需等待所有补充实验才开始写这些已有事实。

## 11. 篇幅与投稿安排

截至核对日期，CVPR 2027 的 [Call for Papers](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) 与 [Dates](https://cvpr.thecvf.com/Conferences/2027/Dates) 可访问：注册截止 2026-11-10 AoE、论文截止 2026-11-16 AoE、补充材料截止 2026-11-23 AoE。相应的北京时间截止为 11 月 11 日、17 日、24 日 19:59:59。

2027 CFP 链接的 Author Guidelines 页面在本轮访问返回 404。先依据 [CVPR 2026 Author Guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines) 的 8 页正文（含图表）+参考文献规划篇幅，待 2027 指南可用后核验，不能把上一届规范写成已经确认的 2027 规则。

临时篇幅预算：Abstract 0.25 页、Introduction 1.25 页、Related Work 0.6 页、Observation 0.6 页、Method 2.0 页、Experiments/Analysis 3.0 页、Conclusion/Limitations 0.3 页，共 8 页。图表计入所属章节；这只是排版预算，不是固定比例。

全文应让读者依次获得：一个清楚的研究问题、一项可复核的设计动机、一个准确且可复现的方法、一组协议明确的主实验，以及与证据强度相匹配的贡献和结论。
