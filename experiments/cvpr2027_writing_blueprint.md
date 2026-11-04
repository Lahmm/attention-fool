# CVPR 2027 写作蓝图：方向方差比与正收缩比例指导选层与随机 token drop

更新日期：2026-11-04。依据用户确认的主线收紧方案，仅保留当前动态单边界方法及其写作安排。PRD 暂作为仓库工作名称；方法名与标题待讨论确定。本文档规划章节、算法、图表和证据准备，论文自然段仍只在对话中交付。

## 1. 目标、中心问题与写作边界

论文研究根据输入表示统计确定随机 token 干预的深度，并通过这一干预优化可迁移对抗扰动。核心设计是**方向方差比 × 正收缩比例选层＋选定边界的均匀随机 token drop**。

故事顺序：**局部—全局关联沿深度变化 → 干预深度的选择问题 → 方向比与正收缩乘积定位 → 单边界随机 drop → 迁移 ASR 验证**。跨层观察建立动态背景，方向比与正收缩乘积定义选层规则，ASR 对照检验设计效果。

i.i.d. RGB 采样经初始投影映射与 RMS 匹配的随机特征增强作为统一设置；具体构造放入补充材料。摘要、引言、贡献、结论和核心图不突出噪声，不单设噪声方法小节。空间配对、梯度平均和投影更新组成完整优化流程。

当前选层已按用户确认迁入代码主线。现有代码默认仍为 `opponent`，显式 `--noise-structure isotropic_rgb` 支持 i.i.d. RGB 采样；已有 opponent 记录保持原标签。最终方法表对应实际 i.i.d. 配置，当前文档不宣称已完成该配置的四源主实验。

## 2. 研究故事与贡献

### 2.1 从动态观察进入干预深度问题

同一位置的局部 token 与全局表示的余弦相似度沿深度变化，高分 patch 区域也随之更替。使用真实样本的跨层 patch-score 图展示这一过程。接着呈现各层局部 token 分数的分布与方差曲线，将读者引向干预深度的确定。

同一 token 跨层变化描述表示演化；同一层不同 token 的分数差异定义定位统计。两者分别呈现，避免用一种现象代替另一种统计。方差的直接含义是局部—全局关联的分散程度。选层有效性由后续匹配设置的 ASR 对照验证。

### 2.2 方向比与正收缩乘积选层与随机干预

每张干净图像在攻击前进行一次完整前向，计算各 block 输出方向方差比与输入／输出正收缩比例的乘积，选择最高分 block。在该 block 的输入边界，从全部局部 token 中均匀抽样并 hard-zero；模型在改变后的状态上继续前向。

“动态”表示根据输入计算选层。边界在该图的全部攻击步与视图中固定，drop 位置按 step/group 重采样。Patch score 用于选层与解释观察，层内位置由均匀随机抽样决定。

### 2.3 贡献组织

1. 提出方向方差比与正收缩比例指导的选层与随机 token 干预方法：从一次干净前向确定输入对应边界，在该边界随机 hard-zero 局部 token，用于迁移攻击优化。
2. 在四源 FPR13 上报告迁移 ASR，并通过选层、token 选择与预算对照检验关键设计。

动态观察融入第一项的动机。贡献不再按噪声与 drop 两个模块拆分。标题围绕 layer selection、random token dropping、transferable adversarial attacks 组织；正式名称另行确认。

## 3. 方法定义与完整算法

### 3.1 必要符号与定位统计

给定干净图像 $x$、真实标签 $y$ 和源模型，攻击约束为 $\|x_{\mathrm{adv}}-x\|_\infty\leq\epsilon$。源模型用于生成扰动，目标模型用于迁移评估。

每张干净图像在攻击前只做一次无 drop、无特征增强的完整前向，保存各原生 block 的局部输入和输出。将每个局部向量及对应全局参考分别在 float64 中做 L2 归一化（模型特征为 float32）。输入和输出分别减去各自的局部 token 中心，平均平方距离分别记为输入、输出分散度。

输出特征沿全局单位参考的投影总体方差，除以“输出分散度／原生特征维数”，得到方向方差比。正收缩比例为 `max(0, 1 - 输出分散度 / 输入分散度)`。两者乘积是唯一主线选层分数。输入或输出分散度为零时分数为零；不人为截断正的小分母。非有限值或零范数特征报错。全图没有正分数时也按原生顺序选择最早的 block。

每图选择最高分 block，并列选择原生前向顺序最早者；在该 block 的输入处 drop。该边界在全部攻击步与视图中固定。统计量仅依赖当前图像，无固定层标签、跨图校准、拟合系数或深度权重。原 float32 余弦分数的总体方差只作为所选层的记录字段，不再直接决定选层。

ViT/DeiT 使用最终归一化后的分类 CLS；CaiT 使用 class-attention 后、最终 LayerNorm 前 CLS；PiT 使用 stage 末端 CLS，最后 stage 使用最终归一化后的 CLS。架构差异在设置中简述、补充材料完整记录。细节见 [方法契约](prd_method_contract.md)。

### 3.2 单边界随机 token drop

设所选输入边界有 $N(x)$ 个局部位置，总比例为 $r$，则 $k(x)=\max(1,\operatorname{round}(N(x)r))$。每个攻击步 $t$、group $g$ 独立抽样：

$$
S_{t,g}(x)\sim\operatorname{Uniform}\{S\subseteq[N(x)]:|S|=k(x)\}.
$$

以 $M_{t,g}$ 表示所选位置的布尔 mask，按通道广播：

$$
\widetilde H_{\mathrm{in}}^{\ell^\star}=(1-M_{t,g})\odot H_{\mathrm{in}}^{\ell^\star}.
$$

清零所选 token 的全部通道，prefix 保持原值，token 数量与原生网格保持不变，后续计算从该状态继续。不同样本可选择不同边界；整批遍历边界并集时只作用于选择当前边界的样本。每张图、每个视图仍只有一个干预边界。

### 3.3 空间配对与优化

默认每组包含原始与反射平移视图。原始视图使用 $M_{t,g}$；平移视图使用其对应空间变换，并以原生网格 occupancy top-k 保持实际 drop 数量。同组两视图使用同一模型边界。平移梯度回传至当前像素 $x_t$。

以 $\mathcal L_{g,v}(x_t,y)$ 表示按照共同增强设置、对应 mask 完成的视图交叉熵。像素梯度直接平均，再以系数 1 累积动量：

$$
g_t=\frac{1}{GV}\sum_{g=1}^{G}\sum_{v=1}^{V}\nabla_{x_t}\mathcal L_{g,v}(x_t,y),\qquad
m_{t+1}=m_t+g_t.
$$

$$
x_{t+1}=\Pi_{\mathcal B_\infty(x,\epsilon)\cap[0,1]^d}
\left(x_t+\alpha\operatorname{sign}(m_{t+1})\right).
$$

默认 seed 42、$T=10$、$G=10$、$V=2$、$\epsilon=16/255$、$\alpha=\epsilon/T$。每图 100 次原始 mask 抽样；平移 mask 由对应变换获得。保留 `--input-diversity-views-per-group` 的 1/2 消融。

### 3.4 Algorithm 1 的逐步内容

| 阶段 | 算法操作 |
| --- | --- |
| 输入 | 干净图像、标签、源模型、扰动预算、步数、group/view 数和 drop 总比例 |
| 一次定位 | 无 drop、无增强噪声的干净前向；逐 block 局部输入／输出 → 方向比 × 正收缩比例 → 最高分 block → 对应输入边界 |
| 初始化 | 记录原生网格与 $k$，设置 $x_0=x$、$m_0=0$ |
| 每步每组抽样 | 在记录的边界全部局部位置均匀、无放回抽取 $k$ 个位置 |
| 空间配对 | 原始／平移视图及对应 mask，边界固定，drop 数量保持 |
| 视图前向 | 按共同增强设置构建前向，在选定输入边界 hard-zero，再继续计算真实标签交叉熵 |
| 梯度聚合 | 求各视图关于当前像素的梯度并直接平均 |
| 像素更新 | 动量累积、sign 更新、L∞ 投影与像素截断 |
| 输出 | 最终对抗图像 |

正文算法将共同增强作为统一设置，不另设噪声机制或颜色方向公式。补充材料如实给出每个视图初始局部特征的增强执行过程。当前 i.i.d. 指 RGB 空间采样，经真实初始投影与 RMS 匹配后的特征可以相关。

## 4. 正文章节与段落任务

| 章节 | 作用 | 产物 |
| --- | --- | --- |
| Title / Abstract | 压缩中心问题、方法与验证 | 方向比与正收缩乘积选层＋随机 drop＋实际最终配置的迁移结果；名称另定 |
| 1. Introduction | 从表示演化建立干预深度问题，概括设计与贡献 | 动态观察 → 层内分布 → 选层＋随机干预 → ASR；承接 Fig. 1 |
| 2. Related Work | 定位干预深度与 token 处理方式 | token/patch attacks、model randomization、中间表示与迁移攻击 |
| 3. Method | 解释统计定位、随机干预和完整优化 | 四个小节，见下表 |
| 4. Experiments | 用 ASR 检验完整方法和关键设计 | 四个小节，见下表 |
| 5. Conclusion | 收束设计与已验证结果 | 表示统计定位干预深度，随机 token 干预与迁移 ASR |
| References / 投稿要求部分 | 支持定位与复核 | 核对原论文；适用范围与研究用途按正式投稿要求安排 |

### 4.1 方法章

| 小节 | 自然段的任务顺序 | 公式与图表 |
| --- | --- | --- |
| 3.1 Preliminaries | 简述任务与预算；引入局部表示和全局参照的必要符号 | 攻击约束，尽快进入设计 |
| 3.2 Depth-wise Local–Global Alignment | ①真实样本的跨层关联变化；②局部关联分布与层内方差，将观察连接到干预深度问题 | Fig. 1、分数和方差定义 |
| 3.3 Direction Variance and Contraction-Guided Layer Selection and Random Token Drop | ①逐图选层与峰前输入边界；②均匀随机集合与 hard-zero；③边界固定、位置重采样及空间对应的执行关系 | 方向比与正收缩乘积规则、随机集合与 mask 公式；方法主体篇幅 |
| 3.4 Attack Optimization | ①空间输入配对及 mask 对应；②视图损失、梯度平均与更新，串联完整算法 | Fig. 2、Algorithm 1、梯度与更新式 |

§3.2 保持紧凑，服务 §3.3 的设计动机。共同增强的详细构造放入补充材料。多架构原生结构和实现细节不占用核心论证篇幅。

### 4.2 实验章

| 小节 | 要回答的问题 | 对照要求 |
| --- | --- | --- |
| 4.1 Experimental Setup | 哪些输入、标签、source/target、权重、预处理、随机设置、预算与 ASR 定义？ | 说明每源总比例和原生网格；共同增强细节见补充材料 |
| 4.2 Main Transfer Results | 最终方法在各源、Transformer/CNN 目标和严格黑盒下的 ASR 如何？ | 同协议外部比较，记录完整计算预算 |
| 4.3 Layer Selection | 当前规则相对历史方差、随机选层及共同底座的效果如何？ | 相同增强、比例、step/group/view，保留真实执行标签 |
| 4.4 Token Dropping and Additional Analysis | 层内选 token、固定/重采样、drop 比例与视图预算的影响？计算代价如何？ | 核验各比较对应配置；成本包括一次干净选层和全部视图前后向 |

最低优先选层矩阵：无 drop、每图随机选一层后保持、历史最大方差与当前乘积选层。强固定单层基线已按用户要求退出。多尺度源报告所选网格、实际 drop 数和比例。该矩阵直接检验中心贡献，先于扩展参数扫描。

## 5. 图表结构

正文优先两张核心图、三张主要表；分析图按证据与篇幅安排。架构配置在 §4.1 简述，完整表放补充材料。

| 标识 | 具体内容 | 论证作用与材料状态 |
| --- | --- | --- |
| Fig. 1 / `FIG1_ALIGNMENT_VARIANCE` | 真实样本的跨层 patch-score 热图、全部层原始方差诊断、方向比与收缩及乘积分数曲线、最高分 block 与其输入边界；必要时加固定 token 曲线 | 把动态观察直接连接到定位规则。已有 `figure1.py` 可导出 ViT 原始分数与热图；方差曲线和完整编排待准备 |
| Fig. 2 / `FIG2_LAYER_RANDOM_DROP` | 上支路一次干净选层，下支路迭代攻击；单边界随机 mask、空间配对、hard-zero、梯度汇合 | 显示边界固定和位置重采样。当前单边界流程图待准备，旧固定双 checkpoint 脚本已删除 |
| Algorithm 1 | §3.4 的完整步骤 | 明确定位仅一次、峰前干预、实际预算与更新 |
| Table 1 | 最终 i.i.d. 配置的四源迁移主结果与同协议比较，主指标为 strict ASR | 当前已有 opponent 动态四源记录，最终配置与外部比较需核对或补齐 |
| Table 2 | 无 drop、随机选层、历史方差与当前乘积选层 | 匹配设置，直接验证选层；完整最终矩阵待准备 |
| Table 3 | 高/低分与全位置随机，固定/重采样策略 | 已有固定 block10 的六组结果；标明实际增强、groups 和边界，最终配置对照待核对 |
| Fig. 3（可选） | drop 比例—ASR，或 ASR—完整计算预算 | 展示参数与成本，按最终配置准备 |

热图可以逐层 min–max 归一化显示，色条标为 Normalized patch score。方差诊断使用原始余弦分数；当前选层另计算归一化特征几何与收缩。显示 min–max 不进入选层。图中区分观测 block 输出与干预 block 输入。选层频次如实呈现，跨源差异和完整分布放补充材料。

图表暂缺时遵循既有 `todonotes` 正文高亮约定，使用上表唯一标识及具体内容，纳入 `\listoftodos`。不在本地生成论文 LaTeX 文档。

## 6. 证据层级与已有材料

### 6.1 方法与协议依据

| 材料 | 用途 |
| --- | --- |
| `AGENTS.md`、`experiments/prd_paper_story.md` | 当前主线、叙事与写作约定 |
| `progressive_attack.py`、`nets/*.py`、`tests/test_dynamic_checkpoints.py` | 一次定位、原生边界、随机 mask、hard-zero 与更新行为 |
| `experiments/prd_method_contract.md`、`main.py` | 执行事实、当前默认与显式 i.i.d. 设置 |
| `transfer_eval.py`、`experiments/fpr_target_protocol.md` | FPR13 权重、预处理和全部样本 ASR |
| `experiments/server_handoff_inventory.md` | 本地原始材料与可移交状态 |

### 6.2 当前方法与历史动态选层评估

当前唯一主线方法已完成 opponent、seed-42、每源 1000 图 FPR13 测量，见 [结果](../results/prd_direction_contraction_s42_1000_fpr/README.md)。独立实验于 2026-11-04 迁入主线，保留测量参数和源码哈希；移除固定层引用后的归档维护见 provenance。

| Source | Overall ASR | Strict ASR |
| --- | ---: | ---: |
| vit | 87.95% | 86.95% |
| cait | 89.14% | 88.37% |
| pit_t | 84.28% | 84.28% |
| deit_b | 95.30% | 95.30% |
| macro | 89.17% | 88.72% |


以下是历史最大方差选层的每源 1000 图像、13 目标原始记录，仍开启 opponent 增强。下表采用与当前参照定义一致的 CaiT 后续运行。

| Source | Overall ASR | Strict ASR | `outputs/attack/` 下运行目录 |
| --- | ---: | ---: | --- |
| ViT-B/16 | 88.13% | 87.15% | `prd_dynamic_variance_s42_20261029T031203Z_vit/` |
| CaiT-S24 | 83.22% | 82.34% | `prd_cait_preln_variance_s42_20261029T070204Z/` |
| PiT-T | 83.60% | 83.60% | `prd_dynamic_variance_s42_20261029T031203Z_pit_t/` |
| DeiT-B distilled | 94.38% | 94.38% | `prd_dynamic_variance_s42_20261029T031203Z_deit_b/` |

每个目录包含 1000 条选层记录、13 行 `transfer.csv`（每目标 total=1000）、参数和汇总 JSON。早期动态 CaiT 的参照不同，使用后续 `preln` 运行对应当前实现。上述目录在服务器本地，不随 Git 提交；最终主表所需轻量记录在明确配置后另行归档。

### 6.3 保留的辅助证据

| 材料 | 已有观察 | 用途 |
| --- | --- | --- |
| 六组 token 选择，10 groups | high-random/low-random/random strict 87.28/87.19/87.24%；random-fix 87.11% | 按实际固定 block10 配置展示层内策略与重采样的直接差值 |
| 六组 token 选择，1 group | random/random-fix strict 75.23/75.35% | 如实呈现视图预算下的差异 |
| 历史最大方差 ViT 频次 | 996/1000 张选中 block10 | 展示真实频次，与当前选择器区分 |
| 其他历史动态选层 | 二阶矩、标准差加绝对均值、仅方向比 | 保留已执行 ASR 与来源，不提供执行选项 |

原件见 [token 证据](prd_token_selection_evidence.md) 和 [结果索引](../results/README.md)。旧固定双／多 checkpoint 算法的结果、消融与专属分析，以及强固定单层基线已经删除，不作为论文材料或待维护路径。

历史随机实现的 seed 42 与当前 PyTorch seed 42 按用户约定统一报告；checkpoint、比例或实际增强分布变化需要对应配置的测量。

## 7. 实验协议与配置记录

论文四源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled；PiT-B 与 Visformer 为目标模型。FPR13 为 7 Transformer＋6 CNN，源与目标权重、预处理、数据顺序和预算在比较时一致。

| Source | 观察范围 | 单图总 drop 比例 | 所选边界的局部网格 |
| --- | --- | ---: | --- |
| ViT-B/16 | 12 blocks | 20/196 | 14×14 |
| CaiT-S24 | 24 个局部 blocks | 30/196 | 14×14 |
| PiT-T | 全部 2+6+4 blocks | 30/196 | 27×27、14×14 或 7×7 |
| DeiT-B distilled | 12 blocks | 20/196 | 14×14 |

按边界原生网格计算实际丢弃数；每图只报告所选单边界的实际数量。最终对照保持总比例及共同设置一致，并报告整数化后的实际预算。

ASR 为全部评估对抗样本的 $1-\mathrm{accuracy}$。Overall 对 13 目标平均；strict 只排除完全相同源目标，ViT/CaiT 为 12 目标，PiT-T/distilled DeiT-B 为 13 目标；四源等权汇总。迁移性与组件效果通过 ASR 和直接差值描述，梯度诊断仅用于内部分析。

## 8. 补充材料

| 部分 | 内容 |
| --- | --- |
| A. Implementation and augmentation | 完整算法、共同 i.i.d. RGB 随机特征增强的采样/投影/RMS/注入位置、空间 mask 映射、动量与保存设置 |
| B. Architecture-specific definitions | 全局参照、原生 block/输入边界、prefix、PiT pooling 与网格、逐样本实际 drop 数 |
| C. Reproducibility and evaluation | 样本、标签、权重、环境、seed、batch、完整参数和原始 ASR 表 |
| D. Extended observation and layer selection | 更多真实样本、原始方差曲线、选层频次与跨源分布 |
| E. Extended ASR comparisons | 逐目标结果、选层与 token 策略、参数和预算分析，配置对应清楚 |
| F. Qualitative cases and cost | 对抗样本、成功/失败案例与完整计算成本 |

## 9. 准备优先级与写作顺序

| 顺序 | 任务 | 完成标准 |
| --- | --- | --- |
| 0 | 确认选层定义、Fig. 1 和最终配置对应关系 | 原始方差、峰值输出与干预输入一致；i.i.d. 与旧记录标签清楚 |
| 1 | 写 §3.2–§3.3 | 动态观察进入统计选层与随机 drop；定义、符号可复核 |
| 2 | 写 §3.1、§3.4、Algorithm 1 与 Fig. 2 方案 | 一次定位、位置采样、空间配对与像素更新完整连通 |
| 3 | 实验设置、最终四源主表与选层矩阵 | 对应实际 i.i.d. 配置，底座与预算匹配；关键结果有原始记录 |
| 4 | token 选择、比例与成本分析 | 结论按直接 ASR 差值组织；现有与最终配置明确对应 |
| 5 | Related Work | 从原论文/实现核对干预深度、token 处理和优化差异 |
| 6 | Introduction | 问题 → 动态观察 → 方向比与正收缩乘积定位与随机干预 → ASR → 两点贡献 |
| 7 | Conclusion、Abstract 与标题 | 只压缩经过验证的最终方法与结果，名称另行确认 |
| 8 | 补充材料与全文核对 | 图、算法、配置、术语、记录和指标一致 |

证据准备优先：最终 i.i.d. 四源记录；无 drop/随机选层/历史方差及当前乘积选层；Fig. 1 原始曲线与新版 Fig. 2。扩展参数扫描按中心问题和篇幅取舍。历史数据仅作为已执行配置的证据，执行代码只保留当前动态方法。

## 10. 文献定位与篇幅

Related Work 围绕 PatchOut/PNA 的局部随机化、Self-Ensemble 的多深度使用、TGR 的 token 梯度处理、FPR 的前向表示处理、Ghost Networks 的模型随机化以及输入多样性方法组织。精读原文与实现，核对干预位置、层的确定方式、token 操作及计算预算后再写差异。保留这些对象作为阅读清单，外部增益只在已有同协议结果时写入。

暂按 8 页正文规划，图表计入章节篇幅，正式篇幅与投稿日期另行核对。优先保证 Fig. 1、选层方法、主结果和选层对照的空间，压缩实现细节与扩展分析。

实际撰写严格按 `AGENTS.md` 逐自然段推进：一次一个英文段落及忠实中文翻译，讨论确定后继续。待补图表使用正文可见的 todonotes 和统一清单；论文正文不写入本地文件。
