# CVPR 2027 写作蓝图：GDI 的有效干预层与迁移攻击

更新日期：2026-11-05。方法名为 GDI（Geometry-Guided Depth Intervention）。沿用既有正文章节与小节结构，重构各部分的论证任务、图表职责和证据安排。论文自然段按 `AGENTS.md` 在对话中逐段交付。

## 1. 研究目标、主张与故事

目标：选择模型内部对迁移攻击有效的输入干预层，生成具有更强迁移效果的对抗扰动。

两项创新：

1. **有效干预层的发现：**模型一次前向传播中，一些层可能承担较重要的信息汇总作用；攻击迭代中干扰这些层的输入可以获得较好的迁移攻击效果。该项贡献通过相应干预的 ASR 证据展开。
2. **模型无关、样本相关的定位：**根据一次干净前向的特征统计，为每张图像计算干预层。GDI 使用方向方差比与正收缩比例的乘积定位，并在所选层输入处随机清零局部 token，以匹配选层对照检验定位收益。

叙事顺序：**干预深度影响迁移效果 → 有效层及其表示特征 → 逐样本定位 → 输入随机干扰 → 迁移 ASR。**

“可能承担信息汇总作用的层”用于表述观察与动机。全文重点落在这些层的迁移攻击价值以及计算方式的定位效果。跨模型统一采用同一统计原则，架构适配提供原生局部特征、参考和输入边界。

## 2. 保留的正文章节结构

| 章节 | 核心任务 | 读者获得的认识 |
| --- | --- | --- |
| Title / Abstract | 压缩问题、两项创新和最终配置的迁移结果 | GDI 选择有效干预层以增强迁移攻击 |
| 1. Introduction | 从攻击迁移性切入，提出干预深度问题，引出有效层的发现与定位需求 | 为什么值得选择干预层、本文如何选择 |
| 2. Related Work | 梳理迁移攻击、内部表示干扰和层选择 | 本工作的研究位置 |
| 3. Method | 沿观察、定位、干预和优化解释 GDI | 方法定义及执行关系 |
| 4. Experiments | 以迁移结果和匹配对照检验两项创新 | 整体 ASR、定位收益及具体设置的效果 |
| 5. Conclusion | 对应两项贡献收束已验证结果 | 有效干预层与逐样本定位的实际价值 |

### 2.1 Method 小节与自然段任务

| 小节 | 段落顺序 | 公式、图表与连接 |
| --- | --- | --- |
| **3.1 Preliminaries** | ①攻击任务、source/target 与 L∞ 预算；②局部 token、全局参考、原生 block 输入／输出符号 | 攻击约束；为 §3.2–§3.3 提供必要符号 |
| **3.2 Depth-wise Local–Global Alignment** | ①真实样本的跨层局部—全局关联变化；②结合 block 前后局部分布观察可能承担信息汇总作用的层，并连接相应干预的迁移效果，提出自动定位需求 | Fig. 1；局部—全局余弦；ASR 证据对应 §4.3 |
| **3.3 Direction Variance and Contraction-Guided Layer Selection and Random Token Drop** | ①定义方向方差比；②定义正收缩比例及乘积分数，逐图选择输入边界；③均匀抽样、hard-zero，以及边界固定和位置重采样 | 定位公式与随机干预公式；方法主体 |
| **3.4 Attack Optimization** | ①简洁说明共同增强与空间配对；②给出视图梯度平均、动量和投影更新，串联完整 GDI | Fig. 2、Algorithm 1、更新式 |

§3.2 的观察引出有效干预位置问题，§3.3 给出计算答案，§3.4 说明如何完成攻击。基础增强和优化手段使用紧凑篇幅，详细构造进入补充设置。

### 2.2 Experiments 小节与自然段任务

| 小节 | 段落顺序与问题 | 对应证据 |
| --- | --- | --- |
| **4.1 Experimental Setup** | ①数据、四源与 FPR13 目标；②共同攻击预算、每源比例、权重与预处理；③ASR、strict 与汇总方式 | 简洁设置表；完整参数放补充材料 |
| **4.2 Main Transfer Results** | ①四源整体与逐目标迁移效果；②按 Transformer/CNN、strict 及跨源结果展开主要观察 | Table 1；同协议外部比较 |
| **4.3 Layer Selection** | ①结合表示观察和干预 ASR 呈现有效层的价值；②匹配条件下比较无 drop、随机选层、历史方差准则与 GDI；③结合分数组成及实际选层分布解释定位效果 | Table 2、Fig. 1 相关干预结果；完整频次放补充材料 |
| **4.4 Token Dropping and Additional Analysis** | ①层内 token 操作与固定／重采样；②drop 比例与视图预算；③一次选层及迭代攻击的完整成本 | Table 3；可选预算图 |

第一项创新由有效层的表示观察及输入干预后的 ASR 支撑，第二项创新由匹配选层与分数组成对照支撑。最终 i.i.d. 配置的证据准备优先服务这两项任务。

## 3. 核心方法定义与公式

### 3.1 攻击任务与局部—全局关联

给定干净图像 $x$、真实标签 $y$ 和源模型，生成满足 $\|x_{\mathrm{adv}}-x\|_\infty\leq\epsilon$ 的对抗图像，以其在目标模型上的 ASR 评估迁移效果。

每张图像先做一次无 drop、无特征增强的干净前向。记 block $\ell$ 的局部输入、输出为 $H_\ell^{\mathrm{in}}$、$H_\ell^{\mathrm{out}}$，原生特征维数为 $d_\ell$，局部位置数为 $N_\ell$。局部向量与全局参考分别 L2 归一化，记为 $u_{\ell,i}^{\mathrm{in}}$、$u_{\ell,i}^{\mathrm{out}}$ 和 $g_\ell$。用于观察的局部—全局关联为 $s_{\ell,i}=\langle u_{\ell,i}^{\mathrm{out}},g_\ell\rangle$。

真实样本的固定位置跨层曲线展示关联随深度的变化，同层 token 分布展示局部关联的差异。Fig. 1 将这些观察与 block 前后分布变化及相应干预效果连接起来。

### 3.2 模型无关、样本相关的定位准则

输入与输出分别围绕各自局部中心计算分散度：

$$
D_\ell^{a}=\frac{1}{N_\ell}\sum_i\|u_{\ell,i}^{a}-\bar u_\ell^{a}\|_2^2,
\qquad a\in\{\mathrm{in},\mathrm{out}\}.
$$

输出沿全局单位方向的总体方差、方向方差比和正收缩比例为：

$$
V_\ell=\frac{1}{N_\ell}\sum_i
\big\langle u_{\ell,i}^{\mathrm{out}}-\bar u_\ell^{\mathrm{out}},g_\ell\big\rangle^2,
\qquad R_\ell=\frac{V_\ell}{D_\ell^{\mathrm{out}}/d_\ell},
\qquad C_\ell=\max\left(0,1-\frac{D_\ell^{\mathrm{out}}}{D_\ell^{\mathrm{in}}}\right).
$$

最终分数与选择为：

$$
S_\ell=R_\ell C_\ell,\qquad
\ell^\star(x)=\operatorname*{arg\,max}_{\ell}S_\ell(x).
$$

方向方差比描述全局方向上的局部差异相对于平均方向方差的大小，正收缩比例描述经过该 block 的局部分布聚拢程度。二者结合，用于定位可能承担信息汇总作用、具有迁移干预价值的层。定位效果由 §4.3 的 ASR 比较展开。

计算使用 float32 特征转 float64 后的归一化几何。输入或输出分散度为零时得分为零；并列及全零时取原生前向顺序最早者。在所选 block 的输入处干预，全部攻击步与视图共用该边界。完整数值规则与架构定义见 [方法契约](gdi_method_contract.md)。

### 3.3 所选层输入处的随机干扰

设所选输入边界有 $N(x)$ 个局部位置，比例为 $r$，令 $k(x)=\max(1,\operatorname{round}(N(x)r))$。每个 step $t$、group $g$ 均匀、无放回抽取：

$$
\mathcal S_{t,g}(x)\sim\operatorname{Uniform}\{\mathcal S\subseteq[N(x)]:|\mathcal S|=k(x)\},
\qquad
\widetilde H_{\ell^\star}^{\mathrm{in}}=(1-M_{t,g})\odot H_{\ell^\star}^{\mathrm{in}}.
$$

mask 沿通道广播，所选局部 token 全通道置零，prefix 与原生网格保持。模型从干扰后的输入继续前向。批内图像各自使用所选边界；执行按边界并集遍历，并对相应样本施加 mask。

### 3.4 共同优化流程

各视图按共同设置施加随机特征增强，同组原始／phase 视图使用空间对应、数量一致的 mask。各视图损失关于当前像素的梯度直接平均，以系数 1 累积动量并做 sign 投影更新：

$$
g_t=\frac{1}{GV}\sum_{g=1}^{G}\sum_{v=1}^{V}\nabla_{x_t}\mathcal L_{g,v}(x_t,y),
\qquad m_{t+1}=m_t+g_t,
$$

$$
x_{t+1}=\Pi_{\mathcal B_\infty(x,\epsilon)\cap[0,1]^d}
\left(x_t+\alpha\operatorname{sign}(m_{t+1})\right).
$$

默认 seed 42、$T=10$、$G=10$、$V=2$、$\epsilon=16/255$、$\alpha=\epsilon/T$。每图有 100 次原始 mask 抽样。增强构造、phase 映射和参数集中写入补充实现设置。

### 3.5 Algorithm 1 的内容

| 阶段 | 操作 |
| --- | --- |
| 输入 | 图像、标签、源模型、扰动预算、步数、视图预算和 drop 比例 |
| 逐图定位 | 干净前向采集原生 block 输入／输出；计算 $R_\ell$、$C_\ell$、$S_\ell$；记录 $\ell^\star$ 输入边界及 $k$ |
| 初始化 | $x_0=x$、$m_0=0$ |
| 每步每组 | 在该图所选边界抽取 $k$ 个位置，构造视图及对应 mask |
| 攻击前向 | 应用共同增强，在所选 block 输入处 hard-zero，继续前向计算交叉熵 |
| 更新 | 平均视图梯度，累积动量，执行 sign 更新和投影 |
| 输出 | 最终对抗图像 |

## 4. 图表内容、作用与准备状态

| 标识 | 内容与布局 | 服务的主张 | 材料状态 |
| --- | --- | --- | --- |
| **Fig. 1 / `FIG1_ALIGNMENT_VARIANCE`** | 实际样本的原图与跨层关联热图；对应方向比、输入／输出分散度、正收缩和乘积分数曲线；配合相应干预的 ASR 比较。标明选定输出 block 及其输入边界 | 有效干预层的表示特征与迁移价值，推动自动定位问题 | `figure1.py` 已提供前 50 张 ViT 样本的原始余弦分数及热图；几何曲线、对应 ASR 及整图编排待准备 |
| **Fig. 2 / `FIG2_LAYER_RANDOM_DROP`** | 上支路一次干净前向定位，下支路迭代攻击；突出逐图边界、随机 mask 和输入处清零，简洁表示梯度更新 | 定位如何转化为 GDI 攻击 | 当前方法流程图待准备 |
| **Algorithm 1** | 一次定位、每组抽样、所选边界输入干预、共同优化 | GDI 的完整执行关系 | 代码和本节步骤可直接用于撰写 |
| **Table 1 / `TAB1_TRANSFER`** | 最终 i.i.d. 四源 FPR13 逐目标、Overall、strict 及同协议方法比较；正文优先 strict | 整体迁移攻击效果 | GDI 四源测量已完成并索引；外部比较按实际记录整理 |
| **Table 2 / `TAB2_LAYER_SELECTION`** | 匹配设置的无 drop、逐图随机层、历史最大方差与 GDI；按证据补充方向比、收缩及乘积分数组成比较 | 有效深度的攻击价值和计算准则的定位效果 | 历史配置已有部分准则比较；最终 i.i.d. 完整矩阵待准备 |
| **Table 3 / `TAB3_TOKEN_DROP`** | 层内 token 策略、固定／重采样及直接 ASR 差值，标明实际边界、增强和视图预算 | 具体干预设置的效果 | 历史固定 block10 数据已保留；最终配置对照按需准备 |
| **Fig. 3（可选）/ `FIG3_BUDGET`** | drop 比例或 ASR—完整计算预算 | 方法对预算的响应及实际成本 | 随中心证据完成进度安排 |

Fig. 1 展示原始观察和迁移效果之间的联系，Table 2 承担完整数值比较。热图逐图逐层 min–max 归一化仅用于颜色显示，曲线保留各统计量的实际定义和尺度。架构全配置、实际选层频次、更多样本及逐目标扩展表放补充材料。

图表待补时，在对应段落后使用 `\todo[inline,color=yellow!30,caption={...}]{...}`，写明本表唯一标识和具体内容，并纳入 `\listoftodos`；沿用主文件已有 `todonotes` 配置。

## 5. 实验与证据对应关系

| 待回答的问题 | 比较方式 | 主要归属 |
| --- | --- | --- |
| GDI 的迁移攻击效果如何？ | 最终配置四源 FPR13 与同协议方法 | §4.2 / Table 1 |
| 选择干预深度带来多少收益？ | 共同设置下无 drop、逐图随机层与有效层定位的 ASR | §4.3 / Table 2 |
| 所提计算方式能否更好定位？ | 共同设置下历史方差、分数组成与乘积准则的 ASR 及直接差值 | §4.3 / Table 2 |
| 哪种 token 干预设置适合该流程？ | 明确实际边界的 token 策略、固定／重采样比较 | §4.4 / Table 3 |
| 效果与预算的关系如何？ | drop 比例、视图数及包含干净定位的完整成本 | §4.4 / 可选 Fig. 3 |

共同条件包括样本、权重、预处理、增强、比例、steps、groups/views 和评估协议。当前优先补齐最终 i.i.d. 的选层矩阵；历史结果按各自执行配置组织，数值与参数由 [结果索引](../results/README.md) 管理。

### 5.1 已完成的最终 i.i.d. 测量

| Source | Overall ASR | Strict ASR |
| --- | ---: | ---: |
| ViT-B/16 | 87.41% | 86.36% |
| CaiT-S24 | 89.75% | 88.98% |
| PiT-T | 84.85% | 84.85% |
| DeiT-B distilled | 95.37% | 95.37% |
| 四源等权平均 | **89.3423%** | **88.8893%** |

测量使用 seed 42、每源 1000 图、10 steps、10 groups × 2 views、i.i.d. RGB 高斯特征增强和配对 mask，见 [主结果归档](../results/gdi_iid_paired_s42_1000_fpr/README.md)。

### 5.2 历史选层及 token 证据

当前乘积准则在历史 opponent 设置下的四源 Overall 为 89.1673%、strict 为 88.7234%；相对同设置最大方差选择器为 +1.8346、+1.8543 个百分点。[归档](../results/gdi_direction_contraction_s42_1000_fpr/README.md) 保留逐源、逐目标、实际选层频次及来源。

仅方向比、二阶原点矩、标准差加绝对均值及归一化变体由 [结果索引](../results/README.md) 管理。固定 block10 的历史 token 对照见 [token 证据](gdi_token_selection_evidence.md)。这些材料为对应配置的 ASR 证据，最终表格按实际条件安排。

## 6. 执行协议与材料位置

| Source | 候选范围 | 总 drop 比例 | 原生网格 | β |
| --- | --- | ---: | --- | ---: |
| ViT-B/16 | 12 blocks | 20/196 | 14×14 | 0.2 |
| CaiT-S24 | 24 局部 blocks | 30/196 | 14×14 | 0.2 |
| PiT-T | 2+6+4 blocks | 30/196 | 27×27、14×14、7×7 | 0.4 |
| DeiT-B distilled | 12 blocks | 20/196 | 14×14 | 0.2 |

FPR13 包含 7 Transformer 与 6 CNN。ASR 为全部对抗样本的 $1-\mathrm{accuracy}$；strict 排除完全相同源目标，ViT/CaiT 各 12 个目标，PiT-T/distilled DeiT-B 各 13 个目标；四源等权汇总。分析采用 ASR 和直接 ASR 差值。

| 材料 | 用途 |
| --- | --- |
| `attack.py`、`main.py`、`nets/*.py` | 当前 GDI 执行事实 |
| `tests/test_dynamic_checkpoints.py`、`tests/test_attack.py`、`tests/test_gdi_adapters.py` | 统计、原生边界、mask、梯度与更新验证 |
| [方法契约](gdi_method_contract.md) | 数值规则、架构参考、共同设置 |
| [FPR13 协议](fpr_target_protocol.md) | 目标、预处理和指标 |
| [交接记录](server_handoff_inventory.md) | 原始运行与文件位置 |
| [论文故事](gdi_paper_story.md) | 两项创新和中心叙事 |

## 7. 补充材料定位

| 部分 | 内容 |
| --- | --- |
| A. Implementation Settings | i.i.d. RGB 采样、初始投影、RMS 匹配、phase 映射、视图梯度与动量等共同设置 |
| B. Architecture-specific Definitions | 各模型全局参考、原生 block 输入边界、prefix、PiT pooling 与实际 drop 数 |
| C. Reproducibility and Evaluation | 样本、权重、环境、seed、batch、原始参数、ASR 与 provenance |
| D. Additional Layer Observations | 更多样本、几何曲线、实际选层频次和跨源分布 |
| E. Extended ASR Analysis | 逐目标、选层和 token 对照、参数与预算 |
| F. Qualitative Cases and Cost | 对抗图像、案例及完整计算成本 |

## 8. 准备优先级与写作顺序

| 顺序 | 任务 | 完成标准 |
| --- | --- | --- |
| 0 | Fig. 1 与选层证据准备 | 实际样本、统计曲线和对应 ASR 能围绕有效干预层共同展开 |
| 1 | §3.2–§3.3 | 从有效层观察进入统一定位准则与随机输入干预 |
| 2 | §3.1、§3.4、Algorithm 1、Fig. 2 | 任务、一次选层和迭代攻击完整连通 |
| 3 | §4.1–§4.3 | 最终配置主结果与匹配选层比较对应两项创新 |
| 4 | §4.4 | token 设置、必要预算和成本分析清楚对应执行记录 |
| 5 | Related Work | 从原论文和实现核对相关工作的操作与层选择方式 |
| 6 | Introduction | 以迁移效果串联问题、观察、定位及两项贡献 |
| 7 | Conclusion、Abstract、标题 | 围绕已写定的两项创新与实际结果压缩表达 |
| 8 | 补充材料与全文核对 | 术语、主张、图表、算法、参数和来源一致 |

## 9. 写作声音与交付方式

使用直接、流畅、肯定的语言，逐句推进研究主张及证据。涉及信息汇总时采用“可能承担信息汇总作用的层”，论述重心始终落在干预后的迁移效果。严格执行用户确认的禁止防御性表达要求。

基础增强和优化使用简洁说明，详细参数进入实现设置。Related Work 围绕 PatchOut/PNA、Self-Ensemble、TGR、FPR、Ghost Networks 和输入多样性等相关原论文及实现准备。正文暂按 8 页组织，正式投稿篇幅按官方要求核对。

每次交付一个英文自然段及忠实中文翻译，讨论确认后继续；公式在对话中正确渲染。文件保存规划与证据，论文正文由用户放入 LaTeX 编辑器。
