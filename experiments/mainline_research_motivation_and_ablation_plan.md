# Patch-Score Routing Attack：研究动机、方法故事与后续实验计划

## 1. 研究定位

当前主线不是继续堆叠更多攻击技巧，也不是单纯追求 ASR 的局部提升，而是围绕两个具有明确职责分工的机制建立一套自洽的方法：

1. **Patch-score 引导的 patch drop：决定扰动应该放在哪里。**
2. **RGB opponent-channel 随机噪声：决定保留信息应该如何被扰动。**

两者分别对应攻击的空间选择和扰动形态。核心假设是：

> 对模型正在使用的语义证据做选择性擦除，同时对剩余证据施加具有颜色结构的随机扰动，比无差别 patch dropout 或无结构 feature Gaussian 更能打破模型对局部证据的稳定依赖。

可以将方法概括为：

```text
patch-score answers where to erase
opponent-channel noise answers how to perturb what remains
```

文章的主要价值应体现在机制解释、可检验假设和有针对性的消融上。ASR 仍然是迁移性的验证指标，但不再是后续工作的唯一或首要优化目标。

## 2. 为什么选择这两个组件

### 2.1 Patch-score：解决随机 patch drop 不知道“删哪里”的问题

普通 patch dropout 只控制删除数量，不回答哪些 patch 真正参与了当前分类判断。随机删除可能落在背景、冗余区域或对当前类别无关的区域，攻击预算因此被浪费。

当前方法从白盒模型的最终语义层提取：

```text
global token
local patch token
```

再通过二者的余弦相似度计算：

```text
patch-score = similarity(local patch, global representation)
```

直观上，高分 patch 的局部表示更贴近模型当前的整体判别语义。因此，攻击先从高分 patch 中构造候选集，再从候选集中随机抽取少量位置进行 drop。

这使 patch-score 不再只是一个排序步骤，而可以被解释为：

> 一个把有限攻击预算路由到模型判别性语义证据上的模型感知信息瓶颈。

不过，高 score 目前首先表示语义相关性，不应未经验证地直接等同于因果重要性。后续必须检查高分 patch 是否真的比低分 patch 更能改变 loss、logit、全局表示和模型预测。

### 2.2 Opponent-channel noise：解决普通 Gaussian 缺少输入结构的问题

普通 IID Gaussian 只能表达“加入随机扰动”，但没有利用 RGB 输入的颜色几何，也没有考虑模型第一层如何把 RGB 通道混合为初始特征。

当前默认噪声在三个 opponent-channel 方向上采样：

```text
亮度方向
红—绿对抗方向
黄—蓝对抗方向
```

随后通过模型真实的首层 RGB projection 映射到初始特征空间，并按照当前特征 RMS 进行尺度匹配。

它的研究动机是：

> 颜色对抗方向提供了比独立 RGB/feature Gaussian 更有结构的随机扰动坐标，而首层 RGB projection 使该坐标与模型自身的输入通道混合方式保持一致。

噪声只注入 kept tokens，使两个核心机制承担不同职责：

```text
patch-score：删除哪些语义证据
opponent noise：扰动剩余证据的颜色表达和局部表征
```

这使 opponent-channel noise 不只是“另一种随机数生成器”，而是一个具有输入空间解释和模型输入适配的结构化扰动机制。

## 3. 如何建立自洽的 motivation 故事

### Observation 1：局部 patch 对模型判别决策的作用并不均匀

一张图像中的 patch 并不都同等重要。有些 patch 对应主体、关键形状或有判别力的纹理，有些 patch 则是背景或冗余上下文。

因此，无差别随机 patch drop 往往无法稳定地破坏模型真正使用的证据。

### Observation 2：攻击应优先擦除模型正在使用的语义证据

最终层 global token 可以看作模型当前的整体判别语义，local patch token 则表示各个局部区域。二者的相似度可以作为局部区域与当前整体语义之间的相关性信号：

```text
global-local similarity
→ semantic relevance
→ patch routing
→ targeted evidence erasure
```

这构成 patch-score-guided patch drop 的 motivation。

### Observation 3：只擦除部分证据后，模型可能利用剩余冗余证据恢复判断

如果只删除少量高分 patch，模型仍可能从剩余 patch 中恢复原有类别判断。因此，除了删除核心证据，还需要扰动保留下来的证据。

### Observation 4：剩余证据应沿着有颜色意义的方向被扰动

直接使用各向同性 feature Gaussian 无法表达 RGB 输入的颜色结构。Opponent-channel noise 在亮度、红绿、黄蓝方向上施加扰动，并通过首层 RGB projection 进入模型特征空间，因此能够同时保留：

- RGB 颜色结构的解释性；
- 模型首层输入投影的适配性；
- 对不同架构的统一实现接口。

### 核心方法故事

当前方法可以被表述为：

> 模型的分类决策依赖一组分布不均匀的局部语义证据。我们首先利用最终语义层的 global-local 相似度定位模型当前依赖的高相关 patch，并进行选择性 patch drop，切断主要判别证据；随后，在保留的局部证据上施加 RGB opponent-channel 随机噪声，并通过首层 RGB projection 映射到模型特征空间，破坏模型对剩余证据的稳定表达。前者负责证据选择，后者负责证据表达扰动，二者构成互补的攻击机制。

## 4. 三个核心研究假设

### H1：语义路由假设

在相同 drop budget 下，patch-score 引导的 drop 比 random drop 更能改变模型的判别性特征和输出。

需要验证的证据包括：

- 高分 patch 被遮挡后 source loss 增幅更大；
- 真实类别 logit 下降更多；
- global feature 改变量更大；
- score 与单 patch occlusion importance 存在相关性；
- 该效果在不同白盒架构中具有一致性。

### H2：颜色结构假设

在相同 RMS 强度和相同注入位置下，opponent-channel noise 比 RGB Gaussian 或 feature IID Gaussian 产生更有结构、更稳定的特征扰动。

需要验证的证据包括：

- RGB/feature 协方差不同；
- 颜色方向的能量分布可解释；
- 初始 feature、global/local token 和 logit 的变化不同；
- 三个 opponent 方向对不同架构的影响具有一定稳定性；
- 首层 projection 对噪声效果具有必要性。

### H3：机制互补假设

patch-score drop 和 opponent-channel noise 作用在两个不同层面：

```text
patch-score drop：改变证据集合
opponent noise：改变证据表达
```

二者组合不应只是重复同一种扰动，而应表现出互补效果。需要通过严格的 2×2 或因子实验验证组合收益是否来自真正的机制协同。

## 5. 当前主线的组件分工

当前默认主线为：

```text
final-layer patch score
→ high-score candidate stochastic patch drop
→ original / phase-shifted view pair
→ kept-only opponent-channel noise at initial RGB projection
→ raw 20-view gradient mean
→ Gaussian gradient residual
→ MI-FGSM update
```

其中：

| 组件 | 当前作用 | 文章定位 |
|---|---|---|
| final-layer patch-score | 定位语义相关 patch | 核心机制 |
| high-score stochastic patch drop | 选择性擦除判别证据 | 核心机制 |
| opponent-channel noise | 扰动保留证据的颜色表达 | 核心机制 |
| initial RGB projection | 将 RGB 结构映射到模型特征空间 | opponent noise 的关键实现 |
| kept-only injection | 保持“删除”和“扰动”职责分离 | 核心消融点 |
| phase pair | 检查轻微空间错位下的稳定性 | 支撑组件 |
| 20-view raw mean | 聚合多种 mask/视图的共同梯度 | 支撑组件 |
| Gaussian residual | 补充空间平滑梯度 | 优化组件 |
| MI-FGSM | 进行最终扰动更新 | 优化器 |
| ASR | 检查迁移性 | 验证指标 |

phase pair 不是频域相位操作，而是对同一个 patch-drop 结果构造：

```text
View A：原始位置的遮挡图像
View B：将遮挡图像和 mask 同时做轻微像素平移后的视图
```

它的作用是减少梯度对固定 patch 边界和像素对齐方式的依赖，不应成为文章主要创新点。

## 6. 第一优先级：验证 patch-score 是否真的有语义路由价值

### 6.1 先做单 patch 因果验证

对每个 patch 单独遮挡，记录：

- 真实类别 logit 下降；
- cross-entropy loss 增加；
- global feature 改变量；
- 最终预测变化；
- local/global 相似度变化。

随后计算：

```text
patch-score 与真实 occlusion importance 的相关性
```

这是最关键的基础实验。如果二者没有关系，就不能直接把高 score 写成“因果重要性”，而应该更谨慎地称为“语义相关性路由信号”。

### 6.2 比较不同 patch 路由策略

在完全相同的 drop 数量下比较：

```text
high-score drop
low-score drop
random drop
score-weighted drop
extreme top-score drop
all-patch sampling
```

重点观察：

- source loss 增幅；
- global feature 变化；
- global/local 相似度变化；
- patch mask 的空间位置；
- 梯度方向和梯度稳定性；
- 跨白盒架构的一致性。

### 6.3 比较不同 score 层

当前主线使用最终语义层，但需要验证这一选择：

```text
早期层
中间层
最终层
多层平均
```

如果最终层最有效，可以支持“最终语义层更适合作为判别证据路由器”的论点。如果中间层更有效，应相应调整文章 motivation，而不是预先假定最终层一定正确。

### 6.4 扫描 drop budget

建议比较：

```text
5%、10%、15%、20%、30%
```

同时分析：

- score-guided 相对 random 的优势是否在小预算下更明显；
- drop 过多后是否失去选择性；
- 当前 15% 是否存在机制上的最佳区间；
- 不同模型 patch 网格下是否需要按 patch 数量校准。

### 6.5 分析 mask 稳定性

记录：

- 不同随机 seed 的 mask IoU；
- 不同迭代步之间的 mask IoU；
- 不同白盒模型之间的 mask overlap；
- high-score 区域是否集中在主体、边缘或纹理区域。

这可以判断方法是在寻找稳定的语义证据，还是只是在制造随机遮挡。

## 7. 第二优先级：验证 opponent-channel noise 的结构价值

### 7.1 基础噪声对照

必须控制相同 RMS、相同注入位置和相同 kept-only mask，比较：

```text
RGB IID Gaussian
RGB opponent-channel noise
feature IID Gaussian
projected opponent-channel noise
```

尤其要区分：

```text
RGB opponent noise，但不经过首层 projection
RGB opponent noise，并经过首层 projection
```

这样才能判断首层 RGB projection 是否是必要组成部分。

### 7.2 三个颜色方向的消融

分别比较：

```text
只用亮度方向
只用红绿方向
只用黄蓝方向
亮度 + 红绿
亮度 + 黄蓝
红绿 + 黄蓝
三者完整组合
```

分析：

- 哪个方向最影响初始特征；
- 哪个方向最影响类别 logit；
- 哪个方向对不同架构最稳定；
- 三个方向是否存在互补关系。

### 7.3 比较噪声注入位置

在相同噪声能量下比较：

```text
像素空间
initial RGB projection
中间 feature layer
最终语义层
```

如果 initial RGB projection 最稳定，可以形成如下方法论结论：

> 初始投影位置既保留 RGB 结构，又能通过模型自身的输入 projection 适配不同架构，是颜色结构噪声较合理的注入点。

### 7.4 比较噪声作用范围

比较：

```text
kept-only
dropped-only
all-token
high-score-only
low-score-only
```

kept-only 最符合当前的职责分离故事：patch drop 负责删除，opponent noise 负责扰动剩余证据。但该假设需要通过实验确认，不能只因为叙事方便而预设结论。

### 7.5 分析噪声的真实结构

记录噪声的：

- RGB 协方差矩阵；
- opponent-channel 协方差；
- 空间频谱；
- 三个方向的能量比例；
- 初始 feature 的 RMS 和 cosine change；
- global token、local token、logit 的变化。

这样才能证明 opponent noise 在机制上确实不同于普通 Gaussian，而不仅仅是更换了随机数生成方式。

## 8. 第三优先级：核心 2×2 因子实验

固定图片、标签、seed、epsilon、steps、view 数、MI-FGSM 和噪声 RMS，只改变 patch 路由和噪声类型。

| Patch 路由 | Noise | 实验目的 |
|---|---|---|
| Random | None | 基础 patch 扰动 |
| Score | None | 单独验证语义路由 |
| Random | RGB Gaussian | 普通输入噪声对照 |
| Random | Feature Gaussian | 普通特征噪声对照 |
| Random | Opponent | 单独验证颜色结构 |
| Score | RGB Gaussian | score + 普通噪声 |
| Score | Feature Gaussian | score + 无结构 feature 噪声 |
| Score | Opponent | 完整主线 |

最关键的比较是：

```text
score drop + IID Gaussian
score drop + opponent noise
random drop + opponent noise
```

它们分别回答：

- 提升主要来自 patch-score 还是噪声？
- opponent noise 是否独立有效？
- 两个机制是否存在真正互补？

进一步可加入：

```text
Score + Opponent，但 noise 注入 all-token
Score + Opponent，但不经过 RGB projection
Score + Opponent，但只保留单一颜色方向
```

## 9. 推荐的评价指标

ASR 不应被删除，但应放在机制指标之后。

### Patch-score 机制指标

- score 与 occlusion importance 的相关性；
- 高分/低分/随机 drop 的 loss 增量；
- 真实类别 logit 下降；
- global feature cosine change；
- mask 稳定性和跨模型 overlap；
- patch drop 前后 global/local 相似度变化。

### Opponent noise 机制指标

- RGB 和 feature 噪声协方差；
- 三个颜色方向的能量分布；
- 注入后初始 feature 的 RMS、cosine change；
- global/local token 变化；
- logit 变化；
- 不同架构之间的响应一致性。

### 梯度稳定性指标

- 多视图梯度与最终平均梯度的 cosine similarity；
- sign agreement；
- effective rank；
- 动量方向与当前梯度的 cosine similarity。

### 最终有效性指标

- source model loss/accuracy change；
- Transformer transfer accuracy/ASR；
- CNN transfer accuracy/ASR；
- 严格黑盒结果；
- 不同白盒模型之间的跨架构一致性。

ASR 应作为“机制是否具有迁移价值”的最终验证，而不是后续搜索过程中唯一的优化信号。

## 10. 实验控制原则

为了保证结论能够归因到两个核心机制，后续实验应尽量遵守：

1. 相同数据、相同 seed、相同扰动预算和相同迭代步数。
2. 不同噪声之间使用相同 RMS 或相同有效能量。
3. 不同 patch 路由之间使用完全相同的 drop 数量。
4. 固定 phase pair、view 数、gradient aggregation 和 MI-FGSM，减少支撑组件干扰。
5. 先用小规模数据做机制筛选，再在固定配置下用 1000 样本确认。
6. 报告 paired per-image difference、均值和置信区间，不只报告一个总体 ASR。
7. 对没有实际运行的模型或配置不做结果性表述。
8. 在声称“创新”或“首次”前进行完整相关工作核查；当前文档只定义方法假设，不预先宣称绝对优先权。

## 11. 后续不应继续扩展成主要创新的组件

以下组件应保持为优化器、稳定化手段或对照项：

- phase pair；
- Gaussian gradient residual；
- MI-FGSM；
- NI、DIM、TI；
- 增加更多随机 view；
- 继续细调 patch drop 比例来追逐 ASR；
- 大量缺少机制解释的 feature noise 变体。

它们可以参与消融，但不应和 patch-score、opponent-channel noise 平分文章贡献。

## 12. 建议的文章结构

### Introduction

提出两个问题：

1. 迁移攻击如何定位模型真正依赖的局部语义证据？
2. 在证据被选择后，如何施加具有输入结构解释的随机扰动？

### Motivation

说明：

```text
局部证据不均匀
→ random drop 不够有效
→ 需要 semantic routing
```

以及：

```text
剩余证据仍可恢复判断
→ 无结构 Gaussian 缺少输入几何
→ 需要 opponent-channel perturbation
```

### Method

依次介绍：

1. final-layer patch-score routing；
2. stochastic high-score patch drop；
3. kept-only opponent-channel noise；
4. initial RGB projection；
5. 多视图梯度估计和 MI-FGSM 优化。

### Experiments

按以下顺序组织：

1. patch-score 的因果/语义有效性；
2. opponent-channel 的结构有效性；
3. 两个机制的互补性；
4. 跨架构一致性；
5. ASR 迁移性验证。

## 13. 最终核心结论应该是什么

后续工作真正需要证明的不是“某个配置又提高了几个百分点 ASR”，而是以下四点：

```text
1. 高分 patch 确实更接近模型的判别性语义证据；
2. opponent-channel noise 确实具有不同于 Gaussian 的颜色/特征结构；
3. 两个机制分别作用于证据选择和证据表达；
4. 二者组合具有可解释的互补性。
```

只要这四点被实验验证，文章的 motivation、method 和实验结论就能够形成闭环，ASR 则作为方法具有迁移价值的最终佐证。
