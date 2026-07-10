# Input Diversity 四种方案实验报告 (Final)

**日期**: 2026-07-10 | **GPU**: NVIDIA RTX 3080 | **白盒**: ViT-B/16  
**固定配置**: epsilon=16/255, steps=10, MI decay=1.0, TI=0, DIM=off  
feature_layer=12, token_score_cls_noise=learned  
HighScore, random sampling, ratio=0.3, ZeroToken + OpponentChannel, 100 samples

---

## 全体结果汇总

| # | 实验 | 方案 | Overall | ViT | CNN | Δ Baseline |
|---|------|------|---------|-----|-----|-------------|
| — | **Baseline (HighScore L12 20×1)** | — | 70.0% | 74.3% | 62.5% | — |
| 7 | **phase_pair 10×2 mean** | **一** | **77.6%** | **80.9%** | **72.0%** | **+7.6% ✅** |
| 8 | pairdiff 10×2 λ=0.05 | 三 | 76.4% | 79.9% | 70.2% | +6.4% ✅ |
| 9 | pairdiff 10×2 λ=0.10 | 三 | 73.0% | 76.4% | 67.0% | +3.0% |
| 4 | rotation α=0.05 | 四 | 69.6% | 73.9% | 62.3% | -0.4% |
| 5 | rotation α=0.10 | 四 | 67.9% | 73.4% | 58.2% | -2.1% |
| 1 | transport α=0.10 | 二 | 67.9% | 72.9% | 59.2% | -2.1% |
| 6 | rotation α=0.20 | 四 | 64.1% | 68.7% | 56.0% | -5.9% |
| 2 | transport α=0.20 | 二 | 65.2% | 69.0% | 58.5% | -4.8% |
| 3 | transport α=0.30 | 二 | 54.7% | 56.3% | 52.0% | -15.3% |

---

## 逐模型 ASR 对比

| 模型 | Baseline | 方案一 10×2 | Δ | 方案三 λ=0.05 | 方案四 α=0.05 | 方案二 α=0.10 |
|------|----------|-------------|---|---------------|---------------|---------------|
| cait_s24_224 | 89% | 87% | -2% | 88% | 91% | 88% |
| deit_base | 78% | **82%** | +4% | 80% | 75% | 74% |
| pit_b_224 | 72% | **82%** | +10% | 81% | 69% | 71% |
| tnt_s | 74% | **81%** | +7% | 77% | 73% | 71% |
| convit_base | 65% | **79%** | +14% | 78% | 68% | 66% |
| visformer | 72% | **79%** | +7% | 79% | 71% | 70% |
| levit_256 | 70% | 76% | +6% | 76% | 70% | 70% |
| **ViT avg** | 74.3% | **80.9%** | +6.6% | 79.9% | 73.9% | 72.9% |
| resnet101 | 65% | **73%** | +8% | 68% | 65% | 61% |
| inc_v4 | 62% | 71% | +9% | 71% | 64% | 63% |
| inc_v3 | 61% | **71%** | +10% | 71% | 65% | 56% |
| incRes_v2 | 62% | **73%** | +11% | 71% | 55% | 57% |
| **CNN avg** | 62.5% | **72.0%** | +9.5% | 70.3% | 62.3% | 59.3% |

---

## 详细分析

### 方案一 (P2): Phase Pair 10×2 — ✅ 大幅超越 Baseline

```
Overall: 77.6% (+7.6%) | ViT: 80.9% (+6.6%) | CNN: 72.0% (+9.5%)
```

**假设验证通过。** 将 20 个 view 的预算从「20 个独立 mask sample」重新分配到「10 组双 phase 对」，
Jacobian diversity 的收益远超 routing-mask diversity 的损失。

关键现象：
- **最受益的模型**：convit_base (+14%), pit_b_224 (+10%), incRes_v2 (+11%)
- **CNN 提升幅度 (9.5%) 甚至超过 ViT (6.6%)**，说明不同的 patch embedding Jacobian 产生了更通用的像素梯度方向
- **CaiT-S/24 微降 (-2%)**: 从 89% 降到 87%，但仍然是最高 ASR 的模型。CaiT 的 deeper CLS 机制使它已经对 patch routing 较少依赖

### 方案三 (P3): Pair-Difference Gradient

```
λ=0.05: 76.4% (+6.4%) | λ=0.10: 73.0% (+3.0%)
```

- λ=0.05 的 pair-difference 在 ViT 上接近方案一的 plain mean（79.9% vs 80.9%），但 CNN 差 2%
- **λ=0 即方案一的 plain mean**，pair-difference 是 plain mean + 正则化项，正则化降低了少许性能
- 结论：**plain pair mean 是此预算下的最优梯度聚合方法**

### 方案二 (P0): Cross-Patch Transport — ❌

```
α=0.10: -2.1% | α=0.20: -4.8% | α=0.30: -15.3%
```

rotate180 排列完全破坏了 ViT 的 token-position 路由。PiT-B 在 α=0.30 时跌至 43%。

### 方案四 (P1): Kept-Token Rotation — ❌

```
α=0.05: -0.4% | α=0.10: -2.1% | α=0.20: -5.9%
```

α=0.05 与 baseline 几乎持平（统计噪声范围），但无增益。更大的 α 单调下降。

---

## 最终结论

### 最优方法: 方案一 — 10×2 Phase Pair Plain Mean

```
--input-diversity-groups 10
--input-diversity-views-per-group 2
--input-diversity-phase-shift-set "4,4;8,8;12,12"
--input-diversity-pair-aggregation mean
```

### 核心发现

1. **Phase diversity beats mask diversity.** 在固定 20-view 预算下，用 10 对 phase-diverse view 替代 20 个独立 single view，获得 +7.6% overall ASR。patch-embedding Jacobian 的多样性远比额外的独立 mask sample 更有价值。

2. **Token-space perturbation (方案二/四) consistently hurts.** ViT 的 attention routing 对 kept-patch token 的 content-position / channel 扰动高度敏感，任何扰动都倾向于降低而非提升 transferability。

3. **CNN 从 phase diversity 中受益最大 (+9.5%)，** 暗示 phase-shifted patch embedding 产生的像素梯度在跨架构迁移时更具泛化性。

4. **Plain pair mean 优于 pair-difference mix.** 不需要额外的 g⁻ 正则化项——直接平均两个 phase view 的梯度已经足够。

5. **CaiT-S/24 体现了"上限效应"**——当 ASR 已经很高时（~89%），几乎所有方法都无法进一步提升它，收益主要体现在中等 ASR 模型上。
