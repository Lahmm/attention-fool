# 一个可严格保证有效的低 attention 数据增强方法

## 结论

在不对未知黑盒模型做任何假设时，不存在可以由有限实验结果推出的“对所有模型、所有样本、所有步长都一定提升攻击成功率”的数据增强。严格数学上能保证的对象必须是一个明确的 surrogate，例如黑盒代理模型上的一阶 loss 方向导数。

因此，当前可以严格成立的“必然有效”方法不是固定无条件打开某个随机增强，而是：

> **C-LABA: Certified Low-Attention Background Augmentation**  
> 使用 `DIM + low-attention background dropout/jitter/freq` 作为候选增强；每一步只在它通过方向导数证书时启用，否则回退 identity。由于 identity 在候选集中，C-LABA 在证书定义的 surrogate 上一定不差；证书为正时，一阶意义下严格更好。

在当前仓库已有实验里，C-LABA 的默认候选应取：

```text
DIM on
guide_aug_area = background        # 实际含义是 low-attention 区域 Q=1-M
guide_aug_method = dropout,jitter,freq
guide_aug_build = patch
attention_guide_type = qk_cls
layers = 0,1,4,9,11
MI on
normalize_grad off
```

对应当前最强经验配置：

```text
ours_dim_background_patch_pre_fpridx_no_normgrad_mifgsm_s40_500
avg transfer success = 0.8015
```

相邻 ablation 都低于它：foreground `0.76575`，all `0.76975`，去掉 DIM `0.4575`，普通 DIM-MI `0.66125`。因此实验结果给出的候选方向非常明确：**低 attention 区域的 dropout/jitter/freq 组合，加 DIM，加 MI，但不做全局梯度归一化。**

## 方法定义

记 high-attention guide map 为：

```text
M(x) in [0,1]
Q(x) = 1 - M(x)
```

这里 `Q` 就是当前代码中 `background` 实际作用的 low-attention 区域。

三个候选增强为：

```text
A_drop(x) = M x + Q[(1-a)x + a(0.5 U + 0.5 B_5 x)]
A_jit(x)  = M x + Q[(1+b)x + eta]
A_freq(x) = M x + Q[(1-a)x + a(0.7 B_9 x + 0.3 B_9 U)]
```

其中：

- `B_5` 是 `5x5 avg_pool`。
- `B_9` 是 `9x9 avg_pool`。
- `U` 是 uniform noise。
- `eta` 是 zero-mean white noise。
- `a` 是增强强度，当前实验为 `0.2`。
- `b` 是 per-image brightness jitter。

C-LABA 的候选增强集合为：

```text
cal A = {I, A_drop, A_jit, A_freq, A_djf}
```

其中 `I` 是 identity，`A_djf` 表示对 `dropout,jitter,freq` 的 loss 做平均：

```text
L_djf(x) = (1/3) sum_{m in {drop,jit,freq}} E_{T in DIM, a_m} CE(f(T(A_m(x; a_m))), y)
```

当前代码的 `guide_aug_method=dropout,jitter,freq` 与 `guide_aug_copies=3` 正是在近似这个期望。

## 证书

令源模型当前候选增强 `A` 产生的 MI 更新方向为：

```text
g_A = E_{T in DIM, aug}[grad_x CE(f_src(T(A(x))), y)]
m_A = mu m_{t-1} + g_A
u_A = sign(m_A)
```

取一组代理黑盒模型 `B = {b_1, ..., b_K}`。对每个代理模型：

```text
h_b = grad_x CE(f_b(x), y)
```

定义一阶方向导数：

```text
D_b(A) = <u_A, h_b>
```

若要强调当前实验发现的有效区域，也可以用 low-attention low/mid-frequency 版本：

```text
D_b^{Q,LM}(A) =
  <Q P_LM u_A, Q P_LM h_b>
```

其中 `P_LM = P_low + P_mid` 是低频和中频投影。

为了得到严格的不劣化规则，定义保守证书：

```text
C(A) =
  min_{b in B} [D_b(A) - D_b(I)]
  - z_delta * stderr_A
  - (alpha H / 2) * (||u_A||_2^2 + ||u_I||_2^2)
```

含义：

- 第一项要求所有代理模型上的方向导数都优于 identity。
- 第二项是随机采样误差的下置信界修正。
- 第三项是把二阶 Taylor 余项除以 `alpha` 后的保守扣除，`H` 是局部 Hessian 谱范数上界，`alpha` 是攻击步长。

C-LABA 选择：

```text
A* = argmax_{A in cal A} C(A)
if C(A*) > 0:
    use A*
else:
    use I
```

## 严格证明

对任意代理模型 `b`，假设交叉熵 loss 在当前 `epsilon` 邻域内二阶光滑，Hessian 谱范数不超过 `H`。Taylor 展开给出：

```text
L_b(x + alpha u_A, y)
= L_b(x,y) + alpha <u_A, h_b> + R_A
|R_A| <= (alpha^2 H / 2) ||u_A||_2^2
```

比较候选增强 `A` 与 identity：

```text
L_b(x + alpha u_A, y) - L_b(x + alpha u_I, y)
>= alpha [D_b(A) - D_b(I)]
   - (alpha^2 H / 2)(||u_A||_2^2 + ||u_I||_2^2)
```

如果 `C(A)>0`，则对所有代理模型 `b`，在扣除采样误差和二阶余项后仍有：

```text
L_b(x + alpha u_A, y) > L_b(x + alpha u_I, y)
```

也就是说，启用该增强比不增强在代理黑盒的一阶攻击目标上严格更好。

如果所有候选 `C(A)<=0`，C-LABA 回退 `I`，于是：

```text
L_b(x + alpha u_{C-LABA}, y) = L_b(x + alpha u_I, y)
```

因此 C-LABA 在该 surrogate 上**一定不差**，证书为正时**严格更好**。这就是数学上能成立的“必然有效”。

## 为什么默认候选是 dropout/jitter/freq，而不是别的增强

当前实验结果支持三个事实。

### 1. 最优结构稳定指向 low-attention background

最强实验：

```text
ours_dim_background_patch_pre_fpridx_no_normgrad_mifgsm_s40_500
avg = 0.8015
```

相邻替换均下降：

```text
foreground: 0.76575, delta = -0.03575
all:        0.76975, delta = -0.03175
nodim:      0.45750, delta = -0.34400
dim_mifgsm: 0.66125, delta = -0.14025
```

并且对 8 个黑盒模型的相邻 ablation delta 都是正的，不是单一模型偶然性。

### 2. 背景方向导数解释了优势

已有方向导数分析显示，最强 no-normgrad 相比 foreground gradnorm 的迁移方向导数增益：

```text
total delta <u,g_b> = +0.364804
foreground delta    = -0.024038
background delta    = +0.388842
```

符号说明优势来自 `Q=1-M` 的 low-attention 项，而不是 foreground 项。

### 3. dropout/jitter/freq 的频域作用正好匹配这个方向

忽略 clamp 后，三个增强的 Jacobian 频响为：

```text
dropout: H_drop(w) ~= 1-a + 0.5a H_avg5(w)
freq:    H_freq(w) ~= 1-a + 0.7a H_avg9(w)
jitter:  Jacobian 不选频，但加性噪声期望会平滑高频敏感 loss 分量
```

`dropout` 和 `freq` 都是 low-attention 区域的低通/中低频偏置；`jitter` 提供亮度和噪声平滑，使方向不依赖固定背景纹理。DIM 又通过随机 resize/pad 削弱位置绑定高频方向。所以这个组合正好把更新投到：

```text
low-attention + low/mid-frequency + transform-stable
```

这个公共子空间和已有实验中正的 background direction derivative 一致。

## 最终推荐方法

如果只需要当前代码能直接跑的经验最强方法，用：

```text
--dim
--mi
--mi-decay 1.0
--guide-aug
--guide-aug-area background
--guide-aug-method dropout,jitter,freq
--guide-aug-copies 3
--guide-aug-strength 0.2
--attention-guide-models deit_base_patch16_224,pit_s_224,cait_s24_224
--attention-guide-type qk_cls
--attention-guide-build-method patch
--layers 0,1,4,9,11
```

并且不要加 `--normalize-grad`。

如果要让“必然有效”在数学上严格成立，应加 C-LABA 的选择证书：

```text
每步比较 identity 与 dropout/jitter/freq candidate；
只在 C(A)>0 时使用增强；
否则使用 identity。
```

这样得到的是一个有保证的增强规则，而不是一个无条件随机开关。

## 下一步实现建议

1. 在攻击循环里加入 `--certified-guide-aug` 模式。
2. 每隔 `k` 步或每个 batch 估计候选 `A in {I, drop, jit, freq, djf}` 的 `D_b(A)-D_b(I)`。
3. 代理模型可先用已有 `attention_guide_models`，不额外引入外部模型。
4. 先用总方向导数 `D_b(A)` 做证书；验证稳定后再切到 `D_b^{Q,LM}(A)`。
5. 记录 `C(A)`、被选择的增强、以及后续 transfer success，检验证书通过率和真实迁移提升的相关性。

## 最短可验证命题

在当前实验上下文中，可以把要验证的数学命题写成：

```text
For A = background(dropout,jitter,freq) with DIM and no_normgrad,
if min_b [D_b(A)-D_b(I)] is positive after confidence and curvature correction,
then A is guaranteed to be at least as effective as identity on the proxy
black-box one-step CE objective; if positive, it is strictly more effective.
```

当前实验结果已经说明这个候选是所有已有配置中最强的；C-LABA 只是把“经验上最强”变成“证书通过才启用，因此 surrogate 上必然不差”的数学规则。
