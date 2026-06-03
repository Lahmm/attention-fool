# DIM 与低 attention 增强的频域梯度机制：数学实验设计与结论

## 目标

本文记录一个数学化的实验框架，用来回答三个问题：

1. 当前 DIM 主要改变哪些频域的攻击梯度。
2. 低 attention 区域上的 `dropout/jitter/freq` 增强分别改变哪些频域的攻击梯度。
3. 为什么 `DIM + low-attention background augmentation` 的组合在迁移攻击里有效，以及下一步可以从数学上推进什么。

这里的 `background` 指当前实现中的低 attention 区域，不是语义分割背景。记 guide map 为 `M(x) in [0,1]`，高值区域为 high-attention foreground，低值区域为 `Q(x)=1-M(x)`。

## 当前实现对应的算子

当前攻击中每个 loss term 的前向可以写成：

```text
ell(x) = CE(f(T_dim(A_R(x; a)); y), y)
g = grad_x ell(x)
```

其中：

- `T_dim` 是随机缩小 resize 后 zero padding 回原尺寸。
- `A_R` 是 guide augmentation；当区域为 background 时，

```text
A_bg(x; a) = M * x + Q * A_m(x; a)
```

`m` 是 `dropout/jitter/freq` 等方法。

对一次固定采样，梯度满足链式法则：

```text
g = J_A(x)^T J_T(z)^T grad_z CE(f(z), y)
z = T_dim(A_bg(x; a))
```

所以 DIM 和 guide augmentation 对梯度频谱的作用，本质上是它们的 Jacobian transpose 对上游模型梯度做了什么频域滤波、重采样和区域门控。

## 1. DIM 影响什么频域的梯度

当前 DIM 是：

```text
T_{s,t}(x) = P_t R_s x
```

`R_s` 是 bilinear resize，`s in [0.85, 1.0]`；`P_t` 是随机 padding/placement。对上游梯度 `h = grad_{T(x)} L`，反传到原图：

```text
g_dim = R_s^T P_t^T h
```

其中 `P_t^T` 是 crop，`R_s^T` 是 bilinear resize 的伴随算子。对二维 Fourier 频率 `omega=(omega_x,omega_y)`，bilinear 插值的主频响可近似写成：

```text
H_bilinear(omega) ~= sinc(omega_x / 2)^2 * sinc(omega_y / 2)^2
```

因此 `R_s^T` 对高频梯度天然有衰减。随机 offset `t` 不改变单次梯度的功率谱相位模长，但在多次 EOT/MI 期望中会让依赖固定空间相位的位置特异梯度互相抵消。于是 DIM 的数学作用不是简单“增强低频”，而是：

```text
E_{s,t}[R_s^T P_t^T h_{s,t}]
```

会保留在随机缩放和随机位置下相位稳定的低/中频方向，削弱高频、位置绑定、白盒局部纹理方向。

### 可证明的算子实验

不用模型即可先证明 DIM 算子的通带。对每个 Fourier 基函数：

```text
phi_omega[u,v] = exp(i <omega, (u,v)>)
```

计算：

```text
G_dim(omega) = E_{s,t} || R_s^T P_t^T P_t R_s phi_omega ||_2^2 / ||phi_omega||_2^2
```

预测：

- `G_dim(omega)` 随频率半径上升而下降。
- 高频处下降来自 bilinear resize 的低通性质和随机 placement 的相位不稳定。
- 如果把 `dim_resize_range` 改得更小，例如 `[0.6,1.0]`，高频衰减会更强，但可能损害白盒优化。

### 模型梯度实验

对同一批样本，估计：

```text
g_plain = grad_x CE(f(x), y)
g_dim   = E_{s,t} grad_x CE(f(T_{s,t}(x)), y)
```

用径向频带投影 `P_low, P_mid, P_high` 分解：

```text
rho_B(g) = ||P_B g||_2^2 / ||g||_2^2
C_B      = cos(P_B g_source, P_B g_blackbox)
D_B      = <P_B sign(m_source), P_B g_blackbox>
```

关键检验不是只看 `rho_B`，而是看 `D_B`。迁移成功的一阶近似由黑盒方向导数决定：

```text
L_b(x + alpha u, y) = L_b(x,y) + alpha <u, grad_x L_b(x,y)> + O(alpha^2)
```

如果 DIM 有效，应该看到 `D_low + D_mid` 的增益大于 `D_high` 的增益，尤其在 MI 后期更明显。

## 2. 低 attention 上的 dropout/jitter/freq 影响什么频域

下面忽略 clamp 的饱和边界，并把随机噪声看作与输入独立。对 background 区域，`Q=1-M`。

### dropout

当前代码里的 `dropout` 不是 Bernoulli dropout，而是：

```text
A_dropout(x) = (1-alpha) x + alpha * (0.5 U + 0.5 B_5 x)
```

其中 `U` 是 uniform noise，`B_5` 是 `5x5 avg_pool`。background blend 后：

```text
A_bg(x) = M*x + Q*A_dropout(x)
```

其 Jacobian 近似为：

```text
J_dropout ~= M I + Q[(1-alpha)I + 0.5 alpha B_5]
```

频域主响应为：

```text
H_dropout(omega) ~= 1-alpha + 0.5 alpha H_avg5(omega)
```

在低频 `H_avg5(omega) ~= 1`，在高频 `H_avg5(omega) ~= 0` 或振荡变小。因此 dropout 对 background 梯度的显式线性部分是低通偏置：低频保留更强，高频被乘上约 `1-alpha` 到 `1-0.5alpha` 的较小系数。随机 `U` 不直接贡献输入 Jacobian，但它改变 forward distribution，等价于对 background loss 做随机平滑，进一步降低对高频纹理的依赖。

### jitter

当前 `jitter` 是：

```text
A_jitter(x) = (1+b)x + eta
```

`b` 是每张图一个 brightness 标量，`eta` 是 white Gaussian-like noise。忽略 clamp：

```text
J_jitter ~= M I + Q(1+b)I
```

所以它的 Jacobian 本身没有固定频率选择性，只是按区域缩放所有频率。但从期望目标看：

```text
E_eta L(f(x + Q eta), y)
```

是只在 low-attention 区域做随机平滑。若局部近似为加性高斯噪声，loss 的 Fourier 分量会被乘上：

```text
exp(-sigma^2 ||omega||^2 / 2)
```

因此 jitter 的频域效果主要来自随机平滑：抑制 background 区域中对高频噪声敏感的梯度，保留低/中频结构；brightness 项则测试跨亮度尺度稳定的方向。

### freq

当前 `freq` 是：

```text
A_freq(x) = (1-alpha)x + alpha * (0.7 B_9 x + 0.3 B_9 U)
```

其中 `B_9` 是 `9x9 avg_pool`。background blend 后：

```text
J_freq ~= M I + Q[(1-alpha)I + 0.7 alpha B_9]
```

对应频响：

```text
H_freq(omega) ~= 1-alpha + 0.7 alpha H_avg9(omega)
```

`B_9` 比 `B_5` 更强低通，因此 `freq` 是三者里最明确的低频/中低频 background 梯度选择器。它会压制 background 高频纹理梯度，同时注入低通随机上下文。

### mask 的作用

区域乘法 `Q*g` 在频域中不是简单保留某个频带，而是卷积：

```text
F[Q*g] = Q_hat * G
```

如果 guide map 使用 patch build，`Q` 主要在 patch 尺度变化，`Q_hat` 的能量集中在低频到 patch-grid 频率附近。这会把增强限制在低 attention 区域，同时避免像素级 mask 的细碎边界制造过多高频伪影。这解释了当前最优配置里 `patch + qk_cls + fpridx` 比一些 pixel 版本更稳。

## 3. 为什么 DIM + low-attention 增强组合有效

组合后的期望梯度是：

```text
g_combo = E_{s,t,a}[J_A(x,a)^T J_T(A(x,a),s,t)^T grad_z L(f(z),y)]
```

它同时施加两个稳定性约束：

1. `T_dim` 要求方向在随机缩放和随机 placement 下仍然提高 loss，过滤位置绑定和高频局部纹理。
2. `A_bg` 要求方向在低 attention 区域被随机平滑/扰动后仍然提高 loss，过滤只依赖固定 background 纹理的梯度。

这两个约束的交集更接近跨模型共享子空间：

```text
S_shared ~= S_spatially_stable ∩ S_low_attention_context_stable
```

已有记录支持这个解释：

- `outputs/analysis/why_background_gradient_transfers_report.md` 显示 background raw gradient 整体不一定比 foreground 更同向，但 background 的低频比例更高、模型间幅值更稳定。
- `outputs/analysis/deep_gradient_root_cause_report.md` 显示最强 no-normgrad 相比 fg-gradnorm 的黑盒方向导数增益几乎全部来自 background 项。
- `outputs/analysis/best_attack_final_conclusion.md` 显示最强配置是 `DIM + background augmentation + MI + no_normgrad`，且相邻 ablation 都下降。

因此当前组合有效的更精确数学表述是：

```text
DIM 提高空间变换稳定性；
background augmentation 提高低 attention 上下文稳定性；
MI/no_normgrad 保留这些稳定方向的自然尺度并在后期累积；
最终提升的是 black-box direction derivative 的 low/mid-frequency background 分量。
```

不是“background 全频梯度天然更迁移”，也不是“越低频越好”。真正有效的是 low-attention 区域中，在 DIM 和随机增强下仍然相位稳定、跨模型方向导数为正的低/中频公共子空间。

## 4. 组合有效性的数学实验

### 实验 A：算子频响证明

对 DIM、dropout、jitter、freq 分别测：

```text
G_A(omega) = E_a ||J_A(a)^T J_A(a) phi_omega||_2^2 / ||phi_omega||_2^2
```

区域版本：

```text
G_{A,bg}(omega) = E_a ||Q J_A(a)^T J_A(a) Q phi_omega||_2^2 / ||Q phi_omega||_2^2
```

预期排序：

```text
high-frequency attenuation:
freq > dropout > jitter(Jacobian only)
```

但在 forward random smoothing 下，jitter 的高频模型梯度也应下降。

### 实验 B：源模型梯度频带分解

对每个样本和每个攻击设置，采样同一批随机种子估计：

```text
g_variant = E_seed grad_x CE(f(T_seed(x)), y)
```

频带：

```text
low:  r <= 0.12
mid:  0.12 < r <= 0.35
high: r > 0.35
```

记录：

```text
energy_B      = ||P_B g||_2^2 / ||g||_2^2
fg_energy_B   = ||M P_B g||_2^2 / ||M g||_2^2
bg_energy_B   = ||Q P_B g||_2^2 / ||Q g||_2^2
seed_coh_B    = ||E_seed P_B g_seed||_2 / E_seed ||P_B g_seed||_2
```

`seed_coh_B` 是关键指标：如果某频带在随机 DIM/augmentation 下不稳定，它的平均后相干度会低。

### 实验 C：黑盒方向导数分解

对攻击过程中的 update `u_t=sign(m_t)`，计算：

```text
D_{R,B}^{(b)}(t) = < M_R P_B u_t, M_R P_B grad_x L_b(x_t,y) >
```

`R in {fg,bg}`，`B in {low,mid,high}`。比较：

```text
plain MI
DIM only
bg dropout/jitter/freq only
DIM + bg dropout/jitter/freq
DIM + foreground augmentation
DIM + all augmentation
```

如果本文机制正确，最强配置的优势应集中在：

```text
Delta D_{bg,low} + Delta D_{bg,mid} > 0
Delta D_{bg,high} 较小或不稳定
Delta D_{fg,*} 不是主要正增益来源
```

### 实验 D：交互项而非加和

定义频带交互增益：

```text
S_{R,B} =
  D_{R,B}(DIM + bg_aug)
  - D_{R,B}(DIM only)
  - D_{R,B}(bg_aug only)
  + D_{R,B}(plain)
```

如果 DIM 和 background augmentation 是互补而非简单叠加，应观察到：

```text
S_{bg,low/mid} > 0
S_{fg,high} <= 0 或不稳定
```

这能直接证明组合有效性来自 low-attention low/mid frequency 的交互项。

## 5. 潜在推进方向

1. 显式频带权重优化  
   把更新写成：

   ```text
   g_t = sum_B lambda_B P_B g_t
   ```

   用黑盒代理或多源模型估计 `lambda_low, lambda_mid, lambda_high`，优先放大 `D_{bg,low/mid}` 为正的频带，而不是盲目做全频 sign。

2. DIM range 的频响校准  
   当前 `[0.85,1.0]` 是经验值。可以用 `G_dim(omega)` 选择 resize range，使其通带刚好压掉不稳定高频，但不过度损失 mid-frequency patch geometry。

3. 区域-频带耦合增强  
   当前 `freq` 是全 background 低通腐蚀。可以改成：

   ```text
   foreground: low-frequency mild jitter
   background: mid/high randomized smoothing
   ```

   或使用 wavelet 分解实现 `fg-low + bg-high/mid` 的可控组合。

4. 以方向导数为目标的选择器  
   每一步估计多个候选增强的 `D_{R,B}` 代理值，选择能提高 `bg low/mid` 黑盒代理方向导数的增强，而不是固定 `dropout,jitter,freq`。

5. mask 频谱正则  
   让 guide map 不只看 attention 高低，也控制 `Q_hat` 的频谱，避免过碎 mask 带来边界高频。数学目标可以是：

   ```text
   min_M TV(M) + beta ||P_high M||_2^2
   ```

   同时保持 high-attention 排名。

6. no_normgrad 的频域解释验证  
   现有证据显示 no_normgrad 保留自然区域尺度更强。下一步应分频带看 momentum：

   ```text
   ||Q P_B m_t||_1 / ||M P_B m_t||_1
   ```

   验证 no_normgrad 是否主要保留 `bg low/mid` 的 momentum 比例，而不是所有 background 频带。

## 当前结论

数学上，当前 DIM 的主要作用是通过 resize/pad 的伴随 Jacobian 和随机 placement 的期望，削弱高频、位置绑定的梯度，保留低/中频空间稳定方向。低 attention 上的 `dropout` 和 `freq` 具有明确低通 Jacobian；`jitter` 的 Jacobian 不直接低通，但通过 background 随机平滑抑制高频敏感 loss 分量。二者结合有效，是因为它们共同把 MI 累积方向推向 `low-attention + low/mid-frequency + cross-transform stable` 的公共子空间，而已有方向导数分析显示迁移优势正来自 background 项。

后续最有价值的实验证明不是只画扰动频谱，而是直接计算：

```text
D_{bg,low/mid} = <Q P_{low/mid} sign(m_t), Q P_{low/mid} grad_x L_blackbox>
```

并验证 `DIM + bg_aug` 的交互项 `S_{bg,low/mid}` 为正。
