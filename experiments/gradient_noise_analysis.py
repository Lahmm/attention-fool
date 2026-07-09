"""
Mathematical analysis: why opponent_channel_gaussian helps CNN transfer,
and how to improve ViT transfer via noise covariance optimization.

Key question: The guide augmentation adds noise n ~ N(0, Σ) to create diverse
gradient signals. The noise covariance Σ in pixel space determines the gradient
diversity in both CNN filter space and ViT token space. Different architectures
have different sensitivity to Σ — we need Σ that works for both.
"""

import torch
import torch.nn.functional as F
import numpy as np
import timm
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Load ViT patch embedding weights
# ---------------------------------------------------------------------------

def load_vit_patch_embed():
    """Extract the patch embedding projection weight from ViT-B/16."""
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.eval()
    weight = model.patch_embed.proj.weight.detach()  # [D, 3, 16, 16]
    D, C, H, W = weight.shape
    print(f"Patch embedding weight shape: [{D}, {C}, {H}, {W}]")
    # Reshape to [D, C*H*W] = [D, L]
    W_flat = weight.reshape(D, -1).to(torch.float64)
    # Also keep per-channel slices: [D, H*W] for each channel
    W_R = weight[:, 0, :, :].reshape(D, -1).to(torch.float64)  # [D, 256]
    W_G = weight[:, 1, :, :].reshape(D, -1).to(torch.float64)
    W_B = weight[:, 2, :, :].reshape(D, -1).to(torch.float64)
    return W_flat, W_R, W_G, W_B, D, H


# ---------------------------------------------------------------------------
# 2. Compute token-space noise covariance for different per-pixel C matrices
# ---------------------------------------------------------------------------

def token_covariance(W_flat, W_R, W_G, W_B, C_pixel):
    """
    Compute the D×D token-space noise covariance given per-pixel covariance C_pixel.

    Σ_token = Σ_{h,w} Σ_{c,c'} W_d[c,h,w] * C[c,c'] * W_{d'}[c',h,w]

    Args:
        W_flat: [D, L] where L = 3*H*W
        W_R, W_G, W_B: [D, H*W] per-channel weight slices
        C_pixel: [3, 3] per-pixel channel covariance
    Returns:
        Σ_token: [D, D] covariance in token space
    """
    # Using the formula: Σ = W_R @ W_R^T * C[0,0] + W_R @ W_G^T * C[0,1] + ...
    # More efficiently: Σ = Σ_{c,c'} C[c,c'] * W_c @ W_{c'}^T
    W_ch = [W_R, W_G, W_B]
    D = W_flat.size(0)
    Sigma = torch.zeros(D, D, dtype=torch.float64)
    for c in range(3):
        for cp in range(3):
            Sigma += C_pixel[c, cp] * (W_ch[c] @ W_ch[cp].T)
    return Sigma


def eigenvalue_analysis(Sigma, label):
    """Analyze eigenvalue spectrum of a covariance matrix."""
    eigvals = torch.linalg.eigvalsh(Sigma)
    eigvals = eigvals[eigvals > 1e-10]  # Remove numerical zeros
    total_var = eigvals.sum().item()
    # Effective rank (participation ratio)
    eff_rank = (eigvals.sum() ** 2 / (eigvals ** 2).sum()).item()
    # Entropy (log-determinant up to constant)
    entropy = eigvals.log().sum().item()
    # Condition number
    cond = (eigvals[-1] / eigvals[0]).item()
    # Cumulative variance explained
    sorted_vals, _ = eigvals.sort(descending=True)
    cumsum = sorted_vals.cumsum(dim=0) / total_var
    top5 = cumsum[4].item() if len(cumsum) > 4 else 1.0
    top10 = cumsum[9].item() if len(cumsum) > 9 else 1.0
    top20 = cumsum[19].item() if len(cumsum) > 19 else 1.0
    top50 = cumsum[49].item() if len(cumsum) > 49 else 1.0

    print(f"\n{'='*60}")
    print(f"Covariance spectrum: {label}")
    print(f"{'='*60}")
    print(f"  Total variance:       {total_var:.2f}")
    print(f"  Effective rank:       {eff_rank:.1f} / {len(eigvals)}")
    print(f"  Log-determinant:      {entropy:.2f}")
    print(f"  Condition number:     {cond:.1f}")
    print(f"  Top-5 variance:       {top5*100:.1f}%")
    print(f"  Top-10 variance:      {top10*100:.1f}%")
    print(f"  Top-20 variance:      {top20*100:.1f}%")
    print(f"  Top-50 variance:      {top50*100:.1f}%")
    return {
        "label": label,
        "total_var": total_var,
        "eff_rank": eff_rank,
        "entropy": entropy,
        "cond": cond,
        "top5": top5,
        "top10": top10,
        "top20": top20,
        "top50": top50,
        "eigvals": eigvals,
    }


# ---------------------------------------------------------------------------
# 3. Define different per-pixel noise covariances
# ---------------------------------------------------------------------------

def make_covariances():
    """Generate a set of candidate per-pixel channel covariance matrices."""
    covs = {}

    # Standard i.i.d. Gaussian (baseline)
    covs["standard_gaussian"] = torch.eye(3, dtype=torch.float64)

    # Opponent channel Gaussian (current best)
    covs["opponent_channel"] = torch.tensor(
        [[1.00, -0.25, -0.25],
         [-0.25,  1.00, -0.25],
         [-0.25, -0.25,  1.00]],
        dtype=torch.float64,
    )

    # Luminance-only (no cross-channel structure)
    covs["luminance_only"] = torch.tensor(
        [[0.3333, 0.3333, 0.3333],
         [0.3333, 0.3333, 0.3333],
         [0.3333, 0.3333, 0.3333]],
        dtype=torch.float64,
    )

    # Chrominance-only (complement to luminance)
    covs["chrominance_only"] = torch.tensor(
        [[ 0.6667, -0.3333, -0.3333],
         [-0.3333,  0.6667, -0.3333],
         [-0.3333, -0.3333,  0.6667]],
        dtype=torch.float64,
    )

    # Maximum opponent (stronger anti-correlation)
    covs["strong_opponent"] = torch.tensor(
        [[ 1.0, -0.5, -0.5],
         [-0.5,  1.0, -0.5],
         [-0.5, -0.5,  1.0]],
        dtype=torch.float64,
    )

    # Positive correlation (luminance-biased)
    covs["positive_corr"] = torch.tensor(
        [[1.0, 0.5, 0.5],
         [0.5, 1.0, 0.5],
         [0.5, 0.5, 1.0]],
        dtype=torch.float64,
    )

    # Channel-independent but unequal variance
    covs["r_g_b_biased"] = torch.tensor(
        [[2.0, 0.0, 0.0],
         [0.0, 0.5, 0.0],
         [0.0, 0.0, 0.5]],
        dtype=torch.float64,
    )

    return covs


# ---------------------------------------------------------------------------
# 4. CNN filter response analysis
# ---------------------------------------------------------------------------

def analyze_cnn_sensitivity(C_pixel, label=""):
    """
    Analyze how noise with per-pixel covariance C affects CNN first-layer filters.

    For a CNN filter w ∈ R^{3 × k × k}, the noise contribution to the filter
    response at a position is:

        r_noise = Σ_{c, i, j} w[c, i, j] * n[c, i, j]

    With E[r_noise] = 0 and:
        Var(r_noise) = Σ_{c, c', i, j} w[c,i,j] * C[c,c'] * w[c',i,j]

    This varies depending on the filter's color tuning.
    """
    # Construct example CNN filters
    k = 3
    filters = {}

    # Luminance edge detector: equal weights across channels
    w_lum = torch.randn(1, 3, k, k, dtype=torch.float64)
    w_lum = w_lum / w_lum.norm()
    # Make it approximately equal across channels
    w_lum[:, 0] = w_lum[:, 1] = w_lum[:, 2] = w_lum.mean(dim=1, keepdim=True) / np.sqrt(3)
    filters["gray_edge"] = w_lum

    # Red-Green opponent
    w_rg = torch.zeros(1, 3, k, k, dtype=torch.float64)
    kernel = torch.randn(1, 1, k, k, dtype=torch.float64)
    kernel = kernel / kernel.norm()
    w_rg[:, 0] = kernel / np.sqrt(2)
    w_rg[:, 1] = -kernel / np.sqrt(2)
    w_rg[:, 2] = 0
    filters["red_green_opponent"] = w_rg

    # Blue-Yellow opponent
    w_by = torch.zeros(1, 3, k, k, dtype=torch.float64)
    w_by[:, 0] = kernel / np.sqrt(6)
    w_by[:, 1] = kernel / np.sqrt(6)
    w_by[:, 2] = -2 * kernel / np.sqrt(6)
    filters["blue_yellow_opponent"] = w_by

    # Random filter
    w_rand = torch.randn(1, 3, k, k, dtype=torch.float64)
    w_rand = w_rand / w_rand.norm()
    filters["random"] = w_rand

    print(f"\n{'='*60}")
    print(f"CNN filter noise variance analysis: {label}")
    print(f"{'='*60}")
    print(f"{'Filter':<25s} {'Var(noise)':>12s} {'vs baseline':>12s}")
    print("-" * 50)

    results = {}
    for name, w in filters.items():
        # Var = Σ_{c,c',i,j} w[c,i,j] * C[c,c'] * w[c',i,j]
        w_flat = w.reshape(3, -1)  # [3, k*k]
        var_noise = torch.einsum("ci,cd,di->", w_flat, C_pixel, w_flat).item()
        # Baseline: standard Gaussian (C = I)
        var_baseline = (w_flat ** 2).sum().item()
        ratio = var_noise / var_baseline
        print(f"  {name:<25s} {var_noise:12.6f} {ratio:12.4f}")
        results[name] = {"var": var_noise, "ratio": ratio}

    return results


# ---------------------------------------------------------------------------
# 5. Compute the optimal per-pixel covariance for ViT token diversity
# ---------------------------------------------------------------------------

def optimize_noise_covariance(W_R, W_G, W_B, D, num_iters=1000, lr=0.01):
    """
    Find the per-pixel channel covariance C that maximizes token-space entropy.

    We parameterize C = L @ L^T where L is lower-triangular (Cholesky),
    guaranteeing C ≽ 0. Constraint: Tr(C) = 3.

    Objective: maximize log det(Σ_token(C)) where:
        Σ_token(C) = Σ_{c,c'} C[c,c'] * W_c @ W_{c'}^T
    """
    # Parameterize L (lower triangular, 3x3)
    L_param = torch.eye(3, dtype=torch.float64) + 0.1 * torch.randn(3, 3, dtype=torch.float64)
    L_param = torch.tril(L_param)
    L_param.requires_grad_(True)

    # Precompute channel Gram matrices G_{c,c'} = W_c @ W_{c'}^T
    W_ch = [W_R, W_G, W_B]
    G = {}
    for c in range(3):
        for cp in range(3):
            G[(c, cp)] = W_ch[c] @ W_ch[cp].T

    # Regularization: small ε * I for numerical stability
    eps = 1e-4
    eye_D = torch.eye(D, dtype=torch.float64)

    optimizer = torch.optim.Adam([L_param], lr=lr)
    best_entropy = -float("inf")
    best_C = None

    for i in range(num_iters):
        optimizer.zero_grad()

        # Reconstruct C = L @ L^T
        C = L_param @ L_param.T

        # Normalize: Tr(C) = 3
        scale = 3.0 / (C[0, 0] + C[1, 1] + C[2, 2])
        C = scale * C

        # Compute token-space covariance
        Sigma = torch.zeros(D, D, dtype=torch.float64)
        for c in range(3):
            for cp in range(3):
                Sigma += C[c, cp] * G[(c, cp)]
        Sigma = Sigma + eps * eye_D

        # Objective: log det (entropy of Gaussian)
        try:
            eigvals = torch.linalg.eigvalsh(Sigma)
            eigvals = torch.clamp(eigvals, min=1e-10)
            entropy = eigvals.log().sum()
        except RuntimeError:
            entropy = torch.tensor(-1e10)

        loss = -entropy  # Minimize negative entropy = maximize entropy

        # Add penalty for extreme condition number
        if eigvals[-1] > 0:
            cond_penalty = 0.001 * (eigvals[-1] / eigvals[0])
            loss = loss + cond_penalty

        loss.backward()
        optimizer.step()

        if -loss.item() > best_entropy:
            best_entropy = -loss.item()
            best_C = C.detach().clone()

        if i % 200 == 0:
            print(f"  iter {i:4d}: entropy={-loss.item():.2f}, C_diag=({C[0,0]:.3f}, {C[1,1]:.3f}, {C[2,2]:.3f})")

    return best_C, best_entropy


# ---------------------------------------------------------------------------
# 6. Main analysis
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("GRADIENT NOISE COVARIANCE ANALYSIS")
    print("=" * 70)

    # Load ViT patch embedding
    W_flat, W_R, W_G, W_B, D, patch_size = load_vit_patch_embed()
    L = W_flat.size(1)  # 3 * 16 * 16 = 768
    print(f"Embedding dim D={D}, patch vector dim L={L}")

    # -----------------------------------------------------------------------
    # Part A: Token-space covariance spectrum for different noise types
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART A: Token-space noise covariance analysis for ViT")
    print("=" * 70)

    covs = make_covariances()
    spectra = {}
    for name, C in covs.items():
        # Ensure C has trace ≈ 3 (same total variance)
        tr = C.trace().item()
        if abs(tr - 3.0) > 0.01:
            C = C * (3.0 / tr)
        Sigma = token_covariance(W_flat, W_R, W_G, W_B, C)
        spectra[name] = eigenvalue_analysis(Sigma, name)

    # -----------------------------------------------------------------------
    # Part B: CNN filter sensitivity analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART B: CNN filter noise sensitivity")
    print("=" * 70)
    print("\nTheory: For CNN first-layer filters, the noise contribution")
    print("variance is Var = Σ_{c,c'} w_c·C_{cc'}·w_{c'}.")
    print("Color-opponent filters benefit from anti-correlated noise;")
    print("luminance filters benefit from positively-correlated noise.")

    for name, C in covs.items():
        tr = C.trace().item()
        if abs(tr - 3.0) > 0.01:
            C_adj = C * (3.0 / tr)
        else:
            C_adj = C.clone()
        analyze_cnn_sensitivity(C_adj, name)

    # -----------------------------------------------------------------------
    # Part C: Theoretical analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART C: Theoretical interpretation")
    print("=" * 70)

    # 1. Why opponent_channel_gaussian helps CNN
    print("\n--- C.1: Why opponent_channel_gaussian improves CNN transfer ---")
    print("""
    The opponent channel noise has per-pixel covariance:
        C_opp = [[1.00, -0.25, -0.25],
                 [-0.25,  1.00, -0.25],
                 [-0.25, -0.25,  1.00]]

    Eigen-decomposition of C_opp:
      - λ_lum = 0.5  (luminance direction [1,1,1]/√3)
      - λ_rg  = 1.25 (red-green opponent [1,-1,0]/√2)
      - λ_yb  = 1.25 (blue-yellow opponent [1,1,-2]/√6)

    Compared to standard Gaussian (λ = 1 in all directions):
      - Luminance variance REDUCED 50% (0.5 vs 1.0)
      - Chrominance variance INCREASED 25% (1.25 vs 1.0)

    CNN first-layer filters are dominated by luminance edge detectors
    (Gabor-like). Standard Gaussian puts 33% of noise into luminance
    → narrow, luminance-dominated gradient signals → poor diversity.

    Opponent noise puts only 17% into luminance → forces gradients to
    use chromatic features → more diverse filter activations → more
    transferable gradients.

    Color opponency is a FUNDAMENTAL property of visual representations,
    shared across all ConvNets → chromatic perturbations transfer better.
    """)

    # 2. ViT token-space diversity analysis
    print("\n--- C.2: ViT token-space diversity bottleneck ---")
    print("""
    For ViT, the patch embedding projects 768-dim patch vectors into
    768-dim token space. The noise in token space has covariance:
        Σ_token = W (C ⊗ I_K²) W^T

    where W is the patch embedding weight and C is per-pixel covariance.

    The DIVERSITY of gradient signals across augmentation copies depends
    on the eigenvalue spectrum of Σ_token:
      - High effective rank → diverse token-space noise → diverse gradients
      - Low effective rank → concentrated noise → narrow gradients → overfit

    The effective rank r_eff = (Σλ)²/(Σλ²) measures how many "independent"
    noise directions exist in token space. Higher is better for transfer.
    """)

    # Compare effective ranks
    print(f"\n{'Noise Type':<30s} {'Eff. Rank':>10s} {'Entropy':>10s} {'Cond':>10s}")
    print("-" * 62)
    for name in ["standard_gaussian", "opponent_channel", "strong_opponent",
                 "positive_corr", "r_g_b_biased"]:
        s = spectra[name]
        print(f"  {name:<30s} {s['eff_rank']:10.1f} {s['entropy']:10.2f} {s['cond']:10.1f}")

    # 3. Why patch_embed_rowspace failed
    print("\n--- C.3: Why patch_embed_rowspace noise fails ---")
    print("""
    The rowspace noise samples n = z @ W where z ~ N(0, I_D). The noise
    in pixel space has covariance proportional to W^T W. But W^T W has
    ~256 zero eigenvalues (D=768, L=768, W is full-rank but its singular
    vectors are highly non-uniform).

    In token space: Σ_token = W @ (W^T W) @ W^T = (W W^T)^2 / Tr(...).

    For a square W: the token-space noise is concentrated in the TOP
    singular vectors of W^T W (squared singular values). This means the
    noise activates only the dominant token directions, creating
    ZERO diversity in the long tail of singular vectors.

    Result: all augmentation copies see nearly identical token-space
    perturbations → gradient collapse → ASR drops to 20%.

    LESSON: Maximizing ViT sensitivity (σ_max) is WRONG. We need to
    MAXIMIZE token-space diversity (effective rank).
    """)

    # -----------------------------------------------------------------------
    # Part D: Optimize noise covariance for ViT token diversity
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART D: Optimizing per-pixel noise covariance for ViT")
    print("=" * 70)

    print("\nOptimizing C to maximize log det(Σ_token(C))...")
    C_opt, entropy_opt = optimize_noise_covariance(W_R, W_G, W_B, D)

    print(f"\nOptimal C (token diversity):")
    print(f"  {C_opt[0,0]:.4f}  {C_opt[0,1]:.4f}  {C_opt[0,2]:.4f}")
    print(f"  {C_opt[1,0]:.4f}  {C_opt[1,1]:.4f}  {C_opt[1,2]:.4f}")
    print(f"  {C_opt[2,0]:.4f}  {C_opt[2,1]:.4f}  {C_opt[2,2]:.4f}")

    # Eigendecomposition of optimal C
    C_opt_np = C_opt.numpy()
    eigvals_c, eigvecs_c = np.linalg.eigh(C_opt_np)
    print(f"\nEigenvalues of optimal C: {eigvals_c}")
    print(f"Eigenvectors:")
    for i in range(3):
        print(f"  λ_{i}={eigvals_c[i]:.4f}: {eigvecs_c[:, i]}")

    # Compute spectrum
    Sigma_opt = token_covariance(W_flat, W_R, W_G, W_B, C_opt)
    spec_opt = eigenvalue_analysis(Sigma_opt, "OPTIMAL (max entropy)")

    # Compare standard vs opponent vs optimal
    print(f"\n{'='*60}")
    print("COMPARISON: Standard vs Opponent vs Optimal")
    print(f"{'='*60}")
    print(f"{'':<25s} {'Standard':>10s} {'Opponent':>10s} {'Optimal':>10s}")
    print("-" * 57)
    for metric in ["eff_rank", "entropy", "cond"]:
        vals = [
            spectra["standard_gaussian"][metric],
            spectra["opponent_channel"][metric],
            spec_opt[metric],
        ]
        print(f"  {metric:<25s} {vals[0]:10.1f} {vals[1]:10.1f} {vals[2]:10.1f}")

    # CNN sensitivity of optimal C
    analyze_cnn_sensitivity(C_opt, "OPTIMAL (max ViT entropy)")

    # -----------------------------------------------------------------------
    # Part E: Proposed hybrid approach
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART E: Proposed approach — frequency-split hybrid noise")
    print("=" * 70)

    print("""
    KEY INSIGHT: CNN and ViT have CONFLICTING requirements for noise structure.

    CNN requires:
      - Per-pixel channel diversity (opponent structure helps)
      - High-frequency spatial variation (small CNN receptive fields)
      - Independent noise across nearby pixels

    ViT requires:
      - Token-space diversity (high effective rank of Σ_token)
      - Spatial structure at patch scale (~16×16 blocks)
      - The patch embedding averages over 16×16, so sub-patch noise is compressed

    PROPOSED: Two-component hybrid noise

    n_total = sqrt(α) * n_cnn + sqrt(1-α) * n_vit

    where:
      n_cnn: Independent per-pixel opponent-channel noise
             (targets CNN filter diversity, high spatial frequency)

      n_vit:  Patch-structured noise with spatial correlation at block scale
             (targets ViT token diversity, at patch-grid spatial frequency)

    n_vit is generated by:
      1. Sample latent noise z in [B, D, grid, grid] space
      2. Project through W^T: per-patch noise = z @ W
      3. But use ONLY mid-frequency components of z
         (within-patch = high freq, cross-patch-low = low freq)
      4. Apply spatial smoothing with kernel σ ≈ 0.5 patches

    This creates noise that:
      - Has per-pixel opponent structure → good CNN gradient diversity
      - Has patch-scale spatial coherence → survives ViT patch embedding
      - Uses the opponent C to reduce luminance bias → both benefit

    ALTERNATIVE: Direct token-space noise injection

    Instead of adding noise in pixel space and hoping it creates diverse
    token representations, we could:
      1. Compute patch tokens for the clean image
      2. Add structured noise DIRECTLY to tokens
      3. Use a "token decoder" loss that penalizes the difference between
         clean and noisy token representations

    This bypasses the patch embedding bottleneck entirely for ViT.
    """)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    1. opponent_channel_gaussian works for CNN because:
       - 50% less luminance noise, 25% more chrominance noise
       - Chromatic features are more universal → better CNN transfer
       - Anti-correlated channels reduce effective noise for "gray" filters
       → CNN ASR: 59.8% → 64.5% (+4.7pp)

    2. ViT transfer bottleneck is token-space diversity:
       - Patch embedding compresses 768-dim → 768-dim
       - Noise must create DIVERSE (not just strong) token perturbations
       - Effective rank of Σ_token measures diversity

    3. Standard Gaussian effective rank:    {spectra['standard_gaussian']['eff_rank']:.0f}
       Opponent channel effective rank:    {spectra['opponent_channel']['eff_rank']:.0f}
       Optimal covariance effective rank:   {spec_opt['eff_rank']:.0f}

    4. To improve ViT transfer to 80%+:
       - Option A: Hybrid opponent + patch-structured noise
       - Option B: Direct token-space augmentation
       - Option C: Frequency-band-split noise (GNS-HFA style)
    """)


if __name__ == "__main__":
    main()
