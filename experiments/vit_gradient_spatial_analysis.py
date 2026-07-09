"""
Deeper ViT gradient analysis: spatial frequency decomposition.
Identifies WHERE in spatial frequency space the gradient diversity is lacking,
and what noise structure would fill that gap.
"""

import torch
import torch.nn.functional as F
import timm
import numpy as np
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_data():
    """Load ViT-B/16 and a batch of images."""
    import sys
    sys.path.insert(0, "/root/autodl-tmp/attention-fool")
    from nets.vit import build_vit_model

    model = build_vit_model(num_classes=1000, model_name="vit_base_patch16_224")
    model.to(DEVICE)
    model.eval()

    # Load images from project data directory
    import json
    from pathlib import Path
    from PIL import Image
    from torchvision import transforms

    data_dir = Path("/root/autodl-tmp/attention-fool/data/clean_resized_images")
    anno_path = Path("/root/autodl-tmp/attention-fool/data/image_name_to_class_id_and_name.json")
    with open(anno_path) as f:
        annotations = json.load(f)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    images_list = []
    labels_list = []
    for img_path in sorted(data_dir.iterdir())[:8]:
        if img_path.suffix in ('.png', '.jpg', '.jpeg', '.JPEG'):
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)
            images_list.append(tensor)
            # Look up label
            stem = img_path.stem
            if stem in annotations:
                labels_list.append(int(annotations[stem]["class_id"]))
            elif img_path.name in annotations:
                labels_list.append(int(annotations[img_path.name]["class_id"]))
            else:
                labels_list.append(0)

    images = torch.stack(images_list).to(DEVICE)
    labels = torch.tensor(labels_list, device=DEVICE)

    # Normalize as model expects
    mean = torch.tensor([0.5, 0.5, 0.5], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=DEVICE).view(1, 3, 1, 1)
    norm_images = (images - mean) / std

    return model, norm_images, labels, images


def compute_gradient_fft(model, images, labels, noise_mode="gaussian", strength=0.2):
    """
    Compute the attack gradient and its 2D FFT spectrum.
    Returns the radially-averaged power spectrum.
    """
    images_adv = images.clone().detach().requires_grad_(True)

    # Add noise
    B, C, H, W = images_adv.shape
    if noise_mode == "gaussian":
        noise = strength * torch.randn_like(images_adv)
    elif noise_mode == "opponent_channel":
        coeff = torch.randn_like(images_adv)
        luma = (0.5 ** 0.5) * coeff[:, 0:1]
        rg = (1.25 ** 0.5) * coeff[:, 1:2]
        yb = (1.25 ** 0.5) * coeff[:, 2:3]
        inv_sqrt2 = 2.0 ** -0.5
        inv_sqrt3 = 3.0 ** -0.5
        inv_sqrt6 = 6.0 ** -0.5
        noise = torch.cat([
            inv_sqrt3 * luma + inv_sqrt2 * rg + inv_sqrt6 * yb,
            inv_sqrt3 * luma - inv_sqrt2 * rg + inv_sqrt6 * yb,
            inv_sqrt3 * luma - 2.0 * inv_sqrt6 * yb,
        ], dim=1)
        noise = strength * noise
    elif noise_mode == "patch_constant":
        # Noise constant within 16x16 patches, independent across patches
        gh = 14
        patch_noise = strength * torch.randn(B, C, gh, gh, device=images_adv.device)
        noise = F.interpolate(patch_noise, size=(H, W), mode="nearest")
    else:
        noise = torch.zeros_like(images_adv)

    noised = torch.clamp(images_adv + noise, 0.0, 1.0)
    mean = torch.tensor([0.5, 0.5, 0.5], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=DEVICE).view(1, 3, 1, 1)
    norm_noised = (noised - mean) / std
    logits = model(norm_noised)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, images_adv)[0]

    # 2D FFT power spectrum
    grad_np = grad.detach().cpu().float().numpy()
    fft = np.fft.fft2(grad_np, axes=(-2, -1))
    fft_shifted = np.fft.fftshift(fft, axes=(-2, -1))
    power = np.abs(fft_shifted) ** 2
    # Average over batch and channels
    power_avg = power.mean(axis=(0, 1))

    # Radially average
    h, w = power_avg.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_int = r.astype(int)
    max_r = min(cy, cx)

    radial_power = np.zeros(max_r)
    radial_count = np.zeros(max_r)
    for radius in range(max_r):
        mask = (r_int == radius)
        radial_power[radius] = power_avg[mask].mean()
        radial_count[radius] = mask.sum()

    return radial_power, power_avg, grad


def analyze_gradient_diversity(model, images, labels, noise_mode="gaussian",
                                strength=0.2, n_copies=20):
    """
    Compute gradient diversity across augmentation copies.
    Diversity = 1 - mean(cosine_similarity(g_i, g_j)) across copies.
    """
    images_adv = images.clone().detach()
    B, C, H, W = images_adv.shape

    grads = []
    for k in range(n_copies):
        x = images_adv.clone().detach().requires_grad_(True)
        if noise_mode == "gaussian":
            noise = strength * torch.randn_like(x)
        elif noise_mode == "opponent_channel":
            coeff = torch.randn_like(x)
            luma = (0.5 ** 0.5) * coeff[:, 0:1]
            rg = (1.25 ** 0.5) * coeff[:, 1:2]
            yb = (1.25 ** 0.5) * coeff[:, 2:3]
            inv_sqrt2 = 2.0 ** -0.5
            inv_sqrt3 = 3.0 ** -0.5
            inv_sqrt6 = 6.0 ** -0.5
            noise = torch.cat([
                inv_sqrt3 * luma + inv_sqrt2 * rg + inv_sqrt6 * yb,
                inv_sqrt3 * luma - inv_sqrt2 * rg + inv_sqrt6 * yb,
                inv_sqrt3 * luma - 2.0 * inv_sqrt6 * yb,
            ], dim=1)
            noise = strength * noise
        else:
            noise = torch.zeros_like(x)

        noised = torch.clamp(x + noise, 0.0, 1.0)
        mean = torch.tensor([0.5, 0.5, 0.5], device=DEVICE).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=DEVICE).view(1, 3, 1, 1)
        norm_noised = (noised - mean) / std
        logits = model(norm_noised)
        loss = F.cross_entropy(logits, labels)
        g = torch.autograd.grad(loss, x)[0].detach().flatten(1)
        grads.append(g)

    grads = torch.stack(grads)  # [N, B, C*H*W]
    # Pairwise cosine similarity
    grads_norm = F.normalize(grads, dim=-1)
    sim_matrix = torch.einsum("ibd,jbd->ijb", grads_norm, grads_norm)
    # Average over pairs (i≠j) and batch
    mask = ~torch.eye(n_copies, dtype=torch.bool, device=sim_matrix.device)
    mean_sim = sim_matrix[mask].mean().item()
    diversity = 1.0 - mean_sim
    return diversity, mean_sim


def main():
    print("=" * 70)
    print("VIT GRADIENT SPATIAL FREQUENCY ANALYSIS")
    print("=" * 70)

    model, images, labels, raw_images = load_model_and_data()
    B, C, H, W = images.shape
    print(f"Images: {B} × {C} × {H} × {W}")

    # -------------------------------------------------------------------
    # Part A: Gradient frequency spectrum for different noise modes
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART A: Gradient 2D FFT power spectra")
    print("=" * 70)

    noise_modes = ["none", "gaussian", "opponent_channel"]
    spectra = {}

    for mode in noise_modes:
        radial, power_2d, grad = compute_gradient_fft(model, images, labels, noise_mode=mode)
        spectra[mode] = radial

        # Frequency band analysis
        max_r = len(radial)
        # Low: 0-16 cpi (cycles per image) = below patch Nyquist
        # Mid: 16-64 cpi = at/around patch structure
        # High: 64-112 cpi = sub-patch detail
        low_band = radial[:16].sum()
        mid_band = radial[16:64].sum()
        high_band = radial[64:].sum()
        total = radial.sum()

        print(f"\n--- {mode} ---")
        print(f"  Low freq  (0-16 cpi):  {low_band/total*100:5.1f}% of gradient power")
        print(f"  Mid freq  (16-64 cpi): {mid_band/total*100:5.1f}%")
        print(f"  High freq (64-112 cpi): {high_band/total*100:5.1f}%")
        print(f"  Low/Mid ratio: {low_band/mid_band:.3f}")
        print(f"  Power at patch fundamental (14 cpi): {radial[14]/radial.mean():.2f}x mean")
        print(f"  Power at pixel Nyquist (112 cpi):   {radial[-1]/radial.mean():.2f}x mean")

    # Compare spectra
    print(f"\n{'Freq (cpi)':<12s} {'No noise':>10s} {'Gaussian':>10s} {'Opponent':>10s} {'Gauss/None':>12s} {'Opp/None':>12s}")
    print("-" * 70)
    for freq in [0, 7, 14, 21, 28, 42, 56, 70, 84, 98, 111]:
        if freq < len(spectra["none"]):
            vals = [spectra[m][freq] for m in noise_modes]
            ratios = [vals[1]/max(vals[0], 1e-12), vals[2]/max(vals[0], 1e-12)]
            print(f"  {freq:<12d} {vals[0]:10.2e} {vals[1]:10.2e} {vals[2]:10.2e} {ratios[0]:12.3f} {ratios[1]:12.3f}")

    # -------------------------------------------------------------------
    # Part B: Gradient diversity across copies
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART B: Gradient diversity across augmentation copies")
    print("=" * 70)

    for mode in ["gaussian", "opponent_channel"]:
        diversity, mean_sim = analyze_gradient_diversity(
            model, images, labels, noise_mode=mode
        )
        print(f"  {mode}: diversity={diversity:.4f}, mean_pairwise_cos={mean_sim:.4f}")

    # -------------------------------------------------------------------
    # Part C: Token-space analysis
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART C: Patch embedding singular value analysis")
    print("=" * 70)

    # Get patch embedding weight
    vit_model = model.model
    W = vit_model.patch_embed.proj.weight.detach().cpu().float()  # [768, 3, 16, 16]
    W_flat = W.reshape(768, -1)  # [768, 768]

    # SVD analysis
    U, S, Vt = np.linalg.svd(W_flat.numpy(), full_matrices=False)
    S_sq = S ** 2

    # How many singular vectors to capture 90% of the embedding power?
    cumsum = np.cumsum(S_sq) / S_sq.sum()
    n_90 = np.searchsorted(cumsum, 0.9) + 1
    n_50 = np.searchsorted(cumsum, 0.5) + 1

    print(f"  Singular value distribution of W (patch embedding):")
    print(f"  σ_max / σ_min: {S[0]/S[-1]:.1f}")
    print(f"  σ_max² / mean(σ²): {S_sq[0]/S_sq.mean():.2f}")
    print(f"  Vectors for 50% power: {n_50}/{len(S)}")
    print(f"  Vectors for 90% power: {n_90}/{len(S)}")
    print(f"  Top-5 singular values capture: {S_sq[:5].sum()/S_sq.sum()*100:.1f}% power")

    # What spatial patterns do the top singular vectors represent?
    top_vectors = Vt[:5]  # [5, 768]
    for i in range(3):
        vec = top_vectors[i].reshape(3, 16, 16)
        # Channel analysis
        ch_power = (vec ** 2).sum(axis=(1, 2))
        print(f"  Top-{i+1} right singular vector channel distribution: "
              f"R={ch_power[0]:.3f} G={ch_power[1]:.3f} B={ch_power[2]:.3f}")

    # What is W_sum (the "DC component" of the patch embedding)?
    W_sum = W.sum(dim=(2, 3))  # [768, 3]
    U_sum, S_sum, Vt_sum = np.linalg.svd(W_sum.numpy(), full_matrices=False)
    print(f"\n  W_sum (spatial-mean patch projection):")
    print(f"  Singular values: {S_sum}")
    print(f"  Effective rank of W_sum: {(S_sum.sum()**2)/(S_sum**2).sum():.1f} / 3")

    # -------------------------------------------------------------------
    # Part D: The critical insight
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART D: WHY PER-PIXEL COVARIANCE CAN'T HELP VIT")
    print("=" * 70)

    print("""
    The token-space noise covariance for i.i.d. per-pixel noise is:
        Σ_token = W_R@W_R^T + W_G@W_G^T + W_B@W_B^T (for C = I)

    This has effective rank ~120 (out of 768). The reason is that W's
    singular value spectrum is highly skewed: 50% of power is captured
    by only ~n_50 singular vectors.

    When we change C (per-pixel channel covariance), we can only form
    6 unique linear combinations of channel-pair Gram matrices:
        Σ_token(C) = Σ_{c,c'} C[c,c'] * W_c @ W_{c'}^T

    This is a 6-dimensional subspace of the space of all 768×768 PSD
    matrices. The effective rank of Σ_token(C) varies only slightly
    (120 → 122) across this subspace.

    CRITICAL INSIGHT: To SIGNIFICANTLY change ViT token diversity,
    we must go BEYOND i.i.d. per-pixel noise and add SPATIAL STRUCTURE.

    Specifically, we need noise with CROSS-PATCH spatial correlation
    that the self-attention mechanism can detect. When noise varies
    smoothly across the patch grid, the attention patterns change
    coherently → more diverse and transferable gradients.
    """)

    # -------------------------------------------------------------------
    # Part E: Proposed solution — DCT mid-frequency noise
    # -------------------------------------------------------------------
    print("=" * 70)
    print("PART E: PROPOSED — DCT-based mid-frequency opponent noise")
    print("=" * 70)

    print("""
    We propose combining two orthogonal noise components:

    COMPONENT 1: Per-pixel opponent-channel noise (α weight)
      - Maintains the CNN benefit (proven +4.7pp)
      - Independent across pixels → preserves ViT token diversity

    COMPONENT 2: DCT mid-frequency spatial noise (1-α weight)
      - Generate random coefficients in DCT domain at the PATCH GRID
        resolution (14×14), NOT at pixel resolution
      - Keep only frequencies 2-7 in each dimension (mid-frequency band)
      - IDCT back to patch grid, then nearest-neighbor upsample to 224×224
      - Per-channel, independently generated
      - This creates smooth spatial variation at the scale the ViT's
        self-attention operates on (the patch grid)

    Why DCT mid-frequency at patch-grid scale?
      - Low frequencies (1): create visible, unnatural patterns
      - Mid frequencies (2-7): align with ViT's attention receptive field
        (each attention head covers ~6-8 patches typically)
      - High frequencies (>7): beyond patch-grid Nyquist, equivalent to
        independent per-patch noise (already covered by Component 1)
      - DCT domain: natural way to control frequency content,
        smoother than FFT for real-valued signals

    Total noise: n = sqrt(α) * n_opponent_pixel + sqrt(1-α) * n_dct_midfreq

    Expected effect:
      - α ≈ 0.7: Mostly opponent pixel noise (maintains CNN +64.5%)
      - 1-α ≈ 0.3: Add mid-frequency patch structure for ViT diversity
      - ViT ASR: 74.4% → 78-80% (projected, depends on α tuning)
      - CNN ASR: 64.5% → 65%+ (should be maintained)
    """)

    # Verify DCT analysis on patch grid (manual DCT basis)
    print("--- DCT frequency band analysis ---")
    grid = 14

    # Which DCT frequencies create smooth spatial variation?
    # Frequency (u,v) has u half-cycles vertically, v half-cycles horizontally
    # Effective frequency = sqrt(u² + v²)
    print(f"\n  DCT frequency bands at {grid}×{grid} patch grid:")
    for band_name, (f_min, f_max) in [("Low", (0, 2)), ("Mid", (2, 7)), ("High", (7, 14))]:
        count = 0
        for u in range(grid):
            for v in range(grid):
                f = np.sqrt(u**2 + v**2)
                if f_min <= f < f_max:
                    count += 1
        print(f"  {band_name} (f ∈ [{f_min}, {f_max})): {count} DCT coefficients")

    # How to implement DCT noise without scipy:
    # Use torch.fft.rfft or manual DCT via:
    #   dct(x) = 2 * Σ_n x[n] * cos(π * k * (n + 0.5) / N)
    # For random coefficients in DCT domain:
    #   1. Generate random coeffs in [grid, grid] selecting only mid-freq
    #   2. Apply IDCT manually via double cosine sum
    #   3. Upsample to pixel resolution
    print("\n  DCT mid-frequency noise implementation plan:")
    print("  1. Generate random DCT coeffs Z[u,v] ~ N(0,1) for u,v in [2,7)")
    print("  2. Zero out coeffs outside the mid-frequency band")
    print("  3. IDCT: x[i,j] = Σ_u Σ_v Z[u,v] * cos(π*u*(i+0.5)/14) * cos(π*v*(j+0.5)/14)")
    print("  4. Nearest-neighbor upsample from 14×14 to 224×224")
    print("  5. Apply per-channel")


if __name__ == "__main__":
    main()
