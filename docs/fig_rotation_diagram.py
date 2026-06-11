"""Generate a Givens rotation diagram for the LM-DSS + rotation mechanism."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), subplot_kw={"aspect": "equal"})
fig.subplots_adjust(wspace=0.45)

# ── shared setup ──────────────────────────────────────────────────────────
# original gradient vector components (lowmid_norm, high_norm)
lm0, h0 = 0.70, 0.55
orig_angle = np.arctan2(h0, lm0)

# rotation parameters
strengths = [0.0, 0.50, 0.85]
labels = [
    r"$\theta = 0$  (no rotation)",
    r"$\theta = %.2f$  (moderate agreement)" % (strengths[1] * orig_angle),
    r"$\theta = %.2f$  (high agreement)"   % (strengths[2] * orig_angle),
]
titles = [
    "Original gradient",
    "DSS agreement = 0.69\n(cautious rotation)",
    "DSS agreement = 0.96\n(aggressive rotation)",
]
annot_y = [0.05, -0.07, -0.07]

# unit circle
theta_circle = np.linspace(0, 2 * np.pi, 400)
cx, cy = np.cos(theta_circle), np.sin(theta_circle)

for col, (s, lbl, ttl) in enumerate(zip(strengths, labels, titles)):
    ax = axes[col]

    # unit circle (reference)
    ax.plot(cx, cy, "k", lw=0.6, alpha=0.25)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.3)

    # rotation angle
    theta = s * orig_angle
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # rotated vector
    lm1 =  lm0 * cos_t + h0 * sin_t
    h1  = -lm0 * sin_t + h0 * cos_t

    # original vector
    ax.arrow(0, 0, lm0, h0,
             head_width=0.04, head_length=0.06, fc="#E74C3C", ec="#E74C3C",
             lw=2.5, alpha=0.7, length_includes_head=True, label="original g")

    # rotated vector
    ax.arrow(0, 0, lm1, h1,
             head_width=0.04, head_length=0.06, fc="#2980B9", ec="#2980B9",
             lw=2.5, alpha=0.9, length_includes_head=True, label="rotated g'")

    # arc showing rotation
    arc_radii = 0.30
    arc_angles = np.linspace(0, -theta, 60) if theta > 0 else np.linspace(0, -theta, 60)
    arc_x = arc_radii * np.cos(np.linspace(0, -theta, 60))
    arc_y = arc_radii * np.sin(np.linspace(0, -theta, 60))
    ax.plot(arc_x, arc_y, "g-", lw=1.5, alpha=0.8)
    mid_idx = len(arc_x) // 2
    ax.annotate(
        r"$\theta$", (arc_x[mid_idx], arc_y[mid_idx]),
        fontsize=11, color="green",
        xytext=(arc_x[mid_idx] + 0.12, arc_y[mid_idx] + 0.06),
        arrowprops=dict(arrowstyle="->", color="green", lw=0.8),
    )

    # axis labels
    ax.set_xlim(-0.2, 1.05)
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel("Low/mid freq magnitude", fontsize=11)
    ax.set_ylabel("High freq magnitude", fontsize=11, labelpad=2)
    ax.set_title(ttl, fontsize=10, pad=8)

    # color-filled region showing 'lowmid dominance'
    ax.fill_between([0, 1], [0, 0], [0.35, 0.35], alpha=0.06, color="#2980B9")
    ax.text(0.35, 0.18, "low/mid dominant", fontsize=8, color="#2980B9", alpha=0.5, ha="center")

    # energy distribution bar (inset in bottom-right corner)
    ax_inset = ax.inset_axes([0.60, 0.05, 0.35, 0.30])
    energies_before = np.array([lm0**2, h0**2])
    energies_after  = np.array([lm1**2, h1**2])
    total_before = energies_before.sum()
    total_after  = energies_after.sum()
    x_pos = np.array([0, 1])
    width = 0.35
    ax_inset.bar(x_pos - width/2, energies_before / total_before, width,
                 color=["#3498DB", "#E74C3C"], alpha=0.6, label="before")
    ax_inset.bar(x_pos + width/2, energies_after / total_after, width,
                 color=["#3498DB", "#E74C3C"], alpha=0.9, label="after",
                 hatch="//")
    ax_inset.set_xticks(x_pos)
    ax_inset.set_xticklabels(["low/mid", "high"], fontsize=7)
    ax_inset.set_ylabel("energy ratio", fontsize=7)
    ax_inset.tick_params(axis="both", labelsize=6)
    ax_inset.set_ylim(0, 1.05)
    if col == 0:
        ax_inset.legend(fontsize=5, loc="upper left", framealpha=0.7)

    # projection lines to axes (for original vector)
    ax.plot([lm0, lm0], [0, h0], ":", color="#E74C3C", lw=0.8, alpha=0.3)
    ax.plot([0, lm0], [h0, h0], ":", color="#E74C3C", lw=0.8, alpha=0.3)

    if col == 0:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)

plt.suptitle("Givens rotation of gradient modulated by LM-DSS agreement",
             fontsize=14, y=1.02)
plt.savefig("docs/fig_rotation_effectiveness.png", dpi=200, bbox_inches="tight")
plt.close()
print("Diagram saved to docs/fig_rotation_effectiveness.png")
