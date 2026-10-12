# Result archive

The promoted research mainline is Progressive Route Disruption (PRD), implemented by `progressive_attack.py` and dispatched by `main.py`.

PRD uses uniformly random local-token masks at architecture-specific checkpoints. The current implementation adds RGB opponent-channel noise to all initial local features before checkpoint traversal. The archived results below use the earlier kept-only policy; see `experiments/prd_noise_injection_change.md` for the change contract. Full-local transfer ASR was measured on 2026-10-12 and is archived separately as an additional result for the new noise policy.

The earlier kept-only PRD version's four completed 1000-image runs (`98cb7cf1`) are recorded in:

- `outputs/csv/outputs_attack_prd1000_vit_seed20260907.csv`;
- `outputs/csv/outputs_attack_prd1000_cait_seed20260907.csv`;
- `outputs/csv/outputs_attack_prd1000_pit_seed20260907.csv`;
- `outputs/csv/outputs_attack_prd1000_visformer_seed20260907.csv`.

Their measurements are consolidated into:

- `prd_cross_arch_s1000.csv`: the complete 4-source × 14-target transfer table;
- `prd_gradient_diagnostics_s1000.csv`: internal experimental analysis records and progressive-schedule counts.

Aggregate and per-target results are documented in `experiments/progressive_cross_arch_mainline_s1000.md`.

Transfer ASR is the paper's measure of transferability. Gradient diagnostics,
including effective rank, are retained as internal process records.

The exact attack parameters, gradient diagnostics, and losslessly compressed replay
manifests of all four formal runs are in `prd_run_artifacts/`. The historical
patch-rank observation is retained there as motivation only. The SHA256 list
for the 4000 formal adversarial PNGs is `prd1000_image_sha256.tsv.gz`.
`server_untracked_files.tsv.gz` lists every server-side file or symlink omitted
from Git; see `experiments/server_handoff_inventory.md` for download guidance.

## Additional full-local results (2026-10-12)

The new full-local policy was evaluated with the same four source configurations, seed `20260907`, and ordered 1000-image sample set. All 56 source-target evaluations used 1000 samples with zero skipped images. Four-source Overall ASR is 84.55% (kept-only: 84.20%; +0.36 percentage points); strict black-box ASR is 83.49% (kept-only: 83.10%; +0.39 points). CNN ASR rises from 78.59% to 79.44%, while Transformer ASR is essentially unchanged at 88.39%.

- [Full-local report and configuration](prd_full_local_s1000.md).
- [Full-local 56-pair transfer table](prd_full_local_cross_arch_s1000.csv).
- [Per-target old/new comparison](prd_full_local_vs_kept_only_per_target_s1000.csv).
- [Summary old/new comparison](prd_full_local_vs_kept_only_summary_s1000.csv).
- `prd_full_local_run_artifacts/`: full-local run provenance, original attack parameters, and original transfer CSVs.

The kept-only CSVs, reports and `prd_run_artifacts/` retain their original measurements. Full-local files provide the additional evaluation of the current method.
