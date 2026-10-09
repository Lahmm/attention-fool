# Result archive

The promoted research mainline is Progressive Route Disruption (PRD), implemented by `progressive_attack.py` and dispatched by `main.py`.

PRD uses uniformly random local-token masks at architecture-specific checkpoints. The other paper mechanism is kept-only RGB opponent-channel noise projected through the source model's initial RGB convolution.

The current PRD implementation's four completed 1000-image runs are recorded in:

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
