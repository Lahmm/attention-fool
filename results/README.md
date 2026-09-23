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
- `prd_gradient_diagnostics_s1000.csv`: the corresponding ensemble-rank and progressive-schedule diagnostics.

Aggregate and per-target results are documented in `experiments/progressive_cross_arch_mainline_s1000.md`.
