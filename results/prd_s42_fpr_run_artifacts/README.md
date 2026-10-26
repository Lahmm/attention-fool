# Versioned FPR13 run records

This directory contains the actual attack parameters and transfer CSV/JSON records for the current seed-42, 1000-image, four-source 2×2 PRD ablation. Each `cell_source/` directory retains one run. PiT-T and distilled DeiT-B `drop/full` are the 2026-10-23 reruns; the other twelve cells retain the 2026-10-19 measurements.

The original `transfer.csv` and `transfer.json` files are copied byte-for-byte. They record each target's correct count, total sample count and ASR, plus the source/cell summaries. `attack_params.json` is extracted from the corresponding transfer JSON. Checkpoint and drop-ratio fields in `base/noise` are inactive because progressive drop was disabled. Per-run batch sizes, source paths, SHA256 checksums and seed convention are in [the provenance record](../prd_s42_1000_fpr_provenance.json).

See [the main FPR13 report](../prd_s42_1000_fpr_ablation.md) for the ASR and direct-difference tables. Original images, per-image predictions and logs stay in ignored `outputs/`; this versioned archive supplies lightweight numerical records to other checkouts.
