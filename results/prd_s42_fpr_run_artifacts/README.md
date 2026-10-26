# Versioned FPR13 run records

This directory contains the actual attack parameters and transfer CSV/JSON records for the current seed-42, 1000-image, four-source 2×2 PRD ablation. Each `cell_source/` directory retains one run. PiT-T and distilled DeiT-B `drop/full` are the 2026-10-23 reruns; the other twelve cells retain the 2026-10-19 measurements.

`paired_outcomes.npz` contains `success` with boolean shape `(4,4,1000,13)` and axes `(source,cell,image,target)`. A true value means the predicted class differs from the ground truth. Keys `sources`, `cells`, `sample_ids`, `labels`, and `targets` give the axis identities. Load with `numpy.load(..., allow_pickle=False)`. All 16 cells share the same image order and labels. These indicators reproduce every ASR, paired difference, bootstrap interval and McNemar count without requiring server-local original predictions.

The original `transfer.csv` and `transfer.json` files are copied byte-for-byte. `attack_params.json` is extracted from the corresponding recorded transfer JSON. Parameter fields naming old checkpoints or ratios in `base/noise` are inactive because progressive drop was disabled. Per-run batch sizes, source paths, SHA256 checksums, seed convention and bootstrap specification are in [the provenance record](../prd_s42_1000_fpr_provenance.json).

See [the main FPR13 report](../prd_s42_1000_fpr_ablation.md) for all result tables. Original images, prediction class IDs and logs stay in ignored `outputs/`; this versioned archive supplies lightweight numerical records to other checkouts.
