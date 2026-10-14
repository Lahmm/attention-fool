# PRD transfer evaluation with the FPR Table 2 target suite

Reference: [CVPR 2025 paper](https://arxiv.org/html/2503.15404v1),
[official model loader](https://github.com/RYC-98/FPR/blob/main/transferattack/utils.py).

The thirteen targets, in table order, are ViT-B/16, CaiT-S24, PiT-B,
Visformer-S, Swin-T, DeiT-T, CoaT-T, ResNet-18, VGG-16, DenseNet-121,
EfficientNet-B0, MobileNet-V3-Small, and ResNeXt-50-32x4d.
The previous project target list contains the first four; the other nine
are additional targets. This machine initially had no cached model weights
or adversarial images from the recorded main experiment.

Use `python experiments/download_fpr_models.py` to download all target
weights into `data/huggingface` and `data/torch`. The seven ViTs use the
installed timm pretrained configuration; CNNs use torchvision `DEFAULT`,
as in the FPR release. FPR supplies filenames and file sizes for its local
ViT binaries, but no downloadable binaries or checksums. Architecture
alignment is established; byte-identical ViT checkpoint provenance is not.

`transfer_eval.py --protocol fpr` evaluates all thirteen models with direct
224x224 image resizing, conversion to [0,1], and model normalization.
This follows the official FPR image loader without the center crop used
by the original project evaluation. MobileNet-V3 means the small model
in the official loader. ASR is one minus adversarial accuracy over all
1000 samples, without clean-correct filtering.

`python experiments/run_prd_fpr_targets.py` generates PRD samples using
the current project sources: ViT-B/16, CaiT-S24, PiT-B, and Visformer-S.
It reads checkpoint/drop/noise settings from the recorded main experiment,
uses seed 20260907, ten iterations, ten phase pairs per iteration, and
epsilon 16/255. Batch size is eight for the local 10GB RTX 3080.
Random replay is keyed by image and attack context, independent of batch
position. Evaluation uses FP32 with TF32 disabled, batch size 32.

Outputs are stored separately under `outputs/fpr_targets/`: attack/evaluation
logs, four per-sample prediction JSON files, and `transfer_asr.csv`.
Each row reports all thirteen ASRs, the overall mean, and the black-box
mean excluding its source target. These are new samples and measurements;
the original 2026-10-12 fourteen-target results remain the recorded main
experiment. The FPR paper itself uses different sources (PiT-T and DeiT-B
instead of PiT-B and Visformer-S), so these rows do not constitute a
source-matched replication of FPR.

Local initial checks: seven transfer-evaluation tests pass. Full suite
has an existing gradient-byte-hash regression failure under PyTorch
2.10.0+cu128 (37 tests initially, one failure, two skips); attack code was
not changed. A Git push was attempted after modifications and failed
because the machine has no GitHub HTTPS authentication.

All thirteen checkpoints were downloaded successfully and passed a real
single-image forward check with finite `(1, 1000)` logits. Runtime:
Python environment `/root/miniconda3/envs/att-atk`, PyTorch 2.10.0+cu128,
torchvision 0.25.0+cu128, timm 1.0.24, NVIDIA RTX 3080 10GB.
