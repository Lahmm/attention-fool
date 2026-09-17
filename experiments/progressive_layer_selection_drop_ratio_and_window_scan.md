# Superseded layer-selection scan

The detailed layer-selection scan was a historical tuning artifact and no
longer defines an executable or documented ViT configuration.

The current and only retained ViT-B/16 progressive attack setting is:

- checkpoints: `block3,block10`;
- drop ratios: `0.051020408163,0.051020408163`;
- drop counts: `10,10` on the 14 x 14 token grid;
- selector: `high`;
- score window ratio: `0.5`;
- opponent noise strength: `0.2`.

With 10 attack steps and 10 augmentation groups, this configuration builds
100 fresh schedules and performs 200 checkpoint mask selections per image.
Scores at `block10` are computed from the sequential state after the
`block3` mask has already been applied. The two checkpoint selections remain
independent and may select the same spatial position.

Reproduce the retained setting with:

```bash
python main.py \
  --whitebox-model vit_base_patch16_224 \
  --checkpoints block3,block10 \
  --drop-ratios 0.051020408163,0.051020408163 \
  --progressive-patch-selector high
```

Superseded ViT checkpoint schedules and their tuning tables were removed from
this active document to prevent them from being mistaken for a reference or
default configuration.
