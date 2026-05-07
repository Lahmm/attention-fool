# Attention-fool

## Workflow

- Install dependencies: `pip install -r requirements.txt`
- Generate adversarial samples with MI-FGSM:
  `python main.py --mode attack --max-attacked-samples 5 --epsilon 0.0313725 --steps 10 --decay 1.0 --output-dir outputs/attack`
- Export correctly classified clean samples:
  `python main.py --mode clean --max-attacked-samples 5 --output-dir outputs/clean`
- Evaluate transfer on saved adversarial samples:
  `python transfer_eval.py --image-dir outputs/attack --prefix adv_`
- Model selection is configured in `nets.py`.

## Visualization

- Attention, A@V, token FFT stability, image 2D FFT low/high frequency, patch score, and overlap visualization:
  `python visualize_attention_patchscores.py --image-dir outputs/clean --pattern "clean_*.png" --max-images 5 --output-dir outputs/attention_patchscores`
