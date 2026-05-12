# Attention-fool

## Workflow

- Install dependencies: `pip install -r requirements.txt`
- Generate adversarial samples with MI-FGSM:
  `python main.py --mode attack --max-attacked-samples 5 --epsilon 0.0313725 --steps 10 --decay 1.0 --output-dir outputs/attack`
- Generate adversarial samples with FFT-stable patch score residual pollution MI-FGSM:
  `python main.py --mode attack --attack-type fft-residual-pollution --max-attacked-samples 5 --epsilon 0.0313725 --steps 10 --decay 1.0 --layers=-4,-2,-1 --lambda-pollution 1.0 --lambda-residual 1.0 --fft-topk 1 --output-dir outputs/attack_fft_residual`
- Export correctly classified clean samples:
  `python main.py --mode clean --max-attacked-samples 5 --output-dir outputs/clean`
- Evaluate transfer on saved adversarial samples:
  `python transfer_eval.py --image-dir outputs/attack --prefix adv_`
- Evaluate transfer on selected black-box timm models:
  `python transfer_eval.py --image-dir outputs/attack --prefix adv_ --model-name deit_base_patch16_224,swin_tiny_patch4_window7_224,pvt_v2_b2,cait_s24_224`
- Model selection is configured in `nets.py`.

## Visualization

- Attention, A@V, token FFT stability, image 2D FFT low/high frequency, patch score, and overlap visualization:
  `python visualize_attention_patchscores.py --image-dir outputs/clean --pattern "clean_*.png" --max-images 5 --output-dir outputs/attention_patchscores`
