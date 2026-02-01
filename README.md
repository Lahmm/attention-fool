# Attention-fool
## 流程
  - `pip install -r requirements.txt` 安装依赖，确保各脚本可运行。
  - 生成对抗样本 `python main.py --mode attack --max-attacked-samples 5 --attn-target-mode avg --attn-map-to-patch --output-dir outputs/attack`
  - 仅导出被正确分类的干净样本 `python main.py --mode clean --max-attacked-samples 5 --pgd-step-size 0.0313725 --output-dir outputs/clean`
  - 所有模型在nets.py中修改，修改基本模型名称
## 可视化注意力
  - 单张图片 `python visualize_attention_from_images.py --image-dir outputs/clean --pattern "adv_*.png" --max-images 5 --cls-layer last --output-dir outputs/attention_from_images`
  - 模型所有层 `python visualize_attention_all_layers.py --image-path outputs/clean/clean_00000.png`