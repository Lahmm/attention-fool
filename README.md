# Attention-fool
## 流程
  - pip install -r requirements.txt 安装依赖，确保各脚本可运行。
  - python train.py --dataset cifar10 --data-dir data/cifar10 --download
    --epochs 20 --batch-size 128 --output-path checkpoints/vit_cifar10.pth 在
    CIFAR-10 上微调 ViT；--download 会在缺数据时拉取，训练完成后得到模型权重。
  - 生成对抗样本（使用 CIFAR‑10 预训练权重）
    python main.py --weights-path checkpoints/vit_cifar10.pth --dataset cifar10 --mode attack --max-attacked-samples 20 --pgd-step-size 8e-3 --output-dir outputs
  - 仅导出被正确分类的干净样本
    python main.py --weights-path checkpoints/vit_cifar10.pth --dataset cifar10 --mode clean --max-attacked-samples 20 --output-dir outputs/clean
    需要使用自定义图片/标注或修改 batch 大小时，可以在 main() 里追加关键字参
    数，如 python main.py ... --mode attack --pgd-step-size 4e-3 并在代码调用
    main(..., image_dir="...", annotations_path="...")
## 可视化注意力
  - 对之前保存在 outputs 目录的对抗图片生成注意力热力图
    python visualize_attention_from_images.py --weights-path checkpoints/vit_cifar10.pth --image-dir outputs --pattern "adv_*.png" --max-images 5 --cls-layer last --output-dir outputs/attention_from_images
