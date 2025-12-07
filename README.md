# Attention-fool
## 流程
  - pip install -r requirements.txt 安装依赖，确保各脚本可运行。
  - python train.py --dataset cifar10 --data-dir data/cifar10 --download
    --epochs 20 --batch-size 128 --output-path checkpoints/vit_cifar10.pth 在
    CIFAR-10 上微调 ViT；--download 会在缺数据时拉取，训练完成后得到模型权重。
  - python main.py --dataset cifar10 --data-dir data/cifar10 --weights-path
    checkpoints/vit_cifar10.pth --max-attacked-samples 5 --pgd-step-size 0.0314
    --output-dir outputs 用刚训练好的权重在 CIFAR-10 测试集上先评估再执行注意力
    补丁攻击，成功的对抗样本会保存在 outputs/。
