# DeiT-B distilled 源配置

当前 DeiT-B 源模型为 `deit_base_distilled_patch16_224`。与 ViT-B/16 共享 PRD 遍历，checkpoint 为 block3、block10，各清零 10/196 个局部 token；opponent strength 为 0.2。

初始状态将局部 token 与两个 prefix（CLS、distillation）分开。初始噪声和 checkpoint masks 仅作用于局部 token，最终调用模型原生 forward_head，在 eval 模式下平均两个分类 head。

新执行统一使用 seed 42，通过 `main.py --whitebox-model deit_base_distilled_patch16_224` 调用。原始／平移视图为默认设置，保留视图数消融参数。目标协议由 `transfer_eval.py --protocol project|fpr` 单独选择。

测试覆盖原生与恢复前向一致、两个 prefix 保护、实际清零数量及有限非零像素梯度。旧实验结果保留原测量含义，不因本次源支持调整改写。
