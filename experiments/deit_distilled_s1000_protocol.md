# DeiT-B distilled 源配置

更新日期：2026-11-04。当前源模型为 `deit_base_distilled_patch16_224`。对每张干净图像计算十二个完整 block 输出的原始局部 token／最终归一化分类 CLS 余弦分数，同时观察各 block 输入／输出的归一化局部特征，以输出方向方差比 × 正收缩比例最大者定位，在该 block 输入处随机 hard-zero。每图只选一次边界，总 drop 比例为 20/196；每个 step/group 重采样位置。

CLS 与 distillation 两个 prefix 从局部 token 中排除，初始随机特征增强与 drop 都仅作用于局部位置。模型原生 forward_head 在 eval 模式下平均两个分类 head。方法细节见 [执行契约](prd_method_contract.md)。

论文采用共同 i.i.d. RGB 投影特征增强，显式运行 `main.py --whitebox-model deit_base_distilled_patch16_224 --noise-structure isotropic_rgb`。代码当前默认仍为 opponent，β=0.2。新执行 seed 42，原始／平移配对，保留 `--input-diversity-views-per-group` 消融；迁移评估使用 `transfer_eval.py --protocol fpr`。

当前乘积选层/opponent 千图记录位于 `outputs/attack/prd_direction_contraction_s42_20261104T044138Z_deit_b/`，Strict ASR 95.30%，见 [归档](../results/prd_direction_contraction_s42_1000_fpr/README.md)。历史最大方差动态/opponent 千图记录位于 `outputs/attack/prd_dynamic_variance_s42_20261029T031203Z_deit_b/`。旧固定 block2/block9 各 10 token 的结果保留原含义。最终 i.i.d. 主表须对应实际运行，证据索引见 [故事记录](prd_paper_story.md)。
