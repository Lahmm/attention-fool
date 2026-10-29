# ViT patch 选择六组独立实验

入口为 `tests/vit_drop_selection.py`，通过脚本内的子类复用 PRD 的前向、噪声、平移视图、梯度与投影更新；不修改 `main.py`、`progressive_attack.py` 或 `nets/`。这是单独的分析实验。

默认固定在 block10 输出（block11 输入）置零，每张图每个视图丢弃 20/196 个 local tokens；CLS 保持不变。干净图像完整前向一次，以 block10 的原始局部表示与最终 `model.norm` 后 CLS 的带符号余弦相似度评分。分数和候选池在整个攻击中固定。

| Case | 候选范围及选择方式 | 更新方式 |
| --- | --- | --- |
| high-fix | 分数最高的 20 个位置 | 全部步／组复用 |
| high-random | 分数较高的 50%，即 98 个位置中，均匀无放回抽 20 个 | 每个 step/group 重抽 |
| low-fix | 分数最低的 20 个位置 | 全部步／组复用 |
| low-random | 分数较低的 50%，即 98 个位置中，均匀无放回抽 20 个 | 每个 step/group 重抽 |
| random-fix | 全部 196 个位置中，均匀无放回抽 20 个 | 每张图抽一次后复用 |
| random | 全部 196 个位置中，均匀无放回抽 20 个 | 每个 step/group 重抽 |

`--drop-ratio` 可改变共同预算，实际数量为 `max(1, round(196*r))`，范围为 `(0,0.5]`；表中的 20 是默认值。排序使用原始分数，不取绝对值、不做 min–max；分数相同时按 patch 索引升序排列，前 98 个分给 low，后 98 个分给 high，因此两池互不重叠。high-fix 取这一排序最后 k 个，low-fix 取最前 k 个。

同组原始／平移视图共享空间对应的 mask，沿用 occupancy 映射保持丢弃数量。平移 mask 不在平移视图上重新评分，映射后的网格位置也不再按原始候选池限制。fix 指原始视图位置固定；相位变化时对应的平移 mask 会变化。

六组均为 seed 42，默认 10 steps、epsilon=16/255、步长 epsilon/steps、10 groups×2 views、opponent β=0.2。每个 batch 的六组恢复同一 PyTorch RNG 起点，全部 case 每组均消耗相同形状的全网格随机排列；fixed case 也采样后丢弃候选。脚本校验各 case 的 RNG 终点一致，因此共享噪声随机系数与平移抽样序列。特征 RMS 随对抗样本变化，匹配后的噪声数值不要求相同。结果依赖所记录的 batch 布局。

运行全部六组并完成 FPR13 评估：

```bash
python tests/vit_drop_selection.py --stage all \
  --output-dir outputs/attack/vit_drop_selection_s42
```

只生成对抗图，或稍后单独评估：

```bash
python tests/vit_drop_selection.py --stage attack \
  --output-dir outputs/attack/vit_drop_selection_s42
python tests/vit_drop_selection.py --stage evaluate \
  --output-dir outputs/attack/vit_drop_selection_s42
```

可用 `--cases high-fix high-random` 选择子集；评估时也需指定相应子集。攻击阶段要求输出目录为空，已有数据不会清空或覆盖。默认 1000 图、攻击 batch 48、评估 batch 32。`--input-diversity-views-per-group 1` 保留单视图消融。GPU 内存不足时降低 `--batch-size`，同一次六组对照始终共用相同 batch。

验证脚本而不执行正式实验：

```bash
python -m unittest tests.test_vit_drop_selection
```

小规模真实模型检查（不代表正式 ASR）：

```bash
python tests/vit_drop_selection.py --stage attack --max-attacked-samples 2 \
  --batch-size 2 --steps 1 --input-diversity-groups 1 \
  --output-dir outputs/attack/vit_drop_selection_smoke_s42
```

每组目录保存 `adv_*.png`、`attack_params.json`，以及 `selection_traces/batch_*.pt`：逐图干净分数、候选池索引、每个 step/group 的实际原始 mask 索引。根目录的 `experiment.json` 保存图像顺序、标签、图像及相关代码 SHA256、模型配置、参数与完成状态。

评估复用 `transfer_eval.py --protocol fpr --no-tf32`，不使用 AMP。保存每组 FPR13 逐目标结果，以及根目录的 `asr_comparison.csv`、`asr_per_target.csv`。ASR 使用所有对抗样本的 `1 - adversarial accuracy`；严格黑盒排除源 ViT-B/16。差值为百分点，fix case 以 random-fix 为参照，重采样 case 以 random 为参照；未运行对应参照时差值留空。

high-fix／low-fix 与各自 random 版本的差异同时包含极端 top-k 与半池采样的覆盖差异；random 与 random-fix 则直接比较同一候选池中是否重新采样。分析六组结果时保留这一区别。
