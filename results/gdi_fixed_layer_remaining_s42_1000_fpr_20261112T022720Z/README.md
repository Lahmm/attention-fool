# GDI：另外三个源模型的逐层固定输入干预消融

20/48 个配置已完成并通过逐图预测与 ASR 校验；状态见 `run_status.json`。

CaiT-S24 固定 24 个局部 block 输入（不含 class-attention），PiT-T 固定原生 2/6/4 stage 内全部 12 个 block 输入，蒸馏 DeiT-B 固定 12 个 block 输入。每配置均为 seed 42、同序 1000 图、10 steps、10 groups × 2 views、动量 1。增强强度按 CaiT/PiT-T/DeiT 为 0.2/0.4/0.2，攻击 batch 为 16/32/48，drop 比例为 30/196、30/196、20/196，并沿用主结果实际参数小数。PiT 三 stage 实际清零 112/30/8 个 token。

FPR13 评估 batch 32、关闭 TF32、无 AMP；ASR 为全部对抗样本的 1 - adversarial accuracy。Strict 按完全同源目标排除：CaiT 为 12 目标，PiT-T/蒸馏 DeiT-B 为 13 目标。

| 源 | 固定层 | 原生输入边界 | Strict ASR | 相对对应动态 GDI（百分点） |
| --- | ---: | --- | ---: | ---: |
| cait | 1 | block0 | 81.1917% | -7.7917 |
| cait | 2 | block1 | 77.5667% | -11.4167 |
| cait | 3 | block2 | 90.5667% | +1.5833 |
| cait | 4 | block3 | 80.3500% | -8.6333 |
| cait | 5 | block4 | 85.2917% | -3.6917 |
| cait | 6 | block5 | 81.1167% | -7.8667 |
| cait | 7 | block6 | 82.0250% | -6.9583 |
| cait | 8 | block7 | 75.1250% | -13.8583 |
| cait | 9 | block8 | 74.8250% | -14.1583 |
| cait | 10 | block9 | 76.0417% | -12.9417 |
| cait | 11 | block10 | 76.9833% | -12.0000 |
| cait | 12 | block11 | 72.3667% | -16.6167 |
| cait | 13 | block12 | 69.9250% | -19.0583 |
| cait | 14 | block13 | 75.3667% | -13.6167 |
| cait | 15 | block14 | 71.8417% | -17.1417 |
| cait | 16 | block15 | 67.3000% | -21.6833 |
| cait | 17 | block16 | 74.0417% | -14.9417 |
| cait | 18 | block17 | 84.8417% | -4.1417 |
| cait | 19 | block18 | 78.1250% | -10.8583 |
| cait | 20 | block19 | 84.6000% | -4.3833 |

`summary.csv` 和 `per_target.csv` 仅包含已校验的测量及直接 ASR 差值。对应动态主结果按原字节保存在 `baselines/`，保留原参数标签及源码身份；RNG 实现差异按项目约定统一为 seed-42 标准。

精确参数、边界、网格、命令和源码版本见 `execution_manifest.json`；样本顺序、标签和文件哈希见 `samples.json`；源及 FPR13 目标权重身份见 `weights.json`。`runs/<配置>/` 保留参数、逐目标 ASR、逐图预测、退出状态、日志、执行与校验记录、对抗图哈希；原始对抗图位置见 manifest。`provenance.json` 保存归档哈希清单。
