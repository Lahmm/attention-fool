# 动态攻击内部精简验证（2026-11-04）

本记录对应 commit 2122605 的结构精简；后续 2026-11-05 的独立视图改动会改变采样行为，不属于此次等价验证。

本记录用于代码维护。当前攻击、默认配置和历史测量的含义保持不变。

## 改动范围

`progressive_attack.py` 从 508 行减为 412 行。mask schedule 使用普通元组，保留一个带形状、布尔类型、预算和活动样本校验的 mask 记录类型；批内边界布局使用普通元组。删除四个测试专用访问属性，将反射平移参考实现移至测试。视图数量直接用实际梯度列表长度检查，梯度仍执行 `torch.stack(..., dim=0).mean(dim=0)`，归约顺序不变。

原来仅测试调用的 mask 数量检查移入测试，并增加每图恰好一个活动边界的独立断言。原生网格、样本数、mask 类型与范围、phase 约束、适配器状态及重放网格检查保留。选层、RGB 投影增强、随机调用顺序、视图顺序、动量和投影更新未改动。

基准为 commit `0760579f8c15809bda459f31ef4900fe45ee7a90`：

- 修改前 `progressive_attack.py` SHA256：`5724d824d988c109312269a1aa08c95960521769b06597e0dd33df26f8d5da83`。
- 修改后 SHA256：`a4a2bdaae3d5b2255ef11ae03528c89980e52b1bfb5fb6bbb38a58caf6c62451`。

## 验证范围

启用实际架构适配检查的完整测试共 43 项，全部通过：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RUN_REAL_PRD_ADAPTERS=1 python -B -m unittest discover -s tests
```

临时对照程序从修改前快照和当前实现分别运行，统一 seed 42、输入、batch 与参数。27 组对照逐元素相等，核对逐图选层、原始和相位 mask、相位抽样、投影增强、视图损失、逐步像素梯度、最终对抗输出、完整参数及 CPU/CUDA RNG 状态。

| 范围 | 对照设置 |
| --- | --- |
| 测试模型 CPU，9 组 | opponent、isotropic、单视图、零相位、无增强、无 drop、二者关闭、修改比例与强度、默认 10 steps × 10 groups × 2 views |
| 测试模型 CUDA，9 组 | 同上；各组先后运行 batch 3 和 batch 2，覆盖 distilled 双 prefix 与不同图像选择不同边界 |
| 四个实际预训练源 CUDA，8 组 | 每源分别 opponent/isotropic；排序前两张真实输入图，2 steps × 2 groups × 2 views，实际执行干净选层 |
| 预训练 PiT-T CUDA，1 组 | 同批两个样本分别干预 stage1 与 stage3 的不同原生网格；边界由测试注入以检验混合 batch 重放 |

上述 GPU 逐元素比较在临时验证进程中设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 和 `torch.use_deterministic_algorithms(True)`，生产入口与默认 CUDA 设置不变。普通 CUDA 设置下，修改前实现自身重复运行也出现末位梯度差异；该自对照的输出和随机状态相同。因此逐元素等价结论对应上述确定性验证条件。

完整权重缓存和结果目录的 470 个条目经过路径、大小、修改时间和符号链接核对，均未变化。此次没有重新生成主实验 ASR，也没有修改其归档。
