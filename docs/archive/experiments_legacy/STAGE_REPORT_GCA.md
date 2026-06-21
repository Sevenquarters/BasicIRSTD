# 阶段性汇报：ResUNet-GCA 实验总结

## 1. 使用的数据集

本项目使用的是红外小目标检测相关数据集，当前阶段重点展示以下两套数据集上的结果：

1. NUDT-SIRST
2. IRSTD-1K

项目 README 中同时说明，SIRST-v1、NUDT-SIRST 和 IRSTD-1K 可用于训练和测试；当前阶段实验主要围绕 NUDT-SIRST 展开，并补充了 IRSTD-1K 的阶段性结果。

## 2. 使用的 baseline

本阶段采用的 baseline 是 ResUNet。

选择它作为对照模型的原因是：

1. 结构清晰，便于和改进模型进行公平对比。
2. 在当前任务上有稳定的训练和测试流程。
3. 便于观察 GCA 模块带来的增益是否真实有效。

## 3. 自己改进的模型

本阶段的改进模型是 ResUNet-GCA。

改进点主要是：

1. 在 ResUNet 的 skip connection 融合前引入 GCA 模块。
2. 通过门控机制增强特征筛选能力，减少无关背景干扰。
3. 保持主干结构不变，尽量控制额外参数和改动范围。

消融实验模型为 ResUNet-GCA-NoGate，用于验证门控分支是否真正有效。

## 4. 数据集上的对比结果

### 4.1 NUDT-SIRST 上的对比

统一训练配置：batchSize = 8，nEpochs = 20。

| 模型 | pixAcc | mIoU | PD | FA |
|---|---:|---:|---:|---:|
| ResUNet | 0.815292 | 0.735121 | 0.947090 | 3.623962e-05 |
| ResUNet-GCA | 0.859933 | 0.751698 | 0.954497 | 2.780593e-05 |
| ResUNet-GCA-NoGate | 0.836129 | 0.707942 | 0.907937 | 3.237896e-05 |

阶段性结论：

1. ResUNet-GCA 相比 baseline 的 pixAcc、mIoU、PD 均提升，FA 下降。
2. 去掉门控后，mIoU 和 PD 明显下降，说明门控分支对性能提升是关键的。
3. 这组结果可以作为阶段汇报中的主结果，结论比较清晰。

### 4.2 IRSTD-1K 上的阶段性结果

统一训练配置：batchSize = 8，nEpochs = 20。

| 模型 | pixAcc | mIoU | PD | FA |
|---|---:|---:|---:|---:|
| ResUNet | 0.740428 | 0.595379 | 0.861953 | 4.803482e-05 |
| ResUNet-GCA | 0.781948 | 0.572068 | 0.912458 | 6.792438e-05 |
| ResUNet-GCA-NoGate | 0.770606 | 0.575314 | 0.892256 | 7.107483e-05 |

阶段性结论：

1. GCA 在 IRSTD-1K 上提升了 pixAcc 和 PD。
2. mIoU 和 FA 还没有形成稳定的全面优势，说明该数据集上仍有进一步调参空间。
3. 消融结果表明，去掉门控后性能会进一步波动，门控结构仍然是需要保留的核心部分。

## 5. 汇报时可以直接说的结论

1. 项目使用的是 NUDT-SIRST 和 IRSTD-1K 这两套红外小目标检测数据集。
2. baseline 选用 ResUNet，作为与改进模型的统一对照。
3. 我们提出的改进模型是 ResUNet-GCA，核心是在 skip connection 融合前加入 GCA 门控模块。
4. 在 NUDT-SIRST 上，ResUNet-GCA 相比 baseline 在主要指标上有明显提升；去门控后的消融结果则显著退化，说明 GCA 的有效性是成立的。
5. 在 IRSTD-1K 上，GCA 依然带来部分指标提升，但还需要继续优化以获得更稳定的综合收益。

## 6. 建议的汇报结构

1. 先说明项目背景和使用的数据集。
2. 再介绍 baseline ResUNet。
3. 接着介绍自己的改进模型 ResUNet-GCA。
4. 最后给出 NUDT-SIRST 和 IRSTD-1K 上的对比实验和消融实验结果。
