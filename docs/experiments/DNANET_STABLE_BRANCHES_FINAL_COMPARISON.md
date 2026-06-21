# DNANet Stable 分支总对比与避雷结论

## 1. 目的

本文件用于收束 `DNANet-LDEM-Gate-Stable` 之后的各条改进分支，保留最有价值的实验事实与研究结论，减少冗长过程记录。

当前重点关注的分支包括：

- `DNANet-LDEM-Gate-Stable`
- `DNANet-LDEM-Gate-Stable-Shape`
- `DNANet-LDEM-Gate-Stable-ShapeLite`
- `DNANet-LDEM-Gate-Stable-AuxShapeLoss`
- `DNANet-LDEM-Gate-Stable-Context`
- `DNANet-LDEM-Gate-Stable-ContextGate`


## 2. 当前主结论

### 2.1 主模型判断

如果只看当前最可靠、最适合作为主线基础的模型：

- `DNANet-LDEM-Gate-Stable` 仍然是最稳的基础版本

如果看当前最值得继续推进、最可能带来新提升的结构方向：

- `DNANet-LDEM-Gate-Stable-Context` 是当前优先级最高的新主线

如果看“方向判断是否正确、但实现还需要再打磨”的版本：

- `DNANet-LDEM-Gate-Stable-ContextGate`

如果看最有研究增量、但更适合作为辅线消融的方向：

- `DNANet-LDEM-Gate-Stable-AuxShapeLoss`


## 3. 双数据集关键结果

### 3.1 IRSTD-1K

| 模型 | Best Epoch | Best mIoU | Final mIoU | Stability Gap | Best PD | Final PD | Best FA | Final FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DNANet-LDEM-Gate-Stable | 31 | 0.6387 | 0.6331 | 0.0055 | 0.8990 | 0.8889 | 2.6266e-05 | 1.9833e-05 |
| DNANet-LDEM-Gate-Stable-Shape | 23 | 0.6420 | 0.6192 | 0.0227 | 0.9192 | 0.9226 | 3.9646e-05 | 5.9232e-05 |
| DNANet-LDEM-Gate-Stable-ShapeLite | 29 | 0.6470 | 0.5870 | 0.0600 | 0.9057 | 0.9158 | 3.5034e-05 | 7.1227e-05 |
| DNANet-LDEM-Gate-Stable-AuxShapeLoss | 33 | 0.6614 | 0.6176 | 0.0438 | 0.9125 | 0.8889 | 2.2622e-05 | 2.6190e-05 |
| DNANet-LDEM-Gate-Stable-Context | 36 | 0.6475 | 0.6371 | 0.0104 | 0.9125 | 0.9293 | 4.7200e-05 | 4.4562e-05 |
| DNANet-LDEM-Gate-Stable-ContextGate | 30 | 0.6458 | 0.6126 | 0.0332 | 0.8788 | 0.8889 | 2.2547e-05 | 2.2300e-05 |

### 3.2 NUDT-SIRST

当前用于判断 `Context` 泛化价值的关键对照：

| 模型 | Best Epoch | Best mIoU | Final mIoU | Stability Gap | Best PD | Final PD | Best FA | Final FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DNANet-LDEM-Gate-Stable | 40 | 0.8803 | 0.8803 | 0.0000 | 0.9757 | 0.9757 | 6.6872e-06 | 6.6872e-06 |
| DNANet-LDEM-Gate-Stable-Context | 39 | 0.8226 | 0.7843 | 0.0384 | 0.9693 | 0.9841 | 6.8021e-06 | 1.6201e-05 |
| DNANet-LDEM-Gate-Stable-ContextGate | 38 | 0.8554 | 0.8263 | 0.0291 | 0.9852 | 0.9545 | 1.0686e-05 | 1.4983e-05 |


## 4. 各分支保留结论

### 4.1 Stable

实现方式：

- 在 `LDEM + Gate` 之后增加 `StabilityRefinementUnit`
- 通过受控残差、语义反校正和跨层一致性抑制后期波动

得到结果：

- 明显修复了 `Gate` 在 `IRSTD-1K` 上后期崩塌的问题
- 成为当前最稳的基础版本

结论：

- 这是后续所有改进的基础底座


### 4.2 Shape

实现方式：

- 在浅层稳定特征上直接注入较强 shape prior refinement

得到结果：

- 峰值略高于 `Stable`
- 但 final 明显下降

结论：

- 直接强 shape 注入会扰动主干后期稳定性
- 不推荐继续作为主线推进


### 4.3 ShapeLite

实现方式：

- 以更温和的 shape modulation 替代强 shape refinement

得到结果：

- 峰值仍有提升
- final 掉点更明显

结论：

- “轻注入”优于“强注入”
- 但浅层主干仍然不适合直接长期承载 shape 约束


### 4.4 AuxShapeLoss

实现方式：

- 不修改主干推理路径
- 仅增加辅助 shape 监督头和辅助损失

得到结果：

- 拿到当前这些分支里最高的 best mIoU：`0.6614`
- final 仍低于 `Stable`

结论：

- shape 信息本身是有用的
- 但更适合作为训练监督，而不是直接改写主干特征
- 这条线保留为“辅线科研增量方向”


### 4.5 Context

实现方式：

- 不继续改浅层稳定路径
- 在中层融合特征 `x2_2` 上增加轻量 `ContextAggregationUnit`
- 通过局部卷积、条带卷积和全局池化聚合上下文，并用有界残差注入

得到结果：

- `IRSTD-1K` 上 best mIoU 超过 `Stable`
- `IRSTD-1K` 上 final mIoU 也超过 `Stable`
- `NUDT-SIRST` 上则明显不如 `Stable`
- 后期仍有波动，说明上下文注入强度对不同数据集的适配性并不一致

结论：

- `Context` 当前更像“复杂背景特化增强分支”
- 它并不适合作为无条件替代 `Stable` 的通用主模型
- 它更符合 `IRSTD-1K` 这类复杂背景场景下“需要更大范围上下文”的问题本质


### 4.6 ContextGate

实现方式：

- 保留 `Context` 的中层上下文分支
- 在上下文注入前增加自适应门控
- 目标是只在复杂背景和困难区域启用更强上下文

得到结果：

- `NUDT-SIRST` 上明显优于直接 `Context`
- 说明“上下文需要自适应启用”的判断是对的
- 但 `IRSTD-1K` 和 `NUDT-SIRST` 上的 final 仍都不如 `Stable`
- 同时在 `IRSTD-1K` 上也没有超过 `Context`

结论：

- `ContextGate` 是一次很有价值的中间验证
- 它证明了“条件式上下文”方向正确
- 但当前实现还不够轻，也不够稳，还不能作为最终主线


## 5. 避雷总结

后续如果继续做新结构，建议优先避开以下方向：

- 不要继续在 `x0_0` 浅层路径上叠加更强 shape 注入
- 不要把“峰值略涨”误判为“整体更优”，必须同时看 final 和 stability gap
- 不要把 `AuxShapeLoss` 当作新的主模型，它更适合做训练监督型创新点

更推荐的方向：

- 继续完善 `Context`
- 继续完善 `ContextGate`
- 或者探索 `Stable + Context + 更温和监督` 的组合方式
- 不建议直接把当前 `Context` 版本当作双数据集统一最优解


## 6. 当前研究判断

- `Stable`：最稳的双数据集通用主线
- `Context`：当前第一优先级复杂背景特化分支
- `ContextGate`：验证“条件式上下文”方向正确，但当前实现未收敛成最优解
- `AuxShapeLoss`：保留为辅线研究方向
- `Shape / ShapeLite`：完成验证价值后不再作为主要推进对象
