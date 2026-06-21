# DNANet 上下文分支阶段进展

## 1. 目的

本文件用于统一记录 `DNANet-LDEM-Gate-Stable` 之后围绕“中层上下文建模”展开的两条路线：

- `DNANet-LDEM-Gate-Stable-Context`
- `DNANet-LDEM-Gate-Stable-ContextGate`

核心问题不是“上下文有没有用”，而是：

- 如何让上下文在复杂背景上带来增益
- 同时尽量减少对简单场景的副作用


## 2. 当前已完成内容

- 已完成 `Context` 结构实现与双数据集 40 epoch 训练
- 已完成 `ContextGate` 结构实现与双数据集 40 epoch 训练
- 已完成最小化前向与反向验证


## 3. 双数据集结果

### 3.1 IRSTD-1K

| 模型 | Best Epoch | Best mIoU | Final mIoU | Stability Gap | Best PD | Final PD | Best FA | Final FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DNANet-LDEM-Gate-Stable | 31 | 0.6387 | 0.6331 | 0.0055 | 0.8990 | 0.8889 | 2.6266e-05 | 1.9833e-05 |
| DNANet-LDEM-Gate-Stable-Context | 36 | 0.6475 | 0.6371 | 0.0104 | 0.9125 | 0.9293 | 4.7200e-05 | 4.4562e-05 |
| DNANet-LDEM-Gate-Stable-ContextGate | 30 | 0.6458 | 0.6126 | 0.0332 | 0.8788 | 0.8889 | 2.2547e-05 | 2.2300e-05 |

结论：

- `Context` 是当前 `IRSTD-1K` 上最好的上下文增强版本
- `ContextGate` 虽然也能抬升 best，但 final 回落更明显
- 这说明“直接上下文”在复杂背景中依然更有效，但还缺乏稳定控制


### 3.2 NUDT-SIRST

| 模型 | Best Epoch | Best mIoU | Final mIoU | Stability Gap | Best PD | Final PD | Best FA | Final FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DNANet-LDEM-Gate-Stable | 40 | 0.8803 | 0.8803 | 0.0000 | 0.9757 | 0.9757 | 6.6872e-06 | 6.6872e-06 |
| DNANet-LDEM-Gate-Stable-Context | 39 | 0.8226 | 0.7843 | 0.0384 | 0.9693 | 0.9841 | 6.8021e-06 | 1.6201e-05 |
| DNANet-LDEM-Gate-Stable-ContextGate | 38 | 0.8554 | 0.8263 | 0.0291 | 0.9852 | 0.9545 | 1.0686e-05 | 1.4983e-05 |

结论：

- `ContextGate` 明显优于 `Context`
- 这说明“上下文需要自适应启用”这个判断成立
- 但 `ContextGate` 仍未追平 `Stable`


## 4. 当前研究判断

### 4.1 已经确认的事情

- 中层上下文建模方向是有效的
- `IRSTD-1K` 确实能从更大范围背景信息中获益
- `NUDT-SIRST` 不适合无条件强上下文注入
- 因此“条件式上下文”是后续必须保留的设计思想

### 4.2 当前最合理的角色定义

- `Stable`：双数据集通用主线
- `Context`：复杂背景特化分支
- `ContextGate`：过渡性验证版本，证明“条件式上下文”方向正确


## 5. 下一步问题定义

第五版不应该继续盲目加大上下文强度，而应重点解决：

1. 为什么 `Context` 在 `IRSTD-1K` 有效，但在 `NUDT-SIRST` 上干扰过强
2. 为什么 `ContextGate` 在 `NUDT-SIRST` 上更合理，但在 `IRSTD-1K` 上又削弱了 final
3. 如何做一个“更轻、更局部、更难样本优先”的上下文版本


## 6. 下一步推荐

- 不继续做重型上下文
- 不回到浅层强 shape 注入
- 优先推进第五版：
  - `DNANet-LDEM-Gate-Stable-ContextLiteGate`
