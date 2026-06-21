# DNANet 第五版改进方向

## 1. 第四版之后的问题

截至当前，围绕上下文分支已经得到两条关键结论：

- `Context` 在 `IRSTD-1K` 上有效，但会明显干扰 `NUDT-SIRST`
- `ContextGate` 证明了“上下文需要自适应启用”这一判断是正确的，但当前实现仍不够稳

因此第五版的目标不应该是“继续做更强上下文”，而应该是：

- 保留复杂背景上的收益
- 进一步减小简单场景上的副作用
- 减少后期训练波动


## 2. 第五版推荐模型名

- `DNANet-LDEM-Gate-Stable-ContextLiteGate`


## 3. 为什么推荐这条路线

### 3.1 不是继续做更重模块

当前问题已经不是“特征不够强”。

如果继续堆更多上下文、多尺度、甚至更重的全局模块，最大的风险是：

- `IRSTD-1K` 峰值可能还会涨一点
- 但 `NUDT-SIRST` 会进一步被扰动
- 研究故事会变得混乱，不利于后续写作

### 3.2 不是回到浅层路径折腾

前面的 `Shape / ShapeLite` 已经说明：

- 直接在浅层主干做强介入，虽然可能抬高局部峰值
- 但很容易破坏最终稳定性

### 3.3 最合理的下一步

所以第五版最合理的方向是：

- 保留“条件式上下文”思想
- 但把上下文分支做得更轻、更克制、更局部


## 4. 第五版核心思路

### 4.1 用更轻的上下文替代当前完整 Context

建议只保留最有效的几类信息：

- `3x3 depthwise local context`
- `1x7 + 7x1 strip context`
- `global pooled scene gate`

建议删除或弱化：

- 当前 `Context` 里较重的 `5x5` 分支
- 过强的多路融合残差

### 4.2 让 gate 只做“困难区域启用”

第五版的 gate 不应该做“整块特征普遍重加权”，而应该更像：

- 对局部复杂背景区域开放上下文
- 对简单区域尽量保持 `Stable`

也就是说，gate 的目标不是“增强全部区域”，而是“只增强需要上下文的那部分区域”。

### 4.3 继续保留有界残差

因为前面几版已经证明：

- 无界残差和过强注入都容易带来后期波动

所以第五版仍然建议：

- `refined = tanh(refined) * small_residual_scale`

但可以把初始 `residual_scale` 再调小一点。


## 5. 结构建议

### 插入位置

仍然放在：

- `x2_2`

原因：

- 这里是当前最适合做中层上下文补充的位置
- 不会直接破坏浅层细节主路径
- 已经经过前面两版验证

### 建议模块组成

1. `LightContextBranch`
- `3x3 depthwise conv`
- `1x7 strip conv`
- `7x1 strip conv`

2. `LiteContextFuse`
- 轻量 `1x1` 融合

3. `HardRegionGate`
- 输入建议使用：
  - 原始 `x2_2`
  - 轻量 context feature
  - `abs(x - context)`
- 输出 `gate_map`

4. `SceneCoefficient`
- 用全局池化生成样本级控制因子
- 防止整张干净图都被过度上下文化

5. `BoundedResidualInjection`
- 保留 `tanh * residual_scale`


## 6. 第五版希望解决什么

### 在 IRSTD-1K 上

- 尽量保留 `Context` 的 best / final 收益
- 减少 `ContextGate` 后期回落过大的问题

### 在 NUDT-SIRST 上

- 至少要优于当前 `Context`
- 最理想情况是进一步逼近 `Stable`


## 7. 实验优先级

建议实验顺序：

1. 先实现 `ContextLiteGate`
2. 先做最小化验证
3. 先跑 `IRSTD-1K + 40 epoch`
4. 如果 `IRSTD-1K` 表现合理，再补 `NUDT-SIRST + 40 epoch`

原因：

- 第五版的主要目标仍然是保住复杂背景收益
- `NUDT-SIRST` 是第二阶段用来检验副作用是否减弱的


## 8. 当前一句话判断

第五版最应该做的，不是“更强上下文”，而是“更轻、更克制、只在困难区域生效的上下文”。
