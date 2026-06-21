# DNANet-LDEM-Gate-Stable-Context 设计说明

## 1. 设计动机

当前 `DNANet-LDEM-Gate-Stable` 的优势在于浅层细节保留和后期训练稳定性较好，但在复杂背景下仍可能遇到两类问题：

- 伪目标点在局部纹理上和真实红外小目标相似，仅依赖浅层增强不够
- `IRSTD-1K` 这类场景中，后期波动往往和“缺乏更大范围背景对比信息”有关

因此这一版不再继续强化浅层分支，而是转向中层上下文建模。

## 2. 核心思路

在 `Stable` 主干保持不变的前提下，于 `x2_2` 中层融合特征后插入一个轻量上下文模块 `ContextAggregationUnit`，用来补充长程背景抑制信息。

该模块由四类上下文分支构成：

- `3x3 depthwise conv`：保留局部上下文
- `5x5 depthwise conv`：扩大近邻感受野
- `1x7 + 7x1 strip conv`：建模横向/纵向长条状空间依赖
- `global average pooling` 分支：提供整幅图像级背景统计

然后将多分支上下文融合，并通过一个门控分支控制其注入强度，最后采用有界残差形式回写：

- `refined = tanh(refined) * residual_scale`
- `output = ReLU(x + refined)`

这样做的目的是：

- 引入更大范围的背景对比信息
- 不直接扰动最脆弱的浅层细节路径
- 保持和 `Stable` 一致的“受控残差”风格，降低后期震荡风险

## 3. 代码插入位置

- 上下文模块定义：
  - [model_DNANet.py](D:\Program Files (x86)\IRSTD\BasicIRSTD\model\DNANet\model_DNANet.py:365)
- 新模型定义：
  - [model_DNANet.py](D:\Program Files (x86)\IRSTD\BasicIRSTD\model\DNANet\model_DNANet.py:1032)
- 模型导出注册：
  - [__init__.py](D:\Program Files (x86)\IRSTD\BasicIRSTD\model\__init__.py:8)
- 训练入口注册：
  - [net.py](D:\Program Files (x86)\IRSTD\BasicIRSTD\net.py:41)

## 4. 与前几版的区别

- `LDEM / Gate / Stable`：主要解决浅层小目标细节保留与误增强问题
- `Shape / ShapeLite / AuxShapeLoss`：主要围绕形状约束与局部结构先验
- `Context`：重点转向“中层长程背景建模”

这意味着它的研究叙事更偏向：

- 为什么复杂背景仍会造成误检
- 为什么需要更大范围的上下文来辅助伪目标抑制
- 为什么将上下文放在中层而非浅层，更有利于兼顾稳定性

## 5. 当前状态

已完成：

- 结构实现
- `Net` 注册
- 随机输入前向测试
- 反向传播测试
- 推理模式输出测试

当前可以直接进入正式训练与对比实验。
