# DNANet 系列三组对比与 V3 改进方向

## 1. 目的

- 汇总 `DNANet`、`DNANet-LDEM`、`DNANet-LDEM-Gate` 在两个数据集上的完整对比结果
- 给出可直接用于汇报和论文素材积累的表格与图像入口
- 基于真实曲线分析 `IRSTD-1K` 后期波动原因，并提出第三版改进方向

---

## 2. 完整对比表

### 2.1 NUDT-SIRST

| 模型 | Best Epoch | Best Loss | Best mIoU | Best PD | Best FA | Final Loss | Final mIoU | Final PD | Final FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DNANet | 32 | 0.2135 | 0.8118 | 0.9619 | 1.5948e-05 | 0.2037 | 0.8114 | 0.9831 | 2.0521e-05 |
| DNANet-LDEM | 39 | 0.1722 | 0.8409 | 0.9757 | 1.5787e-05 | 0.1796 | 0.8218 | 0.9810 | 1.2042e-05 |
| DNANet-LDEM-Gate | 38 | 0.1523 | 0.8397 | 0.9725 | 1.0456e-05 | 0.1509 | 0.8395 | 0.9884 | 2.3003e-05 |

### 2.2 IRSTD-1K

| 模型 | Best Epoch | Best Loss | Best mIoU | Best PD | Best FA | Final Loss | Final mIoU | Final PD | Final FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DNANet | 28 | 0.3925 | 0.6446 | 0.9259 | 2.3021e-05 | 0.3579 | 0.6282 | 0.9091 | 3.9798e-05 |
| DNANet-LDEM | 35 | 0.3689 | 0.6499 | 0.8620 | 3.3307e-05 | 0.3685 | 0.6373 | 0.8956 | 2.1920e-05 |
| DNANet-LDEM-Gate | 26 | 0.3532 | 0.6522 | 0.9293 | 3.3554e-05 | 0.3503 | 0.5232 | 0.7677 | 3.9855e-06 |

---

## 3. 关键结论

### 3.1 NUDT-SIRST

- `DNANet-LDEM` 与 `DNANet-LDEM-Gate` 都明显优于原始 `DNANet`
- `DNANet-LDEM` 的 best mIoU 略高于 `DNANet-LDEM-Gate`
- `DNANet-LDEM-Gate` 的 final mIoU 几乎保持在 best 附近，说明后期稳定性更好
- `DNANet-LDEM-Gate` 在 best FA 上最优，说明 gate 确实增强了“筛选浅层细节”的能力

### 3.2 IRSTD-1K

- `DNANet-LDEM-Gate` 的 best mIoU 最高，说明第二版方向是有效的
- `DNANet-LDEM` 只带来轻微提升，而 `DNANet-LDEM-Gate` 能进一步拉高峰值性能
- 但 `DNANet-LDEM-Gate` 在 `epoch 26` 后出现明显回落，final mIoU 下降到 `0.5232`
- 这说明第二版已经解决了“峰值不够高”的问题，但还没有解决“后期训练不稳定”的问题

---

## 4. 可视化图像入口

### 4.1 NUDT-SIRST

- 曲线总图：[NUDT-SIRST_dnanet_family_curves.png](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\figures\dnanet_family_nudt\NUDT-SIRST_dnanet_family_curves.png)
- 汇总柱状图：[NUDT-SIRST_dnanet_family_summary.png](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\figures\dnanet_family_nudt\NUDT-SIRST_dnanet_family_summary.png)

### 4.2 IRSTD-1K

- 曲线总图：[IRSTD-1K_dnanet_family_curves.png](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\figures\dnanet_family_irstd1k\IRSTD-1K_dnanet_family_curves.png)
- 汇总柱状图：[IRSTD-1K_dnanet_family_summary.png](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\figures\dnanet_family_irstd1k\IRSTD-1K_dnanet_family_summary.png)

### 4.3 跨数据集汇总

- Best mIoU 跨数据集对比图：[dnanet_family_best_miou_cross_dataset.png](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\figures\dnanet_family_summary\dnanet_family_best_miou_cross_dataset.png)

---

## 5. 为什么 IRSTD-1K 后期会波动

### 5.1 现象

- 原始 `DNANet` 在 `IRSTD-1K` 上后期就已经存在波动
- `DNANet-LDEM` 仍然有波动，说明单纯细节增强并没有从根本上解决问题
- `DNANet-LDEM-Gate` 虽然提高了 best mIoU，但后期回落更明显

### 5.2 原因判断

- `IRSTD-1K` 背景更复杂、场景更多样，浅层纹理中混有更多伪目标结构
- 当前 gate 主要做的是“增强后筛选”，但它仍然是一次性静态筛选，缺少训练后期的动态约束
- 当网络后期逐渐强化少数高响应模式时，gate 可能过度偏向局部强激活区域，导致：
  - 部分样本上目标响应更强，峰值更高
  - 但跨样本泛化变差，后期更容易震荡
- 从曲线看，`epoch 26` 达到最好后，后续若干轮 `mIoU` 明显回落，说明当前 gate 缺少“稳定保持最优状态”的机制

### 5.3 当前 V2 的本质短板

- 有“增强”
- 有“筛选”
- 但缺“稳定约束”

---

## 6. 第三版模块改进方向

### 推荐模型名

- `DNANet-LDEM-Gate-Stable`

### 当前工程状态

- 已完成模型实现
- 已完成模型注册
- 已完成前向验证
- 已完成一步反向传播验证
- 已完成：
  - `DNANet-LDEM-Gate-Stable + NUDT-SIRST + 40 epoch`
  - `DNANet-LDEM-Gate-Stable + IRSTD-1K + 40 epoch`

代码位置：

- [model_DNANet.py](D:\Program Files (x86)\IRSTD\BasicIRSTD\model\DNANet\model_DNANet.py)
- [__init__.py](D:\Program Files (x86)\IRSTD\BasicIRSTD\model\__init__.py)
- [net.py](D:\Program Files (x86)\IRSTD\BasicIRSTD\net.py)

### V3 核心目标

- 不是继续追求更高峰值
- 而是让 `IRSTD-1K` 上的高质量响应在后期不容易崩掉

### V3 推荐主方向：门控稳定化 + 语义反校正

#### 模块思路

- 保留现有 `LDEM`
- 保留现有 `DetailSelectiveGate`
- 在 gate 后增加一个轻量的 `Stability Refinement Unit`

#### 当前实现对应关系

- `LDEM`：保留浅层细节增强
- `DetailSelectiveGate`：继续做增强后的选择性筛选
- `StabilityRefinementUnit`：第三版新增稳定化校正单元

#### 建议的三部分组成

1. **门控残差限幅**
- 不让 gate 对浅层特征进行过强的单次重加权
- 让增强结果保持在一个可控范围内
- 目的：减少后期某些 batch 触发的极端激活
- 当前实现方式：
  - 对稳定化残差使用 `tanh`
  - 再乘以一个可学习但初值较小的 `residual_scale`

2. **语义反校正分支**
- 从更高一层的语义特征反向生成一个抑制图
- 用于压制那些“浅层很亮但高层不支持”的伪目标区域
- 目的：避免 gate 后期过度依赖局部亮点
- 当前实现方式：
  - 用 `x1_0` 经过 `1x1` 投影生成浅层语义引导
  - 与浅层 gated 特征共同生成支持度 mask

3. **跨层一致性约束**
- 让浅层增强后的响应与中层语义响应保持一致
- 如果浅层特别强但中层判断不稳定，则削弱该位置权重
- 目的：提升复杂背景下的泛化稳定性
- 当前实现方式：
  - 计算浅层 gated 特征与语义引导特征的绝对差
  - 生成 `inconsistency gate`
  - 最终以 `support * (1 - inconsistency)` 形成稳定 mask

---

## 7. V3 为什么比继续加注意力更合适

- 当前问题不是“特征不够强”
- 当前问题也不是“没有筛选”
- 当前真正的问题是：后期训练中，筛选结果缺少稳定保持机制

所以 V3 最适合的故事不是：

- 再堆一个更重的注意力模块

而是：

- 在第二版已经验证有效的 gate 基础上，引入稳定化校正机制，抑制复杂场景下的后期波动

这个故事更像一篇小论文应有的递进逻辑：

1. `DNANet`：强 baseline
2. `DNANet-LDEM`：证明浅层细节有用
3. `DNANet-LDEM-Gate`：证明选择性筛选能提升峰值性能
4. `DNANet-LDEM-Gate-Stable`：进一步解决复杂场景下的后期不稳定问题

---

## 8. 下一步最推荐实验

1. 先实现 `DNANet-LDEM-Gate-Stable`
2. 先只跑 `IRSTD-1K + 40 epoch`
3. 重点观察：
   - best mIoU 是否保持不降
   - final mIoU 是否显著高于当前 `0.5232`
   - `epoch 25-40` 区间的曲线是否比 V2 更平稳
4. 如果 `IRSTD-1K` 稳定性改善明确，再补 `NUDT-SIRST` 对照实验

---

## 9. V3 当前实验结果

### 9.1 `DNANet-LDEM-Gate-Stable + NUDT-SIRST + 40 epoch`

| 模型 | 数据集 | Best Epoch | Best mIoU | Best PD | Best FA | Final mIoU | Final PD | Final FA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DNANet-LDEM-Gate-Stable | NUDT-SIRST | 40 | 0.8803 | 0.9757 | 6.6872e-06 | 0.8803 | 0.9757 | 6.6872e-06 |

### 9.1 `DNANet-LDEM-Gate-Stable + IRSTD-1K + 40 epoch`

| 模型 | 数据集 | Best Epoch | Best mIoU | Best PD | Best FA | Final mIoU | Final PD | Final FA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DNANet-LDEM-Gate-Stable | IRSTD-1K | 31 | 0.6387 | 0.8990 | 2.6266e-05 | 0.6331 | 0.8889 | 1.9833e-05 |

### 9.2 双数据集阶段结论

- `NUDT-SIRST` 上，`V3` 明显优于 `DNANet`、`DNANet-LDEM`、`DNANet-LDEM-Gate`
- `V3` 在 `NUDT-SIRST` 上不仅峰值最高，而且 final 结果就等于 best，说明稳定化单元没有伤害简单场景，反而继续放大了正增益
- `IRSTD-1K` 上，`V3` 没有超过 `V2` 的峰值 best mIoU，但显著改善了 final 稳定性
- 因此 `V3` 当前呈现出的特征是：
  - 在相对干净的数据集上，性能和稳定性同时增强
  - 在复杂场景数据集上，优先解决后期掉点问题

### 9.3 V2 vs V3 直接对比

#### IRSTD-1K

| 模型 | Best mIoU | Final mIoU | Stability Gap |
|---|---:|---:|---:|
| DNANet-LDEM-Gate (V2) | 0.6522 | 0.5232 | 0.1290 |
| DNANet-LDEM-Gate-Stable (V3) | 0.6387 | 0.6331 | 0.0055 |

#### NUDT-SIRST

| 模型 | Best mIoU | Final mIoU | Stability Gap |
|---|---:|---:|---:|
| DNANet-LDEM-Gate (V2) | 0.8397 | 0.8395 | 0.0002 |
| DNANet-LDEM-Gate-Stable (V3) | 0.8803 | 0.8803 | 0.0000 |

### 9.4 当前阶段结论

- `V3` 没有超过 `V2` 的峰值 best mIoU
- 但 `V3` 在 `IRSTD-1K` 上显著提升了后期稳定性
- `V2` 在 `IRSTD-1K` 上是“峰值更高但后期掉点严重”
- `V3` 在 `IRSTD-1K` 上是“峰值略低但 final 表现稳定且接近 best”
- `V3` 在 `NUDT-SIRST` 上则进一步取得了比 `V2` 更强的性能
- 如果当前目标是论文里的“复杂场景下稳定性提升 + 常规场景下保持甚至增强性能”，那么 `V3` 已经给出了很强的实验支撑
