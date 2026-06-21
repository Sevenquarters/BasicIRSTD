# Baseline 对比与 V1 改进方案

## 1. 目的

- 汇总当前已经完成的 baseline 实验结果
- 给出第一轮可执行的模块改进方案
- 为后续小论文材料积累提供统一入口

---

## 2. 当前已完成 baseline

### 2.1 实验配置口径

- 训练轮数：`40 epoch`
- 每个 epoch 记录：
  - `loss`
  - `mIoU`
  - `PD`
  - `FA`
- 评估频率：`每个 epoch`
- checkpoint 保存：
  - `20 epoch`
  - `40 epoch`
  - `best`

### 2.2 已完成实验

| 模型 | 数据集 | 日志目录 | 指标文件 |
|---|---|---|---|
| RDIAN | NUDT-SIRST | `log/exp_baseline_ridian_nudt_40e/` | `NUDT-SIRST_RDIAN_metrics.csv` |
| DNANet | NUDT-SIRST | `log/exp_baseline_dnanet_nudt_40e/` | `NUDT-SIRST_DNANet_metrics.csv` |
| DNANet | IRSTD-1K | `log/exp_baseline_dnanet_irstd1k_40e/` | `IRSTD-1K_DNANet_metrics.csv` |
| DNANet-LDEM | NUDT-SIRST | `log/exp_dnanet_ldem_nudt_40e/` | `NUDT-SIRST_DNANet-LDEM_metrics.csv` |
| DNANet-LDEM | IRSTD-1K | `log/exp_dnanet_ldem_irstd1k_40e/` | `IRSTD-1K_DNANet-LDEM_metrics.csv` |

---

## 3. 关键结果对比表

### 3.1 最优结果对比

| 模型 | 数据集 | 最优 Epoch | Best Loss | Best mIoU | Best PD | Best FA |
|---|---|---:|---:|---:|---:|---:|
| RDIAN | NUDT-SIRST | 39 | 0.2886 | 0.7386 | 0.9323 | 2.8633e-05 |
| DNANet | NUDT-SIRST | 32 | 0.2135 | 0.8118 | 0.9619 | 1.5948e-05 |
| DNANet | IRSTD-1K | 28 | 0.3925 | 0.6446 | 0.9259 | 2.3021e-05 |
| DNANet-LDEM | NUDT-SIRST | 39 | 0.1722 | 0.8409 | 0.9757 | 1.5787e-05 |
| DNANet-LDEM | IRSTD-1K | 35 | 0.3689 | 0.6499 | 0.8620 | 3.3307e-05 |

### 3.2 40 epoch 最终结果对比

| 模型 | 数据集 | Final Loss | Final mIoU | Final PD | Final FA |
|---|---|---:|---:|---:|---:|
| RDIAN | NUDT-SIRST | 0.2886 | 0.7250 | 0.9397 | 4.4191e-05 |
| DNANet | NUDT-SIRST | 0.2037 | 0.8114 | 0.9831 | 2.0521e-05 |
| DNANet | IRSTD-1K | 0.3579 | 0.6282 | 0.9091 | 3.9798e-05 |
| DNANet-LDEM | NUDT-SIRST | 0.1796 | 0.8218 | 0.9810 | 1.2042e-05 |
| DNANet-LDEM | IRSTD-1K | 0.3685 | 0.6373 | 0.8956 | 2.1920e-05 |

> 注：`DNANet-LDEM + IRSTD-1K` 当前仅记录到 `epoch 37`，上表“Final”对应当前最新结果，不是完整 40 epoch 收尾结果。

### 3.3 `DNANet` vs `DNANet-LDEM` 直接对比

| 数据集 | 指标口径 | DNANet | DNANet-LDEM | 变化 |
|---|---|---:|---:|---:|
| NUDT-SIRST | Best mIoU | 0.8118 | 0.8409 | `+0.0291` |
| NUDT-SIRST | Best PD | 0.9619 | 0.9757 | `+0.0138` |
| NUDT-SIRST | Best FA | 1.5948e-05 | 1.5787e-05 | 略优 |
| NUDT-SIRST | Final mIoU | 0.8114 | 0.8218 | `+0.0104` |
| NUDT-SIRST | Final FA | 2.0521e-05 | 1.2042e-05 | 明显下降 |
| IRSTD-1K | Best mIoU | 0.6446 | 0.6499 | `+0.0053` |
| IRSTD-1K | Best PD | 0.9259 | 0.8620 | `-0.0639` |
| IRSTD-1K | Best FA | 2.3021e-05 | 3.3307e-05 | 变差 |
| IRSTD-1K | Latest mIoU | 0.6282 | 0.6373 | `+0.0091` |
| IRSTD-1K | Latest FA | 3.9798e-05 | 2.1920e-05 | 改善 |

---

## 4. baseline 阶段结论

### 4.1 模型选择结论

- `DNANet` 是当前更强的 baseline
- `RDIAN` 是当前更轻量、可作为效率型对照的 baseline

### 4.2 数据集表现结论

- `DNANet` 在 `NUDT-SIRST` 上表现明显强于 `RDIAN`
- `DNANet` 在 `IRSTD-1K` 上相比 `NUDT-SIRST` 有明显性能下降
- `DNANet` 在 `IRSTD-1K` 上后期波动更明显，说明跨数据集稳定性仍有改进空间
- `DNANet-LDEM` 在 `NUDT-SIRST` 上带来了明确且稳定的增益
- `DNANet-LDEM` 在 `IRSTD-1K` 上仅表现出轻微 mIoU 提升，收益不如 `NUDT-SIRST` 明确

### 4.3 对后续研究最有价值的信号

- 当前最值得研究的问题不只是“继续提单数据集精度”
- 更值得研究的是：
  - 如何提升 `DNANet` 在不同数据集上的稳定性
  - 如何降低复杂背景和分布变化带来的性能波动
  - 如何在不大幅增加复杂度的前提下提升泛化能力

### 4.4 为什么浅层细节增强在两个数据集上表现不同

- `NUDT-SIRST` 更像是“浅层细节可直接转化为有效目标线索”的数据集
- 该数据集中，小目标的亮点、局部边缘和弱纹理更容易和真实目标对齐，因此 `LDEM` 对浅层响应的增强可以直接转化为更好的分割判别
- 从结果看，`NUDT-SIRST` 上 `LDEM` 同时提升了 `mIoU` 和 `PD`，并且最终 `FA` 还有下降，说明增强到的浅层信息大部分是有益信号
- `IRSTD-1K` 的背景更复杂、场景分布更杂，浅层纹理里混有更多噪声边缘、杂波结构和伪亮点
- 在这种情况下，单纯“增强浅层细节”容易把有用细节和干扰细节一起放大，因此 `mIoU` 可能略有提升，但 `PD / FA` 的收益不稳定
- 这说明当前 `LDEM` 的主要问题不是“没增强到细节”，而是“缺少选择性抑制”和“缺少跨场景自适应控制”

---

## 5. 第一版模块改进方案（V1）

### 5.1 主 baseline

- 主 baseline：`DNANet`
- 轻量对照：`RDIAN`

### 5.2 方法目标

第一版方法不追求大改结构，而是优先验证：

- 能否提升 `DNANet` 在不同数据集上的稳定性
- 能否减少后期训练波动
- 能否在 `IRSTD-1K` 上获得更稳的 `mIoU / PD / FA`

### 5.3 推荐改进方向

#### 方向 A：浅层细节增强

借鉴来源：

- `HintU`
- `ILNet`

核心思想：

- 红外小目标的亮度差异、边缘和微弱结构往往首先出现在浅层特征
- 在 DNANet 的早期特征层中增加轻量细节保留分支
- 避免高层语义过强时压制弱小目标的早期响应

适合原因：

- 对工程改动相对小
- 容易做消融
- 很适合解释 `IRSTD-1K` 上的稳定性问题

#### 方向 B：轻量方向感知增强

借鉴来源：

- `RDIAN`

核心思想：

- 在 DNANet 中层或 skip 交互前加入轻量方向感知模块
- 提升模型对复杂背景中微弱目标结构的分辨能力

适合原因：

- 可以自然解释为什么引入 `RDIAN` 做对照
- 容易写成“强 baseline + 经典有效模块迁移”的论文故事

### 5.4 V1 推荐组合

第一版先只做这一个组合：

- `DNANet + 浅层细节增强分支`

先不要同时叠加多个模块。原因：

- 便于判断增益来自哪里
- 更适合做第一轮消融
- 更利于后续论文写作

如果第一轮有效，再继续做：

- `DNANet + 浅层细节增强分支 + 轻量方向感知模块`

### 5.5 当前工程中的 V1 实现

当前已经在工程中新增第一版变体：

- 模型名：`DNANet-LDEM`
- 对应模块：`LDEM`
  - 全称：`Low-level Detail Enhancement Module`

代码位置：

- [model_DNANet.py](D:\Program Files (x86)\IRSTD\BasicIRSTD\model\DNANet\model_DNANet.py)
- [__init__.py](D:\Program Files (x86)\IRSTD\BasicIRSTD\model\__init__.py)
- [net.py](D:\Program Files (x86)\IRSTD\BasicIRSTD\net.py)

实现说明：

- `LDEM` 插入在 `x0_0` 后
- 浅层特征先经过：
  - 两层 `3x3 Conv + BN`
  - 轻量通道权重
  - 残差增强
- 然后再把增强后的 `x0_0` 送入后续 `x0_1 / x0_2 / x0_3 / x0_4` 的 nested interaction

这样做的目的：

- 保持原始 `DNANet` 主体不变
- 只增强最浅层局部细节
- 避免直接改动过深层级导致论文故事不清楚
- 更方便做第一轮消融

---

## 5.6 第二版模块改进方案（V2）

### 模型名

- `DNANet-LDEM-Gate`

### 核心思路

- 保留 `LDEM` 对浅层细节的增强能力
- 在浅层增强后的 `x0_0` 后增加一个轻量 `DetailSelectiveGate`
- gate 不再默认“全部增强都有效”，而是引入选择性筛选

### Gate 的三类输入线索

- 浅层增强特征本身
- 局部对比度响应
- 来自下一层编码器特征 `x1_0` 的语义引导

### 设计动机

- `NUDT-SIRST` 的结果说明浅层细节增强本身是有效的
- `IRSTD-1K` 的结果说明纯增强会把目标细节和背景干扰一起放大
- 因此第二版的重点不是“继续增强”，而是“增强后做筛选”

### 当前实现位置

- [model_DNANet.py](D:\Program Files (x86)\IRSTD\BasicIRSTD\model\DNANet\model_DNANet.py)
- [__init__.py](D:\Program Files (x86)\IRSTD\BasicIRSTD\model\__init__.py)
- [net.py](D:\Program Files (x86)\IRSTD\BasicIRSTD\net.py)

### 当前状态

- 已完成模型注册
- 已完成前向验证
- 已完成一步反向传播验证
- 已完成：
  - `DNANet-LDEM-Gate + NUDT-SIRST + 40 epoch`
  - `DNANet-LDEM-Gate + IRSTD-1K + 40 epoch`

### V2 当前结果

| 模型 | 数据集 | 最优 Epoch | Best mIoU | Best PD | Best FA | Final mIoU | Final PD | Final FA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DNANet-LDEM-Gate | NUDT-SIRST | 38 | 0.8397 | 0.9725 | 1.0456e-05 | 0.8395 | 0.9884 | 2.3003e-05 |
| DNANet-LDEM-Gate | IRSTD-1K | 26 | 0.6522 | 0.9293 | 3.3554e-05 | 0.5232 | 0.7677 | 3.9855e-06 |

### 当前阶段性观察

- `NUDT-SIRST` 上，`DNANet-LDEM-Gate` 明显优于原始 `DNANet`，并基本追平 `DNANet-LDEM`
- `IRSTD-1K` 上，`DNANet-LDEM-Gate` 的 best mIoU 已超过 `DNANet` 和已完成的 `DNANet-LDEM`
- 但 `IRSTD-1K` 上后期波动仍然较大，说明 gate 的确提升了峰值性能，但训练稳定性和后期保持能力还需要继续优化

---

## 6. 推荐实验顺序

### 第一轮

1. 保留当前 baseline 结果
2. 在 `DNANet` 上使用 `DNANet-LDEM`
3. 先跑：
   - `NUDT-SIRST`
   - `IRSTD-1K`
4. 仍然使用 `40 epoch` 快速验证

### 第二轮

若第一轮有效，再加入方向感知模块：

1. `DNANet + Detail`
2. `DNANet + Direction`
3. `DNANet + Detail + Direction`

形成标准消融表。

### 第二版建议方向

第二版不建议简单继续堆更强的细节分支，而建议做：

1. `DNANet + LDEM + 轻量选择性门控`
2. `DNANet + LDEM + 方向感知抑噪`
3. `DNANet + LDEM + 细节/语义双分支重标定`

推荐优先级最高的是第 1 个方向。

原因：

- 第一版已经证明“浅层细节有价值”
- 当前真正缺的不是更多细节，而是判断“哪些细节该保留、哪些细节该压制”
- 这样最容易解释为什么在 `NUDT-SIRST` 有效、在 `IRSTD-1K` 提升有限

建议的 V2 具体设计为：

- 保留当前 `LDEM`，不要删除
- 在 `LDEM` 输出后增加一个轻量 gate
- gate 的输入同时参考：
  - 浅层响应强度
  - 局部对比度
  - 来自下一层语义特征的引导信息
- 输出为逐通道或逐像素权重，用于抑制背景纹理过强的位置，再送入后续 nested block

这类 V2 更适合写成论文中的方法动机：

- V1：证明浅层细节对红外小目标是有效的
- V2：进一步解决复杂场景下“细节增强伴随噪声放大”的问题
- 这样可以自然形成“增强 - 失稳 - 抑噪校正”的完整科研故事

---

## 7. 下一步建议

当前最推荐的下一步不是继续补更多 baseline，而是：

1. 以 `DNANet` 为主 baseline 开始做第一版改进
2. 先训练已经实现好的 `DNANet-LDEM`
3. 继续保持当前这种每个 epoch 记录 `loss / mIoU / PD / FA` 的训练方式

### 7.1 推荐立即执行的实验

#### NUDT-SIRST

```powershell
.\venv\Scripts\python train.py --model_names DNANet-LDEM --dataset_names NUDT-SIRST --batchSize 8 --nEpochs 40 --intervals 1 --eval_intervals 1 --save_intervals 20 --save ./log/exp_dnanet_ldem_nudt_40e
```

#### IRSTD-1K

```powershell
.\venv\Scripts\python train.py --model_names DNANet-LDEM --dataset_names IRSTD-1K --batchSize 8 --nEpochs 40 --intervals 1 --eval_intervals 1 --save_intervals 20 --save ./log/exp_dnanet_ldem_irstd1k_40e
```

---

## 8. 相关文件入口

- [RDIAN NUDT metrics](D:\Program Files (x86)\IRSTD\BasicIRSTD\log\exp_baseline_ridian_nudt_40e\NUDT-SIRST_RDIAN_metrics.csv)
- [DNANet NUDT metrics](D:\Program Files (x86)\IRSTD\BasicIRSTD\log\exp_baseline_dnanet_nudt_40e\NUDT-SIRST_DNANet_metrics.csv)
- [DNANet IRSTD-1K metrics](D:\Program Files (x86)\IRSTD\BasicIRSTD\log\exp_baseline_dnanet_irstd1k_40e\IRSTD-1K_DNANet_metrics.csv)
- [DNANet-LDEM NUDT metrics](D:\Program Files (x86)\IRSTD\BasicIRSTD\log\exp_dnanet_ldem_nudt_40e\NUDT-SIRST_DNANet-LDEM_metrics.csv)
- [DNANet-LDEM IRSTD-1K metrics](D:\Program Files (x86)\IRSTD\BasicIRSTD\log\exp_dnanet_ldem_irstd1k_40e\IRSTD-1K_DNANet-LDEM_metrics.csv)
- `DNANet-LDEM-Gate` 日志目录：
  - `log/exp_dnanet_ldem_gate_nudt_40e/`
  - `log/exp_dnanet_ldem_gate_irstd1k_40e/`
