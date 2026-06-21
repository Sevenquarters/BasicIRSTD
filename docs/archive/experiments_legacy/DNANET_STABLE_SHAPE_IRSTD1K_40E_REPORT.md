# DNANet-LDEM-Gate-Stable-Shape 在 IRSTD-1K 上的 40 Epoch 结果复盘

## 1. 本次实验

- 模型：`DNANet-LDEM-Gate-Stable-Shape`
- 数据集：`IRSTD-1K`
- 训练策略：`10 -> 20 -> 30 -> 40` 分段续跑
- 训练参数：
  - `batchSize = 8`
  - `patchSize = 256`
  - `eval_intervals = 1`
  - `save_intervals = 10`

日志与指标文件：

- [metrics.csv](D:\Program Files (x86)\IRSTD\BasicIRSTD\log\exp_dnanet_ldem_gate_stable_shape_irstd1k_10e_retry\IRSTD-1K_DNANet-LDEM-Gate-Stable-Shape_metrics.csv)
- [best checkpoint](D:\Program Files (x86)\IRSTD\BasicIRSTD\log\exp_dnanet_ldem_gate_stable_shape_irstd1k_10e_retry\IRSTD-1K\DNANet-LDEM-Gate-Stable-Shape_best.pth.tar)
- [epoch 40 checkpoint](D:\Program Files (x86)\IRSTD\BasicIRSTD\log\exp_dnanet_ldem_gate_stable_shape_irstd1k_10e_retry\IRSTD-1K\DNANet-LDEM-Gate-Stable-Shape_40.pth.tar)


## 2. Shape 版关键结果

### 2.1 Best

- Best Epoch：`23`
- Best Loss：`0.3763`
- Best mIoU：`0.6420`
- Best PD：`0.9192`
- Best FA：`3.9646e-05`

### 2.2 Final

- Final Epoch：`40`
- Final Loss：`0.3592`
- Final mIoU：`0.6192`
- Final PD：`0.9226`
- Final FA：`5.9232e-05`

### 2.3 稳定性

- Stability Gap = `Best mIoU - Final mIoU = 0.0227`


## 3. 与其他 DNANet 系列对比

> 说明：`DNANet-LDEM` 当前仍沿用历史实验中已有的最新记录，最新点为 `epoch 37`。

| 模型 | Best Epoch | Best mIoU | Final / Latest mIoU | Stability Gap | 结论 |
|---|---:|---:|---:|---:|---|
| DNANet | 28 | 0.6446 | 0.6282 | 0.0163 | 原始 baseline，整体较稳 |
| DNANet-LDEM | 35 | 0.6499 | 0.6373 | 0.0125 | 峰值与最终值都不错 |
| DNANet-LDEM-Gate | 26 | 0.6522 | 0.5232 | 0.1290 | 峰值最高，但后期崩塌明显 |
| DNANet-LDEM-Gate-Stable | 31 | 0.6387 | 0.6331 | 0.0055 | 稳定性最佳 |
| DNANet-LDEM-Gate-Stable-Shape | 23 | 0.6420 | 0.6192 | 0.0227 | 比 Gate 稳很多，但未超过 Stable |


## 4. 这版 Shape 模块带来了什么

### 4.1 正向效果

- 相比 `DNANet-LDEM-Gate`：
  - 峰值略低，但最终结果大幅更稳
  - 说明 shape 约束确实对“后期崩塌”有抑制作用

- 相比 `DNANet-LDEM-Gate-Stable`：
  - Best mIoU 略高一点：`0.6420 > 0.6387`
  - 说明 shape 分支在某些 epoch 的确能强化复杂背景下的结构判别

### 4.2 不足

- Final mIoU 低于 `Stable`：
  - `0.6192 < 0.6331`
- Stability gap 也明显大于 `Stable`：
  - `0.0227 > 0.0055`

这说明当前这版 `ShapePriorRefinementUnit` 虽然能提升一部分峰值判别能力，但：

- 对后期训练的稳定保持不如 `Stable` 纯稳定版
- shape mask 仍可能在部分 epoch 对真实目标响应造成过筛选
- 或者对高层语义支持不足的位置产生了额外抑制


## 5. 当前最合理的科研判断

### 5.1 结论

`DNANet-LDEM-Gate-Stable-Shape` 目前**不是最终优于 `DNANet-LDEM-Gate-Stable` 的版本**。

更准确地说，它呈现的是：

- **比 Gate 更稳**
- **比 Stable 更激进**
- **能带来一点峰值收益，但破坏了 Stable 的后期保持能力**

### 5.2 这意味着什么

这不是“Shape 方向没用”，而是：

- **当前 shape 约束插入方式还偏硬**
- **shape mask 的作用强度或耦合方式还需要继续调**


## 6. 下一步最推荐改法

### 6.1 优先方向

建议不要放弃 shape，而是做一版更保守的：

- `DNANet-LDEM-Gate-Stable-ShapeLite`

### 6.2 具体改法

1. 降低 shape 残差强度
- 把 `residual_scale` 从当前较激进的设置进一步减小
- 目标：减少后期对主干特征的扰动

2. 把 shape mask 从“直接 refinement”改成“辅助权重”
- 不再让 shape 分支单独决定修正强度
- 改成只参与 support mask 的一部分

3. 对 shape 分支做残差旁路保护
- 避免在真实点状弱目标上过度抑制

4. 先做 `IRSTD-1K 10 epoch`
- 重点看 `epoch 15-30` 区间是否比当前 Shape 版更稳


## 7. 现阶段结论一句话

- `Stable` 目前仍是 IRSTD-1K 上最均衡、最可靠的版本；
- `Stable-Shape` 证明了 shape prior 有潜力，但当前实现还没有把这份潜力稳定兑现成最终收益。
