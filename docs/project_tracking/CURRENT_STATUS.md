# 当前项目状态

## 1. 当前主线

当前红外小目标检测主线围绕 `DNANet` 系列递进改进展开：

- `DNANet`
- `DNANet-LDEM`
- `DNANet-LDEM-Gate`
- `DNANet-LDEM-Gate-Stable`
- `DNANet-LDEM-Gate-Stable-Context`
- `DNANet-LDEM-Gate-Stable-ContextGate`

当前最重要的研究判断是：

- `Stable` 是目前最稳的主线基础版本
- `Context` 是目前最强的复杂背景特化分支
- `ContextGate` 证明了“上下文需要自适应启用”这一判断是对的
- `AuxShapeLoss` 是一个有研究价值但更适合作为辅线消融的方向


## 2. 已完成实验

### 2.1 基础递进主线

已完成双数据集或单数据集验证的模型包括：

- `DNANet`
- `DNANet-LDEM`
- `DNANet-LDEM-Gate`
- `DNANet-LDEM-Gate-Stable`

### 2.2 Stable 后续分支

已完成或已验证可运行的后续分支包括：

- `DNANet-LDEM-Gate-Stable-Shape`
- `DNANet-LDEM-Gate-Stable-ShapeLite`
- `DNANet-LDEM-Gate-Stable-AuxShapeLoss`
- `DNANet-LDEM-Gate-Stable-Context`
- `DNANet-LDEM-Gate-Stable-ContextGate`


## 3. 当前最关键结果

### 3.1 IRSTD-1K

当前最关键的三条主线结果：

| 模型 | Best Epoch | Best mIoU | Final mIoU | Best PD | Final PD | Best FA | Final FA |
|---|---:|---:|---:|---:|---:|---:|---:|
| DNANet-LDEM-Gate-Stable | 31 | 0.6387 | 0.6331 | 0.8990 | 0.8889 | 2.6266e-05 | 1.9833e-05 |
| DNANet-LDEM-Gate-Stable-Context | 36 | 0.6475 | 0.6371 | 0.9125 | 0.9293 | 4.7200e-05 | 4.4562e-05 |
| DNANet-LDEM-Gate-Stable-ContextGate | 30 | 0.6458 | 0.6126 | 0.8788 | 0.8889 | 2.2547e-05 | 2.2300e-05 |

当前结论：

- `Context` 在 `IRSTD-1K` 上已经超过 `Stable` 的 best 和 final mIoU
- 这说明“中层上下文建模”方向是有效的
- `ContextGate` 也能抬升峰值，但 final 回落更明显
- 当前 `IRSTD-1K` 上最优复杂背景分支仍然是 `Context`

### 3.2 NUDT-SIRST

当前已完成主线结果：

| 模型 | Best Epoch | Best mIoU | Final mIoU | Best PD | Final PD | Best FA | Final FA |
|---|---:|---:|---:|---:|---:|---:|---:|
| DNANet-LDEM-Gate-Stable | 40 | 0.8803 | 0.8803 | 0.9757 | 0.9757 | 6.6872e-06 | 6.6872e-06 |
| DNANet-LDEM-Gate-Stable-Context | 39 | 0.8226 | 0.7843 | 0.9693 | 0.9841 | 6.8021e-06 | 1.6201e-05 |
| DNANet-LDEM-Gate-Stable-ContextGate | 38 | 0.8554 | 0.8263 | 0.9852 | 0.9545 | 1.0686e-05 | 1.4983e-05 |

补充说明：

- `Context` 在 `NUDT-SIRST` 上明显不如 `Stable`
- `ContextGate` 比 `Context` 明显回稳，说明“条件式上下文”方向成立
- 但 `ContextGate` 仍未超过 `Stable`
- 当前已形成 `Stable vs Context` 双数据集完整对照
- 下一步不再是“是否能跑通”，而是“如何让上下文分支既保住 IRSTD-1K 收益，又少伤 NUDT-SIRST”


## 4. 当前文档入口

### 4.1 设计文档

- `Context` 设计说明：
  - [DNANET_LDEM_GATE_STABLE_CONTEXT_DESIGN.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\DNANET_LDEM_GATE_STABLE_CONTEXT_DESIGN.md)
- `第五版` 方向说明：
  - [DNANET_V5_DIRECTION.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\DNANET_V5_DIRECTION.md)

### 4.2 主线对比文档

- 主线稳定分支最终对比：
  - [DNANET_STABLE_BRANCHES_FINAL_COMPARISON.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\DNANET_STABLE_BRANCHES_FINAL_COMPARISON.md)
- 上下文分支阶段总结：
  - [DNANET_CONTEXT_BRANCH_PROGRESS.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\DNANET_CONTEXT_BRANCH_PROGRESS.md)

### 4.3 日志与指标

- `Stable + IRSTD-1K`：
  - `log/exp_dnanet_ldem_gate_stable_irstd1k_40e/`
- `Context + IRSTD-1K`：
  - `log/exp_dnanet_ldem_gate_stable_context_irstd1k_40e/`
- `Stable + NUDT-SIRST`：
  - `log/exp_dnanet_ldem_gate_stable_nudt_40e/`
- `Context + NUDT-SIRST`：
  - `log/exp_dnanet_ldem_gate_stable_context_nudt_40e/`
- `ContextGate + IRSTD-1K`：
  - `log/exp_dnanet_ldem_gate_stable_contextgate_irstd1k_40e/`
- `ContextGate + NUDT-SIRST`：
  - `log/exp_dnanet_ldem_gate_stable_contextgate_nudt_40e/`


## 5. 当前最推荐的下一步

1. 整理 `Stable vs Context vs ContextGate` 的双数据集完整对照表
2. 明确第五版目标：保留 `Context` 的 IRSTD-1K 收益，同时减少对 `NUDT-SIRST` 的干扰
3. 第五版优先考虑“更轻、更受控、只在困难区域生效”的上下文路线
4. 再决定是否继续做 `Context + supervision` 的组合路线
