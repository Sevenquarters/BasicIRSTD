# GCA 实验报告（ResUNet-GCA）

## 1. 实验基本信息

- **实验时间**：2026-05-14 18:16（本地时间）
- **实验阶段**：GCA 模块接入后的结构与可运行性验证
- **实验环境**：
  - 项目路径：`D:\Program Files (x86)\IRSTD\BasicIRSTD`
  - Python 环境：`venv`
  - 数据集：`NUDT-SIRST`

---

## 2. 实验目的

1. 验证 `ResUNet-GCA` 是否可正常创建、前向推理和批量推理。  
2. 验证 `ResUNet-GCA` 是否已正确注册到 `Net` 框架入口。  
3. 对比 `ResUNet` 与 `ResUNet-GCA` 的参数量和 CPU 推理速度。  

---

## 3. 实验内容与执行命令

### 3.1 功能验证

```powershell
python test_resunet_gca.py
```

### 3.2 结构与性能分析

```powershell
python analyze_models_gca.py --all
```

---

## 4. 实验结果记录

### 4.1 模型功能验证结果（`test_resunet_gca.py`）

- 模型创建：✅ 成功
- 前向传播：✅ 成功（输入/输出均为 `torch.Size([1, 1, 256, 256])`）
- 批量推理：✅ 成功（输入/输出均为 `torch.Size([4, 1, 256, 256])`）
- Net 注册验证：✅ 成功（`Net('ResUNet-GCA', 'train')` 可正常调用）

### 4.2 参数规模对比（`analyze_models_gca.py --all`）

| 模型 | 参数数量 | 参数量(M) | 相对增长 |
|---|---:|---:|---:|
| ResUNet | 904,785 | 0.90 | 基线 |
| ResUNet-GCA | 958,887 | 0.96 | +6.0% |

- **GCA 模块参数增量**：`54,102`

### 4.3 GCA 连接位置检查

- `gca_2_0`：作用于 `x2_0 (H/4, W/4)`，再与 `up(x3_0)` 融合
- `gca_1_0`：作用于 `x1_0 (H/2, W/2)`，再与 `up(x2_1)` 融合
- `gca_0_0`：作用于 `x0_0 (H, W)`，再与 `up(x1_2)` 融合

> 结论：连接方式符合“3 条 skip connection 融合前插入 GCA（pre-fusion）”设计。

### 4.4 CPU 推理速度对比

| 模型 | 平均推理时间(ms) | 吞吐量(img/s) |
|---|---:|---:|
| ResUNet | 114.85 | 34.8 |
| ResUNet-GCA | 165.00 | 24.2 |

---

## 5. 阶段性结论

1. `ResUNet-GCA` 已完成工程接入，功能正确，可进入训练阶段。  
2. 与基线相比，参数量增加较小（+6.0%），属于可控开销。  
3. 在 CPU 上推理速度下降明显（约慢 43%），后续需结合精度收益评估性价比。  

---

## 6. 下一步实验计划

1. 运行正式训练对比实验（同数据集、同训练配置）：

```powershell
python train.py --model_names ResUNet --dataset_names NUDT-SIRST --batchSize 8 --nEpochs 20
python train.py --model_names ResUNet-GCA --dataset_names NUDT-SIRST --batchSize 8 --nEpochs 20
```

2. 记录并对比指标：
   - `mIoU`
   - `PD`
   - `FA`

3. 形成结论：
   - 若 `PD` 提升且 `FA` 下降，说明 GCA 在红外小目标检测上有效；
   - 若精度提升不明显，考虑调节 GCA 放置层或门控结构参数。  

---

## 7. 备注

- 本文档用于持续追加 GCA 实验记录。  
- 建议每次实验追加“时间-命令-结果-结论”四段内容，便于论文复现实验链路。  

---

## 8. 正式训练对比实验（2026-05-14）

### 8.1 实验命令

```powershell
python train.py --model_names ResUNet --dataset_names NUDT-SIRST --batchSize 8 --nEpochs 20
python train.py --model_names ResUNet-GCA --dataset_names NUDT-SIRST --batchSize 8 --nEpochs 20
```

### 8.2 训练日志摘要

- **ResUNet**
  - Epoch 10 loss: `0.655523`
  - Epoch 20 loss: `0.292588`
- **ResUNet-GCA**
  - Epoch 10 loss: `0.685163`
  - Epoch 20 loss: `0.280577`

### 8.3 最终指标对比

| 模型 | pixAcc | mIoU | PD | FA |
|---|---:|---:|---:|---:|
| ResUNet | 0.815292 | 0.735121 | 0.947090 | 3.623962e-05 |
| ResUNet-GCA | 0.859933 | 0.751698 | 0.954497 | 2.780593e-05 |

### 8.4 相对变化（ResUNet-GCA 相对基线）

- pixAcc：`+0.044641`（约 +5.48%）
- mIoU：`+0.016577`（约 +2.25%）
- PD：`+0.007407`（约 +0.78%）
- FA：`-8.433699e-06`（约 -23.27%，更低更好）

### 8.5 结论

在 NUDT-SIRST 数据集、相同训练配置（batchSize=8, nEpochs=20）下，`ResUNet-GCA` 相比 `ResUNet`：

1. **精度提升**：mIoU、PD、pixAcc 全部提升；
2. **误报下降**：FA 进一步降低；
3. **总体有效**：当前 GCA 方案在该任务上表现为正向增益，可作为后续模块迭代的基线版本。

---

## 9. 40-epoch 曲线对比（2026-05-17）

### 9.1 最终 Epoch 指标（Epoch 40）

| 模型 | pixAcc | mIoU | PD | FA | loss |
|---|---:|---:|---:|---:|---:|
| ResUNet | 0.843581 | 0.772510 | 0.935450 | 1.162793e-05 | 0.220761 |
| ResUNet-GCA | 0.858450 | 0.783869 | 0.952381 | 1.240926e-05 | 0.226620 |

### 9.2 最佳 mIoU（Best Epoch）

| 模型 | 最佳 mIoU | Epoch |
|---|---:|---:|
| ResUNet | **0.799266** | 32 |
| ResUNet-GCA | 0.790850 | 34 |

### 9.3 结论（40 epoch）

1. **最终收敛点**：GCA 在 mIoU/PD/pixAcc 上优于基线，但 FA 与 loss 略高。  
2. **最佳点**：基线的最佳 mIoU 略优于 GCA。  
3. **结论**：若以**最终收敛**为准，GCA 有优势；若以**最佳点**为准，基线略强。  

---

## 10. 消融实验：去门控（ResUNet-GCA-NoGate，2026-05-26）

### 10.1 实验命令

```powershell
python train.py --model_names ResUNet-GCA-NoGate --dataset_names NUDT-SIRST --batchSize 8 --nEpochs 20 --intervals 1 --eval_intervals 1
```

### 10.2 最终 Epoch 指标（Epoch 20）

| 模型 | pixAcc | mIoU | PD | FA | loss |
|---|---:|---:|---:|---:|---:|
| ResUNet-GCA-NoGate | 0.836129 | 0.707942 | 0.907937 | 3.237896e-05 | 0.278698 |

### 10.3 最佳 mIoU

| 模型 | 最佳 mIoU | Epoch |
|---|---:|---:|
| ResUNet-GCA-NoGate | 0.737475 | 17 |

### 10.4 消融结论

1. 去掉门控后，**整体性能明显低于有门控的 GCA**（mIoU 与 PD 均下降，FA 上升）。  
2. 说明**动态门控分支是 GCA 的关键有效组件**，不能简单删除。  
3. 推荐保留门控结构作为正式模型版本。  

---

## 11. 第二轮实验设计：IRSTD-1K（低信噪比场景）

### 11.1 实验动机

NUDT-SIRST 的信噪比较高，基线模型已能达到较好性能。为了验证 GCA 在低信噪比条件下的优势，采用 **IRSTD-1K** 作为第二轮实验数据集。

### 11.2 实验设置（保持公平对比）

- **模型**：ResUNet / ResUNet-GCA / ResUNet-GCA-NoGate  
- **数据集**：IRSTD-1K  
- **batchSize**：8  
- **nEpochs**：先跑 20（如仍有提升空间，再续训练到 40）  
- **eval_intervals**：1（记录每个 epoch 指标）  
- **随机种子**：42（默认）  

### 11.3 实验命令

```powershell
# 1) 基线
python train.py --model_names ResUNet --dataset_names IRSTD-1K --batchSize 8 --nEpochs 20 --intervals 1 --eval_intervals 1

# 2) 门控 GCA
python train.py --model_names ResUNet-GCA --dataset_names IRSTD-1K --batchSize 8 --nEpochs 20 --intervals 1 --eval_intervals 1

# 3) 无门控消融
python train.py --model_names ResUNet-GCA-NoGate --dataset_names IRSTD-1K --batchSize 8 --nEpochs 20 --intervals 1 --eval_intervals 1
```

### 11.4 产出文件（用于曲线与报告）

- 指标 CSV：
  - `.\log\IRSTD-1K_ResUNet_metrics.csv`
  - `.\log\IRSTD-1K_ResUNet-GCA_metrics.csv`
  - `.\log\IRSTD-1K_ResUNet-GCA-NoGate_metrics.csv`
- 最佳模型：
  - `.\log\IRSTD-1K\ResUNet_best.pth.tar`
  - `.\log\IRSTD-1K\ResUNet-GCA_best.pth.tar`
  - `.\log\IRSTD-1K\ResUNet-GCA-NoGate_best.pth.tar`

### 11.5 对比指标与结论模板

请记录以下指标：
- pixAcc / mIoU / PD / FA
- 最佳 mIoU 及对应 Epoch
- 最终 Epoch 指标（Epoch 20）

如需继续：

```powershell
# 续训到 40（可选）
python train.py --model_names ResUNet --dataset_names IRSTD-1K --resume .\log\IRSTD-1K\ResUNet_20.pth.tar --nEpochs 40 --batchSize 8 --intervals 1 --eval_intervals 1
python train.py --model_names ResUNet-GCA --dataset_names IRSTD-1K --resume .\log\IRSTD-1K\ResUNet-GCA_20.pth.tar --nEpochs 40 --batchSize 8 --intervals 1 --eval_intervals 1
python train.py --model_names ResUNet-GCA-NoGate --dataset_names IRSTD-1K --resume .\log\IRSTD-1K\ResUNet-GCA-NoGate_20.pth.tar --nEpochs 40 --batchSize 8 --intervals 1 --eval_intervals 1
```

预期结论方向：
1. 若 **GCA 在 IRSTD-1K 上提升更明显**，说明其在低 SNR 场景下有效。  
2. 若 **NoGate 明显退化**，进一步证实门控分支的重要性。  

---

## 12. NUAA-SIRST 对比实验（2026-05-27）

### 12.1 实验命令

```powershell
# 20 epoch
python train.py --model_names ResUNet --dataset_names NUAA-SIRST --batchSize 8 --nEpochs 20 --intervals 1 --eval_intervals 1
python train.py --model_names ResUNet-GCA --dataset_names NUAA-SIRST --batchSize 8 --nEpochs 20 --intervals 1 --eval_intervals 1
python train.py --model_names ResUNet-GCA-NoGate --dataset_names NUAA-SIRST --batchSize 8 --nEpochs 20 --intervals 1 --eval_intervals 1
python train.py --model_names ResUNet-GCA-DSPG --dataset_names NUAA-SIRST --batchSize 8 --nEpochs 20 --intervals 1 --eval_intervals 1

# 续训到 40 epoch
python train.py --model_names ResUNet --dataset_names NUAA-SIRST --resume .\log\NUAA-SIRST\ResUNet_20.pth.tar --nEpochs 40 --batchSize 8 --intervals 1 --eval_intervals 1
python train.py --model_names ResUNet-GCA --dataset_names NUAA-SIRST --resume .\log\NUAA-SIRST\ResUNet-GCA_20.pth.tar --nEpochs 40 --batchSize 8 --intervals 1 --eval_intervals 1
python train.py --model_names ResUNet-GCA-NoGate --dataset_names NUAA-SIRST --resume .\log\NUAA-SIRST\ResUNet-GCA-NoGate_20.pth.tar --nEpochs 40 --batchSize 8 --intervals 1 --eval_intervals 1
python train.py --model_names ResUNet-GCA-DSPG --dataset_names NUAA-SIRST --resume .\log\NUAA-SIRST\ResUNet-GCA-DSPG_20.pth.tar --nEpochs 40 --batchSize 8 --intervals 1 --eval_intervals 1
```

### 12.2 最终 Epoch 指标（Epoch 40）

| 模型 | pixAcc | mIoU | PD | FA | loss |
|---|---:|---:|---:|---:|---:|
| ResUNet | 0.773071 | 0.660516 | 0.916350 | 7.841078e-05 | 0.269085 |
| ResUNet-GCA | 0.824404 | 0.681428 | 0.950570 | 7.539234e-05 | 0.263678 |
| ResUNet-GCA-NoGate | 0.791362 | 0.662714 | 0.935361 | 7.573534e-05 | 0.311156 |
| ResUNet-GCA-DSPG | 0.847888 | 0.687494 | 0.950570 | 7.285411e-05 | 0.262994 |

### 12.3 最佳 mIoU（Best Epoch）

| 模型 | 最佳 mIoU | Epoch | PD | FA |
|---|---:|---:|---:|---:|
| ResUNet | 0.694981 | 31 | 0.908745 | 5.597830e-05 |
| ResUNet-GCA | 0.690303 | 24 | 0.954373 | 5.940834e-05 |
| ResUNet-GCA-NoGate | 0.675566 | 39 | 0.923954 | 5.721311e-05 |
| ResUNet-GCA-DSPG | **0.699268** | 30 | 0.927757 | **3.155639e-05** |

### 12.4 结论（面向论文叙事）

1. **DSPG 在 NUAA-SIRST 上取得最高 mIoU（0.699268）且 FA 最低（3.16e-05）**。与基线 ResUNet 相比：mIoU **+0.00429**、PD **+0.0190**，FA **下降约 43.6%**。  
2. 在国防与预警类红外检测中，**误报（FA）直接影响告警可信度**，DSPG 的“先验流抑噪”机制提供了明确的工程价值：**在精度略升的同时显著降低误报**。  
3. 该优势目前集中体现在 NUAA-SIRST；其他数据集仍需进一步验证其泛化稳定性。  
