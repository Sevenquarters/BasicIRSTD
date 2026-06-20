# 文献比较表

> 说明：本表聚焦你要求的结构化对比字段：`Backbone / Detection Head / Fusion Strategy / Attention Mechanism / Dataset / Metrics / FLOPs / Parameters`。  
> 状态：IRSTD 主线已补得较完整；多模态与融合方向仍在继续补 FLOPs / Params。

---

## 1. IRSTD 方法比较表

| 方法 | 类别 | Backbone | Detection Head | Fusion Strategy | Attention Mechanism / 关键模块 | 主要数据集 | 主要指标 | FLOPs | Params |
|---|---|---|---|---|---|---|---|---|---|
| ALCNet | Traditional / Hybrid | CNN | Segmentation-style output | 无 | Local Contrast + Attentional Modulation | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 0.378G† | 0.427M† |
| Nonconvex Tensor Low-Rank Approximation | Traditional / Model-driven | Tensor low-rank model | Saliency / segmentation-style output | 无 | Non-convex low-rank approximation | 常见复杂背景红外数据 | Pd / Fa / BSF 等 | 未报告 / 待核 | 未报告 / 待核 |
| Schatten Capped p-Norm-Based Non-Convex Tensor Low-Rank Approximation | Traditional / Model-driven | Tensor low-rank model | Saliency / segmentation-style output | 无 | Schatten capped p-norm low-rank approximation | 复杂背景红外场景 | Pd / Fa / BSF 等 | 未报告 / 待核 | 未报告 / 待核 |
| Nonconvex Tensor Fibered Rank Approximation | Traditional / Model-driven | Tensor low-rank model | Saliency / segmentation-style output | 无 | Fibered rank approximation | 复杂背景红外场景 | Pd / Fa / BSF 等 | 未报告 / 待核 | 未报告 / 待核 |
| DNANet | CNN-based IRSTD | U-Net-like CNN | Segmentation-style output | Dense nested interaction | DNIM + CSAM | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 14.261G† | 4.697M† |
| UIU-Net | CNN-based IRSTD | U-Net in U-Net | Segmentation-style output | Encoder-decoder interaction | RM-DS + IC-A | SIRST / Synthetic / ATR / BasicIRSTD benchmark | IoU / Pd / Fa | 54.426G† | 50.540M† |
| ISNet | CNN-based IRSTD | CNN | Segmentation-style output | 无 | Shape-aware modeling | IRSTD-1K / BasicIRSTD benchmark | IoU / Pd / Fa | 30.618G† | 0.966M† |
| RDIAN | CNN-based IRSTD | CNN | Segmentation-style output | 无 | Receptive-field + Direction Attention | IRDST / BasicIRSTD benchmark | IoU / Pd / Fa | 3.718G† | 0.217M† |
| ISTDU-Net | CNN-based IRSTD | U-Net variant | Segmentation-style output | 无 | U-Net adaptation | 常见 IRSTD benchmark | IoU / Pd / Fa | 7.944G† | 2.752M† |
| SCTransNet | Transformer-based IRSTD | U-shaped + Transformer | Segmentation-style output | Cross-encoder global interaction | SCTB / SSCA / CFN | NUDT-SIRST / NUAA-SIRST / IRSTD-1K | IoU / Pd / Fa | 未报告 / 待核 | 未报告 / 待核 |
| RPCANet | New-paradigm IRSTD | Deep Unfolding | Segmentation-style output | 无 | RPCA unfolding | 常见 IRSTD benchmark | IoU / Pd / Fa | 未报告 / 待核 | 未报告 / 待核 |
| HintU | CNN-based IRSTD | U-Net family | Segmentation-style output | Local feature enhancement | Underappreciated local features | NUDT-SIRST / SIRST-v2 / IRSTD-1K | IoU / Pd / Fa | 未报告 / 待核 | 未报告 / 待核 |

---

## 2. BasicIRSTD Benchmark 数值表

> 该表直接来源于当前工作区 `README.md` 中的统一 benchmark 结果，适合作为“统一复现口径”的引用依据。

| 方法 | SIRST-v1 IoU | SIRST-v1 Pd | SIRST-v1 Fa | NUDT-SIRST IoU | NUDT-SIRST Pd | NUDT-SIRST Fa | IRSTD-1K IoU | IRSTD-1K Pd | IRSTD-1K Fa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALCNet | 61.047 | 87.072 | 55.978 | 61.131 | 97.249 | 29.093 | 58.088 | 92.929 | 74.453 |
| ISNet | 70.491 | 95.057 | 67.983 | 81.236 | 97.778 | 6.343 | 61.852 | 90.236 | 31.561 |
| RDIAN | 70.737 | 95.057 | 48.158 | 82.419 | 98.836 | 14.845 | 59.939 | 87.205 | 33.307 |
| DNANet | 74.815 | 93.536 | 38.279 | 94.192 | 99.259 | 2.436 | 65.735 | 89.562 | 12.336 |
| ISTDU-Net | 75.928 | 96.198 | 38.897 | 91.762 | 98.519 | 3.769 | 65.014 | 93.939 | 26.437 |
| UIU-Net | 77.531 | 92.395 | 9.330 | 90.517 | 98.836 | 8.342 | 65.690 | 91.246 | 13.475 |

---

## 3. 红外-可见融合方法比较表

| 方法 | 类别 | Backbone | Detection Head | Fusion Strategy | Attention Mechanism / 关键模块 | 主要数据集 | 主要指标 | FLOPs | Params |
|---|---|---|---|---|---|---|---|---|---|
| Interactive Compensatory Attention Adversarial Learning | Infrared-Visible Fusion | Fusion CNN + GAN | 无直接检测头 | Image fusion | Interactive compensatory attention + adversarial learning | 常见融合 benchmark（待原文细核） | EN / MI / VIF / SSIM 等 fusion quality metrics | 未报告 / 待核 | 未报告 / 待核 |
| Decomposition-based and Interference Perception (`UMCFuse`, TIP 2025) | Infrared-Visible Fusion | Decomposition-based fusion network | 无直接检测头 | Decomposition-based fusion | Interference perception | 复杂场景融合数据 + 下游任务评测 | Fusion + segmentation / detection / SOD / depth metrics | 未报告 / 待核 | 未报告 / 待核 |
| Beyond Night Visibility | Infrared-Visible Fusion | Multi-scale fusion network | 无直接检测头 | Adaptive multi-scale fusion | Local significant feature extraction + semantic guidance | 夜间融合与检测场景 | Fusion quality + downstream object detection metrics | 未报告 / 待核 | 未报告 / 待核 |
| TSJNet | Task-driven Fusion | Fusion net + task-guided subnetworks | Detection + Segmentation guided | Task-driven fusion | Target-aware + Semantic-aware joint guidance | MSRS / M3FD / RoadScene / LLVIP | mAP / mIoU / fusion quality | 未报告 / 待核 | 未报告 / 待核 |

---

## 4. 多模态小目标检测方法比较表

| 方法 | 类别 | Backbone | Detection Head | Fusion Strategy | Attention Mechanism / 关键模块 | 主要数据集 | 主要指标 | FLOPs | Params |
|---|---|---|---|---|---|---|---|---|---|
| MBNet | RGB-T Detection | Dual-stream CNN | Two-stage pedestrian detection head | RGB-T feature fusion | DMAF + illumination-aware feature alignment | KAIST / CVC-14 | MR / AP | 未报告 / 受限 | 未报告 / 受限 |
| Anchor-free Small-scale Multispectral Pedestrian Detection | RGB-T / Multispectral Detection | Single-stage detector | Anchor-free head | RGB-T fusion | Center-scale estimation | KAIST | MR | 未报告 / 受限 | 未报告 / 受限 |
| DroneVehicle / Uncertainty-Aware Learning | RGB-IR Detection | Cross-modality detector | Detection head | Uncertainty-aware RGB-IR fusion | UAM + illumination-aware NMS | DroneVehicle（56,878 图像） | AP / mAP | 未报告 / 受限 | 未报告 / 受限 |
| ICAFusion | Multispectral Object Detection | Dual-stream detector | Detector-agnostic head | Iterative cross-attention fusion | Dual cross-attention transformer | KAIST / FLIR / VEDAI | mAP | 15.056G* | 23.266M* |

---

## 5. 可直接引用的比较结论

### 5.1 IRSTD 主线
- `RDIAN` 是当前表中**最轻量**的强基线之一：`0.217M / 3.718G`
- `ALCNet` 也非常轻量：`0.427M / 0.378G`，适合作为传统先验 + 深度学习混合基线
- `DNANet` 在统一 benchmark 下对 `NUDT-SIRST` 的表现非常强，是精度型 baseline
- `UIU-Net` 在 `SIRST-v1` 上有突出表现，但复杂度代价最大
- `ISNet` 在 shape-aware 路线中具有较高代表性，而且参数量相对低
- `ISTDU-Net` 是典型的性能与复杂度折中方案

### 5.2 融合与多模态主线
- `MBNet` 代表“模态不平衡”问题建模
- `Anchor-free MSPD` 代表小尺度多光谱检测中的 anchor-free 路线
- `DroneVehicle / UA-CMDet` 代表无人机 RGB-IR 场景下的不确定性感知检测
- `ICAFusion` 代表 cross-attention 跨模态融合方向，且当前已可稳定按 `Pattern Recognition 2023` 引用
- `TSJNet` 代表 task-driven fusion 方向
- 融合方向与检测方向已经表现出明显分化：前者更关注 fused representation 的任务有效性，后者更关注跨模态特征交互与检测头适配
- `Interactive Compensatory` 更偏图像质量驱动融合，`TSJNet` 更偏下游任务驱动融合，二者适合作为融合范式转变的对照组

---

## 6. 当前表格仍待补充的关键空白

1. `SCTransNet / RPCANet / HintU / Non-Convex Tensor Low-Rank Approximation` 的 FLOPs 与 Params
2. 多模态检测方向的 FLOPs 与 Params
3. 红外-可见融合方向更统一的 benchmark 指标口径
4. 传统 IRSTD 方法池中更多非深度方法的结构字段与统一数据集口径

### 6.1 复杂度字段缺失的当前证据说明

- `MBNet`：公开仓库可以稳定核实其为 `ECCV 2020` 工作，并给出训练、测试与模型权重说明，但 README 中未直接给出统一的 `FLOPs / Params` 数字。
- `MBNet`：公开实现为 `tensorflow1.14 + keras2.1` 的旧版双流检测框架；当前环境下复现成本较高，暂未完成本地复杂度复核。
- `ICAFusion`：当前已基于公开实现 `yolov5s_Transfusion_FLIR.yaml` 在本地用 `thop` 完成一轮工程统计，得到约 `15.056G FLOPs / 23.266M Params`（输入尺寸 `640x640`）；这不是作者论文正文直接报告值，而是基于公开实现的复核口径。
- `DroneVehicle / UA-CMDet`：公开项目页与数据集说明已能坐实数据规模与正式 DOI，但未见作者在公开 README 中直接报告统一复杂度数字。
- `DroneVehicle / UA-CMDet`：官方代码仓已核验存在，但其实现基于老版 `mmdetection + mmcv 0.4.3 + 自定义 CUDA ops`，并依赖 `PyTorch 1.1 / CUDA 10.0`；当前环境下直接复核复杂度的工程成本较高。
- `Anchor-free MSPD`：公开 GitHub 仓库当前只提供检测结果文件与论文链接，未提供可直接实例化的模型源码，因此暂无法做本地 `Params / FLOPs` 复核。
- `Interactive Compensatory`：当前已能依据作者 GitHub README 按 `IEEE Transactions on Multimedia 2022` 口径引用，但公开说明中仍未见统一 `FLOPs / Params` 数字。
- `TSJNet / Beyond Night Visibility / UMCFuse`：当前仍未获得稳定可比的复杂度统计；其中 `TSJNet` 已确认存在官方 GitHub 仓库，`UMCFuse` 已在 arXiv 页面标注 `IEEE TIP 2025` 并给出代码入口，但公开材料中仍未见统一 `FLOPs / Params` 数字。

### 6.2 复杂度证据口径

- `作者报告`：
  - 指复杂度数字可直接追溯到论文正文、官方 benchmark 或作者公开材料
- `统一基准复现†`：
  - 指复杂度数字来自当前工作区 `BasicIRSTD README` 的统一 benchmark 结果，不等同于论文作者原文逐篇报告值
- `本地复核*`：
  - 指复杂度数字来自**本地基于公开实现的工程统计**，不是作者论文正文直接报告值
- `未复核 / 受限`：
  - 指当前没有直接获得标准化复杂度数字，或受仓库形态、实现年代、依赖环境限制而暂未完成本地复核

### 6.3 当前已完成的本地复核

- `ICAFusion`
  - 配置文件：`yolov5s_Transfusion_FLIR.yaml`
  - 输入尺寸：`640x640`
  - 统计工具：`thop`
  - 统计时间：`2026-06-18`
  - 结果：`15.056G FLOPs / 23.266M Params`

### 6.4 当前未复核 / 受限的代表情况

- `MBNet`
  - 原因：公开实现基于 `tensorflow1.14 + keras2.1`，当前环境下复核成本高
- `Anchor-free MSPD`
  - 原因：公开 GitHub 仓库只提供检测结果文件，未开放可直接实例化的模型源码
- `DroneVehicle / UA-CMDet`
  - 原因：官方实现基于老版 `mmdetection + mmcv 0.4.3 + 自定义 CUDA ops`，并依赖 `PyTorch 1.1 / CUDA 10.0`

---

## 7. 字段说明

- `†`：
  - 表示该复杂度数字来自当前工作区 `BasicIRSTD README` 的统一 benchmark 结果
  - 适合做同口径横向比较，但不应表述成“作者原文直接报告”
- `*`：
  - 表示该复杂度数字来自本地基于公开实现的工程复核结果
- `未报告 / 待核`：
  - 表示当前公开可访问的论文摘要、代码仓库或本地 benchmark 材料中，没有找到明确复杂度数字，后续仍可继续补查
- `未报告 / 受限`：
  - 表示当前不仅缺少明确复杂度数字，而且受公开代码形态、老版本依赖或缺源码等限制，短期内不适合低成本复核
- `Fusion quality metrics`：
  - 指 EN、SD、MI、VIF、SSIM 等融合质量指标，具体使用哪组指标需要按论文原文补充
- `MR / AP`：
  - 常见于 pedestrian / object detection 的 log-average miss rate 和平均精度评价
