# 红外小目标检测与多模态检测文献综述草稿

> 状态：草稿进行中  
> 范围：2020 至今，覆盖 IRSTD、红外-可见融合、RGB-T / 多光谱小目标检测  
> 说明：本文件优先搭建“最终综述成品”的结构，后续继续补齐缺失条目、FLOPs、参数量、BibTeX。

---

## 1. 分类框架（A-F）

### A. Traditional IRSTD methods
- Attentional Local Contrast Networks for Infrared Small Target Detection（ALCNet，严格说属于 model-driven + deep hybrid）
- 其他纯局部对比、低秩分解、稀疏表示、张量分解类方法待继续补齐

### B. CNN-based IRSTD
- Dense Nested Attention Network for Infrared Small Target Detection（DNANet）
- UIU-Net: U-Net in U-Net for Infrared Small Object Detection
- ISNet: Shape Matters for Infrared Small Target Detection
- RDIAN: Receptive-Field and Direction Induced Attention Network
- ISTDU-Net: Infrared Small-Target Detection U-Net
- HintU: Lost in UNet

### C. Transformer-based IRSTD
- Infrared Small-Dim Target Detection with Transformer under Complex Backgrounds
- SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection
- Improved Dense Nested Attention Network Based on Transformer for Infrared Small Target Detection

### D. Diffusion / Foundation / New Paradigm based IRSTD
- RPCANet: Deep Unfolding RPCA Based Infrared Small Target Detection
- Single-Point Supervision IRSTD
- Foundation-driven IRSTD（后续年份条目可放趋势展望）

### E. Infrared-Visible Fusion
- Infrared and Visible Image Fusion via Interactive Compensatory Attention Adversarial Learning
- Decomposition-based and Interference Perception for Infrared and Visible Image Fusion in Complex Scenes
- Beyond Night Visibility: Adaptive Multi-Scale Fusion of Infrared and Visible Images
- TSJNet: Multi-modality Target and Semantic Awareness Joint-driven Fusion

### F. Multimodal Small Target Detection（RGB-T / 多光谱）
- MBNet: Modality Balance Network
- Anchor-free Small-scale Multispectral Pedestrian Detection
- Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning
- ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection

---

## 2. 代表论文结构化整理（第一批）

### 2.1 ALCNet
- **问题**：纯数据驱动 IRSTD 容易忽略红外弱小目标固有的局部对比先验
- **核心思想**：将传统 local contrast measure 模块化为可嵌入深度网络的无参非线性 refinement layer，再结合自底向上的 attention 调制
- **网络结构**：深度卷积 backbone + cyclic shift local contrast refinement + bottom-up attentional modulation
- **创新点**：把传统局部对比先验显式写进网络，兼顾物理可解释性与端到端训练
- **优点**：解释性强；对弱小目标友好；是传统方法到深度方法之间的桥梁
- **局限**：全局建模能力有限；相比后续 U-Net / Transformer 路线，跨尺度表达仍偏弱

### 2.2 DNANet
- **问题**：小目标在深层网络中易被 pooling 和语义压缩淹没
- **核心思想**：通过 dense nested interaction 持续融合高低层特征，并用 cascaded channel-spatial attention 强化目标响应
- **网络结构**：U-Net / nested encoder-decoder 风格 + DNIM + CSAM
- **创新点**：反复交互的 nested 多层特征融合；提出 NUDT-SIRST 数据集与评价指标体系
- **优点**：对小目标保真好；是后续大量 U-Net 系改进工作的强 baseline
- **局限**：结构较重；长程依赖主要依赖卷积堆叠；跨域泛化仍有限

### 2.3 UIU-Net
- **问题**：标准 backbone 越深越容易导致小目标细节丢失，同时红外小目标亮暗变化大、对局部与全局对比要求高
- **核心思想**：在大 U-Net 中嵌入小 U-Net，做多层次分辨率保持；再用 IC-A 强化局部细节与高层语义交互
- **网络结构**：U-Net in U-Net + RM-DS + IC-A
- **创新点**：提出“嵌套 U-Net”思想，兼顾全局语义与局部对比
- **优点**：训练稳定；适合从头训练；泛化到视频序列也较好
- **局限**：主要还是 CNN 局部建模；全局关系表达不如 Transformer 路线

### 2.4 ISNet
- **问题**：仅从强度角度建模不足以准确刻画红外小目标的真实几何与形状特征
- **核心思想**：强调 shape-aware 表征，把目标形状信息纳入检测框架
- **网络结构**：官方仓库表明其围绕 IRSTD-1K 与 shape-aware 检测展开；具体模块细节待继续补
- **创新点**：明确将“shape matters”作为 IRSTD 关键约束，并同步推出 IRSTD-1K 数据集
- **优点**：对真实复杂目标形态更友好；数据集贡献很大
- **局限**：依赖更强的形状表达；在极弱目标或近点目标条件下，形状信息可能仍不稳定

### 2.5 RDIAN
- **问题**：复杂背景下的 dim target 既需要足够大感受野，又需要方向敏感的细粒度表征
- **核心思想**：通过 receptive-field 与 direction induced attention 建模目标与背景差异
- **网络结构**：卷积 backbone + 方向诱导注意力模块
- **创新点**：引入大规模 IRDST 数据集，并从方向感知角度强化目标判别
- **优点**：对 dim target 更敏感；适合复杂纹理背景
- **局限**：注意力仍以 CNN 范式为主；显式长程依赖不足

### 2.6 ISTDU-Net
- **问题**：红外小目标弱、背景复杂，单纯下采样会破坏目标细节
- **核心思想**：基于 U-Net 设计专用于 IRSTD 的结构，强调小目标特征的保持与恢复
- **网络结构**：U-Net 变体
- **创新点**：面向 IRSTD 的轻量 U-Net 适配
- **优点**：工程实现清晰；容易复现和改造
- **局限**：方法思想相对朴素；与后续多尺度 attention / Transformer 方法相比上限有限

### 2.7 SCTransNet
- **问题**：现有 U-shaped IRSTD 模型对全局信息建模不足，目标与背景高度相似时容易混淆
- **核心思想**：在长跳连上引入 Spatial-channel Cross Transformer Block，通过编码器间 cross attention 和多尺度前馈网络增强全尺度语义差异
- **网络结构**：U-shaped backbone + SCTB（SSCA + CFN）
- **创新点**：空间-通道交叉 Transformer；全编码器输出参与混合与再分发
- **优点**：显式建模全局依赖；对难分背景更有效
- **局限**：复杂度高于纯 CNN；对数据规模和训练稳定性要求更高

### 2.8 RPCANet
- **问题**：深度 IRSTD 网络缺少可解释性，像黑盒；传统 RPCA 又计算复杂
- **核心思想**：将 relaxed RPCA 的迭代优化步骤展开成可训练网络，同时保留目标稀疏、背景低秩的建模思想
- **网络结构**：deep unfolding network
- **创新点**：把 IRSTD 写成 sparse target extraction + low-rank background estimation + reconstruction 的统一框架
- **优点**：解释性强；连接传统低秩方法与深度学习
- **局限**：表达能力上可能不如完全自由学习的大模型；结构设计依赖先验假设

### 2.9 MBNet
- **问题**：RGB-T pedestrian detection 中两模态贡献不平衡，导致融合和优化困难
- **核心思想**：提出 modality balance，利用 DMAF 与 illumination-aware alignment 改善模态互补
- **网络结构**：双流检测网络 + DMAF + illumination-aware feature alignment
- **创新点**：把“模态不平衡”明确成一类核心问题
- **优点**：在 KAIST / CVC-14 上有效；启发后续很多 RGB-T 平衡融合方法
- **局限**：任务主要面向 pedestrian；对更一般的小目标场景迁移仍需验证

### 2.10 Anchor-free Small-scale Multispectral Pedestrian Detection
- **问题**：两阶段 anchor-based 检测器对小尺度、遮挡行人不够友好
- **核心思想**：用单阶段 anchor-free 思想替代 handcrafted anchors，直接学习 center 和 scale
- **网络结构**：single-stage anchor-free multispectral detector
- **创新点**：在多光谱 pedestrian detection 中系统引入 anchor-free 设计
- **优点**：对 small-scale pedestrian 更有效；推理更简洁
- **局限**：任务定义偏行人检测；跨类别泛化有限

### 2.11 DroneVehicle / UA-CMDet
- **问题**：无人机 RGB-IR 车辆检测中，跨模态差异大，低照度下信息互补与冗余并存
- **核心思想**：通过 uncertainty-aware 模块估计模态可信度，并设计 illumination-aware NMS
- **网络结构**：cross-modality detector + uncertainty-aware weighting + illumination-aware NMS
- **创新点**：显式建模跨模态不确定性；同步构建大规模 DroneVehicle 数据集
- **优点**：场景真实；很适合研究 RGB-IR 小目标 / 车辆检测
- **局限**：主要聚焦车辆；与 IRSTD 的像素级检测任务存在任务差异

### 2.12 ICAFusion
- **问题**：传统 RGB-T / 多光谱融合在全局交互和互补建模上不足
- **核心思想**：设计 dual cross-attention transformer，同步做跨模态全局交互，并通过 iterative interaction 共享参数以降低复杂度
- **网络结构**：双流检测框架 + dual cross-attention transformer + iterative fusion block
- **创新点**：迭代式 cross-attention 多模态特征融合
- **优点**：可插拔；适配不同 backbone / detector；在 KAIST、FLIR、VEDAI 等数据集表现强
- **局限**：主要服务 detection 任务；若迁移到像素级 IRSTD，需要重新设计输出头与监督方式

### 2.13 TSJNet
- **问题**：传统融合方法常只服务单一高级任务，难以同时兼顾目标检测与语义分割需求
- **核心思想**：串联 fusion、detection、segmentation 三个子网，用目标与语义信息共同驱动融合过程
- **网络结构**：series structure of fusion + detection + segmentation subnetworks
- **创新点**：target-aware + semantic-aware joint-driven fusion
- **优点**：非常适合“融合是否提升下游任务”这一研究问题
- **局限**：系统复杂；训练成本高；不一定适合轻量部署

---

## 3. 初版比较表（持续补充）

| 方法 | 类别 | Backbone | Detection Head | Fusion Strategy | Attention / 关键模块 | 数据集 | 指标 | FLOPs | Params |
|---|---|---|---|---|---|---|---|---|---|
| ALCNet | A/B | CNN | Segmentation-style output | 无 | Local Contrast + Attentional Modulation | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 0.378G | 0.427M |
| DNANet | B | U-Net-like CNN | Segmentation-style output | 多层特征交互 | DNIM + CSAM | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 14.261G | 4.697M |
| UIU-Net | B | U-Net in U-Net | Segmentation-style output | 编解码器交互 | RM-DS + IC-A | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 54.426G | 50.540M |
| ISNet | B | CNN | Segmentation-style output | 无 | Shape-aware modeling | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 30.618G | 0.966M |
| RDIAN | B | CNN | Segmentation-style output | 无 | Receptive-field + Direction Attention | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 3.718G | 0.217M |
| ISTDU-Net | B | U-Net variant | Segmentation-style output | 无 | U-Net adaptation | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 7.944G | 2.752M |
| SCTransNet | C | U-shaped + Transformer | Segmentation-style output | 跨编码器全局交互 | SCTB / SSCA / CFN | NUDT-SIRST / NUAA-SIRST / IRSTD-1K | IoU / Pd / Fa | 待补 | 待补 |
| RPCANet | D | Deep Unfolding | Segmentation-style output | 无 | RPCA unfolding | 常见 IRSTD 集 | IoU / Pd / Fa | 待补 | 待补 |
| MBNet | F | 双流 CNN | Detection head | RGB-T feature fusion | DMAF + illumination-aware alignment | KAIST / CVC-14 | MR / AP | 待补 | 待补 |
| Anchor-free MSPD | F | Single-stage detector | Anchor-free head | RGB-T fusion | Center-scale modeling | KAIST | MR | 待补 | 待补 |
| UA-CMDet | F | Cross-modality detector | Detection head | uncertainty-aware RGB-IR fusion | UAM + illumination-aware NMS | DroneVehicle | AP / mAP | 待补 | 待补 |
| ICAFusion | F | 双流 detector | YOLOv5-style / detector-agnostic | Iterative cross-attention fusion | Dual cross-attention transformer | KAIST / FLIR / VEDAI / 扩展集 | mAP | 待补 | 待补 |
| TSJNet | E | Fusion net + downstream nets | Detection + Segmentation guided | task-driven fusion | target-aware + semantic-aware | MSRS / M3FD / RoadScene / LLVIP | mAP / mIoU / fusion quality | 待补 | 待补 |

### 3.1 来自 BasicIRSTD Benchmark 的已核性能数字

> 下表直接来自当前工作区 `README.md` 中的 benchmark 表，便于后续在论文 related work 中引用“统一复现口径”的对比结果。

| 方法 | SIRST-v1 IoU / Pd / Fa | NUDT-SIRST IoU / Pd / Fa | IRSTD-1K IoU / Pd / Fa |
|---|---|---|---|
| ALCNet | 61.047 / 87.072 / 55.978 | 61.131 / 97.249 / 29.093 | 58.088 / 92.929 / 74.453 |
| ISNet | 70.491 / 95.057 / 67.983 | 81.236 / 97.778 / 6.343 | 61.852 / 90.236 / 31.561 |
| RDIAN | 70.737 / 95.057 / 48.158 | 82.419 / 98.836 / 14.845 | 59.939 / 87.205 / 33.307 |
| DNANet | 74.815 / 93.536 / 38.279 | 94.192 / 99.259 / 2.436 | 65.735 / 89.562 / 12.336 |
| ISTDU-Net | 75.928 / 96.198 / 38.897 | 91.762 / 98.519 / 3.769 | 65.014 / 93.939 / 26.437 |
| UIU-Net | 77.531 / 92.395 / 9.330 | 90.517 / 98.836 / 8.342 | 65.690 / 91.246 / 13.475 |

### 3.2 当前可直接提炼出的比较结论

- **轻量高效代表**：
  - `RDIAN` 参数量仅 `0.217M`，但在 `NUDT-SIRST` 上达到 `82.419 / 98.836 / 14.845`
  - `ALCNet` 参数量仅 `0.427M`，复杂度最低的一档，适合做“轻量 baseline”
- **精度强 baseline**：
  - `DNANet` 在 `NUDT-SIRST` 上表现非常强，`IoU=94.192`、`Fa=2.436`
  - `UIU-Net` 在 `SIRST-v1` 上 `IoU=77.531`，是 U-Net 系强基线，但代价是 `50.540M` 参数和 `54.426G` FLOPs
- **形状与结构先验代表**：
  - `ISNet` 参数量仅 `0.966M`，但在 `NUDT-SIRST` 与 `IRSTD-1K` 上都保持较强竞争力，说明 shape-aware 约束对真实场景有效
- **工程性折中代表**：
  - `ISTDU-Net` 的参数量和 FLOPs 明显低于 `UIU-Net`，但性能仍维持在较高水平，适合做“性能-复杂度折中 baseline”

---

## 4. 当前已形成的技术脉络判断

### 4.1 IRSTD 演化主线
1. **传统局部对比 / 稀疏低秩方法**
   - 优点是解释性强、先验明确
   - 缺点是对复杂背景和数据分布变化适应性有限
2. **CNN-based U-Net / nested U-Net 时代**
   - 关注多尺度、小目标保真、skip connection、attention
   - 代表：DNANet、UIU-Net、ISNet、RDIAN、ISTDU-Net
3. **Transformer 引入全局建模**
   - 解决目标与背景高相似时的歧义
   - 代表：SCTransNet
4. **可解释展开 / 弱监督 / foundation-driven 新范式**
   - 强调解释性、低标注成本、统一范式
   - 代表：RPCANet、single-point supervision、foundation-driven IRSTD

### 4.2 多模态检测演化主线
1. **早期 RGB-T 检测**：重点是模态融合位置和融合方式
2. **模态平衡与对齐**：如 MBNet，开始显式建模“模态不平衡”
3. **anchor-free 与轻量化设计**：强调小目标和低照度场景
4. **cross-attention / Transformer 融合**：如 ICAFusion，开始重视全局跨模态交互
5. **任务驱动融合**：如 TSJNet，不再只追求 fused image 视觉效果，而是优化 detection / segmentation 结果

---

## 5. 下一轮补充重点

1. 补齐每篇论文的正式 venue 与 DOI
2. 补齐 FLOPs / Params
3. 补传统 IRSTD 方法代表作
4. 把 A-F 六类扩成正式综述段落
5. 单独整理 BibTeX 文件

---

## 6. Chronological Development Roadmap

### 2020
- IRSTD 仍以传统先验与 CNN 结合为主，代表方法是 `ALCNet`
- RGB-T / 多光谱检测开始从简单 early/late fusion 转向更细的模态平衡与 anchor-free 设计
- 代表工作：
  - ALCNet
  - MBNet
  - Anchor-free Small-scale Multispectral Pedestrian Detection
  - DroneVehicle / uncertainty-aware RGB-IR detection

### 2021
- IRSTD 进入以 `DNANet` 为代表的 nested CNN 强化阶段
- 研究重点从“能否检测”转为“如何保留弱小目标细节并抑制复杂背景”
- Transformer 开始试探性进入 IRSTD

### 2022
- U-Net 系 IRSTD 方法快速成熟，强调 shape-aware、nested U-Net、task-oriented redesign
- 代表工作：
  - UIU-Net
  - ISNet
  - ISTDU-Net
- 同期 RGB-T / 融合方向开始重视任务驱动，而不只看融合图像质量

### 2023
- RDIAN 强化方向感知与大感受野建模
- 多模态检测中，cross-attention / Transformer 融合成为主流趋势
- 代表工作：
  - RDIAN
  - ICAFusion

### 2024
- IRSTD 中 Transformer 路线进一步清晰，代表是 `SCTransNet`
- 深度展开、弱监督、任务驱动融合成为新热点
- 代表工作：
  - SCTransNet
  - RPCANet
  - TSJNet
  - HintU

### 2025 及以后（趋势展望）
- foundation model adaptation
- few-shot multispectral detection
- single-point / weakly supervised IRSTD
- 统一 single-frame 与 multi-frame 的 infrared target detection

---

## 7. Taxonomy Figure (Text Format)

```text
Infrared Small Target Detection and Multimodal Small Object Detection
|
|-- A. Traditional / Model-driven IRSTD
|   |-- Local contrast based
|   |-- Low-rank / sparse representation based
|   |-- Tensor decomposition based
|   |-- Hybrid deep + prior based
|       |-- ALCNet
|
|-- B. CNN-based IRSTD
|   |-- Plain encoder-decoder
|   |-- Nested U-Net / multi-scale interaction
|   |   |-- DNANet
|   |   |-- UIU-Net
|   |-- Shape-aware / direction-aware modeling
|   |   |-- ISNet
|   |   |-- RDIAN
|   |-- Lightweight IRSTD adaptation
|       |-- ISTDU-Net
|       |-- HintU
|
|-- C. Transformer-based IRSTD
|   |-- Global context modeling
|   |-- Cross-scale / cross-channel Transformer
|       |-- SCTransNet
|       |-- Transformer-enhanced DNANet variants
|
|-- D. Diffusion / Foundation / New Paradigm IRSTD
|   |-- Deep unfolding
|   |   |-- RPCANet
|   |-- Weak supervision
|   |   |-- Single-point supervision
|   |-- Foundation model adaptation
|
|-- E. Infrared-Visible Fusion
|   |-- Image-quality-oriented fusion
|   |-- Attention-guided fusion
|   |-- Decomposition-based fusion
|   |-- Task-driven fusion
|       |-- TSJNet
|
|-- F. Multimodal Small Object Detection (RGB-T / Multispectral)
    |-- Pedestrian detection
    |   |-- MBNet
    |   |-- Anchor-free multispectral pedestrian detection
    |-- Vehicle / drone detection
    |   |-- DroneVehicle
    |-- General multispectral object detection
        |-- ICAFusion
```

---

## 8. Survey-style Summary for Thesis Related Work

### 8.1 IRSTD 相关工作综述段落草稿

红外小目标检测经历了从传统先验建模到深度特征学习、再到全局建模与统一范式建模的发展过程。早期方法多依赖局部对比、低秩稀疏分解和张量分解等显式先验，通过利用小目标在局部邻域中的灰度突变特性实现检测。这类方法具有较强的物理可解释性，但在复杂背景、非均匀噪声和跨场景泛化方面存在明显局限。为缓解这一问题，研究者开始将传统局部对比先验嵌入深度网络中，ALCNet 即是这一阶段的代表，它通过将局部对比测度模块化，实现了传统先验与端到端学习的结合。

随着公开数据集逐步完善，基于 CNN 的 IRSTD 方法成为主流。此类方法大多采用 U-Net 或其变体作为基础框架，并围绕多尺度特征保持、跳跃连接增强、注意力建模和细粒度结构恢复展开改进。典型代表包括 DNANet、UIU-Net、ISNet、RDIAN 与 ISTDU-Net。DNANet 强调 dense nested interaction 与多层特征反复交互，适合弱小目标保持；UIU-Net 通过 U-Net in U-Net 的设计兼顾全局语义和局部细节；ISNet 则进一步指出目标形状信息在 IRSTD 中的重要性；RDIAN 从方向注意力与感受野建模角度增强 dim target 的可分性。总体来看，CNN-based 方法在 2021 至 2023 年间构成了 IRSTD 研究的主干。

近年来，随着 Transformer 在视觉任务中的广泛应用，研究重心又转向如何显式建模长程依赖与全局上下文。SCTransNet 代表了这一趋势，其在 U-shaped 框架中引入 spatial-channel cross transformer block，以增强全局尺度上的目标背景区分能力。与此同时，研究者也开始重新审视 IRSTD 的可解释性与标注成本问题。一方面，RPCANet 通过 deep unfolding 将 RPCA 优化过程展开为可训练网络，在保持低秩背景与稀疏目标假设的同时提升了表达能力；另一方面，single-point supervision 与 foundation-driven IRSTD 等方向则反映出该领域正逐步迈向弱监督、统一建模与大模型适配的新阶段。

### 8.2 红外-可见融合与多模态检测综述段落草稿

与单模态 IRSTD 相比，红外-可见融合和 RGB-T 多模态检测的核心问题并不只是“增加一个模态”，而是如何有效利用模态间的互补性，并抑制由成像机理差异带来的噪声、不确定性和信息不平衡。早期 RGB-T 检测方法主要围绕特征级融合位置、融合层次和模态对齐展开，典型代表如 MBNet。该方法明确提出模态不平衡问题，并通过动态特征重加权与光照感知对齐改善热红外与可见光分支的协同建模。此后，Anchor-free Small-scale Multispectral Pedestrian Detection 进一步表明，在小尺度目标场景下，摆脱预定义 anchor 设计有助于提升多光谱检测性能。

在更复杂的无人机和通用目标场景中，多模态检测逐渐从“简单拼接”走向“显式交互”。DroneVehicle 及其对应检测框架通过 uncertainty-aware learning 建模跨模态不确定性，说明不同光照和观测条件下各模态的贡献并不恒定。随后，ICAFusion 将 dual cross-attention transformer 引入多光谱检测，通过迭代式跨模态注意力实现更充分的全局交互，代表了当前 RGB-T / 多光谱检测领域的重要发展方向。

另一方面，红外-可见融合领域也在发生明显变化。传统融合方法大多以增强视觉质量为目标，而近期工作越来越强调 fused image 对下游任务的真实增益。TSJNet 等任务驱动方法将检测与语义分割信号直接引入融合过程，使融合网络从“重建最清晰图像”转向“生成最有利于任务决策的表示”。这一趋势与多模态检测的演进高度一致，即研究重点正在从静态图像质量指标逐渐转向面向 detection / segmentation / recognition 的联合优化。

---

## 9. Research Trend Notes

### 9.1 IRSTD 演化趋势
- 从显式先验走向可学习先验
- 从局部对比走向多尺度细节保持
- 从 CNN 局部建模走向 Transformer 全局建模
- 从黑盒检测走向可解释展开与 foundation-driven 统一范式

### 9.2 多模态检测演化趋势
- 从简单 early / late fusion 走向 feature balancing
- 从手工对齐走向跨模态 attention
- 从图像级融合走向任务驱动融合
- 从单任务 detection 走向 detection + segmentation + fusion 联合优化

### 9.3 主要技术瓶颈
- 红外小目标尺度极小、信噪比低、标注困难
- 不同数据集之间分布差异大，泛化差
- 多模态之间存在配准误差、光照偏差、模态不平衡
- 很多方法在复杂度上升后，精度收益并不总是稳定
- 评价标准仍偏碎片化，尤其 fusion 与 downstream task 间缺少统一口径

### 9.4 开放挑战
- 超弱目标与复杂海天/云层/城市背景下的稳定检测
- 小样本、弱监督、点监督条件下的 IRSTD
- 单帧与多帧 IRSTD 的统一建模
- 面向真实部署的轻量化、高鲁棒、多模态协同检测
- 融合图像质量与下游任务增益之间的统一评价

### 9.5 未来研究方向
- foundation model adaptation for IRSTD
- task-driven infrared-visible fusion
- RGB-T / IR-RGB small object detection under low-light and UAV scenarios
- cross-dataset generalization and domain adaptation
- deep unfolding + Transformer / foundation hybrid methods
