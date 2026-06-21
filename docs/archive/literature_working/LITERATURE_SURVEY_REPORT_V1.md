# 红外小目标检测与多模态小目标检测文献综述 V1

> 版本：V1  
> 时间范围：2020 至今  
> 主题范围：
> 1. 红外小目标检测（IRSTD）
> 2. 红外-可见融合用于小目标检测
> 3. 基于红外与 RGB 传感器的多模态小目标检测
>
> 说明：本版本基于当前已核验的论文、代码仓库、数据集与本地 benchmark 信息整理而成，已经具备“论文综述骨架 + 定量比较 + 相关工作草稿”的形态。后续将继续补全个别方法的正式 venue、DOI、FLOPs 与 Params。

---

## 1. Structured Literature Review

### 1.1 Category A: Traditional / Model-driven IRSTD

#### 代表工作 1：ALCNet
- **论文**：Attentional Local Contrast Networks for Infrared Small Target Detection
- **作者**：Yimian Dai, Yiquan Wu, Fei Zhou, Kobus Barnard
- **年份**：2020
- **Venue**：IEEE TGRS
- **DOI**：10.1109/TGRS.2020.3044958
- **数据集**：SIRST
- **代码**：可获取
- **问题**：传统 IRSTD 方法虽然可解释，但难以适配复杂背景；纯 CNN 方法又容易忽略局部对比这一红外先验。
- **核心思想**：将局部对比测度显式模块化，并嵌入深度网络，通过注意力调制增强小目标响应。
- **网络结构**：CNN backbone + local contrast refinement + attentional modulation
- **创新点**：把传统先验直接写入端到端网络，是从传统方法走向深度方法的重要桥梁。
- **优点**：解释性强；参数量小；适合弱小目标。
- **局限**：全局依赖建模能力弱，多尺度表达不如后续 U-Net / Transformer 路线。

### 1.2 Category B: CNN-based IRSTD

#### 代表工作 2：DNANet
- **论文**：Dense Nested Attention Network for Infrared Small Target Detection
- **作者**：Boyang Li, Chao Xiao, Longguang Wang, Yingqian Wang, Zaiping Lin, Miao Li, Wei An, Yulan Guo
- **年份**：2021
- **Venue**：待进一步补正式期刊/会议归属
- **链接**：arXiv:2106.00487
- **数据集**：NUDT-SIRST 等
- **代码**：可获取
- **问题**：小目标在深层特征提取过程中容易被下采样和语义压缩淹没。
- **核心思想**：通过 dense nested interaction 保持高低层特征交互，并用 channel-spatial attention 强化目标响应。
- **网络结构**：nested encoder-decoder + DNIM + CSAM
- **创新点**：提出密集嵌套交互结构，并带动 NUDT-SIRST 基准使用。
- **优点**：是 IRSTD 中非常强的 CNN baseline，特别适合复杂背景。
- **局限**：结构较重，远程依赖主要依赖卷积堆叠。

#### 代表工作 3：UIU-Net
- **论文**：UIU-Net: U-Net in U-Net for Infrared Small Object Detection
- **作者**：Xin Wu, Danfeng Hong, Jocelyn Chanussot
- **年份**：2022
- **Venue**：IEEE TIP
- **DOI**：10.1109/TIP.2022.3228497
- **数据集**：SIRST、Synthetic、ATR ground/air video sequence
- **代码**：GitHub 可获取
- **问题**：标准 U-Net 深度增加后，小目标细节容易丢失。
- **核心思想**：在大 U-Net 中嵌入小 U-Net，以增强局部结构保持和多尺度表征能力。
- **网络结构**：U-Net in U-Net + RM-DS + IC-A
- **创新点**：提出嵌套式 U-Net 设计，兼顾细节恢复与全局语义。
- **优点**：在多个数据集上表现强，结构清晰。
- **局限**：模型规模大，参数和 FLOPs 成本明显偏高。

#### 代表工作 4：ISNet
- **论文**：ISNet: Shape Matters for Infrared Small Target Detection
- **作者**：Mingjing Zhang, Rui Zhang, Yuxiang Yang, Haichen Bai, Jing Zhang, Jie Guo
- **年份**：2022
- **Venue**：CVPR 2022
- **数据集**：IRSTD-1K
- **代码**：GitHub 可获取
- **问题**：很多方法主要基于强度或纹理建模，忽略了小目标的几何形态信息。
- **核心思想**：引入 shape-aware 建模思想，将目标形状先验纳入检测过程。
- **网络结构**：shape-aware CNN framework
- **创新点**：强调“shape matters”，并同步推出 IRSTD-1K 数据集。
- **优点**：在真实复杂场景中泛化较好；参数量低。
- **局限**：对极弱目标或点状目标，形状先验可能不稳定。

#### 代表工作 5：RDIAN
- **论文**：Receptive-Field and Direction Induced Attention Network for Infrared Dim Small Target Detection With a Large-Scale Dataset IRDST
- **作者**：Heng Sun, Junxiang Bai, Fan Yang, Xiangzhi Bai
- **年份**：2023
- **Venue**：IEEE TGRS
- **DOI**：10.1109/TGRS.2023.3235150
- **数据集**：IRDST
- **代码**：GitHub 可获取
- **问题**：dim target 在复杂背景下对方向和感受野建模非常敏感。
- **核心思想**：通过方向诱导注意力和大感受野建模提升目标背景分离能力。
- **网络结构**：CNN backbone + direction-induced attention
- **创新点**：从方向敏感性角度建模红外 dim target。
- **优点**：参数量小；推理复杂度较低；复杂背景下表现好。
- **局限**：仍是 CNN 范式，长程依赖显式建模不足。

#### 代表工作 6：ISTDU-Net
- **论文**：ISTDU-Net: Infrared Small-Target Detection U-Net
- **作者**：Q. Hou, L. Zhang, F. Tan, Y. Xi, H. Zheng, N. Li
- **年份**：2022
- **Venue**：IEEE GRSL
- **DOI**：10.1109/LGRS.2022.3141584
- **代码**：GitHub 可获取
- **问题**：标准 U-Net 在小目标细节保留上不足。
- **核心思想**：面向 IRSTD 重新设计 U-Net 结构。
- **网络结构**：U-Net variant
- **创新点**：专门面向红外小目标检测的结构裁剪与适配。
- **优点**：性能与复杂度折中较好。
- **局限**：相比 Transformer 或大规模多尺度方法，上限有限。

### 1.3 Category C: Transformer-based IRSTD

#### 代表工作 7：SCTransNet
- **论文**：SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection
- **作者**：Shuai Yuan, Hanlin Qin, Xiang Yan, Naveed Akhtar, Ajmal Mian
- **年份**：2024
- **Venue**：IEEE TGRS
- **DOI**：10.1109/TGRS.2024.3383649
- **数据集**：NUDT-SIRST、NUAA-SIRST、IRSTD-1K
- **代码**：已公开
- **问题**：CNN 对全局上下文建模不足，难以在高相似背景中稳定识别目标。
- **核心思想**：在 U-shaped 框架中引入 spatial-channel cross transformer block，对空间与通道依赖联合建模。
- **网络结构**：U-shaped backbone + SCTB（SSCA + CFN）
- **创新点**：跨空间、跨通道的交叉 Transformer 设计。
- **优点**：更擅长处理复杂背景、全局依赖和跨尺度信息。
- **局限**：复杂度和训练要求更高。

### 1.4 Category D: Diffusion / Foundation / New Paradigm IRSTD

#### 代表工作 8：RPCANet
- **论文**：RPCANet: Deep Unfolding RPCA Based Infrared Small Target Detection
- **作者**：Fengyi Wu, Tianfang Zhang, Lei Li, Yian Huang, Zhenming Peng
- **年份**：2024
- **Venue**：WACV 2024
- **代码**：待进一步核
- **问题**：深度网络黑盒性强，而传统低秩稀疏方法计算复杂但可解释。
- **核心思想**：将 RPCA 优化过程展开为可训练神经网络，在保留低秩背景与稀疏目标假设的同时提升性能。
- **网络结构**：deep unfolding network
- **创新点**：连接传统模型驱动方法与现代深度网络。
- **优点**：可解释性强，适合作为“新范式”代表。
- **局限**：自由表达能力可能不如完全数据驱动大模型。

### 1.5 Category E: Infrared-Visible Fusion

#### 代表工作 9：TSJNet
- **论文**：TSJNet: A Multi-modality Target and Semantic Awareness Joint-driven Image Fusion Network
- **作者**：Yuchan Jie, Yushen Xu, Xiaosong Li, Huafeng Li, Haishu Tan, Feiping Nie
- **年份**：2024
- **数据集**：MSRS、M3FD、RoadScene、LLVIP、TNO、UMS
- **代码**：已公开
- **问题**：传统融合方法多只关注视觉质量，难以保证对下游检测和分割真正有帮助。
- **核心思想**：联合 fusion、detection 和 segmentation 子网，用目标与语义双重信号驱动融合。
- **网络结构**：fusion + detection + segmentation series framework
- **创新点**：真正的 task-driven fusion。
- **优点**：与你的研究目标高度相关，尤其适合研究“融合是否促进检测”。
- **局限**：结构复杂，训练成本高。

### 1.6 Category F: Multimodal Small Object Detection

#### 代表工作 10：MBNet
- **论文**：Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems
- **作者**：Kailai Zhou, Linsen Chen, Xun Cao
- **年份**：2020
- **数据集**：KAIST、CVC-14
- **代码**：GitHub 可获取
- **问题**：RGB-T 多模态检测中，两种模态对最终预测贡献不平衡。
- **核心思想**：通过 DMAF 与 illumination-aware alignment 解决模态不平衡。
- **网络结构**：dual-stream detector
- **创新点**：明确提出 modality imbalance 问题。
- **优点**：是 RGB-T detection 中非常典型的基准方法。
- **局限**：任务主要面向 pedestrian。

#### 代表工作 11：Anchor-free Small-scale Multispectral Pedestrian Detection
- **作者**：Alexander Wolpert, Michael Teutsch, M. Saquib Sarfraz, Rainer Stiefelhagen
- **年份**：2020
- **数据集**：KAIST
- **代码**：GitHub 可获取
- **问题**：anchor-based detector 对小尺度多光谱行人不够友好。
- **核心思想**：采用 anchor-free 检测头建模中心与尺度。
- **网络结构**：single-stage anchor-free multispectral detector
- **创新点**：在多光谱 pedestrian detection 中系统引入 anchor-free 范式。
- **优点**：适合小尺度目标。
- **局限**：通用性仍以行人检测为主。

#### 代表工作 12：Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning
- **作者**：Yiming Sun, Bing Cao, Pengfei Zhu, Qinghua Hu
- **年份**：2020
- **数据集**：DroneVehicle
- **代码**：GitHub 可获取
- **问题**：无人机 RGB-IR 车辆检测中，模态可靠性会随环境变化而波动。
- **核心思想**：通过 uncertainty-aware learning 对模态可信度动态建模。
- **网络结构**：cross-modality detector + uncertainty-aware weighting + illumination-aware NMS
- **创新点**：显式建模跨模态不确定性。
- **优点**：数据集真实、工程意义强。
- **局限**：更偏车辆检测任务。

#### 代表工作 13：ICAFusion
- **论文**：ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection
- **作者**：Jifeng Shen, Yifei Chen, Yue Liu, Xin Zuo, Heng Fan, Wankou Yang
- **年份**：2023
- **目标期刊**：Pattern Recognition（arXiv 页注明 minor revision）
- **数据集**：KAIST、FLIR、VEDAI
- **代码**：GitHub 可获取
- **问题**：传统 RGB-T / 多光谱检测缺少充分的跨模态全局交互。
- **核心思想**：使用 dual cross-attention transformer 和 iterative interaction 做特征融合。
- **网络结构**：dual-stream detector + dual cross-attention transformer
- **创新点**：迭代式跨模态交叉注意力融合。
- **优点**：适合迁移其中的 cross-attention 思路到其他框架。
- **局限**：主要面向检测，不直接适配像素级 IRSTD。

---

## 2. Chronological Development Roadmap

### 2020
- IRSTD：传统先验与 CNN 混合阶段，代表是 ALCNet
- 多模态检测：开始关注模态不平衡、anchor-free、小尺度行人和无人机 RGB-IR 车辆检测

### 2021
- IRSTD：CNN-based nested interaction 进入主导地位，DNANet 成为重要 baseline
- Transformer 开始进入 IRSTD 研究

### 2022
- IRSTD：U-Net 系方法成熟，UIU-Net、ISNet、ISTDU-Net 分别从嵌套结构、形状建模、轻量适配角度改进

### 2023
- IRSTD：RDIAN 强调方向与感受野
- RGB-T：ICAFusion 将 cross-attention / Transformer 融合推向主流

### 2024
- IRSTD：SCTransNet 代表全局建模路线进一步清晰
- 新范式：RPCANet、单点监督、foundation-driven IRSTD
- 融合：TSJNet 代表任务驱动融合路线

---

## 3. Taxonomy Figure (Text Format)

```text
IRSTD and Multimodal Small Object Detection
|
|-- Traditional / Model-driven IRSTD
|   |-- local contrast
|   |-- low-rank / sparse
|   |-- hybrid prior + deep
|       |-- ALCNet
|
|-- CNN-based IRSTD
|   |-- nested feature interaction
|   |   |-- DNANet
|   |-- nested U-Net
|   |   |-- UIU-Net
|   |-- shape-aware
|   |   |-- ISNet
|   |-- direction-aware
|   |   |-- RDIAN
|   |-- lightweight U-Net adaptation
|       |-- ISTDU-Net
|
|-- Transformer-based IRSTD
|   |-- global context modeling
|   |-- spatial-channel cross Transformer
|       |-- SCTransNet
|
|-- New-paradigm IRSTD
|   |-- deep unfolding
|   |   |-- RPCANet
|   |-- weak supervision
|   |-- foundation model adaptation
|
|-- Infrared-Visible Fusion
|   |-- image-quality-oriented fusion
|   |-- decomposition-based fusion
|   |-- task-driven fusion
|       |-- TSJNet
|
|-- Multimodal Small Object Detection
    |-- modality balance
    |   |-- MBNet
    |-- anchor-free multispectral detection
    |-- uncertainty-aware RGB-IR detection
    |   |-- DroneVehicle
    |-- cross-attention fusion
        |-- ICAFusion
```

---

## 4. Comparison Table

### 4.1 IRSTD 方法对比

| 方法 | 类别 | Backbone | Detection Head | Fusion Strategy | Attention Mechanism | Dataset | Metrics | FLOPs | Params |
|---|---|---|---|---|---|---|---|---|---|
| ALCNet | A/B | CNN | Segmentation-style | 无 | Local Contrast + Attentional Modulation | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 0.378G | 0.427M |
| DNANet | B | U-Net-like CNN | Segmentation-style | Dense nested interaction | DNIM + CSAM | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 14.261G | 4.697M |
| UIU-Net | B | U-Net in U-Net | Segmentation-style | Encoder-decoder interaction | RM-DS + IC-A | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 54.426G | 50.540M |
| ISNet | B | CNN | Segmentation-style | 无 | Shape-aware modeling | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 30.618G | 0.966M |
| RDIAN | B | CNN | Segmentation-style | 无 | Receptive-field + Direction Attention | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 3.718G | 0.217M |
| ISTDU-Net | B | U-Net variant | Segmentation-style | 无 | U-Net adaptation | SIRST-v1 / NUDT-SIRST / IRSTD-1K | IoU / Pd / Fa | 7.944G | 2.752M |
| SCTransNet | C | U-shaped + Transformer | Segmentation-style | Cross-encoder global interaction | SCTB / SSCA / CFN | NUDT-SIRST / NUAA-SIRST / IRSTD-1K | IoU / Pd / Fa | 待补 | 待补 |
| RPCANet | D | Deep Unfolding | Segmentation-style | 无 | RPCA unfolding | 常见 IRSTD 集 | IoU / Pd / Fa | 待补 | 待补 |

### 4.2 BasicIRSTD 统一复现结果

| 方法 | SIRST-v1 IoU / Pd / Fa | NUDT-SIRST IoU / Pd / Fa | IRSTD-1K IoU / Pd / Fa |
|---|---|---|---|
| ALCNet | 61.047 / 87.072 / 55.978 | 61.131 / 97.249 / 29.093 | 58.088 / 92.929 / 74.453 |
| ISNet | 70.491 / 95.057 / 67.983 | 81.236 / 97.778 / 6.343 | 61.852 / 90.236 / 31.561 |
| RDIAN | 70.737 / 95.057 / 48.158 | 82.419 / 98.836 / 14.845 | 59.939 / 87.205 / 33.307 |
| DNANet | 74.815 / 93.536 / 38.279 | 94.192 / 99.259 / 2.436 | 65.735 / 89.562 / 12.336 |
| ISTDU-Net | 75.928 / 96.198 / 38.897 | 91.762 / 98.519 / 3.769 | 65.014 / 93.939 / 26.437 |
| UIU-Net | 77.531 / 92.395 / 9.330 | 90.517 / 98.836 / 8.342 | 65.690 / 91.246 / 13.475 |

### 4.3 多模态检测方法对比（当前已补结构字段）

| 方法 | 类别 | Backbone | Detection Head | Fusion Strategy | Attention Mechanism | Dataset | Metrics | FLOPs | Params |
|---|---|---|---|---|---|---|---|---|---|
| MBNet | F | Dual-stream CNN | Detection head | RGB-T feature fusion | DMAF + illumination-aware alignment | KAIST / CVC-14 | MR / AP | 待补 | 待补 |
| Anchor-free MSPD | F | Single-stage detector | Anchor-free head | RGB-T fusion | Center-scale modeling | KAIST | MR | 待补 | 待补 |
| DroneVehicle | F | Cross-modality detector | Detection head | uncertainty-aware RGB-IR fusion | UAM + illumination-aware NMS | DroneVehicle | AP / mAP | 待补 | 待补 |
| ICAFusion | F | Dual-stream detector | Detector-agnostic | Iterative cross-attention fusion | Dual cross-attention transformer | KAIST / FLIR / VEDAI | mAP | 待补 | 待补 |
| TSJNet | E | Fusion + downstream nets | Detection + Segmentation guided | Task-driven fusion | Target-aware + semantic-aware | MSRS / M3FD / RoadScene / LLVIP | mAP / mIoU / fusion quality | 待补 | 待补 |

---

## 5. Survey-style Summary for Thesis Related Work

红外小目标检测的发展大体经历了四个阶段。第一阶段以传统先验为核心，研究者主要利用局部对比、低秩稀疏、张量分解等模型来抑制背景并增强目标响应。这类方法具有良好的物理解释性，但面对复杂背景、非均匀噪声和场景变化时常常不够稳定。第二阶段进入 CNN 主导时期，代表方法如 DNANet、UIU-Net、ISNet、RDIAN 和 ISTDU-Net，它们围绕多尺度特征交互、细粒度目标保持、形状先验和方向感知展开改进，显著提升了复杂场景下的小目标检测能力。第三阶段开始引入 Transformer 与全局建模思想，以 SCTransNet 为代表的方法强调空间与通道的联合建模，试图解决小目标与复杂背景高度相似时的歧义问题。第四阶段则朝着弱监督、可解释展开和 foundation-driven 统一范式发展，例如 RPCANet 和单点监督方法，表明该领域正在从“设计更深网络”逐步转向“设计更合理范式”。

与单模态 IRSTD 相比，多模态检测和红外-可见融合的核心问题在于如何真正利用不同模态的互补性，而不是简单拼接输入。RGB-T 检测早期主要关注特征融合位置和模态平衡问题，MBNet 是这一方向的重要代表。随后，Anchor-free Small-scale Multispectral Pedestrian Detection 说明 anchor-free 机制对小尺度多光谱目标具有天然优势。再往后，DroneVehicle 及其相关检测框架开始显式建模跨模态不确定性，使模态权重能够随场景变化动态调整。近年来，以 ICAFusion 为代表的工作进一步将 cross-attention 和 Transformer 引入多模态特征融合中，强化了全局交互能力。另一方面，在红外-可见融合领域，研究重心也从单纯追求融合图像的主观质量，转向关注融合结果是否对下游任务真正有增益。TSJNet 这类 task-driven 方法通过联合检测与分割信号来驱动融合过程，代表了融合研究从“图像增强导向”向“任务决策导向”的转变趋势。

从技术发展趋势来看，IRSTD 与多模态小目标检测正在逐步出现若干共同方向。第一，研究者越来越重视显式先验与可学习特征之间的结合，既希望获得高性能，也希望保留一定解释性。第二，多尺度与全局依赖建模仍是提升小目标检测性能的核心抓手，尤其在低信噪比与复杂背景条件下更为明显。第三，任务驱动融合逐渐替代单纯视觉质量驱动融合，成为红外-可见融合研究的新主线。第四，随着 foundation model 和弱监督方法的发展，未来 IRSTD 很可能从专门设计单任务网络，转向更统一、更可迁移、更低标注成本的框架。

---

## 6. Research Trends

### 6.1 IRSTD 方法演化
- 从局部对比和低秩稀疏先验，演化到 CNN 多尺度特征学习
- 从 CNN 局部建模，演化到 Transformer 全局建模
- 从纯黑盒深度网络，演化到可解释 deep unfolding 与 foundation-driven 范式

### 6.2 多模态检测方法演化
- 从 early / late fusion 到 modality balance
- 从简单特征拼接到跨模态 attention
- 从图像质量导向的融合到 detection / segmentation 驱动的任务融合

### 6.3 主要瓶颈
- 小目标尺度极小，容易在多层特征压缩中消失
- 背景复杂，红外亮点与噪声、云层、海杂波、地物高亮区域容易混淆
- 数据集规模普遍有限，跨数据集泛化能力不足
- 多模态场景中存在模态不平衡、错位配准、环境依赖性强等问题
- 复杂模型性能提升不总是稳定，且常伴随显著复杂度增长

### 6.4 开放挑战
- 超弱目标和复杂背景中的稳定检测
- 单帧与多帧 IRSTD 的统一建模
- 弱监督、点监督、小样本 IRSTD
- 面向部署的轻量高鲁棒多模态检测
- 融合图像质量与下游任务效果的统一评价

### 6.5 未来方向
- Foundation model adaptation for IRSTD
- Task-driven infrared-visible fusion
- RGB-T / UAV 场景的小目标检测
- Cross-dataset generalization and domain adaptation
- Deep unfolding + Transformer / large-model hybrid designs

---

## 7. BibTeX References

> 参考文献草稿已单独整理在：
- [LITERATURE_SURVEY_BIBTEX.bib](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_BIBTEX.bib)

当前已包含：
- ALCNet
- DNANet
- UIU-Net
- ISNet
- RDIAN
- ISTDU-Net
- SCTransNet
- RPCANet
- HintU
- MBNet
- Anchor-free MSPD
- DroneVehicle
- ICAFusion
- TSJNet

---

## 8. 当前版本的完成度判断

- 已完成：
  - 分类框架
  - 第一批代表论文结构化摘要
  - 时间线
  - taxonomy 文本图
  - thesis 风格 related work 草稿
  - 第一版 BibTeX
  - 第一批带复杂度和 benchmark 数字的比较表
- 仍在补充：
  - 个别条目的正式 venue / DOI
  - 多模态检测方向的 FLOPs / Params
  - 更完整的传统 IRSTD 方法池
  - 最终面向论文正文的精炼语言版本

---

## 9. Source Reliability Notes

### 9.1 当前已较稳核验的正式发表条目
- ALCNet：IEEE TGRS + DOI 已核
- UIU-Net：IEEE TIP + DOI 已核
- SCTransNet：IEEE TGRS + DOI 已核
- ISNet：CVPR 2022 已核
- RDIAN：IEEE TGRS + DOI 已核
- ISTDU-Net：IEEE GRSL + DOI 已核
- RPCANet：WACV 2024 已核

### 9.2 当前以 arXiv + 官方代码仓库为主的条目
- DNANet
- MBNet
- Anchor-free Small-scale Multispectral Pedestrian Detection
- Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning
- ICAFusion
- TSJNet

### 9.3 使用建议
- 如果是硕士论文 related work，这些条目可以正常引用，但建议：
  - 对 `已核 DOI / 正式 venue` 的方法，优先作为“正式代表工作”
  - 对 `当前主要依赖 arXiv + GitHub` 的方法，在文中表述为“预印本/公开实现”更严谨
  - 如果后续你要写开题、论文初稿或答辩 PPT，我建议再单独做一轮“正式发表状态清洗”

---

## 10. 对你当前研究主线的直接借鉴建议

> 这一部分不是综述硬性要求，但对你把文献结果转成实验计划很有帮助。

### 10.1 最适合作为你当前 baseline 对照组的工作
- `DNANet`
  - 原因：在 BasicIRSTD 统一 benchmark 中精度很强，尤其是 NUDT-SIRST
- `ISNet`
  - 原因：shape-aware 思路与你当前关注的小目标几何显著性有较强互补性
- `RDIAN`
  - 原因：方向感知 + 大感受野，且复杂度低，适合做“轻量强基线”
- `UIU-Net`
  - 原因：是 U-Net 系中的高性能代表，适合做“重模型上界”对照

### 10.2 最适合迁移到你现有 ResUNet-GCA 主线的模块
- `ISNet` 的 shape-aware 约束
  - 适合与你当前的曲率/几何先验路线结合
- `RDIAN` 的方向诱导注意力
  - 适合加在 skip connection 或浅层特征增强处
- `SCTransNet` 的 cross-scale global interaction
  - 适合作为“第二阶段高阶增强模块”，但复杂度会明显上升
- `ICAFusion` 的 cross-attention 融合思想
  - 如果你后面做红外-可见融合或 RGB-T 检测，这是最值得迁移的多模态模块之一
- `TSJNet` 的 task-driven fusion 思想
  - 适合你后面研究“融合是否真正促进检测”时做方法论参考

### 10.3 现阶段最不建议优先投入的方向
- 直接做超大模型或 foundation model adaptation
  - 原因：你当前主线实验还在验证模块有效性和跨数据集稳定性，过早上大模型会让变量过多
- 先做复杂多任务系统
  - 原因：如果单模态 IRSTD 主线还没完全稳住，过早上 fusion + detection + segmentation 三联任务会明显增加实验负担

