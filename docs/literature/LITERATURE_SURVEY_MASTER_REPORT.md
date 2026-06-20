# 红外小目标检测与多模态小目标检测文献调研总报告

> 版本：Master Draft  
> 时间范围：2020 至今  
> 研究范围：
> 1. Infrared Small Target Detection (IRSTD)
> 2. Infrared-Visible Fusion for Small Target Detection
> 3. Multimodal Small Object Detection using Infrared and RGB Sensors
>
> 说明：本文件作为当前调研包的统一入口，显式覆盖用户要求的 5 项输出：
> 1. Structured literature review
> 2. Chronological development roadmap
> 3. Taxonomy figure (text format)
> 4. Survey-style summary suitable for a master's thesis related work chapter
> 5. BibTeX references

---

## 0. 调研材料入口

> 使用建议：
> - 若用于论文正文写作，优先参考本文件第 `1 / 2 / 3 / 4` 节。
> - 若需要判断某篇论文的证据等级、正式出处是否已坐实、复杂度字段为何缺失，优先参考 [LITERATURE_SURVEY_AUDIT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_AUDIT.md)。

- 主报告（当前最完整综述稿）：
  - [LITERATURE_SURVEY_MASTER_REPORT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_MASTER_REPORT.md)
- 旧版综述主稿：
  - [LITERATURE_SURVEY_REPORT_V1.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_REPORT_V1.md)
- 论文目录总表：
  - [LITERATURE_PAPER_CATALOG.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_PAPER_CATALOG.md)
- 方法比较表：
  - [LITERATURE_COMPARISON_TABLE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_COMPARISON_TABLE.md)
- BibTeX：
  - [LITERATURE_SURVEY_BIBTEX.bib](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_BIBTEX.bib)
- 审计摘要：
  - [LITERATURE_SURVEY_AUDIT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_AUDIT.md)
- 第二章写作指南：
  - [THESIS_CHAPTER_2_GUIDE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\THESIS_CHAPTER_2_GUIDE.md)
- 第二章章节草稿：
  - [THESIS_CHAPTER_2_DRAFT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\THESIS_CHAPTER_2_DRAFT.md)
- 检索与核验记录：
  - [LITERATURE_SURVEY_PROGRESS.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_PROGRESS.md)
- 导航索引：
  - [LITERATURE_SURVEY_INDEX.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_INDEX.md)

---

## 1. Structured Literature Review

### 1.1 Category A: Traditional / Model-driven IRSTD

这一类方法主要基于红外小目标在局部邻域中的显著性、低秩背景与稀疏目标分离、张量分解以及局部对比等先验。其优势在于物理解释性强、对训练数据依赖较弱，但劣势是对复杂背景、非均匀噪声以及跨场景变化的适应性有限。近年来，这一路线并未退出主流讨论，而是逐步与深度网络结合，形成“模型驱动 + 可学习特征”的混合范式。ALCNet 是这类工作的典型代表，它通过显式局部对比模块保留传统方法的优点，同时利用深度网络增强判别能力。除局部对比之外，传统路线还包括各向同性约束、非凸低秩近似、张量分解等代表思路。例如，基于 isotropic constraint 的方法试图借助几何一致性抑制复杂背景中的假亮点；Nonconvex Tensor Low-Rank Approximation 已可稳定按 `IEEE TGRS 2022` 引用，强调通过更强的低秩背景建模提高目标-背景分离能力；而 `Schatten Capped p-Norm-Based Non-Convex Tensor Low-Rank Approximation (IEEE GRSL 2023)` 与 `Nonconvex Tensor Fibered Rank Approximation (IEEE TGRS 2022)` 进一步代表了非凸张量秩建模的两条正式发表分支。相比之下，Double-Weighted Multi-Granularity Patch Tensor Model with Tensor-Train Decomposition 则从多粒度补丁张量与高阶低秩表示角度建模目标与背景差异。进一步地，RPCANet 以 deep unfolding 的方式把低秩稀疏建模写入神经网络结构，体现出传统模型向“可解释深度网络”过渡的清晰趋势。

### 1.2 Category B: CNN-based IRSTD

2021 到 2023 年是 CNN-based IRSTD 的主导阶段。此类方法大多围绕 U-Net、nested U-Net、多尺度 encoder-decoder 和 attention-enhanced skip connection 展开。DNANet 通过 dense nested interaction 与 cascaded attention 实现多层特征反复交互，是公认的高性能基线之一。UIU-Net 从嵌套 U-Net 的角度提升多尺度表征能力，适合强调局部细节保持的场景。ISNet 聚焦于 target shape-aware 建模，说明小目标几何形状本身可以作为稳定先验。RDIAN 则从方向感知与大感受野角度增强 dim target 对复杂背景的鲁棒性。ISTDU-Net 代表了一类结构相对轻量、工程实现清晰、适合做性能-复杂度折中的方法。值得注意的是，近期一些工作又开始重新强调低层特征的重要性，例如 ILNet 与 HintU 都指出，红外小目标的弱纹理、局部亮度差异和边缘信息往往首先出现在浅层特征中，若网络过度追求高层语义，反而会削弱对极小目标的敏感性。因此，“保留 low-level feature” 正逐渐成为 CNN 路线中的一条清晰分支。

### 1.3 Category C: Transformer-based IRSTD

随着视觉 Transformer 的兴起，研究者开始意识到仅靠局部卷积建模很难解决小目标与复杂背景高度相似时的歧义问题。Transformer-based IRSTD 的核心价值在于显式引入长程依赖与全局上下文交互。SCTransNet 是其中的代表，它通过 spatial-channel cross transformer block 将空间关系与通道关系联合建模，在跨尺度特征混合中引入全局交互。除此之外，也有工作尝试在 DNANet 这类 CNN 基线上引入 Swin Transformer 或 ACmix 等混合模块，体现出“CNN 主体 + Transformer 插件化增强”的过渡路线。

### 1.4 Category D: Diffusion / Foundation / New-paradigm IRSTD

这一类别更多代表方法论层面的变化，而不是单纯更深或更复杂的网络。RPCANet 通过 RPCA 深度展开，把传统低秩稀疏模型和神经网络统一起来；single-point supervision 则代表了低标注成本和弱监督方向；foundation-driven IRSTD 则反映了该领域开始尝试吸收大模型与统一表征范式的力量。尽管这些方法中有些年份稍晚，但它们对判断未来方向很重要：IRSTD 不再只是在设计更复杂 backbone，而是在探索统一建模、可解释学习、弱监督与大模型迁移的新路径。

### 1.5 Category E: Infrared-Visible Fusion

红外-可见融合领域早期主要关注融合图像的视觉质量，如边缘清晰度、纹理保留与主观可读性。然而，对于小目标检测任务而言，仅获得视觉上更清晰的融合结果并不足以保证检测性能提升。近年来，研究重点逐步转向 task-driven fusion，即直接优化融合表征对 detection / segmentation 等下游任务的促进作用。TSJNet 是这一方向的重要代表，它将目标感知与语义感知显式纳入融合过程。与此同时，Interactive Compensatory Attention Adversarial Learning、Decomposition-based and Interference Perception（其后续公开版本在 arXiv 页面中已标注为 `UMCFuse`, `IEEE TIP 2025`）、Beyond Night Visibility 等工作分别从注意力补偿、复杂场景干扰抑制和多尺度自适应融合等角度推动了该方向的发展。总体而言，这一分支的研究价值应更多通过下游检测指标而非单纯的融合图像质量指标来评估。

### 1.6 Category F: Multimodal Small Object Detection

RGB-T / 多光谱目标检测的核心在于充分利用不同模态之间的互补性，但这并不意味着简单拼接两路特征即可获得稳定增益。MBNet 显式提出模态不平衡问题，是这一方向的重要起点，并可稳定视为 `ECCV 2020` 的代表作。Anchor-free Small-scale Multispectral Pedestrian Detection 进一步表明，小尺度目标在多模态场景下同样能够从 anchor-free 检测范式中受益，可作为 `BMVC 2020` 的代表工作。DroneVehicle 及其相关检测框架则通过 uncertainty-aware learning 将跨模态可靠性建模纳入系统，其正式期刊版本可以稳定追溯到 `IEEE TCSVT 2022`。ICAFusion 代表了多模态 cross-attention 融合路线，且已经能够稳定按 `Pattern Recognition 2023` 引用，这说明 Transformer 式全局交互不仅对单模态 IRSTD 有意义，在多模态检测场景中同样能够有效增强跨模态特征建模能力。总体而言，该方向的研究重点已经从“是否融合”转向“如何在检测头之前建立更稳定、更可信的跨模态交互机制”。

---

## 2. Chronological Development Roadmap

### 2020
- IRSTD：传统先验与深度网络结合开始受到关注，ALCNet 与 isotropic constraint 代表了“局部对比增强”和“模型驱动抑噪”两条并行思路
- RGB-T / 多光谱检测：模态不平衡、anchor-free、小尺度 pedestrian 与无人机 RGB-IR 车辆检测开始形成代表路线，其中 `MBNet(ECCV)` 与 `Anchor-free MSPD(BMVC)` 是关键起点

### 2021
- IRSTD：以 DNANet 为代表的 nested CNN 结构成为主流，CNN-based baseline 进入系统化竞争阶段
- Transformer 开始进入 IRSTD 研究视野

### 2022
- IRSTD：UIU-Net、ISNet、ISTDU-Net 等 U-Net 系方法快速发展
- 研究重点转向多尺度特征保持、形状先验与轻量结构设计

### 2023
- IRSTD：RDIAN 强调方向与感受野建模
- Multimodal detection：ICAFusion 将 cross-attention / Transformer 融合带入主流，检测任务开始更重视跨模态显式交互，并形成可直接引用的 `Pattern Recognition 2023` 基线

### 2024
- IRSTD：SCTransNet 明确了空间-通道联合 Transformer 的路线
- New paradigm：RPCANet、single-point supervision 等体现出可解释与低标注成本方向
- Fusion：TSJNet 代表任务驱动融合，融合研究开始更明确地接受下游任务监督
- CNN 支线：HintU、ILNet 等方法重新强调 low-level feature preservation 的重要性

### 2025 及以后
 - foundation model adaptation 成为 IRSTD 新增长点
 - few-shot multispectral detection 开始与视觉语言模型结合
 - single-frame / multi-frame unified IRSTD 逐步出现统一建模趋势
 - task-driven multimodal fusion 更强调下游检测或分割监督

---

## 3. Taxonomy Figure (Text Format)

```text
Infrared Small Target Detection and Multimodal Small Object Detection
|
|-- A. Traditional / Model-driven IRSTD
|   |-- local contrast
|   |-- low-rank / sparse
|   |-- non-convex low-rank approximation
|   |-- tensor decomposition
|   |-- prior + deep hybrid
|       |-- ALCNet
|       |-- RPCANet
|
|-- B. CNN-based IRSTD
|   |-- nested feature interaction
|   |   |-- DNANet
|   |-- nested U-Net
|   |   |-- UIU-Net
|   |-- shape-aware
|   |   |-- ISNet
|   |-- direction-aware
|   |   |-- RDIAN
|   |-- low-level feature preserving
|   |   |-- ILNet
|   |   |-- HintU
|   |-- lightweight U-Net adaptation
|       |-- ISTDU-Net
|
|-- C. Transformer-based IRSTD
|   |-- global context modeling
|   |-- spatial-channel cross Transformer
|       |-- SCTransNet
|       |-- Transformer-enhanced DNANet variants
|
|-- D. Diffusion / Foundation / New-paradigm IRSTD
|   |-- deep unfolding
|   |   |-- RPCANet
|   |-- weak supervision
|   |-- foundation model adaptation
|
|-- E. Infrared-Visible Fusion
|   |-- image-quality-oriented fusion
|   |-- decomposition-based fusion
|   |-- multi-scale adaptive fusion
|   |-- task-driven fusion
|       |-- TSJNet
|
|-- F. Multimodal Small Object Detection
    |-- modality balance
    |   |-- MBNet
    |-- anchor-free multispectral detection
    |-- uncertainty-aware RGB-IR detection
    |   |-- DroneVehicle
    |-- cross-attention fusion
        |-- ICAFusion
```

---

## 4. Survey-style Summary Suitable for a Master's Thesis Related Work Chapter

红外小目标检测的发展脉络总体上可以概括为三个阶段：传统先验驱动阶段、深度特征建模阶段以及统一范式探索阶段。早期研究主要依赖局部对比、低秩稀疏分解、非凸低秩近似与张量分解等显式先验，其主要优势在于物理可解释性强、对大规模标注数据依赖较弱，但在复杂背景、弱信噪比以及跨场景泛化条件下往往存在性能上限。随着公开数据集逐步成熟，基于深度学习的 IRSTD 方法逐渐成为主流，其中大量工作以 U-Net 及其变体为基础框架，通过多尺度交互、注意力机制和细粒度结构增强来提升弱小目标的检测性能。DNANet、UIU-Net、ISNet、RDIAN 与 ISTDU-Net 分别从密集嵌套交互、嵌套式 U-Net、形状先验、方向感知和轻量化适配等角度推动了该阶段的发展。

在此基础上，Transformer 被逐步引入 IRSTD，以缓解卷积网络在长程依赖和全局上下文建模方面的局限。SCTransNet 是该趋势中的代表性工作，它通过空间-通道交叉 Transformer 结构提升跨尺度特征融合与全局关系表达能力。与此同时，研究者开始进一步反思 IRSTD 在可解释性、监督成本和统一建模方面的不足，进而催生了 RPCANet 这类 deep unfolding 方法，以及单点监督、foundation-driven IRSTD 等新方向。由此可见，当前 IRSTD 的研究重点已经从单纯设计更深或更复杂的 backbone，逐步转向设计更统一、更可解释且更适合跨场景迁移的学习范式。

相较于单模态 IRSTD，红外-可见融合与 RGB-T 多模态小目标检测更加关注不同模态之间的互补性建模。早期工作主要围绕模态融合层次、模态对齐和模态不平衡问题展开，其中 MBNet 是具有代表性的起点，其正式出处可稳定对应 `ECCV 2020`。随后，Anchor-free Small-scale Multispectral Pedestrian Detection 说明，小尺度目标在多模态场景下同样能够从 anchor-free 检测范式中受益，可视作 `BMVC 2020` 的代表方法。DroneVehicle 相关研究则表明，在无人机 RGB-IR 车辆检测场景中，不同模态的可靠性会随环境条件显著变化，因此 uncertainty-aware learning 成为必要机制，其期刊化版本可稳定对应 `IEEE TCSVT 2022`。进一步地，ICAFusion 等方法将 cross-attention 与 Transformer 引入多模态特征交互过程，推动该方向从浅层融合逐步演进为更深层的交互式融合，并形成了 `Pattern Recognition 2023` 级别的稳定引用入口。由此可见，多模态检测的关键问题已不再是“是否融合”，而是“如何建立对场景变化、尺度变化和模态可靠性变化均更鲁棒的交互机制”。

在融合研究领域，近年来最显著的变化体现在评价目标的转移。传统红外-可见融合方法更重视融合图像的视觉质量，而新一代方法则更关注融合结果是否能够真实提升检测、分割等下游任务性能。TSJNet 这类 task-driven fusion 方法通过显式引入目标感知与语义感知信号，体现了融合研究从“图像增强导向”向“任务决策导向”的转变。与此同时，Decomposition-based and Interference Perception（后续公开版本为 `UMCFuse`）与 Beyond Night Visibility 也说明，融合研究正在从单纯优化像素质量，转向在复杂场景和夜间场景下提升下游任务可用性。对于以小目标检测性能提升为核心目标的研究而言，这一变化具有直接的方法论意义，即融合方法的优劣应优先通过下游检测指标而非融合图像的主观观感来评判。

### 4.1 Main Technical Bottlenecks

- 弱小目标与复杂背景的对比度过低，导致局部纹理、边缘和噪声响应高度混叠。
- 单模态 IRSTD 中，小目标尺寸极小、语义极弱，深层特征虽然更抽象，却可能同时丢失最关键的浅层亮度差异和边缘信息。
- 多模态检测中，不同模态的有效性随昼夜、天气、尺度和遮挡动态变化，静态融合策略很难始终保持最优。
- 融合任务与检测任务之间存在目标不一致问题：视觉上更“好看”的融合结果，并不一定对检测最有利。
- 当前公开数据集之间的分布差异较大，导致跨数据集泛化与统一 benchmark 评价仍不充分。

### 4.2 Open Challenges

- 如何在不显著增加计算量的前提下，同时保留浅层细节、跨尺度信息和全局上下文。
- 如何为 IRSTD 建立更强的统一建模框架，使单帧、多帧、弱监督和 foundation model adaptation 能在一个体系中兼容。
- 如何在 RGB-T / RGB-IR 场景中显式估计模态可靠性，并将其稳定注入检测头之前的特征交互过程。
- 如何建立真正面向小目标检测的融合评测体系，而不是仅依赖传统融合图像质量指标。
- 如何补齐多模态与融合方向的复杂度、延迟和部署指标，支撑更公平的工程比较。

### 4.3 Future Research Directions

- 面向 IRSTD 的 foundation model adaptation 与参数高效微调，有望提升跨数据集泛化能力。
- 将 deep unfolding、低秩稀疏先验与神经网络进一步统一，是提升可解释性的重要方向。
- 面向小目标的 task-driven fusion 可能成为红外-可见融合与检测结合的主线，而不再只是独立的图像增强任务。
- 结合不确定性建模、cross-attention 和动态门控的多模态交互机制，将是 RGB-T / RGB-IR 检测的重要演化方向。
- 轻量化与部署友好型设计仍然必要，特别是在无人机、边缘设备和实时预警场景中。

---

## 5. BibTeX References

BibTeX 草稿已单独整理在：
- [LITERATURE_SURVEY_BIBTEX.bib](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_BIBTEX.bib)

当前已覆盖的主要条目包括：
- ALCNet
- isotropic constraint
- Non-Convex Tensor Low-Rank Approximation
- DNANet
- UIU-Net
- ISNet
- RDIAN
- ISTDU-Net
- SCTransNet
- RPCANet
- HintU
- MBNet
- Anchor-free Small-scale Multispectral Pedestrian Detection
- DroneVehicle
- ICAFusion
- TSJNet

---

## 6. 辅助文件说明

### 论文目录总表
- [LITERATURE_PAPER_CATALOG.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_PAPER_CATALOG.md)

### 方法比较表
- [LITERATURE_COMPARISON_TABLE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_COMPARISON_TABLE.md)

### 导航索引
- [LITERATURE_SURVEY_INDEX.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_INDEX.md)

---

## 7. 当前完成度判断

### 已完成的部分
- 已形成显式覆盖 5 项输出要求的统一主报告
- 已整理第一批 IRSTD、融合、多模态检测代表论文
- 已建立目录表、比较表、BibTeX、进度记录和导航页
- 已补入一批可直接引用的 benchmark 数字与复杂度信息
- 已形成可直接转化为 thesis 风格 related work 的正文草稿与研究趋势分析
- 已核实一批多模态代表工作的正式出处，包括 `MBNet(ECCV 2020)`、`Anchor-free MSPD(BMVC 2020)`、`DroneVehicle / UA-CMDet(TCSVT 2022)`、`ICAFusion(Pattern Recognition 2023)`
- 当前目录层已形成 `29` 篇代表论文的可审计清单，覆盖 `A/B/C/D/E/F = 6/7/3/4/4/5`

### 仍待继续补强的部分
- 传统 IRSTD 方法池已较初版更完整，但仍可继续扩展到更多低秩/张量类代表作
- 多模态与融合方向的 FLOPs / Params 还不完整
- 个别方法的正式 venue / DOI 仍需继续核验，尤其是 `TSJNet` 与 `Beyond Night Visibility` 等 fusion-only 工作；其中 `Interactive Compensatory Attention Adversarial Learning` 已可依据 Crossref 独立题录按 `IEEE Transactions on Multimedia 2023` 与 DOI `10.1109/TMM.2022.3228685` 引用，`Decomposition-based and Interference Perception` 的后续公开版本已在 arXiv 页面中标注为 `UMCFuse`，并给出 `IEEE TIP 2025` 与代码入口
- 主报告语言仍可继续精修，但已具备较强的论文正文改写基础
- 复杂度字段缺失并非单纯遗漏，更多是由于部分公开论文/仓库没有提供统一口径的 `FLOPs / Params` 统计

---

## 8. 任务完成映射

### Step 1: 搜集代表论文
- 已完成的载体：
  - [LITERATURE_PAPER_CATALOG.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_PAPER_CATALOG.md)
  - [LITERATURE_SURVEY_PROGRESS.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_PROGRESS.md)
- 当前覆盖内容：
  - 题目
  - 作者
  - 年份
  - venue
  - DOI 或 GitHub / arXiv 链接
  - 数据集
  - 代码可用性
- 当前缺口：
  - 少数条目的正式发表状态仍未完全坐实，但多模态主线中的 `MBNet / Anchor-free MSPD / DroneVehicle / ICAFusion` 已基本坐实；fusion-only 方向中，`Interactive Compensatory` 已具备独立 DOI 级题录证据，`UMCFuse` 已获得 arXiv 页面标注的 `IEEE TIP 2025` 去向与代码入口，其余如 `TSJNet`、`Beyond Night Visibility` 仍需继续补强正式题录

### Step 2: 按 A-F 六类分类
- 已完成的载体：
  - 本文件第 `1` 节
  - [LITERATURE_PAPER_CATALOG.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_PAPER_CATALOG.md)
  - [LITERATURE_SURVEY_INDEX.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_INDEX.md)
- 当前状态：
  - `A-F` 六类均已建立
  - 其中 `A 类传统 IRSTD` 已补入新的低秩近似代表作，但仍可继续扩充

### Step 3: 每篇论文总结问题、核心思想、结构、创新、优缺点
- 已完成的载体：
  - [LITERATURE_PAPER_SUMMARIES.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_PAPER_SUMMARIES.md)
  - [LITERATURE_SURVEY_REPORT_V1.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_REPORT_V1.md)
  - 本文件第 `1` 节
- 当前状态：
  - 已对目录表中的当前代表论文完成摘要级结构化总结
  - 仍可在后续继续补充更细的实验设置、损失函数、训练策略和消融细节

### Step 4: 比较表
- 已完成的载体：
  - [LITERATURE_COMPARISON_TABLE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_COMPARISON_TABLE.md)
  - [LITERATURE_SURVEY_REPORT_V1.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_REPORT_V1.md)
- 已补字段：
  - Backbone
  - Detection Head
  - Fusion Strategy
  - Attention Mechanism
  - Dataset
  - Metrics
  - 一部分 FLOPs / Params
- 当前缺口：
  - 多模态和融合方向的复杂度字段还不齐，且其中一部分缺口来自作者公开材料本身未提供标准化复杂度统计

### Step 5: 趋势分析
- 已完成的载体：
  - 本文件第 `2 / 4 / 7` 节
  - [LITERATURE_SURVEY_REPORT_V1.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_REPORT_V1.md)
- 已覆盖：
  - IRSTD 方法演化
  - 多模态检测方法演化
  - 主要瓶颈
  - 开放挑战
  - 未来方向
- 关键支撑位置：
  - 本文件第 `4.1 / 4.2 / 4.3` 节

---

## 9. 五类最终输出对应关系

### 1. Structured literature review
- 主文件：本文件第 `1` 节
- 辅助文件：[LITERATURE_SURVEY_REPORT_V1.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_REPORT_V1.md)

### 2. Chronological development roadmap
- 主文件：本文件第 `2` 节

### 3. Taxonomy figure (text format)
- 主文件：本文件第 `3` 节

### 4. Survey-style summary suitable for a master's thesis related work chapter
- 主文件：本文件第 `4` 节

### 5. BibTeX references
- 主文件：[LITERATURE_SURVEY_BIBTEX.bib](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_BIBTEX.bib)

---

## 10. 当前最值得继续补的三项

1. 把 `MBNet / Anchor-free / DroneVehicle / ICAFusion / TSJNet` 的复杂度字段继续补齐，尽量形成可比的工程表格
2. 把 `TSJNet / Beyond Night Visibility` 的正式出处、代码状态和复杂度进一步核实，并把 `UMCFuse` 的正式期刊题录补到更强证据级别
3. 把传统 IRSTD 再补 1-2 篇正式出处更强的低秩/张量方法，并继续精修主报告语言

