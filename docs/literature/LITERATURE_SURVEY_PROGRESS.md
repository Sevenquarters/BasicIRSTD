# 文献调研进度记录（IRSTD / 红外可见融合 / RGB-T 多模态检测）

> 状态：进行中  
> 说明：本文件是阶段性工作底稿，优先记录“已找到的代表性论文、代码、数据集线索”。  
> 原则：只写当前已经核到的信息；正式发表 venue / DOI 仍在继续核验的条目标注为“待核”。

---

## 1. 当前已明确的调研主线

### A. Infrared Small Target Detection（IRSTD）
- 传统/模型驱动方法
- CNN-based 方法
- Transformer-based 方法
- Foundation / Diffusion / 弱监督新范式

### B. Infrared-Visible Fusion
- 面向视觉质量的融合
- 面向下游检测/分割任务的任务驱动融合

### C. Multimodal Small Object Detection（RGB-T / 多光谱检测）
- RGB-T pedestrian / vehicle detection
- 通用多光谱目标检测
- 面向小目标/夜间场景/无人机场景的多模态检测

---

## 2. 已收集的 IRSTD 候选论文

### 2.1 传统 / 模型驱动 / 混合方法

1. **Attentional Local Contrast Networks for Infrared Small Target Detection**
   - 作者：Yimian Dai, Yiquan Wu, Fei Zhou, Kobus Barnard
   - 年份：2020
   - 来源：arXiv（正式 venue 待核）
   - 链接：[arXiv](https://arxiv.org/abs/2012.08573)
   - 关键词：local contrast、model-driven deep network、attentional modulation
   - 数据集：SIRST
   - 代码：论文摘要提到公开，仓库需进一步核对
   - 备注：这篇非常重要，适合作为“模型驱动 + 深度学习融合”的代表作

2. **Infrared small target detection based on isotropic constraint under complex background**
   - 作者：Fan Wang
   - 年份：2020
   - 来源：arXiv（更偏传统方法）
   - 链接：[arXiv](https://arxiv.org/abs/2011.12059)
   - 关键词：multilayer gray difference、isotropic constraint、Hessian
   - 数据集：摘要未明确列出公开 benchmark，后续需补
   - 代码：未见
   - 备注：可归入传统方法对照组

### 2.2 CNN-based IRSTD

1. **Dense Nested Attention Network for Infrared Small Target Detection**
   - 作者：Boyang Li, Chao Xiao, Longguang Wang, Yingqian Wang, Zaiping Lin, Miao Li, Wei An, Yulan Guo
   - 年份：2021
   - 来源：arXiv（正式 venue / DOI 待核）
   - 链接：[arXiv](https://arxiv.org/abs/2106.00487)
   - 数据集：NUDT-SIRST + 其他公开数据
   - 代码：需进一步核 GitHub
   - 备注：BasicIRSTD 中的核心基线之一，方法关键词是 DNIM + CSAM

2. **UIU-Net: U-Net in U-Net for Infrared Small Object Detection**
   - 作者：Xin Wu, Danfeng Hong, Jocelyn Chanussot
   - 年份：2022
   - 来源：arXiv；GitHub 名称显示为 `IEEE_TIP_UIU-Net`
   - 链接：[arXiv](https://arxiv.org/abs/2212.00968)
   - 数据集：SIRST、Synthetic、ATR ground/air video sequence
   - 代码：[GitHub](https://github.com/danfenghong/IEEE_TIP_UIU-Net)
   - 备注：适合作为 U-Net 系方法的重要代表

3. **ISNet: Shape Matters for Infrared Small Target Detection**
   - 作者：待补
   - 年份：待核
   - 来源：BasicIRSTD README 已给出论文与代码入口
   - 链接：代码仓库 [GitHub](https://github.com/RuiZhang97/ISNet)
   - 数据集：IRSTD-1K
   - 代码：有
   - 备注：需补齐正式题录、venue、DOI

4. **RDIAN**
   - 作者：待补
   - 年份：待核
   - 来源：BasicIRSTD README 已给出论文与代码入口
   - 链接：代码仓库 [GitHub](https://github.com/sun11999/RDIAN)
   - 数据集：IRDST
   - 代码：有
   - 备注：需补齐正式题录、venue、DOI

5. **ISTDU-Net**
   - 作者：待补
   - 年份：待核
   - 来源：BasicIRSTD README 已给出论文与代码入口
   - 链接：代码仓库 [GitHub](https://github.com/zhanglw882/ISTDU-Net)
   - 数据集：待核
   - 代码：有
   - 备注：需补齐正式题录、venue、DOI

6. **Lost in UNet: Improving Infrared Small Target Detection by Underappreciated Local Features**
   - 作者：Wuzhou Quan, Wei Zhao, Weiming Wang, Haoran Xie, Fu Lee Wang, Mingqiang Wei
   - 年份：2024
   - 来源：arXiv
   - 链接：[arXiv](https://arxiv.org/abs/2406.13445)
   - 数据集：NUDT-SIRST、SIRSTv2、IRSTD1K
   - 代码：[GitHub](https://github.com/Wuzhou-Quan/HintU)
   - 备注：更偏“对 U-Net 系方法的结构增强/插件化改造”

### 2.3 Transformer-based IRSTD

1. **Infrared Small-Dim Target Detection with Transformer under Complex Backgrounds**
   - 作者：Fangcen Liu, Chenqiang Gao, Fang Chen, Deyu Meng, Wangmeng Zuo, Xinbo Gao
   - 年份：2021
   - 来源：arXiv
   - 链接：[arXiv](https://arxiv.org/abs/2109.14379)
   - 数据集：两个公开数据集，正式名称待补
   - 代码：待核
   - 备注：属于较早将 Transformer 引入 IRSTD 的代表工作

2. **SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection**
   - 作者：Shuai Yuan, Hanlin Qin, Xiang Yan, Naveed Akhtar, Ajmal Mian
   - 年份：2024
   - 来源：arXiv
   - 链接：[arXiv](https://arxiv.org/abs/2401.15583)
   - 数据集：NUDT-SIRST、NUAA-SIRST、IRSTD-1K
   - 代码：摘要中提到 GitHub 将公开
   - 备注：当前与你项目主线最相关，适合重点精读

3. **Improved Dense Nested Attention Network Based on Transformer for Infrared Small Target Detection**
   - 作者：Chun Bao, Jie Cao, Yaqian Ning, Tianhua Zhao, Zhijun Li, Zechen Wang, Li Zhang, Qun Hao
   - 年份：2023
   - 来源：arXiv
   - 链接：[arXiv](https://arxiv.org/abs/2311.08747)
   - 数据集：NUDT-SIRST、SIRST、BIT-SIRST
   - 代码：[GitHub](https://github.com/EdwardBao1006/bit_sirst)
   - 备注：DNANet + Swin Transformer + ACmix 的混合路线

### 2.4 Foundation / Weakly Supervised / 新范式

1. **Refined Infrared Small Target Detection Scheme with Single-Point Supervision**
   - 作者：Jinmiao Zhao, Zelin Shi, Chuang Yu, Yunpeng Liu
   - 年份：2024
   - 来源：arXiv
   - 链接：[arXiv](https://arxiv.org/abs/2408.02773)
   - 数据集：摘要未完整列出，需补
   - 代码：待核
   - 备注：可归到“弱监督/低标注成本 IRSTD”

2. **SPIRIT: Adapting Vision Foundation Models for Unified Single- and Multi-Frame Infrared Small Target Detection**
   - 作者：Qian Xu, Xi Li, Fei Gao, Jie Guo, Haojuan Yuan, Shuaipeng Fan, Mingjin Zhang
   - 年份：2026
   - 来源：arXiv
   - 链接：[arXiv](https://arxiv.org/abs/2602.01843)
   - 数据集：多个 IRSTD benchmark
   - 代码：待核
   - 备注：超出当前用户要求的时间上限，但对“未来方向”很有参考价值

3. **Rethinking Infrared Small Target Detection: A Foundation-Driven Efficient Paradigm**
   - 作者：Chuang Yu, Jinmiao Zhao, Yunpeng Liu, Yaokun Li, Xiujun Shu, Yuanhao Feng, Bo Wang, Yimian Dai, Xiangyu Yue
   - 年份：2025
   - 来源：arXiv
   - 链接：[arXiv](https://arxiv.org/abs/2512.05511)
   - 数据集：多个公开数据集
   - 代码：[GitHub](https://github.com/YuChuang1205/FDEP-Framework)
   - 备注：虽然超出 2024，但很适合写“foundation model 趋势展望”

---

## 3. 已收集的红外-可见融合候选论文

1. **Infrared and Visible Image Fusion via Interactive Compensatory Attention Adversarial Learning**
   - 作者：Zhishe Wang, Wenyu Shao, Yanlin Chen, Jiawei Xu, Xiaoqin Zhang
   - 年份：2023
   - 来源：IEEE Transactions on Multimedia
   - 链接：[arXiv](https://arxiv.org/abs/2203.15337)
   - DOI：`10.1109/TMM.2022.3228685`
   - 数据集：待补
   - 代码：[GitHub](https://github.com/Zhishe-Wang/ICAFusion)
   - 备注：偏经典融合网络，名字与下方多光谱检测的 ICAFusion 不是同一篇

2. **Decomposition-based and Interference Perception for Infrared and Visible Image Fusion in Complex Scenes**
   - 作者：Xilai Li, Xiaosong Li, Haishu Tan
   - 年份：2024
   - 来源：arXiv
   - 链接：[arXiv](https://arxiv.org/abs/2402.02096)
   - 数据集：复杂场景融合数据，后续需补具体 benchmark
   - 代码：截至 `2026-06-18` 未检索到稳定公开仓库
   - 备注：摘要中明确提到下游任务包括 object detection / semantic segmentation；Crossref 未检到同名正式题录

3. **Beyond Night Visibility: Adaptive Multi-Scale Fusion of Infrared and Visible Images**
   - 作者：Shufan Pei, Junhong Lin, Wenxi Liu, Tiesong Zhao, Chia-Wen Lin
   - 年份：2024
   - 来源：arXiv
   - 链接：[arXiv](https://arxiv.org/abs/2403.01083)
   - 数据集：待补
   - 代码：摘要称 peer review 后释放；截至 `2026-06-18` 未检索到稳定公开仓库
   - 备注：很适合你“融合是否真正提升检测”的研究线；Crossref 未检到同名正式题录

4. **TSJNet: A Multi-modality Target and Semantic Awareness Joint-driven Image Fusion Network**
   - 作者：Yuchan Jie, Yushen Xu, Xiaosong Li, Huafeng Li, Haishu Tan, Feiping Nie
   - 年份：2024
   - 来源：arXiv
   - 链接：[arXiv](https://arxiv.org/abs/2402.01212)
   - 数据集：MSRS、M3FD、RoadScene、LLVIP
   - 代码：未检索到稳定公开仓库
   - 备注：这是“融合 + 检测/分割下游联合优化”方向的重要候选；Crossref 未检到同名正式题录

---

## 4. 已收集的 RGB-T / 多模态目标检测候选论文

1. **Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning**
   - 作者：Yiming Sun, Bing Cao, Pengfei Zhu, Qinghua Hu
   - 年份：2022
   - 来源：IEEE TCSVT 2022（预印本为 2020 arXiv）
   - 链接：[arXiv](https://arxiv.org/abs/2003.02437)
   - 数据集：DroneVehicle（56,878 张图像，RGB / IR 各半）
   - 代码 / 数据：[GitHub](https://github.com/VisDrone/DroneVehicle)
   - 备注：无人机 RGB-IR 车辆检测的代表性早期工作

2. **Anchor-free Small-scale Multispectral Pedestrian Detection**
   - 作者：Alexander Wolpert, Michael Teutsch, M. Saquib Sarfraz, Rainer Stiefelhagen
   - 年份：2020
   - 来源：arXiv
   - 链接：[arXiv](https://arxiv.org/abs/2008.08418)
   - 数据集：KAIST Multispectral Pedestrian Detection Benchmark
   - 代码：[GitHub](https://github.com/HensoldtOptronicsCV/MultispectralPedestrianDetection)
   - 备注：强调小尺度 pedestrian detection，和你的“小目标”有直接相关性

3. **Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems**
   - 作者：Kailai Zhou, Linsen Chen, Xun Cao
   - 年份：2020
   - 来源：ECCV 2020（预印本可由 arXiv 获取）
   - 链接：[arXiv](https://arxiv.org/abs/2008.03043)
   - 数据集：KAIST、CVC-14
   - 代码：[GitHub](https://github.com/CalayZhou/MBNet)
   - 备注：MBNet，是 RGB-T detection 中很常见的对照方法

4. **ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection**
   - 作者：Jifeng Shen, Yifei Chen, Yue Liu, Xin Zuo, Heng Fan, Wankou Yang
   - 年份：2023
   - 来源：Pattern Recognition 2023（预印本可由 arXiv 获取）
   - 链接：[arXiv](https://arxiv.org/abs/2308.07504)
   - 数据集：KAIST、FLIR、VEDAI
   - 代码：[GitHub](https://github.com/chanchanchan97/ICAFusion)
   - 备注：跨模态 cross-attention 融合，适合放在 Transformer / attention 融合路线

5. **From Words to Wavelengths: VLMs for Few-Shot Multispectral Object Detection**
   - 作者：Manuel Nkegoum, Minh-Tan Pham, Élisa Fromont, Bruno Avignon, Sébastien Lefèvre
   - 年份：2025
   - 来源：arXiv
   - 链接：[arXiv](https://arxiv.org/abs/2512.15971)
   - 数据集：FLIR、M3FD
   - 代码：摘要未明确
   - 备注：超出当前时间范围，但非常适合 future direction

---

## 5. 当前已确认的数据集线索

### IRSTD 常见数据集
- SIRST-v1
- SIRST-v2 / OpenDeepInfrared
- NUDT-SIRST
- NUAA-SIRST
- IRSTD-1K
- NUDT-SIRST-Sea
- IRDST
- BIT-SIRST

### 红外-可见融合 / RGB-T 常见数据集
- KAIST Multispectral Pedestrian
- CVC-14
- FLIR
- DroneVehicle
- LLVIP
- M3FD
- RoadScene
- MSRS
- VEDAI

---

## 6. 我接下来要补的关键信息

1. 核验每篇论文的**正式发表 venue**
2. 核验可公开获取的 **DOI**
3. 统一整理：
   - backbone
   - detection head
   - fusion strategy
   - attention mechanism
   - datasets
   - metrics
   - FLOPs / Params
4. 从当前候选池中筛出：
   - **最适合做 baseline 的论文**
   - **最适合迁移模块到你现有 ResUNet-GCA 主线的论文**
   - **最适合写 related work 的“代表作 + 对比作”**

---

## 7. 当前阶段判断

- **IRSTD 主线**：以 `ALCNet / DNANet / ISNet / UIU-Net / ISTDU-Net / RDIAN / SCTransNet / HintU` 为骨架，基本能搭出 2020-2024 的主演化链条。
- **红外-可见融合主线**：不能只看图像质量指标，必须优先关注“是否对下游 detection 真正增益”的方法。
- **RGB-T / 多模态检测主线**：`MBNet / Anchor-free multispectral detection / DroneVehicle / ICAFusion` 已经能构成较稳的第一批代表作集合。
- **当前最大的未完成项**：正式 venue / DOI / FLOPs / Params 还没有完全核齐，后续需要逐篇补。

---

## 8. 已核验的关键条目（本轮更新）

> 以下信息已通过 arXiv 论文页或官方 GitHub README 进一步核验。

1. **Attentional Local Contrast Networks for Infrared Small Target Detection**
   - 作者：Yimian Dai, Yiquan Wu, Fei Zhou, Kobus Barnard
   - 年份：2020
   - 正式出处：**IEEE TGRS**
   - DOI：`10.1109/TGRS.2020.3044958`
   - 任务定位：model-driven + deep network 混合式 IRSTD
   - 数据集：SIRST
   - 状态：已核 DOI，代码可获取

2. **UIU-Net: U-Net in U-Net for Infrared Small Object Detection**
   - 作者：Xin Wu, Danfeng Hong, Jocelyn Chanussot
   - 年份：2022
   - 正式出处：**IEEE Transactions on Image Processing (TIP)**
   - DOI：`10.1109/TIP.2022.3228497`
   - 数据集：SIRST、Synthetic、ATR ground/air video sequence
   - 代码：[GitHub](https://github.com/danfenghong/IEEE_TIP_UIU-Net)
   - 状态：已核正式期刊与 DOI

3. **SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection**
   - 作者：Shuai Yuan, Hanlin Qin, Xiang Yan, Naveed Akhtar, Ajmal Mian
   - 年份：2024
   - 正式出处：**IEEE Transactions on Geoscience and Remote Sensing (TGRS)**
   - DOI：`10.1109/TGRS.2024.3383649`
   - 数据集：NUDT-SIRST、NUAA-SIRST、IRSTD-1K
   - 代码：GitHub 已在 arXiv 页给出
   - 状态：已核正式期刊与 DOI

4. **ISNet: Shape Matters for Infrared Small Target Detection**
   - 作者：Mingjing Zhang, Rui Zhang, Yuxiang Yang, Haichen Bai, Jing Zhang, Jie Guo
   - 年份：2022
   - 正式出处：**CVPR 2022**
   - 代码：[GitHub](https://github.com/RuiZhang97/ISNet)
   - 数据集：IRSTD-1K（官方仓库明确称其为 largest realistic dataset）
   - 状态：已核官方仓库与 venue；DOI 仍待补

5. **Receptive-Field and Direction Induced Attention Network for Infrared Dim Small Target Detection With a Large-Scale Dataset IRDST**
   - 简称：RDIAN
   - 作者：Heng Sun, Junxiang Bai, Fan Yang, Xiangzhi Bai
   - 年份：2023
   - 正式出处：**IEEE TGRS**
   - DOI：`10.1109/TGRS.2023.3235150`
   - 数据集：IRDST
   - 代码：[GitHub](https://github.com/sun11999/RDIAN)
   - 状态：已核官方仓库题录、DOI、数据集链接

6. **ISTDU-Net: Infrared Small-Target Detection U-Net**
   - 作者：Q. Hou, L. Zhang, F. Tan, Y. Xi, H. Zheng, N. Li
   - 年份：2022
   - 正式出处：**IEEE Geoscience and Remote Sensing Letters (GRSL)**
   - DOI：`10.1109/LGRS.2022.3141584`
   - 代码：[GitHub](https://github.com/zhanglw882/ISTDU-Net)
   - 状态：已核官方仓库题录与 DOI

7. **RPCANet: Deep Unfolding RPCA Based Infrared Small Target Detection**
   - 作者：Fengyi Wu, Tianfang Zhang, Lei Li, Yian Huang, Zhenming Peng
   - 年份：2024
   - 正式出处：**WACV 2024**
   - 代码：待进一步核 GitHub
   - 状态：已核会议归属；适合归入“可解释 / 深度展开 / 新范式”

8. **Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems**
   - 简称：MBNet
   - 作者：Kailai Zhou, Linsen Chen, Xun Cao
   - 年份：2020
   - 数据集：KAIST、CVC-14
   - 代码：[GitHub](https://github.com/CalayZhou/MBNet)
   - 状态：`ECCV 2020` 已核；公开仓库未见统一 `FLOPs / Params`

9. **Anchor-free Small-scale Multispectral Pedestrian Detection**
   - 作者：Alexander Wolpert, Michael Teutsch, M. Saquib Sarfraz, Rainer Stiefelhagen
   - 年份：2020
   - 数据集：KAIST Multispectral Pedestrian Detection Benchmark
   - 代码：[GitHub](https://github.com/HensoldtOptronicsCV/MultispectralPedestrianDetection)
   - 状态：`BMVC 2020` 已核；公开仓库未见统一 `FLOPs / Params`

10. **ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection**
    - 作者：Jifeng Shen, Yifei Chen, Yue Liu, Xin Zuo, Heng Fan, Wankou Yang
    - 年份：2023
    - 正式出处：**Pattern Recognition 2023**
    - 数据集：KAIST、FLIR、VEDAI
    - 代码：[GitHub](https://github.com/chanchanchan97/ICAFusion)
    - 状态：DOI 已核；公开说明未给出统一 `FLOPs / Params`

11. **Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning**
    - 作者：Yiming Sun, Bing Cao, Pengfei Zhu, Qinghua Hu
    - 年份：2022
    - 数据集：DroneVehicle（56,878 张图像，RGB / IR 各半）
    - 代码 / 数据：[GitHub](https://github.com/VisDrone/DroneVehicle)
    - 状态：`IEEE TCSVT 2022` 与 DOI 已核；公开说明未给出统一 `FLOPs / Params`

12. **TSJNet: A Multi-modality Target and Semantic Awareness Joint-driven Image Fusion Network**
    - 作者：Yuchan Jie, Yushen Xu, Xiaosong Li, Huafeng Li, Haishu Tan, Feiping Nie
    - 年份：2024
    - 数据集：MSRS、M3FD、RoadScene、LLVIP
    - 代码：未检索到稳定公开仓库
    - 状态：当前仍以 arXiv 证据为主；截至 `2026-06-18` 未检到同名 Crossref 题录；非常适合作为“融合是否促进检测/分割”的代表方法
