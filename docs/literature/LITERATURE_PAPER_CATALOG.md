# 代表论文目录总表

> 说明：本表按 `A-F` 六类整理当前已收集的代表论文，优先记录可核验的 `题目 / 作者 / 年份 / venue / DOI或代码链接 / 数据集 / 代码可用性`。  
> 用途：可直接作为后续论文 related work、开题汇报、实验对照方法筛选的“目录层”材料。  
> 状态：持续补充中。  
> 口径说明：多模态检测主线中的 `MBNet / Anchor-free MSPD / DroneVehicle / ICAFusion` 已基本坐实正式出处；部分 fusion-only 工作当前仍以 `arXiv` 预印本与公开项目页作为主要证据来源。
> 当前统计：共收录 `29` 篇代表论文，其中 `A/B/C/D/E/F` 六类分别为 `6/7/3/4/4/5` 篇。

---

## A. Traditional / Model-driven IRSTD

| 题目 | 作者 | 年份 | Venue | DOI / 链接 | 数据集 | 代码可用性 |
|---|---|---:|---|---|---|---|
| Attentional Local Contrast Networks for Infrared Small Target Detection | Yimian Dai, Yiquan Wu, Fei Zhou, Kobus Barnard | 2020 | IEEE TGRS | DOI: `10.1109/TGRS.2020.3044958` | SIRST | 可获取 |
| Infrared small target detection based on isotropic constraint under complex background | Fan Wang | 2020 | arXiv | [arXiv](https://arxiv.org/abs/2011.12059) | 复杂背景红外场景（摘要未列出标准公开集名称） | 暂未明确 |
| Nonconvex Tensor Low-Rank Approximation for Infrared Small Target Detection | Ting Liu, Jungang Yang, Boyang Li, Chao Xiao, Yang Sun, Yingqian Wang, Wei An | 2022 | IEEE TGRS | DOI: `10.1109/TGRS.2021.3130310` / [arXiv](https://arxiv.org/abs/2105.14974) / [GitHub](https://github.com/LiuTing20a/ASTTV-NTLA) | 多种复杂场景红外数据 | 可获取 |
| Infrared Small Target Detection via Schatten Capped p-Norm-Based Non-Convex Tensor Low-Rank Approximation | Fuju Yan, Guili Xu, Junpu Wang, Quan Wu, Zhengsheng Wang | 2023 | IEEE GRSL | DOI: `10.1109/LGRS.2022.3227550` | 复杂背景红外场景 | 暂未明确 |
| Infrared Small Target Detection via Nonconvex Tensor Fibered Rank Approximation | Xuan Kong, Chunping Yang, Siying Cao, Chaohai Li, Zhenming Peng | 2022 | IEEE TGRS | DOI: `10.1109/TGRS.2021.3068465` | 复杂背景红外场景 | 暂未明确 |
| Infrared Small Target Detection Using Double-Weighted Multi-Granularity Patch Tensor Model With Tensor-Train Decomposition | Guiyu Zhang, Qunbo Lv, Zui Tao, Baoyu Zhu, Zheng Tan, Yuan Ma | 2023 | arXiv / 正式出处待继续核 | [arXiv](https://arxiv.org/abs/2310.05347) | 多种复杂场景红外数据 | 暂未明确 |

---

## B. CNN-based IRSTD

| 题目 | 作者 | 年份 | Venue | DOI / 链接 | 数据集 | 代码可用性 |
|---|---|---:|---|---|---|---|
| Dense Nested Attention Network for Infrared Small Target Detection | Boyang Li, Chao Xiao, Longguang Wang, Yingqian Wang, Zaiping Lin, Miao Li, Wei An, Yulan Guo | 2021 | arXiv（当前以预印本与代码仓为主要可核验来源） | [arXiv](https://arxiv.org/abs/2106.00487) | NUDT-SIRST 等 | 可获取 |
| UIU-Net: U-Net in U-Net for Infrared Small Object Detection | Xin Wu, Danfeng Hong, Jocelyn Chanussot | 2022 | IEEE TIP | DOI: `10.1109/TIP.2022.3228497` | SIRST, Synthetic, ATR ground/air video | [GitHub](https://github.com/danfenghong/IEEE_TIP_UIU-Net) |
| ISNet: Shape Matters for Infrared Small Target Detection | Mingjing Zhang, Rui Zhang, Yuxiang Yang, Haichen Bai, Jing Zhang, Jie Guo | 2022 | CVPR 2022 | [GitHub](https://github.com/RuiZhang97/ISNet) | IRSTD-1K | 可获取 |
| Receptive-Field and Direction Induced Attention Network for Infrared Dim Small Target Detection With a Large-Scale Dataset IRDST | Heng Sun, Junxiang Bai, Fan Yang, Xiangzhi Bai | 2023 | IEEE TGRS | DOI: `10.1109/TGRS.2023.3235150` | IRDST | [GitHub](https://github.com/sun11999/RDIAN) |
| ISTDU-Net: Infrared Small-Target Detection U-Net | Q. Hou, L. Zhang, F. Tan, Y. Xi, H. Zheng, N. Li | 2022 | IEEE GRSL | DOI: `10.1109/LGRS.2022.3141584` | 常见 IRSTD 基准 | [GitHub](https://github.com/zhanglw882/ISTDU-Net) |
| Lost in UNet: Improving Infrared Small Target Detection by Underappreciated Local Features | Wuzhou Quan, Wei Zhao, Weiming Wang, Haoran Xie, Fu Lee Wang, Mingqiang Wei | 2024 | arXiv | [arXiv](https://arxiv.org/abs/2406.13445) | NUDT-SIRST, SIRST-v2, IRSTD-1K | [GitHub](https://github.com/Wuzhou-Quan/HintU) |
| ILNet: Low-level Matters for Salient Infrared Small Target Detection | Haoqing Li, Jinfu Yang, Runshi Wang, Yifei Xu | 2023 | arXiv / 正式出处待继续核 | [arXiv](https://arxiv.org/abs/2309.13646) | NUAA-SIRST, IRSTD-1K | [GitHub](https://github.com/Li-Haoqing/ILNet) |

---

## C. Transformer-based IRSTD

| 题目 | 作者 | 年份 | Venue | DOI / 链接 | 数据集 | 代码可用性 |
|---|---|---:|---|---|---|---|
| Infrared Small-Dim Target Detection with Transformer under Complex Backgrounds | Fangcen Liu, Chenqiang Gao, Fang Chen, Deyu Meng, Wangmeng Zuo, Xinbo Gao | 2021 | arXiv | [arXiv](https://arxiv.org/abs/2109.14379) | 两个公开数据集（摘要未显式列名） | 待核 |
| SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection | Shuai Yuan, Hanlin Qin, Xiang Yan, Naveed Akhtar, Ajmal Mian | 2024 | IEEE TGRS | DOI: `10.1109/TGRS.2024.3383649` | NUDT-SIRST, NUAA-SIRST, IRSTD-1K | 已公开 |
| Improved Dense Nested Attention Network Based on Transformer for Infrared Small Target Detection | Chun Bao, Jie Cao, Yaqian Ning, Tianhua Zhao, Zhijun Li, Zechen Wang, Li Zhang, Qun Hao | 2023 | arXiv | [arXiv](https://arxiv.org/abs/2311.08747) | NUDT-SIRST, SIRST, BIT-SIRST | [GitHub](https://github.com/EdwardBao1006/bit_sirst) |

---

## D. Diffusion / Foundation / New-paradigm IRSTD

| 题目 | 作者 | 年份 | Venue | DOI / 链接 | 数据集 | 代码可用性 |
|---|---|---:|---|---|---|---|
| RPCANet: Deep Unfolding RPCA Based Infrared Small Target Detection | Fengyi Wu, Tianfang Zhang, Lei Li, Yian Huang, Zhenming Peng | 2024 | WACV 2024 | 会议已核 | 常见 IRSTD 集 | 待进一步核 |
| Refined Infrared Small Target Detection Scheme with Single-Point Supervision | Jinmiao Zhao, Zelin Shi, Chuang Yu, Yunpeng Liu | 2024 | arXiv | [arXiv](https://arxiv.org/abs/2408.02773) | 弱监督 IRSTD 场景；ICPR 2024 相关挑战验证 | 待核 |
| Rethinking Infrared Small Target Detection: A Foundation-Driven Efficient Paradigm | Chuang Yu, Jinmiao Zhao, Yunpeng Liu, Yaokun Li, Xiujun Shu, Yuanhao Feng, Bo Wang, Yimian Dai, Xiangyu Yue | 2025 | arXiv | [arXiv](https://arxiv.org/abs/2512.05511) / [GitHub](https://github.com/YuChuang1205/FDEP-Framework) | 多个公开数据集 | 可获取 |
| SPIRIT: Adapting Vision Foundation Models for Unified Single- and Multi-Frame Infrared Small Target Detection | Qian Xu, Xi Li, Fei Gao, Jie Guo, Haojuan Yuan, Shuaipeng Fan, Mingjin Zhang | 2026 | arXiv | [arXiv](https://arxiv.org/abs/2602.01843) | 多个 IRSTD benchmark | 待核 |

---

## E. Infrared-Visible Fusion

| 题目 | 作者 | 年份 | Venue | DOI / 链接 | 数据集 | 代码可用性 |
|---|---|---:|---|---|---|---|
| Infrared and Visible Image Fusion via Interactive Compensatory Attention Adversarial Learning | Zhishe Wang, Wenyu Shao, Yanlin Chen, Jiawei Xu, Xiaoqin Zhang | 2023 | IEEE Transactions on Multimedia | DOI: `10.1109/TMM.2022.3228685` / [arXiv](https://arxiv.org/abs/2203.15337) / [GitHub](https://github.com/Zhishe-Wang/ICAFusion) | 常见红外-可见融合 benchmark（摘要未显式列出，待正文核） | 可获取；更偏图像质量导向 |
| Decomposition-based and Interference Perception for Infrared and Visible Image Fusion in Complex Scenes (`UMCFuse`) | Xilai Li, Xiaosong Li, Haishu Tan | 2025 | IEEE TIP（arXiv 页面标注 published as `UMCFuse`） | [arXiv](https://arxiv.org/abs/2402.02096) / [Code](https://github.com/Linfeng-Tang/UMCFuse) | 复杂场景融合 benchmark；并评估 segmentation / detection / SOD / depth | 可获取；arXiv 页面已标注 `Published in IEEE-TIP 2025` |
| Beyond Night Visibility: Adaptive Multi-Scale Fusion of Infrared and Visible Images | Shufan Pei, Junhong Lin, Wenxi Liu, Tiesong Zhao, Chia-Wen Lin | 2024 | arXiv | [arXiv](https://arxiv.org/abs/2403.01083) | 夜间红外-可见融合与检测场景（摘要未完全列出） | 摘要称 peer review 后公开；截至 `2026-06-18` 未检索到稳定公开仓库，Crossref 未检到同名正式题录 |
| TSJNet: A Multi-modality Target and Semantic Awareness Joint-driven Image Fusion Network | Yuchan Jie, Yushen Xu, Xiaosong Li, Huafeng Li, Haishu Tan, Feiping Nie | 2024 | arXiv | [arXiv](https://arxiv.org/abs/2402.01212) / [GitHub](https://github.com/XylonXu01/TSJNet) | MSRS, M3FD, RoadScene, LLVIP | 可获取官方代码；截至 `2026-06-18` 仍未检到稳定正式 venue / DOI |

---

## F. Multimodal Small Object Detection（RGB-T / 多光谱）

| 题目 | 作者 | 年份 | Venue | DOI / 链接 | 数据集 | 代码可用性 |
|---|---|---:|---|---|---|---|
| Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems | Kailai Zhou, Linsen Chen, Xun Cao | 2020 | ECCV 2020 | [arXiv](https://arxiv.org/abs/2008.03043) / [GitHub](https://github.com/CalayZhou/MBNet) | KAIST, CVC-14 | 可获取 |
| Anchor-free Small-scale Multispectral Pedestrian Detection | Alexander Wolpert, Michael Teutsch, M. Saquib Sarfraz, Rainer Stiefelhagen | 2020 | BMVC 2020 | [arXiv](https://arxiv.org/abs/2008.08418) / [GitHub](https://github.com/HensoldtOptronicsCV/MultispectralPedestrianDetection) | KAIST | 可获取 |
| Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning | Yiming Sun, Bing Cao, Pengfei Zhu, Qinghua Hu | 2022 | IEEE TCSVT | DOI: `10.1109/TCSVT.2022.3168279` / [GitHub](https://github.com/VisDrone/DroneVehicle) | DroneVehicle（56,878 张图像，RGB/IR 各半） | 可获取 |
| ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection | Jifeng Shen, Yifei Chen, Yue Liu, Xin Zuo, Heng Fan, Wankou Yang | 2023 | Pattern Recognition | DOI: `10.1016/j.patcog.2023.109913` / [GitHub](https://github.com/chanchanchan97/ICAFusion) | KAIST, FLIR, VEDAI | 可获取 |
| From Words to Wavelengths: VLMs for Few-Shot Multispectral Object Detection | Manuel Nkegoum, Minh-Tan Pham, Élisa Fromont, Bruno Avignon, Sébastien Lefèvre | 2025 | arXiv | [arXiv](https://arxiv.org/abs/2512.15971) | FLIR, M3FD | 待核 |

---

## 当前目录的使用建议

### 如果你要写论文 related work
- 优先从 `A/B/C/D/E/F` 六类中各选 2-4 篇最有代表性的工作写主线
- 正式发表已核验的方法可优先作为主引用
- `arXiv + GitHub` 条目建议在文中表述为“预印本/公开实现”

### 如果你要设计实验对照组
- IRSTD 主线优先看：
  - `DNANet`
  - `ISNet`
  - `RDIAN`
  - `UIU-Net`
- 多模态主线优先看：
  - `MBNet`
  - `ICAFusion`
  - `DroneVehicle`
  - `TSJNet`

### 如果你要筛模块迁移
- 形状先验：`ISNet`
- 方向注意力：`RDIAN`
- 全局交互：`SCTransNet`
- 跨模态 cross-attention：`ICAFusion`
- 任务驱动融合：`TSJNet`
