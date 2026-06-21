# 文献调研包审计摘要

> 目的：给当前文献调研包提供一个“可快速检查”的状态面板，明确哪些内容已经形成稳定引用口径，哪些内容仍属于 `arXiv / 项目页` 级证据，哪些字段仍待补齐。  
> 更新时间：2026-06-18

---

## 1. 当前收录规模

- 当前共收录 `29` 篇代表论文
- 类别分布：
  - `A. Traditional / Model-driven IRSTD`：`6`
  - `B. CNN-based IRSTD`：`7`
  - `C. Transformer-based IRSTD`：`3`
  - `D. Diffusion / Foundation / New-paradigm IRSTD`：`4`
  - `E. Infrared-Visible Fusion`：`4`
  - `F. Multimodal Small Object Detection`：`5`

---

## 2. 已基本坐实的正式出处

- `ALCNet`：`IEEE TGRS 2020`
- `UIU-Net`：`IEEE TIP 2022`
- `ISNet`：`CVPR 2022`
- `RDIAN`：`IEEE TGRS 2023`
- `ISTDU-Net`：`IEEE GRSL 2022`
- `SCTransNet`：`IEEE TGRS 2024`
- `RPCANet`：`WACV 2024`
- `Nonconvex Tensor Low-Rank Approximation for Infrared Small Target Detection`：`IEEE TGRS 2022`
- `Infrared Small Target Detection via Schatten Capped p-Norm-Based Non-Convex Tensor Low-Rank Approximation`：`IEEE GRSL 2023`
- `Infrared Small Target Detection via Nonconvex Tensor Fibered Rank Approximation`：`IEEE TGRS 2022`
- `MBNet`：`ECCV 2020`
- `Anchor-free Small-scale Multispectral Pedestrian Detection`：`BMVC 2020`
- `Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning`：`IEEE TCSVT 2022`
- `ICAFusion`：`Pattern Recognition 2023`
- `Infrared and Visible Image Fusion via Interactive Compensatory Attention Adversarial Learning`：`IEEE Transactions on Multimedia 2023`，DOI `10.1109/TMM.2022.3228685`

---

## 3. 当前仍主要依赖 arXiv / 项目页证据的条目

### IRSTD

- `Dense Nested Attention Network for Infrared Small Target Detection`
- `ILNet`
- `HintU`
- `Infrared Small-Dim Target Detection with Transformer under Complex Backgrounds`
- `Improved Dense Nested Attention Network Based on Transformer for Infrared Small Target Detection`
- `Refined Infrared Small Target Detection Scheme with Single-Point Supervision`
- `Rethinking Infrared Small Target Detection: A Foundation-Driven Efficient Paradigm`
- `SPIRIT`
- `Infrared small target detection based on isotropic constraint under complex background`
- `Infrared Small Target Detection Using Double-Weighted Multi-Granularity Patch Tensor Model With Tensor-Train Decomposition`

### Infrared-Visible Fusion

- `Beyond Night Visibility`
- `TSJNet`

### Multimodal Detection

- `From Words to Wavelengths: VLMs for Few-Shot Multispectral Object Detection`

---

## 4. 当前最稳定的比较表字段

- `Backbone`
- `Detection Head`
- `Fusion Strategy`
- `Attention Mechanism / 关键模块`
- `Dataset`
- `Metrics`

这些字段在当前主线论文中已基本可用，适合直接支撑综述表格与 related work 对比。

---

## 5. 当前最不完整的字段

- `FLOPs`
- `Parameters`

### 缺失原因

- 一部分论文或公开仓库根本没有提供统一口径的复杂度统计
- 一部分方法只强调“更轻量 / 更高效 / 更快推理”，但没有给出标准化数字
- fusion-only 方向很多仍停留在预印本或项目页层面，公开复杂度信息更少
- 一部分多模态方法虽然有公开实现，但受老版本框架、自定义 CUDA 算子或仓库只提供检测结果等限制，当前环境下直接复核复杂度成本较高

---

## 6. 对五项任务的审计结论

### Step 1: 搜集代表论文

- 结论：`基本完成`
- 说明：目录层已经形成 `29` 篇代表论文清单，且一批早期占位条目已被替换为更完整的真实元数据；但少数条目的正式发表状态仍待继续核实

### Step 2: 分类

- 结论：`完成`
- 说明：`A-F` 六类已经建立且条目已归类

### Step 3: 逐篇摘要

- 结论：`基本完成`
- 说明：当前目录中的代表论文已具备摘要级总结；后续仍可继续补实验设置、损失函数和消融细节

### Step 4: 比较表

- 结论：`部分完成`
- 说明：结构字段较完整，且复杂度字段的证据口径已进一步区分为 `统一基准复现† / 本地复核* / 未报告 / 待核 / 未报告 / 受限`，但多模态与 fusion-only 方向的数字仍明显不全

### Step 5: 趋势分析

- 结论：`完成`
- 说明：主报告中已覆盖方法演化、技术瓶颈、开放挑战与未来方向

---

## 7. 当前最值得继续补的三项

1. 继续补 `MBNet / DroneVehicle / TSJNet` 的 `FLOPs / Params`，并视情况补强更多可复核的多模态复杂度条目
2. 继续核实 `TSJNet / Beyond Night Visibility` 的正式出处与代码状态，并补强 `UMCFuse` 的正式期刊题录证据
3. 继续扩充 `A 类传统 IRSTD` 的正式出处更强的低秩/张量方法

---

## 8. 最近一轮补齐的高价值元数据

- `Nonconvex Tensor Low-Rank Approximation for Infrared Small Target Detection` 已进一步坐实为 `IEEE TGRS 2022`，DOI 为 `10.1109/TGRS.2021.3130310`
- 新补入两篇正式发表的传统张量/低秩方法：`Schatten Capped p-Norm-Based Non-Convex Tensor Low-Rank Approximation (IEEE GRSL 2023)` 与 `Nonconvex Tensor Fibered Rank Approximation (IEEE TGRS 2022)`
- `DroneVehicle` 数据集规模已修正为 `56,878` 张图像（RGB / IR 各半）
- `Interactive Compensatory` 已进一步补强为独立 DOI 级证据：`IEEE Transactions on Multimedia 2023`, DOI `10.1109/TMM.2022.3228685`
- `Decomposition-based and Interference Perception` 的后续公开版本已在 arXiv 页面中标注为 `UMCFuse`，并给出 `IEEE TIP 2025` 去向与代码入口，因此其证据等级已由“仅 arXiv”提升为“arXiv + 代码 + 明确期刊去向”
- `TSJNet` 已确认存在官方 GitHub 仓库，但截至 `2026-06-18` 仍未检到稳定正式 venue / DOI
- `Beyond Night Visibility` 仍主要停留在 `arXiv` 层面，当前未检到稳定公开仓库与正式题录
- `ICAFusion` 已完成一轮本地工程复杂度统计：基于公开实现 `yolov5s_Transfusion_FLIR.yaml`、输入 `640x640` 和 `thop`，得到约 `15.056 GFLOPs / 23.266M Params`
- `MBNet` 的公开实现已核验为 `tensorflow1.14 + keras2.1` 老版本双流框架，当前环境下直接复核复杂度成本较高
- `Anchor-free MSPD` 的公开 GitHub 仓库已核验为“仅提供检测结果文件”，未开放可直接实例化的模型源码，因此当前无法本地复核复杂度
- `DroneVehicle / UA-CMDet` 的官方代码仓已核验存在，但其实现基于老版 `mmdetection + mmcv 0.4.3 + 自定义 CUDA ops`，并依赖 `PyTorch 1.1 / CUDA 10.0`，因此当前环境下直接复核复杂度成本较高

---

## 8.1 当前复杂度字段的证据口径

- `统一基准复现†`
  - 指数字来自当前工作区 `BasicIRSTD README` 的统一 benchmark 结果，适合同口径横向比较
  - 这类数字不应直接表述成“论文作者原文逐篇报告”
- `本地复核*`
  - 指数字来自基于公开实现的本地工程统计，如 `ICAFusion`
- `未报告 / 待核`
  - 指当前未在公开材料中定位到标准化复杂度数字，但理论上仍可能继续补查
- `未报告 / 受限`
  - 指除缺少数字外，还受老版本依赖、仓库不含可实例化模型源码或自定义算子等限制，短期内复核成本偏高

---

## 9. 与原始目标逐项对照

### 输出 1：Structured literature review

- 状态：`完成`
- 主要载体：
  - [LITERATURE_SURVEY_MASTER_REPORT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_MASTER_REPORT.md) 第 `1` 节

### 输出 2：Chronological development roadmap

- 状态：`完成`
- 主要载体：
  - [LITERATURE_SURVEY_MASTER_REPORT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_MASTER_REPORT.md) 第 `2` 节

### 输出 3：Taxonomy figure (text format)

- 状态：`完成`
- 主要载体：
  - [LITERATURE_SURVEY_MASTER_REPORT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_MASTER_REPORT.md) 第 `3` 节

### 输出 4：Survey-style summary suitable for a master's thesis related work chapter

- 状态：`完成`
- 主要载体：
  - [LITERATURE_SURVEY_MASTER_REPORT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_MASTER_REPORT.md) 第 `4` 节
  - [THESIS_CHAPTER_2_DRAFT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\THESIS_CHAPTER_2_DRAFT.md)
  - [THESIS_CHAPTER_2_GUIDE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\THESIS_CHAPTER_2_GUIDE.md)

### 输出 5：BibTeX references

- 状态：`完成`
- 主要载体：
  - [LITERATURE_SURVEY_BIBTEX.bib](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_BIBTEX.bib)

---

## 10. 当前仍未完全封版的原因

- 多模态与 fusion-only 方向的 `FLOPs / Params` 仍不完整，尽管其空白项已能区分“待继续检索”和“工程复核受限”
- 少数 fusion-only 工作仍缺少稳定可复核的正式发表信息
- 少数传统 IRSTD 预印本方法仍可继续增强出处严谨性

这意味着：

- 从“是否可用于论文写作”的角度看，当前材料已经达到可用标准
- 从“是否所有元数据均已最终坐实”的角度看，当前仍属于高完成度但未完全封版状态

