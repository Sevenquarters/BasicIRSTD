# 第二章相关工作写作指南

> 用途：帮助将当前文献调研包直接转化为硕士论文第二章“相关工作”初稿。  
> 配套材料：
> - 主报告：[LITERATURE_SURVEY_MASTER_REPORT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_MASTER_REPORT.md)
> - 目录总表：[LITERATURE_PAPER_CATALOG.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_PAPER_CATALOG.md)
> - 逐篇摘要：[LITERATURE_PAPER_SUMMARIES.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_PAPER_SUMMARIES.md)
> - 比较表：[LITERATURE_COMPARISON_TABLE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_COMPARISON_TABLE.md)
> - 审计摘要：[LITERATURE_SURVEY_AUDIT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_AUDIT.md)

---

## 1. 推荐章节结构

### 2.1 红外小目标检测研究现状

- 第一段：说明 IRSTD 的任务定义、应用背景与主要难点
- 第二段：概述传统 / 模型驱动方法
- 第三段：概述 CNN-based 方法
- 第四段：概述 Transformer-based 与新范式方法

### 2.2 红外-可见融合研究现状

- 第一段：说明传统融合更重视觉质量
- 第二段：说明任务驱动融合逐渐成为主线
- 第三段：指出融合方法评价应结合下游检测

### 2.3 多模态小目标检测研究现状

- 第一段：说明 RGB-T / RGB-IR 的互补性与主要挑战
- 第二段：介绍模态不平衡、anchor-free、不确定性感知路线
- 第三段：介绍 cross-attention / Transformer 融合路线

### 2.4 小结与问题归纳

- 总结当前方法的共同优势
- 总结当前方法的共同瓶颈
- 自然引出你的课题切入点

---

## 2. 建议写作顺序

1. 先用 [LITERATURE_SURVEY_MASTER_REPORT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_MASTER_REPORT.md) 第 `4` 节生成第二章主体文字。
2. 再用 [LITERATURE_COMPARISON_TABLE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_COMPARISON_TABLE.md) 补充表格与定量比较结论。
3. 用 [LITERATURE_PAPER_SUMMARIES.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_PAPER_SUMMARIES.md) 回填单篇方法的创新点、优点和局限。
4. 用 [LITERATURE_SURVEY_BIBTEX.bib](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_BIBTEX.bib) 管理参考文献。
5. 用 [LITERATURE_SURVEY_AUDIT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_AUDIT.md) 判断哪些条目适合直接写成正式出处，哪些条目应保守表述为预印本或公开实现。

---

## 3. 每节建议引用的代表方法

### 2.1 IRSTD 主线

- 传统 / 模型驱动：`ALCNet`、`Non-Convex Tensor Low-Rank Approximation`
- CNN-based：`DNANet`、`UIU-Net`、`ISNet`、`RDIAN`、`ISTDU-Net`
- Transformer / 新范式：`SCTransNet`、`RPCANet`

### 2.2 融合主线

- 图像质量导向：`Interactive Compensatory Attention Adversarial Learning`
- 复杂场景鲁棒性：`Decomposition-based and Interference Perception`
- 夜间场景导向：`Beyond Night Visibility`
- 任务驱动融合：`TSJNet`

### 2.3 多模态检测主线

- 模态不平衡：`MBNet`
- 小尺度 anchor-free：`Anchor-free Small-scale Multispectral Pedestrian Detection`
- 不确定性感知：`DroneVehicle / UA-CMDet`
- cross-attention 融合：`ICAFusion`

---

## 4. 正文中建议采用的口径

### 可直接写正式出处的条目

- `MBNet(ECCV 2020)`
- `Anchor-free MSPD(BMVC 2020)`
- `DroneVehicle / UA-CMDet(IEEE TCSVT 2022)`
- `ICAFusion(Pattern Recognition 2023)`
- 以及主线中已核实 DOI / 会议的 IRSTD 方法

### 建议保守表述的条目

- `TSJNet`
- `Beyond Night Visibility`
- `Decomposition-based and Interference Perception`
- `Interactive Compensatory Attention Adversarial Learning`
- 若干仍主要依赖 `arXiv` 的 IRSTD 预印本方法

建议表述方式：

- “某预印本工作提出……”
- “某公开实现工作表明……”
- “已有工作尝试从……角度改进……”

避免在正文中把未完全坐实的条目直接写成正式期刊或会议版本。

---

## 5. 建议保留到表格或附录的信息

- `FLOPs / Params` 不完整的方法
- 仅在项目页中出现、未在论文正文稳定报告的部署指标
- 尚未最终坐实的正式 venue
- 代码状态存在不确定性的 fusion-only 工作

这些信息建议优先放在：

- [LITERATURE_COMPARISON_TABLE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_COMPARISON_TABLE.md)
- [LITERATURE_SURVEY_AUDIT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_AUDIT.md)

---

## 6. 第二章结尾建议怎么收束

建议在第二章结尾明确写出三点：

1. 现有 IRSTD 方法在复杂背景、小目标弱纹理和跨场景泛化方面仍存在不足。
2. 现有融合与多模态检测方法虽然引入了任务驱动、cross-attention 和不确定性感知机制，但复杂度、统一评测与部署友好性仍存在明显空白。
3. 因此，有必要围绕你的课题进一步研究更鲁棒、更轻量或更具可解释性的目标检测方法。

---

## 7. 一句话使用建议

若你现在开始正式写第二章，最稳妥的流程是：

先写 [LITERATURE_SURVEY_MASTER_REPORT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_MASTER_REPORT.md) 第 `4` 节，再用 [LITERATURE_COMPARISON_TABLE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_COMPARISON_TABLE.md) 补表格，用 [LITERATURE_SURVEY_AUDIT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_AUDIT.md) 控制证据口径。

