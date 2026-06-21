# 文献综述索引与总表

> 用途：作为当前调研包的导航页，方便后续直接定位综述正文、BibTeX、进度记录与方法总表。  
> 更新时间：2026-06-18

---

## 1. 文件导航

### 主报告
- [LITERATURE_SURVEY_MASTER_REPORT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_MASTER_REPORT.md)
  - 当前最完整、最接近最终交付的版本
  - 包含：结构化综述、时间线、taxonomy、thesis 风格 related work、趋势分析、任务映射、完成度判断

### 审计摘要
- [LITERATURE_SURVEY_AUDIT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_AUDIT.md)
  - 用于快速查看当前收录规模、证据等级、已坐实正式出处、未补齐字段与阶段完成度

### 论文写作指南
- [THESIS_CHAPTER_2_GUIDE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\THESIS_CHAPTER_2_GUIDE.md)
  - 用于将当前调研包直接转化为硕士论文第二章“相关工作”初稿

### 第二章章节草稿
- [THESIS_CHAPTER_2_DRAFT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\THESIS_CHAPTER_2_DRAFT.md)
  - 更接近论文正文的章节级草稿，可直接改写使用

### 旧版综述主稿
- [LITERATURE_SURVEY_REPORT_V1.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_REPORT_V1.md)
  - 保留较详细的中间过程内容
  - 适合补充查看早期结构化整理痕迹

### 综述草稿
- [LITERATURE_SURVEY_DRAFT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_DRAFT.md)
  - 早期骨架版，保留更多过程性信息

### 检索与核验记录
- [LITERATURE_SURVEY_PROGRESS.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_PROGRESS.md)
  - 记录论文候选池、已核验出处、数据集线索、代码仓库

### BibTeX
- [LITERATURE_SURVEY_BIBTEX.bib](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_BIBTEX.bib)
  - 当前已整理的 BibTeX 草稿

### 目录与比较表
- [LITERATURE_PAPER_CATALOG.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_PAPER_CATALOG.md)
  - 29 篇代表论文目录层总表
- [LITERATURE_COMPARISON_TABLE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_COMPARISON_TABLE.md)
  - 结构字段最完整的比较表
- [LITERATURE_PAPER_SUMMARIES.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_PAPER_SUMMARIES.md)
  - 逐篇摘要层材料

---

## 2. A-F 六类分类索引

### A. Traditional / Model-driven IRSTD
- ALCNet
- 传统局部对比、低秩稀疏、张量分解类方法仍待继续扩充

### B. CNN-based IRSTD
- DNANet
- UIU-Net
- ISNet
- RDIAN
- ISTDU-Net
- HintU

### C. Transformer-based IRSTD
- Infrared Small-Dim Target Detection with Transformer under Complex Backgrounds
- SCTransNet
- Transformer-enhanced DNANet variants

### D. Diffusion / Foundation / New Paradigm IRSTD
- RPCANet
- Single-point supervision IRSTD
- Foundation-driven IRSTD（趋势展望）

### E. Infrared-Visible Fusion
- Interactive Compensatory Attention Adversarial Learning
- Decomposition-based and Interference Perception
- Beyond Night Visibility
- TSJNet

### F. Multimodal Small Object Detection
- MBNet
- Anchor-free Small-scale Multispectral Pedestrian Detection
- DroneVehicle / uncertainty-aware RGB-IR detection
- ICAFusion

---

## 3. 当前已较稳核验的代表论文清单

| 方法 | 年份 | 正式出处 | DOI / 链接 | 状态 |
|---|---:|---|---|---|
| ALCNet | 2020 | IEEE TGRS | DOI 已核 | 稳 |
| UIU-Net | 2022 | IEEE TIP | DOI 已核 | 稳 |
| ISNet | 2022 | CVPR 2022 | GitHub / CVPR 已核 | 稳 |
| RDIAN | 2023 | IEEE TGRS | DOI 已核 | 稳 |
| ISTDU-Net | 2022 | IEEE GRSL | DOI 已核 | 稳 |
| SCTransNet | 2024 | IEEE TGRS | DOI 已核 | 稳 |
| RPCANet | 2024 | WACV 2024 | 会议已核 | 稳 |
| MBNet | 2020 | ECCV 2020 | arXiv / GitHub 已核 | 稳 |
| Anchor-free MSPD | 2020 | BMVC 2020 | arXiv / GitHub 已核 | 稳 |
| DroneVehicle / UA-CMDet | 2022 | IEEE TCSVT | DOI / GitHub 已核 | 稳 |
| ICAFusion | 2023 | Pattern Recognition | DOI / GitHub 已核 | 稳 |

---

## 4. 当前仍以 arXiv / 项目页证据为主的代表论文

| 方法 | 年份 | 当前可确认信息 | 备注 |
|---|---:|---|---|
| DNANet | 2021 | arXiv + GitHub + benchmark 使用广泛 | 适合做 baseline |
| TSJNet | 2024 | arXiv；截至 `2026-06-18` 未检索到同名 Crossref 题录与稳定公开仓库 | task-driven fusion 代表 |
| Beyond Night Visibility | 2024 | arXiv；截至 `2026-06-18` 未检索到同名 Crossref 题录与稳定公开仓库 | 夜间融合与检测相关代表 |
| Decomposition-based and Interference Perception | 2024 | arXiv；截至 `2026-06-18` 未检索到同名 Crossref 题录与稳定公开仓库 | 复杂场景融合代表 |
| Interactive Compensatory Attention Adversarial Learning | 2023 | `IEEE Transactions on Multimedia`, DOI `10.1109/TMM.2022.3228685` | 偏图像质量导向融合代表 |

---

## 5. 可直接引用的定量比较结论

### 轻量高效代表
- `RDIAN`：`0.217M / 3.718G`
- `ALCNet`：`0.427M / 0.378G`

### 精度强基线
- `DNANet`：在 BasicIRSTD benchmark 中 `NUDT-SIRST` 表现极强
- `UIU-Net`：`SIRST-v1` 上精度突出，但复杂度明显更高

### 结构先验代表
- `ISNet`：shape-aware 建模
- `RDIAN`：direction-aware 建模

### 性能-复杂度折中代表
- `ISTDU-Net`

---

## 6. 对你当前项目最值得关注的方法

### 最值得作为 baseline 的方法
- DNANet
- ISNet
- RDIAN
- UIU-Net

### 最值得借鉴的模块
- ISNet：shape-aware 约束
- RDIAN：方向诱导注意力
- SCTransNet：跨尺度全局交互
- ICAFusion：跨模态 cross-attention
- TSJNet：task-driven fusion

---

## 7. 当前缺口

1. 传统 IRSTD 方法代表论文数量还不够多
2. 多模态检测方向的 `FLOPs / Params` 尚不完整
3. 个别方法的正式发表状态仍需继续核
4. 还可以继续增加“适合你当前课题迁移的模块推荐表”

---

## 8. 建议的下一步使用方式

### 如果你要写硕士论文
- 主用 [LITERATURE_SURVEY_REPORT_V1.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_REPORT_V1.md)
- 引文从 [LITERATURE_SURVEY_BIBTEX.bib](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_BIBTEX.bib) 补充

### 如果你要做汇报 / 开题
- 优先使用主报告中的：
  - 时间线
  - taxonomy
  - 比较表
  - 研究趋势与未来方向

### 如果你要继续做实验
- 先看主报告中的：
  - “对你当前研究主线的直接借鉴建议”
  - benchmark 对比表
  - baseline 推荐清单

