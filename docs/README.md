# 项目文档导航

> 用途：作为当前项目文档的统一入口，方便你自己以及后续其他 agent 快速理解“文献调研、项目进度、实验记录”分别放在哪里。

---

## 1. 文档分区

### 文献调研
- 目录：`docs/literature/`
- 适用场景：
  - 查看已收集文献
  - 阅读综述主报告
  - 查方法对比表
  - 写 related work / 开题 / 小论文引言

核心入口：
- [LITERATURE_SURVEY_MASTER_REPORT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_MASTER_REPORT.md)
- [LITERATURE_PAPER_CATALOG.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_PAPER_CATALOG.md)
- [LITERATURE_COMPARISON_TABLE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_COMPARISON_TABLE.md)
- [LITERATURE_SURVEY_AUDIT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_AUDIT.md)

### 项目跟踪
- 目录：`docs/project_tracking/`
- 适用场景：
  - 了解当前项目做到哪一步
  - 看阶段总结
  - 看快速开始说明
  - 看项目结构说明

核心入口：
- [CURRENT_STATUS.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\project_tracking\CURRENT_STATUS.md)
- [PHASE_1_COMPLETE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\project_tracking\PHASE_1_COMPLETE.md)
- [PROJECT_STRUCTURE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\project_tracking\PROJECT_STRUCTURE.md)
- [QUICKSTART.txt](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\project_tracking\QUICKSTART.txt)

### 实验记录
- 目录：`docs/experiments/`
- 适用场景：
  - 回看已经做过的实验
  - 查看某个模块的实验结果
  - 积累后续写论文可直接复用的实验材料

核心入口：
- [EXPERIMENT_REPORT_GCA.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\EXPERIMENT_REPORT_GCA.md)
- [STAGE_REPORT_GCA.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\STAGE_REPORT_GCA.md)
- [BASELINE_COMPARISON_AND_V1_PLAN.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\BASELINE_COMPARISON_AND_V1_PLAN.md)
- [DNANET_FAMILY_COMPARISON_AND_V3_PLAN.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\DNANET_FAMILY_COMPARISON_AND_V3_PLAN.md)
- [DNANET_FAMILY_ONEPAGE_BRIEF.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\DNANET_FAMILY_ONEPAGE_BRIEF.md)

---

## 1.1 当前目录结构

```text
docs/
|-- README.md
|-- literature/
|   |-- LITERATURE_SURVEY_MASTER_REPORT.md
|   |-- LITERATURE_PAPER_CATALOG.md
|   |-- LITERATURE_COMPARISON_TABLE.md
|   |-- LITERATURE_PAPER_SUMMARIES.md
|   |-- LITERATURE_SURVEY_AUDIT.md
|   |-- LITERATURE_SURVEY_BIBTEX.bib
|   |-- THESIS_CHAPTER_2_GUIDE.md
|   `-- ...
|-- project_tracking/
|   |-- CURRENT_STATUS.md
|   |-- PHASE_1_COMPLETE.md
|   |-- QUICKSTART.txt
|   `-- PROJECT_STRUCTURE.md
`-- experiments/
    |-- EXPERIMENT_REPORT_GCA.md
    `-- STAGE_REPORT_GCA.md
```

---

## 2. 推荐阅读顺序

### 如果你想快速接上当前研究
1. 看 [CURRENT_STATUS.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\project_tracking\CURRENT_STATUS.md)
2. 看 [LITERATURE_SURVEY_MASTER_REPORT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_MASTER_REPORT.md)
3. 看 [LITERATURE_COMPARISON_TABLE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_COMPARISON_TABLE.md)

### 如果你想继续做实验
1. 看 [QUICKSTART.txt](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\project_tracking\QUICKSTART.txt)
2. 看 [BASELINE_COMPARISON_AND_V1_PLAN.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\BASELINE_COMPARISON_AND_V1_PLAN.md)
3. 看 [DNANET_FAMILY_COMPARISON_AND_V3_PLAN.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\DNANET_FAMILY_COMPARISON_AND_V3_PLAN.md)
4. 看 [EXPERIMENT_REPORT_GCA.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\EXPERIMENT_REPORT_GCA.md)
5. 看 [PROJECT_STRUCTURE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\project_tracking\PROJECT_STRUCTURE.md)

### 如果你想写论文或整理 related work
1. 看 [LITERATURE_SURVEY_MASTER_REPORT.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_MASTER_REPORT.md)
2. 看 [THESIS_CHAPTER_2_GUIDE.md](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\THESIS_CHAPTER_2_GUIDE.md)
3. 看 [LITERATURE_SURVEY_BIBTEX.bib](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\literature\LITERATURE_SURVEY_BIBTEX.bib)

---

## 3. 当前推荐约定

- 文献检索、综述、BibTeX、论文草稿统一放在 `docs/literature/`
- 阶段总结、项目状态、操作说明统一放在 `docs/project_tracking/`
- 单次实验报告、阶段实验记录统一放在 `docs/experiments/`
- 后续新增文档尽量不要再直接丢回根目录，优先放到对应 `docs/` 子目录
