# 📂 项目文件结构说明 - 第一阶段完成

## 🎯 关键文档（当前集中在 `docs/project_tracking/`）

```
BasicIRSTD/
│
├─ docs/
│  └─ project_tracking/
│     ├─ 📖 PHASE_1_COMPLETE.md   ⭐ 【首先阅读】第一阶段完成总结
│     ├─ 📖 CURRENT_STATUS.md     ⭐ 【项目状态】当前工作进度
│     ├─ 📖 QUICKSTART.txt        快速开始实验指南（45-80 分钟）
│     └─ 📖 PROJECT_STRUCTURE.md  本文件（项目文件结构说明）
│
└─ 🔧 COMMIT_GUIDE.py             Git 提交指南（查看提交方法）
```

---

## 📁 项目完整结构

```
BasicIRSTD/
│
├─ 【核心代码】
│  ├─ train.py                    主训练脚本（已修复：设备支持）
│  ├─ test.py                     测试脚本
│  ├─ net.py                      📝 已修改：添加 ResUNet-CBAM 注册
│  ├─ dataset.py                  数据加载
│  ├─ loss.py                     损失函数
│  ├─ metrics.py                  📝 已修改：修复张量转换错误
│  ├─ utils.py                    工具函数
│  └─ cal_params.py               📝 已修改：时间戳处理
│
├─ 【模型文件夹】
│  ├─ model/
│  │  ├─ __init__.py              📝 已修改：导入 ResUNet_CBAM
│  │  ├─ ResUNet/
│  │  │  ├─ model_ResUNet.py               基线 ResUNet
│  │  │  └─ model_ResUNet_CBAM.py  ✨ 新增 改进模型（核心）
│  │  ├─ ACM/
│  │  │  └─ model_ACM.py
│  │  ├─ UIUNet/
│  │  │  ├─ model_UIUNet.py
│  │  │  └─ fusion.py                     多尺度融合参考
│  │  ├─ RDIAN/
│  │  │  ├─ model_RDIAN.py
│  │  │  └─ cbam.py                       CBAM 注意力模块
│  │  ├─ ISNet/
│  │  │  └─ my_functionals/
│  │  │     └─ GatedSpatialConv.py        门控空间卷积
│  │  └─ ... 其他模型 ...
│
├─ 【数据集文件夹】
│  └─ datasets/
│     ├─ NUDT-SIRST/                      1,327 张 256×256 图像 ✓
│     ├─ NUAA-SIRST/
│     └─ IRSTD-1K/
│
├─ 【日志和结果】
│  ├─ log/                         训练日志（自动生成）
│  ├─ results/                     推理结果（自动生成）
│  └─ params_*.txt                 参数统计（自动生成）
│
├─ 【环境配置】
│  ├─ venv/                        虚拟环境目录
│  ├─ requirements.txt             项目依赖
│  ├─ setup_env.bat                一键环境初始化脚本
│  └─ .gitignore                   Git 忽略规则
│
├─ 【第一阶段新增工具】 ✨
│  ├─ analyze_models.py                 模型分析和对比工具
│  ├─ test_resunet_cbam.py              快速验证脚本
│  ├─ run_test.bat                      Windows 启动脚本
│  ├─ QUICKSTART.txt                    实验操作指南
│  ├─ PHASE_1_COMPLETE.md               阶段总结文档
│  ├─ CURRENT_STATUS.md                 项目状态总结
│  ├─ COMMIT_GUIDE.py                   提交指南
│  └─ PROJECT_STRUCTURE.md              本文件
│
└─ 【其他】
   ├─ README.md                   原项目 README
   ├─ .git/                       Git 仓库目录
   └─ ...
```

---

## 🎯 快速定位文件

### **"我想了解前面做了什么"**
👉 打开 `PHASE_1_COMPLETE.md`

### **"我想做对比实验"**
👉 打开 `QUICKSTART.txt`

### **"我想看当前项目状态"**
👉 打开 `CURRENT_STATUS.md`

### **"我想要 commit 并推送代码"**
👉 运行 `python COMMIT_GUIDE.py`

### **"我想分析 ResUNet-CBAM 模型"**
👉 运行 `python analyze_models.py --all`

### **"我想快速验证 ResUNet-CBAM 能用"**
👉 运行 `python test_resunet_cbam.py`

### **"我想了解 ResUNet-CBAM 代码"**
👉 打开 `model/ResUNet/model_ResUNet_CBAM.py`

### **"我想看训练是否工作"**
👉 运行 `python train.py --model_names ResUNet-CBAM --dataset_names NUDT-SIRST --batchSize 8 --nEpochs 1`

---

## 🔑 核心修改说明

### **ResUNet-CBAM 模型**
- 📍 位置: `model/ResUNet/model_ResUNet_CBAM.py`
- 🎯 改进: 在解码器端添加 CBAM 注意力机制
- 📊 参数: 4.1M（基线 3.5M，增长 17%）
- ✅ 验证: 已成功加载并前向传播测试

### **集成到框架**
- 📍 文件: `model/__init__.py` 第 11 行 - 导入 ResUNet_CBAM
- 📍 文件: `net.py` 第 50-51 行 - 注册为 'ResUNet-CBAM'
- ✅ 使用: `Net('ResUNet-CBAM', mode)` 可直接加载

### **代码兼容性修复**
| 文件 | 问题 | 解决 |
|------|------|------|
| `cal_params.py` | 时间戳冒号导致文件名无效 | 添加 `.replace(':', '-')` |
| `net.py` | 坏的导入语句 | 删除第 10 行 |
| `train.py` | 硬编码 .cuda() | 添加设备检测 |
| `metrics.py` | 张量转换错误 | 修复维度处理 |

---

## 📊 第一阶段工作量统计

| 类别 | 数量 | 说明 |
|-----|------|------|
| 新建文件 | 8 | 模型、工具、文档 |
| 修改文件 | 5 | 集成、修复 |
| 代码行数 | ~800 | ResUNet-CBAM 模型代码 |
| 文档行数 | ~1500 | 各类说明文档 |
| 测试运行 | ✅ | 数据加载、模型推理、完整训练流程 |

---

## ⏭️ 第二阶段预告

当对比实验完成后（性能提升 ≥1%），将进行：

### **MAEB 多尺度融合**
- 设计多尺度注意力增强块
- 在 ResUNet-CBAM 基础上添加
- 预期再提升 1-3%

### **伪时空信息**
- 利用多尺度特征间的动态
- 加入 optical flow 或 ConvLSTM
- 预期再提升 1-2%

### **全数据集验证**
- NUAA-SIRST
- IRSTD-1K
- 评估泛化能力

---

## 🔗 文件关系图

```
核心流程：
  数据集 (datasets/NUDT-SIRST/)
    ↓
  train.py
    ↓
  net.py → 加载模型 → model/__init__.py → model_ResUNet_CBAM.py ✨
    ↓
  训练循环
    ↓
  metrics.py (计算 mIoU, PD, FA)
    ↓
  log/ 目录 (自动保存)

分析流程：
  analyze_models.py
    ├─ ResUNet (baseline)
    ├─ ResUNet-CBAM (improved) ✨
    ├─ 参数对比
    └─ 速度测试

文档流程：
  PHASE_1_COMPLETE.md (总体)
    ├─ CURRENT_STATUS.md (快速查看)
    ├─ QUICKSTART.txt (实验操作)
    ├─ COMMIT_GUIDE.py (提交步骤)
    └─ PROJECT_STRUCTURE.md (本文件)
```

---

## ✅ 提交前检查清单

推送到远端前确认：

```bash
# 1. 查看修改状态
git status

# 2. 查看新增文件
git ls-files --others --exclude-standard

# 3. 查看修改内容
git diff

# 4. 检查提交信息
git log -1

# 5. 确认虚拟环境
which python  # 或 Get-Command python

# 6. 确认训练能运行（快速测试）
python train.py --model_names ResUNet-CBAM --dataset_names NUDT-SIRST --batchSize 1 --nEpochs 1
```

---

## 📞 快速参考命令

```bash
# 查看各类文档
cat PHASE_1_COMPLETE.md
cat CURRENT_STATUS.md
cat QUICKSTART.txt

# 运行工具
python analyze_models.py --all
python test_resunet_cbam.py
python COMMIT_GUIDE.py

# 训练命令
python train.py --model_names ResUNet-CBAM --dataset_names NUDT-SIRST --batchSize 8 --nEpochs 20

# Git 操作
git add -A
git commit -m "feat: 阶段一完成"
git push origin main
git log --oneline -5
```

---

**现在你已经了解了整个项目结构！开始工作吧！🚀**
