#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第一阶段提交指南
用于帮助用户正确提交和推送代码到远端

运行此脚本查看：
1. 当前修改的文件
2. 建议的提交步骤
3. 推送命令
"""

import subprocess
import sys

def run_command(cmd, description=""):
    """运行命令并打印输出"""
    if description:
        print(f"\n{'='*70}")
        print(f"📌 {description}")
        print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("⚠️  ", result.stderr)
    return result.returncode == 0

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║              🚀 第一阶段提交与推送指南                                    ║
║                                                                            ║
║  本脚本帮助你：                                                           ║
║  1. 查看当前的文件修改状态                                                ║
║  2. 按步骤提交代码                                                        ║
║  3. 推送到远端 GitHub                                                    ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    # 第一步：查看状态
    run_command("git status", "第 1 步 - 查看当前修改状态")
    
    print("\n" + "="*70)
    print("📋 接下来的手动操作步骤:")
    print("="*70)
    
    print("""
【方式 A】使用自动脚本提交（推荐）

按照以下步骤在命令行中执行：

# 1️⃣ 暂存所有修改和新文件
git add -A

# 2️⃣ 查看将要提交的文件
git status

# 3️⃣ 提交第一个阶段的所有工作
git commit -m "feat: 第一阶段完成 - ResUNet-CBAM 模型和对比框架

- 修复代码兼容性问题（时间戳、坏导入、GPU 支持等）
- 实现 ResUNet-CBAM 改进模型（编码+CBAM注意力）
- 在 Net 类中注册 ResUNet-CBAM 模型
- 新增模型分析工具 analyze_models.py
- 新增实验操作指南 QUICKSTART.txt
- 新增第一阶段完成总结 PHASE_1_COMPLETE.md
- 所有工作已验证，可运行对比实验"

# 4️⃣ 推送到远端
git push origin main

# 5️⃣ 验证推送成功
git log --oneline -3
""")

    print("""
【方式 B】分步提交（更细粒度）

如果希望分开提交代码修复和新功能：

# 第一个 commit：代码修复
git add cal_params.py train.py metrics.py net.py
git commit -m "fix: 修复代码兼容性问题

- 修复 cal_params.py 时间戳冒号问题  
- 删除 net.py 中的坏导入
- 添加 GPU/CPU 自动检测支持
- 修复 metrics.py 张量转换错误"

# 第二个 commit：新功能
git add model/ResUNet/model_ResUNet_CBAM.py model/__init__.py net.py \\
        analyze_models.py test_resunet_cbam.py QUICKSTART.txt \\
        PHASE_1_COMPLETE.md run_test.bat
git commit -m "feat: 实现 ResUNet-CBAM 改进模型

- 新增 ResUNet-CBAM：编码+解码+CBAM注意力
- 集成到 Net 类，支持通过 model_names 加载
- 新增分析工具和实验指南
- 包含第一阶段完整文档"

# 推送所有 commits
git push origin main
""")

    print("""
【提交检查清单】

在提交前，请确认：

✓ 文件状态
  - 新文件（Untracked）应该被 git add
  - 修改文件（Modified）应该被 git add  
  - 无意修改的文件应该被 git restore

✓ 提交信息
  - 第一行是简明扼要的标题（50 字以内）
  - 后面可以跟详细描述（空行分隔）
  - 使用 feat/fix/docs/test 等前缀

✓ 代码质量
  - 确保 python train.py 可以运行
  - 没有明显的 syntax error
  - 新文件中文注释正确编码（UTF-8）

✓ 远端连接
  - GitHub 仓库地址正确
  - 有推送权限（用户名/密码或 SSH key）
  - 网络连接正常
""")

    print("""
【推送后验证】

推送成功后，你可以：

1. 登录 GitHub 网站查看 commits
2. 在新设备上运行：
   git clone <your-repo-url>
   cd BasicIRSTD
   git log --oneline -5  # 查看提交历史
3. 验证所有文件都已同步

【常见问题】

Q: push 被拒绝？
A: 可能是远端有新提交，运行：
   git pull origin main
   git push origin main

Q: 不想提交某个文件？
A: 先 git restore <file>，或在 .gitignore 中添加

Q: 提交信息写错了？
A: 未 push 之前可以修改：
   git commit --amend -m "新的提交信息"

Q: 想回滚某个提交？
A: 查看历史后运行：
   git revert <commit-hash>
   或
   git reset --soft HEAD~1  # 撤销最后一个提交但保留修改
""")

    print("""
【查看提交效果】

执行完 git push 后，可以这样验证：

# 查看本地 commit 历史
git log --oneline --graph --all

# 查看当前分支
git branch -v

# 查看远端状态
git remote -v
git status
""")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"❌ 出错: {e}")
        sys.exit(1)
