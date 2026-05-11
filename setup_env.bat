@echo off
REM ============================================================
REM BasicIRSTD 虚拟环境自动化设置脚本 (Windows)
REM ============================================================
setlocal enabledelayedexpansion

echo.
echo ===== BasicIRSTD 环境设置 =====
echo.

REM 检查 Python 是否存在
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 未在 PATH 中找到，请先安装 Python 3.7+
    pause
    exit /b 1
)

echo [✓] Python 已安装
python --version

REM 创建虚拟环境
echo.
echo [*] 正在创建虚拟环境 'venv'...
if exist venv (
    echo [!] venv 已存在，跳过创建
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] 创建虚拟环境失败
        pause
        exit /b 1
    )
    echo [✓] 虚拟环境创建成功
)

REM 激活虚拟环境
echo.
echo [*] 激活虚拟环境...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] 激活虚拟环境失败
    pause
    exit /b 1
)
echo [✓] 虚拟环境已激活

REM 升级 pip
echo.
echo [*] 升级 pip...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [WARNING] pip 升级失败，继续...
)

REM 安装 PyTorch (CPU 版本)
echo.
echo [*] 安装 PyTorch (CPU 版本)...
echo [Note] 如需 GPU 支持，请使用 GPU 版 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo [ERROR] PyTorch 安装失败
    pause
    exit /b 1
)
echo [✓] PyTorch 已安装

REM 安装其他依赖
echo.
echo [*] 安装其他依赖库...
pip install numpy scipy scikit-image pillow matplotlib tqdm opencv-python
if errorlevel 1 (
    echo [WARNING] 某些依赖库安装失败，继续...
)
echo [✓] 依赖库安装完成

REM 验证安装
echo.
echo [*] 验证安装...
python -c "import torch; import numpy as np; import cv2; import matplotlib; import tqdm; print('[✓] 所有关键库导入成功'); print('PyTorch version:', torch.__version__)"
if errorlevel 1 (
    echo [ERROR] 验证失败
    pause
    exit /b 1
)

echo.
echo ===== 安装完成 =====
echo.
echo [注意] 下次使用前，请先运行以下命令激活虚拟环境：
echo   Windows CMD: venv\Scripts\activate.bat
echo   Windows PowerShell: venv\Scripts\Activate.ps1
echo.
echo [可选] 如需 GPU 支持，请手动安装 CUDA 版 PyTorch：
echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo   或访问 https://pytorch.org 选择合适版本
echo.
pause
