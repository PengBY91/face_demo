@echo off
chcp 65001 >nul
title Face Demo - 单路摄像头人脸识别

REM ========================================
REM 配置区域 - 根据需要修改
REM ========================================
set CONDA_ENV_NAME=face
set PROJECT_DIR=%~dp0

REM ========================================
REM 初始化 Conda 环境
REM ========================================
echo 正在初始化 Conda 环境...

REM 尝试调用 conda hook
call conda activate base >nul 2>&1
if errorlevel 1 (
    REM 如果直接调用失败，尝试通过 conda 初始化脚本
    if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
        call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    ) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
        call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
    ) else if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
        call "C:\ProgramData\anaconda3\Scripts\activate.bat"
    ) else if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
        call "C:\ProgramData\miniconda3\Scripts\activate.bat"
    ) else (
        echo [错误] 未找到 Conda 安装路径，请手动配置
        pause
        exit /b 1
    )
)

REM ========================================
REM 激活项目环境
REM ========================================
echo 正在激活环境: %CONDA_ENV_NAME%
call conda activate %CONDA_ENV_NAME%
if errorlevel 1 (
    echo [错误] 无法激活环境 "%CONDA_ENV_NAME%"
    echo 请确保已创建该环境: conda create -n %CONDA_ENV_NAME% python=3.10
    pause
    exit /b 1
)

REM ========================================
REM 切换到项目目录并运行
REM ========================================
cd /d "%PROJECT_DIR%"
echo.
echo ========================================
echo  启动 demo.py - 单路摄像头人脸识别
echo ========================================
echo.

python demo.py

REM ========================================
REM 结束处理
REM ========================================
echo.
echo 程序已退出
pause
