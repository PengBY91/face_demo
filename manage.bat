@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if "%1"=="" set "1=start"
if /i "%1"=="start" goto :start
if /i "%1"=="stop" goto :stop
if /i "%1"=="restart" goto :restart
if /i "%1"=="status" goto :status
goto :show_usage

:show_menu
echo ====================================
echo    Face Demo 管理脚本
echo ====================================
echo.
echo 请选择操作：
echo   1. 启动服务 (Start)
echo   2. 停止服务 (Stop)
echo   3. 重启服务 (Restart)
echo   4. 查看状态 (Status)
echo   5. 退出
echo.
set /p choice="请输入选项 (1-5): "

if "%choice%"=="1" goto :start
if "%choice%"=="2" goto :stop
if "%choice%"=="3" goto :restart
if "%choice%"=="4" goto :status
if "%choice%"=="5" exit /b 0
echo 无效选项！
pause
goto :show_menu

:show_usage
echo 用法: manage.bat [start^|stop^|restart^|status]
echo.
echo 命令:
echo   start    - 启动服务
echo   stop     - 停止服务
echo   restart  - 重启服务
echo   status   - 查看服务状态
exit /b 1

:start
echo [启动服务]
echo.

echo [1/3] 检查服务是否已运行...
call :check_running
if !SERVER_RUNNING!==1 (
    echo × Server 已在运行 ^(PID: !SERVER_PID!^)
) else (
    echo √ Server 未运行
)
if !DEMO_RUNNING!==1 (
    echo × Demo 已在运行 ^(PID: !DEMO_PID!^)
) else (
    echo √ Demo 未运行
)

if !SERVER_RUNNING!==1 if !DEMO_RUNNING!==1 (
    echo.
    echo 服务已在运行，如需重启请使用 restart 命令
    goto :end
)

echo.
echo [2/3] 创建日志目录...
if not exist logs mkdir logs

echo [3/3] 启动服务...
call conda activate face

if !SERVER_RUNNING!==0 (
    echo   启动 Server.py...
    start /b "" cmd /c "call conda activate face && python server.py > logs\server.log 2>&1"
    timeout /t 2 /nobreak >nul
    echo   √ Server.py 已启动
)

if !DEMO_RUNNING!==0 (
    echo   启动 Demo.py...
    start /b "" cmd /c "call conda activate face && python demo.py > logs\demo.log 2>&1"
    timeout /t 2 /nobreak >nul
    echo   √ Demo.py 已启动
)

echo.
echo ====================================
echo    启动完成！
echo    日志文件：
echo      - logs\server.log
echo      - logs\demo.log
echo ====================================
goto :end

:stop
echo [停止服务]
echo.

echo [1/2] 检查运行状态...
call :check_running

if !SERVER_RUNNING!==0 if !DEMO_RUNNING!==0 (
    echo × 没有服务在运行
    goto :end
)

echo [2/2] 停止服务...
if !SERVER_RUNNING!==1 (
    echo   停止 Server.py ^(PID: !SERVER_PID!^)...
    taskkill /F /PID !SERVER_PID! >nul 2>&1
    if !errorlevel!==0 (
        echo   √ Server.py 已停止
    ) else (
        echo   × Server.py 停止失败
    )
)

if !DEMO_RUNNING!==1 (
    echo   停止 Demo.py ^(PID: !DEMO_PID!^)...
    taskkill /F /PID !DEMO_PID! >nul 2>&1
    if !errorlevel!==0 (
        echo   √ Demo.py 已停止
    ) else (
        echo   × Demo.py 停止失败
    )
)

echo.
echo ====================================
echo    服务已停止
echo ====================================
goto :end

:restart
echo [重启服务]
echo.
call :stop
echo.
timeout /t 2 /nobreak >nul
call :start
goto :end

:status
echo [服务状态]
echo.
call :check_running

echo Server.py:
if !SERVER_RUNNING!==1 (
    echo   状态: 运行中 ✓
    echo   PID:  !SERVER_PID!
) else (
    echo   状态: 未运行 ×
)

echo.
echo Demo.py:
if !DEMO_RUNNING!==1 (
    echo   状态: 运行中 ✓
    echo   PID:  !DEMO_PID!
) else (
    echo   状态: 未运行 ×
)

echo.
echo 日志文件:
if exist logs\server.log (
    for %%A in ("logs\server.log") do echo   Server: logs\server.log ^(%%~zA bytes^)
) else (
    echo   Server: 无日志文件
)

if exist logs\demo.log (
    for %%A in ("logs\demo.log") do echo   Demo:   logs\demo.log ^(%%~zA bytes^)
) else (
    echo   Demo: 无日志文件
)

goto :end

:check_running
set SERVER_RUNNING=0
set DEMO_RUNNING=0
set SERVER_PID=
set DEMO_PID=

for /f "tokens=2" %%i in ('wmic process where "commandline like '%%server.py%%' and name='python.exe'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    set SERVER_RUNNING=1
    set SERVER_PID=%%i
)

for /f "tokens=2" %%i in ('wmic process where "commandline like '%%demo.py%%' and name='python.exe'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    set DEMO_RUNNING=1
    set DEMO_PID=%%i
)
exit /b

:end
echo.
if "%1"=="" (
    pause
    goto :show_menu
)
exit /b 0
