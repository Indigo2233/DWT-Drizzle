@echo off
chcp 936 >nul
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONSTARTUP="
set "PYTHONPATH=%~dp0pydeps;%~dp0;%~dp0core"
echo ============================================
echo   DWT - Drizzle   GPU 加速 检测
echo ============================================
echo.
if exist "py\python.exe" goto :have_py
echo [X] 找不到 py\python.exe，请确认压缩包完整解压。
pause
exit /b 1
:have_py
echo [1/2] 跑一次真实内核自检（不需要 CUDA Toolkit）...
py\python.exe -B "%~dp0tools\gpu_check.py"
if errorlevel 1 goto :fail
echo.
echo [2/2] 结论：GPU 加速可用。点「开始叠加」时会自动优先用 GPU。
echo.
pause
exit /b 0
:fail
echo.
echo [!] GPU 自检失败。常见原因与处理：
echo     1) 没有 NVIDIA 显卡 -^> 用 CPU 叠加即可，功能完整、只是慢；
echo     2) 显卡驱动太旧 -^> 到 nvidia.cn 更新驱动；本软件只用驱动，不需要装 CUDA Toolkit；
echo     3) 显存被别的程序占满 -^> 关掉浏览器/游戏再试；
echo     4) 缺 VC++ 运行库 -^> 装 Microsoft Visual C++ 2015-2022 x64；
echo     5) 笔记本双显卡被禁用 -^> 设备管理器里确认 NVIDIA 显卡没有被停用。
echo.
pause
exit /b 1
