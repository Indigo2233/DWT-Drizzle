@echo off
chcp 936 >nul
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONSTARTUP="
set "PYTHONPATH=%~dp0pydeps;%~dp0;%~dp0core"
echo ============================================
echo   DWT - Drizzle   GPU 加速 检测 / 修复
echo ============================================
echo.
if exist "py\python.exe" goto :have_py
echo [X] 找不到 py\python.exe，请确认压缩包完整解压到本地目录。
pause
exit /b 1
:have_py
echo [1/3] 查找 CUDA 运行库（nvrtc）...
set "CUDA_PATH="
for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12*") do if exist "%%D\bin\nvrtc64_*.dll" set "CUDA_PATH=%%D"
if not defined CUDA_PATH for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11*") do if exist "%%D\bin\nvrtc64_*.dll" set "CUDA_PATH=%%D"
if defined CUDA_PATH (echo    找到：%CUDA_PATH%) else (echo    没找到 CUDA Toolkit；若显卡驱动是新版，常见于只装了驱动没装 CUDA 工具箱。)
echo.
echo [2/3] 试跑 CuPy 自检...
py\python.exe -B "%~dp0tools\gpu_check.py"
if errorlevel 1 goto :fail
echo.
echo [3/3] 结论：GPU 加速可用。叠加时“计算设备”选“自动”或“只用 GPU”即可。
echo.
pause
exit /b 0
:fail
echo.
echo [!] CuPy 自检失败。常见原因与处理：
echo     1) 没有 NVIDIA 显卡 -> 用 CPU 叠加即可，功能不受影响；
echo     2) 驱动太旧 -> 到 nvidia.cn 更新显卡驱动（会自动带上 CUDA 运行库）；
echo     3) 装了 CUDA 但版本不对 -> 本包内置 cupy 对应 CUDA 11.x/12.x 运行库；
echo     4) 缺 VC++ 运行库 -> 安装 Microsoft Visual C++ 2015-2022 x64。
echo.
pause
exit /b 1
