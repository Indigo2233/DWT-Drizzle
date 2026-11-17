@echo off
rem DWT · Drizzle —— 图形界面启动器
rem 自动探测 CUDA（有就启用 GPU 叠加，没有就自动走 CPU），然后打开界面
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONSTARTUP="
set "PYTHONPATH=%~dp0pydeps;%~dp0;%~dp0core"
set "PYTHONIOENCODING=utf-8"

rem —— 找 CUDA：优先环境变量，其次常见的 Toolkit 安装位置 ——
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvrtc*.dll" goto :cuda_ok
set "CUDA_PATH="
for %%V in (v12.6 v12.5 v12.4 v12.3 v12.2 v12.1 v12.0 v11.8 v11.7 v11.6) do (
    if not defined CUDA_PATH if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%V\bin\nvrtc64_*.dll" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%V"
)
:cuda_ok

start "" "%~dp0py\pythonw.exe" "%~dp0gui.py"
