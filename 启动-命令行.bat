@echo off
rem DWT · Drizzle —— 命令行启动器
rem 例：启动-命令行.bat --photos "D:\calibrated" --out "D:\out" --drizzle --bgdeg 1
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONSTARTUP="
set "PYTHONPATH=%~dp0pydeps;%~dp0;%~dp0core"
set "PYTHONIOENCODING=utf-8"

if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvrtc*.dll" goto :cuda_ok
set "CUDA_PATH="
for %%V in (v12.6 v12.5 v12.4 v12.3 v12.2 v12.1 v12.0 v11.8 v11.7 v11.6) do (
    if not defined CUDA_PATH if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%V\bin\nvrtc64_*.dll" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%V"
)
:cuda_ok

if "%~1"=="" (
    "%~dp0py\python.exe" "%~dp0main.py" --cli --help
    echo.
    echo 上面是全部参数。把参数接在本 bat 后面即可，例如：
    echo   启动-命令行.bat --photos "D:\calibrated" --out "D:\out" --drizzle --bgdeg 1
    pause
    goto :eof
)
"%~dp0py\python.exe" "%~dp0main.py" --cli --out "%~dp0out" %*
pause
