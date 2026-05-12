@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: ==========================================
:: AutoModeling 一键启动脚本 (Windows)
:: ==========================================

echo [0/3] 正在同步配置文件...
"D:\miniconda3\envs\p_3_8_fb\python.exe" "%~dp0sync_config.py"

echo [1/2] 正在检查前端编译状态...
if not exist "%~dp0frontend\dist" (
    echo 未发现编译目录 dist，正在执行编译 (npm run build)...
    cd /d "%~dp0frontend" && npm install && npm run build
) else (
    echo 发现已存在的 dist 目录，将使用生产环境包。
)

echo [2/2] 正在启动后端服务 (FastAPI 托管前端)...
start "AutoModeling-Backend" cmd /k "cd /d %~dp0backend && D:\miniconda3\envs\p_3_8_fb\python.exe app/main.py"

timeout /t 2 > nul

echo [3/3] 就绪...
echo ==========================================
echo 项目启动完成！
echo 1. 后端 API 端口请参考 config.yaml
echo 2. 本机访问地址 (由后端托管): http://localhost:8081
echo 3. 局域网访问请使用生成的 IP 地址 (见上文同步日志)
echo ==========================================
pause

