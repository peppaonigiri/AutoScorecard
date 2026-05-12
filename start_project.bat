@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: ==========================================
:: AutoModeling 一键启动脚本 (Windows)
:: ==========================================

echo [0/3] 正在同步配置文件...
"D:\miniconda3\envs\p_3_8_fb\python.exe" "%~dp0sync_config.py"

echo [1/3] 正在启动后端服务 (FastAPI)...
start "AutoModeling-Backend" cmd /k "cd /d %~dp0backend && D:\miniconda3\envs\p_3_8_fb\python.exe app/main.py"

timeout /t 2 > nul

echo [2/3] 正在启动前端服务 (Vite/React)...
start "AutoModeling-Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

timeout /t 5 > nul

echo [3/3] 就绪...
echo ==========================================
echo 项目启动完成！
echo 1. 后端 API 端口请参考 config.yaml
echo 2. 本机访问地址: http://localhost:5173
echo 3. 局域网访问请使用生成的 IP 地址 (见上文同步日志)
echo ==========================================
pause
