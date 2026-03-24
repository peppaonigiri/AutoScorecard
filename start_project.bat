@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: ==========================================
:: AutoModeling 一键启动脚本 (Windows)
:: ==========================================

echo [1/3] 正在启动后端服务 (FastAPI)...
start "AutoModeling-Backend" cmd /k "cd /d %~dp0backend && D:\miniconda3\envs\p_3_8_fb\python.exe app/main.py"

timeout /t 2 > nul

echo [2/3] 正在启动前端服务 (Vite/React)...
start "AutoModeling-Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

timeout /t 5 > nul

echo [3/3] 正在打开浏览器...
start http://localhost:5173

echo ==========================================
echo 项目启动完成！
echo 后端地址: http://localhost:8000
echo 前端地址: http://localhost:5173
echo ==========================================
pause
