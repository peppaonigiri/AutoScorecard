#!/bin/bash

# ==========================================
# AutoModeling 一键启动脚本 (Linux)
# ==========================================

# 设置工作目录为脚本所在目录
BASE_DIR=$(cd "$(dirname "$0")"; pwd)
cd "$BASE_DIR"

# 定义 Python 环境路径（根据实际环境修改，建议使用虚拟环境绝对路径）
PYTHON_EXE="/opt/miniconda3/envs/scorecard/bin/python"
# 如果使用系统的 python3，请确保已安装依赖：PYTHON_EXE=$(which python3)

echo "[0/3] 正在同步配置文件..."
$PYTHON_EXE "$BASE_DIR/sync_config.py"

echo "[1/3] 正在启动后端服务 (FastAPI)..."
# 使用 nohup 后台运行，并将日志输出到 backend.log
cd "$BASE_DIR/backend"
nohup $PYTHON_EXE app/main.py > "$BASE_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

sleep 2

echo "[2/3] 正在启动前端服务 (Vite/React)..."
# 使用 nohup 后台运行，并将日志输出到 frontend.log
cd "$BASE_DIR/frontend"
nohup npm run dev > "$BASE_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

sleep 5

echo "=========================================="
echo "项目启动完成！"
echo "1. 后端 PID: $BACKEND_PID (日志: backend.log)"
echo "2. 前端 PID: $FRONTEND_PID (日志: frontend.log)"
echo "3. 本机访问地址: http://localhost:5173"
echo "=========================================="
echo "提示：使用 'kill $BACKEND_PID $FRONTEND_PID' 停止服务"