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

echo "[1/2] 正在检查前端编译状态..."
cd "$BASE_DIR/frontend"
if [ ! -d "dist" ]; then
    echo "未发现编译目录 dist，正在执行编译 (npm run build)..."
    npm install && npm run build
else
    echo "发现已存在的 dist 目录，将使用生产环境包。"
fi

echo "[2/2] 正在启动后端服务 (FastAPI 托管前端)..."
# 使用 nohup 后台运行，并将日志输出到 backend.log
cd "$BASE_DIR/backend"
nohup $PYTHON_EXE app/main.py > "$BASE_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

sleep 2

echo "=========================================="
echo "项目启动完成！"
echo "1. 后端 PID: $BACKEND_PID (日志: backend.log)"
echo "2. 前端访问地址 (由后端托管): http://localhost:8081"
echo "   (具体端口请参考 config.yaml 中的后端配置)"
echo "=========================================="
echo "提示：使用 'kill $BACKEND_PID' 停止服务"