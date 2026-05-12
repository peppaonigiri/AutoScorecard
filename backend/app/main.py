# -*- coding: utf-8 -*-
"""AutoModeling FastAPI 主入口"""

import sys
import os
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

# 绝对路径加固
_current_dir = os.path.dirname(os.path.abspath(__file__)) 
_backend_root = os.path.dirname(_current_dir) 
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

# 【重要修复】强制使用非交互式后端，防止后台线程绘图时触发 tkinter 崩溃
import matplotlib
matplotlib.use('Agg')

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session

from app.config import SERVER_HOST, SERVER_PORT, SERVER_DEBUG
from app.database import init_db, get_db

# 先创建应用对象，再做其他事情
app = FastAPI(title='AutoModeling Platform', version='0.3.0')

# 基础中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# 延迟导入 API 模块，防止循环引用
from app.api import project, dataset, feature, modeling, strategy, auth, users, agent

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(project.router)
app.include_router(dataset.router)
app.include_router(feature.router)
app.include_router(modeling.router)
app.include_router(strategy.router)
app.include_router(agent.router)    # LLM Agent

@app.on_event('startup')
def startup():
    init_db()
    logging.info('Application startup: Database and Routes ready.')
    
    from app.database import SessionLocal
    from app.models import User
    from app.api.auth import get_password_hash
    
    db = SessionLocal()
    try:
        root_user = db.query(User).filter(User.username == 'root').first()
        if not root_user:
            root_user = User(
                username='root',
                hashed_password=get_password_hash('root'),
                is_admin=1
            )
            db.add(root_user)
            db.commit()
            logging.info('Root user initialized.')
    except Exception as e:
        logging.error(f"Startup init error: {e}")
    finally:
        db.close()

# --- 前端静态文件服务 (生产环境) ---
# 获取前端编译目录的路径
_project_root = os.path.dirname(_backend_root)
frontend_dist = os.path.join(_project_root, 'frontend', 'dist')

# 挂载 /assets 目录 (如果存在)
assets_dir = os.path.join(frontend_dist, "assets")
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# 捕获所有其他请求，返回 index.html (支持 SPA 路由)
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # 排除 API 路径
    if full_path.startswith("api/") or full_path.startswith("v1/"):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    
    if not os.path.exists(frontend_dist):
        return JSONResponse(
            status_code=500, 
            content={
                "error": "前端未编译", 
                "message": f"未找到前端编译目录 {frontend_dist}。请在服务器执行: cd {os.path.dirname(frontend_dist)} && npm install && npm run build"
            }
        )
    
    # 检查请求的文件是否存在（如 favicon.ico 等）
    file_path = os.path.join(frontend_dist, full_path)
    if full_path and os.path.isfile(file_path):
        return FileResponse(file_path)
        
    # 默认返回 index.html
    index_path = os.path.join(frontend_dist, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return JSONResponse(status_code=404, content={"error": "index.html 不存在", "path": index_path})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.main:app', host=SERVER_HOST, port=SERVER_PORT, reload=SERVER_DEBUG)
