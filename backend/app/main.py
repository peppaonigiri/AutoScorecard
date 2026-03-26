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
from starlette.responses import JSONResponse
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
from app.api import project, dataset, feature, modeling, strategy, auth, users

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(project.router)
app.include_router(dataset.router)
app.include_router(feature.router)
app.include_router(modeling.router)
app.include_router(strategy.router)

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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.main:app', host=SERVER_HOST, port=SERVER_PORT, reload=SERVER_DEBUG)
