# -*- coding: utf-8 -*-
"""强制修复版 FastAPI 主入口 - 杜绝 404"""

import sys
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# 绝对路径加固：确保在各类环境下都能无偏见找到 scorecard_core
_current_dir = os.path.dirname(os.path.abspath(__file__)) # app/
_backend_root = os.path.dirname(_current_dir) # backend/
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import SERVER_HOST, SERVER_PORT, SERVER_DEBUG
from app.database import init_db, get_db
from app.models import Project, Dataset, ModelResult, Deployment, Strategy
from app.schemas import DeploymentRequest, DeploymentResponse, StrategyAnalyzeRequest
from app.api import project, dataset, feature, modeling, strategy, auth, users

# 创建应用
app = FastAPI(title='AutoModeling Platform', version='0.2.0-FIXED')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)




# 终极路由插桩：模型上线专线
@app.post("/api/v1/deploy-model/{project_id}/{model_result_id}")
async def deploy_model_force(project_id: int, model_result_id: int, db: Session = Depends(get_db)):
    """模型上线的物理最终解"""
    print(f"FORCED_DEPLOY: 接收到请求 - 项目 {project_id}, 模型 {model_result_id}")
    try:
        # 进行业务处理
        result = db.query(ModelResult).filter(
            ModelResult.id == model_result_id,
            ModelResult.project_id == project_id
        ).first()
        if not result:
            return {"success": False, "detail": "模型结果不存在"}

        # 检查是否已上线
        existing = db.query(Deployment).filter(
            Deployment.model_result_id == model_result_id,
            Deployment.status == 'active'
        ).first()
        if existing:
            return {"success": True, "id": existing.id, "msg": "already_active"}

        deployment = Deployment(
            project_id=project_id,
            model_result_id=model_result_id,
            status='active'
        )
        db.add(deployment)
        db.commit()
        db.refresh(deployment)
        
        return {
            "success": True, 
            "id": deployment.id,
            "deployed_at": deployment.deployed_at.isoformat() if deployment.deployed_at else None
        }
    except Exception as e:
        import logging
        logging.exception(f"插桩逻辑崩溃: {e}")
        return {"success": False, "error": str(e)}

# 注册原有子模块路由（此时由主程序先尝试匹配）
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
    logging.info('数据库表已创建，系统路由已锚定')
    
    # 初始化 root 管理员
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        from app.models import User
        from app.api.auth import get_password_hash
        root_user = db.query(User).filter(User.username == 'root').first()
        if not root_user:
            root_user = User(
                username='root',
                hashed_password=get_password_hash('root'),
                is_admin=1
            )
            db.add(root_user)
            db.commit()
            logging.info('默认 root 管理员已创建')
    except Exception as e:
        logging.error(f"初始化 root 失败: {e}")
    finally:
        db.close()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.main:app', host=SERVER_HOST, port=SERVER_PORT, reload=SERVER_DEBUG)
