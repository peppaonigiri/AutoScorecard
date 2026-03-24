# -*- coding: utf-8 -*-
"""项目管理 API"""

import os
import shutil
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from app.database import get_db
from app.config import UPLOAD_DIR
from app.models import Project, Dataset, ModelResult, Deployment
from app.schemas import (
    ProjectCreate, ProjectResponse, ProjectListResponse, DatasetResponse,
    DeploymentRequest, DeploymentResponse
)

router = APIRouter(prefix='/api/projects', tags=['项目管理'])


@router.post('', response_model=ProjectResponse)
def create_project(req: ProjectCreate, db: Session = Depends(get_db)):
    project = Project(name=req.name, description=req.description or '')
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get('', response_model=ProjectListResponse)
def list_projects(db: Session = Depends(get_db)):
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    return ProjectListResponse(total=len(projects), items=projects)


@router.get('/{project_id}', response_model=ProjectResponse)
def get_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')
    return project


@router.get('/{project_id}/datasets', response_model=List[DatasetResponse])
def list_datasets(project_id: int, db: Session = Depends(get_db)):
    """获取项目下的所有数据集"""
    datasets = db.query(Dataset).filter(Dataset.project_id == project_id).all()
    return datasets


@router.post('/{project_id}/datasets', response_model=DatasetResponse)
def upload_dataset(project_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 检查项目
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    # 保存文件
    project_dir = os.path.join(UPLOAD_DIR, str(project_id))
    os.makedirs(project_dir, exist_ok=True)
    file_path = os.path.join(project_dir, file.filename)

    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    # 读取并计算统计信息
    try:
        if file.filename.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f'文件读取失败: {str(e)}')

    from scorecard_core.data_processor import calculate_dataset_summary
    stats, l1_res = calculate_dataset_summary(df)

    dataset = Dataset(
        project_id=project_id,
        name=file.filename,
        file_path=file_path,
        file_size=os.path.getsize(file_path),
        n_rows=len(df),
        n_cols=len(df.columns),
        columns_info=stats['dtypes'],
        stats_cache=stats,
        l1_results=l1_res
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.delete('/{project_id}')
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')
    db.delete(project)
    db.commit()
    return {'message': '已删除'}

@router.post('/{project_id}/deploy')
def deploy_model(project_id: int, req: DeploymentRequest, db: Session = Depends(get_db)):
    """上线模型（移除强类型校验以排查序列化故障）"""
    print(f"DEBUG: 接收到上线请求 - 项目 ID: {project_id}, 模型 ID: {req.model_result_id}")
    try:
        result = db.query(ModelResult).filter(
            ModelResult.id == req.model_result_id,
            ModelResult.project_id == project_id
        ).first()
        if not result:
            print("DEBUG: 未找到模型结果")
            raise HTTPException(status_code=404, detail='模型结果不存在或不属于该项目')
        
        existing = db.query(Deployment).filter(
            Deployment.model_result_id == req.model_result_id,
            Deployment.status == 'active'
        ).first()
        if existing:
            print(f"DEBUG: 模型已上线 ID: {existing.id}")
            # 手动转换为字典
            return {
                "id": existing.id,
                "project_id": existing.project_id,
                "model_result_id": existing.model_result_id,
                "status": existing.status,
                "deployed_at": existing.deployed_at.isoformat() if existing.deployed_at else None
            }
        
        print("DEBUG: 正在创建新部署...")
        deployment = Deployment(
            project_id=project_id,
            model_result_id=result.id,
            status='active'
        )
        db.add(deployment)
        db.commit()
        db.refresh(deployment)
        print(f"DEBUG: 部署创建成功 ID: {deployment.id}")
        
        return {
            "id": deployment.id,
            "project_id": deployment.project_id,
            "model_result_id": deployment.model_result_id,
            "status": deployment.status,
            "deployed_at": deployment.deployed_at.isoformat() if deployment.deployed_at else None
        }
    except Exception as e:
        import logging
        logging.exception(f"后端执行上线逻辑崩溃: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.get('/{project_id}/deployments')
def list_deployments(project_id: int, db: Session = Depends(get_db)):
    """获取项目下的部署列表"""
    from sqlalchemy import case
    deployments = db.query(Deployment).filter(Deployment.project_id == project_id)\
        .order_by(case({"active": 1, "inactive": 2}, value=Deployment.status, else_=3))\
        .order_by(Deployment.deployed_at.desc()).all()
    # 手动转换为 dict 以避免 Pydantic 响应模型中的日期序列化问题 (如有)
    return [{
        "id": d.id,
        "project_id": d.project_id,
        "model_result_id": d.model_result_id,
        "status": d.status,
        "deployed_at": d.deployed_at.isoformat() if d.deployed_at else None
    } for d in deployments]
