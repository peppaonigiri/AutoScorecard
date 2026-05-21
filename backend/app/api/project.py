# -*- coding: utf-8 -*-
"""项目管理 API"""

import os
import shutil
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.database import get_db
from app.config import UPLOAD_DIR
from app.models import Project, Dataset, ModelResult, Deployment, User
from app.schemas import (
    ProjectCreate, ProjectResponse, ProjectListResponse, DatasetResponse,
    DeploymentRequest, DeploymentResponse, ProjectVisibilityUpdate, ProjectExcludeColsUpdate
)
from app.api.auth import get_current_user

router = APIRouter(prefix='/api/projects', tags=['项目管理'])

def check_project_access(project_id: int, db: Session, user: User, need_write: bool = False):
    """通用项目权限检查 (包含对 Project 表的查询)"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')
    
    # 获取公开状态/所有者
    is_public = project.is_public == 1
    is_owner = project.owner_id == user.id
    is_admin = user.is_admin == 1

    if need_write:
        # 写操作：仅限所有者或管理员
        if not is_owner and not is_admin:
            raise HTTPException(status_code=403, detail='无权修改该项目')
    else:
        # 读操作：公开项目全员可读，私有项目仅限所有者和管理员
        if not is_public and not is_owner and not is_admin:
            raise HTTPException(status_code=403, detail='无权访问该私有项目')
    return project


@router.post('', response_model=ProjectResponse)
def create_project(req: ProjectCreate, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    project = Project(
        name=req.name, 
        description=req.description or '',
        owner_id=current_user.id
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    project.owner_name = current_user.username
    return project


@router.get('', response_model=ProjectListResponse)
def list_projects(db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    query = db.query(Project)
    
    # 如果不是管理员，只显示自己拥有的或公开的项目
    if current_user.is_admin != 1:
        from sqlalchemy import or_
        query = query.filter(or_(Project.owner_id == current_user.id, Project.is_public == 1))
        
    projects = query.order_by(Project.created_at.desc()).all()
    
    # 手动附加 owner_name 用于展示
    for p in projects:
        p.owner_name = p.owner.username if p.owner else "系统"
        
    return ProjectListResponse(total=len(projects), items=projects)


@router.get('/{project_id}', response_model=ProjectResponse)
def get_project(project_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')
        
    # 权限检查：私有项目且不是所有者且不是管理员，禁止访问
    if not project.is_public and project.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail='无权访问该私有项目')
        
    project.owner_name = project.owner.username if project.owner else "系统"
    return project


@router.put('/{project_id}/visibility', response_model=ProjectResponse)
def update_visibility(project_id: int, req: ProjectVisibilityUpdate, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    """修改项目公开状态"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')
        
    # 只有所有者或管理员可以修改可见性
    if project.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail='无权修改该项目')
        
    project.is_public = req.is_public
    db.add(project)
    db.commit()
    db.refresh(project)
    project.owner_name = project.owner.username if project.owner else "系统"
    return project


@router.put('/{project_id}/exclude-cols', response_model=ProjectResponse)
def update_exclude_cols(project_id: int, req: ProjectExcludeColsUpdate, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    """更新项目的全局排除列"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')
        
    # 只有所有者或管理员可以修改
    if project.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail='无权修改该项目')
        
    project.exclude_cols = req.exclude_cols
    db.add(project)
    db.commit()
    db.refresh(project)
    project.owner_name = project.owner.username if project.owner else "系统"
    return project


@router.get('/{project_id}/datasets', response_model=List[DatasetResponse])
def list_datasets(project_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    """获取项目下的所有数据集"""
    check_project_access(project_id, db, current_user, need_write=False)
    datasets = db.query(Dataset).filter(Dataset.project_id == project_id).all()
    return datasets


@router.post('/{project_id}/datasets', response_model=DatasetResponse)
def upload_dataset(project_id: int, file: UploadFile = File(...), db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    # 检查项目权限
    project = check_project_access(project_id, db, current_user, need_write=True)

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
def delete_project(project_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    """删除项目及其所有关联的物理文件"""
    from app.models import ModelReport, MonitoringLog, StrategyMonitoringLog, Strategy

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')
        
    # 只有所有者或管理员可以删除
    if project.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail='无权删除该项目')

    # 清理物理文件：模型报告 (.xlsx)
    for report in db.query(ModelReport).filter(ModelReport.project_id == project_id).all():
        if report.file_path and os.path.exists(report.file_path):
            try: os.remove(report.file_path)
            except Exception: pass

    # 清理物理文件：模型 (.pkl)
    for mr in db.query(ModelResult).filter(ModelResult.project_id == project_id).all():
        if mr.model_path and os.path.exists(mr.model_path):
            try: os.remove(mr.model_path)
            except Exception: pass

    # 清理物理文件：数据集 (csv/parquet，含派生填充文件)
    for ds in db.query(Dataset).filter(Dataset.project_id == project_id).all():
        if ds.file_path and os.path.exists(ds.file_path):
            try: os.remove(ds.file_path)
            except Exception: pass

    # 删除项目上传目录
    project_dir = os.path.join(UPLOAD_DIR, str(project_id))
    if os.path.isdir(project_dir):
        shutil.rmtree(project_dir, ignore_errors=True)

    # ---------------------------------------------------------------
    # 先删除引用 model_results 的子表记录，避免 FK 约束冲突：
    #   deployments.model_result_id -> model_results.id
    #   model_reports.model_result_id -> model_results.id
    # ORM 的 cascade='all, delete-orphan' 只覆盖 Project 的直接子表，
    # 不会递归处理这两张"跨越" model_results 的引用表。
    # ---------------------------------------------------------------
    # 1) 先删 monitoring_logs（依赖 deployments）
    from sqlalchemy import delete as sa_delete
    mr_ids = [r.id for r in db.query(ModelResult.id).filter(ModelResult.project_id == project_id).all()]
    if mr_ids:
        dep_ids = [d.id for d in db.query(Deployment.id).filter(
            Deployment.model_result_id.in_(mr_ids)).all()]
        if dep_ids:
            db.query(MonitoringLog).filter(MonitoringLog.deployment_id.in_(dep_ids)).delete(
                synchronize_session=False)
        # 2) 删 deployments
        db.query(Deployment).filter(Deployment.model_result_id.in_(mr_ids)).delete(
            synchronize_session=False)
        # 3) 删 model_reports
        db.query(ModelReport).filter(ModelReport.model_result_id.in_(mr_ids)).delete(
            synchronize_session=False)

    # ORM 级联删除 Project（会级联删除 model_results、datasets、tasks 等直接子表）
    db.delete(project)
    db.commit()
    return {'message': '已删除'}

@router.post('/{project_id}/deploy')
def deploy_model(project_id: int, req: DeploymentRequest, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    """上线模型"""
    # 权限检查
    check_project_access(project_id, db, current_user, need_write=True)
    try:
        result = db.query(ModelResult).filter(
            ModelResult.id == req.model_result_id,
            ModelResult.project_id == project_id
        ).first()
        if not result:
            raise HTTPException(status_code=404, detail='模型结果不存在或不属于该项目')
        
        existing = db.query(Deployment).filter(
            Deployment.model_result_id == req.model_result_id,
            Deployment.status == 'active'
        ).first()
        if existing:
            return {
                "id": existing.id,
                "project_id": existing.project_id,
                "model_result_id": existing.model_result_id,
                "status": existing.status,
                "deployed_at": existing.deployed_at.isoformat() if existing.deployed_at else None
            }
        
        deployment = Deployment(
            project_id=project_id,
            model_result_id=result.id,
            status='active'
        )
        db.add(deployment)
        db.commit()
        db.refresh(deployment)
        
        
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
def list_deployments(project_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    # 权限检查
    check_project_access(project_id, db, current_user, need_write=False)
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
