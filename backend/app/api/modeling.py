# -*- coding: utf-8 -*-
"""建模 API"""

import sys
import os
from datetime import datetime # Added for simulate_monitor
import numpy as np # Added for simulate_monitor

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.config import MODEL_DIR
from app.models import Project, Dataset, Task, ModelResult, Deployment, MonitoringLog, Strategy, StrategyMonitoringLog, ModelReport
from app.schemas import (
    ModelingRequest, ModelingResponse, TaskResponse,
    ModelResultResponse, DeploymentRequest, DeploymentResponse,
    SimulationRequest, MonitoringLogResponse
)
from app.task_manager import submit_task

from scorecard_core.monitor_engine import simulate_business_intake, run_monitoring_task, run_strategy_monitoring
from scorecard_core.data_processor import load_data

_current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

router = APIRouter(prefix='/api', tags=['建模'])


@router.post('/projects/{project_id}/modeling/train', response_model=ModelingResponse)
async def start_modeling(project_id: int, req: ModelingRequest, db: Session = Depends(get_db)):
    """启动建模任务（异步）"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    dataset = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail='数据集不存在')

    # 创建任务
    task = Task(
        project_id=project_id,
        task_type='modeling',
        status='pending',
        params={
            'dataset_id': req.dataset_id,
            'dep': req.dep,
            'model_type': req.model_type,
            'strategy_type': req.strategy_type,
            'n_trials': req.n_trials,
            'max_depth': req.max_depth,
            'strategy_threshold': req.strategy_threshold,
            'feature_list': req.feature_list or project.feature_list,
            'exclude_cols': req.exclude_cols,
        }
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # 提交异步任务
    from scorecard_core.model_trainer import run_optuna_training
    model_save_dir = os.path.join(MODEL_DIR, str(project_id))

    await submit_task(
        task_id=task.id,
        func=run_optuna_training,
        data_path=dataset.file_path,
        exclude_cols=req.exclude_cols,
        feature_list=req.feature_list,
        dep=req.dep,
        model_type=req.model_type,
        strategy_type=req.strategy_type,
        strategy_threshold=req.strategy_threshold,
        n_trials=req.n_trials,
        max_depth=req.max_depth,
        project_id=project_id,
        model_save_dir=model_save_dir,
        split_ratios=req.split_ratios or project.split_config.get('split_ratios'),
        oot_col=req.oot_col or project.split_config.get('oot_col'),
        oot_start_time=req.oot_start_time or project.split_config.get('oot_start_time'),
        score_config=req.score_config.dict() if req.score_config else None,
    )

    return ModelingResponse(task_id=task.id, status='pending')


@router.get('/tasks/{task_id}', response_model=TaskResponse)
def get_task(task_id: int, db: Session = Depends(get_db)):
    """查询任务状态"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail='任务不存在')
    return task

@router.post('/tasks/{task_id}/heartbeat')
def update_task_heartbeat(task_id: int, db: Session = Depends(get_db)):
    """更新任务心跳时间"""
    from app.models import Task
    task = db.query(Task).filter(Task.id == task_id).first()
    if task:
        task.last_heartbeat = func.now()
        db.commit()
    return {"status": "ok"}


@router.get('/projects/{project_id}/tasks')
def list_tasks(project_id: int, db: Session = Depends(get_db)):
    """列出项目下所有任务"""
    tasks = db.query(Task).filter(Task.project_id == project_id).order_by(Task.created_at.desc()).all()
    return {'total': len(tasks), 'items': [TaskResponse.from_orm(t) for t in tasks]}


@router.get('/projects/{project_id}/results', response_model=Dict[str, Any])
def list_results(project_id: int, db: Session = Depends(get_db)):
    """列出项目下所有模型结果"""
    try:
        results = db.query(ModelResult).filter(
            ModelResult.project_id == project_id
        ).order_by(ModelResult.created_at.desc()).all()
        # 获取活跃部署 ID
        active_dep = db.query(Deployment).filter(
            Deployment.project_id == project_id,
            Deployment.status == 'active'
        ).first()
        active_result_id = active_dep.model_result_id if active_dep else None

        items = []
        for r in results:
            item = {
                'id': r.id,
                'model_type': r.model_type,
                'params': r.params,
                'metrics': r.metrics,
                'feature_importance': r.feature_importance,
                'feature_list': r.feature_list,
                'score_config': r.score_config,
                'score_distribution': r.score_distribution,
                'created_at': r.created_at,
                'is_active': r.id == active_result_id
            }
            items.append(item)

        return {
            'total': len(results),
            'items': items
        }
    except Exception as e:
        import logging
        logging.exception(f"获取结果列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/results/{result_id}', response_model=ModelResultResponse)
def get_result(result_id: int, db: Session = Depends(get_db)):
    """获取模型结果详情"""
    result = db.query(ModelResult).filter(ModelResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail='结果不存在')
    return result





@router.get('/projects/{project_id}/monitor/model_logs')
def list_monitoring_logs(project_id: int, db: Session = Depends(get_db)):
    """获取项目的模型监控日志"""
    return db.query(MonitoringLog).filter(MonitoringLog.project_id == project_id).order_by(MonitoringLog.created_at.desc()).all()


@router.patch('/projects/{project_id}/models/{result_id}/status')
def update_model_status(project_id: int, result_id: int, status_req: Dict[str, str], db: Session = Depends(get_db)):
    """切换模型上线状态（单活模式：一个项目只能有一个 Active 模型）"""
    new_status = status_req.get('status')
    
    if new_status == 'active':
        # 1. 该项目下所有其它部署置为 inactive
        db.query(Deployment).filter(
            Deployment.project_id == project_id,
            Deployment.status == 'active'
        ).update({"status": "inactive"})
        
        # 2. 将当前模型置为 active
        dep = db.query(Deployment).filter(Deployment.model_result_id == result_id).first()
        if not dep:
            dep = Deployment(project_id=project_id, model_result_id=result_id, status='active')
            db.add(dep)
        else:
            dep.status = 'active'
    else:
        # 仅将当前模型置为 inactive
        db.query(Deployment).filter(Deployment.model_result_id == result_id).update({"status": "inactive"})
        
    db.commit()
    return {"message": "状态更新成功"}


@router.delete('/projects/{project_id}/models/{result_id}')
def delete_model(project_id: int, result_id: int, db: Session = Depends(get_db)):
    """删除指定的训练模型及其关联文件、日志"""
    result = db.query(ModelResult).filter(
        ModelResult.id == result_id, 
        ModelResult.project_id == project_id
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="模型不存在")

    # 1. 检查是否为当前正上线的模型，若是则不允许直接删除
    active_dep = db.query(Deployment).filter(
        Deployment.model_result_id == result_id,
        Deployment.status == 'active'
    ).first()
    if active_dep:
        raise HTTPException(status_code=400, detail="处于上线(活跃)状态的模型不允许删除，请先下线")

    try:
        # 2. 删除相关的模型分析报告及物理文件
        reports = db.query(ModelReport).filter(ModelReport.model_result_id == result_id).all()
        for report in reports:
            if report.file_path and os.path.exists(report.file_path):
                try:
                    os.remove(report.file_path)
                except Exception:
                    pass
            db.delete(report)

        # 3. 删除相关的 Deployment 及相关的业务进件模拟日志 MonitoringLog
        deps = db.query(Deployment).filter(Deployment.model_result_id == result_id).all()
        for dep in deps:
            db.query(MonitoringLog).filter(MonitoringLog.deployment_id == dep.id).delete()
            db.delete(dep)

        # 4. 删除物理模型文件 (.pkl)
        if result.model_path and os.path.exists(result.model_path):
            try:
                os.remove(result.model_path)
            except Exception:
                pass

        # 5. 最后删除模型结果记录本身
        db.delete(result)
        db.commit()
        
    except Exception as e:
        db.rollback()
        import logging
        logging.exception(f"删除模型时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"内部错误: 无法完整删除该模型, {str(e)}")

    return {"message": "模型删除成功"}



@router.post('/projects/{project_id}/monitor/simulate_all')
def simulate_all_monitor(project_id: int, db: Session = Depends(get_db)):
    """一键执行模型与策略全量监控模拟"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    # 1. 准备基础数据集
    ref_dataset = project.datasets[0] if project.datasets else None
    if not ref_dataset:
        raise HTTPException(status_code=400, detail='没有基础数据集可供模拟')
    
    try:
        df_ref = load_data(ref_dataset.file_path)
        # 按照您的要求，将偏移量调回稳健水平 (0.05)，单次采样 8000 条
        sim_df = simulate_business_intake(df_ref, n_samples=8000, drift_scale=0.05)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'模拟数据生成失败: {str(e)}')

    batch_name = f"全量模拟_{datetime.now().strftime('%m%d_%H%M')}"
    
    # 2. 执行模型监控 (针对当前活跃部署)
    active_dep = db.query(Deployment).filter(Deployment.project_id == project_id, Deployment.status == 'active').first()
    if active_dep:
        result = active_dep.model_result
        mon_res = run_monitoring_task(
            model_path=result.model_path,
            current_df=sim_df,
            feature_list=result.feature_list,
            score_config=result.score_config,
            score_distribution=result.score_distribution,
            batch_name=batch_name
        )
        if mon_res:
            m_log = MonitoringLog(
                project_id=project_id,
                deployment_id=active_dep.id,
                batch_name=batch_name,
                sample_size=mon_res['sample_size'],
                psi=mon_res['psi'],
                avg_score=mon_res['avg_score'],
                score_dist=mon_res['score_dist'],
                metrics=mon_res['metrics']
            )
            db.add(m_log)

    # 3. 执行策略监控 (针对当前已上线策略)
    active_strategies = db.query(Strategy).filter(
        Strategy.project_id == project_id,
        Strategy.status == 'active'
    ).order_by(Strategy.priority.asc()).all()
    
    if active_strategies:
        s_res = run_strategy_monitoring(sim_df, active_strategies, db, ModelResult)
        if s_res:
            s_log = StrategyMonitoringLog(
                project_id=project_id,
                batch_name=batch_name,
                total_count=s_res['total_count'],
                pass_count=s_res['pass_count'],
                hit_count=s_res['hit_count'],
                approval_rate=s_res['approval_rate'],
                rule_stats=s_res['rule_stats']
            )
            db.add(s_log)

    db.commit()
    return {"message": "全量模拟监控完成", "batch_name": batch_name}

@router.get('/projects/{project_id}/strategy_monitor/logs')
def get_strategy_monitoring_logs(project_id: int, db: Session = Depends(get_db)):
    """获取项目下的策略监控日志"""
    logs = db.query(StrategyMonitoringLog).filter(
        StrategyMonitoringLog.project_id == project_id
    ).order_by(StrategyMonitoringLog.created_at.desc()).limit(20).all()
    return logs

@router.post('/projects/{project_id}/models/{model_id}/report')
async def generate_model_report_api(project_id: int, model_id: int, db: Session = Depends(get_db)):
    """启动报告生成任务"""
    # 创建任务记录
    task = Task(
        project_id=project_id,
        task_type='model_report',
        status='pending',
        params={'model_id': model_id}
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    from scorecard_core.report import run_report_task
    await submit_task(
        task_id=task.id,
        func=run_report_task,
        project_id=project_id,
        model_result_id=model_id
    )
    
    return {"task_id": task.id, "status": "pending"}

@router.get('/projects/{project_id}/models/{model_id}/report')
def get_model_report(project_id: int, model_id: int, db: Session = Depends(get_db)):
    """查看报告详情"""
    report = db.query(ModelReport).filter(
        ModelReport.project_id == project_id,
        ModelReport.model_result_id == model_id
    ).first()
    if not report:
        raise HTTPException(status_code=404, detail="报告尚未生成或模型不存在")
    return report

@router.get('/projects/{project_id}/models/{model_id}/report/download')
def download_model_report(project_id: int, model_id: int, db: Session = Depends(get_db)):
    """导出报告 Excel"""
    report = db.query(ModelReport).filter(
        ModelReport.project_id == project_id,
        ModelReport.model_result_id == model_id
    ).first()
    if not report or not report.file_path or not os.path.exists(report.file_path):
        raise HTTPException(status_code=404, detail="报告文件不存在")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        report.file_path, 
        filename=f"model_report_{model_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
