# -*- coding: utf-8 -*-
"""策略管理 API"""

import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Project, Dataset, Strategy
from app.schemas import (
    Rule, StrategyAnalyzeRequest, StrategyCreate, StrategyResponse,
    StrategyReorderRequest, StrategyStatusUpdateRequest
)
from scorecard_core.data_processor import load_data
from scorecard_core.strategy_engine import run_strategy_analysis, enrich_df_with_model_scores
from scorecard_core.monitor_engine import proba2score
import pickle
from app.models import ModelResult

_current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

router = APIRouter(prefix='/api/strategies', tags=['策略管理'])

@router.post('/analyze')
def analyze_strategy(req: StrategyAnalyzeRequest, db: Session = Depends(get_db)):
    """运行策略分析（不保存）"""
    dataset = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail='数据集不存在')
    
    # 1. 加载数据
    try:
        df = load_data(dataset.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'数据加载失败: {str(e)}')
    
    # 2. 预处理：计算模型分数规则所需的得分
    rules_dict = [r.dict() for r in req.rules]
    df = enrich_df_with_model_scores(df, rules_dict, db, ModelResult)

    # 3. 调用引擎计算
    metrics = run_strategy_analysis(df, rules_dict, combine_logic=req.combine_logic, rule_type=req.rule_type)
    
    if "error" in metrics:
        raise HTTPException(status_code=400, detail=str(metrics["error"]))
        
    return metrics

@router.post('', response_model=StrategyResponse)
def create_strategy(req: StrategyCreate, db: Session = Depends(get_db)):
    """保存策略方案"""
    strategy = Strategy(
        project_id=req.project_id,
        name=req.name,
        description=req.description,
        status='draft',
        priority=0,
        combine_logic=req.combine_logic,
        rule_type=req.rule_type,
        rules=[r.dict() for r in req.rules],
        metrics=req.metrics
    )
    db.add(strategy)
    db.commit()
    db.refresh(strategy)
    return strategy

@router.post('/reorder')
def reorder_strategies(req: StrategyReorderRequest, db: Session = Depends(get_db)):
    """批量更新策略优先级/顺序"""
    for item in req.items:
        db.query(Strategy).filter(Strategy.id == item.id).update({"priority": item.priority})
    db.commit()
    return {"message": "排序已更新"}

@router.patch('/{strategy_id}/status')
def update_strategy_status(strategy_id: int, req: StrategyStatusUpdateRequest, db: Session = Depends(get_db)):
    """更新策略工作状态（draft/active）"""
    strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if not strategy:
        raise HTTPException(status_code=404, detail='策略不存在')
    strategy.status = req.status
    db.commit()
    return {"id": strategy_id, "status": strategy.status}

@router.get('/projects/{project_id}', response_model=List[StrategyResponse])
def list_strategies(project_id: int, db: Session = Depends(get_db)):
    """获取项目下的所有策略（按优先级排序）"""
    return db.query(Strategy).filter(Strategy.project_id == project_id).order_by(Strategy.priority.asc(), Strategy.created_at.desc()).all()

@router.delete('/{strategy_id}')
def delete_strategy(strategy_id: int, db: Session = Depends(get_db)):
    """删除策略方案"""
    strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if not strategy:
        raise HTTPException(status_code=404, detail='策略不存在')
    db.delete(strategy)
    db.commit()
    return {'message': '已删除'}
