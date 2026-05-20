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
    StrategyReorderRequest, StrategyStatusUpdateRequest, AutoMiningRequest,
    SwapAnalysisRequest, SwapAnalysisResponse
)
from app.models import ModelResult, Task
from scorecard_core.data_processor import load_data
from scorecard_core.strategy_engine import run_strategy_analysis, enrich_df_with_model_scores

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

@router.post('/auto-mining')
async def start_auto_mining(req: AutoMiningRequest, db: Session = Depends(get_db)):
    """启动自动化规则挖掘任务"""
    dataset = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail='数据集不存在')
        
    # 1. 记录任务
    new_task = Task(
        project_id=dataset.project_id,
        task_type='strategy_mining',
        status='pending',
        progress=0.0,
        params=req.dict()
    )
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    
    # 2. 提交异步挖掘任务
    from scorecard_core.strategy_mining import run_auto_mining_task
    from app.task_manager import submit_task
    
    await submit_task(new_task.id, run_auto_mining_task, **req.dict())
    
    return {"task_id": new_task.id}


@router.post('/swap-analysis', response_model=SwapAnalysisResponse)
def swap_analysis(req: SwapAnalysisRequest, db: Session = Depends(get_db)):
    """策略置换分析（Swap In/Out）

    支持两种策略传入方式：
    1. old_strategy_id / new_strategy_id：直接引用数据库中已保存的策略，后端自动读取规则和 model_result_id
    2. old_col/old_reject_op/old_reject_val + new_col/new_reject_op/new_reject_val：手动指定单条规则
       - 若策略为分数型（field='score' 或 '_model_result_X'），需额外传 model_result_id
    """
    dataset = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail='数据集不存在')

    try:
        df = load_data(dataset.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'数据加载失败: {str(e)}')

    from app.models import Deployment

    # ── 辅助：从数据库读取策略规则 ─────────────────────────────
    def _load_strategy_rules(strategy_id: int):
        """返回 (rules_list, combine_logic, rule_type, model_result_id)"""
        strat = db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not strat:
            raise HTTPException(status_code=404, detail=f'策略 {strategy_id} 不存在')
        rules_raw = strat.rules or []
        rules_list = [r if isinstance(r, dict) else r.dict() for r in rules_raw]
        mid = None
        for r in rules_list:
            f = r.get('field', '')
            if f.startswith('_model_result_'):
                try:
                    mid = int(f.replace('_model_result_', ''))
                except ValueError:
                    pass
                break
        return rules_list, strat.combine_logic or 'and', strat.rule_type or 'reject', mid

    # ── 辅助：解析字段名对应的 model_result_id ──────────────────
    def _resolve_mid(col: Optional[str]) -> Optional[int]:
        if not col:
            return None
        if col.startswith('_model_result_'):
            try:
                return int(col.replace('_model_result_', ''))
            except ValueError:
                pass
        if col.lower() == 'score':
            dep = db.query(Deployment).filter(
                Deployment.project_id == dataset.project_id,
                Deployment.status == 'active'
            ).first()
            if dep:
                return dep.model_result_id
            latest = db.query(ModelResult).filter(
                ModelResult.project_id == dataset.project_id
            ).order_by(ModelResult.created_at.desc()).first()
            if latest:
                return latest.id
        return None

    # ── 提取旧策略规则 ──────────────────────────────────────────
    old_model_id: Optional[int] = None
    if req.old_strategy_id:
        old_rules_list, old_combine_logic, old_rule_type, old_model_id = _load_strategy_rules(req.old_strategy_id)
    else:
        old_rules_list = [r.dict() for r in req.old_rules] if req.old_rules else []
        if not old_rules_list and req.old_col and req.old_reject_op:
            old_rules_list = [{"field": req.old_col, "op": req.old_reject_op, "val": req.old_reject_val}]
        old_combine_logic = req.old_combine_logic or 'and'
        old_rule_type = req.old_rule_type or 'reject'

    # ── 提取新策略规则 ──────────────────────────────────────────
    new_model_id: Optional[int] = None
    if req.new_strategy_id:
        new_rules_list, new_combine_logic, new_rule_type, new_model_id = _load_strategy_rules(req.new_strategy_id)
    else:
        new_rules_list = [r.dict() for r in req.new_rules] if req.new_rules else []
        if not new_rules_list and req.new_col and req.new_reject_op:
            new_rules_list = [{"field": req.new_col, "op": req.new_reject_op, "val": req.new_reject_val}]
        new_combine_logic = req.new_combine_logic or 'and'
        new_rule_type = req.new_rule_type or 'reject'

    # ── 统一模型打分补全 ─────────────────────────────────────────
    # 显式 model_result_id 优先；其次从策略规则或字段名推断
    explicit_mid: Optional[int] = req.model_result_id
    mids_to_score: set = set()

    def _get_mid_for_rule(r: dict) -> Optional[int]:
        col = r.get('field', '')
        if col in df.columns:
            return None  # 已有该列，不需要打分
        return explicit_mid or old_model_id or new_model_id or _resolve_mid(col)

    # 遍历所有规则，规范化分数字段名并收集需打分的 mid
    for r in old_rules_list + new_rules_list:
        col = r.get('field', '')
        if col in df.columns:
            continue
        mid = _get_mid_for_rule(r)
        if mid:
            r['field'] = f'_model_result_{mid}'  # 规范化
            mids_to_score.add(mid)

    if explicit_mid:
        mids_to_score.add(explicit_mid)

    if mids_to_score:
        enrich_rules = [{'field': f'_model_result_{mid}'} for mid in mids_to_score]
        try:
            df = enrich_df_with_model_scores(df, enrich_rules, db, ModelResult)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'模型打分失败: {str(e)}')

    # ── 确定 Swap 用的 old_col / new_col ────────────────────────
    def _pick_col(req_col: Optional[str], rules: list, fallback_mid: Optional[int]) -> Optional[str]:
        """选出在 df 中实际存在的列名"""
        # 优先：请求中明确指定的列
        if req_col:
            norm = f'_model_result_{fallback_mid or explicit_mid}' if req_col not in df.columns and (fallback_mid or explicit_mid) else req_col
            if norm in df.columns:
                return norm
        # 兜底：rules 第一条的 field（已规范化）
        if rules:
            f = rules[0].get('field')
            if f and f in df.columns:
                return f
        return req_col

    old_col = _pick_col(req.old_col, old_rules_list, old_model_id)
    new_col = _pick_col(req.new_col, new_rules_list, new_model_id)

    try:
        from scorecard_core.swap_analysis import run_swap_analysis
        result = run_swap_analysis(
            df=df,
            old_col=old_col,
            old_reject_op=req.old_reject_op,
            old_reject_val=req.old_reject_val,
            new_col=new_col,
            new_reject_op=req.new_reject_op,
            new_reject_val=req.new_reject_val,
            label_col=req.label_col,
            new_col_bins=req.new_col_bins,
            old_rules=old_rules_list,
            old_combine_logic=old_combine_logic,
            old_rule_type=old_rule_type,
            new_rules=new_rules_list,
            new_combine_logic=new_combine_logic,
            new_rule_type=new_rule_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'分析计算失败: {str(e)}')

    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])

    return result
