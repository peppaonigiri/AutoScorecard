# -*- coding: utf-8 -*-
"""特征工程 API"""

import sys
import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset, Project, Task
from app.schemas import (
    FeatureFilterRequest, IVReportRequest, FeatureReportResponse, TaskResponse
)
from app.task_manager import submit_task

_current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

router = APIRouter(prefix='/api/projects', tags=['特征工程'])


@router.post('/{project_id}/feature/iv-report')
def create_iv_report(project_id: int, req: IVReportRequest, db: Session = Depends(get_db)):
    """生成 IV 和 PSI 报告"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    dataset = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail='数据集不存在')

    # 持久化划分配置
    project.split_config = {
        'split_ratios': req.split_ratios,
        'oot_col': req.oot_col,
        'oot_start_time': req.oot_start_time,
        'oot_pct': req.oot_pct
    }
    db.commit()

    from scorecard_core.data_processor import load_data, split_dataset
    from scorecard_core.feature_engineer import calc_multi_set_metrics

    df = load_data(dataset.file_path)
    
    # 拆分数据集
    datasets = split_dataset(df, dep=req.dep, 
                             ratios=req.split_ratios, 
                             oot_col=req.oot_col, 
                             oot_start_time=req.oot_start_time,
                             oot_pct=req.oot_pct)

    default_exclude = list(set(req.exclude_cols + [req.dep, 'target', 'weight']))
    ft_lst = [c for c in df.columns if c not in default_exclude]

    # 计算多集 IV 和 PSI
    report = calc_multi_set_metrics(datasets, ft_lst, dep=req.dep)
    features_list = report.to_dict(orient='records')

    project.iv_report = features_list
    db.add(project)
    db.commit()

    return {
        'features': features_list,
        'total_features': len(ft_lst),
    }


@router.post('/{project_id}/feature/filter')
def filter_features(project_id: int, req: FeatureFilterRequest, db: Session = Depends(get_db)):
    """变量筛选（集成两层筛选）"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail='项目不存在')

    dataset = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail='数据集不存在')

    # 同步更新划分配置
    project.split_config = {
        'split_ratios': req.split_ratios,
        'oot_col': req.oot_col,
        'oot_start_time': req.oot_start_time,
        'oot_pct': req.oot_pct
    }

    from scorecard_core.data_processor import load_data, split_dataset
    from scorecard_core.feature_engineer import filter_features as do_filter

    df = load_data(dataset.file_path)
    
    datasets = split_dataset(df, dep=req.dep, 
                             ratios=req.split_ratios, 
                             oot_col=req.oot_col, 
                             oot_start_time=req.oot_start_time,
                             oot_pct=req.oot_pct)

    default_exclude = list(set(req.exclude_cols + [req.dep, 'target', 'weight']))
    ft_lst = [c for c in df.columns if c not in default_exclude]

    thresholds = req.thresholds.dict()
    # 执行筛选
    result = do_filter(datasets, ft_lst, req.dep, thresholds, exclude_cols=default_exclude, skip_l1=req.skip_l1, impute_value=dataset.impute_value)

    # 持久化特征列表及筛选报告
    project.feature_list = result['kept_features']
    project.filter_result = result
    db.add(project)
    db.commit()

    return result
