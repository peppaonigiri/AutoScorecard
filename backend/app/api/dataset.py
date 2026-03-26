# -*- coding: utf-8 -*-
"""数据集详情 API"""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset
from app.schemas import DataPreviewResponse, DataStatsResponse

router = APIRouter(prefix='/api/datasets', tags=['数据集管理'])


@router.get('/{dataset_id}/preview')
def preview_dataset(dataset_id: int, rows: int = 100, db: Session = Depends(get_db)):
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail='数据集不存在')

        if dataset.file_path.endswith('.parquet'):
            df = pd.read_parquet(dataset.file_path)
        else:
            df = pd.read_csv(dataset.file_path)

        import json
        preview_df = df.head(rows)
        # 转换为原生 Python 字典，安全处理 np 类型和 NaN
        data_records = json.loads(preview_df.to_json(orient='records', date_format='iso'))

        return DataPreviewResponse(
            columns=list(df.columns),
            dtypes=dataset.columns_info,
            data=data_records,
            total_rows=dataset.n_rows
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/{dataset_id}/stats')
def dataset_stats(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail='数据集不存在')

    # 检查数据库中的统计信息
    force_recalc = False # 之前由于需要强制更新缺失率逻辑开启了重算，现在改回 False
    
    if force_recalc or not dataset.stats_cache:
        try:
            if dataset.file_path.endswith('.parquet'):
                df = pd.read_parquet(dataset.file_path)
            else:
                df = pd.read_csv(dataset.file_path)
            
            from scorecard_core.data_processor import calculate_dataset_summary
            stats, l1_res = calculate_dataset_summary(df)
            dataset.stats_cache = stats
            dataset.l1_results = l1_res
            db.add(dataset)
            db.commit()
            db.refresh(dataset)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"计算统计信息失败: {e}")

    return dataset.stats_cache


from app.schemas import BinningExplorerRequest

@router.post('/{dataset_id}/binning-explorer')
def binning_explorer(dataset_id: int, req: BinningExplorerRequest, db: Session = Depends(get_db)):
    """自定义特征分箱预览 API"""
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail='数据集不存在')

        if dataset.file_path.endswith('.parquet'):
            df = pd.read_parquet(dataset.file_path)
        else:
            df = pd.read_csv(dataset.file_path)

        if req.variable not in df.columns or req.label_col not in df.columns:
            raise HTTPException(status_code=400, detail='变量或标签列不存在')

        import os
        import sys
        # 动态添加 scorecard 路径 (dataset.py 在 backend/app/api 下，需回退 4 层)
        _api_dir = os.path.dirname(os.path.abspath(__file__))
        _root_dir = os.path.dirname(os.path.dirname(os.path.dirname(_api_dir)))
        _scorecard_dir = os.path.join(_root_dir, 'scorecard')
        if _scorecard_dir not in sys.path:
            sys.path.append(_scorecard_dir)

        from new_tools.iv_report import IVCalculator
        from scorecard_core.report import clean_serializable
        import toad

        # 准备分析数据
        subset = df[[req.variable, req.label_col]].dropna()
        if subset.empty:
            return []

        # 构造临时的 IVCalculator 对象
        df_tmp = subset.copy()
        df_tmp['target_tmp'] = 'train'
        
        # 根据方法计算边界
        if req.method == 'decision_tree':
            # 内部会自动拟合树模型
            calc = IVCalculator(df=df_tmp, label=req.label_col, target='target_tmp', keep_list=[req.variable],
                               max_leaf_nodes=req.max_leaf_nodes, min_samples_leaf=req.min_samples_leaf)
            _, details = calc.calculate_iv(df_tmp[req.variable], df_tmp[req.label_col])
        else:
            # 使用 toad 处理 quantile 或 chi
            c = toad.transform.Combiner()
            c.fit(subset, y=req.label_col, method=req.method, n_bins=req.n_bins)
            bins = c.export().get(req.variable, [])
            
            calc = IVCalculator(df=df_tmp, label=req.label_col, target='target_tmp', keep_list=[req.variable])
            import numpy as np
            boundary = [-np.inf] + list(bins) + [np.inf]
            _, details = calc.calculate_iv(df_tmp[req.variable], df_tmp[req.label_col], bins=boundary)

        return clean_serializable(details.to_dict(orient='records'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
