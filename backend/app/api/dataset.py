# -*- coding: utf-8 -*-
"""数据集详情 API"""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset
from app.schemas import DataPreviewResponse, DataStatsResponse
from app.api.auth import get_current_user

router = APIRouter(prefix='/api/datasets', tags=['数据集管理'])


@router.get('/{dataset_id}/preview')
def preview_dataset(dataset_id: int, rows: int = 100, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
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
def dataset_stats(dataset_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
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
            stats, l1_res = calculate_dataset_summary(df, impute_value=dataset.impute_value)
            dataset.stats_cache = stats
            dataset.l1_results = l1_res
            db.add(dataset)
            db.commit()
            db.refresh(dataset)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"计算统计信息失败: {e}")

    return dataset.stats_cache


from app.schemas import BinningExplorerRequest, ImputeRequest


@router.delete('/{dataset_id}')
def delete_dataset(dataset_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    """删除数据集及其物理文件"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail='数据集不存在')

    # 检查是否有运行中的任务正在使用该数据集（精准检查）
    from app.models import Task
    from sqlalchemy import cast, String
    running_task = db.query(Task).filter(
        Task.project_id == dataset.project_id,
        Task.status.in_(['pending', 'running']),
        Task.params['dataset_id'].as_integer() == dataset_id
    ).first()
    if running_task:
        raise HTTPException(status_code=400, detail='该数据集正被运行中的任务使用，请等待完成后再删除')

    import os
    # 删除物理文件
    if dataset.file_path and os.path.exists(dataset.file_path):
        try:
            os.remove(dataset.file_path)
        except Exception:
            pass

    db.delete(dataset)
    db.commit()
    return {"message": f"数据集 '{dataset.name}' 已删除"}

@router.post('/{dataset_id}/impute')
def impute_dataset(dataset_id: int, req: ImputeRequest, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    """对数据集内除排除列外的字段，将 NaN 替换为指定的默认值，并生成一个新的补充文件分支"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail='数据集不存在')

    try:
        # 1. 加载数据
        if dataset.file_path.endswith('.parquet'):
            df = pd.read_parquet(dataset.file_path)
        else:
            df = pd.read_csv(dataset.file_path)

        # 2. 遍历列判断（支持数字和字符串统一转空逻辑，如果是数值列直接处理）
        fill_val = req.fill_value
        cols_to_fill = [col for col in df.columns if col not in req.exclude_cols]
        for col in cols_to_fill:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(fill_val)
        
        # 3. 写回新文件 (不覆盖原快照)
        import os
        import time
        base, ext = os.path.splitext(dataset.file_path)
        timestamp = int(time.time())
        new_file_path = f"{base}_imputed_{timestamp}{ext}"

        if new_file_path.endswith('.parquet'):
            df.to_parquet(new_file_path, index=False)
        else:
            df.to_csv(new_file_path, index=False)
            
        # 4. 在数据库创建新的一条数据集快照分支
        from datetime import datetime
        ts = datetime.now().strftime('%m%d_%H%M')
        new_dataset = Dataset(
            project_id=dataset.project_id,
            name=f"{dataset.name}_已填充({int(fill_val)})_{ts}",
            file_path=new_file_path,
            file_size=os.path.getsize(new_file_path),
            n_rows=dataset.n_rows,
            n_cols=dataset.n_cols,
            columns_info=dataset.columns_info,
            impute_value=fill_val,
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        # 5. 初始计算统计信息并缓存
        from scorecard_core.data_processor import calculate_dataset_summary
        stats, l1_res = calculate_dataset_summary(df, impute_value=fill_val)
        new_dataset.stats_cache = stats
        new_dataset.l1_results = l1_res
        db.add(new_dataset)
        db.commit()
        
        return {"message": f"成功以 {fill_val} 完成缺失值填充", "new_dataset_id": new_dataset.id}

    except Exception as e:
        import traceback
        import logging
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"缺失值填充失败: {str(e)}")


@router.post('/{dataset_id}/binning-explorer')
def binning_explorer(dataset_id: int, req: BinningExplorerRequest, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
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
