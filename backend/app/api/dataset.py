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

    # 如果没有缓存，则计算一次并保存
    if not dataset.stats_cache:
        try:
            if dataset.file_path.endswith('.parquet'):
                df = pd.read_parquet(dataset.file_path)
            else:
                df = pd.read_csv(dataset.file_path)
            
            from scorecard_core.data_processor import calculate_dataset_summary
            stats, l1_res = calculate_dataset_summary(df)
            dataset.stats_cache = stats
            dataset.l1_results = l1_res
            db.commit()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"计算统计信息失败: {e}")

    return dataset.stats_cache
