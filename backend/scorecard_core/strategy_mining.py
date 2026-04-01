# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from .rules_from_tree import RulesFromTree

def run_auto_mining(
    df: pd.DataFrame,
    label_col: str,
    time_col: str = None,
    tree_type: str = 'exrf',
    max_depth: int = 3,
    n_estimators: int = 100,
    min_samples_leaf: int = 100,
    max_features: float = 0.5
) -> List[Dict[str, Any]]:
    """
    运行自动化规则挖掘并返回推荐规则列表
    """
    # 1. 准备数据：如果有时间列则按时间切分，否则随机切分
    df = df.copy()
    if time_col and time_col in df.columns:
        df.sort_values(by=time_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
        cut_index = int(len(df) * 0.7)
        df['target_group'] = 'test'
        df.loc[:cut_index, 'target_group'] = 'train'
    else:
        # 随机 7/3 开
        df['target_group'] = np.random.choice(['train', 'test'], size=len(df), p=[0.7, 0.3])
    
    # 排除非特征列
    ex_list = [label_col, 'target_group']
    if time_col:
        ex_list.append(time_col)
    
    # 筛选数值型特征（决策树通常处理数值型更好，或会自动处理）
    # RulesFromTree 用 eval(rule)，所以特征名最好不要有特殊字符
    x_list = [col for col in df.columns if col not in ex_list and df[col].dtype in [np.int64, np.float64, np.int32, np.float32]]
    
    if not x_list:
        raise ValueError("未找到可用于挖掘的数值型特征列")

    # 2. 初始化并配置挖掘器
    params = {
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features
    }
    if tree_type in ['rf', 'exrf']:
        params['n_estimators'] = n_estimators
    
    rft = RulesFromTree(tree_type, df, x_list, label_col, 'target_group')
    rft.set_params(params)
    
    # 3. 执行挖掘
    # 使用多线程以提高速度，但后端环境下 max_workers 不宜过大
    rules_df = rft.make_rules_df(use_thread=True, max_workers=4)
    
    # 4. 转换结果为前端友好的格式
    # 我们只需要一些关键指标，并按坏率/Lift 排序
    # rules_df 包含：规则, 坏样本比例(训练集), Lift(训练集), 命中率(训练集), PSI 等
    
    recommendations = []
    # 筛选出一些“好”规则：Lift 较高，PSI 较低，命中率适中
    results = rules_df.to_dict(orient='records')
    
    for row in results:
        recommendations.append({
            "rule": row['规则'],
            "train_bad_rate": row['坏样本比例(训练集)'],
            "train_lift": row['Lift(训练集)'],
            "train_hit_rate": row['命中率(训练集)'],
            "test_lift": row['Lift(测试集)'],
            "psi": row['PSI']
        })
        
    return recommendations

def run_auto_mining_task(db, task_id, progress_callback, **kwargs):
    """
    异步任务运行器
    """
    from scorecard_core.data_processor import load_data
    from app.models import Dataset
    
    dataset_id = kwargs.get('dataset_id')
    label_col = kwargs.get('label_col', 'label')
    time_col = kwargs.get('time_col')
    tree_type = kwargs.get('tree_type', 'exrf')
    
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError("数据集不存在")
        
    progress_callback(10.0) # 加载数据中
    df = load_data(dataset.file_path)
    
    progress_callback(30.0) # 正在挖掘规则...
    
    recs = run_auto_mining(
        df=df,
        label_col=label_col,
        time_col=time_col,
        tree_type=tree_type,
        max_depth=kwargs.get('max_depth', 3),
        n_estimators=kwargs.get('n_estimators', 100),
        min_samples_leaf=kwargs.get('min_samples_leaf', 100),
        max_features=kwargs.get('max_features', 0.5)
    )
    
    progress_callback(100.0)
    return recs
