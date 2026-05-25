# -*- coding: utf-8 -*-
"""scorecard_core - 数据预处理模块"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_data(file_path):
    """加载数据文件（CSV/Parquet）"""
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        return pd.read_csv(file_path)


def get_basic_stats(df, dep='label', impute_value=None):
    """获取数据基础统计信息"""
    missing_counts = _calc_missing_counts(df, impute_value)
    stats = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_rates': (missing_counts / len(df)).to_dict(),
    }

    # 数值列统计
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_stats = {}
    for col in numeric_cols:
        desc = df[col].describe()
        numeric_stats[col] = {k: (float(v) if pd.notnull(v) else None) for k, v in desc.items()}
    stats['numeric_stats'] = numeric_stats

    # 标签分布
    if dep in df.columns:
        label_dist = df[dep].value_counts().to_dict()
        stats['label_distribution'] = {str(k): int(v) for k, v in label_dist.items()}

    return stats

def split_dataset(df, target_col='target', dep='label', 
                  ratios=None, oot_col=None, oot_start_time=None,
                  oot_pct=None, random_state=42):
    """
    增强版数据集拆分
    
    参数:
        ratios: [train_ratio, valid_ratio, oot_ratio] (仅在无 target_col 时使用)
        oot_col: 用于划分 OOT 的时间列名
        oot_start_time: OOT 的起始时间点 (>= 该时间为 OOT)
        oot_pct: OOT 百分比 (如 0.1 表示最后10%按时间排序的数据做 OOT)
    """
    datasets = {}
    temp_df = df.copy()

    # 1. 如果指定了 OOT 列，优先按时间截取 OOT 数据
    if oot_col and oot_col in temp_df.columns:
        try:
            col_vals = temp_df[oot_col].dropna()
            # 三路分支：已是 datetime → 直通；整数型 → 按位数格式化；字符串 → 直接解析
            if pd.api.types.is_datetime64_any_dtype(col_vals):
                pass  # 已是正确类型，无需转换
            elif pd.api.types.is_numeric_dtype(col_vals):
                sample = int(col_vals.iloc[0])
                if 190001 <= sample <= 209912:        # YYYYMM，如 202401
                    temp_df[oot_col] = pd.to_datetime(
                        temp_df[oot_col].astype(int).astype(str), format='%Y%m'
                    )
                elif 19000101 <= sample <= 20991231:  # YYYYMMDD，如 20240101
                    temp_df[oot_col] = pd.to_datetime(
                        temp_df[oot_col].astype(int).astype(str), format='%Y%m%d'
                    )
                else:
                    # Unix 秒时间戳
                    temp_df[oot_col] = pd.to_datetime(temp_df[oot_col], unit='s', errors='coerce')
            else:
                # 字符串类型（含 ISO 8601、'2024-10-02'、'2024-09-07T00:00:00.000000' 等）
                temp_df[oot_col] = pd.to_datetime(temp_df[oot_col], errors='coerce')
            temp_df = temp_df.sort_values(oot_col)

            
            if not oot_pct and not oot_start_time:
                # 默认使用 ratios 末尾比例作为 OOT 返回（默认0.2）
                oot_pct = ratios[-1] / sum(ratios) if ratios and len(ratios) >= 3 else 0.2

            if oot_pct and oot_pct > 0:
                # 按百分比切分：最后 oot_pct 的数据做 OOT
                n = len(temp_df)
                split_idx = int(n * (1 - oot_pct))
                datasets['oot'] = temp_df.iloc[split_idx:].copy()
                rest_df = temp_df.iloc[:split_idx].copy()
                logger.info(f"按时间排序百分比 {oot_pct*100:.1f}% 划分 OOT: {len(datasets['oot'])} 行")
            elif oot_start_time:
                # 按时间点切分
                split_time = pd.to_datetime(oot_start_time)
                oot_mask = temp_df[oot_col] >= split_time
                datasets['oot'] = temp_df[oot_mask].copy()
                rest_df = temp_df[~oot_mask].copy()
                logger.info(f"基于时间 {oot_start_time} 划分 OOT: {len(datasets['oot'])} 行")
            else:
                rest_df = temp_df
        except Exception as e:
            logger.warning(f"时间划分失败: {e}. 回退到无 OOT 或随机划分。")
            rest_df = temp_df
    else:
        rest_df = temp_df

    # 2. 从剩余数据 (rest_df) 中划分 Train / Valid
    # 如果数据集中明确含有预设的 target_col 切分标识，则遵从原有规则
    has_target_split = False
    if target_col in rest_df.columns:
        unique_targets = [str(x).lower() for x in rest_df[target_col].unique() if pd.notna(x)]
        if 'train' in unique_targets or 'valid' in unique_targets:
            for name in rest_df[target_col].unique():
                if pd.isna(name): continue
                key_name = str(name).lower()
                # 避免 target 列带有 'oot' 覆盖已经划分出的 OOT
                if key_name == 'oot' and 'oot' in datasets:
                    continue
                datasets[key_name] = rest_df[rest_df[target_col] == name].copy()
            has_target_split = True

    if has_target_split:
        return datasets

    # 3. 随机划分 Train/Valid (或补全 OOT)
    from sklearn.model_selection import train_test_split
    
    if ratios is None:
        ratios = [0.8, 0.2]
    
    if 'oot' not in datasets:
        s = sum(ratios)
        if len(ratios) >= 3:
            tr, va, oo = ratios[0]/s, ratios[1]/s, ratios[2]/s
            train_df, other = train_test_split(rest_df, test_size=(va+oo), random_state=random_state, 
                                               stratify=rest_df[dep] if dep in rest_df.columns else None)
            valid_df, oot_df = train_test_split(other, test_size=(oo/(va+oo)), random_state=random_state, 
                                               stratify=other[dep] if dep in other.columns else None)
            datasets['train'], datasets['valid'], datasets['oot'] = train_df, valid_df, oot_df
        else:
            tr, va = ratios[0]/s, ratios[1]/s
            train_df, valid_df = train_test_split(rest_df, test_size=va, random_state=random_state, 
                                                  stratify=rest_df[dep] if dep in rest_df.columns else None)
            datasets['train'], datasets['valid'] = train_df, valid_df
    else:
        s_rv = ratios[0] + ratios[1]
        tr, va = ratios[0]/s_rv, ratios[1]/s_rv
        train_df, valid_df = train_test_split(rest_df, test_size=va, random_state=random_state, 
                                              stratify=rest_df[dep] if dep in rest_df.columns else None)
        datasets['train'], datasets['valid'] = train_df, valid_df

    return datasets


def screen_features_basic(df, feature_cols, single_value_limit=0.95, null_limit=0.95, impute_value=None):
    """
    L1 基础特征筛选：按缺失率、单一值率、零标准差三个维度剔除无效特征。

    返回:
        (kept_list, removed_dict)
        removed_dict = {'missing': [...], 'freq': [...], 'zero_std': [...]}
    """
    removed = {
        'missing': [],
        'freq': [],
        'zero_std': []
    }
    kept = []

    for col in feature_cols:
        n = len(df)
        if n == 0:
            kept.append(col)
            continue

        # 1. 缺失率（动态：如果有填充值则该值也算缺失）
        if impute_value is not None and pd.api.types.is_numeric_dtype(df[col]):
            null_rate = (df[col].isnull() | (df[col] == impute_value)).mean()
        else:
            null_rate = df[col].isnull().mean()
        if null_rate >= null_limit:
            removed['missing'].append(col)
            continue

        # 2. 单一值占比过高
        vc = df[col].value_counts(normalize=True)
        if len(vc) > 0 and vc.iloc[0] >= single_value_limit:
            removed['freq'].append(col)
            continue

        # 3. 零方差 (零标准差)
        if pd.api.types.is_numeric_dtype(df[col]):
            std_val = df[col].std()
            if std_val is not None and std_val == 0:
                removed['zero_std'].append(col)
                continue

        kept.append(col)

    logger.info(f"L1 筛选: 输入 {len(feature_cols)} 特征, 保留 {len(kept)}, "
                f"剔除(缺失={len(removed['missing'])}, "
                f"单一值={len(removed['freq'])}, "
                f"零标准差={len(removed['zero_std'])})")
    return kept, removed


def _calc_missing_counts(df, impute_value=None):
    """统一的缺失计数逻辑：NaN + 可选的填充值（仅数值列）"""
    missing_counts = df.isnull().sum()
    if impute_value is not None:
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            missing_counts[col] = (df[col].isnull() | (df[col] == impute_value)).sum()
    return missing_counts


def calculate_dataset_summary(df, dep='label', thresholds=None, impute_value=None):
    """
    计算数据集完整统计信息（包含分位数和 L1 初筛结果）
    用于在数据上传后进行持久化缓存。
    """
    if thresholds is None:
        thresholds = {'freq': 0.95, 'missing': 0.95}

    total_rows = len(df)
    missing_counts = _calc_missing_counts(df, impute_value)
    stats = {
        'n_rows': total_rows,
        'n_cols': len(df.columns),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_rates': (missing_counts / total_rows).to_dict(),
    }

    # 计算分位数和数值统计
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_stats = {}
    for col in numeric_cols:
        desc = df[col].describe()
        # 增加分位数对齐
        q_list = [.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]
        quantiles = df[col].quantile(q_list).to_dict()
        
        col_stat = {k: (float(v) if pd.notnull(v) else None) for k, v in desc.items()}
        col_stat.update({f'q{int(q*100)}': (float(v) if pd.notnull(v) else None) for q, v in quantiles.items()})
        numeric_stats[col] = col_stat
    
    stats['numeric_stats'] = numeric_stats

    # L1 初筛逻辑 (单一值率、标准差、缺失率)
    sys_cols = [str(dep).lower(), 'target', 'weight', 'id', 'uuid']
    potential_features = [c for c in df.columns if str(c).lower() not in sys_cols]
    
    l1_kept, l1_removed = screen_features_basic(df, potential_features, 
                                                single_value_limit=thresholds.get('freq', 0.95), 
                                                null_limit=thresholds.get('missing', 0.95),
                                                impute_value=impute_value)
    
    l1_results = {
        'kept': l1_kept,
        'removed_info': l1_removed
    }

    return stats, l1_results


def get_feature_list(df, exclude_cols):
    """获取特征列表（排除指定列）"""
    return [col for col in df.columns if col not in exclude_cols]


def add_weight_column(df, weight_col='weight'):
    """如果没有 weight 列，添加默认权重为 1"""
    if weight_col not in df.columns:
        df[weight_col] = 1
    return df
