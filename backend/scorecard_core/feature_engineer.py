# -*- coding: utf-8 -*-
"""scorecard_core - 特征工程模块"""

import sys
import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# 将 scorecard 包加入 path
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_scorecard_path = os.path.join(_base_dir, 'scorecard')
if _scorecard_path not in sys.path:
    sys.path.insert(0, _scorecard_path)


def calc_multi_set_metrics(datasets, ft_lst, dep='label'):
    """
    计算多集 IV 和 PSI 特征报告
    """
    import toad
    
    # 过滤掉 ft_lst 中不在所有数据集中的列
    common_ft = [c for c in ft_lst if all(c in ds.columns for ds in datasets.values())]
    if not common_ft:
        return pd.DataFrame({'var_names': ft_lst})
    
    report = pd.DataFrame({'var_names': common_ft})
    
    # 1. 计算三集的独立 IV（calc_iv 返回 dict）
    for name, df in datasets.items():
        if len(df) == 0: continue
        iv_dict = calc_iv(df, common_ft, dep)
        iv_df = pd.DataFrame([
            {'var_names': k, f'iv_{name}': v} for k, v in iv_dict.items()
        ])
        report = pd.merge(report, iv_df, on='var_names', how='left')
    
    # 2. 计算两两之间的 PSI (稳定性)
    # 仅对数值型列计算
    train = datasets.get('train')
    valid = datasets.get('valid')
    oot = datasets.get('oot')
    
    numeric_ft = [c for c in common_ft if pd.api.types.is_numeric_dtype(train[c])] if train is not None else []
    
    def _safe_psi(df1, df2, cols, col_name):
        try:
            psi_result = toad.metrics.PSI(df1[cols], df2[cols])
            if isinstance(psi_result, pd.Series):
                psi_df = psi_result.reset_index()
                psi_df.columns = ['var_names', col_name]
            else:
                psi_df = pd.DataFrame({'var_names': cols, col_name: [0.0]*len(cols)})
            return psi_df
        except Exception as e:
            logger.warning(f"PSI 计算失败 ({col_name}): {e}")
            return pd.DataFrame({'var_names': cols, col_name: [0.0]*len(cols)})
    
    if train is not None and valid is not None and numeric_ft:
        report = pd.merge(report, _safe_psi(train, valid, numeric_ft, 'psi_tv'), on='var_names', how='left')
        
    if train is not None and oot is not None and numeric_ft:
        report = pd.merge(report, _safe_psi(train, oot, numeric_ft, 'psi_to'), on='var_names', how='left')
        
    if valid is not None and oot is not None and numeric_ft:
        report = pd.merge(report, _safe_psi(valid, oot, numeric_ft, 'psi_vo'), on='var_names', how='left')
        
    return report.replace([np.inf, -np.inf], np.nan).fillna(0)


def calc_iv(df, feature_cols, dep='label'):
    """计算每个特征的 IV 值"""
    try:
        import toad
        iv_result = toad.quality(df[feature_cols + [dep]], dep, iv_only=True)
        return iv_result.to_dict().get('iv', {})
    except Exception as e:
        logger.warning(f"toad IV 计算失败，使用简易计算: {e}")
        return _simple_iv(df, feature_cols, dep)


def _simple_iv(df, feature_cols, dep):
    """简易 IV 计算（fallback）"""
    iv_dict = {}
    for col in feature_cols:
        try:
            if df[col].dtype in ['object', 'category']:
                iv_dict[col] = 0.0
                continue
            # 等频分箱
            bins = pd.qcut(df[col], q=10, duplicates='drop')
            grouped = df.groupby(bins)[dep].agg(['sum', 'count'])
            grouped['good'] = grouped['count'] - grouped['sum']
            grouped['bad'] = grouped['sum']
            total_good = grouped['good'].sum()
            total_bad = grouped['bad'].sum()
            if total_good == 0 or total_bad == 0:
                iv_dict[col] = 0.0
                continue
            grouped['good_pct'] = grouped['good'] / total_good
            grouped['bad_pct'] = grouped['bad'] / total_bad
            grouped['woe'] = np.log((grouped['good_pct'] + 1e-10) / (grouped['bad_pct'] + 1e-10))
            grouped['iv'] = (grouped['good_pct'] - grouped['bad_pct']) * grouped['woe']
            iv_dict[col] = float(grouped['iv'].sum())
        except Exception:
            iv_dict[col] = 0.0
    return iv_dict


def calc_missing_rate(df, feature_cols):
    """计算缺失率"""
    return (df[feature_cols].isnull().sum() / len(df)).to_dict()


def calc_std_ratio(df, feature_cols):
    """计算标准差占比（检测常量列）"""
    result = {}
    for col in feature_cols:
        if df[col].dtype in ['object', 'category']:
            result[col] = 0.0
        else:
            result[col] = float(df[col].std()) if df[col].std() is not None else 0.0
    return result


def calc_freq(df, feature_cols):
    """计算最高频率占比"""
    result = {}
    for col in feature_cols:
        if len(df) == 0:
            result[col] = 0.0
        else:
            result[col] = float(df[col].value_counts(normalize=True).iloc[0]) if len(df[col].value_counts()) > 0 else 0.0
    return result


def calc_correlation(df, feature_cols):
    """计算相关系数矩阵"""
    numeric_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    if not numeric_cols:
        return {}
    corr_matrix = df[numeric_cols].corr()
    return corr_matrix.to_dict()


def filter_features(df_or_datasets, feature_cols, dep, thresholds, exclude_cols=None, skip_l1=False):
    """
    多轮变量筛选（两层机制）
    L1 (统计): 单一值、缺失率、标准差
    L2 (业务): IV (基于 Train)、PSI (稳定性)、相关性
    
    thresholds: {missing, std, freq, iv, corr, psi}
    df_or_datasets: 可以是单一 DataFrame 或 {'train': df, 'valid': df, 'oot': df} 字典
    """
    import pandas as pd
    import numpy as np
    print(f"DEBUG: filter_features thresholds={thresholds}")
    from scorecard_core.data_processor import screen_features_basic
    if isinstance(df_or_datasets, dict):
        df_main = df_or_datasets.get('train', next(iter(df_or_datasets.values())))
        datasets = df_or_datasets
    else:
        df_main = df_or_datasets
        datasets = {'train': df_or_datasets}

    if exclude_cols is None:
        exclude_cols = []

    # ---- Round 1: L1 基础统计筛选 ----
    if not skip_l1:
        l1_kept, l1_removed = screen_features_basic(
            df_main, 
            feature_cols, 
            single_value_limit=thresholds.get('freq', 0.95),
            null_limit=thresholds.get('missing', 0.95)
        )
        kept = l1_kept
        dropped = {
            'missing': l1_removed['missing'],
            'freq': l1_removed['freq'],
            'zero_std': l1_removed['zero_std']
        }
    else:
        kept = feature_cols
        dropped = {
            'missing': [], 'freq': [], 'zero_std': []
        }
    
    details = {}

    # ---- Round 2: L2 业务筛选 (IV & PSI) ----
    report = calc_multi_set_metrics(datasets, kept, dep=dep)
    
    # IV 筛选 (基于 Train IV)
    iv_thresh = thresholds.get('iv', 0.02)
    iv_col = 'iv_train' if 'iv_train' in report.columns else report.columns[1] # fallback to first iv
    
    drop_iv = report[report[iv_col] < iv_thresh]['var_names'].tolist()
    dropped['iv'] = drop_iv
    kept = [c for c in kept if c not in drop_iv]
    
    # PSI 筛选 (稳定性)
    psi_thresh = thresholds.get('psi', 0.1)
    drop_psi = []
    for psi_col in ['psi_tv', 'psi_to']:
        if psi_col in report.columns:
            bad_psi = report[(report['var_names'].isin(kept)) & (report[psi_col] > psi_thresh)]['var_names'].tolist()
            drop_psi.extend(bad_psi)
    
    drop_psi = list(set(drop_psi))
    dropped['psi'] = drop_psi
    kept = [c for c in kept if c not in drop_psi]

    # ---- Round 2: L2 相关性 ----
    if len(kept) > 1:
        details['iv'] = dict(zip(report['var_names'], report[iv_col]))
        numeric_kept = [c for c in kept if pd.api.types.is_numeric_dtype(df_main[c])]
        if len(numeric_kept) > 1:
            corr_matrix = df_main[numeric_kept].corr().abs()
            drop_corr = set()
            for i in range(len(numeric_kept)):
                for j in range(i + 1, len(numeric_kept)):
                    if corr_matrix.iloc[i, j] >= thresholds.get('corr', 0.9):
                        col_i, col_j = numeric_kept[i], numeric_kept[j]
                        if details['iv'].get(col_i, 0) >= details['iv'].get(col_j, 0):
                            drop_corr.add(col_j)
                        else:
                            drop_corr.add(col_i)
            dropped['corr'] = list(drop_corr)
            kept = [c for c in kept if c not in drop_corr]
        else:
            dropped['corr'] = []
    else:
        dropped['corr'] = []

    return {
        'kept_features': kept,
        'dropped_features': dropped,
        'details': details,
        'summary': {
            'total_input': len(feature_cols),
            'total_kept': len(kept),
            'total_dropped': len(feature_cols) - len(kept),
            'drop_by_missing': len(dropped.get('missing', [])),
            'drop_by_freq': len(dropped.get('freq', [])),
            'drop_by_zero_std': len(dropped.get('zero_std', [])),
            'drop_by_iv': len(dropped.get('iv', [])),
            'drop_by_psi': len(dropped.get('psi', [])),
            'drop_by_corr': len(dropped.get('corr', [])),
        }
    }


def generate_iv_report(df, feature_cols, dep='label'):
    """生成 IV 报告"""
    iv_values = calc_iv(df, feature_cols, dep)
    missing_rates = calc_missing_rate(df, feature_cols)

    report = []
    for col in feature_cols:
        report.append({
            'feature': col,
            'iv': round(iv_values.get(col, 0.0), 6),
            'missing_rate': round(missing_rates.get(col, 0.0), 4),
            'dtype': str(df[col].dtype),
            'nunique': int(df[col].nunique()),
        })

    # 按 IV 降序排列
    report.sort(key=lambda x: x['iv'], reverse=True)
    return report
