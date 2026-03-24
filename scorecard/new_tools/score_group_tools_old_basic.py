"""
score_group_tools.py
This file is a toolbox for score result group analysis, including group KS, AUC, sub top 5% lift, monthly PSI, drawing distribution charts, etc.
date:2025-11-21
version:1.0.0
"""

from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import sys
import warnings
warnings.filterwarnings('ignore')


from new_tools import new_psi


def get_metric_stats(
    group,
    score_col,
    target_col: str = 'target',
    weight_col: str = 'weight',
    metrics=None,
    if_only: bool = True,
    if_weight: bool = False,
    if_score: bool = False
):
    """
    统一计算 KS / AUC / TOP_5_LIFT / SUB_5_LIFT 等指标（针对单个样本集合）。

    Parameters
    ----------
    group : pd.DataFrame
        输入数据，至少包含 `score_col`、`target_col`，当 `if_weight=True` 时还需包含 `weight_col`。
    score_col : str
        用于评估的分数列名，可以是 **proba** 也可以是 **score**。
    target_col : str, default 'target'
        目标变量（标签）列名，约定 1 为“坏样本”、0 为“好样本”。
    weight_col : str, default 'weight'
        样本权重列名，仅在 `if_weight=True` 时使用。
    metrics : list or None, default None
        需要计算的指标名称列表（不区分大小写），支持：
        ['KS', 'AUC', 'TOP_5_LIFT', 'SUB_5_LIFT']。
        为 None 时默认计算 ['KS', 'AUC', 'TOP_5_LIFT', 'SUB_5_LIFT']。
    if_only : bool, default True
        - True：返回 `pd.Series`（只包含指标值）。
        - False：返回 `pd.DataFrame`，附带样本量、正负样本数等信息。
        一般情况下不会使用这里的if_only=False，因为如果需要查看样本量、正负样本数等信息，通常使用get_group_stats_basic函数。
    if_weight : bool, default False
        是否使用 `weight_col` 作为样本权重来计算 KS / AUC。
    if_score : bool, default False
        - False：`score_col` 为 proba，值越大“越坏”。
        - True：`score_col` 为 score，值越小“越坏”，AUC 和 lift 内部会做方向调整。

    Returns
    -------
    pd.Series or pd.DataFrame
        根据 `if_only` 返回不同类型的结果，其中键为指标名称。
    """

    if metrics is None:
        metrics = ['KS', 'AUC', 'TOP_5_LIFT', 'SUB_5_LIFT']
    metrics = [metric.upper() for metric in metrics]
    supported_metrics = {'KS', 'AUC', 'TOP_5_LIFT', 'SUB_5_LIFT'}
    unknown = set(metrics) - supported_metrics
    if unknown:
        raise ValueError(f"暂不支持的指标: {unknown}")

    sample_weight = group[weight_col] if if_weight else None
    results = {}

    # KS / AUC
    if ('KS' in metrics) or ('AUC' in metrics):
        fpr, tpr, _ = roc_curve(group[target_col], group[score_col], sample_weight=sample_weight)
        if 'KS' in metrics:
            results['KS'] = abs(fpr - tpr).max()
        if 'AUC' in metrics:
            results['AUC'] = auc(fpr, tpr)
            if if_score:
                results['AUC'] = 1-results['AUC']

    # top/sub 5% lift（参考给定函数逻辑：sub_5_lift 对应 top5，top_5_lift 对应 bottom5）
    if ('TOP_5_LIFT' in metrics) or ('SUB_5_LIFT' in metrics):
        if if_score:
            proab_score_col = group[score_col].copy()
        else:
            proab_score_col = -group[score_col].copy()
        
        high_thr = proab_score_col.quantile(0.95)
        low_thr = proab_score_col.quantile(0.05)
        top5 = group[proab_score_col >= high_thr]
        bottom5 = group[proab_score_col <= low_thr]
        bad_rate_all = (group[target_col] == 1).mean()
        bad_rate_top5 = (top5[target_col] == 1).mean()
        bad_rate_bottom5 = (bottom5[target_col] == 1).mean()
        lift_top5 = bad_rate_top5 / bad_rate_all if bad_rate_all > 0 else float('nan')
        lift_bottom5 = bad_rate_bottom5 / bad_rate_all if bad_rate_all > 0 else float('nan')

        # 注意：按照你提供的参考函数，这里保持 sub_5_lift=top5 lift，top_5_lift=bottom5 lift
        if 'SUB_5_LIFT' in metrics:
            results['SUB_5_LIFT'] = lift_top5
        if 'TOP_5_LIFT' in metrics:
            results['TOP_5_LIFT'] = lift_bottom5

    if if_only:
        return pd.Series(results)
    extra_stats = {
        '样本量': len(group),
        '负样本数': group[target_col].sum(),
        '正样本数': len(group) - group[target_col].sum()
    }
    extra_stats.update(results)
    return pd.DataFrame([extra_stats])


def get_group_stats_basic(
    label: pd.DataFrame,
    groupby_col,
    target_col: str,
    if_weight: bool = False
):
    """
    计算基础分组统计（样本数/正负样本数/坏账率）。

    Parameters
    ----------
    label : pd.DataFrame
        输入数据，至少包含 `target_col`，若 `if_weight=True` 还需包含 'weight'。
    groupby_col : str or None
        分组字段名称：
        - 为 None：不分组，只统计全量一行，index 名为 'all_data'。
        - 为 str：按该列分组统计，并额外增加一行 'all_data' 代表总体。
    target_col : str
        标签列名，约定 1 为“坏样本”、0 为“好样本”。
    if_weight : bool, default False
        - False：按样本条数统计。
        - True：按 `weight` 求和统计样本量和正负样本数。
        注意这里默认需要的是weight列，没有设置weight列的其他命名选择

    Returns
    -------
    pd.DataFrame
        含以下字段：
        - 当 groupby_col 为 None：['all', '负样本数', '正样本数', '样本数', '坏账率']，即只有all_data这一行，没有分组列
        - 否则： [groupby_col, '正样本数', '负样本数', '样本数', '坏账率']，并含 'all_data' 汇总行。
    """
    if 'index' in label.columns:
        label.rename(columns={'index':'index_'},inplace=True)
    label = label.reset_index()
    if groupby_col is None:
        if if_weight:
            ks_stats_all = label.groupby([target_col])['weight'].sum()
        else:
            ks_stats_all = label.groupby([target_col])['index'].count()
        ks_stats_all.index.name = None
        ks_stats_all = ks_stats_all.to_frame().T
        ks_stats_all.index = ['all_data']
        ks_stats_all = ks_stats_all.reset_index()
        ks_stats_all.rename(columns={1.0:'负样本数',0.0:'正样本数','index':'all'},inplace=True)
    else:
        if if_weight:
            if 'weight' not in label.columns:
                raise ValueError('weight column is not in the label, please check the label and the name!!!')
            ks_stats_all = label.groupby([groupby_col,target_col])['weight'].sum().reset_index().pivot_table(index=groupby_col,columns=target_col,values='weight').reset_index()
        else:
            ks_stats_all = label.groupby([groupby_col,target_col])['index'].count().reset_index().pivot_table(index=groupby_col,columns=target_col,values='index').reset_index()

        ks_stats_all.rename(columns={1.0:'负样本数',0.0:'正样本数'},inplace=True)
        ks_stats_all = pd.concat([ks_stats_all,pd.DataFrame([['all_data',ks_stats_all['正样本数'].sum(),ks_stats_all['负样本数'].sum()]],columns=[groupby_col,'正样本数','负样本数'])],axis=0)
    ks_stats_all.fillna(0,inplace=True)
    ks_stats_all['样本数'] = ks_stats_all['负样本数'] + ks_stats_all['正样本数']
    ks_stats_all['坏账率'] = ks_stats_all['负样本数'] / ks_stats_all['样本数']
    return ks_stats_all


def group_metrics(
    label: pd.DataFrame,
    groupby_col,
    target_col: str,
    proba_cols,
    metrics=None,
    metric_alias=None,
    if_weight: bool = False,
    if_score: bool = False,
    multi_index: bool = False
):
    """
    统一的“单个分组列 / 多个分数列”的指标计算接口，可同时计算 KS / AUC / LIFT 等。

    Parameters
    ----------
    label : pd.DataFrame
        输入数据，需包含 `target_col` 和 `proba_cols` 中指定的列，
        若 `if_weight=True` 需额外包含 'weight'。
    groupby_col : str or None
        分组字段：
        - 为 None：不分组，仅返回总体 'all_data' 的指标。
        - 为 str：按该列分组，并额外增加 'all_data' 总体行。
    target_col : str
        标签列名，约定 1 为“坏样本”、0 为“好样本”。
    proba_cols : str or list of str
        分数列名，可以是：
        - 单个字符串：单一模型分数；
        - 字符串列表：多个模型分数。
        当 `if_score=False` 时视为 proba，值越大“越坏”；当 `if_score=True` 时视为 score，值越小“越坏”。
    metrics : list or None, default None
        需要计算的指标名称列表（不区分大小写），支持：
        ['KS', 'AUC', 'SUB_5_LIFT', 'TOP_5_LIFT']。
        为 None 时默认计算全部四个指标。
    metric_alias : dict or None, default None
        指标别名字典，如 {'KS': 'ks', 'AUC': 'auc'}。
        - 不使用多层列时：列名为 `<proba_col>_<alias>`（当 proba_cols 为多个）；
        - 使用多层列时：列名第二层使用别名。
        若为 None，则默认：
        {'KS': 'ks', 'AUC': 'auc', 'SUB_5_LIFT': 'sub_5_lift', 'TOP_5_LIFT': 'top_5_lift'}。
    if_weight : bool, default False
        是否按 `weight` 列加权计算 KS/AUC/LIFT。
        注意这里默认需要的是weight列，没有设置weight列的其他命名选择
    if_score : bool, default False
        - False：`proba_cols` 为 proba，值越大“越坏”；
        - True：`proba_cols` 为 score，值越小“越坏”，内部会对 AUC / LIFT 做方向调整。
    multi_index : bool, default False
        当为 True 且 `proba_cols` 和 `metrics` 都超过 1 个时，使用双层 MultiIndex：
        - 第一层：分数列名 proba_col；
        - 第二层：指标别名 metric_alias。

    Returns
    -------
    pd.DataFrame
        行：分组取值及 'all_data' 汇总行；
        列：基础统计列 + 指标列（根据 `multi_index` 决定是否为多层列）。
    """
    if metrics is None:
        metrics = ['KS', 'AUC', 'SUB_5_LIFT', 'TOP_5_LIFT']
    metrics = [metric.upper() for metric in metrics]
    supported_metrics = {'KS', 'AUC', 'SUB_5_LIFT', 'TOP_5_LIFT'}
    unknown = set(metrics) - supported_metrics
    if unknown:
        raise ValueError(f"暂不支持的指标: {unknown}")
    if metric_alias is None:
        metric_alias = {
            'KS': 'ks',
            'AUC': 'auc',
            'SUB_5_LIFT': 'sub_5_lift',
            'TOP_5_LIFT': 'top_5_lift',
        }

    if isinstance(proba_cols, str):
        proba_cols = [proba_cols]
        single_proba = True
    else:
        single_proba = False

    # 判断是否使用双层索引
    use_multi_index = multi_index and len(proba_cols) > 1 and len(metrics) > 1

    stats_df = get_group_stats_basic(label,groupby_col,target_col,if_weight)

    if groupby_col is None:
        print('groupby_col is None!!! return all_data metrics!!!')

    # 用于存储所有指标数据的字典
    metric_data_dict = {}
    
    for proba_col in proba_cols:
        for metric in metrics:
            total_value = get_metric_stats(
                    label,
                    score_col=proba_col,
                    target_col=target_col,
                    metrics=[metric],
                    if_weight=if_weight,
                    if_score = if_score
                )[metric]
            col_alias = metric_alias.get(metric, metric)
            
            if use_multi_index:
                # 使用双层索引时，先存储数据，最后统一处理
                if proba_col not in metric_data_dict:
                    metric_data_dict[proba_col] = {}
                metric_data_dict[proba_col][col_alias] = {
                    'total_value': total_value,
                    'groupby_data': None
                }
                
                if groupby_col is not None:
                    metric_df = label.groupby(groupby_col).apply(
                        get_metric_stats,
                        score_col=proba_col,
                        target_col=target_col,
                        metrics=[metric],
                        if_weight=if_weight,
                        if_score = if_score
                    ).reset_index()
                    metric_df = metric_df[[groupby_col, metric]].rename(columns={metric: 'value'})
                    metric_data_dict[proba_col][col_alias]['groupby_data'] = metric_df
            else:
                # 原有逻辑：不使用双层索引
                if not single_proba:
                    col_alias = f'{proba_col}_{col_alias}'

                if groupby_col is not None:
                    all_row = pd.DataFrame([['all_data', total_value]], columns=[groupby_col, col_alias])
                    metric_df = label.groupby(groupby_col).apply(
                        get_metric_stats,
                        score_col=proba_col,
                        target_col=target_col,
                        metrics=[metric],
                        if_weight=if_weight,
                        if_score = if_score
                    ).reset_index()

                    metric_df = metric_df[[groupby_col, metric]].rename(columns={metric: col_alias})

                    
                    metric_df = pd.concat([metric_df, all_row], axis=0)
                    stats_df = pd.merge(stats_df, metric_df, on=groupby_col, how='outer')
                else:
                    all_row = pd.DataFrame([['all_data', total_value]], columns=['all', col_alias])
                    metric_df = all_row
                    stats_df = pd.merge(stats_df, metric_df, on='all', how='outer')
    
    # 如果使用双层索引，统一处理所有数据
    if use_multi_index:
        merge_key = groupby_col if groupby_col is not None else 'all'
        
        # 收集所有指标数据到一个DataFrame中
        all_metric_dfs = []
        
        for proba_col in proba_cols:
            for col_alias in metric_data_dict[proba_col].keys():
                total_value = metric_data_dict[proba_col][col_alias]['total_value']
                groupby_data = metric_data_dict[proba_col][col_alias]['groupby_data']
                
                if groupby_col is not None:
                    all_row = pd.DataFrame([['all_data', total_value]], columns=[groupby_col, 'value'])
                    if groupby_data is not None:
                        metric_df = pd.concat([groupby_data, all_row], axis=0)
                    else:
                        metric_df = all_row
                    metric_df.rename(columns={'value': (proba_col, col_alias)}, inplace=True)
                else:
                    metric_df = pd.DataFrame([['all_data', total_value]], columns=['all', (proba_col, col_alias)])
                
                all_metric_dfs.append(metric_df)
        
        # 合并所有指标数据
        if all_metric_dfs:
            combined_metric_df = all_metric_dfs[0]
            for df in all_metric_dfs[1:]:
                combined_metric_df = pd.merge(combined_metric_df, df, on=merge_key, how='outer')
            
            # 合并到stats_df
            stats_df = pd.merge(stats_df, combined_metric_df, on=merge_key, how='outer')
            
            # 设置双层索引
            basic_cols = [col for col in stats_df.columns if not isinstance(col, tuple)]
            metric_cols = [col for col in stats_df.columns if isinstance(col, tuple)]
            
            if metric_cols:
                # 按照proba_cols和metrics的顺序重新排列metric_cols
                ordered_metric_cols = []
                for proba_col in proba_cols:
                    for metric in metrics:
                        col_alias = metric_alias.get(metric, metric)
                        col_tuple = (proba_col, col_alias)
                        if col_tuple in metric_cols:
                            ordered_metric_cols.append(col_tuple)
                
                # 将basic_cols也转换为MultiIndex格式（第一层为空字符串，第二层为列名）
                # 这样所有列都会显示为两层结构
                basic_multi_cols = [('', col) for col in basic_cols]
                
                # 创建完整的MultiIndex列
                all_multi_cols = basic_multi_cols + ordered_metric_cols
                multi_index = pd.MultiIndex.from_tuples(all_multi_cols, names=['proba_col', 'metric'])
                
                # 重新组织DataFrame的列
                stats_df = stats_df[basic_cols + ordered_metric_cols]
                stats_df.columns = multi_index
            
    return stats_df


def group_metrics_many(
    label: pd.DataFrame,
    groupby_col_list,
    target_col: str,
    proba_cols,
    metrics=None,
    metric_alias=None,
    if_weight: bool = False,
    if_score: bool = False,
    multi_index: bool = False
):
    """
    多个分组列 / 多个分数列的统一指标计算接口。

    相当于对 `group_metrics` 在 `groupby_col_list` 上做循环计算。

    Parameters
    ----------
    label : pd.DataFrame
        输入数据。
    groupby_col_list : list of str or None
        分组字段列表：
        - 为 None：退化为不分组，仅计算总体 'all_data' 指标。
        - 为列表：依次对每个分组字段调用 `group_metrics`。
    target_col : str
        标签列名。
    proba_cols : str or list of str
        分数列名或其列表，含义同 `group_metrics` 中的 `proba_cols`。
    metrics : list or None
        需要计算的指标列表，含义同 `group_metrics`。
    metric_alias : dict or None
        指标别名字典，含义同 `group_metrics`。
    if_weight : bool, default False
        是否使用样本权重。
    if_score : bool, default False
        分数是否为“score”而非 proba，含义同 `group_metrics`。
    multi_index : bool, default False
        是否在返回的每个 DataFrame 中使用 MultiIndex 列。

    Returns
    -------
    dict or pd.DataFrame
        - 当 `groupby_col_list` 为 None：返回单个 DataFrame（与 `group_metrics` 一致）。
        - 否则：返回字典 {groupby_col: DataFrame}。
    """
    if groupby_col_list is None:
        print('groupby_col_list is None!!! return all_data metrics!!!')
        return group_metrics(
            label,
            None,
            target_col,
            proba_cols,
            metrics=metrics,
            metric_alias=metric_alias,
            if_weight=if_weight,
            if_score = if_score,
            multi_index=multi_index
        )
    else:
        all_df_dict = {}
        for groupby_col in groupby_col_list:
            all_df_dict[groupby_col] = group_metrics(
                label,
                groupby_col,
                target_col,
                proba_cols,
                metrics=metrics,
                metric_alias=metric_alias,
                if_weight=if_weight,
                if_score = if_score,
                multi_index=multi_index
            )
        return all_df_dict


def get_cross(
    label: pd.DataFrame,
    groupby_col_cross_list,
    target_col: str,
    proba_col,
    metrics=None,
    metric_alias=None,
    if_weight: bool = False,
    if_score: bool = False,
    multi_index: bool = False
):
    """
    交叉分组（多列组合）下的 KS / AUC / LIFT 等指标计算。

    Parameters
    ----------
    label : pd.DataFrame
        输入数据。
    groupby_col_cross_list : list of str
        需要做交叉的分组列列表，例如 ['qudao3_act', 'month']。
        函数内部会先拼接成临时列 '_tmp_cross_key' 再进行分组。
    target_col : str
        标签列名。
    proba_col : str or list of str
        分数列名或列表，含义同 `group_metrics` 的 `proba_cols`。
    metrics : list or None
        指标列表，默认 ['KS', 'AUC', 'SUB_5_LIFT', 'TOP_5_LIFT']。
    metric_alias : dict or None
        指标别名字典，含义同 `group_metrics`。
    if_weight : bool, default False
        是否使用样本权重。
    if_score : bool, default False
        分数是否为“score”而非 proba。
    multi_index : bool, default False
        是否使用 MultiIndex 列。

    Returns
    -------
    pd.DataFrame
        行：交叉组合后的各组以及 'all_data'；
        列：拆分后的原始分组列 + 指标列。
    """
    new_col = '_tmp_cross_key'
    label[new_col] = label[groupby_col_cross_list].agg('&'.join, axis=1)
    if metrics is None:
        metrics = ['KS', 'AUC', 'SUB_5_LIFT', 'TOP_5_LIFT']

    result = group_metrics(
        label,
        new_col,
        target_col,
        proba_col,
        metrics=metrics,
        metric_alias=metric_alias,
        if_weight=if_weight,
        if_score = if_score,
        multi_index=multi_index
    )
    label.drop(columns=[new_col],inplace=True)

    # 拆分组合列
    split_cols = result[new_col].str.split('&', expand=True)
    split_cols.columns = groupby_col_cross_list[:split_cols.shape[1]]
    result = pd.concat([split_cols, result.drop(columns=[new_col])], axis=1)
    return result


def get_month_psi_many(
    label: pd.DataFrame,
    groupby_col_list,
    proba_cols,
    month_col: str
):
    """
    多个分组列 / 多个分数列的月度 PSI 统一计算接口。

    Parameters
    ----------
    label : pd.DataFrame
        输入数据，至少包含 `month_col` 和 `proba_cols` 中的列。
    groupby_col_list : list of str or None
        分组字段列表：
        - 为 None：不分组，仅计算整体的月度 PSI。
        - 为列表：依次对每个分组字段调用 `get_month_psi`。
    proba_cols : str or list of str
        分数列名或其列表，用于计算 PSI。
    month_col : str
        月份字段列名，值应能按时间顺序比较，如 'YYYY-MM' 或 datetime 类型。

    Returns
    -------
    dict or pd.DataFrame
        - 当 `groupby_col_list` 为 None：返回整体 PSI 的 DataFrame。
        - 否则：返回字典 {groupby_col: DataFrame}。
    """
    all_df_dict = {}
    if groupby_col_list is None:
        print('groupby_col_list is None!!! return all_data psi!!!')
        return get_month_psi(
            label,
            None,
            proba_cols,
            month_col
        )
    else:
        for groupby_col in groupby_col_list:
            all_df_dict[groupby_col] = get_month_psi(
                label,
                groupby_col,
                proba_cols,
                month_col
            )
        return all_df_dict


def get_month_psi(
    label: pd.DataFrame,
    groupby_col,
    proba_cols,
    month_col: str
):
    """
    计算月度 PSI，可针对整体或指定分组。

    Parameters
    ----------
    label : pd.DataFrame
        输入数据，需包含 `month_col` 和 `proba_cols`。
    groupby_col : str or None
        分组字段：
        - 为 None：不分组，仅计算总体 PSI；
        - 为 str：按该列分组，返回每个分组 + 总体的月度 PSI。
    proba_cols : str or list of str
        分数列名或列表，用于计算 PSI。
    month_col : str
        月份字段列名，作为时间维度，内部会以最小月份作为基准。

    Returns
    -------
    pd.DataFrame
        行：月份 moth；
        列：各分数列及分组的 PSI 值，列名中带 `_psi` 后缀。
    """
    min_month = label[month_col].min()
    if isinstance(proba_cols, str):
        proba_cols = [proba_cols]
    if groupby_col is None:
        print('groupby_col is None!!! return all_data psi!!!')
        for proba_col in proba_cols:
            if proba_col == proba_cols[0]:
                all_data_psi_df_month = new_psi.choice_month_psi(label,min_month,proba_col,month_col)
                all_data_psi_df_month.rename(columns={'psi':f'{proba_col}_psi'},inplace=True)
            else:
                temp_psi_df = new_psi.choice_month_psi(label,min_month,proba_col,month_col)
                temp_psi_df.rename(columns={'psi':f'{proba_col}_psi'},inplace=True)
                all_data_psi_df_month = pd.merge(all_data_psi_df_month,temp_psi_df,on='moth',how='left')
    else:
        for proba_col in proba_cols:
            all_data_psi_df_month = new_psi.choice_month_psi(label,min_month,proba_col,month_col)
            all_data_psi_df_month.rename(columns={'psi':f'{proba_col}_psi'},inplace=True)
            groupby_psi_df = label.groupby(groupby_col).apply(
                new_psi.choice_month_psi,
                min_month,
                proba_col,
                month_col
            ).reset_index()
            groupby_psi_df.drop(columns=['level_1'],inplace=True)
            groupby_psi_df_pivot = groupby_psi_df.pivot_table(index='moth',columns=groupby_col,values='psi').reset_index()
            groupby_psi_df_pivot.columns = ['moth'] + list(groupby_psi_df_pivot.columns[1:]+'_psi')
            all_data_psi_df_month = pd.merge(all_data_psi_df_month,groupby_psi_df_pivot,on='moth',how='left')
    return all_data_psi_df_month


def source_plt_combined(
    label2: pd.DataFrame,
    source_col,
    score_col,
    if_score: bool = False,
    save_path: str = None,
    group_list=None
):
    """
    将各 source 分布图拼接成网格（直方图 + KDE），可选择保存或返回图像。

    Parameters
    ----------
    label2 : pd.DataFrame
        输入数据。
    source_col : str or None
        分组字段列名，例如渠道、月份等：
        - 为 None：只绘制整体一张图（'all_data'）。
        - 为 str：按该字段取唯一值或 `group_list` 指定的值分组绘制。
    score_col : str
        分数列名，可以是 proba 或 score。
    if_score : bool, default False
        - False：`score_col` 为 proba，x 轴标题为 'Proba'；
        - True：`score_col` 为 score，x 轴标题为 'Score'。
    save_path : str or None, default None
        - 若为字符串：图片保存到该路径，并调用 `plt.show()`，函数返回 None；
        - 若为 None：不主动显示，返回 `matplotlib.figure.Figure` 对象，方便在外部显示或保存。
    group_list : list or None, default None
        - 为 None：自动使用 `label2[source_col].unique()` 作为分组列表；
        - 为列表：只绘制列表中指定的分组。

    Returns
    -------
    matplotlib.figure.Figure or None
        - `save_path` 为 None：返回 figure 对象；
        - 否则：保存图片并返回 None。
    """
    
    # 设置样式
    plt.style.use('seaborn-v0_8-deep')  # 更新为新版seaborn样式
    # 设置中文字体（黑体）并解决负号显示问题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 确定要绘制的source列表
    if source_col is None:
        print('source_col is None!!! only return all_data plt!!!')
        sources = []
    else:
        if group_list is None:
            print('group_list is None!!! return the unique values of the source_col column!!!')
            sources = list(label2[source_col].unique())
        else:
            sources = group_list.copy()
    sources.append('all_data')

    # 每行3个子图，如果不足3个，则按实际个数显示
    if len(sources)<=3:
        n_cols = len(sources)
    else:
        n_cols = 3  
    n_rows = (len(sources) + n_cols - 1) // n_cols  # 向上取整

    # 创建图形和子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10+5*(n_rows-2)))
    # 处理不同情况：1x1返回单个对象，需要转为数组；其他情况展平为1D数组
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])  # 单个对象转为数组
    else:
        axes = axes.flatten()  # 数组展平为1D
    
    x_axis_name = 'Score' if if_score else 'Proba'

    for i, source in enumerate(sources):
        if source == 'all_data':
            sns.histplot(
                label2[score_col],
                bins=30,
                kde=True,
                alpha=0.7,
                color='skyblue',
                edgecolor='black',
                ax=axes[i]
            )
        else:
            sns.histplot(
                label2[label2[source_col] == source][score_col],
                bins=30,
                kde=True,
                alpha=0.7,
                color='skyblue',
                edgecolor='black',
                ax=axes[i]
            )
        
        # 设置子图样式
        axes[i].set_title(f'{source}', fontsize=12)
        axes[i].set_xlabel(x_axis_name, fontsize=10)
        axes[i].set_ylabel('Count', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(len(sources), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path is not None:
        # 如果提供了路径，保存图片并显示
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return None  # 已保存，无需返回对象
    else:
        # 如果没提供路径，返回 figure 对象，不显示
        return fig


def create_example_data():
    """
    构造一份用于测试的示例数据集。

    字段说明
    --------
    - mobile : str
        虚拟手机号（字符串）。
    - backPointTime : datetime
        回传时间，最近 30 天内的随机时间。
    - target : int
        标签，0/1 二分类。
    - qudao3_act : str
        渠道 / 来源类别，['qudao1', 'qudao2', 'qudao3']。
    - ronghe_proba, ronghe_proba2 : float
        模拟的 proba 分数（坏样本在 0.7~1.0，好样本在 0.0~0.4）。
    - ronghe_score, ronghe_score2 : float
        模拟的 score 分数（坏样本在 300~550，好样本在 550~900）。

    Returns
    -------
    pd.DataFrame
        含上述字段的示例数据，可用于测试本文件中的各类函数。
    """

    # 设置随机种子保证可复现
    np.random.seed(42)

    # 数据总量
    n_samples = 10000

    # 生成target列（95%为1）
    target = np.random.choice([0, 1], size=n_samples, p=[0.05, 0.95])

    # 为不同target生成qudao3_act分类变量
    # target=1: 更倾向"high"类别；target=0: 更倾向"low"类别
    qudao3_act = np.where(target == 1,
                        np.random.choice(['qudao1', 'qudao2', 'qudao3'], size=n_samples, p=[0.6, 0.3, 0.1]),
                        np.random.choice(['qudao1', 'qudao2', 'qudao3'], size=n_samples, p=[0.1, 0.3, 0.6]))

    # 生成DataFrame
    df = pd.DataFrame({
        'mobile': ['1' + ''.join(np.random.choice([str(i) for i in range(10)], 10)) for _ in range(n_samples)],
        
        'backPointTime': [
            datetime.now() - timedelta(days=np.random.randint(0, 30), 
                                    hours=np.random.randint(0, 24),
                                    minutes=np.random.randint(0, 60))
            for _ in range(n_samples)
        ],
        
        'target': target,
        
        'qudao3_act': qudao3_act,
        
        'ronghe_proba': np.where(target == 1,
                                np.random.uniform(0.7, 1.0, n_samples),
                                np.random.uniform(0.0, 0.4, n_samples)),

        'ronghe_proba2': np.where(target == 1,
                                np.random.uniform(0.7, 1.0, n_samples),
                                np.random.uniform(0.0, 0.4, n_samples)),
        
        'ronghe_score': np.where(target == 1,
                                np.random.uniform(300, 550, n_samples),
                                np.random.uniform(550, 900, n_samples)),

        'ronghe_score2': np.where(target == 1,
                                np.random.uniform(300, 550, n_samples),
                                np.random.uniform(550, 900, n_samples))
    })

    # 调整数据类型
    df['mobile'] = df['mobile'].astype(str)
    df['target'] = df['target'].astype(int)
    return df


# if __name__ == '__main__':
#     ### example
#     # data = pd.read_pickle(r'D:\work\06.std_product\xyf\TZ_03F_D1T2Z2_score_proba.pkl') 
#     # 这个数据就是一个打完分的数据，包括mobile backPointTime target qudao3_act target ronghe_proba ronghe_score这几列

#     data = create_example_data() # 一个测试数据，输出结果可以忽略，仅作测试
#     data['month'] = pd.to_datetime(data['backPointTime']).dt.strftime('%Y-%m')

#     # 这里展示的都是proba作为输入特征的情况，如果采用score作为输入特征的话，需要修改if_score=True的参数
#     # group_metrics系列的函数提供两种参数输出的形式，multi_index=True和multi_index=False，分别为是否双层索引，默认为无
#     # 注意所有的proba_col的参数输入既可以是单个col也可以是一组col
#     # 只计算一个分组特征的ks和auc
#     # groupby_col/groupby_col_list == None 的时候向下兼容所有数据，即不分组的数据的各项统计指标计算
#     all_metrics = group_metrics(data,'qudao3_act','target','ronghe_proba') # 默认不限定metrics的话，会计算KS、AUC、SUB_5_LIFT、TOP_5_LIFT
#     ks_auc = group_metrics(data,'qudao3_act','target','ronghe_proba',metrics=['KS','AUC'])
#     single_ks = group_metrics(data,'qudao3_act','target','ronghe_proba',metrics=['KS'])
#     single_auc = group_metrics(data,'qudao3_act','target','ronghe_proba',metrics=['AUC'])
#     single_sub_5_lift = group_metrics(data,'qudao3_act','target','ronghe_proba',metrics=['SUB_5_LIFT'])
#     single_top_5_lift = group_metrics(data,'qudao3_act','target','ronghe_proba',metrics=['TOP_5_LIFT'])

#     # 交叉分析ks和auc
#     cross_result = get_cross(data,['qudao3_act','month'],'target','ronghe_proba',metrics=['KS','AUC'])

#     # psi
#     # psi中也可以计算多个分组或者多个分数列的psi
#     month_psi = get_month_psi(data,['qudao3_act'],'ronghe_proba','month') # 指定列的psi cal
#     month_psi_all = get_month_psi(data,None,'ronghe_proba','month')
#     month_psi_many = get_month_psi_many(data,['qudao3_act','month'],['ronghe_proba'],'month')

#     # 多个分组特征的ks和auc
#     many_ks_auc = group_metrics_many(data,['qudao3_act','month'],'target',['ronghe_proba'],metrics=['KS','AUC'])
#     many_ks = group_metrics_many(data,['qudao3_act','month'],'target',['ronghe_proba'],metrics=['KS'])
#     many_auc = group_metrics_many(data,['qudao3_act','month'],'target',['ronghe_proba'],metrics=['AUC'])

if __name__ == '__main__':

    """
        # data = pd.read_pickle('D:/work/06.std_product/xyf/TZ_03F_D1T2Z2_score_proba.pkl') 
        # 这个数据就是一个打完分的数据，包括mobile backPointTime target qudao3_act target ronghe_proba ronghe_score这几列
        下面是基于测试数据生成的AI测试代码：

        使用 `create_example_data()` 生成的模拟数据，对本文件中的主要函数做一次尽量全面的自测。

        目标：
        - 覆盖：get_metric_stats / get_group_stats_basic / group_metrics / group_metrics_many /
                get_cross / get_month_psi / get_month_psi_many / source_plt_combined
        - 同时演示：proba / score 两种模式、是否加权、是否多分数组合、是否 multi_index 等典型参数组合。
    """

    # ------------------------------------------------------------------
    # 1. 构造基础数据：增加 month 和 weight 列
    # ------------------------------------------------------------------
    data = create_example_data()  # 一个测试数据，输出结果可以忽略，仅作测试
    data['month'] = pd.to_datetime(data['backPointTime']).dt.strftime('%Y-%m')

    # 构造一个简单的 weight 列：坏样本权重大、好样本权重小，便于验证加权效果
    data['weight'] = np.where(data['target'] == 1, 2.0, 0.5)

    print("\n===== 基础数据预览 =====")
    print(data.head())

    # ------------------------------------------------------------------
    # 2. get_metric_stats：单组样本 KS/AUC/LIFT（含加权 / score 模式）
    # ------------------------------------------------------------------
    print("\n===== 测试 get_metric_stats =====")

    # 2.1 仅 KS / AUC（proba）
    metric_basic = get_metric_stats(
        data,
        score_col='ronghe_proba',
        target_col='target',
        metrics=['KS', 'AUC'],
        if_only=True,
        if_weight=False,
        if_score=False
    )
    print("unweighted KS/AUC:\n", metric_basic)

    # 2.2 全部指标 + 加权
    metric_weighted = get_metric_stats(
        data,
        score_col='ronghe_proba',
        target_col='target',
        weight_col='weight',
        metrics=None,          # 默认全指标
        if_only=True,
        if_weight=True,
        if_score=False
    )
    print("weighted all metrics:\n", metric_weighted)

    # 2.3 以 score 模式输入（if_score=True）
    metric_score_mode = get_metric_stats(
        data,
        score_col='ronghe_score',
        target_col='target',
        weight_col='weight',
        metrics=['KS', 'AUC', 'SUB_5_LIFT', 'TOP_5_LIFT'],
        if_only=True,
        if_weight=True,
        if_score=True
    )
    print("score mode metrics:\n", metric_score_mode)

    # ------------------------------------------------------------------
    # 3. get_group_stats_basic：无分组 / 有分组 + 加权 / 不加权
    # ------------------------------------------------------------------
    print("\n===== 测试 get_group_stats_basic =====")
    basic_all = get_group_stats_basic(data, None, 'target', if_weight=False)
    print("no group (count):\n", basic_all)

    basic_group = get_group_stats_basic(data, 'qudao3_act', 'target', if_weight=False)
    print("group by qudao3_act (count):\n", basic_group.head())

    basic_group_weighted = get_group_stats_basic(data, 'qudao3_act', 'target', if_weight=True)
    print("group by qudao3_act (weighted):\n", basic_group_weighted.head())

    # ------------------------------------------------------------------
    # 4. group_metrics：单/多分数列、是否 multi_index、是否加权、score 模式
    # ------------------------------------------------------------------
    print("\n===== 测试 group_metrics =====")

    # 4.1 单分数列，默认全指标，按渠道分组
    gm_all_metrics = group_metrics(
        data,
        groupby_col='qudao3_act',
        target_col='target',
        proba_cols='ronghe_proba'
    )
    print("single proba, all metrics:\n", gm_all_metrics.head())

    # 4.2 限定指标 KS/AUC
    gm_ks_auc = group_metrics(
        data,
        groupby_col='qudao3_act',
        target_col='target',
        proba_cols='ronghe_proba',
        metrics=['KS', 'AUC']
    )
    print("KS/AUC only:\n", gm_ks_auc.head())

    # 4.3 多分数列（proba），不使用 multi_index
    gm_multi_proba = group_metrics(
        data,
        groupby_col='qudao3_act',
        target_col='target',
        proba_cols=['ronghe_proba', 'ronghe_proba2'],
        metrics=['KS', 'AUC', 'SUB_5_LIFT', 'TOP_5_LIFT'],
        multi_index=False
    )
    print("multi proba, flat columns:\n", gm_multi_proba.head())

    # 4.4 多分数列 + multi_index=True
    gm_multi_proba_mi = group_metrics(
        data,
        groupby_col='qudao3_act',
        target_col='target',
        proba_cols=['ronghe_proba', 'ronghe_proba2'],
        metrics=['KS', 'AUC'],
        multi_index=True
    )
    print("multi proba, MultiIndex columns:\n", gm_multi_proba_mi.head())

    # 4.5 不分组（groupby_col=None），仅整体指标
    gm_all_data = group_metrics(
        data,
        groupby_col=None,
        target_col='target',
        proba_cols='ronghe_proba',
        metrics=['KS', 'AUC']
    )
    print("no group, all_data only:\n", gm_all_data)

    # 4.6 加权 + score 模式
    gm_score_weighted = group_metrics(
        data,
        groupby_col='qudao3_act',
        target_col='target',
        proba_cols=['ronghe_score', 'ronghe_score2'],
        metrics=['KS', 'AUC', 'SUB_5_LIFT', 'TOP_5_LIFT'],
        if_weight=True,
        if_score=True,
        multi_index=True
    )
    print("score mode + weighted, MultiIndex:\n", gm_score_weighted.head())

    # ------------------------------------------------------------------
    # 5. group_metrics_many：多个分组字段
    # ------------------------------------------------------------------
    print("\n===== 测试 group_metrics_many =====")
    gm_many = group_metrics_many(
        data,
        groupby_col_list=['qudao3_act', 'month'],
        target_col='target',
        proba_cols=['ronghe_proba'],
        metrics=['KS', 'AUC']
    )
    for gcol, df_tmp in gm_many.items():
        print(f"\n-- group_metrics_many for groupby_col={gcol} --")
        print(df_tmp.head())

    # 不分组模式（groupby_col_list=None）
    gm_many_all = group_metrics_many(
        data,
        groupby_col_list=None,
        target_col='target',
        proba_cols=['ronghe_proba'],
        metrics=['KS', 'AUC']
    )
    print("\n-- group_metrics_many with groupby_col_list=None --")
    print(gm_many_all)

    # ------------------------------------------------------------------
    # 6. get_cross：交叉分组（例如 渠道 x 月份）
    # ------------------------------------------------------------------
    print("\n===== 测试 get_cross =====")
    cross_result = get_cross(
        data,
        groupby_col_cross_list=['qudao3_act', 'month'],
        target_col='target',
        proba_col='ronghe_proba',
        metrics=['KS', 'AUC']
    )
    print("cross KS/AUC by qudao3_act & month:\n", cross_result.head())

    # ------------------------------------------------------------------
    # 7. get_month_psi / get_month_psi_many：PSI 计算（整体 + 分组 + 多分组）
    # ------------------------------------------------------------------
    print("\n===== 测试 get_month_psi / get_month_psi_many =====")

    # 7.1 单分数列 + 单分组
    month_psi_qudao = get_month_psi(
        data,
        groupby_col='qudao3_act',
        proba_cols='ronghe_proba',
        month_col='month'
    )
    print("month PSI with groupby_col='qudao3_act':\n", month_psi_qudao.head())

    # 7.2 单分数列，不分组（整体 PSI）
    month_psi_all = get_month_psi(
        data,
        groupby_col=None,
        proba_cols='ronghe_proba',
        month_col='month'
    )
    print("month PSI all_data:\n", month_psi_all.head())

    # 7.3 get_month_psi_many 不分组（整体 PSI）
    month_psi_many_all = get_month_psi_many(
        data,
        groupby_col_list=None,
        proba_cols=['ronghe_proba'],
        month_col='month'
    )
    print("\n-- get_month_psi_many with groupby_col_list=None --")
    print(month_psi_many_all.head())

    # ------------------------------------------------------------------
    # 8. source_plt_combined：绘制分布图（proba / score，保存 / 返回 figure）
    # ------------------------------------------------------------------
    print("\n===== 测试 source_plt_combined（图像仅示例，不一定在所有环境中展示）=====")
    qudao_list = ['qudao1', 'qudao2']
    month_list = sorted(data['month'].unique())

    # 8.1 不指定 group_list：绘制所有渠道 + all_data，按 proba
    source_plt_combined(
        data,
        source_col='qudao3_act',
        score_col='ronghe_proba',
        if_score=False,
        save_path='example_qudao_proba_dist.png'
    )

    # 8.2 指定 group_list：仅绘制部分渠道
    source_plt_combined(
        data,
        source_col='qudao3_act',
        score_col='ronghe_proba',
        if_score=False,
        save_path='example_qudao_subset_proba_dist.png',
        group_list=qudao_list
    )

    # 8.3 以 score 模式绘制按月分布，保存为图片
    source_plt_combined(
        data,
        source_col='month',
        score_col='ronghe_score',
        if_score=True,
        save_path='example_month_score_dist.png',
        group_list=list(month_list)
    )

    # 8.4 不指定 save_path，直接返回 figure 对象（示例）
    fig = source_plt_combined(
        data,
        source_col='qudao3_act',
        score_col='ronghe_proba',
        if_score=False,
        group_list=qudao_list
    )
    # 在脚本环境中保存，Jupyter 中也可以直接 display(fig)
    fig.savefig('example_qudao_proba_dist_return_fig.png', dpi=300, bbox_inches='tight')

    print("\n===== 自测脚本执行完毕 =====\n")
