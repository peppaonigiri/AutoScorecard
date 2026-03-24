"""
TzScoreGroupTools.py
评分分组分析工具类，封装了KS、AUC、LIFT、PSI等指标的批量计算功能，
支持分组分析、交叉分析、加权计算、多分数列对比等场景。
date: 2025-12-30
version: 2.0.1
"""

from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import sys,os
import warnings
warnings.filterwarnings('ignore')

# sys.path.append(r'D:\work\scorecard')
from new_tools import new_psi
from new_tools import excel_utils
from new_tools import plt_utils


class TzScoreGroupTools:
    """
    评分分组分析工具类
    
    用于计算KS、AUC、LIFT、PSI等指标，支持分组分析、交叉分析、
    加权计算、多分数列对比等功能。
    也可以用来做label ans
    """
    
    def __init__(self, label_df: pd.DataFrame, target_col: str = 'target',month_col = 'month',
                 groupby_list: list = None, proba_list: list = None, 
                 if_weight: bool = False, if_score = False, 
                 metrics_lst: list = None, multi_index: bool = False):
        """
        初始化评分分组工具类
        
        Parameters
        ----------
        label_df : pd.DataFrame
            输入数据框，必须包含target_col和proba_list中指定的列
        target_col : str, default 'target'
            目标变量列名，约定1为"坏样本"、0为"好样本"
        groupby_list : list or None, default None
            默认分组字段列表
        proba_list : list or None, default None
            默认分数列名列表
        if_weight : bool, default False
            是否默认使用样本权重（需要数据包含'weight'列）
        metrics_lst : list or None, default None
            默认计算的指标列表，支持['KS', 'AUC', 'SUB_5_LIFT', 'TOP_5_LIFT']
        multi_index : bool, default False
            是否默认使用多层索引（当proba_list和metrics都多于1个时）
        """
        self.label_df = label_df.copy()
        self.target_col = target_col
        self.month_col = month_col
        self.groupby_list = groupby_list or []
        self.proba_list = proba_list or []
        self.if_weight = if_weight
        self.if_score = if_score
        self.metrics_lst = metrics_lst or ['KS', 'AUC', 'SUB_5_LIFT', 'TOP_5_LIFT']
        self.multi_index = multi_index
        self.reporting_flag = 0
        
        # 验证必要列
        self._validate_columns()
        
        # 设置默认指标别名
        self.metric_alias = {
            'KS': 'ks',
            'AUC': 'auc',
            'SUB_5_LIFT': 'sub_5_lift',
            'TOP_5_LIFT': 'top_5_lift',
        }
    
    def _validate_columns(self):
        """验证数据中是否存在必要的列"""
        if self.target_col not in self.label_df.columns:
            raise ValueError(f"数据中不存在目标列: {self.target_col}")
        
        if self.if_weight and 'weight' not in self.label_df.columns:
            raise ValueError("if_weight=True 但数据中不存在 'weight' 列")
        
        for col in self.proba_list:
            if col not in self.label_df.columns:
                raise ValueError(f"数据中不存在分数列: {col}")
    
    def _get_default(self, value, default_attr):
        """获取参数值，若未提供则使用类属性默认值"""
        return value if value is not None else default_attr
    
    def get_metric_stats(self, group: pd.DataFrame = None, score_col: str = None, 
                        metrics: list = None, if_only: bool = True, 
                        if_weight: bool = None, if_score: bool = None) -> pd.Series or pd.DataFrame:
        """
        计算单个样本集合的指标
        
        Parameters
        ----------
        group : pd.DataFrame or None, default None
            输入数据，如果为None则使用类的label_df
        score_col : str or None, default None
            分数列名，如果为None则使用类的proba_list第一个元素
        metrics : list or None, default None
            需要计算的指标列表
        if_only : bool, default True
            - True: 返回pd.Series（只包含指标值）
            - False: 返回pd.DataFrame，附带样本量、正负样本数等信息
        if_weight : bool or None, default None
            是否使用样本权重，如果为None则使用类的if_weight属性
        if_score : bool, default None
            - False: score_col为proba，值越大"越坏"
            - True: score_col为score，值越小"越坏"，AUC和lift内部会做方向调整
        
        Returns
        -------
        pd.Series or pd.DataFrame
            根据if_only返回不同类型的结果
        """
        group = self._get_default(group, self.label_df)
        score_col = self._get_default(score_col, self.proba_list[0] if self.proba_list else None)
        metrics = self._get_default(metrics, self.metrics_lst)
        if_weight = self._get_default(if_weight, self.if_weight)
        if_score = self._get_default(if_score, self.if_score)
        
        if score_col is None:
            raise ValueError("必须指定score_col或初始化时设置proba_list")
        
        # 验证指标
        metrics = [metric.upper() for metric in metrics]
        supported_metrics = {'KS', 'AUC', 'SUB_5_LIFT', 'TOP_5_LIFT'}
        unknown = set(metrics) - supported_metrics
        if unknown:
            raise ValueError(f"暂不支持的指标: {unknown}")

        sample_weight = group['weight'] if if_weight else None
        results = {}

        # KS / AUC
        if ('KS' in metrics) or ('AUC' in metrics):
            fpr, tpr, _ = roc_curve(group[self.target_col], group[score_col], sample_weight=sample_weight)
            if 'KS' in metrics:
                results['KS'] = abs(fpr - tpr).max()
            if 'AUC' in metrics:
                results['AUC'] = auc(fpr, tpr)
                if if_score:
                    results['AUC'] = 1 - results['AUC']

        # top/sub 5% lift
        if ('TOP_5_LIFT' in metrics) or ('SUB_5_LIFT' in metrics):
            # 1/9 认为不需要这里的反向
            # if if_score:
            #     proab_score_col = group[score_col].copy()
            # else:
            #     proab_score_col = -group[score_col].copy()
            proab_score_col = group[score_col].copy()

            high_thr = proab_score_col.quantile(0.95)
            low_thr = proab_score_col.quantile(0.05)
            top5 = group[proab_score_col >= high_thr]
            bottom5 = group[proab_score_col <= low_thr]
            
            bad_rate_all = (group[self.target_col] == 1).mean()
            bad_rate_top5 = (top5[self.target_col] == 1).mean()
            bad_rate_bottom5 = (bottom5[self.target_col] == 1).mean()
            
            lift_top5 = bad_rate_top5 / bad_rate_all if bad_rate_all > 0 else float('nan')
            lift_bottom5 = bad_rate_bottom5 / bad_rate_all if bad_rate_all > 0 else float('nan')

            if 'SUB_5_LIFT' in metrics:
                results['SUB_5_LIFT'] = lift_bottom5
            if 'TOP_5_LIFT' in metrics:
                results['TOP_5_LIFT'] = lift_top5

        if if_only:
            return pd.Series(results)
        else:
            extra_stats = {
                '样本量': len(group),
                '负样本数': group[self.target_col].sum(),
                '正样本数': len(group) - group[self.target_col].sum()
            }
            extra_stats.update(results)
            return pd.DataFrame([extra_stats])
    
    def get_group_stats_basic(self, groupby_col: str = None, if_weight: bool = None) -> pd.DataFrame:
        """
        计算基础分组统计（样本数/正负样本数/坏账率）
        
        Parameters
        ----------
        groupby_col : str or None, default None
            分组字段，如果为None则使用类的groupby_list第一个元素
        if_weight : bool or None, default None
            是否使用权重，如果为None则使用类的if_weight属性
            
        Returns
        -------
        pd.DataFrame
            包含分组统计信息的数据框
        """
        # groupby_col = self._get_default(groupby_col, self.groupby_list[0] if self.groupby_list else None)
        if_weight = self._get_default(if_weight, self.if_weight)
        
        label = self.label_df.copy()
        if 'index' in label.columns:
            label.rename(columns={'index': 'index_'}, inplace=True)
        label = label.reset_index()
        
        if groupby_col is None:
            # 不分组，只统计全量
            if if_weight:
                ks_stats_all = label.groupby([self.target_col])['weight'].sum()
            else:
                ks_stats_all = label.groupby([self.target_col])['index'].count()
            
            ks_stats_all.index.name = None
            ks_stats_all = ks_stats_all.to_frame().T
            ks_stats_all.index = ['all_data']
            ks_stats_all = ks_stats_all.reset_index()
            ks_stats_all.rename(columns={1.0: '负样本数', 0.0: '正样本数', 'index': 'all'}, inplace=True)
        else:
            # 按分组字段统计
            if if_weight:
                if 'weight' not in label.columns:
                    raise ValueError('weight column is not in the label, please check the label and the name!!!')
                ks_stats_all = label.groupby([groupby_col, self.target_col])['weight'].sum().reset_index().pivot_table(
                    index=groupby_col, columns=self.target_col, values='weight').reset_index()
            else:
                ks_stats_all = label.groupby([groupby_col, self.target_col])['index'].count().reset_index().pivot_table(
                    index=groupby_col, columns=self.target_col, values='index').reset_index()

            ks_stats_all.rename(columns={1.0: '负样本数', 0.0: '正样本数'}, inplace=True)
            
            # 添加汇总行
            total_row = pd.DataFrame([['all', ks_stats_all['正样本数'].sum(), ks_stats_all['负样本数'].sum()]],
                                   columns=[groupby_col, '正样本数', '负样本数'])
            ks_stats_all = pd.concat([ks_stats_all, total_row], axis=0)
        
        ks_stats_all.fillna(0, inplace=True)
        ks_stats_all['样本数'] = ks_stats_all['负样本数'] + ks_stats_all['正样本数']
        ks_stats_all['坏账率'] = ks_stats_all['负样本数'] / ks_stats_all['样本数']
        return ks_stats_all
    
    def group_metrics(self, groupby_col: str = None, proba_cols: list = None, 
                     metrics: list = None, metric_alias: dict = None, 
                     if_weight: bool = None, if_score: bool = None, 
                     multi_index: bool = None) -> pd.DataFrame:
        """
        统一的"单个分组列/多个分数列"的指标计算接口
        
        Parameters
        ----------
        groupby_col : str or None, default None
            分组字段，如果为None则使用类的groupby_list第一个元素
        proba_cols : str or list or None, default None
            分数列名，如果为None则使用类的proba_list
        metrics : list or None, default None
            需要计算的指标列表
        metric_alias : dict or None, default None
            指标别名字典
        if_weight : bool or None, default None
            是否使用权重
        if_score : bool, default None
            是否为score模式
        multi_index : bool or None, default None
            是否使用多层索引，如果为None则使用类的multi_index属性
            
        Returns
        -------
        pd.DataFrame
            包含分组统计和指标计算结果的数据框
        """
        # groupby_col = self._get_default(groupby_col, self.groupby_list[0] if self.groupby_list else None)
        proba_cols = self._get_default(proba_cols, self.proba_list)
        metrics = self._get_default(metrics, self.metrics_lst)
        if_weight = self._get_default(if_weight, self.if_weight)
        multi_index = self._get_default(multi_index, self.multi_index)
        if_score = self._get_default(if_score, self.if_score)
        
        if metric_alias is None:
            metric_alias = self.metric_alias

        if isinstance(proba_cols, str):
            proba_cols = [proba_cols]
        
        single_proba = len(proba_cols) == 1
        use_multi_index = multi_index and len(proba_cols) > 1 and len(metrics) > 1

        # 获取基础统计
        stats_df = self.get_group_stats_basic(groupby_col, if_weight)

        if self.reporting_flag == 0 :
            if groupby_col is None:
                print('groupby_col is None!!! return all_data metrics!!!')

        metric_data_dict = {}
        
        # 遍历所有分数列和指标
        for proba_col in proba_cols:
            for metric in metrics:
                # 计算总体值
                total_value = self.get_metric_stats(
                    group=self.label_df,
                    score_col=proba_col,
                    metrics=[metric],
                    if_weight=if_weight,
                    if_score=if_score
                )[metric]
                
                col_alias = metric_alias.get(metric, metric)
                
                if use_multi_index:
                    # 使用多层索引模式
                    if proba_col not in metric_data_dict:
                        metric_data_dict[proba_col] = {}
                    
                    metric_data_dict[proba_col][col_alias] = {
                        'total_value': total_value,
                        'groupby_data': None
                    }
                    
                    if groupby_col is not None:
                        # 分组计算
                        metric_df = self.label_df.groupby(groupby_col).apply(
                            lambda x: self.get_metric_stats(
                                group=x,
                                score_col=proba_col,
                                metrics=[metric],
                                if_weight=if_weight,
                                if_score=if_score
                            )
                        ).reset_index()
                        metric_df = metric_df[[groupby_col, metric]].rename(columns={metric: 'value'})
                        metric_data_dict[proba_col][col_alias]['groupby_data'] = metric_df
                else:
                    # 使用单层索引模式
                    if not single_proba:
                        col_alias = f'{proba_col}_{col_alias}'

                    if groupby_col is not None:
                        # 创建汇总行
                        all_row = pd.DataFrame([['all', total_value]], columns=[groupby_col, col_alias])
                        
                        # 分组计算
                        metric_df = self.label_df.groupby(groupby_col).apply(
                            lambda x: self.get_metric_stats(
                                group=x,
                                score_col=proba_col,
                                metrics=[metric],
                                if_weight=if_weight,
                                if_score=if_score
                            )
                        ).reset_index()
                        metric_df = metric_df[[groupby_col, metric]].rename(columns={metric: col_alias})
                        
                        # 合并汇总行和分组数据
                        metric_df = pd.concat([metric_df, all_row], axis=0)
                        stats_df = pd.merge(stats_df, metric_df, on=groupby_col, how='outer')
                    else:
                        # 只有汇总数据
                        all_row = pd.DataFrame([['all_data', total_value]], columns=['all', col_alias])
                        stats_df = pd.merge(stats_df, all_row, on='all', how='outer')
        
        # 处理多层索引的情况
        if use_multi_index:
            merge_key = groupby_col if groupby_col is not None else 'all'
            all_metric_dfs = []
            
            for proba_col in proba_cols:
                for col_alias in metric_data_dict[proba_col].keys():
                    total_value = metric_data_dict[proba_col][col_alias]['total_value']
                    groupby_data = metric_data_dict[proba_col][col_alias]['groupby_data']
                    
                    if groupby_col is not None:
                        all_row = pd.DataFrame([['all', total_value]], columns=[groupby_col, 'value'])
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
                
                stats_df = pd.merge(stats_df, combined_metric_df, on=merge_key, how='outer')
                
                # 设置多层索引
                basic_cols = [col for col in stats_df.columns if not isinstance(col, tuple)]
                metric_cols = [col for col in stats_df.columns if isinstance(col, tuple)]
                
                if metric_cols:
                    # 按proba_cols和metrics的顺序重新排列metric_cols
                    ordered_metric_cols = []
                    for p_col in proba_cols:
                        for m in metrics:
                            col_alias = metric_alias.get(m, m)
                            col_tuple = (p_col, col_alias)
                            if col_tuple in metric_cols:
                                ordered_metric_cols.append(col_tuple)
                    
                    # 为所有列创建多层索引
                    basic_multi_cols = [('', col) for col in basic_cols]
                    all_multi_cols = basic_multi_cols + ordered_metric_cols
                    multi_index_obj = pd.MultiIndex.from_tuples(all_multi_cols, names=['proba_col', 'metric'])
                    
                    stats_df = stats_df[basic_cols + ordered_metric_cols]
                    stats_df.columns = multi_index_obj
        
        return stats_df
    
    def group_metrics_many(self, groupby_col_list: list = None, proba_cols: list = None, 
                          metrics: list = None, metric_alias: dict = None, 
                          if_weight: bool = None, if_score: bool = None, 
                          multi_index: bool = None) -> dict or pd.DataFrame:
        """
        多个分组列的指标计算
        
        Parameters
        ----------
        groupby_col_list : list or None, default None
            分组字段列表，如果为None则使用类的groupby_list
        proba_cols : str or list or None, default None
            分数列名
        metrics : list or None, default None
            指标列表
        metric_alias : dict or None, default None
            指标别名字典
        if_weight : bool or None, default None
            是否使用权重
        if_score : bool, default False
            是否为score模式
        multi_index : bool or None, default None
            是否使用多层索引
            
        Returns
        -------
        dict or pd.DataFrame
            当groupby_col_list为None时返回单个DataFrame，否则返回字典
        """
        if self.reporting_flag == 0 :
            if groupby_col_list is None:
                print(
                """groupby_col_list is None!!!\
                    \n But this version cannt cal none with this function, will return all groupby_col_results!!!\
                    \n If you want to return all data metrics, you can use group_metrics function!!!"""
                )

        groupby_col_list = self._get_default(groupby_col_list, self.groupby_list)
        proba_cols = self._get_default(proba_cols, self.proba_list)
        metrics = self._get_default(metrics, self.metrics_lst)
        if_weight = self._get_default(if_weight, self.if_weight)
        multi_index = self._get_default(multi_index, self.multi_index)
        if_score = self._get_default(if_score, self.if_score)
        
        all_df_dict = {}
        for groupby_col in groupby_col_list:
            all_df_dict[groupby_col] = self.group_metrics(
                groupby_col=groupby_col,
                proba_cols=proba_cols,
                metrics=metrics,
                metric_alias=metric_alias,
                if_weight=if_weight,
                if_score=if_score,
                multi_index=multi_index
            )
        return all_df_dict
    
    def get_cross(self, groupby_col_cross_list: list, proba_col: str = None, 
                 metrics: list = None, metric_alias: dict = None, 
                 if_weight: bool = None, if_score: bool = None, 
                 multi_index: bool = None) -> pd.DataFrame:
        """
        交叉分组分析
        
        Parameters
        ----------
        groupby_col_cross_list : list
            需要做交叉的分组列列表，例如['qudao3_act', 'month']
        proba_col : str or None, default None
            分数列名，如果为None则使用类的proba_list第一个元素
        metrics : list or None, default None
            指标列表
        metric_alias : dict or None, default None
            指标别名字典
        if_weight : bool or None, default None
            是否使用权重
        if_score : bool, default False
            是否为score模式
        multi_index : bool or None, default None
            是否使用多层索引
            
        Returns
        -------
        pd.DataFrame
            交叉分组分析结果
        """
        proba_col = self._get_default(proba_col, self.proba_list[0] if self.proba_list else None)
        metrics = self._get_default(metrics, self.metrics_lst)
        if_weight = self._get_default(if_weight, self.if_weight)
        multi_index = self._get_default(multi_index, self.multi_index)
        if_score = self._get_default(if_score, self.if_score)
        
        if proba_col is None:
            raise ValueError("必须指定proba_col或初始化时设置proba_list")
        
        # 创建临时交叉列
        new_col = '_tmp_cross_key'
        self.label_df[new_col] = self.label_df[groupby_col_cross_list].agg('&'.join, axis=1)
        
        result = self.group_metrics(
            groupby_col=new_col,
            proba_cols=proba_col,
            metrics=metrics,
            metric_alias=metric_alias,
            if_weight=if_weight,
            if_score=if_score,
            multi_index=multi_index
        )
        
        self.label_df.drop(columns=[new_col], inplace=True)
        # 拆分组合列
        split_cols = result[new_col].str.split('&', expand=True)
        split_cols.columns = groupby_col_cross_list[:split_cols.shape[1]]
        result = pd.concat([split_cols, result.drop(columns=[new_col])], axis=1)
        return result
    
    def get_month_psi(self, month_col: str='month', groupby_col: str = None, 
                    proba_cols: list = None, multi_index: bool = None) -> pd.DataFrame:
        """
        计算月度PSI
        
        Parameters
        ----------
        month_col : str
            月份字段列名
        groupby_col : str or None, default None
            分组字段
        proba_cols : str or list or None, default None
            分数列名
        multi_index : bool or None, default None
            是否使用多层索引，如果为None则使用类的multi_index属性
                
        Returns
        -------
        pd.DataFrame
            PSI计算结果，行：月份，列：各分数列及分组的PSI值
        """
        proba_cols = self._get_default(proba_cols, self.proba_list)
        multi_index = self._get_default(multi_index, self.multi_index)
        month_col = self._get_default(month_col, self.month_col)

        if isinstance(proba_cols, str):
            proba_cols = [proba_cols]
        
        min_month = self.label_df[month_col].min()

        if self.reporting_flag == 0 :
            if groupby_col is None:
                print('groupby_col is None!!! return all_data psi!!!')
        
        all_dfs = []
        
        for proba_col in proba_cols:
            psi_df = new_psi.choice_month_psi(self.label_df, min_month, proba_col, month_col)
            
            if multi_index:
                psi_df = psi_df.rename(columns={'psi': (proba_col, 'psi')})
            else:
                psi_df = psi_df.rename(columns={'psi': f'{proba_col}_psi'})
            
            all_dfs.append(psi_df)
        
        # 这里算出来是所有proba_cols不分组的psi
        if all_dfs:
            result_df = all_dfs[0]
            for df in all_dfs[1:]:
                result_df = pd.merge(result_df, df, on='moth', how='left')
        else:
            return pd.DataFrame()
        
        if groupby_col is not None:
            for proba_col in proba_cols:

                def compute_group_psi(group):
                    return new_psi.choice_month_psi(group, min_month, proba_col, month_col)
            
                groupby_psi_df = self.label_df.groupby(groupby_col).apply(compute_group_psi).reset_index()

                if 'level_1' in groupby_psi_df.columns:
                    groupby_psi_df = groupby_psi_df.drop(columns=['level_1'])
                groupby_psi_df = groupby_psi_df.rename(columns={'psi': 'psi_value'})

                pivot_df = groupby_psi_df.pivot_table(
                    index='moth',
                    columns=groupby_col,
                    values='psi_value'
                ).reset_index()
                
                rename_dict = {}
                for group_val in pivot_df.columns[1:]:
                    if multi_index:
                        rename_dict[group_val] = (proba_col, f'{group_val}_psi')
                    else:
                        rename_dict[group_val] = f'{proba_col}_{group_val}_psi'
                if rename_dict:
                    pivot_df = pivot_df.rename(columns=rename_dict)
                result_df = pd.merge(result_df, pivot_df, on='moth', how='left')
        
        if multi_index and len(proba_cols) >= 1:
            basic_cols = ['moth']
            multi_cols = [col for col in result_df.columns if isinstance(col, tuple)]
            ordered_multi_cols = []
            for p_col in proba_cols:
                ordered_multi_cols.append((p_col, 'psi'))
                if groupby_col is not None:
                    group_psi_cols = [
                        col for col in multi_cols 
                        if isinstance(col, tuple) and col[0] == p_col and col[1].endswith('_psi') and col[1] != 'psi'
                    ]
                    group_psi_cols.sort(key=lambda x: x[1])
                    ordered_multi_cols.extend(group_psi_cols)
            result_df = result_df[basic_cols + ordered_multi_cols]
            column_tuples = []
            for col in result_df.columns:
                if col == 'moth':
                    column_tuples.append(('', 'moth'))
                else:
                    column_tuples.append(col)
            
            result_df.columns = pd.MultiIndex.from_tuples(
                column_tuples,
                names=['proba_col', 'metric']
            )
        
        return result_df
    
    def get_month_psi_many(self, month_col: str='month', groupby_col_list: list = None, 
                          proba_cols: list = None) -> dict or pd.DataFrame:
        """
        多个分组列的月度PSI计算
        
        Parameters
        ----------
        month_col : str
            月份字段列名
        groupby_col_list : list or None, default None
            分组字段列表，如果为None则使用类的groupby_list
        proba_cols : str or list or None, default None
            分数列名
            
        Returns
        -------
        dict or pd.DataFrame
            当groupby_col_list为None时返回单个DataFrame，否则返回字典
        """
        if self.reporting_flag == 0 :
            if groupby_col_list is None:
                print(
                """groupby_col_list is None!!!\
                    \n But this version cannt cal none with this function, will return all groupby_col_results!!!\
                    \n If you want to return all data psi, you can use get_month_psi function!!!"""
                )

        groupby_col_list = self._get_default(groupby_col_list, self.groupby_list)
        proba_cols = self._get_default(proba_cols, self.proba_list)
        month_col = self._get_default(month_col, self.month_col)

        all_df_dict = {}
        for groupby_col in groupby_col_list:
            if groupby_col == month_col:
                continue
            all_df_dict[groupby_col] = self.get_month_psi(month_col, groupby_col=groupby_col, proba_cols=proba_cols)
        return all_df_dict
    
    def source_plt_combined(self, score_col: str,if_score: bool = None, source_col: str = None, 
                            save_path: str = None, 
                           group_list: list = None):
        """
        绘制分布图
        
        Parameters
        ----------
        score_col : str
            分数列名
        source_col : str or None, default None
            分组字段列名
        if_score : bool, default False
            - False: score_col为proba，x轴标题为'Proba'
            - True: score_col为score，x轴标题为'Score'
        save_path : str or None, default None
            - 若为字符串：图片保存到该路径，并调用plt.show()，函数返回None
            - 若为None：不主动显示，返回matplotlib.figure.Figure对象
        group_list : list or None, default None
            - 为None：自动使用label_df[source_col].unique()作为分组列表
            - 为列表：只绘制列表中指定的分组
            
        Returns
        -------
        matplotlib.figure.Figure or None
            当save_path为None时返回figure对象，否则返回None
        """
        # 设置绘图样式

        if_score = self._get_default(if_score, self.if_score)

        plt.style.use('seaborn-v0_8-deep')
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 确定要绘制的source列表
        if source_col is None:
            if self.reporting_flag == 0 :
                print('source_col is None!!! only return all_data plt!!!')
            sources = []
        else:
            if group_list is None:
                if self.reporting_flag == 0 :
                    print('group_list is None!!! return the unique values of the source_col column!!!')
                sources = list(self.label_df[source_col].unique())
            else:
                sources = group_list.copy()
        
        sources.append('all_data')

        # 计算子图布局
        n_cols = min(len(sources), 3)
        n_rows = (len(sources) + n_cols - 1) // n_cols

        # 创建图形和子图
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()
        
        x_axis_name = 'Score' if if_score else 'Proba'

        # 绘制每个分组的分布图
        for i, source in enumerate(sources):
            if source == 'all_data':
                data_to_plot = self.label_df[score_col]
            else:
                data_to_plot = self.label_df[self.label_df[source_col] == source][score_col]
            
            sns.histplot(
                data_to_plot,
                bins=30,
                kde=True,
                alpha=0.7,
                color='skyblue',
                edgecolor='black',
                ax=axes[i]
            )
            
            axes[i].set_title(f'{source}', fontsize=12)
            axes[i].set_xlabel(x_axis_name, fontsize=10)
            axes[i].set_ylabel('Count', fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(len(sources), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            return None
        else:
            return fig
    

    def get_all_report_df(self):
        """
            这是一个补充的函数，用于获取所有分组的报告数据，展示较为好看些
            但是不能做到完完全全的最优，例如当多个数据渠道时，展示最优的方式是分为两列去展示
            这个单独领出来写是万一这个数据单独需要的时候
        """
        report_all_df = self.group_metrics(multi_index=True)
        if 'all' not in report_all_df.columns:
            report_all_df = report_all_df.loc[:, [i for i in report_all_df.columns if i[0] != '']].stack(level=0).reset_index().drop(columns=['level_0'])
        else:
            report_all_df.drop(columns=['all','正样本数','负样本数','样本数','坏账率'], inplace=True)
            report_all_df = report_all_df.reindex(sorted(report_all_df.columns), axis=1)
            report_all_df.insert(0, 'proba_col', self.proba_list)
            
        return report_all_df

    def plot_all_proba_all_data(self,source_col: str = None,n_cols: int = 3, if_score=None):
        """
            绘制各个概率下的所有数据的分布图
            这个单独拎出来写是万一这个东西是单独需要的时候

            parameters
            ----------
            source_col : str, default None
                分组字段列名
            n_cols : int, default 3
                图片的列数
            if_score : bool, default None
                是否使用score作为x轴的名称（这里是仅仅名称的修改，不是实际的分数）
        """
        if_score = self._get_default(if_score, self.if_score)
        fig_list = []
        for proba_col in self.proba_list:
            fig_list.append(self.source_plt_combined(score_col = proba_col,source_col = source_col, if_score=if_score))
        fig = plt_utils.concat_figures(fig_list,self.proba_list,n_cols=n_cols)
        return fig

    
    def create_report(self,save_path: str = None, phs_n_cols_simmple: int = 3,phs_n_cols_detail: int = 3, if_simple=True, if_detail=True, if_psi=True, if_score=None,if_detail_plot=True):
        """
            生成最终报告
            
            Parameters
            ----------
            save_path : str
                报告的保存文件夹
            phs_n_cols_simmple : int, default 3
                简单报告中图片的列数
            phs_n_cols_detail : int, default 3
                详细报告中图片的列数
            if_simple : bool, default True
                是否生成简单报告
            if_detail : bool, default True
                是否生成详细报告
            if_psi : bool, default True
                是否生成psi报告
            if_score : bool, default None
                是否使用score作为x轴的名称（这里是仅仅名称的修改，不是实际的分数）
            if_detail_plot : bool, default True
                是否生成详细报告中图片
            
            Returns
            -------
            matplotlib.figure.Figure or None
                当save_path为None时返回figure对象，否则返回None
        """
        if_score = self._get_default(if_score, self.if_score)
        print('######################### Start create report #########################')
        self.reporting_flag = 1

        if if_simple:
            print('######################### Start create simple report #########################')
            # part1 数据简报
            # 数据简报也就是所有部分组的报告的汇总
            report_all_df = self.get_all_report_df()
            report_all_psi = self.get_month_psi()
            plot_all_proba_all_data_fig = self.plot_all_proba_all_data(n_cols=phs_n_cols_simmple, if_score=if_score)   
            month_report_df = self.group_metrics(groupby_col=self.month_col)
            # print(month_report_df)
            # self.temp_df = month_report_df

            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            with pd.ExcelWriter(save_path + 'simple_report.xlsx') as writer:
                report_all_df.to_excel(writer, sheet_name='report_all_df', index=False)
                month_report_df.to_excel(writer, sheet_name='month_report_df')
                report_all_psi.to_excel(writer, sheet_name='report_all_psi')
                pd.DataFrame().to_excel(writer, sheet_name='distribution')
                ws = writer.sheets['report_all_df']
                # 找report_all_df的ks 与 auc在第几列，还得考虑没有ks 和 auc的情况
                if 'ks' in report_all_df.columns:
                    ks_col = report_all_df.columns.get_loc('ks')
                else:
                    ks_col = None
                if 'auc' in report_all_df.columns:
                    auc_col = report_all_df.columns.get_loc('auc')
                else:
                    auc_col = None
                for col_num in [ks_col,auc_col]:
                    if col_num is not None:
                        start_cell,end_cell = excel_utils.excel_padding_format(col_num,1,report_all_df.shape[1]-col_num-1,0,report_all_df)
                        excel_utils.apply_percentage_format(ws, start_cell, end_cell) # 修改单元格格式
                        excel_utils.apply_color_scale(ws, start_cell, end_cell) # 修改单元格色阶
                ws = writer.sheets['distribution']
                excel_utils.insert_matplotlib_fig(ws, plot_all_proba_all_data_fig, 'A1',scale=0.3) # 插入图片
                nrows,ncols = excel_utils.estimate_cell_dimensions(plot_all_proba_all_data_fig,scale=0.3) # 估算图片占用的行列数
                excel_utils.merge_cells_from_start(ws, 'A1', nrows, ncols) # 合并单元格
                

        if if_detail:
            # part2 详细分组报告 - 非psi
            print('######################### Start create detail report #########################')
            report_metrics_many_dict = self.group_metrics_many()
            
            with pd.ExcelWriter(save_path + 'detail_report.xlsx') as writer:
                for key,value in report_metrics_many_dict.items():
                    sheet_name = f'groupby_{key}'
                    value.to_excel(writer, sheet_name=sheet_name)
                    ws = writer.sheets[sheet_name]
                    # 对内存要求太高....在rz的分群上暂且搁置下述所有功能...使用source_plt_combined函数绘制分布图...
                    if if_detail_plot:
                        # if len(self.proba_list)*len(self.label_df[key].unique()) >=500:
                        #     print(f'The number of {key} group plots is too large, will not plot!!!\n You can use the function plot_all_proba_all_data or source_plt_combined (this function is recommended)!!!')
                        #     continue
                        fig = self.plot_all_proba_all_data(source_col=key,n_cols=phs_n_cols_detail, if_score=if_score)

                        start_cell = excel_utils.get_cell(value.shape[1]+3,1)
                        excel_utils.insert_matplotlib_fig(ws, fig, start_cell,scale=0.1) # 插入图片
                        nrows,ncols = excel_utils.estimate_cell_dimensions(fig,scale=0.1) # 估算图片占用的行列数
                        excel_utils.merge_cells_from_start(ws, start_cell, nrows, ncols) # 合并单元格

        if if_psi:
            # part3 详细分组报告 - psi
            print('######################### Start create detail report psi #########################')
            report_all_group_psi = self.get_month_psi_many()
            
            with pd.ExcelWriter(save_path + 'detail_report_psi.xlsx') as writer:
                for key,value in report_all_group_psi.items():
                    sheet_name = f'groupby_{key}'
                    value.to_excel(writer, sheet_name=sheet_name)
                    ws = writer.sheets[sheet_name]
        print('######################### End create report #########################')
        self.reporting_flag = 0


    @staticmethod
    def create_example_data() -> pd.DataFrame:
        """
        创建示例数据集
        
        Returns
        -------
        pd.DataFrame
            包含mobile, backPointTime, target, qudao3_act, ronghe_proba等字段的示例数据
        """
        np.random.seed(42)
        n_samples = 10000
        target = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
        qudao3_act = np.where(target == 1,
                            np.random.choice(['qudao1', 'qudao2', 'qudao3'], size=n_samples, p=[0.6, 0.3, 0.1]),
                            np.random.choice(['qudao1', 'qudao2', 'qudao3'], size=n_samples, p=[0.1, 0.3, 0.6]))

        qudao2_act = np.where(target == 1,
                            np.random.choice(['qudao21', 'qudao22', 'qudao23'], size=n_samples, p=[0.6, 0.3, 0.1]),
                            np.random.choice(['qudao21', 'qudao22', 'qudao23'], size=n_samples, p=[0.1, 0.3, 0.6]))


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

            'qudao2_act': qudao2_act,
            
            'ronghe_proba': np.where(target == 1,
                                    np.random.uniform(0.7, 1.0, n_samples),
                                    np.random.uniform(0.0, 0.4, n_samples)),

            'ronghe_proba2': np.where(target == 1,
                                    np.random.uniform(0.7, 1.0, n_samples),
                                    np.random.uniform(0.0, 0.4, n_samples)),
            
            'ronghe_proba3': np.where(target == 1,
                                    np.random.uniform(0.7, 1.0, n_samples),
                                    np.random.uniform(0.0, 0.4, n_samples)),
            'ronghe_proba4': np.where(target == 1,
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


# 使用示例
if __name__ == '__main__':
    # 1. 创建示例数据
    data = TzScoreGroupTools.create_example_data()
    data['month'] = pd.to_datetime(data['backPointTime']).dt.strftime('%Y-%m')
    data['weight'] = np.where(data['target'] == 1, 2.0, 0.5)  # 添加权重列
    
    # 2. 初始化工具类
    tool = TzScoreGroupTools(
        label_df=data,
        target_col='target',
        groupby_list=['qudao3_act','qudao2_act', 'month'],
        month_col='month',
        proba_list=['ronghe_proba', 'ronghe_proba2', 'ronghe_proba3', 'ronghe_proba4'],
        if_weight=False,
        if_score=True,
        metrics_lst=['KS','SUB_5_LIFT', 'TOP_5_LIFT'],
        multi_index=True
    )

    # # 3. 计算单个分组指标
    # result1 = tool.group_metrics(
    #     # groupby_col = None
    #     # groupby_col='qudao3_act',
    #     # metrics=['KS', 'AUC']
    # )
    # print("单个分组指标:\n", result1.head())
    
    # # 4. 计算多个分组指标
    # result2 = tool.group_metrics_many()
    # for key, df in result2.items():
    #     print(f"\n多个分组指标 - {key}:\n", df.head())
    
    # # 5. 交叉分析
    # result3 = tool.get_cross(['qudao3_act', 'month'], proba_col='ronghe_proba')
    # print("\n交叉分析:\n", result3.head())
    
    # # 6. PSI计算
    # result4 = tool.get_month_psi(month_col='month', groupby_col='qudao3_act')
    # print("\n月度PSI:\n", result4.head())
    
    # result5 = tool.get_month_psi_many(month_col='month', groupby_col_list=['qudao3_act', 'qudao2_act'])
    # for key, df in result5.items():
    #     print(f"\n多个分组psi - {key}:\n", df.head())

    # # 7. 绘制分布图
    # fig = tool.source_plt_combined(
    #     score_col='ronghe_proba',
    #     source_col='qudao3_act',
    #     save_path='qudao_distribution.png'
    # )
    # print("\n分布图已保存")

    # 8. 创建报告
    tool.create_report(save_path=r'D:\work\06.std_product\xyf/report_test1230_second/')