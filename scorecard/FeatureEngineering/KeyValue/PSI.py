#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : F. Li
    @Time    : 2019/7/22 14:31
    @Use     : 计算变量的PSI
"""

import numpy as np
import math
import toad
import pandas as pd


class Result(object):
    def __init__(self, threshold, drop_lst, keep_lst, df_res):
        self.threshold = threshold
        self.drop_lst = drop_lst
        self.keep_lst = keep_lst
        self.num_drop = len(drop_lst)
        self.num_keep = len(keep_lst)
        self.res = df_res


def psi_var(dev, val):
    """
    计算变量psi

    :param dev: 训练集
    :param val: 对照集
    :return:
    """
    dev_cnt, val_cnt = sum(dev), sum(val)
    if dev_cnt * val_cnt == 0:
        return None
    PSI = 0
    for i in range(len(dev)):
        dev_ratio = dev[i] / dev_cnt
        val_ratio = val[i] / val_cnt + 1e-10
        psi = (dev_ratio - val_ratio) * math.log(dev_ratio / val_ratio)
        PSI += psi
    return PSI


def psi_proba(model, dev_x, val_x, model_type):
    """
    计算输出概率psi
    :param model: 模型
    :param dev_x: 训练集
    :param val_x: 对照集
    :return:
    """
    if model_type == 'lgb':
        dev_predict_y = model.predict(dev_x)
        val_predict_y = model.predict(val_x)
    else:
        dev_predict_y = [s[1] for s in model.predict_proba(dev_x)]
        val_predict_y = [s[1] for s in list(model.predict_proba(val_x))]

    dev_nrows = dev_x.shape[0]
    dev_predict_y.sort()
    # 等频分箱成10份
    cutpoint = [-100] + [dev_predict_y[int(dev_nrows / 10 * i)] for i in range(1, 10)] + [100]
    cutpoint = list(set(cutpoint))
    cutpoint.sort()
    val_nrows = val_x.shape[0]
    PSI = 0
    # 每一箱之间分别计算PSI
    for i in range(len(cutpoint) - 1):
        start_point, end_point = cutpoint[i], cutpoint[i + 1]
        dev_cnt = [p for p in dev_predict_y if start_point <= p < end_point]
        dev_ratio = len(dev_cnt) / dev_nrows + 1e-10
        val_cnt = [p for p in val_predict_y if start_point <= p < end_point]
        val_ratio = len(val_cnt) / val_nrows + 1e-10
        psi = (dev_ratio - val_ratio) * math.log(dev_ratio / val_ratio)
        PSI += psi
    return PSI


def psi_ft(actual, predict, bins=10):
    """
    计算连续变量和离散变量的PSI值
    :param actual: 一维数组或series，代表训练集中的变量
    :param predict: 一维数组或series，代表测试集中的变量
    :param bins: 违约率段划分个数
    :return: 字典，键值关系为{'psi': PSI值，'psi_fig': 实际和预期占比分布曲线}
    """

    psi_dict = {}
    actual = np.sort(actual)
    actual_distinct = np.sort(list(set(actual)))
    predict = np.sort(predict)
    predict_distinct = np.sort(list(set(predict)))
    actual_len = len(actual)
    actual_distinct_len = len(actual_distinct)
    predict_len = len(predict)
    predict_distinct_len = len(predict_distinct)
    psi_cut = []
    actual_bins = []
    predict_bins = []
    actual_min = actual.min()
    actual_max = actual.max()
    cuts = []
    binlen = (actual_max - actual_min) / bins
    if (actual_distinct_len < bins):
        for i in actual_distinct:
            cuts.append(i)
        for i in range(2, (actual_distinct_len + 1)):
            if i == bins:
                lowercut = cuts[i - 2]
                uppercut = float('Inf')
            else:
                lowercut = cuts[i - 2]
                uppercut = cuts[i - 1]
            actual_cnt = ((actual >= lowercut) & (actual < uppercut)).sum() + 1
            predict_cnt = ((predict >= lowercut) & (predict < uppercut)).sum() + 1
            actual_pct = (actual_cnt + 0.0) / actual_len
            predict_pct = (predict_cnt + 0.0) / predict_len
            psi_cut.append((actual_pct - predict_pct) * math.log(actual_pct / predict_pct))
            actual_bins.append(actual_pct)
            predict_bins.append(predict_pct)
    else:
        for i in range(1, bins):
            cuts.append(actual_min + i * binlen)
        for i in range(1, (bins + 1)):
            if i == 1:
                lowercut = float('-Inf')
                uppercut = cuts[i - 1]
            elif i == bins:
                lowercut = cuts[i - 2]
                uppercut = float('Inf')
            else:
                lowercut = cuts[i - 2]
                uppercut = cuts[i - 1]
            actual_cnt = ((actual >= lowercut) & (actual < uppercut)).sum() + 1
            predict_cnt = ((predict >= lowercut) & (predict < uppercut)).sum() + 1
            actual_pct = (actual_cnt + 0.0) / actual_len
            predict_pct = (predict_cnt + 0.0) / predict_len
            psi_cut.append((actual_pct - predict_pct) * math.log(actual_pct / predict_pct))
            actual_bins.append(actual_pct)
            predict_bins.append(predict_pct)
    psi = sum(psi_cut)
    nbins = len(actual_bins)
    xlab = np.arange(1, nbins + 1)
    psi_dict['psi'] = psi
    return psi_dict


'''
使用方法参考：
columns_select = var_missing_filter
fea_psi_compare = pd.DataFrame(index=['psi_value'],columns=columns_select)
# flag = 0
for column in columns_select:
#     print(flag, column)
    fea_psi_compare.ix['psi_value',column] = fea_psi_calc(train_data[column].dropna(),test_data[column].dropna())['psi']
#     flag += 1

fea_psi = fea_psi_compare.T
print('总体标签稳定性：')
fea_psi.head()

fea_psi.to_excel('变量psi统计结果.xlsx')

fea_psi_select = fea_psi[fea_psi['psi_value']<=0.1].sort_values(by='psi_value',ascending=False)
psi_stable_list = list(fea_psi_select.index)
print(len(psi_stable_list))
'''


def psi_by_month(df, train, ft_lst, var_name):
    df['month_time'] = df[var_name].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m'))
    result = list()
    for ft in ft_lst:
        f = [lambda x: toad.metrics.PSI(train[ft], x)]
        res = df.groupby(['month_time'])[ft].agg(f)
        print(res)
        res_ = pd.DataFrame([list(res['<lambda>'])], columns=list(res.index))
        res_['var_names'] = ft

        res_['max'] = res['<lambda>'].max()
        res_['mean'] = res['<lambda>'].mean()
        result.append(res_)

    result = pd.concat(result)
    result['benchmark'] = 'train'
    return result


def value(df, ft_lst):
    train = df[df['target'] == 'train'][ft_lst]
    valid = df[df['target'] == 'valid'][ft_lst]
    oot = df[df['target'] == 'oot'][ft_lst]
    tv = df[df['target'].isin(['train', 'valid'])][ft_lst]

    psi_df_tv = toad.metrics.PSI(train, valid)
    psi_df_to = toad.metrics.PSI(train, oot)
    psi_df_tvo = toad.metrics.PSI(tv, oot)

    psi_df_tv = psi_df_tv.reset_index().rename(columns={'index': 'var_names', 0: 'psi_tv'})
    psi_df_to = psi_df_to.reset_index().rename(columns={'index': 'var_names', 0: 'psi_to'})
    psi_df_tvo = psi_df_tvo.reset_index().rename(columns={'index': 'var_names', 0: 'psi_tvo'})
    res = pd.merge(psi_df_tv, psi_df_to, on='var_names').sort_values(by='psi_to', ascending=False)
    res = pd.merge(res, psi_df_tvo, on='var_names').sort_values(by='psi_tvo', ascending=False)

    return res


def value1(df, ft_lst):
    """
    如果存在oot1
    :param df:
    :param ft_lst:
    :return:
    """
    train = df[df['target'] == 'train'][ft_lst]
    valid = df[df['target'] == 'valid'][ft_lst]
    oot = df[df['target'] == 'oot'][ft_lst]
    oot1 = df[df['target'] == 'oot1'][ft_lst]
    tv = df[df['target'].isin(['train', 'valid'])][ft_lst]

    psi_df_tv = toad.metrics.PSI(train, valid)
    psi_df_to = toad.metrics.PSI(train, oot)
    psi_df_tvo = toad.metrics.PSI(tv, oot)
    psi_df_tvo1 = toad.metrics.PSI(tv, oot1)
    psi_df_oo = toad.metrics.PSI(oot, oot1)
    psi_df_tv = psi_df_tv.reset_index().rename(columns={'index': 'var_names', 0: 'psi_tv'})
    psi_df_to = psi_df_to.reset_index().rename(columns={'index': 'var_names', 0: 'psi_to'})
    psi_df_tvo = psi_df_tvo.reset_index().rename(columns={'index': 'var_names', 0: 'psi_tvo'})
    psi_df_tvo1 = psi_df_tvo1.reset_index().rename(columns={'index': 'var_names', 0: 'psi_tvo1'})
    psi_df_oo = psi_df_oo.reset_index().rename(columns={'index': 'var_names', 0: 'psi_oo'})
    res = pd.merge(psi_df_tv, psi_df_to, on='var_names', ).sort_values(by='psi_to', ascending=False)
    res = pd.merge(res, psi_df_tvo, on='var_names', ).sort_values(by='psi_tvo', ascending=False)
    res = pd.merge(res, psi_df_tvo1, on='var_names', ).sort_values(by='psi_tvo', ascending=False)
    res = pd.merge(res, psi_df_oo, on='var_names', ).sort_values(by='psi_tvo', ascending=False)
    return res


def filter(df, ex_lst, threshold=0.1):
    """
    :param df: 待分析数据
    :param threshold: 阈值
    :return: res
    """
    ft_lst = [i for i in list(df.columns) if i not in ex_lst]
    res = value(df, ft_lst)

    drop_lst = list(res[res['psi_tvo'] > threshold]['var_names'])
    keep_lst = [i for i in list(df.columns) if i not in drop_lst]

    return Result(threshold, drop_lst, keep_lst, res)


def calculate_psi(base_list, test_list, bins=20, min_sample=10):
    try:
        base_df = pd.DataFrame(base_list, columns=['score'])
        test_df = pd.DataFrame(test_list, columns=['score'])

        # 1.去除缺失值后，统计两个分布的样本量
        base_notnull_cnt = len(list(base_df['score'].dropna()))
        test_notnull_cnt = len(list(test_df['score'].dropna()))

        # 空分箱
        base_null_cnt = len(base_df) - base_notnull_cnt
        test_null_cnt = len(test_df) - test_notnull_cnt

        # 2.最小分箱数
        q_list = []
        if type(bins) == int:
            bin_num = min(bins, int(base_notnull_cnt / min_sample))
            q_list = [x / bin_num for x in range(1, bin_num)]
            break_list = []
            for q in q_list:
                bk = base_df['score'].quantile(q)
                break_list.append(bk)
            break_list = sorted(list(set(break_list)))  # 去重复后排序
            score_bin_list = [-np.inf] + break_list + [np.inf]
        else:
            score_bin_list = bins

        # 4.统计各分箱内的样本量
        base_cnt_list = [base_null_cnt]
        test_cnt_list = [test_null_cnt]
        bucket_list = ["MISSING"]
        for i in range(len(score_bin_list) - 1):
            left = round(score_bin_list[i + 0], 4)
            right = round(score_bin_list[i + 1], 4)
            bucket_list.append("(" + str(left) + ',' + str(right) + ']')

            base_cnt = base_df[(base_df.score > left) & (base_df.score <= right)].shape[0]
            base_cnt_list.append(base_cnt)

            test_cnt = test_df[(test_df.score > left) & (test_df.score <= right)].shape[0]
            test_cnt_list.append(test_cnt)

        # 5.汇总统计结果
        stat_df = pd.DataFrame({"bucket": bucket_list, "base_cnt": base_cnt_list, "test_cnt": test_cnt_list})
        stat_df['base_dist'] = stat_df['base_cnt'] / len(base_df)
        stat_df['test_dist'] = stat_df['test_cnt'] / len(test_df)

        def sub_psi(row):
            # 6.计算PSI
            base_list = row['base_dist']
            test_dist = row['test_dist']
            # 处理某分箱内样本量为0的情况
            if base_list == 0 and test_dist == 0:
                return 0
            elif base_list == 0 and test_dist > 0:
                base_list = 1 / base_notnull_cnt
            elif base_list > 0 and test_dist == 0:
                test_dist = 1 / test_notnull_cnt

            return (test_dist - base_list) * np.log(test_dist / base_list)

        stat_df['psi'] = stat_df.apply(lambda row: sub_psi(row), axis=1)
        stat_df = stat_df[['bucket', 'base_cnt', 'base_dist', 'test_cnt', 'test_dist', 'psi']]
        psi = stat_df['psi'].sum()

    except:
        print('error!!!')
        psi = np.nan
        stat_df = None
    return psi, stat_df