#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import json
import numpy as np
import pandas as pd
import toad


class Result(object):
    def __init__(self, threshold, drop_lst, keep_lst, df_res):
        self.threshold = threshold
        self.drop_lst = drop_lst
        self.keep_lst = keep_lst
        self.num_drop = len(drop_lst)
        self.num_keep = len(keep_lst)
        self.res = df_res


def iv_calc(count_1, count_0):
    """
    计算变量的iv值
    count_1: Series or numpy.array 每个bin中坏样本的数量
    count_0: Series or numpy.array 每个bin中好样本的数量
    """
    bad_dist = count_1 / float(count_1.sum())
    good_dist = count_0 / float(count_0.sum())
    bad_dist = bad_dist.apply(lambda x: 0.0001 if x == 0 else x)
    good_dist = good_dist.apply(lambda x: 0.0001 if x == 0 else x)
    return ((bad_dist - good_dist) * np.log(bad_dist / good_dist)).sum()


# def value(df, lst, dep, target=['train', 'valid', 'oot']):
#     res = pd.DataFrame({'var_names': lst})
#     for i in target:
#         temp = toad.quality(df[df['target'] == i][lst], target=df[df['target'] == 'train'][dep], iv_only=False).\
#             rename(columns={'iv': 'iv_' + i})
#         temp['var_names'] = temp.index
#         res = pd.merge(res, temp['iv_' + i], on='var_names', how='outer')
#
#     iv_total = toad.quality(df[lst], target=df[dep], iv_only=False)
#     iv_total['var_names'] = iv_total.index
#
#     res = res.reset_index().rename(columns={'index': 'var_names'})
#     res = pd.merge(res, iv_total, on='var_names', how='outer')
#
#     return res


def value(df, lst, dep):
    iv_train = toad.quality(df[df['target'] == 'train'][lst], target=df[df['target'] == 'train'][dep], iv_only=False)
    iv_valid = toad.quality(df[df['target'] == 'valid'][lst], target=df[df['target'] == 'valid'][dep], iv_only=False)
    iv_oot = toad.quality(df[df['target'] == 'oot'][lst], target=df[df['target'] == 'oot'][dep], iv_only=False)
    iv_total = toad.quality(df[lst], target=df[dep], iv_only=False)
    iv_total['var_names'] = iv_total.index

    res = pd.merge(iv_train['iv'], iv_valid['iv'], left_index=True, right_index=True, how='outer')
    res = pd.merge(res, iv_oot['iv'], left_index=True, right_index=True, how='outer')
    res = res.reset_index().rename(
        columns={'iv_x': 'iv_train', 'iv_y': 'iv_valid', 'iv': 'iv_oot', 'index': 'var_names'})
    res = pd.merge(res, iv_total, on='var_names', how='outer')

    return res


def filter(df, ex_lst, dep, threshold=0.009):
    """
    :param dep:
    :param ex_lst:
    :param df: 待分析数据
    :param threshold: 阈值
    :return: res
    """
    ft_lst = [i for i in list(df.columns) if i not in ex_lst]
    res = value(df, ft_lst, dep)

    drop_lst = list(res[res['iv_train'] < threshold]['var_names'])
    keep_lst = [i for i in list(df.columns) if i not in drop_lst]

    return Result(threshold, drop_lst, keep_lst, res)


def score_iv(dep, datasets=[]):
    res = pd.DataFrame(columns={'datasets', 'iv'})
    for i in datasets:
        label = i['target'][0]
        iv = toad.quality(i['score'], target=i[dep], iv_only=False)['iv']
        res = res.append({'datasets': label, 'iv': iv}, ignore_index=True)
    return res


def IV(df, score, target):
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total': total, 'bad': bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all.index = range(len(all))
    all = all.sort_values(by=score, ascending=False)

    all['badCumRate'] = all['bad'] / all['bad'].sum()
    all['goodCumRate'] = all['good'] / all['good'].sum()
    all['badCumRate'] = all['badCumRate'].apply(lambda x: 0.0001 if x == 0 else x)
    all['goodCumRate'] = all['goodCumRate'].apply(lambda x: 0.0001 if x == 0 else x)

    all['iv'] = (all['badCumRate'] - all['goodCumRate']) * np.log(all['badCumRate'] / all['goodCumRate'])
    all['totalPcnt'] = all['total'] / all['total'].sum()

    return all['iv'].sum()