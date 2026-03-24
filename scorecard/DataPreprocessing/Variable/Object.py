#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : F. Li
    @Time    : 2019/7/24 23:23
    @Use     : 字符型变量处理
"""

import pandas as pd
import numpy as np
import toad


# 1. 去掉数据中的特殊符号并转化为其他类型
def special_symbol(data, feature, type, replaced_value, replace_value):
    data[feature] = pd.Series(data[feature]).str.replace(replaced_value, replace_value).astype(type)
    # data[feature] = data[feature].replace(["A", "B", "C", "D", "E", "F"], [1, 2, 3, 4, 5, 6])


# 2. one hot 编码
def encoder_onehot(self, data, feature):
    embarked_oht = pd.get_dummies(data[[feature]])
    data.loc[:, embarked_oht] = embarked_oht


def cat_encoder(df, col, target):
    """
    方法：字符分类变量按照坏样本率进行编码
    :param df: 待处理的数据
    :param col: 待处理的变量
    :param target: 目标值
    :return: 返回编码处理好的该变量对应的数据newCol, 该变量每个值对应的编码值
    """
    encoder = {}
    for v in set(df[col]):
        if v == v:
            subDf = df[df[col] == v]
        else:  # 空值的情况
            xList = list(df[col])
            nanInd = [i for i in range(len(xList)) if xList[i] != xList[i]]  # nan所在位置的索引
            subDf = df.loc[nanInd]
        encoder[v] = sum(subDf[target]) * 1.0 / subDf.shape[0]  # 坏样本率
    newCol = [encoder[i] for i in df[col]]
    return newCol, encoder


'''分类变量按照ln(odds) woe 进行编码'''


def ln_odds_encoder(df, col, target):
    """
    :param df: 待处理的数据
    :param col: 待处理的变量
    :param target: 目标值
    :return: 返回编码处理好的该变量对应的数据，和该变量每个值对应的编码值。空值仍保持空值np.nan
    """
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup[col + '_WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)

    encoder = regroup[[col, col + '_WOE']].set_index(col).to_dict(orient='index')
    for k, v in encoder.items():
        encoder[k] = v[col + '_WOE']
    for v in set(df[col]):
        if v != v:
            encoder[v] = v
    # encoder = {}
    # for v in set(df[col]):
    #     if v == v:
    #         encoder[v] = regroup[regroup[col] == v][col + '_WOE'].iloc[0]
    #     else:
    #         encoder[v] = v
    newCol = [encoder[i] for i in df[col]]
    return newCol, encoder


'''分类变量按照ln(odds)进行编码'''


def Ln_odds(df, col, target):
    """
    功能：
    :param df:
    :param col:
    :param target:
    :return: 返回一个拼接后的大表
    """
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup[col + '_WOE'] = regroup.apply(lambda x: np.log(x.bad_pcnt * 1.0 / x.good_pcnt), axis=1)
    df = pd.merge(df, regroup[[col, col + '_WOE']], on=col, how='left')
    return df


"""
画woe图配合使用
for i in categorical_var:
    data=Ln_odds(data, i, 'target')
    sns.pointplot(x=i, y=i+'_WOE', data=data)
    plt.xticks(rotation=90)
    plt.show()
"""


def obj2num(df, ft_lst, threshold_freq):
    for i in ft_lst:
        res = pd.DataFrame(df[i].value_counts())
        lst = list(res[res[i] < threshold_freq].index)
        index_ = list(df[df[i].isin(lst)].index)
        df.loc[index_, i] = u"其它"

    for i in ft_lst:
        dic = dict(df.groupby(i)['label'].mean())
        df[i] = df[i].replace(dic)

    combiner = toad.transform.Combiner()
    combiner.fit(df, df['label'], method='dt', min_samples=0.05, exclude=['label', 'mobile'], n_bins=6)
    data_obj_bin = combiner.transform(df)

    t = toad.transform.WOETransformer()
    df_woe = t.fit_transform(data_obj_bin, data_obj_bin['label'], exclude=['label', 'mobile'])

    return df_woe

