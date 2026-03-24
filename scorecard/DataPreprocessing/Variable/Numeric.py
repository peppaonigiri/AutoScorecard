#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : F. Li
    @Time    : 2019/7/24 23:23
    @Use     : 数值型变量处理
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import Normalizer


""" 一、数值缩放 """
# 1. 对数变换
def map_log(data, feature_list):
    for feature in feature_list:
        log = data[feature].apply(lambda x: np.log(x))
        data.loc[:, feature + '_log'] = log


# 2. 幅度缩放，最大最小值缩放到[0,1]区间内
def minmax_scaler(data, feature_list):
    minmax_scaler = MinMaxScaler()
    for feature in feature_list:
        mm_scaler = minmax_scaler.fit_transform(data[[feature]])
        data.loc[:, feature + '_minmax_scaler'] = mm_scaler


# 3. 幅度缩放，将每一列的数据标准化为正态分布的
def standar_scaler(self, data, feature_list):
    standar_scaler = StandardScaler()
    for feature in feature_list:
        std_scaler = standar_scaler.fit_transform(data[[feature]])
        data.loc[:, feature + '_standar_scaler'] = std_scaler


# 4. 中位数或者四分位数去中心化数据
def preprocess_robust_scaler(data, feature_list):
    for feature in feature_list:
        data.loc[:, feature + '_standar_scaler'] = robust_scale(data[[feature]])


# 5. 将同一行数据规范化,前面的同一变为1以内也可以达到这样的效果
def preprocess_normalizer(data, feature_list):
    normalizer = Normalizer()
    for feature in feature_list:
        data.loc[:, feature + '_normalizer'] = normalizer.fit_transform(data[[feature]])


# 6. 离散化
# 等距切分
def preprocess_discretization_cut(data, feature_list, num):
    for feature in feature_list:
        data.loc[:, feature + '_cut'] = pd.cut(data[feature], num)


# 等频切分
def preprocess_discretization_qcut(data, feature_list, num):
    for feature in feature_list:
        data.loc[:, feature + '_cut'] = pd.qcut(data[feature], num)


# 7. 分箱
def preprocess_bin(data, feature, label):
    alist = list(set(data[feature]))
    badrate = {}
    for x in alist:
        subset = data[data[feature == x]]

        bad = subset[subset[label] == 1].count()
        good = subset[subset[label] == 0].count()

        badrate[x] = bad / (bad + good)

    f = zip(badrate.keys(), badrate.values())
    f = sorted(f, key=lambda x: x[1], reverse=True)
    badrate = pd.DataFrame(f)
    badrate.columns = pd.Series(['cut', 'badrate'])
    badrate = badrate.sort_values('cut')
    print(badrate)
    badrate.plot('cut', 'badrate')





