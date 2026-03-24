#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : F. Li
    @Time    : 2019/7/24 23:23
    @Use     : 日期型变量处理
"""

import pandas as pd
from sklearn.impute import SimpleImputer


# 1. 转换日期格式
def to_datetime(data, feature_list):
    for feature in feature_list:
        data.loc[:, feature + '_t'] = pd.to_datetime(data[feature])


# 2. 取出关键时间信息
# 取出几月份
def get_month(data, feature_list):
    for feature in feature_list:
        data.loc[:, feature + '_month'] = data[feature].dt.month


# 取出来是几号
def get_date(data, feature_list):
    for feature in feature_list:
        data.loc[:, feature + '_day'] = data[feature].dt.day


# 取出一年当中的第几天
def get_dayofyear(data, feature_list):
    for feature in feature_list:
        data.loc[:, feature + '_doy'] = data[feature].dt.dayofyear


# 取出星期几
def get_dayofweek(data, feature_list):
    for feature in feature_list:
        data.loc[:, feature + '_dow'] = data[feature].dt.dayofweek


# 只显示年月并按照时间排序
def by_month(df, var_name):
    df['month_time'] = df[var_name].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m'))
    month_time = pd.DataFrame(df[df['target'] == 'train']['month_time'].value_counts()).sort_index()
    return month_time











