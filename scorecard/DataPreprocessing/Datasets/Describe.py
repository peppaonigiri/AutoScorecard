import matplotlib
import pandas as pd
import numpy as np
import toad
# df = pd.DataFrame()
# var_name = []

# ======================================================================================================================
# df.describe()
# df.info()
# toad.detector.detect(df)
# ======================================================================================================================
"""
按月统计各类指标
"""

def badrate_by_month(df, var_name, target, dep):
    df['month_time'] = df[var_name].map(lambda x: pd.to_datetime(x).strftime('%Y-%m'))
    df = df[df['target'].isin(target)]
    res = pd.DataFrame(df['month_time'].value_counts().sort_index())
    df[dep] = pd.to_numeric(df[dep])
    res.rename(columns={'month_time': 'count'}, inplace=True)
    res['bad'] = df.groupby('month_time')[dep].agg([('bad', 'sum')])
    res['good'] = res['count'] - res['bad']
    res['bad_rate'] = df.groupby('month_time')[dep].agg([('bad_rate', 'mean')])
    return res


def badrate_by_day(df, var_name, target, dep):
    df['day_time'] = df[var_name].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
    df = df[df['target'].isin(target)]
    res = pd.DataFrame(df['day_time'].value_counts().sort_index())
    df[dep] = pd.to_numeric(df[dep])
    res.rename(columns={'day_time': 'count'}, inplace=True)
    res['bad'] = df.groupby('day_time')[dep].agg([('bad', 'sum')])
    res['good'] = res['count'] - res['bad']
    res['bad_rate'] = df.groupby('day_time')[dep].agg([('bad_rate', 'mean')])
    return res

def badrate_by_week(df, var_name, target, dep):
    df['week_time'] = df[var_name].apply(lambda x: pd.to_datetime(x).week)
    df = df[df['target'].isin(target)]
    res = pd.DataFrame(df['week_time'].value_counts().sort_index())
    df[dep] = pd.to_numeric(df[dep])
    res.rename(columns={'week_time': 'count'}, inplace=True)
    res['bad'] = df.groupby('week_time')[dep].agg([('bad', 'sum')])
    res['good'] = res['count'] - res['bad']
    res['bad_rate'] = df.groupby('week_time')[dep].agg([('bad_rate', 'mean')])
    res['date_max'] = df.groupby('week_time')[var_name].agg([('date_max', 'max')])
    res['date_min'] = df.groupby('week_time')[var_name].agg([('date_min', 'min')])
    return res


def feature_distribution_by_month_continue(df, var_name, ft_lst):
    df['month_time'] = df[var_name].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m'))
    result = list()
    for ft in ft_lst:
        f = ['count', 'mean', 'median', 'std', 'min', lambda x: x.quantile(0.01), lambda x: x.quantile(0.25),
             lambda x: x.quantile(0.5), lambda x: x.quantile(0.75), lambda x: x.quantile(0.99), 'max',
             lambda x: x.shape[0]]
        res = df.groupby('month_time')[ft].agg(f)
        res['var_names'] = ft
        res['month_time'] = res.index
        res.reset_index(drop=True, inplace=True)
        result.append(res)

    result = pd.concat(result)
    result.rename(columns={'<lambda_0>': '1%', '<lambda_1>': '25%', '<lambda_2>': '50%', '<lambda_3>': '75%',
                           '<lambda_4>': '99%', '<lambda_5>': 'total'}, inplace=True)
    result['count_NaN'] = result['total'] - result['count']
    result['cover_rate'] = result['count'] / result['total']
    return result[['var_names', 'month_time', 'total', 'count', 'cover_rate', 'count_NaN', 'mean', 'median', 'std',
                   'min', '1%', '25%', '50%', '75%', '99%', 'max']]


def feature_distribution_by_month_descret(df, var_name, ft_lst):
    df['month_time'] = df[var_name].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m'))
    result = list()
    for ft in ft_lst:
        f = ['count', 'mean', 'median', 'std', lambda x: x.shape[0]]
        res = df.groupby(['month_time', ft])[ft].agg(f)
        res['var_names'] = ft
        res.reset_index(inplace=True)
        res.rename(columns={ft: 'value'}, inplace=True)
        result.append(res)

    result = pd.concat(result)
    result.rename(columns={'<lambda_0>': 'total'}, inplace=True)
    result['count_NaN'] = result['total'] - result['count']
    result['cover_rate'] = result['count'] / result['total']
    return result[['var_names', 'month_time', 'value', 'total', 'count', 'cover_rate', 'count_NaN', 'median', 'mean',
                   'std']]


def cover_rate_by_month(df, var_name, ft_lst):
    df['month_time'] = df[var_name].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m'))
    result = list()
    for ft in ft_lst:
        f = ['count', lambda x: x.shape[0]]
        res = df.groupby(['month_time'])[ft].agg(f)

        res['cover_rate'] = res['count'] / res['<lambda_0>']

        res_ = pd.DataFrame([list(res['cover_rate'])], columns=list(res.index))
        res_['var_names'] = ft

        res_['std'] = res['cover_rate'].std()
        res_['mean'] = res['cover_rate'].mean()
        result.append(res_)

    result = pd.concat(result)
    result['cv'] = result['std'] / result['mean']

    return result


def duplicate_value_by_id(df, var_name):
    return pd.DataFrame(df[var_name].value_counts())



# ======================================================================================================================
# 单特征类别统计
# df.label.unique(return_index=True, return_inverse=True)
# df[var_name].value_counts(sort=True, ascending=False, bins=5, dropna=True, normalize=False)

# ======================================================================================================================
# 箱线图
# %matplotlib inline
# df.boxplot(column='Age')

# ======================================================================================================================
# 分布图
# %matplotlib inline
# import seaborn as sns
#
# sns.set(color_codes=True)
# np.random.seed(sum(map(ord, "distributions")))
# sns.distplot(df.Age, kde=True, bins=20, rug=True)

# ======================================================================================================================
# Bar3D
# from pyecharts.charts import Bar3D
# bar3d = Bar3D()
# x_axis = ["12a", "1a", "2a", "3a", "4a", "5a", "6a", "7a", "8a", "9a", "10a", "11a",
#           "12p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "10p", "11p"]
# y_axis = ["Saturday", "Friday", "Thursday", "Wednesday", "Tuesday", "Monday", "Sunday"]
# data = [[0, 0, 5], [0, 1, 1], [0, 2, 0], [0, 3, 0], [0, 4, 0], [0, 5, 0]]
# range_color = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf',
#                '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
# bar3d.add(
#     "",
#     x_axis,
#     y_axis,
#     [[d[1], d[0], d[2]] for d in data],
# #     is_visualmap=True,
# #     visual_range=[0, 20],
# #     visual_range_color=range_color,
#     grid3d_width=200,
#     grid3d_depth=80,
#     is_grid3d_rotate=True,  # 自动旋转
#     grid3d_rotate_speed=180,  # 旋转速度
# )
# bar3d