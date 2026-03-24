
from tqdm import tqdm
import pandas as pd
import numpy as np


class Result(object):
    def __init__(self, threshold, drop_lst, keep_lst, df_res):
        self.threshold = threshold
        self.drop_lst = drop_lst
        self.keep_lst = keep_lst
        self.num_drop = len(drop_lst)
        self.num_keep = len(keep_lst)
        self.res = df_res


# 单变量筛选：1.单变量某一取值占比筛选；2.单变量取值范围个数筛选
def value(df):
    """
    :param X: 待分析的DataFrame数据 lei
    :return: 单变量频率筛选后的变量列表
    """
    num_samples = df.shape[0]
    res = pd.DataFrame()

    for i in df.columns:
        res = res.append({'var_names': i, 'freq': pd.crosstab(index=df[i], columns=np.ones(num_samples))/num_samples},
                         ignore_index=True)
        # pd.crosstab 第一个参数是列, 第二个参数是行,统计交叉项的累计次数

    return res


def filter(df, threshold=0.95):
    """
    :param df: 待分析数据
    :param threshold: 阈值
    :return: res
    """
    numerical_var = list(df.select_dtypes(include=["number"]).columns)

    res = value(df[numerical_var])
    keep = []

    for index, row in res.iterrows():
        if (row['freq'] < threshold).values.all():
            # 所有取值percent都不超过阈值则为True
            keep.append(row['var_names'])

    drop_lst = [i for i in numerical_var if i not in keep]
    keep_lst = [i for i in list(df.columns) if i not in drop_lst]

    return Result(threshold, drop_lst, keep_lst, res)

