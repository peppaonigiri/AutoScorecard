import pandas as pd
import numpy as np
from tqdm import tqdm


def drop_std(df, std_limit, col_list=None):
    """
    去除标准差小于等于 std_limit 的列
    :param df: DataFrame
    :param std_limit: 标准差的阈值
    :param col_list: 列名列表，默认为 None，即 df.columns.tolist()
    :return: 保留的列名列表
    """
    if col_list is None:
        col_list = df.columns.tolist()

    drop_list = []
    keep_list = []

    for col in tqdm(col_list, desc="筛选标准差"):
        temp_std = df[col].std()
        if np.isnan(temp_std) or temp_std <= std_limit:
            drop_list.append(col)
        else:
            keep_list.append(col)

    print(f"舍弃了 {len(drop_list)} 个标准差小于等于 {std_limit} 的列")
    return keep_list


def drop_svr(df, svr_limit, col_list=None):
    """
    去除单一值占比超过 svr_limit 的列
    :param df: DataFrame
    :param svr_limit: 单一值占比的阈值
    :param col_list: 列名列表，默认为 None，即 df.columns.tolist()
    :return: 保留的列名列表
    """
    if col_list is None:
        col_list = df.columns.tolist()

    drop_list = []
    keep_list = []

    for col in tqdm(col_list, desc="筛选单一值率"):
        temp_series = df[col].value_counts(normalize=True)
        if len(temp_series) == 0 or temp_series.values[0] > svr_limit:
            drop_list.append(col)
        else:
            keep_list.append(col)

    print(f"舍弃了 {len(drop_list)} 个单一值占比超过 {svr_limit*100}% 的列")
    return keep_list
