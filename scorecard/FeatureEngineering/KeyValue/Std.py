
import pandas as pd


class Result(object):
    def __init__(self, threshold, drop_lst, keep_lst, df_res):
        self.threshold = threshold
        self.drop_lst = drop_lst
        self.keep_lst = keep_lst
        self.num_drop = len(drop_lst)
        self.num_keep = len(keep_lst)
        self.res = df_res


def value(df):
    """
    :param df: 待分析的DataFrame数据
    :return: 标准差筛选后剩下的变量列表
    """

    numerical_var = list(df.select_dtypes(include=["number"]).columns)
    res = pd.DataFrame(columns=['var_names', 'std'])
    for i in numerical_var:
        res = res.append({'var_names': i, 'std': df[i].std()}, ignore_index=True)
    return res.sort_values(by='std', ascending=False)


def filter(df, threshold=0):
    """
    :param df: 待分析数据
    :param threshold: 阈值
    :return: res
    """
    res = value(df)

    drop_lst = list(res[res['std'] == threshold]['var_names'])
    keep_lst = [i for i in list(df.columns) if i not in drop_lst]

    return Result(threshold, drop_lst, keep_lst, res)

