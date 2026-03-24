import pandas as pd


class Result(object):
    def __init__(self, threshold, drop_lst, keep_lst, df_res):
        self.threshold = threshold
        self.drop_lst = drop_lst
        self.keep_lst = keep_lst
        self.num_drop = len(drop_lst)
        self.num_keep = len(keep_lst)
        self.res = df_res


def row(df):
    row, col = df.shape
    row_miss = []
    row_total = []
    for i in range(row):
        w = df.iloc[i, :].isnull().sum()  # 第i行缺失的总数
        row_total.append(w)
        row_miss.append(w.sum() / col)
    row_miss = pd.Series(row_miss)
    row_total = pd.Series(row_total)
    row_miss.index = df.index  # 要保证row_miss和df的index相同
    return pd.DataFrame({'var_names': row_miss.index, 'count': row_total,
                         'rate': row_miss}).sort_values(by='rate', ascending=False)


def value(df):
    return pd.DataFrame({'var_names': df.columns,
                         'missing_count': df.isnull().sum(),
                         'missing_rate': df.isnull().sum() / df.isnull().count(),
                         'cover_rate': 1 - df.isnull().sum() / df.isnull().count()}).\
        sort_values(by='missing_rate',  ascending=False).reset_index(drop=True)


def filter(df, threshold=0.95):
    """
    :param df: 待分析的DataFrame数据
    :param threshold: 缺失率阈值
    :return: res
    """
    res = value(df)
    keep_lst = list(res[res['missing_rate'] < threshold]['var_names'])
    drop_lst = list(res[res['missing_rate'] >= threshold]['var_names'])

    return Result(threshold, drop_lst, keep_lst, res)


def missing_filter_with_target(data_all, y_list, null_rate=0.95):
    """
    :param data_all: 待分析的DataFrame数据
    :param y_list: 非变量列表，一般含target
    :param null_rate: 缺失率阈值
    :return: 返回缺失率筛选后剩下的变量列表
    """
    n = data_all.shape[0]
    a = pd.DataFrame(data_all.apply(lambda x: sum(x.isnull())), columns=['null_num'])
    b = pd.DataFrame(data_all.apply(lambda x: sum(x.isnull()) / n), columns=['null_rate'])
    k = pd.merge(a, b, left_index=True, right_index=True, how='left')
    k_sort = k.sort_values(by='null_rate', axis=0, ascending=False)
    k_filter = k_sort[k_sort['null_rate'] <= null_rate]
    var_null_filter = list(k_filter.index)
    var_null_delete_y = [i for i in var_null_filter if i not in y_list]
    print("缺失率筛选后剩下的变量个数：{}".format(len(var_null_delete_y)))

    return var_null_delete_y


