
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class Result(object):
    def __init__(self, threshold, drop_lst, keep_lst, df_res):
        self.threshold = threshold
        self.drop_lst = drop_lst
        self.keep_lst = keep_lst
        self.num_drop = len(drop_lst)
        self.num_keep = len(keep_lst)
        self.res = df_res

# def value(X, y, num_feats):
#     # 选择K个最好的特征，返回选择特征后的数据
#     X_norm = MinMaxScaler().fit_transform(X)
#     chi_selector = SelectKBest(chi2, k=num_feats)
#     new_data = chi_selector.fit(X_norm, y)
#     # chi_support = chi_selector.get_support()
#     # chi_feature = X.loc[:, chi_support].columns.tolist()
#     return new_data


def value(data, dep, ex_lst, k):
    ft = list(data.columns)
    for i in ex_lst:
        if i in ft:
            ft.remove(i)
    x = MinMaxScaler().fit_transform(data[ft])
    chi_selector = SelectKBest(chi2, k='all')
    new_data = chi_selector.fit(x, data[dep])
    return pd.DataFrame({'var_names': ft,
                        'chi2': new_data.scores_}).sort_values(by='chi2', ascending=False)


def filter(df, dep, ex_lst, threshold=3):
    """
    :param df: 待分析数据
    :param threshold: 阈值
    :return: res
    """
    res = value(df, dep, ex_lst, 10)

    drop_lst = list(res[res['chi2'] < threshold]['var_names'])
    keep_lst = [i for i in list(df.columns) if i not in drop_lst]

    return Result(threshold, drop_lst, keep_lst, res)

