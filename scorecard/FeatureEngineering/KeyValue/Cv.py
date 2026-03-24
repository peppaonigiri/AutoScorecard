import pandas as pd
# 基于变异系数（Coefficient of Variation，CV）
#
# 1）功能：
#
# step 1. 基于数据分布EDD，选择某个指标（如均值mean）计算变异系数CV，用来衡量变量分布的稳定性。
# step 2. 设置阈值进行筛选。
# 2）指标：变异系数 C·V =（ 标准偏差 SD / 平均值Mean ）× 100%
#
# 在进行数据统计分析时，如果变异系数大于15%，则要考虑该数据可能不正常，应该剔除。 ----百度百科
#


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


def value(data, ex_lst):
    ft_lst = [i for i in data.columns if i not in ex_lst]
    res = pd.DataFrame()
    for ft in ft_lst:
        res = res.append({'var_names': ft,
                          'cv': data[ft].std() / data[ft].mean()}, ignore_index=True)
    return res.sort_values(by='cv', ascending=False)


def filter(df, ex_lst, threshold=3):
    """
    :param df: 待分析数据
    :param threshold: 阈值
    :return: res
    """
    res = value(df, ex_lst)

    drop_lst = list(res[res['cv'] > threshold]['var_names'])
    keep_lst = [i for i in df.columns if i not in drop_lst]

    return Result(threshold, drop_lst, keep_lst, res)

