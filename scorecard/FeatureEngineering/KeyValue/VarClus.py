
from FeatureEngineering.KeyValue import IV
from sklearn.cluster import FeatureAgglomeration
from sklearn import preprocessing

class Result(object):
    def __init__(self, threshold, drop_lst, keep_lst, df_res):
        self.threshold = threshold
        self.drop_lst = drop_lst
        self.keep_lst = keep_lst
        self.num_drop = len(drop_lst)
        self.num_keep = len(keep_lst)
        self.res = df_res

"""
step 1.列聚类，将所有变量进行层次聚类。
step 2.根据聚类结果，剔除IV相对较低的变量。
"""


def value(data_all, ft_lst, dep):
    x_scale = preprocessing.StandardScaler().fit_transform(data_all[ft_lst])
    ward = FeatureAgglomeration(n_clusters=10, linkage='ward')
    ward.fit(x_scale)
    res = IV.value(data_all, ft_lst, dep)
    res['cluster'] = list(ward.labels_)
    return res


def filter(df, ex_lst, dep, rule, threshold=5):
    """
    :param df: 待分析数据
    :param threshold: 阈值
    :return: res
    """
    ft_lst = [i for i in list(df.columns) if i not in ex_lst]
    res = value(df, ft_lst, dep).sort_values(by=['cluster', rule], ascending=False)

    f = [lambda x: [i for i in range(x.shape[0])]]
    drop_lst = list(res[res[rule] > threshold]['var_names'])
    keep_lst = [i for i in list(df.columns) if i not in drop_lst]

    return Result(threshold, drop_lst, keep_lst, res)
