
from sklearn.feature_selection import VarianceThreshold
import pandas as pd


class Result(object):
    def __init__(self, threshold, drop_lst, keep_lst, df_res):
        self.threshold = threshold
        self.drop_lst = drop_lst
        self.keep_lst = keep_lst
        self.num_drop = len(drop_lst)
        self.num_keep = len(keep_lst)
        self.res = df_res


def value(data, threshold):
    var = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    var.fit_transform(data)
    res = pd.DataFrame(columns={'var_names', 'var'})
    return res