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


# 计算变量CSI
def csi(actual, expect, val, q):
    var_cat = pd.qcut(actual[val].values, q=q, duplicates='drop')
    var_cat_cnt = var_cat.value_counts()+0.0001
    var_cnt_proportion = var_cat_cnt/sum(var_cat_cnt)
    age_pred_cnt = pd.cut(expect[val].values, bins=var_cat.categories).value_counts()
    age_pred_cnt_1 = age_pred_cnt + 0.0001
    age_pred_proportion = age_pred_cnt_1/sum(age_pred_cnt_1)
    return sum((var_cnt_proportion-age_pred_proportion) * np.log(var_cnt_proportion/age_pred_proportion))


def value(actual, expect, var_name, q=10):
    res = []
    for i in var_name:
        res.append(csi(actual[i], expect[i], i, q=q))
    return pd.DataFrame(res, columns=['var_names', 'csi'])

