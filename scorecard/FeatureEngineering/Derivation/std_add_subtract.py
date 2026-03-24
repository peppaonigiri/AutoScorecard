import copy

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def run(data, ft_lst, ex_lst):
    stand_scaler = StandardScaler()
    res = copy.deepcopy(data[ex_lst])

    data[ft_lst] = stand_scaler.fit_transform(data[ft_lst])  # 标准化

    length = len(ft_lst)
    for i in range(length):
        for j in range(i, length):
            res[ft_lst[i] + '_add_' + ft_lst[j]] = data[ft_lst[i]] + data[ft_lst[j]]
            res[ft_lst[i] + '_subtract_' + ft_lst[j]] = data[ft_lst[i]] - data[ft_lst[j]]
            res[ft_lst[i] + '_multiply_' + ft_lst[j]] = data[ft_lst[i]] * data[ft_lst[j]]
            res[ft_lst[i] + '_divide_' + ft_lst[j]] = data[ft_lst[i]] * data[ft_lst[j]]
    return res
