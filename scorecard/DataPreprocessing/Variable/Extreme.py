#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  :
    @Time    : 2019/7/24 23:23
    @Use     : 极值缺失值处理
"""

import pandas as pd


# 1、用1%和99%的值替代异常值
def replace_by_percentage(df, var_names, flag_target):
    dev = df[df[flag_target] == 'dev']
    val = df[df[flag_target] == 'val']
    oot = df[df[flag_target] == 'oot']
    df_rpl = pd.DataFrame(columns=('var_names', 'p01', 'p99'))

    for (idx, var_name) in enumerate(var_names):
        p01, p99 = dev[var_name].quantile([0.01, 0.99]).values
        p01, p99 = round(p01, 5), round(p99, 5)
        print(idx, var_name, p01, p99)
        df_rpl = df_rpl.append([{'var_names': var_name, 'p01': p01, 'p99': p99}], ignore_index=True)

        dev.loc[dev[var_name].isnull(), var_name] = -100
        dev.loc[dev[var_name] < p01, var_name] = p01
        dev.loc[dev[var_name] > p99, var_name] = p99

        if not isinstance(val, str):
            val.loc[val[var_name].isnull(), var_name] = -100
            val.loc[val[var_name] < p01, var_name] = p01
            val.loc[val[var_name] > p99, var_name] = p99

        if not isinstance(oot, str):
            oot.loc[oot[var_name].isnull(), var_name] = -100
            oot.loc[oot[var_name] < p01, var_name] = p01
            oot.loc[oot[var_name] > p99, var_name] = p99
    new_df = pd.concat([dev, val, oot])
    new_df.to_csv(new_df.replace(".csv", "_trt.csv"), index=False)


