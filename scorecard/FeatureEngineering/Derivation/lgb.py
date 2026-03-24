# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:08:42 2019

@author:
"""

import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def lgb_dev(data, label, num_boost_round, num_leaves, max_depth):
    params = {
        'num_boost_round': num_boost_round,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': num_leaves,
        'metric': 'auc',
        'max_depth': max_depth,
        'feature_fraction': 1,
        'bagging_fraction': 1,
    }
    lgb_train = lgb.Dataset(data, label, free_raw_data=False)
    model = lgb.train(params, lgb_train)
    leaf = model.predict(data, pred_leaf=True)
    lgb_enc = OneHotEncoder()
    lgb_enc.fit(leaf)
    gl = pd.DataFrame(lgb_enc.transform(leaf).toarray())
    return gl

