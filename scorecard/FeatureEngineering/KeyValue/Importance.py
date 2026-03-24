#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : F. Li
    @Time    :
    @Use     : FeatureImportance
"""

import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import copy


class Result(object):
    def __init__(self, threshold, drop_lst, keep_lst, df_res):
        self.threshold = threshold
        self.drop_lst = drop_lst
        self.keep_lst = keep_lst
        self.num_drop = len(drop_lst)
        self.num_keep = len(keep_lst)
        self.res = df_res


def lr_model(x, y):
    model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, penalty='l2', solver='liblinear',
                               verbose=0, n_jobs=1)
    model.fit(x, y)
    return model


def xgb_model(x, y):
    model = xgb.XGBClassifier(learning_rate=0.1,
                              n_estimators=80,
                              max_depth=3,
                              class_weight='balanced',
                              min_child_weight=1,
                              subsample=0.7,
                              nthread=-1,
                              scale_pos_weight=1,
                              n_jobs=-1,
                              reg_lambda=300)
    model.fit(x, y)
    return model


def lgb_model(x, y, valx, valy):
    model = lgb.LGBMClassifier(num_leaves=30,
                               min_child_weight=0.05,
                               colsample_bytree=0.7,
                               subsample=0.7,
                               min_child_samples=100,
                               objective='binary',
                               max_depth=5,
                               learning_rate=0.005,
                               boosting_type="gbdt",
                               bagging_seed=2020,
                               verbosity=-1,
                               random_state=2020,
                               n_estimators=80)
    model.fit(x, y, eval_metric='auc', eval_set=([(x, y), (valx, valy)]), early_stopping_rounds=500, verbose=False)
    return model


def rf_model(x, y):
    model = RandomForestClassifier(n_estimators=80,
                                   random_state=0,
                                   n_jobs=-1)
    model.fit(x, y)

    return model


def gbdt_model(x, y):
    model = GradientBoostingClassifier(learning_rate=0.1,
                                       n_estimators=80,
                                       max_depth=3,
                                       max_features=None,
                                       min_samples_split=5,
                                       subsample=0.7,
                                       min_samples_leaf=5)
    model.fit(x, y)

    return model


def value(df, dev, val, dep='label', exclude=None, model_switch=[1, 1, 1, 1, 1]):
    data = copy.deepcopy(df)

    lis = list(data.columns)
    for i in exclude:
        if i in lis:
            lis.remove(i)

    devv = data[(data['target'] == dev)]
    vall = data[(data['target'] == val)]
    x, y = devv[lis], devv[dep]
    valx, valy = vall[lis], vall[dep]

    res = pd.DataFrame()
    res['var_names'] = lis

    if model_switch[0] == 1:
        # from sklearn.preprocessing import StandardScaler
        # std_scaler = StandardScaler()
        # data[lis] = std_scaler.fit_transform(data[lis])
        model_lr = lr_model(x, y)
        res['lr_coef'] = model_lr.coef_[0]

    if model_switch[1] == 1:
        model_xgb = xgb_model(x, y)
        res['xgb_gain'] = model_xgb.feature_importances_
        model_xgb.importance_type = 'total_gain'
        res['xgb_total_gain'] = model_xgb.feature_importances_
        model_xgb.importance_type = 'weight'
        res['xgb_weight'] = model_xgb.feature_importances_
        model_xgb.importance_type = 'total_cover'
        res['xgb_total_cover'] = model_xgb.feature_importances_
        model_xgb.importance_type = 'cover'
        res['xgb_cover'] = model_xgb.feature_importances_
        res['xgb_shap_value'] = pd.DataFrame(
            np.abs(model_xgb.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)[:, :-1])).mean()

    if model_switch[2] == 1:
        model_lgb = lgb_model(x, y, valx, valy)
        res['lgb_split'] = model_lgb.feature_importances_
        model_lgb.importance_type = 'gain'
        res['lgb_gain'] = model_lgb.feature_importances_
        res['lgb_shap_value'] = pd.DataFrame(np.abs(model_lgb.predict(x, pred_contrib=True)[:, :-1])).mean()

    if model_switch[3] == 1:
        model_rf = rf_model(x, y)
        res['rf'] = model_rf.feature_importances_

    if model_switch[4] == 1:
        model_gbdt = gbdt_model(x, y)
        res['gbdt'] = model_gbdt.feature_importances_

    return res


def filter(df, ex_lst, dep, type, threshold=0):
    """
    :param df: 待分析数据
    :param threshold: 阈值
    :return: res
    """
    res = value(df, 'train', 'valid', dep, ex_lst)

    drop_lst = list(res[res[type] <= threshold]['var_names'])
    keep_lst = [i for i in list(df.columns) if i not in drop_lst]

    return Result(threshold, drop_lst, keep_lst, res)


def plot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('You need to install matplotlib for plotting.')

    # 加载数据集
    print('加载数据...')
    df_train = pd.read_csv('data/regression.train.txt', header=None, sep='\t')
    df_test = pd.read_csv('data/regression.test.txt', header=None, sep='\t')

    # 取出特征和标签
    y_train = df_train[0].values
    y_test = df_test[0].values
    X_train = df_train.drop(0, axis=1).values
    X_test = df_test.drop(0, axis=1).values

    # 构建lgb中的Dataset数据格式
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # 设定参数
    params = {
        'num_leaves': 5,
        'metric': ('l1', 'l2'),
        'verbose': 0
    }

    evals_result = {}  # to record eval results for plotting

    print('开始训练...')
    # 训练
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=[lgb_train, lgb_test],
                    feature_name=['f' + str(i + 1) for i in range(28)],
                    categorical_feature=[21],
                    evals_result=evals_result,
                    verbose_eval=10)

    print('在训练过程中绘图...')
    ax = lgb.plot_metric(evals_result, metric='l1')
    plt.show()

    print('画出特征重要度...')
    ax = lgb.plot_importance(gbm, max_num_features=10)
    plt.show()

    import matplotlib.pyplot as plt

    print('画出第84颗树...')
    ax = lgb.plot_tree(gbm, tree_index=83, figsize=(20, 8), show_info=['split_gain'])
    plt.show()
