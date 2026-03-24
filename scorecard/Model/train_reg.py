#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  :
    @Time    :
    @Use     : 训练模型
"""

import xgboost as xgb
import lightgbm as lgb
import copy
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error


def show(model, ft_lst, dep, datasets):
    res = pd.DataFrame()
    for key, value in datasets.items():
        y_pred = model.predict(value[ft_lst])

        res = res.append({'datasets': key,
                          'r2_score': r2_score(value[dep], y_pred),
                          'mean_squared_error': mean_squared_error(value[dep], y_pred),
                          'explained_variance_score': explained_variance_score(value[dep], y_pred),
                          'mean_absolute_error': mean_absolute_error(value[dep], y_pred)},
                         ignore_index=True)
    print(res[['datasets', 'mean_squared_error', 'mean_absolute_error', 'r2_score', 'explained_variance_score']])
    return res[['datasets', 'mean_squared_error', 'mean_absolute_error', 'r2_score', 'explained_variance_score']]


def plot(data, dep, target):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    t = data.shape[0]
    plt.figure(facecolor='w')
    plt.plot(t, data[dep], 'r-', linewidth=2, label='真实值')
    plt.plot(t, data[target], 'g-', linewidth=1, label='预测值')
    plt.legend(loc='upper left')
    plt.title("线性回归预测真实值之间的关系", fontsize=20)
    plt.grid(b=True)
    plt.show()


def get_against_weight(train, oot, ft_lst):
    train_c = train.copy()
    oot_c = oot.copy()
    train_c['oot_label'] = 0
    oot_c['oot_label'] = 1
    data = pd.concat([train_c, oot_c])
    # print(data['oot_label'].value_counts())
    # print(data.shape[0])

    clf = xgb.XGBClassifier(learning_rate=0.1,
                            n_estimators=80,
                            max_depth=3,
                            min_child_weight=10,
                            subsample=0.7,
                            nthread=-1,
                            scale_pos_weight=1,
                            random_state=1,
                            n_jobs=-1,
                            reg_lambda=300,
                            predictor='cpu_predictor')

    clf.fit(data[ft_lst], data['oot_label'])
    train_c['weight'] = clf.predict_proba(train_c[ft_lst])[:, 1]

    # print('auc: ', roc_auc_score(train['oot_label'], train['against_weight']))
    return train_c


def lr_model(ft_lst, dep, fit_set='train', datasets=None):

    if datasets is None:
        datasets = {}
    model = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    model.fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], sample_weight=datasets[fit_set]['weight'])
    show(model, ft_lst, dep, datasets)
    return model


def xgb_model(ft_lst, dep, fit_set='train', datasets=None):
    if datasets is None:
        datasets = {}
    model = xgb.XGBRegressor(max_depth=3,
                             learning_rate=0.1,
                             n_estimators=100,
                             objective='reg:squarederror',
                             booster='gbtree',
                             gamma=0,
                             min_child_weight=1,
                             subsample=1,
                             colsample_bytree=1,
                             reg_alpha=0,
                             reg_lambda=1,
                             random_state=0,
                             verbose_eval=1)

    model.fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], sample_weight=datasets[fit_set]['weight'],
              eval_metric="rmse", eval_set=[(datasets['valid'][ft_lst], datasets['valid'][dep])], verbose=True)
    show(model, ft_lst, dep, datasets)
    return model


def lgb_model(ft_lst, dep, fit_set='train', datasets=None):
    if datasets is None:
        datasets = {}
    model = lgb.LGBMRegressor(objective='regression',
                              max_depth=3,
                              learning_rate=0.1,
                              n_estimators=100,
                              metric='rmse',
                              bagging_fraction=0.8,
                              feature_fraction=0.8,
                              boosting_type="gbdt",
                              num_leaves=15,
                              min_child_weight=0.01,
                              min_child_samples=10,
                              reg_lambda=300,
                              bagging_freq=10,
                              random_state=2020
                              )
    model.fit(datasets[fit_set][ft_lst], datasets[fit_set][dep],
              eval_metric='rmse', eval_set=(datasets['valid'][ft_lst], datasets['valid'][dep]), early_stopping_rounds=500,
              sample_weight=datasets[fit_set]['weight'], verbose=False)
    show(model, ft_lst, dep, datasets)
    return model


def rf_model(ft_lst, dep, fit_set='train', datasets=None):
    if datasets is None:
        datasets = {}
    model = RandomForestRegressor(n_estimators=100,
                                  criterion='mse',
                                  max_depth=3,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0,
                                  max_features='auto',
                                  max_leaf_nodes=None,
                                  min_impurity_decrease=0.0,
                                  min_impurity_split=None,
                                  bootstrap=True,
                                  oob_score=False,
                                  n_jobs=None,
                                  random_state=2021,
                                  verbose=0,
                                  warm_start=False)
    model.fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], sample_weight=datasets[fit_set]['weight'])
    show(model, ft_lst, dep, datasets)
    return model


def bi_train(df, dep='label', exclude=None, model_switch=None, if_against_weight=None):
    if model_switch is None:
        model_switch = [1, 1, 1, 1]
    data = copy.deepcopy(df)
    ft_lst = [i for i in data.columns if i not in exclude]
    print('num_features: ', len(ft_lst))

    datasets = {}
    set_lst = list(df['target'].unique())
    for i in set_lst:
        datasets[i] = data[data['target'] == i]
    if if_against_weight is not None:
        datasets['if_against_weight'] = get_against_weight(data[data['target'] == 'train'],
                                                           data[data['target'] == 'oot'], ft_lst)

    res = {}
    if model_switch[0]:
        print("线性回归：")
        model_lr = lr_model(ft_lst, dep, datasets=datasets)
        res['model_lr'] = model_lr
        # print("线性回归反向：")
        # model_lr_inv = lr_model(ft_lst, dep, fit_set='oot', datasets=datasets)
        # res['model_lr_inv'] = model_lr_inv

    if model_switch[1]:
        print("XGBoost：")
        model_xgb = xgb_model(ft_lst, dep, datasets=datasets)
        res['model_xgb'] = model_xgb
        # print("XGBoost反向：")
        # model_xgb_inv = xgb_model(ft_lst, dep, fit_set='oot', datasets=datasets)
        # res['model_xgb_inv'] = model_xgb_inv

    if model_switch[2]:
        print("LightGBM：")
        model_lgb = lgb_model(ft_lst, dep, datasets=datasets)
        res['model_lgb'] = model_lgb
        # print("LightGBM反向：")
        # model_lgb_inv = lgb_model(ft_lst, dep, fit_set='oot', datasets=datasets)
        # res['model_lgb_inv'] = model_lgb_inv

    if model_switch[3]:
        print("RF：")
        model_rf = rf_model(ft_lst, dep, datasets=datasets)
        res['model_lgb'] = model_rf
        # print("RF反向：")
        # model_rf_inv = rf_model(ft_lst, dep, fit_set='oot', datasets=datasets)
        # res['model_rf_inv'] = model_rf_inv

    return res
