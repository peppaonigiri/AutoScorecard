#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  :
    @Time    :
    @Use     : 训练模型
"""

from sklearn.metrics import roc_curve, auc, recall_score, precision_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
import copy
import pandas as pd
from Model.Evaluation.f2_score import f2_score


def show(model, ft_lst, dep, datasets):
    from matplotlib import pyplot as plt
    res = pd.DataFrame()

    for k, v in datasets.items():
        if k != 'test':
            if str(type(model)).find('Regressor') != -1 or str(type(model)).find('OneClassSVM') != -1:
                y_pred = model.predict(v[ft_lst])
                fpr_off, tpr_off, _ = roc_curve(v[dep], y_pred,sample_weight=v['weight'])
                off_ks = abs(fpr_off - tpr_off).max()
                plt.plot(fpr_off, tpr_off, label=k)
                # res = res.append({'datasets': k,
                #                   'auc': auc(fpr_off, tpr_off),
                #                   # 'precision': precision_score(v[dep], y_pred, average='micro'),
                #                   # 'recall': recall_score(v[dep], y_pred, average='micro'),
                #                   'ks': off_ks},
                #                  ignore_index=True)
                add_df = pd.DataFrame({'datasets': k,
                                        'auc': auc(fpr_off, tpr_off),
                                        'ks': off_ks}, index=[0])
                res = pd.concat([res, add_df], axis=0)
            else:
                if str(type(model)).find('DecisionTreeRegressor') != -1:
                    y_pred = model.predict(v[ft_lst])
                else:
                    y_pred = model.predict_proba(v[ft_lst])[:, 1]

                fpr_off, tpr_off, _ = roc_curve(v[dep], y_pred,sample_weight=v['weight'])
                off_ks = abs(fpr_off - tpr_off).max()
                f2, th, precision, recall = f2_score(v[dep], y_pred, v['weight'])
                plt.plot(fpr_off, tpr_off, label=k)

                df_y_pred = pd.DataFrame({'y': y_pred})
                df_y_pred.loc[df_y_pred['y'] >= th, 'y'] = 1
                df_y_pred.loc[df_y_pred['y'] < th, 'y'] = 0

                # res = res.append({'datasets': k,
                #                   'auc': auc(fpr_off, tpr_off),
                #                   'precision': precision,
                #                   'recall': recall,
                #                   'ks': off_ks,
                #                   'f2': f2,
                #                   'th': th,
                #                   'ratio': df_y_pred['y'].mean(),
                #                   'count': len(v)},
                #                  ignore_index=True)
                add_df = pd.DataFrame({'datasets': k,
                                        'auc': auc(fpr_off, tpr_off),
                                        'precision': precision,
                                        'recall': recall,
                                        'ks': off_ks,
                                        'f2': f2,
                                        'th': th,
                                        'ratio': df_y_pred['y'].mean(),
                                        'count': len(v)}, index=[0])
                res = pd.concat([res, add_df], axis=0)

        else:
            if str(type(model)).find('Regressor') != -1 or str(type(model)).find('OneClassSVM') != -1:
                pass
                # y_pred = model.predict(v[ft_lst])
                # fpr_off, tpr_off, _ = roc_curve(v[dep], y_pred)
                # off_ks = abs(fpr_off - tpr_off).max()
                # res = res.append({'datasets': k,
                #                   'auc': auc(fpr_off, tpr_off),
                #                   # 'precision': precision_score(v[dep], y_pred),
                #                   # 'recall': recall_score(v[dep], y_pred),
                #                   'ks': off_ks},
                #                  ignore_index=True)
            else:
                if str(type(model)).find('DecisionTreeRegressor') != -1:
                    df_y_pred = pd.DataFrame({'y': model.predict(v[ft_lst])[:, 1]})
                else:
                    df_y_pred = pd.DataFrame({'y': model.predict_proba(v[ft_lst])[:, 1]})
                best_th = res.loc[res['datasets'] == 'oot', 'th'].values[0]
                df_y_pred.loc[df_y_pred['y'] >= best_th, 'y'] = 1
                df_y_pred.loc[df_y_pred['y'] < best_th, 'y'] = 0

                # res = res.append({'datasets': k,
                #                   'th': best_th,
                #                   'ratio': df_y_pred['y'].mean(),
                #                   'count': df_y_pred['y'].sum()},
                #                  ignore_index=True)
                add_df = pd.DataFrame({'datasets': k,
                                        'th': best_th,
                                        'ratio': df_y_pred['y'].mean(),
                                        'count': df_y_pred['y'].sum()}, index=[0])
                res = pd.concat([res, add_df], axis=0)

    print(res)

    if str(type(model)).find('Regressor') != -1 or str(type(model)).find('OneClassSVM') != -1:
        res = res[['datasets', 'auc', 'ks']]
    else:
        res = res[['datasets', 'auc', 'ks', 'precision', 'recall', 'f2', 'th', 'ratio', 'count']]
    print(res)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

    return res


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


def lr_model(ft_lst, dep, fit_set='train', datasets={}):
    model = LogisticRegression(C=0.1,
                               max_iter=2000,
                               class_weight='balanced',
                               penalty='l2',
                               solver='liblinear',
                               verbose=0,
                               n_jobs=1)
    # model = LogisticRegression(C=0.1,
    #                            max_iter=2000,
    #                            class_weight='balanced',
    #                            penalty='l2',
    #                            solver='liblinear',
    #                            verbose=0,
    #                            n_jobs=1,
    #                            fit_intercept=False)

    model.fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], sample_weight=datasets[fit_set]['weight'])
    res = show(model, ft_lst, dep, datasets)
    return model, res


def xgb_model(ft_lst, dep, fit_set='train', datasets={}):
    model = xgb.XGBClassifier(learning_rate=0.1,
                              n_estimators=80,
                              max_depth=3,
                              min_child_weight=10,
                              subsample=0.7,
                              nthread=-1,
                              scale_pos_weight=1,
                              random_state=2022,
                              n_jobs=-1,
                              reg_lambda=300
                              )
    model.fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], sample_weight=datasets[fit_set]['weight'])
    res = show(model, ft_lst, dep, datasets)
    return model, res


def lgb_model(ft_lst, dep, fit_set='train', datasets={}):
    model = lgb.LGBMClassifier(boosting_type="gbdt",
                               objective='binary',
                               learning_rate=0.1,
                               n_estimators=80,
                               max_depth=3,
                               num_leaves=15,
                               min_child_weight=0.01,
                               min_child_samples=10,
                               reg_lambda=300,
                               feature_fraction=0.7,
                               bagging_freq=10,
                               random_state=2022,
                               verbose = -1,
                               )
    model.fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], eval_metric='auc',
              eval_set=(datasets['valid'][ft_lst], datasets['valid'][dep]), callbacks=[lgb.early_stopping(100,verbose=False)],
              sample_weight=datasets[fit_set]['weight'])
    res = show(model, ft_lst, dep, datasets)
    return model, res


def gbdt_model(ft_lst, dep, fit_set='train', datasets={}):
    model = GradientBoostingClassifier(learning_rate=0.1,
                                       n_estimators=80,
                                       max_depth=3,
                                       max_features=None,
                                       min_samples_split=5,
                                       subsample=0.7,
                                       min_samples_leaf=5)
    model.fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], sample_weight=datasets[fit_set]['weight'])
    res = (model, ft_lst, dep, datasets)
    return model, res


def bi_train(df, dep='label', exclude=None, model_switch=[1, 1, 1, 1], if_against_weight=None):
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
        print("逻辑回归：")
        model_lr = lr_model(ft_lst, dep, datasets=datasets)
        res['model_lr'],  res['model_lr_res'] = model_lr
        # print("逻辑回归反向：")
        # model_lr_inv = lr_model(ft_lst, dep, fit_set='oot', datasets=datasets)
        # res['model_lr_inv'] = model_lr_inv

    if model_switch[1]:
        print("XGBoost：")
        model_xgb = xgb_model(ft_lst, dep, datasets=datasets)
        res['model_xgb'],  res['model_xgb_res'] = model_xgb
        # print("XGBoost反向：")
        # model_xgb_inv = xgb_model(ft_lst, dep, fit_set='oot', datasets=datasets)
        # res['model_xgb_inv'] = model_xgb_inv

    if model_switch[2]:
        print("LightGBM：")
        model_lgb = lgb_model(ft_lst, dep, datasets=datasets)
        res['model_lgb'],  res['model_lgb_res'] = model_lgb
        # print("LightGBM反向：")
        # model_lgb_inv = lgb_model(ft_lst, dep, fit_set='oot', datasets=datasets)
        # res['model_lgb_inv'] = model_lgb_inv

    if model_switch[3]:
        print("GBDT：")
        model_gbdt = gbdt_model(ft_lst, dep, datasets=datasets)
        res['model_gbdt'],  res['model_gbdt_res'] = model_gbdt
        # print("LightGBM反向：")
        # model_gbdt_inv = gbdt_model(ft_lst, dep, fit_set='oot', datasets=datasets)
        # res['model_gbdt_inv'] = model_gbdt_inv

    return res


def bi_train_model(df, model, dep='label', exclude=None, if_against_weight=None, fit_set='train'):
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
    if model[1]:
        print("逻辑回归：")
        model[0].fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], sample_weight=datasets[fit_set]['weight'])
        show(model[0], ft_lst, dep, datasets)
        res['model_lr'], res['model_lr_res'] = model[0]
        # print("逻辑回归反向：")
        # model_lr_inv = lr_model(ft_lst, dep, fit_set='oot', datasets=datasets)
        # res['model_lr_inv'] = model_lr_inv

    if model[1]:
        print("XGBoost：")
        model[1].fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], sample_weight=datasets[fit_set]['weight'])
        show(model[1], ft_lst, dep, datasets)
        res['model_xgb'], res['model_xgb_res'] = model[1]
        # print("XGBoost反向：")
        # model_xgb_inv = xgb_model(ft_lst, dep, fit_set='oot', datasets=datasets)
        # res['model_xgb_inv'] = model_xgb_inv

    if model[2]:
        print("LightGBM：")
        model[2].fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], sample_weight=datasets['train']['weight'])
        show(model[2], ft_lst, dep, datasets)
        res['model_lgb'], res['model_lgb_res'] = model[2]
        # print("LightGBM反向：")
        # model_lgb_inv = lgb_model(ft_lst, dep, fit_set='oot', datasets=datasets)
        # res['model_lgb_inv'] = model_lgb_inv

    if model[3]:
        print("GBDT：")
        model[3].fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], sample_weight=datasets[fit_set]['weight'])
        show(model[3], ft_lst, dep, datasets)
        res['model_gbdt'], res['model_gbdt_res'] = model[3]
        # print("LightGBM反向：")
        # model_gbdt_inv = gbdt_model(ft_lst, dep, fit_set='oot', datasets=datasets)
        # res['model_gbdt_inv'] = model_gbdt_inv

    return res
