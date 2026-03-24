#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    # @Time    : 2019/7/16 10:29
    # @Author  : F.Li
    # @File    : DataPreprocessing .py
"""

import random
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from new_tools.reduce_mem_usage import reduce_mem_usage_dic

def by_stratify(data, test_size, random_state, dep, date):
    data_split = data[data['target'].isin(['train', 'valid'])]
    oot = data[data['target'] == 'oot']
    train, val, y_train, y_valid = train_test_split(data_split, data_split[dep], test_size=test_size,
                                                    stratify=data_split[dep], random_state=random_state)
    res = pd.DataFrame({'count': [data.shape[0], train.shape[0], val.shape[0], oot.shape[0]],
                        'bad_rate': [data[dep].mean(), train[dep].mean(), val[dep].mean(), oot[dep].mean()],
                        'bad': [data[dep].sum(), train[dep].sum(), val[dep].sum(), oot[dep].sum()],
                        'time_min': [data[date].min(), train[date].min(), val[date].min(), oot[date].min()],
                        'time_max': [data[date].max(), train[date].max(), val[date].max(), oot[date].max()]})
    res.rename(index={0: 'all', 1: 'train', 2: 'valid', 3: 'oot'}, inplace=True)
    print(res)
    print('ratio_bad_good: ', train[dep].sum() / (len(train[dep]) - train[dep].sum()))
    data.loc[train.index, 'target'] = 'train'
    data.loc[val.index, 'target'] = 'valid'
    print(data['target'].value_counts())
    return data, res


def by_threshold(data, test_size, random_state, dep, date):
    data_split = data[data['target'].isin(['train', 'valid'])]
    oot = data[data['target'] == 'oot']
    # data_split.reset_index(drop=True, inplace=True)
    # random.shuffle(data_split)
    # train = data_split[data_split.index % 10 >= test_size]
    # val = data_split[data_split.index % 10 < test_size]

    train, val, y_train, y_valid = train_test_split(data_split, data_split[dep], test_size=test_size,
                                                    stratify=data_split[dep], random_state=random_state)

    res = pd.DataFrame({'count': [data.shape[0], train.shape[0], val.shape[0], oot.shape[0]],
                        'bad_rate': [data[dep].mean(), train[dep].mean(), val[dep].mean(), oot[dep].mean()],
                        'bad': [data[dep].sum(), train[dep].sum(), val[dep].sum(), oot[dep].sum()],
                        'time_min': [data[date].min(), train[date].min(), val[date].min(), oot[date].min()],
                        'time_max': [data[date].max(), train[date].max(), val[date].max(), oot[date].max()]})
    res.rename(index={0: 'all', 1: 'train', 2: 'valid', 3: 'oot'}, inplace=True)
    print(res)
    print('ratio_bad_good: ', train[dep].sum() / (len(train[dep]) - train[dep].sum()))
    data.loc[train.index, 'target'] = 'train'
    data.loc[val.index, 'target'] = 'valid'
    print(data['target'].value_counts())
    return data, res


def datasets_drop_dup(df, col_name):
    df = df.drop_duplicates([col_name], keep='first')


def by_time(self, df, data_start, date_end, threshold):
    df['dtn'] = (df[date_end] - df[data_start]).apply(lambda x: x.days)
    self.datasets_train = df[df['dtn'] < threshold]


# ======================================================================================================================
# ********************************************* 样本选择 ****************************************************************
# ======================================================================================================================

class SampleChoose(object):
    def __init__(self, FILE_PATH, SAVE_PATH, dep, ex_lst, df=None):
        self.FILE_PATH = FILE_PATH
        self.SAVE_PATH = SAVE_PATH
        self.dep = dep
        if df is None:
            self.df = pd.read_csv(self.FILE_PATH, encoding='utf-8-sig')
            self.df = reduce_mem_usage_dic(self.df)
        else:
            self.df = df
        self.ex_lst = ex_lst

    def lr_model(self, ft_lst, dep, fit_set='train', datasets={}):
        model = LogisticRegression(C=0.1,
                                   class_weight='balanced',
                                   max_iter=2000,
                                   penalty='l2',
                                   solver='liblinear',
                                   verbose=0,
                                   n_jobs=1)
        model.fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], sample_weight=datasets[fit_set]['weight'])
        return self.show(model, ft_lst, dep, 'lr', datasets)

    def xgb_model(self, ft_lst, dep, fit_set='train', datasets={}):
        model = xgb.XGBClassifier(learning_rate=0.1,
                                  n_estimators=80,
                                  max_depth=3,
                                  min_child_weight=10,
                                  subsample=0.7,
                                  nthread=-1,
                                  scale_pos_weight=1,
                                  random_state=1,
                                  n_jobs=-1,
                                  reg_lambda=300,
                                #   predictor='cpu_predictor'
                                  )
        model.fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], sample_weight=datasets[fit_set]['weight'])

        return self.show(model, ft_lst, dep, 'xgb', datasets)

    def lgb_model(self, ft_lst, dep, fit_set='train', datasets={}):
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
                                   n_estimators=80,
                                   silent=False)
        model.fit(datasets[fit_set][ft_lst], datasets[fit_set][dep], eval_metric='auc',
                  eval_set=(datasets['valid'][ft_lst], datasets['valid'][dep]), early_stopping_rounds=500,
                  sample_weight=datasets[fit_set]['weight'], verbose=False)

        return self.show(model, ft_lst, dep, 'lgb', datasets)

    def show(self, model, ft_lst, dep, model_name, datasets={}):
        res = {}
        for key, value in datasets.items():
            y_pred = model.predict_proba(value[ft_lst])[:, 1]
            fpr_, tpr_, _ = roc_curve(value[dep], y_pred)
            ks = abs(fpr_ - tpr_).max()
            res[key + '_auc_' + model_name] = auc(fpr_, tpr_)
            res[key + '_ks_' + model_name] = ks
        print(res)
        return res

    def bi_train(self, df, model_switch, dep='label', exclude=None):
        data = copy.deepcopy(df)
        ft_lst = [i for i in data.columns if i not in exclude]

        datasets = {}
        set_lst = list(df['target'].unique())
        for i in set_lst:
            datasets[i] = data[data['target'] == i]

        res = []
        if model_switch[0]:
            dic_lr = self.lr_model(ft_lst, dep, datasets=datasets)
            res.append(dic_lr)
            # print("逻辑回归反向：")
            # model_lr_inv = lr_model(ft_lst, dep, fit_set='oot', datasets=datasets)
            # res['model_lr_inv'] = model_lr_inv

        if model_switch[1]:
            dic_xgb = self.xgb_model(ft_lst, dep, datasets=datasets)
            res.append(dic_xgb)
            # print("XGBoost反向：")
            # model_xgb_inv = xgb_model(ft_lst, dep, fit_set='oot', datasets=datasets)
            # res['model_xgb_inv'] = model_xgb_inv

        if model_switch[2]:
            dic_lgb = self.lgb_model(ft_lst, dep, datasets=datasets)
            res.append(dic_lgb)
            # print("LightGBM反向：")
            # model_lgb_inv = lgb_model(ft_lst, dep, fit_set='oot', datasets=datasets)
            # res['model_lgb_inv'] = model_lgb_inv

        return res

    def find(self, flag_start, num_iter, gap, model_switch=[1, 1, 1]):
        flag = flag_start
        df = copy.deepcopy(self.df)
        data_split = df[df['target'].isin(['train', 'valid'])]
        res = pd.DataFrame()
        while flag < num_iter:
            print(flag, '=========================')
            ks_auc = {}
            train, valid, y_train, y_valid = train_test_split(data_split, data_split[self.dep], test_size=0.3,
                                                              stratify=data_split[self.dep], random_state=flag)
            df.loc[train.index, 'target'] = 'train'
            df.loc[valid.index, 'target'] = 'valid'

            dic_res = self.bi_train(df, dep=self.dep, exclude=self.ex_lst, model_switch=model_switch)
            for value_ in dic_res:
                ks_auc.update(value_)
            ks_auc['flag'] = flag
            # res = res.append(ks_auc, ignore_index=True)
            add_df = pd.DataFrame(ks_auc, index=[0])
            res = pd.concat([res, add_df], ignore_index=True)
            flag += gap
        self.res = res.copy()

        return self.res

    def find_by_against_validation(self, num_iter, model_switch=[1, 1, 1]):

        df = copy.deepcopy(self.df)
        data_split = df[df['target'].isin(['train', 'valid'])]

        data = copy.deepcopy(df)
        ft_lst = [i for i in data.columns if i not in self.ex_lst]

        i = 1
        while True:

            random_state = int(np.random.randint(1, 10000000, 1))  # 随机种子
            x_train, x_test, y_train, y_test = train_test_split(data_split, data_split[self.dep], test_size=0.3,
                                                                random_state=random_state,
                                                                stratify=data_split[self.dep], )

            df.loc[x_train.index, 'test_label'] = 0
            df.loc[x_test.index, 'test_label'] = 1

            # 开始验证训练集和验证集分布是否一致，如果越接近0.5，就说明一样
            clf = LogisticRegression()
            score = cross_val_score(clf, df[df['target'].isin(['train', 'valid'])][ft_lst],
                                    df[df['target'].isin(['train', 'valid'])]['test_label'],
                                    cv=5, n_jobs=-1, scoring='roc_auc').mean()

            print('第{}次迭代，随机种子为：{}，AUC为：{}'.format(i, random_state, score))

            if 0.49 <= score <= 0.51:
                print('-' * 100)
                print('最终随机种子为：{}，AUC为：{}'.format(random_state, score))
                break

            i += 1
        return random_state

    def save(self, SAVE_PATH):
        if SAVE_PATH is not None:
            self.res.to_csv(SAVE_PATH, index=False)
        else:
            self.res.to_csv(self.SAVE_PATH, index=False)

    def print(self, type, threshold, SAVE_PATH):
        if SAVE_PATH is not None:
            res = pd.read_csv(SAVE_PATH, index=False)
        else:
            res = self.res
        if type == 'lr':
            res = res[res['dev_ks_lr'] > threshold[0]]
            res = res[res['val_ks_lr'] > threshold[1]]
            res = res[res['oot_ks_lr'] > threshold[2]]
            print(res)
        elif type == 'xgb':
            res = res[res['dev_ks_xgb'] > threshold[0]]
            res = res[res['val_ks_xgb'] > threshold[1]]
            res = res[res['oot_ks_xgb'] > threshold[2]]
            print(res)
        else:
            res = res[res['dev_ks_lgb'] > threshold[0]]
            res = res[res['val_ks_lgb'] > threshold[1]]
            res = res[res['oot_ks_lgb'] > threshold[2]]
            print(res)

        return res

    def find_xgb(self, best_param, flag_start, num_iter, gap):
        flag = flag_start
        df = copy.deepcopy(self.df)

        data_split = df[df['target'].isin(['train', 'valid'])]
        oot = df[df['target'] == 'oot']

        model = xgb.XGBClassifier(**best_param)
        ft_lst = [i for i in df.columns if i not in self.ex_lst]
        res = pd.DataFrame()

        while flag < num_iter:
            print('-+-', flag, '-+-')
            train, valid, y_train, y_valid = train_test_split(data_split, data_split[self.dep], test_size=0.3,
                                                              stratify=data_split[self.dep], random_state=flag)
            df.loc[train.index, 'target'] = 'train'
            df.loc[valid.index, 'target'] = 'valid'

            model.fit(train[ft_lst], train[self.dep])

            y_pred = model.predict_proba(train[ft_lst])[:, 1]
            fpr_dev, tpr_dev, _ = roc_curve(train[self.dep], y_pred)
            train_ks = abs(fpr_dev - tpr_dev).max()

            y_pred = model.predict_proba(valid[ft_lst])[:, 1]
            fpr_val, tpr_val, _ = roc_curve(valid[self.dep], y_pred)
            val_ks = abs(fpr_val - tpr_val).max()

            y_pred = model.predict_proba(oot[ft_lst])[:, 1]
            fpr_off, tpr_off, _ = roc_curve(oot[self.dep], y_pred)
            off_ks = abs(fpr_off - tpr_off).max()

            print('dev_ks: ', train_ks, '  val_ks: ', val_ks, '  oot_ks: ', off_ks,
                  '  dev_auc: ', auc(fpr_dev, tpr_dev), '  val_auc: ', auc(fpr_val, tpr_val),
                  '  oot_auc: ', auc(fpr_off, tpr_off))

            res = res.append({'random_seed': flag,
                              'dev_ks': train_ks,
                              'val_ks': val_ks,
                              'oot_ks': off_ks,
                              'dev_auc': auc(fpr_dev, tpr_dev),
                              'val_auc': auc(fpr_val, tpr_val),
                              'oot_auc': auc(fpr_off, tpr_off)}, ignore_index=True)

            flag += gap

        self.res = res[['random_seed', 'dev_auc', 'val_auc', 'oot_auc', 'dev_ks', 'val_ks', 'oot_ks']]

        return self.res


