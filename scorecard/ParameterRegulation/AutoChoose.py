#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : F. Li
    @Time    :
    @Use     : 自动化调参框架
"""

import xgboost as xgb
from copy import deepcopy
from ScoreCard.Envalue.KS import KS


class AutoChoose(object):
    def __init__(self, datasets, var_names, weight, dep):
        self.datasets = datasets
        self.var_names = var_names
        self.weight = weight
        self.dep = dep

    def params_choose_lr(self):
        dev_data = self.datasets.get['train']
        off_data = self.datasets.get['oot']
        params = {
            "max_depth": 3,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "min_child_weight": 1,
            "subsample": 1,
            "scale_pos_weight": 1
        }
        ks = KS(dev_data)
        model = xgb.XGBClassifier(list(params.items()), random_state=7)
        model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        devks = ks.solve_xgb(model, dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        offks = ks.solve_xgb(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
        train_number = 0
        best_param = {"devks": devks, "offks": offks}
        print("train_number: %s, best_param: %s, params: %s" % (train_number, best_param, params))

        for key, value in params.items():
            if not isinstance(value, list):
                best_param[key] = value
                continue
            for val in value:
                tmp = deepcopy(best_param)
                model = xgb.XGBClassifier(learning_rate=tmp.get("learning_rate", 0.1),
                                          n_estimators=tmp.get("n_estimators", 100),
                                          max_depth=tmp.get("max_depth", 3),
                                          min_child_weight=tmp.get("min_child_weight", 1),
                                          subsample=tmp.get("subsample", 1),
                                          objective=tmp.get("objective", "binary:logistic"),
                                          nthread=tmp.get("nthread", 8),
                                          scale_pos_weight=tmp.get("scale_pos_weight", 1),
                                          random_state=tmp.get("random_state", 7),
                                          n_jobs=tmp.get("n_jobs", 8))
                model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
                offks = ks.solve_xgb(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
                if offks > best_param.get("offks", 0):
                    best_param[key] = val
                    devks = ks.solve_xgb(model, dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
                    best_param["devks"], best_param["offks"] = devks, offks
                print("%s = %s, best_param: %s, offks: %s" % (key, val, best_param, offks))

    def auto_choose_params_lr(self):
        dev_data = self.datasets.get("dev", "")
        off_data = self.datasets.get("off", "")
        params = {
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "min_child_weight": 1,
            "subsample": 1,
            "scale_pos_weight": 1
        }
        ks = KS(dev_data)
        model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),
                                  learning_rate=params.get("learning_rate", 0.05),
                                  n_estimators=params.get("n_estimators", 100),
                                  min_child_weight=params.get("min_child_weight", 1),
                                  subsample=params.get("subsample", 1),
                                  scale_pos_weight=params.get("scale_pos_weight", 1),
                                  nthread=8,
                                  n_jobs=8,
                                  random_state=7)
        model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        offks = ks.solve(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
        train_number = 0
        print("train_number: %s, offks: %s, params: %s" % (train_number, offks, params))
        while True:
            dic = {
                "max_depth": [2, 1, -1, -2],
                "learning_rate": [0.1, 0.001, -0.001, -0.1],
                "n_estimators": [20, 1, -1, -20],
                "min_child_weight": [20, 1, -1, -20],
                "subsample": [0.1, 0.001, -0.001, -0.1],
                "scale_pos_weight": [20, 1, -1, -20]
            }
            outks = []
            for (key, values) in dic.items():
                for v in values:
                    if v + params[key] > 0:
                        params, offks, train_number = self.check_params(dev_data, off_data, params,
                                                                        key, train_number, v, offks)
                        outks.append(offks)
            if len(set(outks)) == 1:
                break
        print("Best params: ", params)
        model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),
                                  learning_rate=params.get("learning_rate", 0.05),
                                  n_estimators=params.get("n_estimators", 100),
                                  min_child_weight=params.get("min_child_weight", 1),
                                  subsample=params.get("subsample", 1),
                                  scale_pos_weight=params.get("scale_pos_weight", 1),
                                  nthread=8,
                                  n_jobs=8,
                                  random_state=7)
        model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        ks.plot_xgb(model=model, bins=20)

    def auto_choose_param_xgb(self, target="offks"):
        """
        :param target:
                "offks": offks最大化;
                "minus": 1-abs(devks-offks) 最大化;
                "avg": (devks+offks)/2  最大化
                "weight": offks + abs(offks - devks) * 0.2 最大化
                "mzh1": offks + (offks - devks) * 0.2 最大化
                "mzh2": (offks + (offks - devks) * 0.2)**2 最大化
                其余取值均使用跨时间测试集offks  最大化
                当业务稳定性较差时，应将0.2改为更大的值
        :return: 输出最优模型变量
        """
        dev_data = self.datasets.get("dev", "")
        off_data = self.datasets.get("off", "")
        params = {
            "max_depth": 5,
            "learning_rate": 0.09,
            "n_estimators": 120,
            "min_child_weight": 50,
            "subsample": 1,
            "scale_pos_weight": 1,
            "reg_lambda": 21
        }
        model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),
                                  learning_rate=params.get("learning_rate", 0.05),
                                  n_estimators=params.get("n_estimators", 100),
                                  min_child_weight=params.get("min_child_weight", 1),
                                  subsample=params.get("subsample", 1),
                                  scale_pos_weight=params.get("scale_pos_weight", 1),
                                  reg_lambda=params.get("reg_lambda", 1),
                                  nthread=8,
                                  n_jobs=8,
                                  random_state=7)
        model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        ks = KS(self.datasets, self.dep, self.var_names, self.weight)
        devks = ks.solve(model, dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        offks = ks.solve(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
        train_number = 0
        print("train_number: %s, devks: %s, offks: %s, params: %s" % (train_number, devks, offks, params))
        dic = {
            "learning_rate": [0.05, -0.05],
            "max_depth": [1, -1],
            "n_estimators": [20, 5, -5, -20],
            "min_child_weight": [20, 5, -5, -20],
            "subsample": [0.05, -0.05],
            "scale_pos_weight": [20, 5, -5, -20],
            "reg_lambda": [10, -10]}
        targetks = self.target_value(target=target, devks=devks, offks=offks, w=0.2)
        old_devks = devks
        old_offks = offks
        while True:
            targetks_lis = []
            for (key, values) in dic.items():
                for v in values:
                    if v + params[key] > 0:
                        params, targetks, train_number = self.check_params_xgb(dev_data, off_data, params, key, train_number,
                                                                           v, target, targetks)
                        targetks_n = self.target_value(target=target, devks=devks, offks=offks, w=0.2)
                        if targetks < targetks_n:
                            old_devks = devks
                            old_offks = offks
                            targetks_lis.append(targetks)
            print("-"*50)
            if not targetks_lis:
                break
        print("Best params: ", params)
        model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),
                                  learning_rate=params.get("learning_rate", 0.05),
                                  n_estimators=params.get("n_estimators", 100),
                                  min_child_weight=params.get("min_child_weight", 1),
                                  subsample=params.get("subsample", 1),
                                  scale_pos_weight=params.get("scale_pos_weight", 1),
                                  reg_lambda=params.get("reg_lambda", 1),
                                  nthread=8,
                                  n_jobs=8,
                                  random_state=7)
        model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        ks.plot_xgb(model=model, bins=20)

 # 参数搜索方案
    def check_params_xgb(self, dev_data, off_data, params, param, train_number, step, target, targetks):
        '''
            当前向搜索对调参策略有提升时， 继续前向搜索。
            否则进行后向搜索
        '''
        while True:
            try:
                if params[param] + step > 0:
                    params[param] += step
                    ks = KS(self.datasets, self.dep, self.var_names, self.weight)
                    model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),
                                              learning_rate=params.get("learning_rate", 0.05),
                                              n_estimators=params.get("n_estimators", 100),
                                              min_child_weight=params.get("min_child_weight", 1),
                                              subsample=params.get("subsample", 1),
                                              scale_pos_weight=params.get("scale_pos_weight", 1),
                                              nthread=10,
                                              n_jobs=10,
                                              random_state=7)
                    model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
                    devks = ks.solve(model, dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
                    offks = ks.solve(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
                    train_number += 1
                    targetks_n = self.target_value(target=target,devks=devks, offks=offks, w=0.2)
                    if targetks < targetks_n:
                        print("(Good) train_number: %s, devks: %s, offks: %s, params: %s" % (
                            train_number, devks, offks, params))
                        targetks = targetks_n
                        old_devks = devks
                        old_offks = offks
                    else:
                        print("(Bad) train_number: %s, devks: %s, offks: %s, params: %s" %
                              (train_number, devks, offks, params))
                        break
                else:
                    break
            except:
                break
        params[param] -= step
        return params, targetks, train_number

    def target_value(self, target, devks, offks, w=0.2):
        if target == "offks":
            return offks
        elif target == "avg":
            return (devks + offks) / 2
        elif target == "minus":
            return 1-abs(devks-offks)
        elif target == "weight":
            return offks - abs(devks - offks) * w
        elif target == "quad":
            return (offks + (offks - devks) * 0.2) ** 2
        else:
            return offks + (offks - devks) * 0.2
