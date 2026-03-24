#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier


def rf(data, ft_lst, dep, scoring_type):
    x = data[data['target'] == 'train'][ft_lst]
    y = data[data['target'] == 'train'][dep]
    x_test = data[data['target'] == 'valid'][ft_lst]
    y_test = data[data['target'] == 'valid'][dep]
    param_test1 = {
        'n_estimators': range(10, 101, 10),
        'max_depth': range(2, 6, 1),
        'min_samples_split': range(2, 102, 50),
        'min_samples_leaf': range(1, 60, 20)
    }
    model = GridSearchCV(estimator=RandomForestClassifier(n_estimators=90,
                                                          criterion='gini',
                                                          max_depth=3,
                                                          min_samples_split=2,
                                                          min_samples_leaf=1,
                                                          max_features='auto',
                                                          bootstrap=True,
                                                          class_weight=None,
                                                          n_jobs=None,
                                                          oob_score=False,
                                                          random_state=666,
                                                          verbose=0,
                                                          warm_start=False),
                         param_grid=param_test1,
                         cv=5,
                         scoring=scoring_type,
                         verbose=1
                         )
    model.fit(x, y)

    y_pred_train = model.predict_proba(x)[:, 1]
    y_pred_valid = model.predict_proba(x_test)[:, 1]

    fpr_dev, tpr_dev, _ = roc_curve(y, y_pred_train)
    train_ks = abs(fpr_dev - tpr_dev).max()

    fpr_val, tpr_val, _ = roc_curve(y_test, y_pred_valid)
    val_ks = abs(fpr_val - tpr_val).max()

    print('train_ks : ', train_ks, '  valid_ks : ', val_ks, '  train_auc : ', auc(fpr_dev, tpr_dev), '  valid_auc : ',
          auc(fpr_val, tpr_val))
    print('用网格搜索找到的最优超参数为:', model.best_params_)
    return model


def xgb(data, ft_lst, dep, scoring_type):
    x = data[data['target'] == 'train'][ft_lst]
    y = data[data['target'] == 'train'][dep]
    x_test = data[data['target'] == 'valid'][ft_lst]
    y_test = data[data['target'] == 'valid'][dep]
    param_test1 = {
        'n_estimators': range(10, 101, 10),
        'max_depth': range(3, 6, 1),
        'min_samples_split': range(2, 102, 50),
        'min_samples_leaf': range(1, 60, 20)
    }
    model = GridSearchCV(estimator=RandomForestClassifier(n_estimators=90,
                                                          criterion='gini',
                                                          max_depth=3,
                                                          min_samples_split=2,
                                                          min_samples_leaf=1,
                                                          max_features='auto',
                                                          bootstrap=True,
                                                          class_weight=None,
                                                          n_jobs=None,
                                                          oob_score=False,
                                                          random_state=666,
                                                          verbose=0,
                                                          warm_start=False),
                         param_grid=param_test1,
                         cv=5,
                         scoring=scoring_type,
                         verbose=1
                         )
    model.fit(x, y)

    y_pred_train = model.predict_proba(x)[:, 1]
    y_pred_valid = model.predict_proba(x_test)[:, 1]

    fpr_dev, tpr_dev, _ = roc_curve(y, y_pred_train)
    train_ks = abs(fpr_dev - tpr_dev).max()

    fpr_val, tpr_val, _ = roc_curve(y_test, y_pred_valid)
    val_ks = abs(fpr_val - tpr_val).max()

    print('train_ks : ', train_ks, '  valid_ks : ', val_ks, '  train_auc : ', auc(fpr_dev, tpr_dev), '  valid_auc : ',
          auc(fpr_val, tpr_val))
    print('用网格搜索找到的最优超参数为:', model.best_params_)
    return model