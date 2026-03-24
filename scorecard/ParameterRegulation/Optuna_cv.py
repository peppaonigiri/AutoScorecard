
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from ParameterRegulation.Optuna_strategy import strategy, param_lr, param_xgb, param_lgb


def model_fit(model, datasets, ft_lst, dep):
    model.fit(datasets['train'][ft_lst], datasets['train'][dep],  sample_weight=datasets['train']['weight'])

    res = dict()

    for k, v in datasets.items():
        y_pred = model.predict_proba(v[ft_lst])[:, 1]
        fpr_off, tpr_off, _ = roc_curve(v[dep], y_pred)
        off_ks = abs(fpr_off - tpr_off).max()

        res[k + '_ks'] = off_ks
        res[k + '_auc'] = auc(fpr_off, tpr_off)

    return res


def objective_cv(data_all, trial, ex_lst, strategy_type, cv=5, model_type='xgb', dep='label'):
    ft_lst = [i for i in data_all.columns if i not in ex_lst]
    data_split = data_all[data_all['target'].isin(['train', 'valid'])].copy()
    data_split.reset_index(drop=True, inplace=True)
    datasets = {}
    for i in list(data_all['target'].unique()):
        datasets[i] = data_all[data_all['target'] == i]

    if model_type == 'lr':
        param = param_lr(trial)
        model = LogisticRegression(**param)
    elif model_type == 'xgb':
        param = param_xgb(trial)
        model = xgb.XGBClassifier(**param)
    else:
        param = param_lgb(trial)
        model = lgb.LGBMClassifier(**param)

    res = pd.DataFrame()
    kf = StratifiedKFold(n_splits=cv, random_state=2021, shuffle=True).split(datasets['train'][ft_lst].values,
                                                                             datasets['train'][[dep]].values)

    for i, (train_fold, test_fold) in enumerate(kf):
        print(i, train_fold, test_fold)
        x_train, x_valid, y_train, y_valid = datasets['train'][ft_lst].values[train_fold, :], \
                                             datasets['train'][ft_lst].values[test_fold, :], \
                                             datasets['train'][[dep]].values[train_fold, :], \
                                             datasets['train'][[dep]].values[test_fold, :]
        model.fit(x_train, y_train)

        auc_ks = model_fit(model, datasets, ft_lst, dep)
        auc_ks['test_fold'] = i
        res = res.append(auc_ks, ignore_index=True)

    mean_ = {'test_fold': 'mean'}
    for k, v in datasets.items():
        mean_[k+'_ks'] = res[k+'_ks'].mean()

    res = res.append(mean_, ignore_index=True)
    print(res)

    return strategy(res, strategy_type)


def run(objective_cv, data_all, ex_lst, strategy, exist_oot, n_trials=100, model_type='xgb'):
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(lambda trial: objective_cv(data_all, trial, ex_lst, strategy_type=strategy, cv=5,
                                              model_type=model_type, dep='label', exist_oot=exist_oot),
                   n_trials=n_trials)

    print(study.best_trial)
    # plot_slice(study)
    # hist = study.trials_dataframe()
    # hist.head()
    return study
