
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from random import randint
from optuna.visualization import plot_slice
from ParameterRegulation.Optuna_strategy_reg import strategy, param_lr, param_xgb, param_lgb
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error


def model_fit(model, datasets, ft_lst, dep):
    model.fit(datasets['train'][ft_lst], datasets['train'][dep])

    res = dict()

    for k, v in datasets.items():
        y_pred = model.predict(v[ft_lst])

        # res[k + '_r2_score'] = r2_score(v[dep], y_pred)
        res[k + '_mean_squared_error'] = mean_squared_error(v[dep], y_pred)
        # res[k + '_explained_variance_score'] = explained_variance_score(v[dep], y_pred)
        res[k + '_mean_absolute_error'] = mean_absolute_error(v[dep], y_pred)

    return res


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).


def objective_fix_sample(data_all, trial, ex_lst, strategy_type, model_type='xgb', dep='label'):
    ft_lst = [i for i in data_all.columns if i not in ex_lst]
    data_split = data_all[data_all['target'].isin(['train', 'valid'])].copy()
    data_split.reset_index(drop=True, inplace=True)

    datasets = {}
    for i in list(data_all['target'].unique()):
        datasets[i] = data_all[data_all['target'] == i]

    if model_type == 'lr':
        param = param_lr(trial)
        model = linear_model.LinearRegression(**param)
    elif model_type == 'xgb':
        param = param_xgb(trial)
        model = xgb.XGBRegressor(**param)
    else:
        param = param_lgb(trial)
        model = lgb.LGBMRegressor(**param)

    res = pd.DataFrame()
    auc_ks = model_fit(model, datasets, ft_lst, dep)
    res = res.append(auc_ks, ignore_index=True)

    print(res)

    return strategy(res, strategy_type)


def run(objective_fix_sample, data_all, ex_lst, strategy, n_trials=100, model_type='xgb'):
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(lambda trial: objective_fix_sample(data_all, trial, ex_lst, strategy_type=strategy,
                                                      model_type=model_type), n_trials=n_trials)
    # study = optuna.create_study(
    #     pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    # )
    # study.optimize(objective_xgb, n_trials=100)
    print(study.best_trial)
    # plot_slice(study)
    # hist = study.trials_dataframe()
    # hist.head()
    return study



