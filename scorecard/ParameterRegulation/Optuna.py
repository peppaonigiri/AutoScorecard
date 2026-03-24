import catboost
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn import tree
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from ParameterRegulation.Optuna_strategy import strategy, param_lr, param_xgb, param_lgb, param_dt, param_rf, \
    param_catboost
from Model.Evaluation.f2_score import f2_score
import numpy as np
import json
from Model.Evaluation.divergence import js_divergence
from ScoreCard.show_strategy import show_stage1
import scipy


# def model_fit(model, datasets, ft_lst, dep):
#     model.fit(datasets['train'][ft_lst], datasets['train'][dep],  sample_weight=datasets['train']['weight'])
#
#     res = dict()
#
#     for k, v in datasets.items():
#         y_pred = model.predict_proba(v[ft_lst])[:, 1]
#         fpr_off, tpr_off, _ = roc_curve(v[dep], y_pred)
#         off_ks = abs(fpr_off - tpr_off).max()
#
#         res[k + '_ks'] = off_ks
#         res[k + '_auc'] = auc(fpr_off, tpr_off)
#         res[k + '_f2'], res[k + '_th'], res[k + '_precision'], res[k + '_recall'] = f2_score(v[dep], y_pred)
#     return res

# def model_fit(model, datasets, ft_lst, dep):
#     model.fit(datasets['train'][ft_lst], datasets['train'][dep],  sample_weight=datasets['train']['weight'])
#
#     res = dict()
#
#     for k, v in datasets.items():
#         if k != 'test':
#             y_pred = model.predict_proba(v[ft_lst])[:, 1]
#             fpr_off, tpr_off, _ = roc_curve(v[dep], y_pred)
#             off_ks = abs(fpr_off - tpr_off).max()
#
#             res[k + '_ks'] = off_ks
#             res[k + '_auc'] = auc(fpr_off, tpr_off)
#             res[k + '_f2'], res[k + '_th'], res[k + '_precision'], res[k + '_recall'] = f2_score(v[dep], y_pred)
#
#             df_y_pred = pd.DataFrame({'y': y_pred})
#             df_y_pred.loc[df_y_pred['y'] >= res[k + '_th'], 'y'] = 1
#             df_y_pred.loc[df_y_pred['y'] < res[k + '_th'], 'y'] = 0
#             res[k + '_ratio'] = df_y_pred['y'].mean() if df_y_pred['y'] is not np.nan else 0
#         else:
#             df_y_pred = pd.DataFrame({'y': model.predict_proba(v[ft_lst])[:, 1]})
#             best_th = res['oot_th']
#             df_y_pred.loc[df_y_pred['y'] >= best_th, 'y'] = 1
#             df_y_pred.loc[df_y_pred['y'] < best_th, 'y'] = 0
#             res[k + '_ratio'] = df_y_pred['y'].mean() if df_y_pred['y'] is not np.nan else 0
#
#     return res

def non_decreasing(lst: list):
    return all(x - y <= 0.001 for x, y in zip(lst, lst[1:]))


def model_fit(model, datasets, ft_lst, dep, model_type='xgb'):
    if model_type == 'catboost':
        cat_fea_list = list(datasets['train'][ft_lst].dtypes[datasets['train'][ft_lst].dtypes == 'object'].index)
        cat_features_index = [ft_lst.index(cat_fea) for cat_fea in cat_fea_list]
        model.fit(datasets['train'][ft_lst], datasets['train'][dep], sample_weight=datasets['train']['weight'],
                  cat_features=cat_features_index)
    else:
        model.fit(datasets['train'][ft_lst], datasets['train'][dep],  sample_weight=datasets['train']['weight'])
    res = dict()

    for k, v in datasets.items():
        if k != 'test':
            if str(type(model)).find('DecisionTreeRegressor') != -1:
                y_pred = model.predict(v[ft_lst])
            else:
                y_pred = model.predict_proba(v[ft_lst])[:, 1]

            fpr_off, tpr_off, _ = roc_curve(v[dep], y_pred, sample_weight=v['weight'])
            off_ks = abs(fpr_off - tpr_off).max()

            res[k + '_ks'] = off_ks
            # res[k + '_auc'] = auc(fpr_off, tpr_off)
            res[k + '_auc'] = roc_auc_score(v[dep], y_pred, sample_weight=v['weight'])
            res[k + '_f2'], res[k + '_th'], res[k + '_precision'], res[k + '_recall'] = f2_score(v[dep], y_pred, v['weight'])
            # 查看各分位下的lift
            sub_5 = np.percentile(y_pred, 5)
            top_5 = np.percentile(y_pred, 95)
            label = v[dep].values
            sub_5_rate = label[y_pred <= sub_5].sum() / len(label[y_pred <= sub_5])
            top_5_rate = label[y_pred >= top_5].sum() / len(label[y_pred >= top_5])
            all_rate = label.sum() / len(label)
            sub_5_lift = sub_5_rate / all_rate
            top_5_lift = top_5_rate / all_rate
            res[k + '_sub_5_lift'] = sub_5_lift
            res[k + '_top_5_lift'] = top_5_lift

            df_y_pred = pd.DataFrame({'y': y_pred})
            df_y_pred.loc[df_y_pred['y'] >= res[k + '_th'], 'y'] = 1
            df_y_pred.loc[df_y_pred['y'] < res[k + '_th'], 'y'] = 0
            res[k + '_ratio'] = df_y_pred['y'].mean() if df_y_pred['y'] is not np.nan else 0

            v1 = v[['label']].copy()
            v1['proba'] = y_pred
            v1['score_cut'] = pd.cut(v1['proba'], [i / 100 for i in range(100)])
            # s1 = pd.DataFrame(v1[v1['label'] == 0]['score_cut'].value_counts() / len(v1[v1['label'] == 0]['score_cut']))
            # s2 = pd.DataFrame(v1[v1['label'] == 1]['score_cut'].value_counts() / len(v1[v1['label'] == 1]['score_cut']))
            # s = pd.merge(s1, s2, left_index=True, right_index=True)
            s1 = v1[v1['label'] == 0]['score_cut'].value_counts(normalize=True)
            s2 = v1[v1['label'] == 1]['score_cut'].value_counts(normalize=True)
            s = pd.concat([s1, s2], axis=1)
            s.columns = ['score_cut_x','score_cut_y']
            res[k + '_KL_divergence'] = js_divergence(s['score_cut_x'], s['score_cut_y'])
            bad_rate_lst = show_stage1(v1)['bad rate']
            res[k + '_if_badrate_mono'] = non_decreasing(bad_rate_lst)

        else:
            if str(type(model)).find('DecisionTreeRegressor') != -1:
                df_y_pred = pd.DataFrame({'y': model.predict(v[ft_lst])[:, 1]})
            else:
                df_y_pred = pd.DataFrame({'y': model.predict_proba(v[ft_lst])[:, 1]})

            best_th = res['oot_th']
            df_y_pred.loc[df_y_pred['y'] >= best_th, 'y'] = 1
            df_y_pred.loc[df_y_pred['y'] < best_th, 'y'] = 0
            res[k + '_ratio'] = df_y_pred['y'].mean() if df_y_pred['y'] is not np.nan else 0

    return res


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).


def objective_fix_sample(data_all, trial, ex_lst, df, strategy_type, model_type='xgb', dep='label', max_depth=6):
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
        param = param_xgb(trial, max_depth)
        model = xgb.XGBClassifier(**param)
    elif model_type == 'dt':
        param = param_dt(trial, max_depth)
        model = tree.DecisionTreeRegressor(**param)
    elif model_type == 'lgb':
        param = param_lgb(trial, max_depth)
        model = lgb.LGBMClassifier(**param)
    elif model_type == 'rf':
        param = param_rf(trial, max_depth)
        model = RandomForestClassifier(**param)
    elif model_type == 'exrf':
        param = param_rf(trial, max_depth)
        model = ExtraTreesClassifier(**param)
    elif model_type == 'catboost':
        param = param_catboost(trial, max_depth)
        model = catboost.CatBoostClassifier(**param)

    res = pd.DataFrame()
    # res = res.append(model_fit(model, datasets, ft_lst, dep), ignore_index=True)
    # 改写append为concat
    add_df = pd.DataFrame(model_fit(model, datasets, ft_lst, dep, model_type), index=[0])
    res = pd.concat([res, add_df], axis=1)
    
    res['value'] = strategy(res, strategy_type)
    res['model_param'] = json.dumps(param)

    df.append(res)

    # print(model)
    # print(res)
    return res['value']


def run(objective_fix_sample, data_all, ex_lst, strategy, file_path, n_trials=100, model_type='xgb', dep='label', max_depth=6,if_return_df = False): # 防止过拟合可以再调小点
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    df = []
    study.optimize(lambda trial: objective_fix_sample(data_all, trial, ex_lst, df, strategy_type=strategy,
                                                    model_type=model_type, dep=dep, max_depth=max_depth),
                                                    n_trials=n_trials)
    df = pd.concat(df)
    if not if_return_df:
        df.to_excel(file_path)
    # study = optuna.create_study(
    #     pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    # )
    # study.optimize(objective_xgb, n_trials=100)
    print(study.best_trial)
    # plot_slice(study)
    # hist = study.trials_dataframe()
    # hist.head()
    if if_return_df:
        return study,df
    else:
        return study



