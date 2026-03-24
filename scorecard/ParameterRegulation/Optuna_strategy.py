
def strategy(res, type):
    #  +++++++++++++++++++++++++++++++++++++++++ KS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if type == 1:
        return res['oot_ks'].mean() - 0.2 * abs(res['train_ks'].mean() - res['oot_ks'].mean()) \
               - 0.2 * abs(res['valid_ks'].mean() - res['oot_ks'].mean())
    elif type == 2:
        return res['train_ks'].mean() - 0.2 * abs(res['train_ks'].mean() - res['oot_ks'].mean())
    elif type == 3:
        threshold = 0.03
        return res['oot_ks'].mean() if (abs(res['train_ks'].mean() - res['oot_ks'].mean()) < threshold) \
                                       and (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < threshold) \
                                       and (abs(res['oot_ks'].mean() - res['valid_ks'].mean()) < threshold) \
            else -(abs(res['train_ks'].mean() - res['oot_ks'].mean())
                   + abs(res['train_ks'].mean() - res['valid_ks'].mean()))
    elif type == 4:
        threshold = 0.02
        return res['oot_ks'].mean() if (abs(res['train_ks'].mean() - res['oot_ks'].mean()) < threshold) \
                                       and (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < threshold) \
                                       and (abs(res['oot_ks'].mean() - res['valid_ks'].mean()) < threshold) \
            else -(abs(res['train_ks'].mean() - res['oot_ks'].mean())
                   + abs(res['train_ks'].mean() - res['valid_ks'].mean()))
    elif type == 5:
        threshold = 0.04
        return res['oot_ks'].mean() if (abs(res['train_ks'].mean() - res['oot_ks'].mean()) < threshold) \
                                       and (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < threshold) \
                                       and (abs(res['oot_ks'].mean() - res['valid_ks'].mean()) < threshold) \
            else -(abs(res['train_ks'].mean() - res['oot_ks'].mean())
                   + abs(res['train_ks'].mean() - res['valid_ks'].mean()))
    elif type == 6:
        return res['oot_ks'].mean() if (abs(res['train_ks'].mean() - res['oot_ks'].mean()) < 0.3) \
                                       and (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < 0.1) \
            else -(abs(res['train_ks'].mean() - res['oot_ks'].mean())
                   + abs(res['train_ks'].mean() - res['valid_ks'].mean()))
    elif type == 7:
        return res['valid_ks'].mean() if (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < 0.03
                                          and res['oot1_ks'].mean() > 0.28 and res['oot_ks'].mean() > 0.29) \
            else -abs(res['train_ks'].mean() - res['valid_ks'].mean())
    elif type == 8:
        return res['valid_ks'].mean() if (abs(res['train_auc'].mean() - res['valid_auc'].mean()) < 0.03) and \
                                         (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < 0.03) \
            else -abs(res['train_ks'].mean() - res['valid_ks'].mean())
    elif type == 9:
        return res['oot_ks'].mean() if (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < 0.2) and \
                                         (abs(res['train_ks'].mean() - res['oot_ks'].mean()) < 0.2) \
            else -(abs(res['train_ks'].mean() - res['oot_ks'].mean())
                   + abs(res['train_ks'].mean() - res['valid_ks'].mean()))
    elif type == 10:
        return (res['valid_ks'].mean() + res['oot_ks'].mean()) / 2 if (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < 0.1) and \
                                         (abs(res['train_ks'].mean() - res['oot_ks'].mean()) < 0.1) \
            else -(abs(res['train_ks'].mean() - res['oot_ks'].mean())
                   + abs(res['train_ks'].mean() - res['valid_ks'].mean()))       
    elif type == 1348: 
        return (res['valid_ks'].mean() + res['oot_ks'].mean()) / 2 if (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < 0.05) and \
                                            (abs(res['train_ks'].mean() - res['oot_ks'].mean()) < 0.05) \
            else -(abs(res['train_ks'].mean() - res['oot_ks'].mean())
                    + abs(res['train_ks'].mean() - res['valid_ks'].mean()))
    elif type == 1349: 
        return (res['valid_ks'].mean() + res['oot_ks'].mean()) / 2 if (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < 0.03) and \
                                            (abs(res['train_ks'].mean() - res['oot_ks'].mean()) < 0.03) \
            else -(abs(res['train_ks'].mean() - res['oot_ks'].mean())
                    + abs(res['train_ks'].mean() - res['valid_ks'].mean()))
    elif type == 1350: 
        return (res['valid_ks'].mean() + res['oot_ks'].mean()) / 2 if (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < 0.02) and \
                                            (abs(res['train_ks'].mean() - res['oot_ks'].mean()) < 0.02) \
            else -(abs(res['train_ks'].mean() - res['oot_ks'].mean())
                    + abs(res['train_ks'].mean() - res['valid_ks'].mean()))
    #  +++++++++++++++++++++++++++++++++++++++++ lift +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elif type == 201:
        return (res['valid_ks'].mean() + res['oot_ks'].mean())*0.9 + res['oot_top_5_lift']*0.1 if (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < 0.1) and \
                                         (abs(res['train_ks'].mean() - res['oot_ks'].mean()) < 0.1) \
            else -(abs(res['train_ks'].mean() - res['oot_ks'].mean())
                   + abs(res['train_ks'].mean() - res['valid_ks'].mean()))
    #  无oot
    elif type == 11:
        return res['valid_ks'].mean() if abs(res['train_ks'].mean() - res['valid_ks'].mean()) < 0.03 \
            else -abs(res['train_ks'].mean() - res['valid_ks'].mean())
    
    elif type == 12:
        return res['valid_ks'].mean() if abs(res['train_ks'].mean() - res['valid_ks'].mean()) < 0.05 \
            else -abs(res['train_ks'].mean() - res['valid_ks'].mean())

    #  +++++++++++++++++++++++++++++++++++++++++ AUC +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elif type == 21:
        return res['valid_auc'].mean() if (abs(res['train_auc'].mean() - res['valid_auc'].mean()) < 0.02
                                           and res['train_f2_score'].mean() > 0.19 and res
                                               ['valid_f2_score'].mean() > 0.19) \
            else -abs(res['train_auc'].mean() - res['valid_auc'].mean())
    elif type == 20:
        return res['valid_auc'].mean() if (abs(res['train_auc'].mean() - res['valid_auc'].mean()) < 0.02) \
            else -abs(res['train_auc'].mean() - res['valid_auc'].mean())
    elif type == 23:
        return res['train_auc'].mean() if (abs(res['train_auc'].mean() - res['valid_auc'].mean()) < 0.01) \
            else -abs(res['train_auc'].mean() - res['valid_auc'].mean())
    elif type == 22:
        return res['train_auc'].mean() if (abs(res['train_auc'].mean() - res['valid_auc'].mean()) < 0.03) \
            else -abs(res['train_auc'].mean() - res['valid_auc'].mean())
    elif type == 24:
        return res['valid_auc'].mean() if (abs(res['train_auc'].mean() - res['valid_auc'].mean()) < 0.03) and \
                                          (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < 0.03) \
            else -abs(res['train_auc'].mean() - res['valid_auc'].mean())
    elif type == 25:
        threshold = 0.03
        return res['oot_auc'].mean() if (abs(res['train_auc'].mean() - res['valid_auc'].mean()) < threshold) and \
                                          (abs(res['oot_ks'].mean() - res['train_ks'].mean()) < threshold) \
            else -abs(res['train_auc'].mean() - res['oot_auc'].mean())
    elif type == 26:
        threshold = 0.03
        return res['oot_auc'].mean() if (abs(res['train_auc'].mean() - res['oot_auc'].mean()) < threshold) \
                                       and (abs(res['train_auc'].mean() - res['valid_auc'].mean()) < threshold) \
                                       and (abs(res['oot_auc'].mean() - res['valid_auc'].mean()) < threshold) \
            else -(abs(res['train_auc'].mean() - res['oot_auc'].mean())
                   + abs(res['train_auc'].mean() - res['valid_auc'].mean()))
    elif type == 27:
        threshold = 0.03
        return res['oot_auc'].mean() if (abs(res['train_auc'].mean() - res['oot_auc'].mean()) < threshold) and \
                                          (abs(res['oot_ks'].mean() - res['train_ks'].mean()) < threshold) \
            else -abs(res['train_auc'].mean() - res['oot_auc'].mean())
    #  +++++++++++++++++++++++++++++++++++++++++ F2 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elif type == 31:
        return res['valid_f2_score'].mean() if (abs(res['train_auc'].mean() - res['valid_auc'].mean()) < 0.02
                                                and abs
                    (res['valid_f2_score'].mean() - res['train_f2_score'].mean()) < 0.1) \
            else -abs(res['valid_f2_score'].mean() - res['train_f2_score'].mean())
    elif type == 32:
        return res['valid_auc'].mean() if (abs(res['train_auc'].mean() - res['valid_auc'].mean()) < 0.01) \
            else -abs(res['train_auc'].mean() - res['valid_auc'].mean())
    elif type == 33:
        return 0.4 * res['valid_auc'].mean() + 0.6 * res['valid_f2'].mean() if (
                abs(res['train_f2'].mean() - res['valid_f2'].mean()) < 0.1) \
            else -abs(res['train_f2'].mean() - res['valid_f2'].mean())
    elif type == 34:
        return 0.5 * res['train_f2'].mean() + 0.5 * res['valid_f2'].mean() if (
                abs(res['train_f2'].mean() - res['valid_f2'].mean()) < 0.1) \
            else -abs(res['train_f2'].mean() - res['valid_f2'].mean())
    elif type == 35:
        return res['oot_f2'].mean() if (abs(res['oot_th'].mean() - res['valid_th'].mean()) < 0.01) and \
                                       (abs(res['train_th'].mean() - res['oot_th'].mean()) < 0.01) and \
                                       (abs(res['train_th'].mean() - res['valid_th'].mean()) < 0.01) \
            else -res['oot_f2'].mean()
    elif type == 36:
        return res['oot_f2'].mean() if (abs(res['oot_th'].mean() - res['valid_th'].mean()) < 0.01) and \
                                       (abs(res['train_th'].mean() - res['oot_th'].mean()) < 0.01) and \
                                       (abs(res['train_th'].mean() - res['valid_th'].mean()) < 0.01) and \
                                       (abs(res['oot_th'].mean() - 0.1) < 0.03) \
            else -res['oot_f2'].mean()
    elif type == 37:
        return res['train_f2'].mean() if (abs(res['oot_th'].mean() - res['valid_th'].mean()) < 0.01) and \
                                         (abs(res['train_th'].mean() - res['oot_th'].mean()) < 0.01) and \
                                         (abs(res['train_th'].mean() - res['valid_th'].mean()) < 0.01) and \
                                         (abs(res['oot_th'].mean() - 0.1) < 0.02) \
            else -res['train_f2'].mean()
    elif type == 38:
        return res['oot_f2'].mean() if (abs(res['oot_th'].mean() - res['valid_th'].mean()) < 0.01) and \
                                       (abs(res['train_th'].mean() - res['oot_th'].mean()) < 0.01) and \
                                       (abs(res['train_th'].mean() - res['valid_th'].mean()) < 0.01) and \
                                       (abs(res['oot_th'].mean() - 0.1) < 0.02) \
            else -res['oot_f2'].mean()
    elif type == 39:
        return res['oot_f2'].mean() if (abs(res['oot_th'].mean() - res['valid_th'].mean()) < 0.01) and \
                                       (abs(res['train_th'].mean() - res['oot_th'].mean()) < 0.01) and \
                                       (abs(res['train_th'].mean() - res['valid_th'].mean()) < 0.01) and \
                                       (res['oot_th'].mean() - 0.1 > - 0.02) \
            else -res['oot_f2'].mean()
    elif type == 40:
        return res['oot_f2'].mean() if (abs(res['oot_th'].mean() - res['valid_th'].mean()) < 0.01) and \
                                       (abs(res['train_th'].mean() - res['oot_th'].mean()) < 0.01) and \
                                       (abs(res['train_th'].mean() - res['valid_th'].mean()) < 0.01) and \
                                       (abs(res['oot_th'].mean() - 0.1) < 0.02) and (res['oot_precision'].mean() > 0.23) \
            else -res['oot_f2'].mean()
    elif type == 41:
        return res['oot_f2'].mean() if (abs(res['oot_th'].mean() - res['valid_th'].mean()) < 0.01) and \
                                       (abs(res['train_th'].mean() - res['oot_th'].mean()) < 0.01) and \
                                       (abs(res['train_th'].mean() - res['valid_th'].mean()) < 0.01) and \
                                       (abs(res['oot_ratio'].mean() - res['test_ratio'].mean()) < 0.004) \
            else -res['oot_f2'].mean()
    elif type == 42:
        return res['oot_f2'].mean() if (abs(res['oot_th'].mean() - res['valid_th'].mean()) < 0.03) and \
                                       (abs(res['train_th'].mean() - res['oot_th'].mean()) < 0.03) and \
                                       (abs(res['train_th'].mean() - res['valid_th'].mean()) < 0.03) \
            else -res['oot_f2'].mean()
    elif type == 43:
        return res['oot_f2'].mean() if (abs(res['oot_th'].mean() - res['valid_th'].mean()) < 0.05) and \
                                       (abs(res['train_th'].mean() - res['oot_th'].mean()) < 0.05) and \
                                       (abs(res['train_th'].mean() - res['valid_th'].mean()) < 0.05) \
            else -res['oot_f2'].mean()

    #  +++++++++++++++++++++++++++++++++++++++++ KL div ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elif type == 51:
        threshold = 0.03
        return res['KL_divergence'].mean() if (abs(res['train_ks'].mean() - res['oot_ks'].mean()) < threshold) \
                                       and (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < threshold) \
                                       and (abs(res['oot_ks'].mean() - res['valid_ks'].mean()) < threshold) \
            else -(abs(res['train_ks'].mean() - res['oot_ks'].mean())
                   + abs(res['train_ks'].mean() - res['valid_ks'].mean()))
    elif type == 52:
        return res['valid_ks'].mean() if ((res['oot_ks'].mean() > 0.2) and
                                        (res['train_ks'].mean() > 0.2)
                                        ) else \
            -(abs(res['train_ks'].mean() - res['oot_ks'].mean()) + abs(res['train_ks'].mean() - res['valid_ks'].mean()))

    elif type == 101:

        threshold = 0.03
        return res['oot_ks'].mean() if (abs(res['train_ks'].mean() - res['oot_ks'].mean()) < threshold) \
                                       and (abs(res['train_ks'].mean() - res['valid_ks'].mean()) < threshold) \
                                       and (abs(res['oot_ks'].mean() - res['valid_ks'].mean()) < threshold) \
                                       and (res['oot_if_badrate_mono'][0])\
            else -(abs(res['train_ks'].mean() - res['oot_ks'].mean())
                   + abs(res['train_ks'].mean() - res['valid_ks'].mean()))
    






def param_xgb(trial, max_depth):
    param = {
        "verbosity": 0,
        # "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "booster": "gbtree",
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 500.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 30, 100, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, max_depth),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "col_sample_bytree": trial.suggest_float("col_sample_bytree", 0.1, 1, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 50, log=True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["subsample"] = trial.suggest_float("subsample", 0.5, 1.0, log=True)
        # param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        # param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    return param


def param_lr(trial):
    param = {
        "penalty": 'l2',  # 'l1', 'elasticnet', 'None'
        "C": trial.suggest_float("C", 0.01, 1, log=True),
        "fit_intercept": True,
        "class_weight": "balanced",
        "max_iter": trial.suggest_int("max_iter", 100, 5000, log=True),
        "solver": 'liblinear',
        # "multi_class": 'auto',
    }

    # if param["multi_class"] == "multinomial":
    #     param['solver'] = trial.suggest_categorical('solver', ['newton - cg', 'sag', 'saga'])
    #
    # elif param["multi_class"] == 'ovr' or param["multi_class"] == 'auto':
    #     param['solver'] = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])

    return param


def param_lgb(trial, max_depth):
    param = {
        "boosting_type": "gbdt",  # "dart", "goss", "rf"
        "objective": "binary",  # "multiclass"
        "random_state": 2022,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 10, 500, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, max_depth),
        "num_leaves": trial.suggest_int("num_leaves", 2** 3, 2 ** 6),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 10, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 50, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 500.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 500.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 1e-8, 1.0, log=True),
        "max_bin": trial.suggest_int("max_bin", 127, 511, log=True),
        "subsample_for_bin": trial.suggest_int("subsample_for_bin", 100000, 300000, log=True),
        "subsample": trial.suggest_float("subsample", 0.3, 1, log=True),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10000, log=True)
    }

    return param


def param_dt(trial, max_depth):
    param = {
        "min_samples_leaf": 150,
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 300, log=True),
        'criterion': "squared_error",
        'splitter': "best",
        'max_depth': trial.suggest_int("max_depth", 1, max_depth, log=True),
        'min_weight_fraction_leaf': 0.0,
        'max_features': None,  # int, float或{"auto"， "sqrt"， "log2"}，默认=None
        # 'random_state': trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        'max_leaf_nodes': trial.suggest_int("max_leaf_nodes", 20, 100, log=True),
        # 'min_impurity_decrease': trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        # 'ccp_alpha': trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
    }

    return param

def param_rf(trial, max_depth):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, max_depth),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 300, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 300, log=True),
        # "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "max_features": trial.suggest_float("max_features", 0.01, 1.0, log=True),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0, log=True),
        "bootstrap": True,
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "n_jobs": -1,
    }

    return param


def param_catboost(trial, max_depth):
    param = {
        "loss_function": "Logloss",
        "task_type": "CPU",
        "random_seed": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations": trial.suggest_int("iterations", 100, 1000, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "depth": trial.suggest_int("depth", 1, max_depth),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 300, log=True),
        "border_count": trial.suggest_int("border_count", 100, 254, log=True),
    }

    return param
