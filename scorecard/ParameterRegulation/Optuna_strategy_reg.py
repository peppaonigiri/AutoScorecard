

def strategy(res, type):
    if type == 1:
        return - res['oot_mean_squared_error'].mean()
    elif type == 2:
        return - res['oot_mean_squared_error'].mean() \
            if (abs(res['train_mean_squared_error'].mean() - res['oot_mean_squared_error'].mean()) < 0.005) \
               and (abs(res['train_mean_squared_error'].mean() - res['valid_mean_squared_error'].mean()) < 0.005) \
            else -(abs(res['train_mean_squared_error'].mean() - res['oot_mean_squared_error'].mean())
                   + abs(res['train_mean_squared_error'].mean() - res['valid_mean_squared_error'].mean()))
    elif type == 3:
        return - res['oot_mean_absolute_error'].mean()
    elif type == 4:
        return - res['oot_mean_absolute_error'].mean() \
            if (abs(res['train_mean_absolute_error'].mean() - res['oot_mean_absolute_error'].mean()) < 0.02) \
               and (abs(res['train_mean_absolute_error'].mean() - res['valid_mean_absolute_error'].mean()) < 0.02) \
            else -(abs(res['train_mean_absolute_error'].mean() - res['oot_mean_absolute_error'].mean())
                   + abs(res['train_mean_absolute_error'].mean() - res['valid_mean_absolute_error'].mean()))
    elif type == 5:
        return - res['oot_mean_absolute_error'].mean() \
            if (abs(res['train_mean_absolute_error'].mean() - res['oot_mean_absolute_error'].mean()) < 0.01) \
               and (abs(res['train_mean_absolute_error'].mean() - res['valid_mean_absolute_error'].mean()) < 0.01) \
            else -(abs(res['train_mean_absolute_error'].mean() - res['oot_mean_absolute_error'].mean())
                   + abs(res['train_mean_absolute_error'].mean() - res['valid_mean_absolute_error'].mean()))
    elif type == 6:
        return - res['oot_mean_absolute_error'].mean() \
            if (abs(res['train_mean_absolute_error'].mean() - res['oot_mean_absolute_error'].mean()) < 0.04) \
               and (abs(res['train_mean_absolute_error'].mean() - res['valid_mean_absolute_error'].mean()) < 0.04) \
            else -(abs(res['train_mean_absolute_error'].mean() - res['oot_mean_absolute_error'].mean())
                   + abs(res['train_mean_absolute_error'].mean() - res['valid_mean_absolute_error'].mean()))
    elif type == 7:
        return - res['oot_mean_absolute_error'].mean() \
            if (abs(res['train_mean_absolute_error'].mean() - res['oot_mean_absolute_error'].mean()) < 0.03) \
               and (abs(res['train_mean_absolute_error'].mean() - res['valid_mean_absolute_error'].mean()) < 0.03) \
            else -(abs(res['train_mean_absolute_error'].mean() - res['oot_mean_absolute_error'].mean())
                   + abs(res['train_mean_absolute_error'].mean() - res['valid_mean_absolute_error'].mean()))


def param_xgb(trial):
    param = {
        "verbosity": 0,
        # "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "booster": "gbtree",
        "random_state": 2021,
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 500.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 30, 100, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
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
        'fit_intercept': True,
        'normalize': False,
        'copy_X': True,
        'n_job': None,
        'positive': False,
    }
    # if param["multi_class"] == "multinomial":
    #     param['solver'] = trial.suggest_categorical('solver', ['newton - cg', 'sag', 'saga'])
    #
    # elif param["multi_class"] == 'ovr' or param["multi_class"] == 'auto':
    #     param['solver'] = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])

    return param


def param_lgb(trial):
    param = {
        "boosting_type": "gbdt",  # "dart", "goss", "rf"
        "objective": "regression",  # "multiclass"
        "random_state": 2021,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 30, 100, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "num_leaves": trial.suggest_int("num_leaves", 2**3, 2**5),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 10, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 50, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 500.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 500.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 1e-8, 1.0, log=True),
        "max_bin": trial.suggest_int("max_bin", 127, 511, log=True),
        "subsample_for_bin": trial.suggest_int("subsample_for_bin", 100000, 300000, log=True),
        "subsample": trial.suggest_float("subsample", 0.3, 1, log=True),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 10000, log=True)
    }

    return param