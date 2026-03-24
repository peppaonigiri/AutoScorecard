

# 定义逐步回归函数
import statsmodels.api as sm
import pandas as pd
import toad


def stepwise_selection(X, y, initial_list=[], threshold_in=0.05, threshold_out=0.05, verbose=True):  # 0.05
    """
    Perform a forward-backward feature selection based on p-value from statsmodels.api.OLS
    Arguments:
         X - pandas.DataFrame with candidate features
         y - list-like with the target
         initial_list - list of features to start with (column names of X)
         threshold_in - include a feature if its p-value < threshold_in
         threshold_out - exclude a feature if its p-value > threshold_out
         verbose - whether to print the sequence of inclusions and exclusions
    Returns:
        list of selected features
        Always set threshold_in < threshold_out to avoid infinite looping.
        See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
           # if verbose:
            #    print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            #if verbose:
             #   print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# 逐步回归特征筛选，支持向前，向后和双向（推荐）。
# estimator: 用于拟合的模型，支持'ols', 'lr', 'lasso', 'ridge'
# direction: 逐步回归的方向，支持'forward', 'backward', 'both' （推荐）
# criterion: 评判标准，支持 'aic', 'bic', 'ks', 'auc'
# max_iter: 最大循环次数
# return_drop: 是否返回被剔除的列名
# exclude: 不需要被训练的列名，比如ID列和时间列
# tip: 经验证，direction = ‘both’效果最好。estimator = ‘ols’以及criterion = ‘aic’运行速度快且结果对逻辑回归建模有较好的代表性
def selection_by_stepwise(dev_woe_psi, val_woe_psi, off_woe_psi, ex_lis, dep):
    dev_woe_psi_stp = toad.selection.stepwise(dev_woe_psi,
                                              dev_woe_psi[dep],
                                              exclude=ex_lis,
                                              direction='both',
                                              criterion='aic',
                                              estimator='ols',
                                              intercept=False)
    val_woe_psi_stp = val_woe_psi[dev_woe_psi_stp.columns]
    off_woe_psi_stp = off_woe_psi[dev_woe_psi_stp.columns]
    data = pd.concat([dev_woe_psi_stp, val_woe_psi_stp, off_woe_psi_stp])
    print(data.shape)
    return data