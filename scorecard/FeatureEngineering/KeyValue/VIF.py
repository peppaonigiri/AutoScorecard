

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


def calc(df):
    col = list(range(df.shape[1]))
    vif = [variance_inflation_factor(df.iloc[:, col].values, ix) for ix in range(df.iloc[:, col].shape[1])]
    return pd.DataFrame({'var_names': list(df.columns), 'vif': vif}).sort_values(by='vif', ascending=False)


def value(df, ex_lst):
    lst = [i for i in list(df.columns) if i not in ex_lst]
    df = df[lst]
    res = pd.DataFrame(columns={'var_names', 'vif'})

    while not lst:
        vif = calc(df[lst])
        res = res.append({'var_names': vif['var_names'][0], 'vif': vif['vif'][0]}, ignore_index=True)
        lst.remove(df)
    return res


def vif(df, threshold=10):  # 10, 5, 3
    col = list(range(df.shape[1]))
    res = pd.DataFrame(columns=['var_names', 'vif'])
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(df.iloc[:, col].values, ix) for ix in range(df.iloc[:, col].shape[1])]
        maxvif = max(vif)
        maxix = vif.index(maxvif)
        res = res.append({'var_names': df.columns[col[maxix]], 'vif': maxvif}, ignore_index=True)
        if maxvif > threshold:
            del col[maxix]
            dropped = True
    return list(df.columns[col]), res
