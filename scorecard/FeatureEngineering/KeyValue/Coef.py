

import pandas as pd
from sklearn.linear_model import LogisticRegression


def lr_model(x, y, C=0.1):
    model = LogisticRegression(C=C, class_weight='balanced',
                               max_iter=2000,
                               penalty='l2',
                               solver='liblinear',
                               verbose=0,
                               n_jobs=1)
    model.fit(x, y)
    return model


def value(data, dep, ex_lst):
    ft_lst = [i for i in data.columns if i not in ex_lst]
    x, y = data[data['target'] == 'train'][ft_lst], data[data['target'] == 'train'][dep]
    model = lr_model(x, y)
    res = pd.DataFrame(columns=['var_names', 'coef'])
    res['var_names'] = ft_lst
    res['coef'] = model.coef_[0]
    while min(res['coef']) < 0 or max(res['coef']) > 1:
        drop_lst = list(res[res['coef'] < 0]['var_names']) + list(res[res['coef'] >= 1]['var_names'])
        ft_lst = [i for i in ft_lst if i not in drop_lst]
        x, y = data[data['target'] == 'train'][ft_lst], data[data['target'] == 'train'][dep]
        model = lr_model(x, y)
        print(model.coef_[0])
        res = pd.DataFrame(columns=['var_names', 'coef'])
        res['var_names'] = ft_lst
        res['coef'] = model.coef_[0]
    return ft_lst