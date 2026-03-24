import pandas as pd
import numpy as np
import math
import toad


def KS(df, score, target):
    '''
    :param df: the dataset containing probability and bad indicator
    :param score:'predictions'
    :param target:'target'
    :return: KS值
    '''

    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total': total, 'bad': bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all.index = range(len(all))
    all = all.sort_values(by=score, ascending=False)
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    all['totalPcnt'] = all['total'] / all['total'].sum()
    ks = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return max(abs(ks))


def IV(df, score, target):
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total': total, 'bad': bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all.index = range(len(all))
    all = all.sort_values(by=score, ascending=False)

    all['badCumRate'] = all['bad'] / all['bad'].sum()
    all['goodCumRate'] = all['good'] / all['good'].sum()
    all['badCumRate'] = all['badCumRate'].apply(lambda x: 0.0001 if x == 0 else x)
    all['goodCumRate'] = all['goodCumRate'].apply(lambda x: 0.0001 if x == 0 else x)

    all['iv'] = (all['badCumRate'] - all['goodCumRate']) * np.log(all['badCumRate'] / all['goodCumRate'])
    all['totalPcnt'] = all['total'] / all['total'].sum()

    return all['iv'].sum()


def value(df, ex_lst, dep, target='train',if_train_oot=False):
    import toad
    combiner = toad.transform.Combiner()
    combiner.fit(df[df['target'] == target], df[df['target'] == target][dep], method='dt', min_samples=0.05, n_bins=6,
                exclude=ex_lst)
    transfer = toad.transform.WOETransformer()
    dev_bin = combiner.transform(df[df['target'] == target])
    transfer.fit(dev_bin, dev_bin[dep], exclude=ex_lst)

    data_bins = combiner.transform(df)
    # data_woe = transfer.transform(data_bins)

    ft_lst = [i for i in df.columns if i not in ex_lst]
    res = pd.DataFrame(columns=['var_names', 'ks'])

    train = data_bins[data_bins['target'] == 'train']
    valid = data_bins[data_bins['target'] == 'valid']
    oot = data_bins[data_bins['target'] == 'oot']
    tv = data_bins[data_bins['target'].isin(['train', 'valid'])]

    for i in ft_lst:
        add_df = pd.DataFrame({'var_names': i,
                                'ks': KS(data_bins, i, dep),
                                'iv_bin_train': IV(train, i, dep),
                                'iv_bin_valid': IV(valid, i, dep),
                                'iv_bin_oot': IV(oot, i, dep),
                                'iv_bin_total': IV(data_bins, i, dep)},index=[0])
        res = pd.concat([res, add_df], axis=0, ignore_index=True)

        # res = res.append({'var_names': i,
        #                   'ks': KS(data_bins, i, dep),
        #                   'iv_bin_train': IV(train, i, dep),
        #                   'iv_bin_valid': IV(valid, i, dep),
        #                   'iv_bin_oot': IV(oot, i, dep),
        #                   'iv_bin_total': IV(data_bins, i, dep)},
        #                   ignore_index=True)

    psi_df_tv = toad.metrics.PSI(train[ft_lst], valid[ft_lst])
    psi_df_to = toad.metrics.PSI(train[ft_lst], oot[ft_lst])
    psi_df_vo = toad.metrics.PSI(valid[ft_lst], oot[ft_lst])
    psi_df_tvo = toad.metrics.PSI(tv[ft_lst], oot[ft_lst])
    psi_df_tv = psi_df_tv.reset_index().rename(columns={'index': 'var_names', 0: 'psi_tv_bin'})
    psi_df_to = psi_df_to.reset_index().rename(columns={'index': 'var_names', 0: 'psi_to_bin'})
    psi_df_vo = psi_df_vo.reset_index().rename(columns={'index': 'var_names', 0: 'psi_vo_bin'})
    psi_df_tvo = psi_df_tvo.reset_index().rename(columns={'index': 'var_names', 0: 'psi_tvo_bin'})

    res = pd.merge(res, psi_df_tv, on='var_names', )
    res = pd.merge(res, psi_df_to, on='var_names', )
    res = pd.merge(res, psi_df_vo, on='var_names', )
    res = pd.merge(res, psi_df_tvo, on='var_names', )
    res = res.sort_values(by='ks', ascending=False)
    if if_train_oot:
        return res, combiner, transfer, train, oot
    else:
        return res, combiner, transfer


def KS_weighted(model, X, Y, Weight):
    Y_predict = model.predict(X)
    nrows = X.shape[0]
    lis = [(Y_predict,  Y.values[i], Weight[i]) for i in range(nrows)] # 还原权重
    ks_lis = sorted(lis, key=lambda x: x[0], reverse=True)
    KS = list()
    bad = sum([w for (p, y, w) in ks_lis if y > 0.5])
    good = sum([w for (p, y, w) in ks_lis if y <= 0.5])
    bad_cnt, good_cnt = 0, 0
    for (p, y, w) in ks_lis:
        if y > 0.5:
            bad_cnt += w  # 1*w 即加权样本个数
        else:
            good_cnt += w  # 1*w 即加权样本个数
        ks = math.fabs((bad_cnt/bad)-(good_cnt/good))
        KS.append(ks)
    return max(KS)


def KS_bucket(prob_off, offy):
    return toad.metrics.KS_bucket(prob_off, offy, bucket=20, method='quantile')
