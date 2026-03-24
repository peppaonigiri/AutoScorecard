from FeatureEngineering.KeyValue import Missing
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve


class Analysis(object):
    def __int__(self, data, ex_lst, dep='label'):
        self.data = data
        self.ex_lst = ex_lst
        self.ft_lst = [i for i in data.columns if i not in ex_lst]
        self.dep = dep

    def cover_rate(self):
        missing = Missing.value(self.data)
        missing['cover_rate'] = 1 - missing['missing_rate']
        return missing

    def quantile(self):
        res = pd.DataFrame(self.data.apply(lambda x: x.max()))
        res['var_names'] = res.index
        res = res.rename(columns={0: 'max'})
        res['min'] = self.data.apply(lambda x: x.min())
        quantile = self.data.quantile([.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95], numeric_only=True)
        for i in [.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95]:
            res[str(i*100) + '%分位数'] = quantile.loc[i, :]
        return res

    def merge(self, merge_lst):
        res = merge_lst[0][['var_names']].copy()
        for i in merge_lst:
            res = pd.merge(res, merge_lst[i], on='var_names', how='outer')
        return res

    def value_counts(self):
        self.value_counts = {}
        for i in self.ft_lst:
            self.value_counts[i] = pd.DataFrame(self.data[i].value_counts())

    def single_var(self, ft):
        values = list(set(self.data[ft]))
        dic = {}
        for i in values:
            a = self.data[self.data[ft] == i]
            good = a[a[self.dep] == 0].count()
            bad = a[a[self.dep] == 1].count()
            dic[i] = bad / bad + good
        return dic


n_sample = 1000

df_score = pd.DataFrame({
    'user_id': [u for u in range(n_sample)],
    'label': np.random.randint(2, size=n_sample),
    'score': 900*np.random.random(size=n_sample),
    'term': 20201+np.random.randint(5, size=n_sample)
})


df_score.groupby('term').agg(total=('label', 'count'),
                             bad=('label', 'sum'),
                             bad_rate=('label', 'mean'))


def get_auc(ytrue, yprob):
    auc = roc_auc_score(ytrue, yprob)
    if auc < 0.5:
        auc = 1 - auc
    return auc


def get_ks(ytrue, yprob):
    fpr, tpr, thr = roc_curve(ytrue, yprob)
    ks = max(abs(tpr - fpr))
    return ks


def get_gini(ytrue, yprob):
    auc = get_auc(ytrue, yprob)
    gini = 2 * auc - 1
    return gini


df_metrics = pd.DataFrame({
    'auc': df_score.groupby('term').apply(lambda x: get_auc(x['label'], x['score'])),
    'ks': df_score.groupby('term').apply(lambda x: get_ks(x['label'], x['score'])),
    'gini': df_score.groupby('term').apply(lambda x: get_gini(x['label'], x['score']))
})


#PSI

df_score['score_bin'] = pd.cut(df_score['score'], [0, 500, 700, 800, 900])

df_total = pd.pivot_table(df_score,
                          values='user_id',
                          index='score_bin',
                          columns=['term'],
                          aggfunc="count",
                          margins=True)
df_ratio = df_total.div(df_total.iloc[-1, :], axis=1)

eps = np.finfo(np.float32).eps
lst_psi = list()
for idx in range(1, len(df_ratio.columns)-1):
    last, cur = df_ratio.iloc[0, -1: idx-1]+eps, df_ratio.iloc[0, -1: idx]+eps
    psi = sum((cur-last) * np.log(cur / last))
    lst_psi.append(psi)
df_ratio.append(pd.Series([np.nan]+lst_psi+[np.nan],
                          index=df_ratio.columns,
                          name='psi'))


#总人数比例和坏客户比例

df_total = pd.pivot_table(df_score,
                          values='user_id',
                          index='score_bin',
                          columns=['term'],
                          aggfunc="count",
                          margins=True)
df_ratio = df_total.div(df_total.iloc[-1, :], axis=1)

df_bad = pd.pivot_table(df_score[df_score['label'] == 1],
                        values='user_id',
                        index='score_bin',
                        columns=['term'],
                        aggfunc="count",
                        margins=True)
df_bad_rate = df_bad/df_total

#做图

colormap = sns.diverging_palette(130, 20, as_cmap=True)
df_ratio.drop('All').T.plot(kind='bar', stacked=True, colormap=colormap)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

colormap = sns.diverging_palette(130, 20, as_cmap=True)
df_bad_rate.drop('All').T.plot(kind='line', colormap=colormap)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

