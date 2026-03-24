

from toad.metrics import KS, F1, PSI
import toad
import pandas as pd
import numpy as np
# from pyecharts.charts import *
# from pyecharts import options as opts
# from pylab import *
# import seaborn as sns
import math
from sklearn.metrics import roc_curve, auc, roc_auc_score
from FeatureEngineering.KeyValue import IV
from FeatureEngineering.KeyValue import Importance
from FeatureEngineering.KeyValue import PSI
from FeatureEngineering.KeyValue import Chi2
from FeatureEngineering.KeyValue import Missing
from DataPreprocessing.Datasets.Describe import badrate_by_month
from FeatureEngineering.KeyValue import KS


class Report(object):
    def __init__(self, df_init, ex_lst, dep, SAVEPATH):
        self.df = df_init.fillna(-999)
        # self.df = df_init.copy()
        self.df_init = df_init
        self.ex_lst = ex_lst
        self.ft_lst = [i for i in list(df_init.columns) if i not in ex_lst]
        self.dep = dep
        self.SAVEPATH = SAVEPATH

    def iv_ks_psi(self):
        print('正在计算IV......')
        self.ks, self.combiner, self.transfer = KS.value(self.df, self.ex_lst, self.dep)
        self.ks.to_excel(self.SAVEPATH + 'iv_ipt_psi_chi2_missing.xlsx')
        print('正在转存......')
        self.dev_bin = self.combiner.transform(self.df[self.df['target'] == 'train'])
        bins = self.bins_detail(self.dev_bin, self.combiner, self.transfer, self.ft_lst)

        return self.ks, bins

    def iv_ks_psi_chi2(self):
        print('正在计算IV......')
        self.ks, self.combiner, self.transfer = KS.value(self.df, self.ex_lst, self.dep)
        print('正在计算CHI2......')
        self.chi2 = Chi2.value(self.df, self.dep, self.ex_lst, 10)

        res = pd.merge(self.ks, self.chi2, on='var_names', how='outer')

        print('正在转存......')
        self.dev_bin = self.combiner.transform(self.df[self.df['target'] == 'train'])
        bins = self.bins_detail(self.dev_bin, self.combiner, self.transfer, self.ft_lst)
        res.to_excel(self.SAVEPATH+'iv_ipt_psi_chi2.xlsx')
        return res

    def iv_ks_psi_chi2_missing_quantile(self, target='train',if_train_oot=False):
        print('正在计算IV......')
        if if_train_oot:
            self.ks, self.combiner, self.transfer,train,oot = KS.value(self.df, self.ex_lst, self.dep,if_train_oot=if_train_oot)
        else:
            self.ks, self.combiner, self.transfer = KS.value(self.df, self.ex_lst, self.dep,if_train_oot=if_train_oot)
        print('正在计算CHI2......')
        self.chi2 = Chi2.value(self.df, self.dep, self.ex_lst, 10)
        print('正在计算COVER RATE......')
        self.missing = Missing.value(self.df_init[self.ft_lst])

        res = pd.merge(self.ks, self.chi2, on='var_names', how='outer')
        res = pd.merge(res, self.missing, on='var_names', how='outer')
        print('正在计算QUANTILE......')
        min_ = pd.DataFrame(self.df_init[self.ft_lst].min()).rename(columns={0: 'min'})
        quantile_ = self.df_init[res['var_names']].quantile([.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99],
                                                            numeric_only=True).T
        max_ = pd.DataFrame(self.df_init[self.ft_lst].max()).rename(columns={0: 'max'})
        qtl = pd.merge(min_, quantile_, left_index=True, right_index=True)
        qtl = pd.merge(qtl, max_, left_index=True, right_index=True)
        res = pd.merge(res, qtl, left_on='var_names', right_index=True)

        print('正在保存结果......')
        # 注意这里的分箱是target == 'train'
        self.dev_bin = self.combiner.transform(self.df[self.df['target'] == target])
        bins = self.bins_detail(self.dev_bin, self.combiner, self.transfer, self.ft_lst)
        res.to_excel(self.SAVEPATH+'iv_ipt_psi_chi2_missing.xlsx')
        print('已保存')
        if if_train_oot:
            return res, bins, train, oot
        else:
            return res, bins

    def iv_ks_psi_missing_quantile(self, target='train'):
        print('正在计算IV......')
        self.ks, self.combiner, self.transfer = KS.value(self.df, self.ex_lst, self.dep)
        print('正在计算COVER RATE......')
        self.missing = Missing.value(self.df_init[self.ft_lst])
        res = pd.merge(self.ks, self.missing, on='var_names', how='outer')
        print('正在计算QUANTILE......')
        min_ = pd.DataFrame(self.df_init[self.ft_lst].min()).rename(columns={0: 'min'})
        quantile_ = self.df_init[res['var_names']].quantile([.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99],
                                                            numeric_only=True).T
        max_ = pd.DataFrame(self.df_init[self.ft_lst].max()).rename(columns={0: 'max'})
        qtl = pd.merge(min_, quantile_, left_index=True, right_index=True)
        qtl = pd.merge(qtl, max_, left_index=True, right_index=True)
        res = pd.merge(res, qtl, left_on='var_names', right_index=True)

        print('正在保存结果......')
        self.dev_bin = self.combiner.transform(self.df[self.df['target'] == target])
        bins = self.bins_detail(self.dev_bin, self.combiner, self.transfer, self.ft_lst)
        res.to_excel(self.SAVEPATH + 'iv_ipt_psi_chi2_missing.xlsx')
        print('已保存')
        return res, bins

    def iv_ks_ipt_psi_chi2_missing_quantile(self, target='train'):
        print('正在计算IV......')
        self.iv = IV.value(self.df, self.ft_lst, self.dep)
        print('正在计算KS......')
        self.ks, self.combiner, self.transfer = KS.value(self.df, self.ex_lst, self.dep, target=target)
        print('正在计算IPT......')
        self.ipt = Importance.value(self.df, 'train', 'valid', dep=self.dep, exclude=self.ex_lst)
        print('正在计算PSI......')
        self.psi = PSI.value(self.df, self.ft_lst)
        print('正在计算CHI2......')
        self.chi2 = Chi2.value(self.df, self.dep, self.ex_lst, 10)
        print('正在计算COVER RATE......')
        self.missing = Missing.value(self.df_init[self.ft_lst])

        res = pd.merge(self.iv, self.ipt, on='var_names', how='outer')
        res = pd.merge(res, self.ks, on='var_names', how='outer')
        res = pd.merge(res, self.psi, on='var_names', how='outer')
        res = pd.merge(res, self.chi2, on='var_names', how='outer')
        res = pd.merge(res, self.missing, on='var_names', how='outer')

        print('正在计算QUANTILE......')
        min_ = pd.DataFrame(self.df_init[self.ft_lst].min()).rename(columns={0: 'min'})
        quantile_ = self.df_init[res['var_names']].quantile([.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99],
                                                            numeric_only=True).T
        max_ = pd.DataFrame(self.df_init[self.ft_lst].max()).rename(columns={0: 'max'})
        qtl = pd.merge(min_, quantile_, left_index=True, right_index=True)
        qtl = pd.merge(qtl, max_, left_index=True, right_index=True)
        res = pd.merge(res, qtl, left_on='var_names', right_index=True)

        print('正在转存......')
        self.dev_bin = self.combiner.transform(self.df[self.df['target'] == target])
        bins = self.bins_detail(self.dev_bin, self.combiner, self.transfer, self.ft_lst)
        res.to_excel(self.SAVEPATH+'iv_ipt_psi_chi2_missing.xlsx')
        return res, bins

    def distribution(self, var_time, analyse_list):
        """
        :param analyse_list: []
        :return:
        """
        for i in analyse_list:
            res = badrate_by_month(self.df, var_time, i, self.dep)
            res.to_excel(self.SAVEPATH+'distribution'+i+'.xlsx')

    def iv_bin_woe(self, combiner, t):
        res = pd.DataFrame(columns={'var_names', 'bin', 'woe'})
        self.ft_lst = [i for i in list(self.df.columns) if i not in self.ex_lst]
        for i in self.ft_lst:
            dics = {}
            for j in t[i]['value']:
                dics[j] = t[i]['woe'][j]
            res = res.append({'var_names': i, 'bin': combiner[i], 'woe': dics}, ignore_index=True)

        res = pd.merge(self.iv, res, on='var_names', how='outer')
        res.to_excel(self.SAVEPATH+'iv_bin_woe.xlsx')
        return res

    def score_card(self, combiner, t, lr_intercept, base_score=500, base_odds=0.25, pdo=50, rate=2):
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.rate = rate
        self.combiner = combiner
        self.t = t
        self.lr_intercept = lr_intercept
        from toad.scorecard import ScoreCard
        self.card = ScoreCard(combiner=combiner,
                         transer=t, C=0.1,
                         class_weight='balanced',
                         base_score=base_score,
                         base_odds=base_odds,
                         pdo=pdo,
                         rate=rate)
        self.card.fit(self.df[self.ft_lst], self.df[self.dep])
        self.card.export(to_frame=True).to_excel(self.SAVEPATH+'card.xlsx')
        iv = self.iv[['var_names', 'iv_train']]
        res = pd.DataFrame(columns={'var_names', 'bin', 'woe', 'scores', 'weight'})

        for i in self.ft_lst:
            res = res.append({'var_names': i, 'bin': combiner[i], 'woe': self.card[i]['woes'], 'weight': self.card[i]['weight'],
                              'scores': self.card[i]['scores']}, ignore_index=True)
            res = res.append({'var_names': "  "}, ignore_index=True)
        res = pd.merge(iv, res, on='var_names', how='outer')
        res.to_excel(self.SAVEPATH+'score_card.xlsx')
        self.A = self.base_score - self.pdo / np.log(self.rate) * np.log(self.base_odds)
        self.q = self.A
        self.B = self.pdo / np.log(self.rate)
        self.p = -self.B
        self.base_score_sc = - self.B * self.lr_intercept + self.A
        self.base_score_ft = self.base_score_sc / len(self.ft_lst)
        return self.card, res

    def proba_score(self, model, datasets=[]):
        self.model = model
        self.test_sets = datasets
        self.dev = self.df[self.df['target'] == 'train'].copy()
        self.val = self.df[self.df['target'] == 'valid'].copy()
        self.oot = self.df[self.df['target'] == 'oot'].copy()

        for i in [self.dev, self.val, self.oot]:
            i['proba'] = model.predict_proba(i[self.ft_lst])[:, 1]
            i['score'] = self.card.proba_to_score(i['proba'])

        self.train_valid = pd.concat([self.dev[['label', 'score']], self.val[['label', 'score']]])
        self.train_valid.to_excel(self.SAVEPATH + 'train_valid_label_score.xlsx')
        self.oot[['label', 'score']].to_excel(self.SAVEPATH + 'oot_label_score.xlsx')

        self.test_sets_woe = datasets.copy()
        for i in range(len(datasets)):
            datasets[i] = datasets[i].fillna(-999)
            self.test_sets_woe[i][self.ft_lst] = self.t.transform(self.combiner.transform(self.test_sets_woe[i][self.ft_lst]))
            self.test_sets_woe[i]['proba'] = model.predict_proba(self.test_sets_woe[i][self.ft_lst])[:, 1]
            self.test_sets_woe[i]['score'] = self.card.proba_to_score(self.test_sets_woe[i]['proba'])
            self.test_sets_woe[i][['label', 'score']].to_excel(self.SAVEPATH +
                                                               self.test_sets_woe[i]['target'][0]+'_label_score.xlsx')

        self.plot('score')


    def eva_model(self):
        # AUC, KS, F1, PSI
        self.dic = {}
        for i in [self.dev, self.val, self.oot] + self.test_sets_woe:
            i = i.reset_index(drop=True)
            fpr, tpr, _ = roc_curve(i['label'], i['proba'])
            ks = abs(fpr - tpr).max()
            auc_ = auc(fpr, tpr)
            label = i['target'][0]
            f1 = F1(i['proba'], i['label'])
            self.dic[label] = [fpr, tpr, ks, auc_, f1]
            print(label, ': ', ' ks : ', ks, ' auc : ', auc_, ' f1: ', f1)
        print('模型PSI:', PSI(self.dev['proba'], self.oot['proba']))

        from matplotlib import pyplot as plt
        for key, value in self.dic.items():
            plt.plot(value[0], value[1], label=key)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.show()

        return self.dic

    def bins(self):
        self.dev_bin = self.combiner.transform(self.dev, self.dev[self.dep], exclude=self.ex_lst)
        t = toad.transform.WOETransformer()
        self.dev_woe = t.fit_transform(self.dev_bin, self.dev_bin[self.dep], exclude=self.ex_lst)

        ft_lst = [i for i in self.df.columns if i not in self.ex_lst]
        self.bins = {}

        for i in ft_lst:
            total = self.dev_bin.groupby([i])[self.dep].count()
            bad = self.dev_bin.groupby([i])[self.dep].sum()
            res = pd.DataFrame({'total': total, 'bad': bad})
            res['good'] = res['total'] - res['bad']
            res['value'] = res.index

            bins = []
            sp_l = [float('-inf')] + list(self.combiner[i]) + [float('inf')]
            for j in range(len(sp_l) - 1):
                bins.append('[' + str(sp_l[j]) + ', ' + str(sp_l[j + 1]) + ')')
            res['bins'] = bins

            res['bins_prop'] = res['total'] / res['total'].sum()
            res['bad_cumrate'] = res['bad'].cumsum() / res['bad'].sum()
            res['good_cumrate'] = res['good'].cumsum() / res['good'].sum()
            res['ks'] = res['bad_cumrate'] - res['bad_cumrate']

            res['bad_prop'] = res['bad'] / res['bad'].sum()
            res['good_prop'] = res['good'] / res['good'].sum()
            res['bad_prop'] = res['bad_prop'].apply(lambda x: 0.0001 if x == 0 else x)
            res['good_prop'] = res['good_prop'].apply(lambda x: 0.0001 if x == 0 else x)

            res['iv'] = (res['bad_prop'] - res['good_prop']) * np.log(res['bad_prop'] / res['good_prop'])
            res['total_iv'] = res['iv'].sum()
            res['total_ks'] = res['ks'].max()
            res['woe'] = t[i]['woe']

            self.bins[i] = res
        self.bins = pd.concat(self.bins)
        self.bins.to_excel(self.SAVEPATH + 'bins.xlsx')

    def bins_detail(self, dev_bin, combiner, t, ft_lst):
        BINS = {}
        for i in ft_lst:
            count = dev_bin.groupby([i])['label'].count()
            bad = dev_bin.groupby([i])['label'].sum()
            res = pd.DataFrame({'count': count, 'bad': bad})
            res['good'] = res['count'] - res['bad']
            res['value'] = res.index
            res['variable'] = i

            bins = []
            sp_l = [float('-inf')] + list(combiner[i]) + [float('inf')]
            for j in range(len(sp_l) - 1):
                bins.append('[' + str(sp_l[j]) + ', ' + str(sp_l[j + 1]) + ')')

            res['bin'] = bins
            res['count_distr'] = res['count'] / res['count'].sum()
            res['bad_cumrate'] = res['bad'].cumsum() / res['bad'].sum()
            res['good_cumrate'] = res['good'].cumsum() / res['good'].sum()
            res['ks'] = abs(res['bad_cumrate'] - res['good_cumrate'])

            res['badprob'] = res['bad'] / res['bad'].sum()
            res['goodprob'] = res['good'] / res['good'].sum()
            res['badprob'] = res['badprob'].apply(lambda x: 0.0001 if x == 0 else x)
            res['goodprob'] = res['goodprob'].apply(lambda x: 0.0001 if x == 0 else x)

            res['iv'] = (res['badprob'] - res['goodprob']) * np.log(res['badprob'] / res['goodprob'])
            res['total_iv'] = res['iv'].sum()
            res['total_ks'] = res['ks'].max()
            res['woe'] = t[i]['woe']

            BINS[i] = res[['variable', 'value', 'bin', 'count', 'count_distr', 'good', 'goodprob', 'bad', 'badprob', 'iv',
                           'total_iv', 'ks', 'total_ks', 'woe']]
        pd_bins = pd.concat(BINS)
        pd_bins.to_excel(self.SAVEPATH + 'bins_detail.xlsx')
        return pd_bins


def multi_report_task(data, ex_lst, dep, FILE_PATH, i, return_dict):
    print(str(i), 'report')
    report = Report(data, ex_lst, dep, FILE_PATH + str(i) + '_')
    #     key_value, bins_detail = report.iv_ks_psi()
    #     key_value, bins_detail = report.iv_ks_psi_chi2()
    key_value, bins_detail = report.iv_ks_psi_chi2_missing_quantile()
    return_dict[i] = key_value.shape[0]
