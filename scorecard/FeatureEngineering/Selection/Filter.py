
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : F. Li
    @Time    : 2019/7/24 23:23
    @Use     :
"""
import toad

from FeatureEngineering.KeyValue import Missing
from FeatureEngineering.KeyValue import Std
from FeatureEngineering.KeyValue import Freq
from FeatureEngineering.KeyValue import Chi2
from FeatureEngineering.KeyValue import IV
from FeatureEngineering.KeyValue import Corr
from FeatureEngineering.KeyValue import Importance
from FeatureEngineering.KeyValue import PSI


class Result(object):
    def __init__(self, threshold, drop_lst, keep_lst):
        self.threshold = threshold
        self.drop_lst = drop_lst
        self.keep_lst = keep_lst
        self.num_drop = len(drop_lst)
        self.num_keep = len(keep_lst)


class Filter(object):
    def __init__(self, df, dep, ex_lst, SAVE_PATH):
        self.df = df
        self.dep = dep
        self.ex_lst = ex_lst
        self.ft_lst = [i for i in df.columns if i not in ex_lst]
        self.dev = self.df[self.df['target'] == 'train']
        self.val = self.df[self.df['target'] == 'valid']
        self.oot = self.df[self.df['target'] == 'oot']
        self.SAVE_PATH = SAVE_PATH

    # def save(self, SAVE_PATH, res):

    def print(self, dic):
        for key, value in dic.items():
            print(key, '  num_drop: %d,   num_keep: %d ' % (value.num_drop, value.num_keep))

    def drop(self, dic):
        ft_lst = [i for i in list(self.df.columns) if i not in self.ex_lst]
        return self.df

    def round_1(self, threshold_dict, f_print=True, f_save=True):
        """
        :param threshold_dict: {missing: 0.8, std: 0.95, freq: 0.95}
        :return:
        """
        missing = Missing.filter(self.dev[self.ft_lst], threshold_dict['missing'])
        std = Std.filter(self.dev[missing.keep_lst], threshold_dict['std'])
        freq = Freq.filter(self.dev[std.keep_lst], threshold_dict['freq'])

        res = {'missing': missing, 'std': std, 'freq': freq}

        if f_print:
            self.print(res)

        if f_save:
            missing.res.to_csv(self.SAVE_PATH+'missing.csv', index=False)
            std.res.to_csv(self.SAVE_PATH + 'std.csv', index=False)
            freq.res.to_csv(self.SAVE_PATH + 'freq.csv', index=False)

        return res

    def round_2(self, threshold_dict, f_print=True, f_save=True):
        """
        :param threshold_dict: {chi2: >3, corr: < 0.9, iv: >0.009, psi: <0.1, importance: >0}
        :return:
        """
        chi2 = Chi2.filter(self.dev, self.dep, self.ex_lst, threshold_dict['chi2'])
        iv = IV.filter(self.df, self.ex_lst, self.dep, threshold_dict['iv'])
        corr = Corr.filter(self.dev, iv.res, self.ex_lst, threshold_dict['corr'])
        importance = Importance.filter(self.df, self.ex_lst, self.dep, 'xgb_gain', threshold_dict['importance'])
        psi = PSI.filter(self.dev, self.ex_lst, threshold_dict['psi'])

        res = {'chi2': chi2, 'iv': iv, 'corr': corr, 'importance': importance, 'psi': psi}

        if f_print:
            self.print(res)

        if f_save:
            chi2.res.to_csv(self.SAVE_PATH + 'chi2.csv', index=False)
            iv.res.to_csv(self.SAVE_PATH + 'iv.csv', index=False)
            corr.res.to_csv(self.SAVE_PATH + 'corr.csv', index=False)
            importance.res.to_csv(self.SAVE_PATH + 'importance.csv', index=False)
            psi.res.to_csv(self.SAVE_PATH + 'psi.csv', index=False)

        return res

    def round_toad(self, threshold_dict, f_print=True, f_save=False):
        dev_slct, drop_lst = toad.selection.select(self.dev, self.dev[self.dep],
                                                   empty=threshold_dict['missing'],
                                                   iv=threshold_dict['iv'],
                                                   corr=threshold_dict['corr'],
                                                   return_drop=True,
                                                   exclude=self.ex_lst)
        missing = Result(threshold_dict['missing'], drop_lst['empty'],
                         [i for i in self.df.columns if i not in drop_lst['empty']])
        iv = Result(threshold_dict['iv'], drop_lst['iv'],
                    [i for i in self.df.columns if i not in drop_lst['iv']])
        corr = Result(threshold_dict['corr'], drop_lst['corr'],
                      [i for i in self.df.columns if i not in drop_lst['corr']])

        res = {'missing': missing, 'iv': iv, 'corr': corr}

        if f_print:
            self.print(res)

        # if f_save:
        #     missing.res.to_csv(self.SAVE_PATH + 'std.csv', index=False)
        #     iv.res.to_csv(self.SAVE_PATH + 'freq.csv', index=False)
        #     corr.res.to_csv(self.SAVE_PATH + 'freq.csv', index=False)





