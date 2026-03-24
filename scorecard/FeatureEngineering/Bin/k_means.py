import pandas as pd
import toad


class Kmeans(object):
    def __int__(self, dev, val, oot, dep, ft_lst, ex_lst):
        self.df = pd.concat([dev, val, oot])
        self.dev = dev.fillna(-999)
        self.val = val.fillna(-999)
        self.oot = oot.fillna(-999)
        self.dep = dep
        self.ft_lst = ft_lst
        self.ex_lst = ex_lst
        self.rules = {}

    def combiner_toad(self, n_bins=6, empty_separate=True):
        self.combiner = toad.transform.Combiner()
        self.combiner.fit(self.dev, self.dev[self.dep], method='kmeans', exclude=self.ex_lst,
                          n_bins=n_bins, empty_separate=empty_separate)
        return self.combiner