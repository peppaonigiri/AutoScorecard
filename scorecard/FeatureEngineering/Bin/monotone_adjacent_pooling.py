
import numpy as np
import pandas as pd
import toad


class MAPA(object):
    def __init__(self, data, ex_lst, dep='label'):
        self.data = data
        self.ex_lst = ex_lst
        self.ft_lst = [i for i in data.columns if i not in ex_lst]
        self.combiner = toad.transform.Combiner()
        self.dep = dep

    def fit(self, method='dt', min_samples=0.05, n_bins=10):
        self.combiner.fit(self.data, self.data[self.dep], method=method, min_samples=min_samples, n_bins=n_bins,
                          exclude=self.ex_lst)
        data_bin = self.combiner.transform(self.data)

        res = dict()
        for ft in self.ft_lst:
            split_points = self.combiner[ft]
            total = data_bin.groupby([ft])[self.dep].count()
            bad = data_bin.groupby([ft])[self.dep].sum()
            temp = pd.DataFrame({'total': total, 'bad': bad})
            temp['good'] = temp['total'] - temp['bad']
            temp['bad_rate'] = temp['bad'] / temp['total']
            temp[ft] = temp.index
            temp.index = [i for i in range(len(temp))]
            temp.sort_values(by=ft, ascending=True, inplace=True)
            temp['badCum'] = temp['bad'].cumsum()
            temp['totalCum'] = temp['total'].cumsum()
            temp['badrateCumRate'] = temp['badCum'].cumsum() / temp['totalCum'].sum()
            # print(ft)
            # print(split_points)
            cut_points, idx = [], 0
            bad = temp['bad'].to_list()
            total = temp['total'].to_list()

            while idx <= temp[ft].max():
                r = [-float('inf')] * idx
                b = [sum(bad[idx:i]) for i in range(idx + 1, len(bad) + 1)]
                t = [sum(total[idx:i]) for i in range(idx + 1, len(total) + 1)]
                r += [b[i] / t[i] for i in range(len(b))]
                # print(r)
                temp['iter_' + str(idx)] = r
                idx = r.index((max(r))) + 1
                if idx - 2 >= 0:
                    cut_points.append(split_points[idx - 2])

            res[ft] = {'df': temp, 'split_points': cut_points}

        return res

    def transform(self, data, dic_bin, ex_lst):
        ft_lst = [i for i in data.columns if i not in ex_lst]
        data_bin = data[ex_lst].copy()
        for ft in ft_lst:
            labels = [i for i in range(len(dic_bin[ft]['split_points']) + 1)]
            data_bin[ft] = pd.cut(data[ft], [- float('inf')] + dic_bin[ft]['split_points'] + [float('inf')],
                                  labels=labels)
        return data_bin
