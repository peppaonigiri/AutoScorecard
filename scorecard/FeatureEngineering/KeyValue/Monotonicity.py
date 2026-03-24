
import toad
from toad.plot import bin_plot
from functools import reduce
import numpy as np


class Result(object):
    def __init__(self, threshold, drop_lst, keep_lst, df_res):
        self.threshold = threshold
        self.drop_lst = drop_lst
        self.keep_lst = keep_lst
        self.num_drop = len(drop_lst)
        self.num_keep = len(keep_lst)
        self.res = df_res


class MonotonicityAdjuster(object):
    def __init__(self, ft_lst):
        self.ft_lst = ft_lst
        self.ascending_lst = []
        self.descending_lst = []
        self.peak_lst = []
        self.valley_lst = []
        self.adjust_lst = []

    def type_monotonicity(self, table):
        derivative = []
        diff_count = 0
        extreme_points = []

        for i in range(table.shape[0] - 1):
            derivative.append(table.loc[i, 'badprob'] - table.loc[i + 1, 'badprob'])

        for i in range(len(derivative) - 1):
            if derivative[i] * derivative[i+1] < 0:
                diff_count += 1
                extreme_points.append(i+1)

        if diff_count == 0 and np.mean(derivative) > 0:
            return ['descending', extreme_points]
        elif diff_count == 0 and np.mean(derivative) < 0:
            return ['ascending', extreme_points]
        elif diff_count == 1 and derivative[0] > derivative[-1]:
            return ['valley', extreme_points[0]]
        elif diff_count == 1 and derivative[0] < derivative[-1]:
            return ['peak', extreme_points[0]]
        else:
            return ['non_type', extreme_points]

    def type_consistency(self, ft, df_bins_dic):
        monotonicity_lst = [self.type_monotonicity(value[ft])[0] for key, value in df_bins_dic.items()]
        consistency_lst = [self.type_monotonicity(value[ft])[1] for key, value in df_bins_dic.items()]
        # print('monotonicity_lst: ', monotonicity_lst)
        # print('consistency_lst: ', consistency_lst)

        if len(set(monotonicity_lst)) == 1:
            if monotonicity_lst[0] == 'peak' or monotonicity_lst[0] == 'valley':
                if len(set(consistency_lst)) == 1:
                    return True, monotonicity_lst[0]
                else:
                    return False, 'non_type'
            elif monotonicity_lst[0] == 'descending' or monotonicity_lst[0] == 'ascending':
                return True, monotonicity_lst[0]
            else:
                return False, 'non_type'
        else:
            return False, 'non_type'

    def classify(self, df_bins_dic):
        for ft in self.ft_lst:
            # print(ft)
            if_consistency, class_ = self.type_consistency(ft, df_bins_dic)
            # print(if_consistency, 'class: ', class_)
            if if_consistency:
                if class_ == 'descending':
                    self.descending_lst.append(ft)
                elif class_ == 'ascending':
                    self.ascending_lst.append(ft)
                elif class_ == 'valley':
                    self.valley_lst.append(ft)
                elif class_ == 'peak':
                    self.peak_lst.append(ft)
                else:
                    self.adjust_lst.append(ft)
            else:
                self.adjust_lst.append(ft)


