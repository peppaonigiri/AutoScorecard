from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import toad
import numpy as np
import scorecardpy as sc


class DecisionTree(object):
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
        self.combiner.fit(self.dev, self.dev[self.dep], method='dt', min_samples=0.05, exclude=self.ex_lst, n_bins=n_bins,
                          empty_separate=empty_separate)
        return self.combiner

    def get_boundary(self, x, y, criterion='entropy', max_leaf_nodes=6, min_samples_leaf=0.05):
        boundary = []
        clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                     max_leaf_nodes=6,  # 最大叶子节点数
                                     min_samples_leaf=0.05)  # 叶子节点样本数量最小占比
        clf.fit(x, y)

        # tree.plot_tree(clf) #打印决策树的结构图
        # plt.show()

        n_nodes = clf.tree_.node_count  # 决策树的节点数
        children_left = clf.tree_.children_left  # node_count大小的数组，children_left[i]表示第i个节点的左子节点
        children_right = clf.tree_.children_right  # node_count大小的数组，children_right[i]表示第i个节点的右子节点
        threshold = clf.tree_.threshold  # node_count大小的数组，threshold[i]表示第i个节点划分数据集的阈值

        for i in range(n_nodes):
            if children_left[i] != children_right[i]:  # 非叶节点
                boundary.append(threshold[i])

        boundary.sort()
        # min_x = x.min()
        # max_x = x.max() + 0.1
        boundary = [float('-inf')] + boundary + [float('inf')]

        return boundary

    def combiner_dtc(self, criterion='entropy', max_leaf_nodes=6, min_samples_leaf=0.05):
        y = self.df[self.dep].value
        res = {}
        for i in self.ft_lst:
            x = self.df[i].value
            res[i] = self.get_boundary(self, x, y)
        return res

    def calc_cut_point(self,sample_set, var):
        '''
        计算相邻评分的中位数
        '''
        var_list = list(np.unique(sample_set[var]))
        var_median_list = []
        for i in range(len(var_list) - 1):
            var_median = (var_list[i] + var_list[i + 1]) / 2
            var_median_list.append(var_median)
        return var_median_list

    # var表示需要进行分箱的变量名，返回一个样本变量中位数的list

    def choose_best_split(self, sample_set, var, min_sample):
        '''
        使用CART分类决策树选择最好的样本切分点
        param min_sample: 待切分样本的最小样本量(限制条件)
        '''
        # 根据样本评分计算相邻不同分数的中间值
        cut_point_list = self.calc_cut_point(sample_set, var)
        median_len = len(cut_point_list)
        sample_cnt = sample_set.shape[0]
        sample1_cnt = sum(sample_set[self.dep])
        sample0_cnt = sample_cnt - sample1_cnt
        gini = 1 - np.square(sample1_cnt / sample_cnt) - np.square(sample0_cnt / sample_cnt)

        best_gini, best_split_point, best_split_position = 0.0, 0.0, 0.0

        for i in range(median_len):
            left, right = sample_set[sample_set[var] < cut_point_list[i]], sample_set[sample_set[var] > cut_point_list[i]]
            left_cnt, right_cnt = left.shape[0], right.shape[0]
            left1_cnt, right1_cnt = sum(left[self.dep]), sum(right[self.dep])
            left0_cnt, right0_cnt = left_cnt - left1_cnt, right_cnt - right1_cnt
            left_ratio, right_ratio = left_cnt / sample_cnt, right_cnt / sample_cnt

            if left_cnt < min_sample or right_cnt < min_sample:
                continue

            gini_left = 1 - np.square(left1_cnt / left_cnt) - np.square(left0_cnt / left_cnt)
            gini_right = 1 - np.square(right1_cnt / right_cnt) - np.square(right0_cnt / right_cnt)
            gini_temp = gini - (left_ratio * gini_left + right_ratio * gini_right)
            if gini_temp > best_gini:
                best_gini = gini_temp
                best_split_point = cut_point_list[i]
                if median_len > 1:
                    best_split_position = i / (median_len - 1)
                else:
                    best_split_position = i / median_len
            else:
                continue

        gini = gini - best_gini
        return best_split_point, best_split_position

    def bining_data_split(self, sample_set, var, min_sample, split_list):
        '''
        划分数据找到最优分割点list
        param sample_set: 待切分样本
        param var: 分割变量名称
        param min_sample: 待切分样本的最小样本量(限制条件)
        param split_list: 最优分割点list
        '''
        split, position = self.choose_best_split(sample_set, var, min_sample)
        if split != 0.0:
            split_list.append(split)
        # 根据分割点划分数据集，继续进行划分
        sample_set_left = sample_set[sample_set[var] < split]
        sample_set_right = sample_set[sample_set[var] > split]
        # 如果左子树样本量超过2倍最小样本量，且分割点不是第一个分割点，则切分左子树
        if len(sample_set_left) >= min_sample * 2 and position not in [0.0, 1.0]:
            self.bining_data_split(sample_set_left, var, min_sample, split_list)
        else:
            None
        # 如果右子树样本量超过2倍最小样本量，且分割点不是最后一个分割点，则切分右子树
        if len(sample_set_right) >= min_sample * 2 and position not in [0.0, 1.0]:
            self.bining_data_split(sample_set_right, var, min_sample, split_list)
        else:
            None

    def combiner_gini(self, var, min_sample=0.05):
        min_samples = min_sample * self.dev.shape[0]
        res = {}
        for i in self.ft_lst:
            split_list = []
            self.bining_data_split(self.dev, var, min_samples, split_list)
            res[i] = [float('-inf')] + split_list.sort() + [float('inf')]
        return res

    def combiner_scorecardpy(self):
        res = sc.woebin(self.dev, y=self.dep, method='dt', min_perc_coarse_bin=0.05, stop_limit=0.1,
                        special_values=[-999])
        return res


