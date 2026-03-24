
import math
import toad
import pandas as pd
import numpy as np
import pydotplus
from six import StringIO
import os
from sklearn import tree
import warnings
from sklearn.tree import _tree

warnings.filterwarnings("ignore")


class AutoTree(object):
    def __init__(self,
                 datasets,
                 ex_lis,
                 dep='label',
                 min_samples=0.05,  # 分箱时最小箱样本占总比
                 min_samples_leaf=200,  # 决策树子节点最小样本个数
                 min_samples_split=20,  # 决策树划分前，父结点最小样本个数
                 max_leaf_nodes=20,
                 max_depth=4,  # 决策树最大深度
                 is_bin=True):  # 是否进行卡方分箱
        self.datasets = datasets
        self.ex_lis = ex_lis
        self.dep = dep
        self.ft_lst = [i for i in datasets.columns if i not in ex_lis]

        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_leaf_nodes = max_leaf_nodes
        self.is_bin = is_bin
        self.param = None

        self.bins = 0
        self.result = {}

    def fit_plot(self):
        os.environ["PATH"] += os.pathsep + 'C:/Users/lifan/.conda/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz'
        dtree = tree.DecisionTreeRegressor(max_depth=self.max_depth,
                                           random_state=0,
                                           min_samples_leaf=self.min_samples_leaf,
                                           min_samples_split=self.min_samples_split,
                                           max_leaf_nodes=self.max_leaf_nodes)
        del_lis = []
        for i in self.ex_lis:
            if i in list(self.datasets.columns):
                del_lis.append(i)
        x = self.datasets.drop(del_lis, axis=1)
        y = self.datasets[self.dep]

        if self.is_bin:
            combiner = toad.transform.Combiner()
            combiner.fit(x, y, method='chi', min_samples=self.min_samples)

            x_bin = combiner.transform(x)
            self.bins = combiner.export()
        else:
            combiner = 0
            self.bins = []
            x_bin = x.copy()

        dtree = dtree.fit(x_bin, y)
        self.estimator = dtree

        df_bin = x_bin.copy()
        df_bin[self.dep] = y

        dot_data = StringIO()
        tree.export_graphviz(dtree,
                             out_file=dot_data,
                             feature_names=x_bin.columns,
                             class_names=[self.dep],
                             filled=True,
                             rounded=True,
                             special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

        self.result['combiner'] = combiner
        self.result['bins'] = self.bins
        self.result['df_bin'] = df_bin
        self.result['graph'] = graph.create_png()
        self.result['model'] = dtree
        tree_ = self.result['model'].tree_
        self.result['features'] = [self.ft_lst[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
        self.result['features'] = list(set(self.result['features']))
        if "undefined!" in self.result['features']:
            self.result['features'].remove("undefined!")

        self.param = pd.DataFrame()
        self.param = self.param.append({'param': 'min_samples',
                                        'value': self.min_samples,
                                        'ps': '分箱时最小箱样本占总比'}, ignore_index=True)
        self.param = self.param.append({'param': 'min_samples_leaf',
                                        'value': self.min_samples_leaf,
                                        'ps': '决策树子节点最小样本个数'}, ignore_index=True)
        self.param = self.param.append({'param': 'min_samples_split',
                                        'value': self.min_samples_split,
                                        'ps': '决策树划分前父结点最小样本个数'}, ignore_index=True)
        self.param = self.param.append({'param': 'max_depth',
                                        'value': self.max_depth,
                                        'ps': '决策树最大深度'}, ignore_index=True)
        self.param = self.param.append({'param': 'is_bin',
                                        'value': self.is_bin,
                                        'ps': '是否进行卡方分箱'}, ignore_index=True)

        return self.result

    def transfer(self, data):
        data_bin = self.result['combiner'].transform(data[self.result['df_bin'].columns].fillna(-999))
        data_bin['pred'] = self.result['model'].predict(data[self.result['df_bin'].columns[:-1].to_list()])

        total = data_bin.groupby(['pred'])['label'].count()
        bad = data_bin.groupby(['pred'])['label'].sum()
        res = pd.DataFrame({'count': total, 'bad': bad})
        res['good'] = res['count'] - res['bad']
        res = res.sort_values(by='pred', ascending=False)
        res['distr'] = res['count'] / res['count'].sum()
        res['bad_rate'] = res['bad'] / res['count']
        res['bad_rate_init'] = res['bad'].sum() / res['count'].sum()
        res['lift'] = res['bad_rate'] / res['bad_rate_init']

        return data_bin, res

    def predict(self, data):
        data_bin = self.result['combiner'].transform(data[self.result['df_bin'].columns].fillna(-999))
        return self.result['model'].predict(data_bin[self.ft_lst])

    def get_samples_by_nodes(self, data, node_num):
        data_bin = self.result['combiner'].transform(data.fillna(-999))
        data_bin['node_num'] = self.result['model'].apply(data_bin[self.ft_lst], check_input=True)
        temp = data_bin[data_bin['node_num'] == int(node_num)]

        return pd.DataFrame({'bad_rate': temp['label'].mean(),
                             'count': temp.shape[0],
                             'bad': temp['label'].sum(),
                             'good': temp.shape[0] - temp['label'].sum(),
                             'bad_rate_init': data['label'].mean(),
                             'bad_prob': temp['label'].sum() / data_bin['label'].sum(),
                             'good_prob': (temp.shape[0] - temp['label'].sum()) / (data_bin.shape[0] - data_bin['label'].sum()),
                             'count_prob': temp.shape[0] / data_bin.shape[0],
                             'lift': temp['label'].mean() / data['label'].mean()
                             }, index=[0])

    def get_samples(self, data, node_num):
        data_bin = self.result['combiner'].transform(data.fillna(-999))
        data_bin['node_num'] = self.result['model'].apply(data_bin[self.ft_lst], check_input=True)
        return data_bin[data_bin['node_num'] == int(node_num)]

    def get_samples_nodes(self, data):
        data_bin = self.result['combiner'].transform(data.fillna(-999))
        data['node_num'] = self.result['model'].apply(data_bin[self.ft_lst], check_input=True)
        return data

    def get_detail(self):
        n_nodes = self.estimator.tree_.node_count
        children_left = self.estimator.tree_.children_left
        children_right = self.estimator.tree_.children_right
        feature = self.estimator.tree_.feature
        threshold = self.estimator.tree_.threshold

        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent parent depth

        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # if we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append(((children_left[node_id], parent_depth + 1)))
                stack.append(((children_right[node_id], parent_depth + 1)))
            else:
                is_leaves[node_id] = True
        print("The binary tree structure has %s nodes and has "
              "the following tree structure:"
              % n_nodes)

        for i in range(n_nodes):
            if is_leaves[i]:
                print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            else:
                print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                      "node %s."
                      % (node_depth[i] * "\t",
                         i,
                         children_left[i],
                         feature[i],
                         threshold[i],
                         children_right[i],
                         ))
        print()

    # 查看分箱结果
    def value(self, col):
        print('变量： ', col)
        print('分箱结果', self.result['bins'][col])
        print('分箱后取值', set(self.result['df_bin'][col]))

    # 计算负样本占比
    def badrate(self, data, dep):
        bad = sum(data[dep] == 1)
        good = sum(data[dep] == 0)
        print('负样本：', bad, '正样本：', good, '负样本占比', bad / (bad + good))

    # 读取数据
    def read_data(self, path, type):
        if type == 'excel':
            return pd.read_excel(path)
        elif type == 'csv' or type == 'txt':
            return pd.read_csv(path)
        elif type == 'sas':
            return pd.read_sas(path)
        else:
            return '未知类型文件'

    # 展示图像
    def image(self):
        graph = self.result['graph']
        from IPython.display import Image
        return Image(graph)

    # 规则输出
    def tree_to_code(self):
        tree_ = self.result['model'].tree_
        feature_name = [self.ft_lst[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
        name = feature_name[0]
        res = []
        result = pd.DataFrame()

        def recurse(node, depth, name, res):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                if node > 0:
                    name += ' and ' + feature_name[node]
                threshold = tree_.threshold[node]
                recurse(tree_.children_left[node], depth + 1, name + ' < ' +
                        str(self.result['bins'][feature_name[node]][math.floor(threshold)]), res)
                recurse(tree_.children_right[node], depth + 1, name + ' >= ' +
                        str(self.result['bins'][feature_name[node]][math.floor(threshold)]), res)
            else:
                # print('rule: ', name, 'value: ',tree_.value[node][0][0], 'samples: ', tree_.n_node_samples[node])
                res.append({'rule': name, 'bad_rate': tree_.value[node][0][0], 'count': tree_.n_node_samples[node],
                            'node_num': node})
            return res

        recurse(0, 1, name, res)

        for i in res:
            result = result.append(i, ignore_index=True)
        result['bad'] = result['count'] * result['bad_rate']
        result['good'] = result['count'] - result['bad']
        result['bad_rate_init'] = result['bad'].sum() / result['count'].sum()
        result['bad_prob'] = result['bad'] / result['bad'].sum()
        result['good_prob'] = result['good'] / result['good'].sum()
        result['count_prob'] = result['count'] / result['count'].sum()
        result['lift'] = result['bad_rate'] / result['bad_rate_init']
        return result

    def tree_to_code_s(self):
        tree_ = self.result['model'].tree_
        feature_name = [self.ft_lst[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
        name = '(x.' + feature_name[0]
        res = []
        result = pd.DataFrame()

        def recurse(node, depth, name, res):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                if node > 0:
                    name += ' )&( ' + 'x.' + feature_name[node]
                threshold = tree_.threshold[node]
                recurse(tree_.children_left[node], depth + 1, name + ' < ' +
                        str(self.result['bins'][feature_name[node]][math.floor(threshold)]), res)
                recurse(tree_.children_right[node], depth + 1, name + ' >= ' +
                        str(self.result['bins'][feature_name[node]][math.floor(threshold)]), res)
            else:
                # print('rule: ', name, 'value: ',tree_.value[node][0][0], 'samples: ', tree_.n_node_samples[node])
                res.append({'rule': name, 'bad_rate': tree_.value[node][0][0], 'count': tree_.n_node_samples[node],
                            'node_num': node})
            return res

        recurse(0, 1, name, res)

        for i in res:
            result = result.append(i, ignore_index=True)
        result['bad'] = result['count'] * result['bad_rate']
        result['good'] = result['count'] - result['bad']
        result['bad_rate_init'] = result['bad'].sum() / result['count'].sum()
        result['bad_prob'] = result['bad'] / result['bad'].sum()
        result['good_prob'] = result['good'] / result['good'].sum()
        result['count_prob'] = result['count'] / result['count'].sum()
        result['lift'] = result['bad_rate'] / result['bad_rate_init']
        return result

    def create_test_code_py(self, split, ret):
        r = self.tree_to_code_s()
        print('def fun_test_py(x): ')
        df_lst = [r[r['bad_rate'] <= split[0]]]
        for i in range(1, len(split)):
            df_lst.append(r[(r['bad_rate'] > split[i - 1]) & (r['bad_rate'] <= split[i])])
        df_lst.append(r[r['bad_rate'] > split[-1]])
        for i in range(len(df_lst)):
            s = '    if '
            l = df_lst[i]['rule'].to_list()
            for j in range(len(l) - 1):
                s += '(' + l[j] + ')) |'
            s += '(' + l[-1] + ')):'
            print(s)
            print('        return "' + ret[i] + '"')

    def analysis(self, data_all):
        res = self.tree_to_code()
        res.sort_values(by='bad_rate', ascending=False, inplace=True)
        data_comp = self.get_samples_nodes(data_all)
        res['original_bad_rate'] = res['bad_rate']
        r = pd.DataFrame(data_comp['node_num'].value_counts())
        r.rename(columns={'node_num': 'count'}, inplace=True)
        r['node_num'] = r.index
        r['bads'] = r.apply(lambda x: data_comp[data_comp['node_num'] == x['node_num']]['label'].sum(), axis=1)
        r['goods'] = r['count'] - r['bads']
        r['bad_rate'] = r['bads'] / r['count']
        r['bad_rate_init'] = data_comp['label'].mean()
        r['bad_prob'] = r['bads'] / r['bads'].sum()
        r['good_prob'] = r['goods'] / r['goods'].sum()
        r['count_prob'] = r['count'] / r['count'].sum()
        r['lift'] = r['bad_rate'] / r['bad_rate_init']
        r = pd.merge(res[['rule', 'node_num', 'original_bad_rate']], r, on='node_num')
        r = r.sort_values(by='original_bad_rate', ascending=False)
        r['cum_count_prob'] = r['count_prob'].cumsum()
        r['cum_bads'] = r['bads'].cumsum()
        r['cum_count'] = r['count'].cumsum()
        r['通过率'] = 1 - r['cum_count_prob']
        r['通过人群整体坏账率'] = r.apply(lambda x: (r['bads'].sum() - x['cum_bads']) / (r['count'].sum() - x['cum_count']),
                                 axis=1)
        return r
