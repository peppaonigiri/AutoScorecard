from six import StringIO
from sklearn import tree
import pydotplus
import toad
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import itertools
from itertools import combinations
from pathlib import Path
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder


class auto_tree(object):
    def __init__(self, dataset, ex_lis, y, min_samples, min_samples_leaf, min_samples_split, max_depth, is_bin=True):
        """
        :param dataset: 数据集 dataframe格式
        :param ex_lis: 不参与建模的特征，如id，时间切片等。 list格式
        :param y:
        :param min_samples: 分箱时最小箱的样本占总比 numeric格式
        :param min_samples_leaf: 决策树子节点最小样本个数 numeric格式
        :param min_samples_split: 决策树划分前，父节点最小样本个数 numeric格式
        :param max_depth: 决策树最大深度 numeric格式
        :param is_bin: 是否进行卡方分箱 bool格式（True/False）
        """

        self.dataset = dataset
        self.ex_lis = ex_lis
        self.y = y
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.is_bin = is_bin
        self.bins = 0
        self.result = {}

    def fit_plot(self):
        # os.environ["PATH"] += os.pathsep +
        # "D:\\ProgramFiles(x86)\\Python-3.6\\Python36\\ideProject\\xinlan\\tanzhi\\all_tools_package\\" #'/root/india-pywork'
        # print('os.environ["PATH"] : ',os.environ["PATH"])
        dtree = tree.DecisionTreeClassifier(max_depth=self.max_depth,
                                            min_samples_leaf=self.min_samples_leaf,
                                            min_samples_split=self.min_samples_split)

        self.ex_lis.append(self.y)
        x = self.dataset.drop(columns=self.ex_lis, axis=1)
        y = self.dataset[self.y]

        if self.is_bin:
            # 分箱
            combiner = toad.transform.Combiner()
            combiner.fit(x, y, method='chi', min_samples=self.min_samples)

            x_bin = combiner.transform(x)
            self.bins = combiner.export()
        else:
            combiner = None
            x_bin = x.copy()

        dtree = dtree.fit(x_bin, y)

        df_bin = x_bin.copy()

        df_bin[self.y] = y

        with open("dt.dot", "w") as f:
            tree.export_graphviz(dtree, out_file=f)

        dot_data = StringIO()

        pprob = []

        for i in range(dtree.tree_.value.shape[0]):
            class_0 = dtree.tree_.value[i][0][0]
            class_1 = dtree.tree_.value[i][0][1]
            pprob.append(class_1 / (class_0 + class_1))

        # print('pprob: ',pprob)

        if max(pprob) < 0.9:
            return self.result

        tree.export_graphviz(dtree, out_file=dot_data,
                             feature_names=x_bin.columns,
                             #  class_names=['0', '1'],
                             node_ids=True,
                             proportion=False,
                             #  rotate =True,
                             filled=True, rounded=True,
                             special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

        self.result['combiner'] = combiner
        self.result['bins'] = self.bins
        self.result['df_bin'] = df_bin
        self.result['graph'] = graph.create_png()
        self.result['model'] = dtree
        return self.result

        # 查看分箱结果

    def value(result, col):
        # print('变量：', col)
        # print('分箱结果', result['bins'][col])
        # print('分箱后取值', set(result['df_bin'][col]))
        return dict(zip(set(result['df_bin'][col]), result['bins'][col]))

    # 计算负样本占比  
    def badrate(data, y):
        bad = sum(data[y] == 1)
        good = sum(data[y] == 0)
        print('负样本：', bad, '正样本：', good, '负样本占比: ', bad / (bad + good))


def GetTree(data, y, cols_ls, filename, url, min_samples=0.001, min_samples_leaf=30, min_samples_split=100, max_depth=3):
    """
    data: DataFrame
    y: 目标字段名
    cols_ls: 特征字段 list
    """
    cols = [y] + cols_ls
    result = auto_tree(dataset=data[cols],
                       ex_lis=[],
                       y=y,
                       min_samples=min_samples,
                       min_samples_leaf=min_samples_leaf,
                       min_samples_split=min_samples_split,
                       is_bin=True,
                       max_depth=max_depth).fit_plot()

    if not bool(result):
        return {}

    value_dic = {}
    for col in cols_ls:
        value_dic[col] = auto_tree.value(result, col)

    # url = "D:\\ProgramFiles(x86)\\Python-3.6\\Python36\\ideProject\\xinlan\\tanzhi\\all_tools_package\\dst_tree\\"
    # filename = 'tree.jpg'
    image = Image.open(BytesIO(result['graph']))
    # image = (result['graph'])
    image.save(url + filename)

    dic = {
        'value': value_dic,
        'url': url,
        'filename': filename
    }
    return dic


def func_policy(data, threefeatures, y, url):
    k = 0
    for i in tqdm(threefeatures):
        k = k + 1

        if k < 0:
            continue

        df_col = pd.DataFrame()

        new_featurn = list(i)
        feature_c = '__'.join(new_featurn)
        try:
            dic = GetTree(data, y, new_featurn, feature_c + '.png', url, min_samples_leaf=100)
            f = open(dic['url'] + dic['filename'].split('.')[0] + '.txt', 'w', encoding='utf-8')
            # print('file : ',dic['url'] + dic['filename'].split('.')[0] + '.txt')
            f.write(str(dic['value']) + '\n')
            f.close()
        except:
            pass

# threefeatures = list(itertools.combinations(col_m,3))
# func_policy(t_data, threefeatures, 'y')
