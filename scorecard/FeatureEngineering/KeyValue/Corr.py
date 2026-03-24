
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pyecharts.charts import HeatMap
import pyecharts.options as opts
from copy import copy

data = pd.DataFrame()


def pandas(df):
    corr = df.corr(method='pearson', min_periods=1)
    res = pd.DataFrame(columns={'var_names', 'corr_value'})
    lst = corr.columns
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            res = res.append({'var_names': lst[i] + ',' + lst[j],
                              'corr_value': corr.iloc[i, j]}, ignore_index=True)
    res = res.sort_values(by='corr_value', ascending=False)
    return res

# 参数说明：
# method：可选值为{‘pearson’, ‘kendall’, ‘spearman’}
# - pearson：Pearson相关系数来衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算，针对非线性数据便会有误差。
# - kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据
# - spearman：非线性的，非正太分析的数据的相关系数
# - min_periods：样本最少的数据量


class Result(object):
    def __init__(self, threshold, drop_lst, keep_lst, df_res):
        self.threshold = threshold
        self.drop_lst = drop_lst
        self.keep_lst = keep_lst
        self.num_drop = len(drop_lst)
        self.num_keep = len(keep_lst)
        self.res = df_res


def value(df, ft_lst):
    """
    :param train: 模型中的训练DataFrame
    :param features1_iv:  IV筛选后剩余变量列表
    :return: res
    """
    res = pd.DataFrame(columns={'var_names', 'corr'})
    for i in range(len(ft_lst) - 1):
        for j in range(i + 1, len(ft_lst)):
            res = res.append({'var_names': [ft_lst[i], ft_lst[j]], 'corr': df[[ft_lst[i], ft_lst[j]]].corr().iloc[0, 1]},
                             ignore_index=True)
    return res


def filter(df, df_iv, ex_lst, threshold=0.95):
    """
    注意：是否替换为woe再计算相关性
    :param train: 模型中的训练DataFrame
    :return: 返回corr筛选后删除的变量列表
    说明：计算的是woe替换后的corr
    """
    ft_lst = [i for i in list(df.columns) if i not in ex_lst]
    drop_lst = []
    res = value(df, ft_lst)
    for index, row in res.iterrows():
        if abs(row['corr']) > threshold:
            iv_0 = df_iv[df_iv['var_names'] == row['var_names'][0]]
            iv_1 = df_iv[df_iv['var_names'] == row['var_names'][1]]
            if iv_0 < iv_1:
                drop_lst.append(row['var_names'][0])
            else:
                drop_lst.append(row['var_names'][1])

    keep_lst = [i for i in df.columns if i not in drop_lst]

    return Result(threshold, drop_lst, keep_lst, res)


class CalCorr(object):
    """
    相关性高的两个变量，删除和其它变量整体相关更高的那个，这个方法就是我们今天要介绍的，其目的就是在一定的相关阈值之下，尽量多地保留变量，其具体算法过程如下：
    计算所有变量的相关矩阵；
    1、挑选出相关系数最高的一对变量A和B；
    2、分别对A和B计算其与其它变量相关系数的平均值α，β；
    3、如果α > β，删除变量A，否则删除B。
    4、重复2 - 4步直到所有变量两两之间的相关系数低于给定阈值。

    """
    def __init__(self, df, n_clusters=5, threshold=0.7):
        # 对所有变量矩阵先进行处理
        self.df = self.handle_df(df.copy())
        # Kmeans聚类的时候指定聚成几类
        self.n_clusters = n_clusters
        # 相关性筛选时的阈值
        self.threshold = threshold
        # 变量的相关矩阵存下来
        self.corr = None

    @property
    def corr_matrix(self):
        # 变量的相关矩阵
        return self.corr

    @property
    def corr_pairs(self):
        # 变量两两间的相关系数对
        return self.pairs

    def handle_df(self, df):

        """
        先处理一下变量矩阵：
        类别型变量做编码处理
        缺失值填充为中位数
        所有变量做一下均一化处理
        :param df:
        :return:
        """

        for item in df.select_dtypes(include=["object"]):
            df[item] = df[item].astype('category').cat.codes

        df = df.fillna(df.median())
        df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
        print(df.shape)
        return df

    def order_by_kmeans(self):
        """
           根据Kmeans聚类结果对变量进行排序
        :return:
        """

        kk = KMeans(n_clusters=self.n_clusters)
        res = kk.fit_predict(self.df.T)

        self.df = self.df.append(pd.Series(res, index=self.df.columns), ignore_index=True)
        self.df.sort_values(by=self.df.shape[0] - 1, axis=1, inplace=True)
        self.df.drop(self.df.shape[0] - 1, inplace=True)

        print(self.df.shape)

    def corr_heat_map(self):
        """
        计算相关矩阵，并使用pyecharts的heat_map呈现
        :return:
        """

        self.order_by_kmeans()

        self.corr = self.df.corr(method="pearson")
        self.corr = self.corr.round(3)
        self.corr = self.corr.apply(lambda x: abs(x))
        myvalues = []
        for i in range(len(self.corr.index)):
            for j in range(len(self.corr.index)):
                tmp = [i, j, self.corr.iloc[i, j]]
                myvalues.append(copy(tmp))
        self.__setattr__('pairs', myvalues)

        heat_map = HeatMap(init_opts=opts.InitOpts(width="1440px", height="1440px")) \
            .add_xaxis(list(self.corr.columns)) \
            .add_yaxis("corr", list(self.corr.index), myvalues) \
            .set_global_opts(
            title_opts=opts.TitleOpts(title="模型变量相关性"),
            datazoom_opts=[opts.DataZoomOpts(is_show=True, is_realtime=True), ],
            visualmap_opts=opts.VisualMapOpts(min_=-1.2, max_=1.2, pos_right=20),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category", is_scale=True, is_inverse=True,
                                     axislabel_opts=opts.LabelOpts(is_show=True, rotate=-60)),
            yaxis_opts=opts.AxisOpts(is_scale=True, is_inverse=False,
                                     axislabel_opts=opts.LabelOpts(is_show=True, position="right")),
            tooltip_opts=opts.TooltipOpts(is_show=True)) \
            .set_series_opts(label_opts=opts.LabelOpts(is_show=True, position="insideBottom"))

        heat_map.render()

    def drop_hight_corr(self):
        """
        根据上述算法，删除相关性高的变量
        :return:
        """
        cor_pair = self.pairs
        cor_pair.sort(key=lambda x: x[2], reverse=True)
        del_pair = []
        del_col = []
        for item in cor_pair:
            if (item[0] == item[1]) | (set(item[:2]) in del_pair) | (item[0] in del_col) | (item[1] in del_col):
                continue

            if item[2] > self.threshold:
                c1 = self.corr.iloc[item[0], [x for x in range(self.corr.shape[0]) if x not in del_col]].mean()
                c2 = self.corr.iloc[item[1], [x for x in range(self.corr.shape[0]) if x not in del_col]].mean()

                del_col.append(item[0] if c1 > c2 else item[1])
                del_pair.append(set(item[:2]))
            else:
                break

        del_col_name = self.corr.iloc[:, del_col].columns

        return list(del_col_name)


if __name__ == '__main__':
    c = CalCorr(df, n_clusters=5, threshold=0.7)
    # 在当前路径下产生的render.html文件即为相关矩阵热力图
    c.corr_heat_map()
    del_col = c.drop_hight_corr()
