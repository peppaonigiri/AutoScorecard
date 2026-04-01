from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


def optimal_binning_boundary(x,y,max_leaf_nodes=6,min_samples_leaf=0.05):
    """
    使用决策树计算最优分箱边界
    :param x: pandas Series
    :param y: pandas Series
    :param max_leaf_nodes: int, 决策树最大叶子节点数
    :param min_samples_leaf: float, 叶子节点最小样本数
    :return: list
    """
    boundary = []
    min_x = -np.inf
    max_x = np.inf
    x = x.values
    y = y.values
    clf = DecisionTreeClassifier(criterion='gini',max_leaf_nodes=max_leaf_nodes,min_samples_leaf=min_samples_leaf)
    clf.fit(x.reshape(-1,1),y)
    n_nodes = clf.tree_.node_count # 决策树的节点数
    children_left = clf.tree_.children_left # node_count大小的数组，children_left[i]表示第i个节点的左子节点
    children_right = clf.tree_.children_right # node_count大小的数组，children_right[i]表示第i个节点的右子节点
    threshold = clf.tree_.threshold # node_count大小的数组，threshold[i]表示第i个节点划分数据集的阈值

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:
            boundary.append(threshold[i])
    boundary.sort()
    boundary = [min_x]+boundary+[max_x]
    return boundary


def feature_woe_iv_bins(x,y,bins=None,print_iv=False):
    """
    计算特征的分箱信息, 包括WOE、IV、LIFT、KS、卡方值等
    :param x: pandas Series, 特征
    :param y: pandas Series, 目标变量
    :param data: pandas DataFrame, 包含特征和目标变量
    :param bins: list, 若不为None, 则使用bins作为分箱边界
    :param print_iv: bool, 是否打印IV
    :return: pandas DataFrame, 包含分箱信息
    """
    x_name = x.name
    y_name = y.name
    if bins == None:
        boundary = optimal_binning_boundary(x,y)
    else:
        boundary = bins
    df = pd.concat([x,y],axis=1)
    # 如果返回的分箱是[-inf,inf]
    if isinstance(boundary,int):
        df['bin'] = pd.qcut(x,bins,duplicates='drop')
    elif len(boundary) <= 2:
        df['bin'] = pd.qcut(x,2,duplicates='drop')
    else:
        df['bin'] = pd.cut(x,bins=boundary,right=True)
    grouped = df.groupby('bin', observed=True)
    result_df = grouped[y_name].agg([('good',lambda y:(y==0).sum()),('bad',lambda y:(y==1).sum()),('total', 'count')]).reset_index()
    result_df['bin'] = result_df['bin'].map(lambda x:str(x))
    result_df['good_pct'] = result_df['good'] / result_df['good'].sum()
    result_df['bad_pct'] = result_df['bad'] / result_df['bad'].sum()
    result_df['total_pct'] = result_df['total'] / result_df['total'].sum()
    # 'good_pct','bad_pct'等于0会导致后面WOE和IV计算出问题，所以替换为0.0001
    result_df[['good_pct','bad_pct']] = result_df[['good_pct','bad_pct']].replace({0:0.0001})
    
    result_df['bad_rate'] = result_df['bad'] / result_df['total']
    result_df['bad_pct_cumsum'] = result_df['bad'].cumsum()/result_df['bad'].sum()
    result_df['good_pct_cumsum'] = result_df['good'].cumsum() / result_df['good'].sum()
    result_df['KS(%)'] = abs(result_df['bad_pct_cumsum']-result_df['good_pct_cumsum'])*100
    # 计算LIFT
    result_df['lift'] = (result_df['bad']/result_df['total'])/(result_df['bad'].sum()/result_df['total'].sum())
    # 计算卡方值，对该列求和即可
    result_df['X2'] = ((result_df['bad']-(result_df['bad'].sum()/result_df['total'].sum())*result_df['total'])**2)/((result_df['bad'].sum()/result_df['total'].sum())*result_df['total'])
    result_df['woe'] = np.log(result_df['good_pct'] / result_df['bad_pct'])
    result_df['iv'] = (result_df['good_pct'] - result_df['bad_pct']) * result_df['woe']
    result_df['total_iv'] = result_df['iv'].sum()
    result_df.insert(0,'variable',x_name)
    result_df['good_distr'] = result_df['good'] / result_df['total'].sum()
    result_df['bad_distr'] = result_df['bad'] / result_df['total'].sum()
    if print_iv:
        print("变量  {}  的IV = {}".format(x_name,result_df[~result_df['iv'].isin([np.inf,-np.inf])]['iv'].sum()))
    
    return result_df

def plot_bin(binx, title=None, show_iv=True):
    '''
    绘制分箱的分布图
    :param binx: pandas DataFrame, 包含分箱信息
    :param title: str, 图标题
    :param show_iv: bool, 是否显示IV
    :return: None
    '''
    # y_right_max
    y_right_max = np.ceil(binx['bad_rate'].max()*10)
    if y_right_max % 2 == 1: y_right_max=y_right_max+1
    if y_right_max - binx['bad_rate'].max()*10 <= 0.3: y_right_max = y_right_max+2
    y_right_max = y_right_max/10
    if y_right_max>1 or y_right_max<=0 or y_right_max is np.nan or y_right_max is None: y_right_max=1
    ## y_left_max
    y_left_max = np.ceil(binx['total_pct'].max()*10)/10
    if y_left_max>1 or y_left_max<=0 or y_left_max is np.nan or y_left_max is None: y_left_max=1
    # title
    title_string = binx.loc[0,'variable']+"  (iv:"+str(round(binx.loc[0,'total_iv'],4))+")" if show_iv else binx.loc[0,'variable']
    title_string = title+'-'+title_string if title is not None else title_string
    # param
    ind = np.arange(len(binx.index))    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence
    ###### plot ######
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # ax1
    p1 = ax1.bar(ind, binx['good_distr'], width, color=(24/254, 192/254, 196/254))
    p2 = ax1.bar(ind, binx['bad_distr'], width, bottom=binx['good_distr'], color=(246/254, 115/254, 109/254))
    for i in ind:
        ax1.text(i, binx.loc[i,'total_pct']*1.02, str(round(binx.loc[i,'total_pct']*100,1))+'%, '+str(binx.loc[i,'total']), ha='center')
    # ax2
    ax2.plot(ind, binx['bad_rate'], marker='o', color='blue')
    for i in ind:
        ax2.text(i, binx.loc[i,'bad_rate']*1.02, str(round(binx.loc[i,'bad_rate']*100,1))+'%', color='blue', ha='center')
    # settings
    ax1.set_ylabel('Bin count distribution')
    ax2.set_ylabel('Bad probability', color='blue')
    ax1.set_yticks(np.arange(0, y_left_max+0.2, 0.2))
    ax2.set_yticks(np.arange(0, y_right_max+0.2, 0.2))
    ax2.tick_params(axis='y', colors='blue')
    plt.xticks(ind, binx['bin'])
    plt.title(title_string, loc='left')
    plt.legend((p2[0], p1[0]), ('bad', 'good'), loc='upper right')
    # show plot
    # plt.show()
    return fig