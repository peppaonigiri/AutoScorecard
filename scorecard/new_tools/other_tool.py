import pandas as pd
import numpy as np
import toad
import scorecardpy as sc
import toad.plot

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def toad_iv_plot(df, target, features):
    """
    绘制IV值图
    :param df: pandas DataFrame
    :param target: 目标变量
    :param features: 特征列表
    :return: None
    """
    bins_process = toad.transform.Combiner()
    bins_process.fit(df[features], y=df[target], method='chi', min_samples=0.05, empty_separate=False)
    for col in features:
        toad.plot.bin_plot(bins_process.transform(df[[col,target]],labels=True),x=col,target=target)


def sc_iv_plot(df, target, features, bins_dic=None):
    """
    绘制IV值图,相比toad_iv_plot更美观一点
    :param df: pandas DataFrame
    :param target: 目标变量
    :param features: 特征列表
    :param bins_dic: 自定义分箱字典
    :return: None
    """
    if bins_dic is None:
        bins_process = toad.transform.Combiner()
        bins_process.fit(df[features], y=df[target], method='chi', min_samples=0.05, empty_separate=False)
        bins_dic = bins_process.export()
    else:
        bins_dic = {col: bins for col, bins in bins_dic.items() if col in features}
    
    bins_adj = sc.woebin(df[features + [target]], y=target, breaks_list=bins_dic)
    sc.woebin_plot(bins_adj)


def get_bins(df, target, features, bins_method='chi', min_samples=0.05, empty_separate=False):
    """
    获取分箱字典
    :param df: pandas DataFrame
    :param target: 目标变量
    :param features: 特征列表
    :param bins_method: 分箱方法
    :param min_samples: 最小样本数
    :param empty_separate: 是否空值分箱
    :return: bins_dic
    """
    bins_process = toad.transform.Combiner()
    bins_process.fit(df[features], y=df[target], method=bins_method, min_samples=min_samples, empty_separate=empty_separate)
    bins_dic = bins_process.export()
    return bins_dic


def get_roc_ks(actual, predicted, sample_weight):
    """
    计算ROC值和KS值
    :param actual: 真实标签
    :param predicted: 预测值
    :return: ROC值和KS值
    """

    fpr, tpr, thresh_lr = roc_curve(actual, predicted, pos_label=1)  # pos_label=1表示1为响应，0为未响应
    ks_value = max(abs(fpr - tpr))
    auc_value = roc_auc_score(actual, predicted, sample_weight=sample_weight)

    return auc_value, ks_value


def plot_roc_ks(actual, predicted, figsize=(7, 7)):
    """
    绘制ROC曲线和KS曲线
    :param actual: 真实标签
    :param predicted: 预测值
    :return: None
    :示例
    >>> actual = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]
    >>> predicted = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.9,0.8,0.7,0.6,0.5,0.4,0.3]
    >>> plot_roc_ks(actual, predicted)
    """
    # 计算ROC曲线和KS值
    fpr, tpr, thresh_lr = roc_curve(actual, predicted, pos_label=1)  # pos_label=1表示1为响应，0为未响应
    ks_value = max(abs(fpr - tpr))
    auc_value = roc_auc_score(actual, predicted)

    print('AUC = {:.4f}'.format(auc_value))
    print('KS = {:.4f}'.format(ks_value))

    # 绘制ROC曲线
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label='ROC curve (AUC = {:.4f})'.format(auc_value))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='upper left')
    plt.show()

    # 绘制KS曲线
    # 找出 tpr 和 fpr 大于 0 的阈值索引
    valid_indices = np.where(np.logical_or(tpr > 0, fpr > 0))[0]

    # 使用这些索引更新 thresh_lr, tpr, fpr
    thresh_lr = thresh_lr[valid_indices]
    tpr = tpr[valid_indices]
    fpr = fpr[valid_indices]

    plt.figure(figsize=figsize)
    plt.plot(thresh_lr, tpr, label='Bad')
    plt.plot(thresh_lr, fpr, label='Good')
    plt.plot(thresh_lr, tpr - fpr, label='KS = {:.4f}\nbest threshold = {:.4f}'.format(ks_value, thresh_lr[np.argmax(tpr - fpr)]))
    
    # 标记KS值
    x = np.argwhere(abs(fpr - tpr) == ks_value)[0, 0]
    plt.plot((thresh_lr[x], thresh_lr[x]), (0, ks_value), color='r', marker='o', markerfacecolor='r', markersize=3)
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('KS Curve')
    plt.legend(loc='upper right')
    plt.show()