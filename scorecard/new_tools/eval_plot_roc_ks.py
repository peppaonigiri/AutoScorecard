import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score

def get_roc_ks(y_true, y_pred, pos_label=1):
    """
    计算ROC值和KS值
    :param actual: 真实标签
    :param predicted: 预测值
    :param pos_label: 响应标签
    :return: ROC值和KS值
    """

    fpr, tpr, thresh_lr = roc_curve(y_true, y_pred, pos_label=pos_label)  # pos_label=1表示1为响应，0为未响应
    ks_value = max(abs(fpr - tpr))
    
    if len(set(y_true)) > 1:
        auc_value = roc_auc_score(y_true, y_pred)
    else:
        auc_value = None

    return auc_value, ks_value


def dataset_eval(df, label='label', proba='paroba', target='target', pos_label=1):
    """
    对数据集进行ROC和KS评估，并返回结果
    :param df: 包含标签、预测值、目标列的DataFrame
    :param label: 标签列名
    :param proba: 预测值列名
    :param target: 目标列名
    :param pos_label: 响应标签
    :return: 包含ROC和KS值和最低5%和最高5%的lift的DataFrame
    """
    df_list = []
    for target_name in df[target].unique():
        temp_dic = {}
        y_true = df[df[target] == target_name][label].values
        y_pred = df[df[target] == target_name][proba].values
        # 计算最低5%和最高5%的lift值
        sub_5 = np.percentile(y_pred, 5)
        top_5 = np.percentile(y_pred, 95)
        sub_5_rate = y_true[y_pred <= sub_5].sum() / len(y_true[y_pred <= sub_5])
        top_5_rate = y_true[y_pred >= top_5].sum() / len(y_true[y_pred >= top_5])
        all_rate = y_true.sum() / len(y_true)
        sub_5_lift = sub_5_rate / all_rate
        top_5_lift = top_5_rate / all_rate
        # 计算ROC值和KS值
        auc_value, ks_value = get_roc_ks(y_true, y_pred, pos_label)
        # 转化为DataFrame并加入列表
        temp_dic['dataset'] = target_name
        temp_dic['auc'] = auc_value
        temp_dic['ks'] = ks_value
        temp_dic['sub_5_lift'] = sub_5_lift
        temp_dic['top_5_lift'] = top_5_lift
        temp_df = pd.DataFrame(temp_dic, index=[0])
        df_list.append(temp_df)
    result_df = pd.concat(df_list, axis=0, ignore_index=True)
    return result_df


def plot_roc(y_true, y_pred, figsize=(7, 7)):
    """
    绘制ROC曲线
    :param actual: 真实标签
    :param predicted: 预测值
    :return: None
    :示例
    >>> actual = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]
    >>> predicted = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.9,0.8,0.7,0.6,0.5,0.4,0.3]
    >>> plot_roc(y_true, y_pred)
    """
    # 计算ROC曲线和KS值
    fpr, tpr, thresh_lr = roc_curve(y_true, y_pred, pos_label=1)  # pos_label=1表示1为响应，0为未响应
    ks_value = max(abs(fpr - tpr))
    auc_value = roc_auc_score(y_true, y_pred)

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


def plot_ks(labels, pred_probs, figsize=(7, 7)):
    """
    计算KS值并绘制KS图

    参数:
    labels (list or array): 真实标签，包含0和1，1为响应标签
    pred_probs (list or array): 预测分数或概率，如果该值与响应标签正相关，KS是负的
    """
    # 将输入转换为DataFrame
    data = pd.DataFrame({'label': labels, 'pred_prob': pred_probs})
    
    # 按预测分数排序
    data = data.sort_values(by='pred_prob')
    
    # 计算正负样本的累积分布
    data['cum_pos_rate'] = data['label'].cumsum() / data['label'].sum()
    data['cum_neg_rate'] = (1 - data['label']).cumsum() / (1 - data['label']).sum()
    
    # 计算KS值
    data['ks'] = (data['cum_pos_rate'] - data['cum_neg_rate']).abs()
    ks_value = data['ks'].abs().max()
    
    # 找到KS值对应的预测分数
    ks_prob = data.loc[data['ks'].abs().idxmax(), 'pred_prob']
    
    # 绘制KS图
    plt.figure(figsize=figsize)
    plt.plot(data['pred_prob'], data['cum_pos_rate'], label='Bad')
    plt.plot(data['pred_prob'], data['cum_neg_rate'], label='good')
    plt.plot(data['pred_prob'], data['ks'], label=f'KS = {ks_value:.4f}\nbest threshold = {ks_prob:.4f}', linestyle='--')
    plt.plot((ks_prob, ks_prob), (0, ks_value), color='r', marker='o', markerfacecolor='r', markersize=3)
    plt.legend()
    plt.xlabel('Prediction')
    plt.ylabel('Cumulative Rate')
    plt.title('KS Curve')
    plt.show()