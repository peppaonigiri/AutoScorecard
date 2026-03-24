
from sklearn.metrics import roc_curve, auc, recall_score, roc_auc_score, fbeta_score, precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support, precision_score
import numpy as np


def f2_score(y, y_pred, weight):
    precision, recall, thresholds = precision_recall_curve(y, y_pred, sample_weight=weight)
    f2score = [5 * recall[i] * precision[i] / (4 * precision[i] + recall[i] + 0.000001) for i in range(len(precision))]
    return np.max(f2score), \
           thresholds[list(f2score).index(max(f2score))], \
           precision[[list(f2score).index(max(f2score))]][0], \
           recall[[list(f2score).index(max(f2score))]][0]


def f2_score_th(y, y_pred, weight, th=0.1):
    y_pred_copy = y_pred.copy()
    y_pred_copy[y_pred_copy >= th] = 1
    y_pred_copy[y_pred_copy < th] = 0
    recall = recall_score(y, y_pred_copy, sample_weight=weight)
    precision = precision_score(y, y_pred_copy, sample_weight=weight)
    return 5 * recall * precision / (4 * precision + recall)