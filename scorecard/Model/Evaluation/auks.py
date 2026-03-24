
from sklearn.metrics import roc_curve, auc, recall_score, precision_score


def auks(y, y_pred):
    fpr_off, tpr_off, _ = roc_curve(y, y_pred)
    ks = fpr_off - tpr_off
    return abs(sum(ks))
