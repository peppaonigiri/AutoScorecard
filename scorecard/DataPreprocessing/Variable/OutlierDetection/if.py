
from pyod.models.iforest import IForest
import numpy as np


# Isolation Forest
def isolation_forest(dev, val, var_names, x_train, threshold=0.7):
    clf = IForest(behaviour='new', bootstrap=False, contamination=0.1, max_features=1.0,
                  max_samples='auto', n_estimators=500, n_jobs=-1, random_state=None, verbose=0)
    # clf.fit(val)
    clf.fit(dev)
    dev_pred = clf.predict_proba(val, method='linear')[:, 1]
    val_pred = clf.predict_proba(dev, method='linear')[:, 1]

    dev['out_pred'] = dev_pred
    val['out_pred'] = val_pred

    x_train = dev[x_train.out_pred < threshold][var_names]
    y_train = dev[x_train.out_pred < threshold]['dep']
    x_test = val[x_train.out_pred < threshold][var_names]
    y_test = val[x_train.out_pred < threshold]['dep']

    # 预测值大于0.7的样本值为1，小于0.7的为0
    x_train['for_pred'] = np.where(x_train.out_pred > threshold, 1, 0)
    x_train.for_pred.groupby(x_train.obs_mth).sum() / x_train.for_pred.groupby(x_train.obs_mth).count()
    return x_train, y_train, x_test, y_test
