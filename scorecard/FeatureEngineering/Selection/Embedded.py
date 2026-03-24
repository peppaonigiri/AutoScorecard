
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import pandas as pd

# 使用SelectFromModel选择特征 (Variable selection using SelectFromModel)
# 基于L1的特征选择 (L1-based feature selection)
# 使用L1范数作为惩罚项的线性模型(Linear models)会得到稀疏解：大部分特征对应的系数为0。当你希望减少特征的维度以用于其它分类器时，
# 可以通过 feature_selection.SelectFromModel 来选择不为0的系数。
# 特别指出，常用于此目的的稀疏预测模型有 linear_model.Lasso（回归）， linear_model.LogisticRegression 和 svm.LinearSVC（分类）


def L1(df, dep, ex_lst):
    lst = [i for i in list(df.columns) if i not in ex_lst]
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(df[lst], df[dep])
    model = SelectFromModel(lsvc, prefit=True)
    X_embed = model.transform(df[lst])
    return pd.DataFrame({'var_names': list(X_embed.columns), 'coef': lsvc.coef_[0]}).sort_values(by=['coef'], ascending=False)


def SVM(data, feature_list, label):
    X = data[feature_list]
    y = data[label]
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)

    model = SelectFromModel(lsvc, prefit=True)
    feature = pd.DataFrame(
        {'name': model.booster_.feature_name(),
         'importance': model.feature_importances_}).sort_values(by=['importance'], ascending=False)