from sklearn.preprocessing import PolynomialFeatures


# 1. 四则运算
def sum(data, fl1, fl2, new):
    data.loc[:, new] = data[fl1] + data[fl2] + 1


def multi(data, fl1, fl2, new):
    data.loc[:, new] = data[fl1] * data[fl2]


def der(data, fl1, fl2, new):
    data.loc[:, new] = data[fl1] / (data[fl2] + 0.00001)


def min(data, fl1, fl2, new):
    data.loc[:, new] = data[fl1] - data[fl2]


# 2. 多项式
def polynomial(data, feature_list):
    poly = PolynomialFeatures(degree=len(feature_list),  interaction_only=False, include_bias=True)
    poly.fit_transform(data[feature_list])


# 4. 借助条件去判断获取组合特征
def logistic(data, fl1, fl2, new):
    data.loc[:, new] = (data[fl1] == 0) & (data[fl2] == 0)