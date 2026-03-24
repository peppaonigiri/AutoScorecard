#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : F. Li
    @Time    : 2019/7/24 23:23
    @Use     : 缺失值处理
"""

import pandas as pd
from sklearn.impute import SimpleImputer
import re

data = pd.DataFrame()
var_names = []


# 2. 简单删除
def drop(df_miss_row, df_miss_col, lower_row,  supper_row, lower_col, supper_col):
    slct_lst_raw = list(df_miss_row[lower_row < df_miss_row['percent'] < supper_row])
    slct_lst_col = list(df_miss_col[lower_col < df_miss_col['percent'] < supper_col])
    return slct_lst_raw, slct_lst_col


# 2. 添加是否为Nan作为一个新的特征
def add_is_NaN(df, df_miss_col, lower, supper):
    slct_lst_col = list(df_miss_col[lower < df_miss_col['percent'] < supper])
    fd = pd.DataFrame()
    for ft in slct_lst_col:
        fd[ft + '_is_NaN'] = df[ft].isnull()
    return fd

# ========================================================================================================================
# pandas fillna
# - value : 变量, 字典, Series, or DataFrame
# - inplace参数的取值：True 直接修改原对象、False 创建一个副本，修改副本，原对象不变（缺省默认)
# - method参数的取值 ： {‘pad’, ‘ffill’,‘backfill’, ‘bfill’, None}, default None
#   pad/ffill：用前一个非缺失值去填充该缺失值
#   backfill/bfill：用下一个非缺失值填充该缺失值
#   None：指定一个值去替换缺失值（缺省默认这种方式）
# - limit参数：限制填充个数
# - axis参数：修改填充方向 1 行   0 列
# - downcast : dict, 默认是 None； 如果可能的话，把 item->dtype 的字典将尝试向下转换为适当的相等类型的字符串（例如，如果可能的话，从float64到int64）
#
# pandas isna
# - numpy里边查找NaN值的话，就用np.isnan()。
# - pandas里边查找NaN值的话，要么.isna()，要么.isnull()。
#
# pandas dropna
# - axis: default 0指行,1为列
# - how: {‘any’, ‘all’}, default ‘any’指带缺失值的所有行;'all’指清除全是缺失值的
# - thresh: int,保留含有int个非空值的行
# - subset: 对特定的列进行缺失值删除处理
# - inplace: 这个很常见,True表示直接在原数据上更改
# ========================================================================================================================


# 3.单独编码 如所有的空值都用“unknown”填充。这样将形成另一个概念，可能导致严重的数据偏离，一般不使用。
data[var_names].fillna(-999)  # 999

# 4. 用0填充
data[var_names].fillna(0.0, inplace=True)

# 5. 用平均值填充 数值型
data[var_names].fillna(value=data[var_names].mean(), inplace=True)

# ======================================== Imputer 填充 =================================================================
# imp = Imputer(missing_values='NaN',strategy='median',axis=0,verbose=0,copy=True)
# trainset1 = imp.fit_transform(trainset)
# missing_values：缺失值，可以为整数或NaN(缺失值numpy.nan用字符串‘NaN’表示)，默认为NaN
# strategy：替换策略，字符串，默认用均值‘mean’替换。mean均值；median中位数；most_frequent众数
# axis：指定轴数，默认axis=0代表列，axis=1代表行
# copy：设置为True代表不在原数据集上修改，设置为False时，就地修改
# ======================================================================================================================

# 6. 用中位数填充 数值型
imp = SimpleImputer(strategy='median')
data[var_names] = imp.fit_transform(data[var_names])

# data = data.apply(lambda x: x.fillna(x.median()))

# 7. 用众数填充， 非数值型
imp = SimpleImputer(strategy='most_frequent')
data[var_names] = imp.fit_transform(data[var_names])

# data = data.apply(lambda x: x.fillna(x.mode()))

# 8. 用每组的均值填充缺失值
data[var_names] = data.groupby("label").transform(lambda x: x.fillna(x.mean()))
# data = data.apply(lambda x: x.fillna(x.mean()))


# 9、用bad rate最近的箱的中位值填充
def fillna_badrate_median(X_df,var_woe):
    """
    X_df: 待填充的dataframe数据
    var_woe: 含分箱的bin, badprob的表格数据
    return: 返回缺失值填充好的df
    """
    for var in X_df.columns:
        df_woe = var_woe.loc[var]
        df_woe['diff'] = 100
        for i in range(1, df_woe.shape[0]):
            df_woe.loc[i, 'diff'] = abs(df_woe.loc[i, 'badprob'] - df_woe[df_woe['bin'] == 'missing'].badprob.loc[0])
        temp = re.split(r'[\[,\)]', df_woe[df_woe['diff'] == df_woe['diff'].min()].bin.values[0])
        appro_bin = [i for i in temp if i != '']
        med = X_df.loc[(X_df[var] >= float(appro_bin[0])) & (X_df[var] < float(appro_bin[1]))][var].median()
        X_df[var].fillna(med, inplace=True)
    return X_df


# 10、使用随机森林填补一个特征的缺失值的函数
def fill_missing_rf(X,y,to_fill):
    """
    X：要填补的特征矩阵
    y：完整的，没有缺失值的标签
    to_fill：字符串，要填补的那一列的名称
    """
    #构建我们的新特征矩阵和新标签
    df = X.copy()
    fill = df.loc[:, to_fill]
    df = pd.concat([df.loc[:, df.columns != to_fill], pd.DataFrame(y)], axis=1)

    # 找出我们的训练集和测试集
    Ytrain = fill[fill.notnull()]  # 特征不缺失的值
    Ytest = fill[fill.isnull()]  # 特征缺失的值
    Xtrain = df.iloc[Ytrain.index, :]  # 特征不缺失的值对应其他n-1个特征+本来的标签
    Xtest = df.iloc[Ytest.index, :]  # 特征缺失的值对应其他n-1个特征+本来的标签

    #用随机森林回归来填补缺失值
    from sklearn.ensemble import RandomForestRegressor as rfr
    rfr = rfr(n_estimators=100)
    rfr = rfr.fit(Xtrain, Ytrain)
    Ypredict = rfr.predict(Xtest)

    return Ypredict
# ======================================================================================================================
# 对于一个特征缺失，其他特征也有缺失值
# 遍历所有的特征，从缺失最少的特征开始填补（因为填补缺失值最少的特征所需要的准确信息最少）。填补一个特征时，先将其他特征的缺失值用0代题，
# 没完成一次回归预测，就将预测值放到原本的特征矩阵中，再继续填补下一个特征，每一次填补完毕，有缺失值的特征会减少一个，所以每次循环后，
# 需要用0填补的特征就越来越少，当进行到最后一个特征时（这个特征应该时所有特征缺失值最多的），已经没有其他特征需要用0来进行填补，遍历所有的特征后，
# 数据就完整，不再有缺失值。
# ======================================================================================================================














