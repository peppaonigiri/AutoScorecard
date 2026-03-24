#!/usr/bin/env python
# -*- coding: utf-8 -*-

from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler


# 1. 朴素随机过采样 replacement = true 有放回采样 可以实现自助法(boostrap)抽样
def RandomOverSampler(x, y):
    ros = RandomOverSampler(random_state=0, replacement = True)
    X_resampled, y_resampled = ros.fit_sample(x, y)

    sorted(Counter(y_resampled).items())


# 2. SMOTE
def OverSamplerSMOTE(x, y):
    X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(x, y)
    sorted(Counter(y_resampled_smote).items())


# 3. adasyn
def OverSamplerADASYN(x, y):
    X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_sample(x, y)
    sorted(Counter(y_resampled_adasyn).items())


# 4. 随机欠采样
def RandomUnderSampler(x, y):
    x_resampled, y_resampled = RandomUnderSampler().fit_sample(x, y)
    sorted(Counter(y_resampled).items())


# 5. 原型生成(prototype generation) 每一个类别的样本都会用K-Means算法的中心点来进行合成, 而不是随机从原始样本进行抽取.
def KmeansUnderSampler(x, y):
    X_resampled, y_resampled = ClusterCentroids(random_state=0).fit_sample(x, y)
    sorted(Counter(y_resampled).items())



