#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  :
    @Time    : 2022/2/14
    @Use     : Max Divergence
"""

import pandas as pd
from qpsolvers import solve_qp
from qpsolvers import dense_solvers
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.linear_model import LogisticRegression
from Model.evaluation.f2_score import f2_score


def show(model, ft_lst_bin, dep, datasets):
    from matplotlib import pyplot as plt
    res = pd.DataFrame()
    for key, value in datasets.items():
        y_pred = model.predict_proba(value[ft_lst_bin])[:, 1]
        fpr_, tpr_, _ = roc_curve(value[dep], y_pred)
        ks = abs(fpr_ - tpr_).max()
        plt.plot(fpr_, tpr_, label=key)
        f2, th, precision, _ = f2_score(value[dep], y_pred)
        res = res.append({'datasets': key,
                          'auc': auc(fpr_, tpr_),
                          'presicion': precision,
                          'ks': ks,
                          'f2': f2,
                          'th': th},
                         ignore_index=True)
    print(res[['datasets', 'auc', 'ks', 'presicion', 'f2', 'th']])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

    return res[['datasets', 'auc', 'ks', 'presicion', 'f2', 'th']]


class MaxDivergence(object):
    def __init__(self, data_bin, ft_lst, dep, constraints=None, P=None, q=None, G=None, h=None, A=None, b=None,
                 lb=None, ub=None, delta=None, penalty=None):

        constraints0 = {
            'Centering_constraints': 1,
            'No_Information_constraints': None,
            'Cross_restrictions': None,  # 需要直接把矩阵放进去
            'Monotonic_constraints': None,  # {'age': 1},  # 需要限制单调性的放dict里，1代表递增，-1递减
            'Other_constraints': None}  # 需要直接把矩阵放进去

        self.constraints = constraints if constraints is not None else constraints0
        self.ft_lst = ft_lst
        self.dep = dep
        self.data_all, self.data, self.ft_lst_bin = self.create_data(data_bin)
        # print("number of ft_lst_bin: {}".format(len(self.ft_lst_bin)))
        self.XdG = self.data[self.data[self.dep] == 0][self.ft_lst_bin]
        self.XdB = self.data[self.data[self.dep] == 1][self.ft_lst_bin]

        self.C = None
        self.penalty = penalty if penalty is not None else 0.001
        self.P = P if P is not None else self.create_P()
        self.q = q if q is not None else self.create_q()
        self.delta = delta if delta is not None else self.get_max_divergence()

        self.G = G if G is not None else self.create_G()
        self.h = h if h is not None else self.create_h()
        self.A = A if A is not None else self.create_A()
        self.b = b if b is not None else self.create_b()
        self.lb = lb if lb is not None else self.create_lb()
        self.ub = ub if ub is not None else self.create_ub()
        self.beta = None
        self.solution = None
        self.coef_ = None

        self.feature_importances_ = None

        matrix_char_lst = ['P', 'q', 'G', 'h', 'A', 'b', 'lb', 'ub']
        matrix_lst = [self.P, self.q, self.G, self.h, self.A, self.b, self.lb, self.ub]
        for i in range(len(matrix_char_lst)):
            print("Shape of matrix {0}: {1}".format(matrix_char_lst[i], matrix_lst[i].shape))
        print("Value of delta {}".format(self.delta))

    def create_data(self, df):
        ft_lst_bin = []
        for i in self.ft_lst:
            lst = sorted(df[i].unique())
            print(lst)
            for ft in lst:
                df[i + '_' + str(ft)] = df.apply(lambda x: 1 if x[i] == ft else 0, axis=1)
                ft_lst_bin.append(i + '_' + str(ft))
        return df, df[df['target'] == 'train'], ft_lst_bin

    def get_max_divergence(self):
        model = LogisticRegression(C=0.1,
                                   max_iter=2000,
                                   class_weight='balanced',
                                   penalty='l2',
                                   solver='liblinear',
                                   verbose=0,
                                   n_jobs=1)

        model.fit(self.data[self.ft_lst_bin], self.data[self.dep], sample_weight=self.data['weight'])
        d = np.mean(self.XdG) - np.mean(self.XdB)

        return np.dot(
            d, model.coef_[0]) * np.dot(d, model.coef_[0]) / np.dot(np.dot(model.coef_[0].T, self.C), model.coef_[0].T)

    def create_C(self):
        xdg = self.XdG.values
        xdb = self.XdB.values
        self.C = (np.cov(xdg.T) + np.cov(xdb.T)) / 2 + (self.penalty / len(self.ft_lst_bin)) * np.eye(len(self.ft_lst_bin))

    def create_P(self):
        self.create_C()
        return 2 * self.C

    def create_q(self):
        return np.zeros((len(self.P)))

    def create_G(self):
        res = list()

        char_num = [len(self.data[i].unique()) for i in self.ft_lst]
        high = [sum(char_num[:i]) for i in range(1, len(char_num) + 1)]
        low = [0] + [high[i - 1] for i in range(1, len(self.ft_lst))]

        if self.constraints['Monotonic_constraints']:
            Ap = []
            for k, v in self.constraints['Monotonic_constraints'].items():
                idx = self.ft_lst.index(k)
                array = np.zeros((char_num[idx], len(self.ft_lst_bin)))
                if v > 0:
                    for m in range(char_num[idx] - 1):
                        array[m, low[idx] + m], array[m, low[idx] + m + 1] = 1, -1
                else:
                    for m in range(char_num[idx] - 1):
                        array[m, low[idx] + m], array[m, low[idx] + m + 1] = -1, 1
                Ap.append(array)

            res += Ap

        if self.constraints['Other_constraints']:
            res.append(self.constraints['Other_constraints'])

        if self.constraints['Other_constraints'] is None and self.constraints['Monotonic_constraints'] is None:
            res.append(np.zeros((len(self.ft_lst_bin), len(self.ft_lst_bin))))

        return np.vstack(res)

    def create_h(self):
        return np.zeros((len(self.G)))

    def create_Ac(self):
        res = []

        char_num = [len(self.data[i].unique()) for i in self.ft_lst]
        high = [sum(char_num[:i]) for i in range(1, len(char_num) + 1)]
        low = [0] + [high[i - 1] for i in range(1, len(self.ft_lst))]

        if self.constraints['Centering_constraints']:
            Ac = np.zeros((len(self.ft_lst), len(self.ft_lst_bin)))
            error = np.mean(self.XdG) + np.mean(self.XdB)
            for i in range(len(self.ft_lst)):
                for k in range(int(low[i]), int(high[i])):
                    Ac[i, k] = error[k]
            res.append(Ac)

        if self.constraints['No_Information_constraints']:
            Ac = np.zeros((len(self.ft_lst), len(self.ft_lst_bin)))
            for i in range(len(self.ft_lst)):
                Ac[i, high[i] - 1] = 1
            res.append(Ac)

        if self.constraints['Cross_restrictions']:
            res.append(self.constraints['Cross_restrictions'])

        if self.constraints['Centering_constraints'] is None \
                and self.constraints['No_Information_constraints'] is None \
                and self.constraints['Cross_restrictions'] is None:
            res.append(np.zeros((len(self.ft_lst_bin), len(self.ft_lst_bin))))

        return np.vstack(res)

    def create_A(self):
        d = np.mean(self.XdG) - np.mean(self.XdB)
        Ac = self.create_Ac()
        return np.vstack((np.array(d).T, Ac))

    def create_b(self):
        return np.hstack((np.array([self.delta]).T, np.zeros((self.A.shape[0]) - 1).T))

    def create_lb(self):
        return -float('inf') * np.ones((len(self.ft_lst_bin)))

    def create_ub(self):
        return float('inf') * np.ones((len(self.ft_lst_bin)))

    def fit(self, solver=dense_solvers[2]):
        # self.solution = solve_qp(self.P, self.q, self.G, self.h, self.A, self.b, self.lb, self.ub, solver=solver)
        try:
            self.solution = solve_qp(self.P, self.q, self.G, self.h, self.A, self.b, self.lb, self.ub, solver=solver)
        except Exception as e:
            print("ERROR：%s" % e)
        finally:
            if self.solution is not None:
                print("QP solution: x = {}".format(self.solution))
            else:
                print('There is no solution feasible.')

    def predict_proba(self, data):
        if self.solution is not None:
            proba = 1.0 / (1.0 + np.exp(np.dot(data.values, self.convert_to_woe_scale())))
            return np.array([1 - proba, proba]).T
        else:
            print('There is no solution feasible.')

    def convert_to_woe_scale(self):
        scrldG = np.dot(self.XdG, self.solution)
        scrldB = np.dot(self.XdB, self.solution)
        self.beta = 2 * (np.mean(scrldG) - np.mean(scrldB)) / (np.cov(scrldG.T) + np.cov(scrldB.T))
        self.coef_ = self.beta * self.solution
        self.feature_importances_ = self.coef_
        return self.coef_

    def feature_importances(self):
        return pd.DataFrame({'var_names': self.ft_lst_bin, 'score': self.coef_})


# def transfer2score(score, min_score=350, max_score=850):
#     min_, max_ = min(score), max(score)
#     return [min_score + (i - min_) / (max_ - min_) * (max_score - min_score) for i in score]
