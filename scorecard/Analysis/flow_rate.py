import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
mpl.rcParams['font.sans-serif'] = ['SimHei']


def flowrate(data_all, col_name, bins=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 365]):
    data_all['overdue_days_bin'] = pd.cut(data_all[col_name], bins)
    t = pd.pivot_table(data_all, values='label', columns='overdue_days_bin', index='backDateTime', aggfunc=['count'])
    t['sum'] = [t.loc[i,:].sum() for i in t.index]
    ft = t.columns.tolist()[:-1]
    for i in ft:
        t[i] = t[i] / t.loc[:, ('sum', '')]
    for i in range(len(t.index)):
        for j in range(len(ft)):
            t.iloc[i, j] = t.iloc[i, j] + t.iloc[i, j+1:-1].sum()

    plt.figure(figsize=(20,8))
    plt.plot(t['count'], label=t['count'].columns.tolist())
    plt.legend()
    plt.margins(0)
    plt.grid()
    plt.subplots_adjust(bottom=0.15)
    plt.title("flowrate")

    plt.show()
    return t


def flowrate_without_label(data_all, col_name, bins=[-1, 0, 1, 2, 3, 365]):
    data_all['overdue_days_bin'] = pd.cut(data_all[col_name], bins)
    t = pd.pivot_table(data_all, values=['label'], columns='overdue_days_bin', index='backDateTime', aggfunc=['count'])
    t['sum'] = [t.loc[i,:].sum() for i in t.index]
    ft = t.columns.tolist()[:-1]
    for i in ft:
        t[i] = t[i] / t.loc[:, ('sum', '')]

    plt.figure(figsize=(20, 8))
    plt.plot(t['count'], label=t['count'].columns.tolist())
    plt.legend()
    plt.margins(0)
    plt.grid()
    plt.subplots_adjust(bottom=0.15)
    plt.title("flowrate")

    plt.show()
    return t
