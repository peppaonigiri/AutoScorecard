import pandas as pd
import numpy as np
from scipy.optimize import root

# 将名义利率、内部收益率、年化利率、年化损失率的计算过程用代码实现如下：

def calcAnnual(mode, A, T, R, vtg):
    # mode: 还款方式，等额本金/等额本息/先息后本/等本等息
    # A: 本金
    # T: 产品期数
    # R: 产品年利率:
    #          等额本金/等额本息/先息后本: 年化利率
    #          等本等息：名义利率，非年化利率
    # vtg: vintage损失率

    r = R / 12  # 月利率

    if mode == '等额本金':
        # 计算每月现金流
        # 每月还款本金
        Principal = [A / T] * T
        # 每月还款利息
        Interest = [(A - A / T * i) * r for i in range(T)]
        # 每月还款金额
        b = [ints + princ for ints, princ in zip(Interest, Principal)]
        # 每月剩余本金
        Rest = [A - princ for princ in np.cumsum(Principal)]

    elif mode == '等额本息':
        # 计算每月现金流
        # 每月还款金额
        b = [A * r * (1 + r) ** T / ((1 + r) ** T - 1)] * T
        # 每月还款利息
        Interest = [(A * r - b[0]) * (1 + r) ** (t - 1) + b[0] for t in range(1, T + 1)]
        # 每月还款本金
        Principal = [bi - ints for bi, ints in zip(b, Interest)]
        # 每月剩余本金
        Rest = [A - princ for princ in np.cumsum(Principal)]

    elif mode == '先息后本':
        # 计算每月现金流
        # 每月还款利息
        Interest = [A * r] * T
        # 每月还款本金
        Principal = [0] * (T - 1) + [A]
        # 每月还款金额
        b = [ints + princ for ints, princ in zip(Interest, Principal)]
        # 每月剩余本金
        Rest = [A - princ for princ in np.cumsum(Principal)]

    elif mode == '等本等息':
        # 计算每月现金流
        # 每月还款利息
        Interest = [A * r] * T
        # 每月还款本金
        Principal = [A / T] * T
        # 每月还款金额
        b = [ints + princ for ints, princ in zip(Interest, Principal)]
        # 每月剩余本金
        Rest = [A - princ for princ in np.cumsum(Principal)]

    # 现金流
    CashFlow = pd.DataFrame({'Period': range(1, T + 1), 'Total': b,
                             'Interest': Interest, 'Principal': Principal,
                             'Rest': Rest})

    # 名义利率APR
    APR = sum(Interest) / A * (12 / T)

    # 内部收益率
    d = root(lambda x: A - sum([bi / (1 + x) ** (i + 1) for i, bi in enumerate(b)]), 0.01)
    IRR = 12 * d['x'][0]

    """
    method1: 借款期限内的年均占用本金法
    """
    # 平均资金占用(年月均余额)
    AveAmt = sum([princ * (i + 1) for i, princ in enumerate(Principal)]) / T
    # 年周转次数
    TurnOvers = 12 / T
    # 年化利率
    AnnR = sum(Interest) / AveAmt * TurnOvers
    # 年化损失率
    AnnLoss = A * vtg / AveAmt * TurnOvers

    """
    method2: 一年内的年均占用本金法
    """
    # 平均资金占用(年月均余额)
    AveAmt2 = sum([princ * (i + 1) for i, princ in enumerate(Principal)]) / 12
    # 年周转次数
    TurnOvers2 = 1.0
    # 年化利率
    AnnR2 = sum(Interest) / AveAmt2 * TurnOvers2
    # 年化损失率
    AnnLoss2 = A * vtg / AveAmt2 * TurnOvers2

    """
    method3: 本金满额占用时间法
    """
    # 本金满额占用时间
    EntireTerm = sum([princ * (i + 1) for i, princ in enumerate(Principal)]) / A
    # 年周转次数
    TurnOvers3 = 12 / EntireTerm
    # 年化利率
    AnnR3 = sum(Interest) / A * TurnOvers3
    # 年化损失率
    AnnLoss3 = A * vtg / A * TurnOvers3

    print('-' * 50)
    print('APR(名义利率): {}'.format(round(APR, 4)))
    print('IRR(内部收益率): {}'.format(round(IRR, 4)))
    print('-' * 50)
    print('method1: 借款期限内的年均占用本金法')
    print('AveAmt(年均占用金额): {}'.format(round(AveAmt, 4)))
    print('TurnOvers(年周转次数): {}'.format(round(TurnOvers, 4)))
    print('AnnR(年化利率): {}'.format(round(AnnR, 4)))
    print('AnnLoss(年化损失率): {}'.format(round(AnnLoss, 4)))
    print('-' * 50)
    print('method2: 一年内的年均占用本金法')
    print('AveAmt(年均占用金额): {}'.format(round(AveAmt2, 4)))
    print('TurnOvers(年周转次数): {}'.format(round(TurnOvers2, 4)))
    print('AnnR(年化利率): {}'.format(round(AnnR2, 4)))
    print('AnnLoss(年化损失率): {}'.format(round(AnnLoss2, 4)))
    print('-' * 50)
    print('method3: 本金满额占用时间法')
    print('EntireTerm(满额占用时间): {}'.format(round(EntireTerm, 4)))
    print('TurnOvers(年周转次数): {}'.format(round(TurnOvers3, 4)))
    print('AnnR(年化利率): {}'.format(round(AnnR3, 4)))
    print('AnnLoss(年化损失率): {}'.format(round(AnnLoss3, 4)))
    print('-' * 50)

    return CashFlow


# （1）等额本金的年化损失率：
calcAnnual(mode='等额本金', A=10000, T=9, R=0.24, vtg=0.06)

# （2）等额本息的年化损失率：
calcAnnual(mode='等额本息', A=10000, T=9, R=0.24, vtg=0.06)

# （3）先息后本的年化损失率：
# 因为先息后本的本金一直是满额占用的，所以在Vintage损失率和期数相同的情况下，其年化损失率是最小的，还可以看到，其名义利率和真实利率是相同的。
calcAnnual(mode='先息后本', A=10000, T=9, R=0.24, vtg=0.06)

# （4）等本等息的年化损失率：
# 等本等息中，按照年均占用方法计算的AnnR并不等于IRR，因此这里计算的AnnR并不是年化利率，年化利率是IRR.
calcAnnual(mode='等本等息', A=10000, T=9, R=0.24, vtg=0.06)
