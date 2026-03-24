
import numpy as np
import pandas as pd
import math


def proba2score(prob, pdo=30, rate=2, base_odds=35, base_score=750):
    factor = pdo / np.log(rate)
    offset = base_score - factor * np.log(base_odds)
    return factor * (np.log(1 - prob) - np.log(prob)) + offset


# def woe2score(woe, weight, intercept_, n_features_, pdo, base_score, base_odds, rate):
#     factor = pdo / np.log(rate)
#     offset = base_score - factor * np.log(base_odds)
#     b = (offset - factor * intercept_) / n_features_
#     res = []
#     for i in woe:
#         res.append(-factor * weight * i + b)
#     return res
#
#
# def woe_to_score(ft_lst, lr_coef, transer, intercept_, n_features_, pdo, base_score, base_odds, rate):
#     rules = {}
#     for idx, key in enumerate(ft_lst):
#         weight = lr_coef[idx]
#         woe = list(transer[key])
#         rules[key] = woe2score(woe, weight, intercept_, n_features_, pdo, base_score, base_odds, rate)
#     #             'bins': combiner[key],
#     #                       'woes': woe,
#     #                       'weight': weight,
#     #                       'scores': woe2score(woe, weight,  pdo, base_score,base_odds, rate, intercept_, n_features_)}
#
#     return rules

def woe2score(woe, weight, intercept_, n_features_, pdo, base_score, base_odds, rate):
    factor = pdo / np.log(rate)
    offset = base_score - factor * np.log(base_odds)
    b = (offset - factor * intercept_) / n_features_
    res = []
    for i in woe:
        res.append(-factor * weight * i + b)
    return res


def woe_to_score(ft_lst, lr_coef, transer, intercept_, n_features_, pdo, rate, base_odds, base_score):
    rules = {}
    for idx, key in enumerate(ft_lst):
        weight = lr_coef[idx]
        woe = list(transer[key]['woe'])
        rules[key] = woe2score(woe, weight, intercept_, n_features_, pdo, base_score, base_odds, rate)
    return rules


def get_dicts(ft_used, lr_coef, intercept_, combiner, transer, pdo=30, rate=2, base_odds=35, base_score=750):
    dic_combiner, dic_transer, dic_score = {}, {}, {}
    for i in ft_used:
        dic_combiner[i] = list(combiner[i])
        dic_transer[i] = list(transer[i]['woe'])
    dic_score = woe_to_score(ft_used, lr_coef, transer, intercept_, len(ft_used), pdo=pdo, base_score=base_score,
                             base_odds=base_odds, rate=rate)
    return dic_combiner, dic_transer, dic_score


def lr(data, intercept, coef):
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))
    return sigmoid(np.dot(data, coef) + intercept)


def featute2score(data, intercept, coef, ft_lst, combiner, transer):
    data.fillna(-999, inplace=True)
    data = data.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G'], [1, 2, 3, 4, 5, 6, 7])
    dic = {}
    for i in range(34):
        dic['L' + str(i)] = i
    data = data.replace(dic)

    for i in ft_lst:
        data[i] = pd.cut(data[i], [float('-inf')] + combiner[i] + [float('inf')], labels=transer[i])
        data[i] = pd.to_numeric(data[i])

    data['proba'] = lr(data[ft_lst], intercept, coef)
    data['score'] = proba2score(data['proba'])
    return data


def score_lr(feature_list, weight, param):
    xbeta = 0
    for i in range(len(weight) - 1):
        xbeta += feature_list * weight[i]
    xbeta += weight[-1]
    score = param[0] - param[1] * xbeta / math.log(2)
    return score


def Prob2Score(p, basePoint=600, PDO=20, odds0=1/60):
    from scipy.linalg import solve
    """
    将概率转化成分数且为正整数
    odds0=ratio_bad_good, 当比率为odds0时，评分卡输出分数为基准分
    score公式 = a + b lgodds0
    (1)basePoint = a + b lgodds0
    (2)basePoint + PDO = a + b lg2odds0 odds翻倍，分数增加PDO
    得：b = PDO/lg2, a = basePoint - b * lgodds0  # 此处PDO为-20才对，PDO为正，a公式也为正
    """
    a = np.array([[1, -math.log(odds0)], [1, -math.log(odds0*2)]])
    b = np.array([basePoint, (basePoint-PDO)])
    x = solve(a, b) # 1/60,481.89, 28.85
    # print(x)
    min_p = 1e-6
    max_p = 1 - min_p
    if p >= max_p:
        p = max_p
    elif p <= min_p:
        p = min_p
    lgodds = np.log(p / (1 - p))
    # 客户越坏,p越大，odds越大，log odds越大（例如1：100至1：10），分数越低
    return int(x[0] - x[1] * lgodds)


def score_lr(feature_list, weight, param):
    xbeta = 0
    for i in range(len(weight) - 1):
        xbeta += feature_list * weight[i]
    xbeta += weight[-1]
    score = param[0] - param[1] * xbeta / math.log(2)
    return score


def output(model, output_scores, datasets, uid, dep, var_names):
    for output_score in output_scores:
        tdf = datasets.get(output_score, "")
        if not isinstance(tdf, str):
            f = open("data\%s_score.txt" % output_score, "w")
            f.write("%s\t%s\tscore\n" % (uid, dep))
            UID, X, Y = tdf[uid], tdf[var_names], tdf[dep]
            Result = model.predict(X)
            for i in range(tdf.shape[0]):
                f.write("%s\t%s\t%s\n" % (UID[i], Y[i], Result[i]))
            f.close()


def code_generation_java(ft_lst, combiner, dic_score):
    print('# 上线文档')
    print('# step1 将原始文件中的null替换为 -999')
    print('# step2 将原始文件中的["A", "B", "C", "D", "E", "F", "G"]分别替换为[1, 2, 3, 4, 5, 6, 7]')
    print('# step3 将原始文件中的 ["L1","L2", ......, "L34"]分别替换为[1，2，......, 34]上线文档')
    print('# step4 获取每一个特征的值带入以下函数')
    print('# 将输出值与对照文件进行对比，对照列为"score", 整数及小数点后两位相同即可认为分数转换无误')

    print('public class Demo{')

    print('    public static double replace_str(String str){')
    print('        HashMap<String, Integer> mapreplaces = new HashMap<>();')
    print('        for (int i = 0; i < 34; i++) mapreplaces.put("L" + i, i);')
    print('        mapreplaces.put("A", 1);')
    print('        mapreplaces.put("B", 2);')
    print('        mapreplaces.put("C", 3);')
    print('        mapreplaces.put("D", 4);')
    print('        mapreplaces.put("E", 5);')
    print('        mapreplaces.put("F", 6);')
    print('        mapreplaces.put("G", 7);')
    print('        if(str == null){return -999;}')
    print('        else{replaceStr(str, mapreplaces);')
    print('             return str;')
    print('        }')
    print('    }')
    print(' ')
    print('    public static double replace_num(double num){')
    print('        if(num == null){return -999;}')
    print('    }')
    print(' ')
    r = ''
    for i in ft_lst:
        r += 'double ' + i + ', '
    print('    public static double feature2score(', r[:-2], '){')
    print('        double score = 0.0;')
    for i in ft_lst:

        print('')
        lst = [float('-inf')] + combiner[i] + [float('inf')]
        print('        if( ', i, ' < ', lst[1], '){ score += ', dic_score[i][0], ';}')
        for j in range(1, len(lst) - 2):
            print('        else if (', lst[j], ' <= ', i, '&&', i, ' < ', lst[j + 1], '){score += ', dic_score[i][j],
                  ';}')
        print('        else ', '{ score += ', dic_score[i][-1], ';}')
    print('')
    print('        return score;')
    print('    }')
    print('}')


def code_generation_python(ft_lst, combiner, dic_score):
    r = ''
    for i in ft_lst:
        r += i + ', '

    print('def feature2score(array):')
    print('    score = 0')
    m = 0
    for i in ft_lst:

        print('')
        lst = [float('-inf')] + combiner[i] + [float('inf')]
        print('    if array[', m, ']', ' <= ', lst[1], ': score += ', dic_score[i][0])
        for j in range(1, len(lst) - 2):
            print('    elif', lst[j], ' <= ', 'array[', m, ']', ' < ', lst[j + 1], ': score += ', dic_score[i][j])
        print('    else ', ': score += ', dic_score[i][-1])
        m += 1
    print('')
    print('    return score')


def get_Score(data, feature2score):
    data.fillna(-999, inplace=True)
    data = data.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G'], [1, 2, 3, 4, 5, 6, 7])
    dic = {}
    for i in range(34):
        dic['L' + str(i)] = i
    data = data.replace(dic)
    data['score'] = 0

    for i in range(data.shape[0]):
        data.loc[i, 'score'] = feature2score(data.loc[i, :].values)
    return data