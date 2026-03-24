import pandas as pd
import math
import numpy as np
import toad

def eva_score(data, bins=20):
    nrows = data.shape[0]
    dataset = data.reset_index(drop=True)
    lis = [(dataset.loc[i, 'score'], dataset.loc[i, 'label']) for i in range(nrows)]
    sorted_lis = sorted(lis, key=lambda x: x[0], reverse=False)
    bin_num = int(nrows / bins + 1)
    bad = sum([1 for (p, y) in sorted_lis if y > 0.5])
    good = sum([1 for (p, y) in sorted_lis if y <= 0.5])
    bad_cnt, good_cnt, cum_bad_rate_random = 0, 0, 0
    total_bad = data['label'].sum()
    total_good = nrows - total_bad

    KS = []
    BAD = []
    GOOD = []
    CUM_BAD = []
    CUM_GOOD = []
    KS_PCTG = []
    BADRATE = []
    LIFT = []
    PCTG = []
    MIN = []
    MAX = []
    CUM_BR = []
    CUM_BRR = []
    BR_CUT = []

    dct_report = {}
    for j in range(bins):
        ds = sorted_lis[j * bin_num: min((j + 1) * bin_num, nrows)]
        dr = sorted_lis[j * bin_num: nrows]
        bad1 = sum([1 for (p, y) in ds if y > 0.5])
        good1 = sum([1 for (p, y) in ds if y <= 0.5])
        bad2 = sum([1 for (p, y) in dr if y > 0.5])
        good2 = sum([1 for (p, y) in dr if y <= 0.5])
        bad_cnt += bad1
        good_cnt += good1
        cum_bad_rate_random += 1 / bins
        ks_pctg = round(math.fabs((bad1 / total_bad) - (good1 / total_good)), 4)
        badrate = round(bad1 / (bad1 + good1), 4)
        ks = round(math.fabs((bad_cnt / bad) - (good_cnt / good)), 4)
        bad_rate_cut = round(bad2 / (bad2 + good2), 4)
        lift = round(bad_cnt / (bad_cnt + good_cnt) / cum_bad_rate_random, 4)
        cum_bd = round(bad_cnt / total_bad, 4)
        KS.append(ks)
        BAD.append(bad1)
        GOOD.append(good1)
        CUM_BAD.append(bad_cnt)
        CUM_GOOD.append(good_cnt)
        KS_PCTG.append(ks_pctg)
        BADRATE.append(badrate)
        PCTG.append(str(100 * j / bins) + '%分位数')
        MIN.append(ds[0][0])
        MAX.append(ds[-1][0])
        LIFT.append(lift)
        CUM_BR.append(cum_bd)
        CUM_BRR.append(cum_bad_rate_random)
        BR_CUT.append(bad_rate_cut)

    dct_report['PCTG'] = PCTG
    dct_report['MIN'] = MIN
    dct_report['MAX'] = MAX
    dct_report['BAD'] = BAD
    dct_report['GOOD'] = GOOD
    dct_report['CUM_BAD'] = CUM_BAD
    dct_report['CUM_GOOD'] = CUM_GOOD
    dct_report['KS_PCTG'] = KS_PCTG
    dct_report['BADRATE'] = BADRATE
    dct_report['CUM_BR'] = CUM_BR
    dct_report['CUM_BRR'] = CUM_BRR
    dct_report['KS'] = KS
    dct_report['BR_CUT'] = BR_CUT
    dct_report['LIFT'] = LIFT

    return pd.DataFrame(dct_report)


def eva_score2(data, bins=20):
    nrows = data.shape[0]
    dataset = data.reset_index(drop=True)
    lis = [(dataset.loc[i, 'score'], dataset.loc[i, 'label']) for i in range(nrows)]
    sorted_lis = sorted(lis, key=lambda x: x[0], reverse=False)
    scores = [(dataset.loc[i, 'score']) for i in range(nrows)]
    bad = sum([1 for (p, y) in sorted_lis if y > 0.5])
    good = sum([1 for (p, y) in sorted_lis if y <= 0.5])
    bad_cnt, good_cnt, cum_bad_rate_random = 0, 0, 0
    total_bad = data['label'].sum()
    total_good = nrows - total_bad

    KS = []
    BAD = []
    GOOD = []
    CUM_BAD = []
    CUM_GOOD = []
    KS_PCTG = []
    BADRATE = []
    LIFT = []
    PCTG = []
    MIN = []
    MAX = []
    CUM_BR = []
    CUM_BRR = []
    BR_CUT = []

    dct_report = {}
    for j in range(bins):
        s_upper = np.percentile(scores, cum_bad_rate_random * 100)
        cum_bad_rate_random += 1 / bins
        s_supper = np.percentile(scores, min(cum_bad_rate_random * 100, 100))
        ds = [i for i in sorted_lis if (s_upper <= i[0] < s_supper)]
        dr = [i for i in sorted_lis if (s_upper <= i[0])]

        bad1 = sum([1 for (p, y) in ds if y > 0.5])
        good1 = sum([1 for (p, y) in ds if y <= 0.5])
        bad2 = sum([1 for (p, y) in dr if y > 0.5])
        good2 = sum([1 for (p, y) in dr if y <= 0.5])

        bad_cnt += bad1
        good_cnt += good1
        ks_pctg = round(math.fabs((bad1 / total_bad) - (good1 / total_good)), 4)
        badrate = round(bad1 / (bad1 + good1 + 1e-9), 4)
        ks = round(math.fabs((bad_cnt / bad) - (good_cnt / good)), 4)
        lift = round(bad_cnt / (bad_cnt + good_cnt) / cum_bad_rate_random, 4)
        cum_bd = round(bad_cnt / total_bad, 4)
        bad_rate_cut = round(bad2 / (bad2 + good2), 4)

        KS.append(ks)
        BAD.append(bad1)
        GOOD.append(good1)
        CUM_BAD.append(bad_cnt)
        CUM_GOOD.append(good_cnt)
        KS_PCTG.append(ks_pctg)
        BADRATE.append(badrate)
        PCTG.append(str(100 * j / bins) + '%分位数')
        MIN.append(s_upper)
        MAX.append(s_supper)
        LIFT.append(lift)
        CUM_BR.append(cum_bd)
        BR_CUT.append(bad_rate_cut)
        CUM_BRR.append(cum_bad_rate_random)

    dct_report['PCTG'] = PCTG
    dct_report['MIN'] = MIN
    dct_report['MAX'] = MAX
    dct_report['BAD'] = BAD
    dct_report['GOOD'] = GOOD
    dct_report['CUM_BAD'] = CUM_BAD
    dct_report['CUM_GOOD'] = CUM_GOOD
    dct_report['KS_PCTG'] = KS_PCTG
    dct_report['BADRATE'] = BADRATE
    dct_report['CUM_BR'] = CUM_BR
    dct_report['CUM_BRR'] = CUM_BRR
    dct_report['KS'] = KS
    dct_report['BR_CUT'] = BR_CUT
    dct_report['LIFT'] = LIFT

    return pd.DataFrame(dct_report)


def eva_score_loan(data, bucket=20, method='step', q=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    score_ks = toad.metrics.KS_bucket(data['score'], data['label'],
                                      bucket=bucket,
                                      method=method,
                                      q=q
                                      )
    num = score_ks.shape[0]
    score_ks['min'] = score_ks.apply(lambda x: int(x['min']), axis=1)
    score_ks.loc[0, 'min'] = -float('inf')
    score_ks.loc[num - 1, 'max'] = float('inf')

    for i in range(num - 1):
        score_ks.loc[i, 'max'] = score_ks.loc[i + 1, 'min']
    tb_min, tb_max = score_ks['min'].tolist(), score_ks['max'].tolist()

    bucket = score_ks['min'].tolist()
    bucket.append(float('inf'))
    print(bucket)
    data['bucket'] = pd.cut(data['score'], bucket, labels=[i for i in range(len(bucket) - 1)],
                            include_lowest=True, right=False)
    score_ks = toad.metrics.KS_bucket(data['score'], data['label'], bucket=data['bucket'])

    score_ks['通过人群整体坏账率'] = score_ks.apply(lambda x: data[data['score'] >= x['min']]['label'].mean(), axis=1)
    score_ks['通过率'] = score_ks.apply(lambda x: data[data['score'] >= x['min']].shape[0] / data.shape[0], axis=1)
    score_ks['odds'] = 1 / score_ks['odds']
    score_ks['min'] = tb_min
    score_ks['max'] = tb_max

    # return score_ks[['min', 'max', 'bad_rate', 'odds', 'cum_bads_prop', 'cum_total_prop', 'total', 'cum_lift',
    # '通过率', '通过人群整体坏账率']]

    return score_ks


def eva_score_marketing(data, bucket=20, method='step', q=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    score_ks = toad.metrics.KS_bucket(data['score'], data['label'],
                                      bucket=bucket,
                                      method=method,
                                      q=q
                                      )
    score_ks.sort_values(by='min', ascending=False, inplace=True)
    score_ks['cum_bads_prop'] = score_ks['bads'].cumsum() / score_ks['bads'].sum()
    score_ks['通过率'] = score_ks['total'].cumsum() / score_ks['total'].sum()
    score_ks['通过人群整体响应率'] = score_ks.apply(lambda x: data[data['score'] > x['min']]['label'].mean(), axis=1)
    score_ks['min'] = round(score_ks['min'], 4)
    score_ks['max'] = round(score_ks['max'], 4)
    # return score_ks[['min', 'max', 'bad_rate', 'odds', 'cum_bads_prop', 'cum_lift', '通过率', '通过人群整体响应率']]
    return score_ks
