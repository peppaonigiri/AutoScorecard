import math
import pandas as pd


def show_stage1(x_val):
    x_val_ = x_val.copy()
    x_val_["pred"] = 1 - x_val_["proba"]
#     x_val_["pred"] = model.predict_proba(x_val)[:,0]
    x_val_["FICO"] = x_val_["pred"].apply(lambda x: 300 + int(500*x))

    InputData = x_val_
#     InputData['label'] = pd.DataFrame(y_val)
    #data["FICO"].describe()
    scoreList = list(InputData["FICO"].sort_values())
    npart = int(len(scoreList)/10)

    listBins = [100]
    for i in range(0, len(scoreList), npart):
        partList = scoreList[i : i + npart]
        if len(partList) == npart:
            listBins.append(partList[-1])

    listBins[-1] = 900
    # print(listBins)

    # listBins = [100, 710, 737, 750, 759, 764, 770, 780, 785, 900]
    se = pd.cut(InputData["FICO"], bins=listBins, right=False, duplicates='drop')

    tjData = pd.value_counts(se).sort_index(ascending=False)

    overdueRateList = []
    cumulativeOverdueRateList = []
    ksList = []

    i = len(listBins) - 1
    allBadTotal = InputData[InputData["label"] == 1]["label"].count()
    allGoodTotal = InputData[InputData["label"] == 0]["label"].count()

    while i > 0:
        odData = InputData[(InputData["FICO"] >= listBins[i-1]) & (InputData["FICO"] < listBins[i])]
        total = odData["label"].count()
        bad_total = odData[odData["label"] == 1]["label"].count()
        good_total = odData[odData["label"] == 0]["label"].count()
        bad_pcnt = bad_total/total if total != 0 else 0
        good_pcnt = good_total/total if total != 0 else 0
    #     print(bad_pcnt)
        overdueRateList.append(bad_pcnt)

        odData = InputData[InputData["FICO"] >= listBins[i-1]]
        total = odData["label"].count()
        badCnt = odData[odData["label"] == 1]["label"].count()
        goodCnt = odData[odData["label"] == 0]["label"].count()
        bad_pcnt = badCnt/total if total != 0 else 0
        ks = round(math.fabs((badCnt / allBadTotal) - (goodCnt / allGoodTotal)),3)
        ksList.append(ks)
        cumulativeOverdueRateList.append(bad_pcnt)
        i = i - 1
    sampleTotal = sum(list(tjData))
    sampleList = []
    cumulativeTotalPercent = []
    for amout in list(tjData):
        sampleList.append(amout)
        cumulativeTotalPercent.append(sum(sampleList)/sampleTotal)

    pdData = {
            'score range': list(tjData.index),
            'amount': list(tjData),
            'cumulative total percent': cumulativeTotalPercent,
            'bad rate': overdueRateList,
            'cumulative bad rate': cumulativeOverdueRateList,
            'ks': ksList,
    }
    return pdData