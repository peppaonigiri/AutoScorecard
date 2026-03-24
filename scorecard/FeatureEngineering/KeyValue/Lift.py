
from scipy.stats import scoreatpercentile
import matplotlib.pyplot as plt


def lift_curve_(result):
    result.columns = ['target', 'proba']
    result_ = result.copy()
    proba_copy = result.proba.copy()
    for i in range(10):
        point1 = scoreatpercentile(result_.proba, i*(100/10))
        point2 = scoreatpercentile(result_.proba, (i+1)*(100/10))
        proba_copy[(result_.proba >= point1) & (result_.proba <= point2)] = ((i+1))
    result_['grade'] = proba_copy
    df_gain = result_.groupby(by=['grade'], sort=True).sum()/(len(result)/10)*100
    plt.plot(df_gain['target'], color='red')
    for xy in zip(df_gain['target'].reset_index().values):
        plt.annotate("%s" % round(xy[0][1],2), xy=xy[0], xytext=(-20, 10), textcoords='offset points')
    plt.plot(df_gain.index,[sum(result['target'])*100.0/len(result['target'])]*len(df_gain.index), color='blue')
    plt.title('Lift Curve')
    plt.xlabel('Decile')
    plt.ylabel('Bad Rate (%)')
    plt.xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    plt.savefig("train.png")
    plt.show()


def lift_eval(input_df, score='credit_score', cutoff=600, target='bad', sample_rate=0.1):
    """
    功能：计算抽样和原始样本的lift
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param score: string, 信用分
    :param cutoff: float, 分数阈值，大于阈值则pass，否则reject
    :param target: string, 目标变量
    :param sample_rate: float, 好样本欠采样比例，比如0.1
    ----------------------------------------------------------------------
    :return lift_sam: float, 抽样样本上拒绝人群的lift
            lift_ori: float, 原始样本上拒绝人群的lift
    """
    df = input_df.copy()
    df['result'] = df['score'].apply(lambda x: 'PS' if x >= cutoff else 'RJ')
    rj_df = df[df['result'] == 'RJ']
    ps_df = df[df['result'] == 'PS']

    # 拒绝样本好坏样本数
    rj = len(rj_df)
    bad_rj = rj_df[target].sum()
    good_rj = rj - bad_rj

    # 通过样本好坏样本数
    ps = len(ps_df)
    bad_ps = ps_df[target].sum()
    good_ps = ps - bad_ps

    # 抽样样本上的lift
    lift_sam = (bad_rj / rj) / ((bad_rj + bad_ps) / (rj + ps))

    # 原始样本上的lift
    lift_ori = bad_rj / (bad_rj + bad_ps) * \
               (1 + (sample_rate * bad_ps + good_ps) / (sample_rate * bad_rj + good_rj))

    return lift_sam, lift_ori
