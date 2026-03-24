import pandas as pd


def swapset_table(input_df, score_var1, score_var2, weight_var, target_var, bins):
    """
    功能：新旧模型分交叉后，统计样本量和bad量，进而swap set分析
    ----------------------------------------------------------------------
    :param input_df: pd.DataFrame, 输入数据
    :param score_var1: string, 模型分数变量1
    :param score_var2: string, 模型分数变量2
    :param target_var: string, 目标变量
    :param weight_var: string, 权重变量
    :param bins: int, 分箱数, 典型取值为10或20
    ----------------------------------------------------------------------
    :return stat_df: pd.DataFrame, 统计数据
    ----------------------------------------------------------------------
    用法：
    >>> stat = swapset_table(input_df=test_df,
                             score_var1='score1',
                             score_var2='score2',
                             weight_var='weight',
                             target_var='bad',
                             bins=10)
    >>> stat[['total']].unstack().fillna(0)
    >>> stat[['bad']].unstack().fillna(0)
    """

    def get_bucket(score_list, weight_list, bins):
        final_score_list = []  # 得到总体样本上的分数
        for i in range(len(score_list)):
            final_score_list += [score_list[i]] * round(weight_list[i])

        bucket_lst = pd.qcut(final_score_list, bins, duplicates='drop')  # 等频分箱
        bucket_map = {}
        for x in range(len(bucket_lst)):
            bucket_map[final_score_list[x]] = bucket_lst[x]
        return bucket_map

    df = input_df.copy()
    score1_list = list(df.loc[:, score_var1])
    score2_list = list(df.loc[:, score_var2])
    weight_list = list(df.loc[:, weight_var])

    # 分箱映射
    bucket1_map = get_bucket(score1_list, weight_list, bins)
    bucket2_map = get_bucket(score2_list, weight_list, bins)

    new_score1 = score_var1 + '_bin'
    new_score2 = score_var2 + '_bin'
    df[new_score1] = df[score_var1].apply(lambda x: bucket1_map.get(x) if pd.notna(x) else "missing")
    df[new_score2] = df[score_var2].apply(lambda x: bucket2_map.get(x) if pd.notna(x) else "missing")

    df['total'] = df[weight_var]
    df['bad'] = df[target_var] * df[weight_var]
    stat_df = df.groupby([new_score1, new_score2]).sum()[['total', 'bad']]
    stat_df['total'] = stat_df['total'].apply(lambda x: round(x))
    stat_df['bad'] = stat_df['bad'].apply(lambda x: round(x))

    return stat_df