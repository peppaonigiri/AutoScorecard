import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


def psi_for_continue_var(expected_array, actual_array, bins=10, method='equal_width', detail=False, save_file_path=None):
    '''
    ----------------------------------------------------------------------
    功能: 计算连续型变量的群体性稳定性指标（population stability index ,PSI）
    ----------------------------------------------------------------------
    :param expected_array: numpy array of original values，基准组
    :param actual_array: numpy array of new values, same size as expected，比较组
    :param bins: number of percentile ranges to bucket the values into，分箱数, 默认为10
    :param method: string, 分箱模式，'equal_width'为等距均分，'equal_frequency'为按等频分箱
    :param detail: bool, 取值为True时输出psi计算的完整表格, 否则只输出最终的psi值
    :param save_file_path: string, csv文件保存路径. 默认值=None. 只有当detail=Ture时才生效.
    ----------------------------------------------------------------------
    :return psi_value: 
            当detail=False时, 类型float, 输出最终psi计算值;
            当detail=True时, 类型pd.DataFrame, 输出psi计算的完整表格。最终psi计算值 = list(psi_value['psi'])[-1]
    ----------------------------------------------------------------------
    示例：
    >>> psi_for_continue_var(expected_array=df['score'][:400],
                            actual_array=df['score'][401:], 
                            bins=5, bucket_type='bins', detail=0)
    >>> 0.0059132756739701245
    ------------
    >>> psi_for_continue_var(expected_array=df['score'][:400],
                            actual_array=df['score'][401:], 
                            bins=5, bucket_type='bins', detail=1)
    >>>
        score_range	expecteds	expected(%)	actucalsactucal(%)ac - ex(%)ln(ac/ex)psi	max
    0	[0.021,0.2095]	120.0	30.00	152.0	31.02	1.02	0.033434	0.000341	
    1	(0.2095,0.398]	117.0	29.25	140.0	28.57	-0.68	-0.023522	0.000159	
    2	(0.398,0.5865]	81.0	20.25	94.0	19.18	-1.07	-0.054284	0.000577	<<<<<<<
    3	(0.5865,0.7751]	44.0	11.00	55.0	11.22	0.22	0.019801	0.000045	
    4	(0.7751,0.9636]	38.0	9.50	48.0	9.80	0.30	0.031087	0.000091	
    5	>>> summary	400.0	100.00	489.0	100.00	NaN	NaN	0.001214	<<< result
    ----------------------------------------------------------------------
    知识:
    公式： psi = sum(（实际占比-预期占比）* ln(实际占比/预期占比))
    一般认为psi小于0.1时候变量稳定性很高，0.1-0.25一般，大于0.25变量稳定性差，建议重做。
    相对于变量分布(EDD)而言, psi是一个宏观指标, 无法解释两个分布不一致的原因。但可以通过观察每个分箱的sub_psi来判断。
    ----------------------------------------------------------------------
    '''
    
    expected_array = pd.Series(expected_array).dropna()
    actual_array = pd.Series(actual_array).dropna()
    if isinstance(list(expected_array)[0], str) or isinstance(list(actual_array)[0], str):
        raise Exception("输入数据expected_array只能是数值型, 不能为string类型")
        
    """step1: 确定分箱间隔"""
    def scale_range(input_array, scaled_min, scaled_max):
        '''
        ----------------------------------------------------------------------
        功能: 对input_array线性放缩至[scaled_min, scaled_max]
        ----------------------------------------------------------------------
        :param input_array: numpy array of original values, 需放缩的原始数列
        :param scaled_min: float, 放缩后的最小值
        :param scaled_min: float, 放缩后的最大值
        ----------------------------------------------------------------------
        :return input_array: numpy array of original values, 放缩后的数列
        ----------------------------------------------------------------------
        '''
        input_array += -np.min(input_array) # 此时最小值放缩到0
        if scaled_max == scaled_min:
            raise Exception('放缩后的数列scaled_min = scaled_min, 值为{}, 请检查expected_array数值！'.format(scaled_max))
        scaled_slope = np.max(input_array) * 1.0 / (scaled_max - scaled_min)
        input_array /= scaled_slope
        input_array += scaled_min
        return input_array
    
    # 异常处理，所有取值都相同时, 说明该变量是常量, 返回999999
    if np.min(expected_array) == np.max(expected_array):
        return 999999
    
    all_array = np.hstack([expected_array, actual_array])
    breakpoints = np.arange(0, bins + 1) / (bins) * 100 # 等距分箱百分比
    if 'equal_width' == method:        # 等距分箱
        breakpoints = scale_range(breakpoints, np.min(all_array), np.max(all_array))
    elif 'equal_frequency' == method: # 等频分箱
        breakpoints = np.stack([np.percentile(all_array, b) for b in breakpoints])

    """step2: 统计区间内样本占比"""
    def generate_counts(arr, breakpoints):
        '''
        ----------------------------------------------------------------------
        功能: Generates counts for each bucket by using the bucket values 
        ----------------------------------------------------------------------
        :param arr: ndarray of actual values
        :param breakpoints: list of bucket values
        ----------------------------------------------------------------------
        :return cnt_array: counts for elements in each bucket, length of breakpoints array minus one
        :return score_range_array: 分箱区间
        ----------------------------------------------------------------------
        '''
        def count_in_range(arr, low, high, start):
            '''
            ----------------------------------------------------------------------
            功能: 统计给定区间内的样本数(Counts elements in array between low and high values)
            ----------------------------------------------------------------------
            :param arr: ndarray of actual values
            :param low: float, 左边界
            :param high: float, 右边界
            :param start: bool, 取值为Ture时，区间闭合方式[low, high],否则为(low, high]
            ----------------------------------------------------------------------
            :return cnt_in_range: int, 给定区间内的样本数
            ----------------------------------------------------------------------
            '''
            if start:
                cnt_in_range = len(np.where(np.logical_and(arr >= low, arr <= high))[0])
            else:
                cnt_in_range = len(np.where(np.logical_and(arr > low, arr <= high))[0])
            return cnt_in_range

        cnt_array = np.zeros(len(breakpoints) - 1)
        score_range_array = [''] * (len(breakpoints) - 1)
        for i in range(1, len(breakpoints)):
            cnt_array[i-1] = count_in_range(arr, breakpoints[i-1], breakpoints[i], i==1)
            if 1 == i:
                score_range_array[i-1] = '[' + str(round(breakpoints[i-1], 4)) + ',' + str(round(breakpoints[i], 4)) + ']'
            else:
                score_range_array[i-1] = '(' + str(round(breakpoints[i-1], 4)) + ',' + str(round(breakpoints[i], 4)) + ']'
                                                                                
        return (cnt_array, score_range_array)

    expected_cnt = generate_counts(expected_array, breakpoints)[0]
    expected_percents = expected_cnt / len(expected_array)
    actual_cnt = generate_counts(actual_array, breakpoints)[0]
    actual_percents = actual_cnt / len(actual_array)
    delta_percents = actual_percents - expected_percents
    score_range_array = generate_counts(expected_array, breakpoints)[1]
                                                                                
    """step3: 区间放缩"""
    def sub_psi(e_perc, a_perc):
        '''
        ----------------------------------------------------------------------
        功能: 计算单个分箱内的psi值。Calculate the actual PSI value from comparing the values.
            Update the actual value to a very small number if equal to zero
        ----------------------------------------------------------------------
        :param e_perc: float, 期望占比
        :param a_perc: float, 实际占比
        ----------------------------------------------------------------------
        :return value: float, 单个分箱内的psi值
        ----------------------------------------------------------------------
        '''
        if a_perc == 0: # 实际占比
            a_perc = 0.000001
        if e_perc == 0: # 期望占比
            e_perc = 0.000001
        value = (e_perc - a_perc) * np.log(e_perc * 1.0 / a_perc)
        return value
    
    """step4: 得到最终稳定性指标"""
    sub_psi_array = [sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))]
    if detail:
        psi_value = pd.DataFrame()
        psi_value['score_range'] = score_range_array
        psi_value['expecteds'] = expected_cnt
        psi_value['expected(%)'] = expected_percents * 100
        psi_value['actucals'] = actual_cnt
        psi_value['actucal(%)'] = actual_percents * 100
        psi_value['ac - ex(%)'] = delta_percents * 100
        psi_value['actucal(%)'] = psi_value['actucal(%)'].apply(lambda x: round(x, 2))
        psi_value['ac - ex(%)'] = psi_value['ac - ex(%)'].apply(lambda x: round(x, 2))
        expected_percents_array = np.where(expected_percents == 0, 0.000001, expected_percents)
        actual_percents_array = np.where(actual_percents == 0, 0.000001, actual_percents)
        psi_value['ln(ac/ex)'] = np.log(actual_percents_array * 1.0 / expected_percents_array)
        psi_value['psi'] = sub_psi_array
        flag = lambda x: '<<<<<<<' if x == psi_value.psi.max() else ''
        psi_value['max'] = psi_value.psi.apply(flag)
        add_df = pd.DataFrame([{'score_range':'>>> summary', 
                                'expecteds': sum(expected_cnt),
                                'expected(%)':100,
                                'actucals': sum(actual_cnt),
                                'actucal(%)':100,
                                'ac - ex(%)': np.nan,
                                'ln(ac/ex)': np.nan,
                                'psi': np.sum(sub_psi_array),
                                'max':'<<< result'}])
        psi_value = pd.concat([psi_value, add_df], ignore_index=True)
        if save_file_path:
            if not isinstance(save_file_path, str):
                raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
            elif not save_file_path.endswith('.csv'):
                raise Exception('参数save_file_path不是csv文件后缀，请检查!')
            psi_value.to_csv(save_file_path, encoding='utf-8', index=1)
    else:
        psi_value = np.sum(sub_psi_array)

    return psi_value




def psi_filter(df, var_list, expect_list=['train'], actual_list=['oot'], threshold=0.05, use_numpy=True, method='equal_width'):
    '''
    ----------------------------------------------------------------------
    功能: 过滤变量，根据psi值进行筛选
    ----------------------------------------------------------------------
    :param df: pandas dataframe, 包含变量var_list, 训练验证测试集划分列'target'
    :param var_list: list of string, 待筛选变量名列表
    :param expect_list: list of string, 期望组名列表
    :param actual_list: list of string, 实际组名列表
    :param threshold: float, 阈值，psi值大于等于该值时，保留该变量
    :param use_numpy: bool, 是否使用numpy计算，默认为True
    :param method: string, 分箱方法，'equal_width'（等距）或 'equal_frequency'（等频），默认为'equal_width'
    ----------------------------------------------------------------------
    :return keep_list: list of string, 保留的变量名列表
    :return psi_df: pandas dataframe, 包含变量名var_list, 对应的psi值
    ----------------------------------------------------------------------
    示例：
    >>> df = pd.read_csv('data.csv')
    >>> keep_list, psi_df = psi_filter(df, var_list=['var1', 'var2', 'var3'], expect_list=['train'], actual_list=['oot'], threshold=0.05)
    >>> keep_list
    ['var1', 'var2']
    >>> psi_df
        var  psi
    0   var1  0.0
    1   var2  0.0
    ----------
    该函数通过计算期望组和实际组的psi值，筛选出psi值小于等于阈值的变量。
    ----------------------------------------------------------------------

    '''
    expected_df = df[df['target'].isin(expect_list)]
    actual_df = df[df['target'].isin(actual_list)]
    psi_dic = {}
    for var in tqdm(var_list,desc='筛选变量psi值'):
        expected_array = expected_df[var].values
        actual_array = actual_df[var].values
        if use_numpy:
            psi_value = calculate_psi(expected_array, actual_array, method=method)
        else:
            psi_value = psi_for_continue_var(expected_array, actual_array, bucket_type=method)
        psi_dic[var] = psi_value
    psi_df = pd.DataFrame(psi_dic.items(), columns=['var', 'psi']).sort_values(by='psi', ascending=False)
    keep_list = psi_df[psi_df['psi'] <= threshold]['var'].tolist()
    return keep_list, psi_df




def choice_month_psi(df, choice_moth, var_name, moth_col, use_numpy=True, method='equal_width'):
    '''
    ----------------------------------------------------------------------
    功能: 计算选择不同月份作为期望分布，计算各月份的psi值
    ----------------------------------------------------------------------
    :param df: pandas dataframe, 包含要计算的列名var_name, 月份的列名moth_col
    :param choice_moth: string, 选择的月份
    :param var_name: string, 要计算的列名
    :param moth_col: string, 月份数所在的列名
    :param use_numpy: bool, 是否使用numpy计算，默认为True
    :param method: string, 分箱方法，'equal_width'（等距）或 'equal_frequency'（等频），默认为'equal_width'
    ----------------------------------------------------------------------
    :return psi_df: pandas dataframe, 指定月份与各月份对应的psi值
    ----------------------------------------------------------------------
    示例：
    >>> df = pd.read_csv('data.csv')
    >>> choice_moth = '2024-07'
    >>> var_name = 'score'
    >>> moth_col = 'month_time'
    >>> psi_df = choice_moth_psi(df, choice_moth, var_name, moth_col)
    >>> psi_df
        moth  psi
    0   moth1  0.0
    1   moth2  0.0
    ----------------------------------------------------------------------
    '''
    psi_dic = {}
    excepted_values = df.query(f"{moth_col} == '{choice_moth}'")[var_name].values
    for moth in df[moth_col].unique():
        # print(moth)
        actual_values = df.query(f"{moth_col} == '{moth}'")[var_name].values
        if use_numpy:
            psi_value = calculate_psi(excepted_values, actual_values, method=method)
        else:
            psi_value = psi_for_continue_var(excepted_values,actual_values,bucket_type=method)
        psi_dic[moth] = psi_value
    psi_df = pd.DataFrame.from_dict(psi_dic,orient='index').sort_index().reset_index().rename(columns={'index':'moth',0:'psi'})
    return psi_df


def plot_month_psi_curve(df, moth_col, var_name='score'):
    """
    绘制月度PSI曲线图。

    参数:
    df: pandas.DataFrame, 包含数据的DataFrame。
    moth_col: str, 表示月份的列名。
    var_name: str, 表示变量名称的列名，默认为'score'。

    返回:
    pandas.DataFrame, 经过选择后的包含PSI值的数据DataFrame。
    """
    # 根据输入数据和条件选择合适的月份及对应的变量值，计算PSI值
    psi_df = choice_month_psi(df, df[moth_col].min(), var_name, moth_col)

    # 提取x轴和y轴的数据
    x = psi_df['moth']
    y = psi_df['psi']

    # 绘制PSI曲线
    plt.plot(x, y, linewidth=2, c='g')
    # 设置图表标题
    plt.title("psi curve", fontsize=20)
    # 设置x轴和y轴的标签
    plt.xlabel("month time", fontsize=12)
    plt.ylabel("psi value", fontsize=12)
    # 设置刻度标签的大小
    plt.tick_params(axis='both', labelsize=10)
    # 设置y轴的范围
    plt.ylim(0, psi_df['psi'].max() + 0.005)

    # 设置字体为Times New Roman，以适应中文显示
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    # 显示图表
    plt.show()
    return psi_df


def calculate_psi(expected, actual, num_bins=10, method='equal_width'):
    """
    计算PSI值以衡量两个数据分布之间的差异。

    参数：
    actual (np.ndarray): 实际数据列。
    expected (np.ndarray): 预期数据列。
    num_bins (int): 分箱数量，默认为10。
    method (str): 分箱方法，'equal_width'（等距）或 'equal_frequency'（等频），默认为'equal_width'。

    返回：
    psi (float): 计算得到的PSI值。
    """
    # 确保输入是numpy数组
    actual = np.array(actual)
    expected = np.array(expected)
    all_array = np.hstack([actual, expected])
    
    # print(actual)
    # 分箱
    if method == 'equal_width':
        bins = np.linspace(min(all_array), 
                            max(all_array), 
                            num_bins+1)
    elif method == 'equal_frequency':
        bins = np.unique(np.percentile(all_array, np.linspace(0, 100, num_bins+1)))
    else:
        raise ValueError("method must be 'equal_width' or 'equal_frequency'")
    
    # 计算每个分箱的频率占比
    # print(bins)
    actual_counts = np.histogram(actual, bins=bins)[0]
    # print(actual_counts)
    expected_counts = np.histogram(expected, bins=bins)[0]
    # print(expected_counts)
    
    actual_counts = actual_counts / actual_counts.sum()
    expected_counts = expected_counts / expected_counts.sum()
    
    # 避免除以零的情况，替换零为一个小的值
    actual_counts[actual_counts == 0] = 0.000001
    expected_counts[expected_counts == 0] = 0.000001
    
    # 计算psi值
    # print(actual_counts)
    # print(expected_counts)
    psi = np.sum((actual_counts - expected_counts) * 
                    np.log(actual_counts / expected_counts))
    
    return psi

# # 示例数据
# data1 = np.random.normal(0, 1, 1000)
# data2 = np.random.normal(0, 2, 1000)
# print(f'data1最大值: {data1.max()}, 最小值: {data1.min()}')
# print(f'data2最大值: {data2.max()}, 最小值: {data2.min()}')

# # 计算psi值（默认等距分箱）
# psi_equal_width = calculate_psi(data1, data2, method='equal_width')
# print('等距分箱的PSI值: ', psi_equal_width)

# # 计算psi值（等频分箱）
# psi_equal_frequency = calculate_psi(data1, data2, method='equal_frequency')
# print('等频分箱的PSI值: ', psi_equal_frequency)

# psi_equal_width = psi_for_continue_var(data1, data2, method='equal_width',detail=True)
# print('等距分箱的PSI值: ', psi_equal_width)

# psi_equal_frequency = psi_for_continue_var(data1, data2, method='equal_frequency',detail=False)
# print('等频分箱的PSI值: ', psi_equal_frequency)


