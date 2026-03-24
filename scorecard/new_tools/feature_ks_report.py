# 该代码为hello遗留

import pandas as pd
import numpy as np
import re
from scipy import stats

# 按bins_detail中对应变量的bin进行分箱统计KS值
def calculate_ks_by_bins(data, bins_detail, var_name, label_col='label'):
    """
    根据bins_detail中对应变量的bin进行分箱统计KS值
    
    Parameters:
    data: DataFrame, 包含变量和标签的数据
    bins_detail: DataFrame, 分箱详情数据
    var_name: str, 变量名
    label_col: str, 标签列名
    
    Returns:
    DataFrame: 包含分箱统计和KS值的结果
    """
    
    def parse_bin_interval(bin_str):
        """解析分箱区间字符串，返回min和max值"""
        if pd.isna(bin_str):
            return np.nan, np.nan
        
        # 处理区间字符串，如 "[-inf, 0.02083333395421505)" 或 "[-0.5, inf)"
        pattern = r'\[([^,]+),\s*([^)]+)\)'
        match = re.match(pattern, str(bin_str))
        
        if match:
            left_str = match.group(1).strip()
            right_str = match.group(2).strip()
            
            # 解析左边界
            if left_str == '-inf':
                left_val = -np.inf
            else:
                try:
                    left_val = float(left_str)
                except ValueError:
                    left_val = np.nan
            
            # 解析右边界
            if right_str == 'inf':
                right_val = np.inf
            else:
                try:
                    right_val = float(right_str)
                except ValueError:
                    right_val = np.nan
            
            return left_val, right_val
        else:
            return np.nan, np.nan
    
    # 获取该变量的分箱信息
    var_bins = bins_detail[bins_detail['variable'] == var_name].copy()
    if var_bins.empty:
        print(f"警告: 变量 {var_name} 在bins_detail中未找到")
        return pd.DataFrame(), 0
    
    # 按value排序（value列表示分箱的顺序）
    var_bins = var_bins.sort_values('value')
    
    # 创建分箱结果DataFrame
    result = []
    
    for _, bin_info in var_bins.iterrows():
        bin_name = bin_info['bin']
        bin_value = bin_info['value']
        
        # 解析分箱区间
        min_val, max_val = parse_bin_interval(bin_name)
        
        # 根据分箱条件筛选数据
        if np.isnan(min_val) and np.isnan(max_val):
            # 缺失值分箱
            bin_data = data[data[var_name].isna()]
        elif np.isnan(min_val):
            # 只有最大值
            bin_data = data[data[var_name] <= max_val]
        elif np.isnan(max_val):
            # 只有最小值
            bin_data = data[data[var_name] >= min_val]
        else:
            # 正常区间 - 注意这里使用左开右闭区间
            bin_data = data[(data[var_name] > min_val) & (data[var_name] <= max_val)]
        
        if len(bin_data) == 0:
            continue
            
        # 计算分箱统计
        total = len(bin_data)
        bad_count = bin_data[label_col].sum()
        good_count = total - bad_count
        bad_rate = bad_count / total if total > 0 else 0
        
        result.append({
            'variable': var_name,
            'bin': bin_name,
            'bin_value': bin_value,
            'min': min_val,
            'max': max_val,
            'total': total,
            'bad': bad_count,
            'good': good_count,
            'bad_rate': bad_rate,
            'bin_description': str(bin_name)
        })
    
    result_df = pd.DataFrame(result)
    
    if len(result_df) == 0:
        return result_df, 0
    
    # 计算累积统计和KS值
    result_df = result_df.sort_values('bin_value')
    result_df['cum_bad'] = result_df['bad'].cumsum()
    result_df['cum_good'] = result_df['good'].cumsum()
    result_df['cum_total'] = result_df['total'].cumsum()
    
    total_bad = result_df['bad'].sum()
    total_good = result_df['good'].sum()
    total_samples = result_df['total'].sum()
    
    if total_bad > 0 and total_good > 0:
        result_df['cum_bad_rate'] = result_df['cum_bad'] / total_bad
        result_df['cum_good_rate'] = result_df['cum_good'] / total_good
        result_df['ks'] = abs(result_df['cum_bad_rate'] - result_df['cum_good_rate'])
    else:
        result_df['cum_bad_rate'] = 0
        result_df['cum_good_rate'] = 0
        result_df['ks'] = 0
    
    # 计算总体KS值
    max_ks = result_df['ks'].max() if len(result_df) > 0 else 0
    
    return result_df, max_ks

def month_ks_report(data_end, keep_lst, bins_detail, max_month):

    month_data = data_end[data_end['month_time'] == max_month][keep_lst + ['label']]

    print(f"正在对月份 {max_month} 的数据进行分箱KS统计...")
    print(f"数据量: {len(month_data)}")

    # 对每个变量进行分箱KS统计
    ks_results = []
    for var_name in keep_lst:
        # print(f"处理变量: {var_name}")
        try:
            result_df, max_ks = calculate_ks_by_bins(month_data, bins_detail, var_name)
            if not result_df.empty:
                # 添加总体KS值
                result_df['max_ks'] = max_ks
                ks_results.append(result_df)
                # print(f"  - 变量 {var_name} 最大KS值: {max_ks:.4f}")
            else:
                print(f"  - 变量 {var_name} 无有效分箱数据")
        except Exception as e:
            print(f"  - 变量 {var_name} 处理出错: {str(e)}")

    # 合并所有结果
    if ks_results:
        all_ks_results = pd.concat(ks_results, ignore_index=True)
        
        summary_ks = all_ks_results.groupby('variable')['max_ks'].first().reset_index()
        summary_ks = summary_ks.sort_values('max_ks', ascending=False)
        return summary_ks
    else:
        print("没有成功处理任何变量")
        return None
    