import pandas as pd
from optbinning import OptimalBinning

"""
# basic
# dtype： 'numerical', 'categorical' 配合 cat_cutoff=0.1使用，合并发生率不到10%的类别
"""

x, y = pd.DataFrame(), pd.DataFrame()

optb = OptimalBinning(name='variable', dtype="numerical", solver="cp")
optb.fit(x, y)

binning_table = optb.binning_table.build()
print('if an optimal solution', optb.status)
print('optimal split points', optb.splits)
print('binning table', binning_table)
print('figure: ', binning_table.plot(metric="woe")) # "event_rate"

"""
# woe转换
# "event_rate"， "indices"猜测是bins_label, metric="bins" 返回分箱节点
"""
x_transform_woe = optb.transform(x, metric="woe")

"""
# advanced
# 打印信息 default print_level=1, minimal: print_level=0
"""
print('information: ', optb.information(print_level=1)) # print_level=2 更详细

"""
# 分析结果 statistical analysis of the binning table
# Gini index, Information Value (IV), Jensen-Shannon divergence, and the quality score
"""
binning_table.analysis(pvalue_test="chi2") # "fisher"，

"""
# 单调分箱
# monotonic_trend:“auto” , “ascending”, “descending”, “peak” ,“valley”，
# for large size instances：“auto_heuristic”, “auto_asc_desc”, “peak_heuristic” and “valley_heuristic”.
"""

"""
# 正则项 gamma 用于减少最大和最小箱之间的差异以产生更均匀的解决方案 
"""

"""
# 用户自定义分享节点 user_splits=user_splits, user_splits_fixed=user_splits_fixed
"""
user_splits = [14, 15, 16, 17, 20, 21, 22, 27]
user_splits_fixed = [False, True,  True, False, False, False, False, False]
user_splits_cat = [['Businessman'],
                   ['Working'],
                   ['Commercial associate'],
                   ['Pensioner', 'Maternity leave'],
                   ['State servant'],
                   ['Unemployed', 'Student']]

"""
# 求解器
# solver="mip"， small problems, say less than max_n_prebins<=20
#  solver="cp"， for medium and large problems
"""

"""
# 缺失值和特殊值
"""
special_codes = [-9, -8, -7]
special_codes = special_codes

# 限制求解时间
# min_prebin_size=0.001, time_limit=30

"""
# 回归问题 optimal binning with continuous target
# 如果箱与箱之间的差异不够明显： min_mean_diff=2.0
"""
from optbinning import ContinuousOptimalBinning
optb = ContinuousOptimalBinning(name='variable', dtype="numerical", min_mean_diff=2.0)

"""
# 多分类问题 optimal binning with multiclass target
"""
from optbinning import MulticlassOptimalBinning
optb = MulticlassOptimalBinning(name='variable', solver="cp")

"""
# 分别制定每一类的单调性
"""

optb = MulticlassOptimalBinning(name='variable', solver="mip",
                                monotonic_trend=["ascending", "auto", None],
                                verbose=True)

# optb =  OptimalBinning(cat_cutoff=None, class_weight=None, divergence='iv',
#                dtype='numerical', gamma=0, max_bin_n_event=None,
#                max_bin_n_nonevent=None, max_bin_size=None, max_n_bins=None,
#                max_n_prebins=20, max_pvalue=None,
#                max_pvalue_policy='consecutive', min_bin_n_event=None,
#                min_bin_n_nonevent=None, min_bin_size=None,
#                min_event_rate_diff=0, min_n_bins=None, min_prebin_size=0.05,
#                mip_solver='bop', monotonic_trend='auto', name='mean radius',
#                outlier_detector=None, outlier_params=None,
#                prebinning_method='cart', solver='cp', special_codes=None,
#                split_digits=None, time_limit=100, user_splits=None,
#                user_splits_fixed=None)
