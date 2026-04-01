import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class IVCalculator:
    """
    计算IV值
    参数:
    df (DataFrame): 包含特征、标签、数据集标签的DataFrame
    label (str): y标签列名
    target (str): 数据集划分标签列名
    keep_list (list): 待计算IV的特征列名列表
    max_leaf_nodes (int): 决策树最大叶子节点数，决定分成几箱
    min_samples_leaf (float): 决策树最小样本数，决定每箱最小样本占比

    属性:
    bins_dict (dict): 各个特征的分箱边界
    iv_dict (dict): 各个特征的IV值
    target_list (list): 数据集标签列表
    dataset_dict (dict): 各个数据集的DataFrame
    detail_df_dict (dict): 各个特征的分箱细节
    iv_df: 各个特征按照不同数据集的IV值
    detail_df_all: 所有特征的分箱细节
    compare_report_df: 各个特征的IV值比较报告,含PSI值

    方法:
    get_bins(var): 获取分箱结果
    calculate_iv(x, y, bins=None): 计算IV值
    calculate_iv_for_dataset(var): 按照数据集划分计算IV值
    iv_report(use_thread=False, max_workers=4): 计算IV值报告
    detail_report(var_list, target_name='train'): 计算分箱细节
    compared_report(var_list, target_name='train'): 计算IV值比较报告,含PSI值
    """
    
    def __init__(self, df, label='label', target='target', keep_list=None, max_leaf_nodes=10, min_samples_leaf=0.02): #6 0.05
        print('初始化IV计算器,获取各个数据集的索引...')
        self.df = df
        self.label = label
        self.target = target
        self.keep_list = keep_list
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.bins_dict = {}
        self.iv_dict = {}
        self.target_list = []
        self.dataset_dict = {}

        if self.keep_list is None:
            raise ValueError('keep_list 参数不能为空')
        if 'train' not in df[self.target].unique():
            raise ValueError('数据集中不存在train标签')
        
        self.target_list = df[self.target].unique().tolist()
        self.dataset_dict = {group_name: dataset for  group_name, dataset in df.groupby(self.target)}
        self.detail_df_dict = {var:{} for var in self.keep_list}
        
        print('初始化完成')

    def optimal_binning_boundary(self, x, y):
        """
        使用决策树计算最优分箱边界
        :param x: pandas Series
        :param y: pandas Series
        :return: list
        """
        boundary = []
        min_x = -np.inf
        max_x = np.inf
        x = x.values
        y = y.values
        clf = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=self.min_samples_leaf)
        clf.fit(x.reshape(-1, 1), y)
        n_nodes = clf.tree_.node_count  # 决策树的节点数
        children_left = clf.tree_.children_left  # node_count大小的数组，children_left[i]表示第i个节点的左子节点
        children_right = clf.tree_.children_right  # node_count大小的数组，children_right[i]表示第i个节点的右子节点
        threshold = clf.tree_.threshold  # node_count大小的数组，threshold[i]表示第i个节点划分数据集的阈值

        for i in range(n_nodes):
            if children_left[i] != children_right[i]:
                boundary.append(threshold[i])
        boundary.sort()
        boundary = [min_x] + boundary + [max_x]
        return boundary

    def calculate_iv(self, x, y, bins=None):
        """
        计算IV值

        参数:
        x (Series): 特征
        y (Series): 标签
        bins (list, optional): 分箱边界，默认为None，使用最优分箱

        返回:
        float: 总IV值
        """
        if bins is None:
            boundary = self.optimal_binning_boundary(x, y)
        else:
            boundary = bins

        # 如果返回的分箱是[-inf, inf]，返回0
        if len(boundary) <= 2:
            return 0, pd.DataFrame()

        # 使用numpy进行分箱
        x_values = x.values
        y_values = y.values
        bins = np.array(boundary)
        # np.digitize(x_values, bins)返回x_values在所属的分箱的索引
        bin_indices = np.digitize(x_values, bins, right=True)

        # 计算每个分箱中的“好”样本数、“坏”样本数和总样本数
        unique_bins = np.unique(bin_indices)
        good_counts = np.zeros(len(unique_bins))
        bad_counts = np.zeros(len(unique_bins))
        total_counts = np.zeros(len(unique_bins))

        for i, bin_idx in enumerate(unique_bins):
            mask = bin_indices == bin_idx
            good_counts[i] = np.sum(y_values[mask] == 0)
            bad_counts[i] = np.sum(y_values[mask] == 1)
            total_counts[i] = np.sum(mask)

        # 计算bad_rate
        bad_rate = bad_counts / total_counts

        # 计算百分比
        total_good = np.sum(good_counts)
        total_bad = np.sum(bad_counts)
        total_total = np.sum(total_counts)

        good_pcts = good_counts / total_good
        bad_pcts = bad_counts / total_bad
        total_pcts = total_counts / total_total

        # 处理零值
        good_pcts[good_pcts == 0] = 0.0001
        bad_pcts[bad_pcts == 0] = 0.0001

        # 计算WOE和IV
        woes = np.log(good_pcts / bad_pcts)
        ivs = (good_pcts - bad_pcts) * woes

        total_iv = np.sum(ivs)

        # 计算分箱细节
        details_df = pd.DataFrame({'bin': unique_bins, 'good': good_counts,
                                    'bad': bad_counts, 'total': total_counts,
                                    'bad_rate': bad_rate,
                                    'good_pct': good_pcts, 'bad_pct': bad_pcts,
                                    'total_pct': total_pcts, 'woe': woes,
                                    'iv': ivs, 'total_iv': total_iv})
        
        # 计算 Lift 和 累计指标
        total_bad_rate = total_bad / total_total if total_total > 0 else 0
        details_df['lift'] = (details_df['bad_rate'] / total_bad_rate) if total_bad_rate > 0 else 0
        
        details_df['bad_cum'] = details_df['bad'].cumsum()
        details_df['total_cum'] = details_df['total'].cumsum()
        details_df['cum_bad_rate'] = details_df['bad_cum'] / details_df['total_cum']
        details_df['cum_lift'] = (details_df['cum_bad_rate'] / total_bad_rate) if total_bad_rate > 0 else 0
        
        details_df['badCumRate'] =  details_df['bad'].cumsum() / total_bad if total_bad > 0 else 0
        details_df['goodCumRate'] = details_df['good'].cumsum() / total_good if total_good > 0 else 0
        ks = max(details_df.apply(lambda x: abs(x.badCumRate - x.goodCumRate), axis=1)) if not details_df.empty else 0
        details_df['ks'] = ks
        bins_replace = {i: str(f'({bins[i-1]},{bins[i]}]') for i in range(1, len(bins))}
        details_df['bin'].replace(bins_replace, inplace=True)

        return total_iv, details_df

    def get_bins(self, var):
        """
        获取分箱结果
        :param var: str, 待分箱的变量名
        :return: None
        """
        train_var_series = self.dataset_dict['train'][var]
        train_label_series = self.dataset_dict['train'][self.label]
        bins = self.optimal_binning_boundary(train_var_series, train_label_series)
        self.bins_dict[var] = bins

    def calculate_iv_for_dataset(self, var):
        """
        计算IV值
        :param var: str, 待计算IV的变量名
        :return: None
        """
        bins = self.bins_dict[var]
        self.iv_dict[var] = []
        for target_name in self.target_list:
            temp_dataset = self.dataset_dict[target_name]
            x = temp_dataset[var]
            y = temp_dataset[self.label]
            iv, details_df = self.calculate_iv(x, y, bins)
            details_df.insert(0,'var_name',var)
            self.iv_dict[var].append(iv)
            self.detail_df_dict[var][target_name] = details_df

    def iv_report(self, use_thread=False, max_workers=4):
        """
        计算IV值报告
        :param use_thread: bool, 是否使用多线程计算
        :param max_workers: int, 最大线程数
        :return: pandas DataFrame
        """
        if use_thread:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(executor.map(self.get_bins, self.keep_list), total=len(self.keep_list), desc='多线程获取分箱bins'))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(executor.map(self.calculate_iv_for_dataset, self.keep_list), total=len(self.keep_list), desc='多线程特征计算IV'))
        else:
            for var in tqdm(self.keep_list, desc='获取分箱bins'):
                self.get_bins(var)
            for var in tqdm(self.keep_list, desc='特征计算IV'):
                self.calculate_iv_for_dataset(var)

        iv_df = pd.DataFrame(self.iv_dict).T
        iv_df.columns = [f'{col}_iv' for col in self.target_list]
        iv_df.index.name = 'var_name'
        iv_df.reset_index(inplace=True)
        self.iv_df = iv_df
        return iv_df
    
    def detail_report(self, var_list, target_name='train'):
        """
        计算IV分箱细节报告
        :param var_list: list, 待计算IV细节的变量名列表
        :param target_name: str, 待计算IV细节的目标数据集名,如果为'all',则计算在所有数据集上的IV细节
        :return: pandas DataFrame
        """
        details_df_list = []
        if target_name == 'all':
            for var in tqdm(var_list, desc='计算在所有数据集上的IV分箱细节'):
                bins = self.bins_dict[var]
                x = self.df[var]
                y = self.df[self.label]
                iv, details_df = self.calculate_iv(x, y, bins)
                if len(details_df) > 0:
                    details_df.insert(0,'var_name',var)
                    details_df_list.append(details_df)
        else:
            for var in var_list:
                temp_df = self.detail_df_dict[var][target_name]
                details_df_list.append(temp_df)

        if len(details_df_list) == 0:
            raise ValueError('所有变量的分箱bins均为[-inf, inf], 无法计算分箱细节')
        
        self.detail_df_all = pd.concat(details_df_list)
        return self.detail_df_all
    
    def compare_report(self, var_list):
        """
        计算IV对比报告
        :param var_list: list, 待比较IV的变量名列表
        :return: pandas DataFrame

        tips: 
        1. train_oot_iv_diff 是指 train_IV - oot_IV, train_oot_iv_diff_ratio 是指 (train_IV - oot_IV) / train_IV
        2. PSI值计算的是按照当前分箱每箱样本占比的PSI值，并非等频或者等距分箱的PSI值
        """
        compare_df_dict = {}
        for target_name in self.target_list:
            temp_df = self.detail_report(var_list, target_name)
            temp_df.columns = ['var_name', 'bin'] + [f'{col}_{target_name}' for col in temp_df.columns if col not in ['var_name', 'bin']]
            compare_df_dict[target_name] = temp_df
        
        temp_list = [key for key in compare_df_dict.keys()]
        compare_list = []
        for i in range(len(temp_list)-1):
            for j in range(i+1, len(temp_list)):
                compare_list.append((temp_list[i], temp_list[j]))

        psi_dict = {}
        iv_diff_dict = {}
        iv_diff_ratio_dict = {}
        for compare_1, compare_2 in compare_list:
            temp_df = pd.merge(compare_df_dict[compare_1], compare_df_dict[compare_2], on=['var_name', 'bin'], how='inner')
            temp_df[f'total_pct_{compare_1}'] = temp_df[f'total_pct_{compare_1}'].replace(0, 0.0001)
            temp_df[f'total_pct_{compare_2}'] = temp_df[f'total_pct_{compare_2}'].replace(0, 0.0001)

            psi_list = []
            iv_diff_list = []
            iv_diff_ratio_list = []
            for var in tqdm(var_list, desc=f'计算{compare_1}_{compare_2}对比报告'):
                temp_df_cut = temp_df[temp_df['var_name'] == var]
                temp_psi = (temp_df_cut[f'total_pct_{compare_1}'] - temp_df_cut[f'total_pct_{compare_2}']) * np.log(temp_df_cut[f'total_pct_{compare_1}'] / temp_df_cut[f'total_pct_{compare_2}'])
                psi_list.append(temp_psi.sum())
                temp_iv_diff = temp_df_cut[f'total_iv_{compare_1}'] - temp_df_cut[f'total_iv_{compare_2}']
                iv_diff_list.append(temp_iv_diff.max())
                temp_iv_diff_ratio = (temp_df_cut[f'total_iv_{compare_1}'] - temp_df_cut[f'total_iv_{compare_2}']) / temp_df_cut[f'total_iv_{compare_1}']
                iv_diff_ratio_list.append(temp_iv_diff_ratio.max())
            
            psi_dict[f'{compare_1}_{compare_2}_psi'] = psi_list
            iv_diff_dict[f'{compare_1}_{compare_2}_iv_diff'] = iv_diff_list
            iv_diff_ratio_dict[f'{compare_1}_{compare_2}_iv_diff_ratio'] = iv_diff_ratio_list
        
        psi_df = pd.DataFrame(psi_dict)
        iv_diff_df = pd.DataFrame(iv_diff_dict)
        iv_diff_ratio_df = pd.DataFrame(iv_diff_ratio_dict)
        compare_report_df = pd.concat([psi_df, iv_diff_df, iv_diff_ratio_df], axis=1)
        compare_report_df.insert(0, 'var_name', var_list)
        compare_report_df = pd.merge(self.iv_df, compare_report_df, on='var_name', how='right')
        self.compare_report_df = compare_report_df
        return compare_report_df








# # 示例用法

# df = pd.DataFrame({'label': np.random.randint(0, 1, 10000), 'target': np.random.choice(['train', 'test'], 10000), 'feature1': np.random.normal(0, 1, 10000), 'feature2': np.random.normal(0, 1, 10000)})
# iv_calculator = IVCalculator(df, label='label', target='target', keep_list=['feature1', 'feature2'])
# iv_values = iv_calculator.calculate_iv(df['feature1'], df['label'])
# print(iv_values)
# iv_df = iv_calculator.iv_report(use_thread=False, max_workers=4)
# detail_df = iv_calculator.detail_report(['feature1'], target_name='all')
# compare_df = iv_calculator.compare_report(['feature1', 'feature2'])
# print(iv_df)
# print(detail_df)
