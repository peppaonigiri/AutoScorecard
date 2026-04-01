import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# import dtreeviz
from IPython.display import display



class RulesFromTree:
    """
    从决策树分类器中提取规则。

    参数:
    tree_type (str): 决策树类型，可以是 'rf' (随机森林), 'exrf' (极端随机森林), 或 'dt' (决策树)。
    df (pandas.DataFrame): 包含训练集和测试集的数据集。
    x_list (list): 特征列名列表。
    y_col (str): 目标列名。
    group_col (str): 分组列名。
    params (dict): 决策树分类器参数。

    属性:
    clf (sklearn.tree.DecisionTreeClassifier): 决策树分类器。
    train_df (pandas.DataFrame): 训练集特征数据集。
    train_y (pandas.Series): 训练集目标列。
    test_df (pandas.DataFrame): 测试集特征数据集。
    test_y (pandas.Series): 测试集目标列。
    params (dict): 决策树分类器参数。
    rules (list): 提取的规则列表。
    rules_df (pandas.DataFrame): 提取的规则在训练集和测试集和整体上的表现df。
    rules_set_all (pandas.DataFrame): 规则集在整体上的表现df。
    rules_set_train (pandas.DataFrame): 规则集在训练集上的表现df。
    rules_set_test (pandas.DataFrame): 规则集在测试集上的表现df。
    catch_df (pandas.DataFrame): 规则集捕获的样本df。
    catch_index_list (list): 规则集捕获的样本索引列表。

    方法:
    set_params(params): 设置决策树分类器参数。
    fit(): 训练决策树分类器。
    extract_rules_from_tree(clf, train_list): 从决策树分类器中提取规则。
    get_rules(): 获取提取的规则列表。
    process_rules(): 处理单条规则表现。
    make_rules_df(): 生成提取的规则在训练集和测试集和整体上的表现df。
    make_rules_set(): 生成规则集相应的表现df。
    make_catch_df(): 生成规则集捕获的样本df。
    plot_tree_from_clf(): 绘制决策树。
    plot_tree_by_dtreeviz(): 使用 dtreeviz 库绘制决策树。



    tips:
    - 为了提高效率，可以使用多线程提取规则，但如果线程太多，且数据集较大，容易把内存拉爆
    - 可以调整决策树最小叶子节点样本数 min_samples_leaf 控制规则的覆盖度，和 最大深度 max_depth 来控制规则的准确性和复杂度
    - 对于随机森林和极端随机森林，可以调整 n_estimators 控制生成规则的数量
    - 可以调整 max_features 控制决策树分裂时考虑的特征数量，使得每棵子树包含不同特征，提高生成规则的多样性
    - 建议使用极端森林，训练速度快，且生成规则的多样性高，且规则在不同数据集上更有可能相对随机森林和决策树稳定
    """
    def __init__(self, tree_type, df, x_list, y_col, group_col, params=None):
        self.tree_type = tree_type
        self.x_list = x_list
        self.y_col = y_col
        self.group_col = group_col
        self.df = df
        
        if 'train' not in self.df[self.group_col].unique() or 'test' not in self.df[self.group_col].unique():
            raise ValueError('数据集分组列group_col中必须包含一个名为 train 和 test 的分组')
        
        print('正在准备数据集...')
        self.train_df = df.query(f'{self.group_col} == "train"')[x_list]
        self.train_y = df.query(f'{self.group_col} == "train"')[y_col]
        self.test_df = df.query(f'{self.group_col} == "test"')[x_list]
        self.test_y = df.query(f'{self.group_col} == "test"')[y_col]
        print('数据集准备完成')
        
        self.set_params(params)


    def set_params(self, params):
        """
        设置决策树分类器参数。

        参数:
        params (dict): 要更新的决策树分类器参数。
        """
        default_params = {
            'rf': {
                'bootstrap': True,
                'ccp_alpha': 0.0,
                'class_weight': None,
                'criterion': 'gini',
                'max_depth': 3,
                'max_features': 'sqrt',
                'max_leaf_nodes': None,
                'max_samples': None,
                'min_impurity_decrease': 0.0,
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'min_weight_fraction_leaf': 0.0,
                'n_estimators': 100,
                'n_jobs': None,
                'oob_score': False,
                'random_state': 2024,
                'verbose': 0,
                'warm_start': False
                },
            'exrf': {
                'bootstrap': True,
                'ccp_alpha': 0.0,
                'class_weight': None,
                'criterion': 'gini',
                'max_depth': 3,
                'max_features': 'sqrt',
                'max_leaf_nodes': None,
                'max_samples': None,
                'min_impurity_decrease': 0.0,
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'min_weight_fraction_leaf': 0.0,
                'n_estimators': 100,
                'n_jobs': None,
                'oob_score': False,
                'random_state': 2024,
                'verbose': 0,
                'warm_start': False
                },
            'dt': {
                'ccp_alpha': 0.0,
                'class_weight': None,
                'criterion': 'gini',
                'max_depth': 3,
                'max_features': None,
                'max_leaf_nodes': None,
                'min_impurity_decrease': 0.0,
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'min_weight_fraction_leaf': 0.0,
                'random_state': 2024,
                'splitter': 'best'
                }
        }
        
        self.params = default_params[self.tree_type]
        if params is not None and isinstance(params, dict):
            self.params.update(params)
            print('参数更新完成')


    def fit(self):
        """
        训练决策树分类器。
        """
        classifiers = {
            'rf': RandomForestClassifier,
            'exrf': ExtraTreesClassifier,
            'dt': DecisionTreeClassifier
        }
        if self.tree_type not in classifiers:
            raise ValueError('tree_type 必须是 rf, exrf 或 dt')
        
        self.clf = classifiers[self.tree_type](**self.params)
        print('正在训练决策树分类器...')
        self.clf.fit(self.train_df, self.train_y)
        print('训练完成')
        


    def extract_rules_from_tree(self, clf, train_list):
        """
        从决策树分类器中提取规则。

        参数:
        clf (sklearn.tree.DecisionTreeClassifier): 决策树分类器。
        train_list (list): 决策树分类器训练集特征列名列表。

        返回:
        list: 提取的规则列表。
        """
        # 检查输入参数
        if not hasattr(clf, 'tree_'):
            raise ValueError("输入的分类器不是一个有效的决策树分类器。")
        if not isinstance(train_list, list):
            raise ValueError("输入的训练集特征列名列表不是一个列表。")

        # 获取决策树的属性
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        value = clf.tree_.value

        # 初始化节点深度和叶子节点标志
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]

        # 遍历决策树节点
        while len(stack) > 0:
            node_id, depth = stack.pop()
            node_depth[node_id] = depth
            is_split_node = children_left[node_id] != children_right[node_id]
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        # 获取特征名称
        feature_name = [
            train_list[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in clf.tree_.feature
        ]

        # 初始化用于存储规则的列表
        ways = []
        depth = []
        feat = []
        nodes = []
        rules = []

        # 生成规则
        for i in range(n_nodes):
            if is_leaves[i]:
                while depth[-1] >= node_depth[i]:
                    depth.pop()
                    ways.pop()
                    feat.pop()
                    nodes.pop()
                if children_left[i - 1] == i:
                    #当前节点是上一个节点的左节点，则是小于等于阈值的情况
                    a = '{f} <= {th}'.format(f=feat[-1], th=round(threshold[nodes[-1]], 4))
                    ways[-1] = a
                    last = ' and '.join(ways)
                    rules.append(last)
                else:
                    #当前节点是上一个节点的右节点，则是大于阈值的情况
                    a = '{f} > {th}'.format(f=feat[-1], th=round(threshold[nodes[-1]], 4))
                    ways[-1] = a
                    last = ' and '.join(ways)
                    rules.append(last)
            else:
                # 非叶子节点
                if i == 0:
                    ways.append(round(threshold[i], 4))
                    depth.append(node_depth[i])
                    feat.append(feature_name[i])
                    nodes.append(i)
                else:
                    while depth[-1] >= node_depth[i]:
                        depth.pop()
                        ways.pop()
                        feat.pop()
                        nodes.pop()
                    if i == children_left[nodes[-1]]:
                        w = '{f} <= {th}'.format(f=feat[-1], th=round(threshold[nodes[-1]], 4))
                    else:
                        w = '{f} > {th}'.format(f=feat[-1], th=round(threshold[nodes[-1]], 4))
                    ways[-1] = w
                    ways.append(round(threshold[i], 4))
                    depth.append(node_depth[i])
                    feat.append(feature_name[i])
                    nodes.append(i)

        return rules
    

    def get_rules(self):
        """
        获取提取的规则列表。
        """
        self.fit()
        print('正在提取规则...')
        if self.tree_type in ['rf', 'exrf']:
            self.rules = []
            for tree in tqdm(self.clf.estimators_,desc='从子树中提取规则'):
                rule_single_tree = self.extract_rules_from_tree(tree, self.x_list)
                self.rules.extend(rule_single_tree)
        else:
            self.rules = self.extract_rules_from_tree(self.clf, self.x_list)
    

    def process_rule(self, rule, train_badrate, test_badrate, all_badrate, train_total, test_total, all_total):
        """
        处理单个规则的表现。

        参数:
        rule (str): 规则。
        train_badrate (float): 训练集坏样本比例。
        test_badrate (float): 测试集坏样本比例。
        all_badrate (float): 整体数据集坏样本比例。
        train_total (int): 训练集总样本数。
        test_total (int): 测试集总样本数。
        all_total (int): 整体数据集总样本数。

        返回:
        tuple: 规则、规则在训练集、测试集、整体数据集上的表现。
        """
        temp_train = self.train_df[self.train_df.eval(rule)]
        temp_test = self.test_df[self.test_df.eval(rule)]
        temp_all = self.df[self.df.eval(rule)]
        
        bad_train = self.train_y.loc[temp_train.index].sum()
        all_train = temp_train.shape[0]
        good_train = all_train - bad_train
        badrate_train = bad_train / all_train if all_train > 0 else 0
        lift_train = badrate_train / train_badrate if train_badrate > 0 else 0
        shot_rate_train = all_train / train_total if train_total > 0 else 0
        
        bad_test = self.test_y.loc[temp_test.index].sum()
        all_test = temp_test.shape[0]
        good_test = all_test - bad_test
        badrate_test = bad_test / all_test if all_test > 0 else 0
        lift_test = badrate_test / test_badrate if test_badrate > 0 else 0
        shot_rate_test = all_test / test_total if test_total > 0 else 0
        
        bad_all = temp_all[self.y_col].sum()
        all_all = temp_all.shape[0]
        good_all = all_all - bad_all
        badrate_all = bad_all / all_all if all_all > 0 else 0
        lift_all = badrate_all / all_badrate if all_badrate > 0 else 0
        shot_rate_all = all_all / all_total if all_total > 0 else 0

        # psi = sum((实际占比-预期占比)* ln(实际占比/预期占比))
        psi = (badrate_test-badrate_train) * np.log(badrate_test/badrate_train if badrate_train > 0 else 1)
        
        return (rule, [bad_train, good_train, all_train, badrate_train, lift_train, shot_rate_train,
                    bad_test, good_test, all_test, badrate_test, lift_test, shot_rate_test,
                    bad_all, good_all, all_all, badrate_all, lift_all, shot_rate_all, psi])

    def make_rules_df(self,use_thread=True,max_workers=5):
        """
        生成提取的规则在训练集和测试集和整体上的表现df。

        参数:
        use_thread (bool): 是否使用多线程提取规则。
        max_workers (int): 最大线程数。

        返回:
        pandas.DataFrame: 提取的规则在训练集和测试集和整体上的表现df。
        """
        self.get_rules()
        rules_dict = {}
        
        
        # 计算坏样本比例
        train_badrate = self.train_y.sum() / self.train_df.shape[0] if self.train_df.shape[0] > 0 else 0
        test_badrate = self.test_y.sum() / self.test_df.shape[0] if self.test_df.shape[0] > 0 else 0
        all_badrate = self.df[self.y_col].sum() / self.df.shape[0] if self.df.shape[0] > 0 else 0
        
        # 预计算总样本数
        train_total = self.train_df.shape[0]
        test_total = self.test_df.shape[0]
        all_total = self.df.shape[0]

        if use_thread:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务到线程池
                futures = [executor.submit(self.process_rule, rule, train_badrate, test_badrate, all_badrate, train_total, test_total, all_total) for rule in self.rules]
                
                # 使用 tqdm 监控任务完成的进度
                for future in tqdm(as_completed(futures), total=len(futures), desc='生成规则表现df'):
                    rule, result = future.result()
                    rules_dict[rule] = result
        else:
            for rule in tqdm(self.rules, desc='生成规则表现df'):
                rule, result = self.process_rule(rule, train_badrate, test_badrate, all_badrate, train_total, test_total, all_total)
                rules_dict[rule] = result

        # 生成规则表现df
        self.rules_df = pd.DataFrame.from_dict(rules_dict, orient='index')
        self.rules_df.reset_index(inplace=True)
        self.rules_df.columns = ['规则', '坏样本数(训练集)', '好样本数(训练集)', '总样本数(训练集)', '坏样本比例(训练集)', 'Lift(训练集)', '命中率(训练集)',
                                '坏样本数(测试集)', '好样本数(测试集)', '总样本数(测试集)', '坏样本比例(测试集)', 'Lift(测试集)', '命中率(测试集)',
                                '坏样本数(整体)', '好样本数(整体)', '总样本数(整体)', '坏样本比例(整体)', 'Lift(整体)', '命中率(整体)', 'PSI']
        self.rules_df.sort_values(by='坏样本比例(训练集)', ascending=False, inplace=True)
        self.rules_df.reset_index(drop=True, inplace=True)
        self.rules_df['lift_衰减'] = self.rules_df['Lift(训练集)'] - self.rules_df['Lift(测试集)']
        return self.rules_df


    def process_set(self, df, best_rules_by_sort, rules_df, strategy=1, desc='all'):
        """
        处理单个数据集。

        参数:
        df (pandas.DataFrame): 待处理数据集。
        best_rules_by_sort (list): 按照Lift衰减排序的最佳规则列表。
        rules_df (pandas.DataFrame): 规则在训练集和测试集和整体上的表现df。
        strategy (int): 规则集生成策略：
                        0: 不做区分,所有规则,
                        1: 单个规则必须有新增样本,
                        2: 单个规则必须有新增坏样本,
                        3: 单个规则新增样本中坏样本比例高于整体坏样本比例。

        desc (str): 处理数据集的描述。

        返回:
        pandas.DataFrame: 处理后的结果。
        """
        # 计算训练集的总样本数和坏样本比例
        train_total = self.train_df.shape[0]

        # 计算整体数据集的总样本数和坏样本比例
        all_total = df.shape[0]
        all_badrate = df[self.y_col].sum() / all_total if all_total > 0 else 0


        rule_increase = {}
        index_set = set()
        
        # 生成规则集并计算新增样本数
        for rule in tqdm(best_rules_by_sort, desc=f'生成规则集_{desc}'):
            # 获得单个规则在训练集和测试集上的 Lift 衰减
            lift_reduce = rules_df.loc[rules_df['规则'] == rule, 'lift_衰减'].values[0]

            temp_series = df.eval(rule)
            true_indexes_set = set(temp_series[temp_series].index.tolist())
            increase_indexes = list(true_indexes_set - index_set)
            if strategy == 0:
                pass
            # 如果单独规则没有新增样本，则跳过
            elif strategy == 1:
                if len(increase_indexes) == 0:
                    continue
            # 如果单独规则新增样本中没有坏样本，则跳过
            elif strategy == 2:
                if df.loc[increase_indexes, self.y_col].sum() == 0:
                    continue
            # 如果单独规则新增样本中坏样本比例低于整体坏样本比例，则跳过
            elif strategy == 3:
                inc_badrate = df.loc[increase_indexes, self.y_col].sum() / len(increase_indexes) if len(increase_indexes) > 0 else 0
                if inc_badrate < all_badrate:
                    continue
            else:
                raise ValueError('strategy 参数错误')
            inc_all = len(increase_indexes)
            inc_bad = df.loc[increase_indexes, self.y_col].sum()
            inc_good = inc_all - inc_bad
            rule_increase[rule] = [inc_all, inc_bad, inc_good, lift_reduce]
            index_set.update(increase_indexes)
        
        # 将结果转换为 DataFrame
        rules_set_df = pd.DataFrame.from_dict(rule_increase, orient='index').reset_index()
        rules_set_df.columns = ['规则', '新增总样本数', '新增坏样本数', '新增好样本数', 'Lift衰减']
        
        # 计算累计和
        cumulative_sum = rules_set_df[['新增总样本数', '新增坏样本数', '新增好样本数']].expanding().sum().copy()
        cumulative_sum.columns = ['累计总样本数', '累计坏样本数', '累计好样本数']
        cumulative_sum['累计坏样本比例'] = (cumulative_sum['累计坏样本数'] / cumulative_sum['累计总样本数']).where(cumulative_sum['累计总样本数']>0, 0)
        cumulative_sum['累计命中率'] = cumulative_sum['累计总样本数'] / all_total if all_total > 0 else 0
        cumulative_sum['累计Lift'] = cumulative_sum['累计坏样本比例'] / all_badrate if all_badrate > 0 else 0
        arr_all_bad = np.full(len(cumulative_sum), df[self.y_col].sum()) - cumulative_sum['累计坏样本数']
        arr_all_total = np.full(len(cumulative_sum), all_total) - cumulative_sum['累计总样本数']
        cumulative_sum['策略后整体坏样本比例'] = (arr_all_bad / arr_all_total).where(arr_all_total>0, 0)
        cumulative_sum['策略后整体Lift'] = cumulative_sum['策略后整体坏样本比例'] / all_badrate if all_badrate > 0 else 0
        
        # 合并累计和结果到原始 DataFrame
        rules_set_df = pd.concat([rules_set_df, cumulative_sum], axis=1)

        return rules_set_df

    def make_rules_set(self, strategy=1, sort_by='test', lift_diff=None, psi_cut = None, rule_list=None):
        """
        生成规则集。

        参数:
        strategy (int): 规则集生成策略：
                        0: 不做区分,所有规则,
                        1: 单个规则必须有新增样本,
                        2: 单个规则必须有新增坏样本,
                        3: 单个规则新增样本中坏样本比例高于整体坏样本比例。
        sort_by (str): 规则集叠加顺序依据。'train': 训练集上的坏样本比例, 'test': 测试集上的坏样本比例, 'all': 整体数据集上的坏样本比例。
        lift_diff (float): 控制lift衰减阈值,若为None则不控制。
        psi_cut (float): 控制psi值阈值,若为None则不控制。
        rule_list (list): 指定叠加顺序的规则列表,若为None则按指定策略排序。

        返回:
        pandas.DataFrame: 整体的规则集与性能指标,若要查看train和test上的规则集,请调用 self.rules_set_train 和 self.rules_set_test。
        """
        if not hasattr(self, 'rules_df'):
            raise ValueError("请先调用 make_rules_df 方法生成规则表现df。")
        
        rules_df = self.rules_df.copy()

        if isinstance(rule_list, list):
            # 按指定顺序叠加规则
            best_rules_by_sort = rule_list
        elif rule_list is None:
            # 是否控制lift衰减
            if isinstance(lift_diff, float):
                rules_df = rules_df.query('lift_衰减 <= @lift_diff')
            elif lift_diff is None:
                pass
            else:
                raise ValueError('lift_diff 参数错误')
            
            # 是否控制psi值
            if isinstance(psi_cut, float):
                rules_df = rules_df.query('PSI <= @psi_cut')
            elif psi_cut is None:
                pass
            else:
                raise ValueError('psi_cut 参数错误')

            # 按指定策略排序规则
            if sort_by == 'train':
                best_rules_by_sort = rules_df.sort_values(by='坏样本比例(训练集)', ascending=False)['规则'].to_list()
            elif sort_by == 'test':
                best_rules_by_sort = rules_df.sort_values(by='坏样本比例(测试集)', ascending=False)['规则'].to_list()
            elif sort_by == 'all':
                best_rules_by_sort = rules_df.sort_values(by='坏样本比例(整体)', ascending=False)['规则'].to_list()
            else:
                raise ValueError('sort_by 参数错误')
        else:
            raise ValueError('rule_list 参数错误')
        
        # 先按照策略处理在整体数据集
        self.rules_set_all = self.process_set(self.df, best_rules_by_sort, rules_df, strategy=strategy, desc='all')
        # 获取在整体数据集上的最佳规则集
        temp_rules = self.rules_set_all['规则'].tolist()
        
        # 按照temp_rules处理训练集和测试集，且策略为0不做任何控制
        self.rules_set_train = self.process_set(self.df.query('index in @self.train_df.index'), temp_rules, rules_df, strategy=0, desc='train')
        self.rules_set_test = self.process_set(self.df.query('index in @self.test_df.index'), temp_rules, rules_df, strategy=0, desc='test')

        # 计算规则集在训练集-测试集的PSI
        arr_shot_diff = self.rules_set_test['累计命中率'] - self.rules_set_train['累计命中率']
        arr_shot_div = (self.rules_set_test['累计命中率'] / self.rules_set_train['累计命中率']).where(self.rules_set_train['累计命中率'] != 0, 1)
        self.rules_set_all['训练-测试PSI'] = (arr_shot_diff * np.log(arr_shot_div))

        return self.rules_set_all


    def make_catch_df(self, rule_list=None):
        """
        生成捕获明细df。

        参数:
        rule_list (list): 指定生成捕获明细的规则列表,若为None则使用self全部规则。

        返回:
        pandas.DataFrame: 捕获明细df。
        """
        if rule_list is None:
            rule_list = self.rules

        index_set = set()
        col_set = set()
        catch_rule_dict = {}
        for rule in tqdm(rule_list, desc='生成捕获明细df'):
            temp_series = self.df.eval(rule)
            true_indexes_set = set(temp_series[temp_series].index.tolist())
            increase_indexes = list(true_indexes_set - index_set)
            catch_rule_dict[rule] = increase_indexes
            index_set.update(increase_indexes)
            for single_rule in rule.split(' and '):
                if '<=' in single_rule:
                    col_set.update([single_rule.split(' <= ')[0]])
                elif '>' in single_rule:
                    col_set.update([single_rule.split(' > ')[0]])
        
        # df中用到的列
        col_list = list(col_set)
        # 命中的index,与self.df的index对应
        catch_index_list = list(index_set)
        # 捕获明细df
        catch_df = self.df.loc[catch_index_list, col_list].copy()
        # 命中规则列，叠加关系，表示排序在此规则之前的规则均未命中，添加此条规则后命中该样本
        for rule in catch_rule_dict:
            catch_df.loc[catch_rule_dict[rule], '命中规则'] = rule
        self.catch_df = catch_df
        self.catch_index_list = catch_index_list

        return self.catch_df




    def plot_tree_from_clf(self, num_of_trees=0):
        """
        绘制决策树分类器的决策树。

        参数:
        num_of_trees (int): 绘制第几个决策树。
        """
        if self.clf is None:
            raise ValueError('请先训练决策树分类器')
        
        class_names = [str(c) for c in self.clf.classes_]
        
        plt.figure(figsize=(10, 10),dpi=300)
        if self.tree_type in ['rf', 'exrf']:
            if num_of_trees >= len(self.clf.estimators_):
                raise ValueError('指定的树索引超出范围')
            plot_tree(self.clf.estimators_[num_of_trees], filled=True, rounded=True, class_names=class_names, feature_names=self.x_list)
        else:
            plot_tree(self.clf, filled=True, rounded=True, class_names=class_names, feature_names=self.x_list)
        plt.show()
    

    # def plot_tree_by_dtreeviz(self, clf=None, train_df=None, train_y=None, x_list=None, num_of_trees=0, single_sample=None, viz_params=None, save_path=None, show_img=2):
    #     '''
    #     使用 dtreeviz 绘制决策树,需要先安装Graphviz,并配置环境变量,例如添加D:\Program Files\Graphviz\bin到系统变量Path中。

    #     参数:
    #     clf (sklearn.tree.DecisionTreeClassifier): 决策树分类器,若为None则使用实例化中的默认的分类器。
    #     train_df (pandas.DataFrame): 训练集数据。
    #     train_y (pandas.Series): 训练集标签。
    #     x_list (list): 特征名列表。
    #     num_of_trees (int): 树索引。
    #     single_sample (dict): 单个样本。
    #     viz_params (dict): viz_model参数,如 show_just_path=False, fancy=True。
    #     save_path (str): 保存路径。
    #     show_img (int): 1: 打开网页显示图片, 2: 直接在jupyter中显示图片。
    #     '''
    #     try:
    #         if clf is None:
    #             clf = self.clf.estimators_[num_of_trees]
    #             train_df = self.train_df
    #             train_y = self.train_y
    #             x_list = self.x_list

    #         viz_model = dtreeviz.model(model=clf,
    #                                    X_train=train_df,
    #                                    y_train=train_y,
    #                                    target_name='label',
    #                                    feature_names=x_list,
    #                                    class_names={0: 'good', 1: 'bad'})

    #         if viz_params is None:
    #             viz_params = {'show_just_path': False, 'fancy': True}

    #         v = viz_model.view(x=single_sample, **viz_params)

    #         if show_img == 1:
    #             v.show()
    #         elif show_img == 2:
    #             display(v)

    #         if save_path is not None:
    #             v.save(save_path)
    #     except Exception as e:
    #         print(f"绘制决策树时发生错误: {e}")




# 用例
# df = pd.read_csv('data.csv')
# x_list = ['x1', 'x2', 'x3']
# y_col = 'y'
# group_col = 'target'
# rf = RulesFromTree('rf', df, x_list, y_col, group_col)
# rf.set_params({'n_estimators': 100})
# rf.make_rules_df(use_thread=True, max_workers=20)
# rf.make_rules_set(strategy=1)
# print(rf.rules_df)
# print(rf.rules)
# rf.plot_tree_from_clf(1)

