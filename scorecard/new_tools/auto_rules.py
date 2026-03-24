
import pandas as pd
import prim
import copy
from sklearn.tree import DecisionTreeClassifier
from auto_rules.woe_badrate import cal_bad_rate, cal_woe, cal_bad_pcnt
from sklearn.metrics import precision_score, roc_curve, f1_score, recall_score, average_precision_score, accuracy_score
import itertools
import numpy as np
import re
import joblib
from multiprocessing import Manager, Pool, cpu_count, Lock, Value
from sklearn import tree
import toad
pd.set_option('max_colwidth', 300)  # 设定列最大长度为300


class AutoRules(object):
    def __init__(self, FILE_PATH, SAVE_PATH, ex_lst, date_col, num_var_rule=[2], dep='label'):
        if isinstance(FILE_PATH, str):
            if '.pkl' in FILE_PATH:
                self.dev = pd.read_pickle(FILE_PATH)
            elif '.csv' in FILE_PATH:
                self.dev = pd.read_csv(FILE_PATH, encoding='utf-8-sig')
            elif '.xlsx' in FILE_PATH:
                self.dev = pd.read_excel(FILE_PATH)
        else:
            self.dev = FILE_PATH

        self.date_col = date_col
        self.train = copy.deepcopy(self.dev)
        self.SAVE_PATH = SAVE_PATH
        self.ex_lst = ex_lst
        self.ft_lst = [i for i in self.dev.columns if i not in self.ex_lst]
        self.cat_var = [i for i in self.ft_lst if self.dev.dtypes[i] == "object"]
        self.dep = dep
        self.groups = []
        self.num_var_rule = num_var_rule
        self.encode_dict = {}
        self.p = Pool(processes=60)
        print("训练数据集大小:", self.dev.shape)
        print("类别型变量数量:", len(self.cat_var))
        print("数值型变量数量:", len(self.ft_lst) - len(self.cat_var))

        self.rules_code_py = None
        self.rules_code_sas = None
        self.rules_value = None
        self.com_rules_value = None
        self.model_type = None

        self.combiner = toad.transform.Combiner()
        self.transfer = toad.transform.WOETransformer()
        self.dev_bin = None
        self.dev_woe = None
        self.result = None
        self.cat2num_type = None
        self.num2bin_type = None
        self.num2bin_ft = None

    def category2numeric(self, method='bad_rate'):
        self.cat2num_type = method
        self.train.fillna('nan', inplace=True)
        # if method == 'dummy':
        #     self.train = pd.get_dummies(self.train, columns=self.cat_var, prefix=self.cat_var, prefix_sep='_',
        #                                 dummy_na=False, drop_first=False)
        if method == 'bad_pcnt':
            for i in self.cat_var:
                self.encode_dict[i] = cal_bad_pcnt(self.train, i, self.dep)
                self.train[i] = self.train[i].replace(self.encode_dict[i].keys(), self.encode_dict[i].values())
        elif method == 'bad_rate':
            for i in self.cat_var:
                self.encode_dict[i] = cal_bad_rate(self.train, i, self.dep)
                self.train[i] = self.train[i].replace(self.encode_dict[i].keys(), self.encode_dict[i].values())
        else:
            for i in self.cat_var:
                self.encode_dict[i] = cal_woe(self.train, i, self.dep)
                self.train[i] = self.train[i].replace(self.encode_dict[i].keys(), self.encode_dict[i].values())

    def numeric2bin(self, ft_lst, type, **kwargs):
        self.num2bin_type = type
        self.num2bin_ft = ft_lst

        ex_lst = [i for i in self.train if i not in ft_lst]
        self.combiner.fit(self.train, self.train[self.dep], exclude=ex_lst, **kwargs)
        self.train[ft_lst] = self.combiner.transform(self.train[ft_lst])
        if type == 'woe':
            self.transfer.fit(self.train, self.train[self.dep], exclude=ex_lst)
            self.train[ft_lst] = self.transfer.transform(self.train[ft_lst])

    def create_groups(self):
        for i in self.num_var_rule:
            self.groups.extend(list(itertools.combinations(self.ft_lst, i)))
        print("变量组合数:", len(self.groups))

    def create_rules(self, method, **kwargs):
        if len(self.groups) == 0:
            self.create_groups()

        with Manager() as manager:
            rules_code_py = manager.list()
            rules_code_sas = manager.list()
            rules_value = manager.list()

            for k in self.groups:
                x = self.train[list(k)]
                y = self.train[self.dep]
                if method == 'tree':
                    self.model_type = 'tree'
                    clf = tree.DecisionTreeClassifier(**kwargs).fit(x, y)
                    self.p.apply_async(tree.export_text(decision_tree=clf,
                                                        feature_names=list(x.columns),
                                                        rules_code_py=rules_code_py,
                                                        rules_code_sas=rules_code_sas,
                                                        rules_value=rules_value),
                                       args=())
                elif method == 'prim':
                    self.model_type = 'prim'
                    prim_ = prim.Prim(x=x, y=y, **kwargs)
                    self.p.apply_async(prim_export_text(model=prim_, df=self.train.copy(),
                                                        rules_code_py=rules_code_py,
                                                        rules_code_sas=rules_code_sas,
                                                        rules_value=rules_value),
                                       args=())

                elif method == 'ar':
                    pass

                elif method == 'n-gram':
                    pass

            self.p.close()
            self.p.join()
            self.rules_code_py = pd.DataFrame(list(rules_code_py), columns=['Rule', 'Good', 'Bad'])
            self.rules_code_sas = pd.DataFrame(list(rules_code_sas), columns=['Rule', 'Good', 'Bad'])
            self.rules_value = pd.DataFrame(list(rules_value), columns=['Rule', 'Rule_Var', 'Good', 'Bad'])
            self.rules_value['code_py'] = self.rules_code_py['Rule']
            self.rules_value['code_sas'] = self.rules_code_sas['Rule']
            self.rules_value['rule_complexity'] = self.rules_value[['Rule_Var']].apply(
                lambda x: len(x.Rule_Var.split(',')), axis=1)

    def evaluate_statistic(self, df):
        df['Good'] = df['Good'].astype(np.float)
        df['Bad'] = df['Bad'].astype(np.float)
        df['Total'] = df['Good'] + df['Bad']
        df['Odds'] = df['Good'] / df['Bad']
        df['Bad_Rate'] = df['Bad'] / df['Total']
        df['Catch_Rate'] = df['Bad'] / sum(self.train[self.dep])
        df['Trigger_Rate'] = df['Total'] / self.train.shape[0]
        df['Lift'] = df['Bad_Rate'] / (sum(self.train[self.dep]) / self.train.shape[0])
        df = df[~df.Total.isin([self.train.shape[0], 0])]
        df = df.drop_duplicates(subset=['code_py', 'Good', 'Bad'], keep='first')
        df.sort_values(by='Bad_Rate', ascending=False, inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df

    def business_analysis(self, df, lost, interest):
        total = self.train.shape[0]
        total_bad = self.train[self.dep].sum()
        df['通过率'] = (total - df['Total']) / total
        df['通过人群坏账率'] = (total_bad - df['Bad']) / (total - df['Total'])
        if isinstance(lost, (float, pd.Series, list)):
            df['lost'] = lost
            df['interest'] = interest
            df['业务收益推断'] = df['Bad'] * df['lost'] - df['Good'] * df['interest']
        return df

    def evaluate_model(self, t):
        df = self.train.copy()

        rule_lst = self.rules_value['code_py'].tolist()
        print(len(rule_lst))

        if self.date_col:
            df['month_time'] = df[self.date_col].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m'))
            month_lst = sorted(list(df['month_time'].unique()))
            psi_keys = ['psi_' + str(i) for i in month_lst]

        cols = df.columns.tolist()

        precision, recall, f1, average_precision, accuracy, psi = [], [], [], [], [], []

        for rule in rule_lst:
            exec(rule)

        new_cols = [i for i in df if i not in cols]

        for ft in new_cols:
            precision.append(precision_score(df[self.dep], df[ft]))
            recall.append(recall_score(df[self.dep], df[ft]))
            f1.append(f1_score(df[self.dep], df[ft]))
            average_precision = average_precision_score(df[self.dep], df[ft])
            accuracy = accuracy_score(df[self.dep], df[ft])

            if self.date_col:
                datasets = {}
                for i in month_lst:
                    datasets[i] = df[df['month_time'] == i][ft].tolist()

                if self.model_type == 'tree':
                    psi.append([toad.metrics.PSI(df[ft], i) for i in datasets.values()])
                elif self.model_type == 'prim':
                    psi.append([toad.metrics.PSI(df[ft], i) for i in datasets.values()])
            else:
                psi_keys = 'psi'

        t['accuracy'] = accuracy
        t['precision'] = precision
        t['recall'] = recall
        t['f1_score'] = f1
        t['average_precision'] = average_precision
        t[psi_keys] = psi if isinstance(psi_keys, list) else None

        return t, df

    def evaluate(self, lost=90, interest=50):
        self.rules_value = self.evaluate_statistic(self.rules_value)
        self.rules_value = self.business_analysis(self.rules_value, lost, interest)
        self.rules_value, self.result = self.evaluate_model(self.rules_value)

    def get_python_code(self):
        self.rules_code_py = self.rules_value.copy()

        _rules_code_py_ = ""
        for i in self.rules_code_py['code_py']:
            _rules_code_py_ += i
        py_codes = open(self.SAVE_PATH + 'rules_code_py.txt', mode='w')
        py_codes.write(_rules_code_py_)
        py_codes.close()

        # print(self.FILE_PATH + "_rules_code_py_:", _rules_code_py_)

        self.rules_code_py.to_csv(self.SAVE_PATH + "rules_code_py_table.csv", index=False)
        joblib.dump(_rules_code_py_, self.SAVE_PATH + 'rules_code_py.pkl')
        joblib.dump(self.rules_code_py, self.SAVE_PATH + 'rules_code_py_table.pkl')

    def get_sas_code(self):
        self.rules_code_sas = self.rules_value.copy()
        # b['len'] = b.apply(lambda x: len(re.findall(r"If (.+?) Then", x.Rule)), axis=1)
        # b = b[b.len > 0]
        self.rules_code_sas['_rule_'] = self.rules_code_sas.apply(
            lambda x: re.findall(r"If (.+?) Then", x.code_sas)[0], axis=1)
        self.rules_code_sas = self.rules_code_sas.drop_duplicates(['_rule_'])
        self.rules_code_sas = self.rules_code_sas.drop(['_rule_'], axis=1)

        _rules_code_sas_ = ""
        for i in self.rules_code_sas['code_sas']:
            _rules_code_sas_ += i
        sas_codes = open(self.SAVE_PATH + 'rules_code_sas.txt', mode='w')
        sas_codes.write(_rules_code_sas_)
        sas_codes.close()

        # print(self.FILE_PATH + "_rules_code_sas_:", _rules_code_sas_)

        self.rules_code_sas.to_csv(self.SAVE_PATH + "rules_code_sas_table.csv", index=False)
        joblib.dump(_rules_code_sas_, self.SAVE_PATH + 'rules_code_sas.pkl')
        joblib.dump(self.rules_code_sas, self.SAVE_PATH + 'rules_code_sas_table.pkl')

    def get_rules(self):
        """
        :return: 导出规则及相关参数
        """
        self.rules_value.to_csv(self.SAVE_PATH + "rules_value.csv", index=False)
        joblib.dump(self.rules_value, self.SAVE_PATH + 'rules_value.pkl')

    @staticmethod
    def rules_table(data, OOT_rules_value, Rule, TARGET):
        tmp_data_01 = data[data[Rule] == 1]
        Bad = sum(tmp_data_01[TARGET])
        Total = tmp_data_01.shape[0]
        Good = Total - Bad
        Bad_Rate = Bad / Total
        Catch_Rate = Bad / sum(data[TARGET])
        Trigger_Rate = Total / data.shape[0]
        Lift = Bad_Rate / (sum(data[TARGET]) / data.shape[0])
        new = pd.DataFrame({'Rule': [Rule], 'OOT_Good': [Good], 'OOT_Bad': [Bad], 'OOT_Total': [Total],
                            'OOT_Bad_Rate': [Bad_Rate], 'OOT_Catch_Rate': [Catch_Rate],
                            'OOT_Trigger_Rate': [Trigger_Rate], 'OOT_Lift': [Lift]})
        OOT_rules_value = OOT_rules_value.append(new, ignore_index=True)
        return OOT_rules_value

    def get_test_result(self, test):
        if isinstance(test, str):
            if '.pkl' in test:
                test = pd.read_pickle(test)
            elif '.csv' in test:
                test = pd.read_csv(test, encoding='utf-8-sig')
            elif '.xlsx' in test:
                test = pd.read_excel(test)

        df = test.fillna('nan')

        encode_dict = {}
        # if method == 'dummy':
        #     df = pd.get_dummies(self.train, columns=self.cat_var, prefix=self.cat_var, prefix_sep='_',
        #                                 dummy_na=False, drop_first=False)
        if self.cat2num_type == 'bad_pcnt':
            for i in self.cat_var:
                encode_dict[i] = cal_bad_pcnt(df, i, self.dep)
                for k, v in self.encode_dict.items():
                    encode_dict[k] = v
                df[i] = df[i].replace(encode_dict[i].keys(), encode_dict[i].values())
        elif self.cat2num_type == 'bad_rate':
            for i in self.cat_var:
                encode_dict[i] = cal_bad_rate(df, i, self.dep)
                for k, v in self.encode_dict.items():
                    encode_dict[k] = v
                df[i] = df[i].replace(encode_dict[i].keys(), encode_dict[i].values())
        elif self.cat2num_type == 'woe':
            for i in self.cat_var:
                encode_dict[i] = cal_woe(df, i, self.dep)
                for k, v in self.encode_dict.items():
                    encode_dict[k] = v
                df[i] = df[i].replace(encode_dict[i].keys(), encode_dict[i].values())

        if self.num2bin_type == 'bin':
            df[self.num2bin_ft] = self.combiner.transform(df[self.num2bin_ft])
        elif self.num2bin_type == 'woe':
            df[self.num2bin_ft] = self.combiner.transform(df[self.num2bin_ft])
            df[self.num2bin_ft] = self.transfer.transform(df[self.num2bin_ft])


        # rules_code_py_table = joblib.load('rules_code_py_table.pkl')
        # _rules_code_py_table_ = rules_code_py_table[rules_code_py_table.Lift > Lift_level]
        # _rules_code_py_table_max = _rules_code_py_table_[_rules_code_py_table_['Bad'] >= Min_Bad_Nums]

        _rules_ = []
        test_rules_value = pd.DataFrame(
            columns=['Rule', 'OOT_Good', 'OOT_Bad', 'OOT_Total', 'OOT_Bad_Rate', 'OOT_Catch_Rate',
                     'OOT_Trigger_Rate', 'OOT_Lift'])
        for i in self.rules_value['code_py']:
            _rule_ = re.findall(r"df.'(.+?)'] = np.where.*", i)[0]
            _rules_.append(_rule_)
            exec(i)
            test_rules_value = self.rules_table(df, test_rules_value, _rule_, self.dep)

        self.com_rules_value = pd.merge(self.rules_value, test_rules_value, on='Rule')
        self.com_rules_value["Lift_Rate_OfChang"] = self.com_rules_value.apply(
            lambda x: (x.OOT_Lift - x.Lift) / x.Lift if x.Lift != 0 else np.nan, axis=1)
        self.com_rules_value["Bad_RateR_OfChang"] = self.com_rules_value.apply(
            lambda x: (x.OOT_Bad_Rate - x.Bad_Rate) / x.Bad_Rate if x.Bad_Rate != 0 else np.nan, axis=1)
        self.com_rules_value = self.com_rules_value[
            ['Rule', 'Rule_Var', 'Good', 'Bad', 'Total', 'Bad_Rate', 'Catch_Rate', 'Trigger_Rate', 'Lift',
             'OOT_Good', 'OOT_Bad', 'OOT_Total', 'OOT_Bad_Rate', 'OOT_Catch_Rate', 'OOT_Trigger_Rate', 'OOT_Lift',
             "Bad_RateR_OfChang", "Lift_Rate_OfChang"]]
        self.com_rules_value.to_csv(self.SAVE_PATH + "com_rules_value.csv")
        joblib.dump(self.com_rules_value, self.SAVE_PATH + 'com_rules_value.pkl')


def prim_export_text(model, df, rules_code_py, rules_code_sas, rules_value):
    py_rules_fmt = "df['{}'] = np.where({},1,0)\n"
    sas_rules_fmt = "If {} Then var_{}=1 Else var_{}=0; Lable var_{}='{}';\n"

    rules = []

    box = model.find_box()

    if box.limits.shape[0]:
        for i in range(box.limits.shape[0]):
            ft = box.limits.index[i]
            if box.limits.loc[ft, 'min'] == box.limits.loc[ft, 'max']:
                rules.append('(df.' + ft + '==' + str(box.limits.loc[ft, 'min']) + ')')
            else:
                rules.append('(df.' + ft + '>=' + str(box.limits.loc[ft, 'min']) + ')')
                rules.append('(df.' + ft + '<=' + str(box.limits.loc[ft, 'max']) + ')')

        rule = '&'.join(rules)
        ft_name = rule.replace('df.', '')
        rule_sas = re.sub(r'&', "and", rule)
        _var_num_ = len(rules_code_sas) + 1

        exec(py_rules_fmt.format(ft_name, rule))

        bad = df.loc[df[ft_name] == 1, 'label'].sum()
        good = df[df[ft_name] == 1].shape[0] - bad

        rules_code_py.append([py_rules_fmt.format(ft_name, rule), good, bad])
        rules_code_sas.append([sas_rules_fmt.format(rule_sas, _var_num_, _var_num_, _var_num_, rule[:-1]), good, bad])
        rules_value.append([ft_name, str(box.limits.index.tolist()).replace("'", ''), good, bad])

