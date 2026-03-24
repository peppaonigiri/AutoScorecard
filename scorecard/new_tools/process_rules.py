import pandas as pd
import numpy as np
from rules_from_tree import RulesFromTree
from pathlib import Path

class process_rules:
    def __init__(self, df, ex_list, time_col, y_col, fail_path):
        
        self.ex_list = ex_list + ['target']
        self.time_col = time_col
        self.y_col = y_col
        self.fail_path = Path(fail_path)

        x_list = [col for col in df.columns if col not in ex_list]
        self.x_list = x_list

        df.sort_values(by=time_col, inplace=True)
        df.reset_index(drop=True, inplace=True)

        cut_index = int(len(df)*0.7)
        df.loc[:cut_index, 'target'] = 'train'
        df.loc[cut_index:, 'target'] = 'test'

        group_col = 'target'

        self.rft = RulesFromTree('dt',df,x_list,y_col,group_col)
        self.rft.set_params({'max_depth':3,'min_samples_leaf': 200,'min_samples_split': 200,'max_features': 0.5})


    def get_rules_df(self):
        self.rules_df = self.rft.make_rules_df(use_thread=False,max_workers=1)
        self.rules_df.to_excel(self.fail_path.joinpath('rules_df.xlsx'), index=False)
    
    def get_rules_set(self):
        self.rules_set_all = self.rft.make_rules_set(strategy=3,sort_by='test',lift_diff=0.1,psi_cut=0.1,rule_list=None)
        self.rules_set_train = self.rft.rules_set_train
        self.rules_set_test = self.rft.rules_set_test

        self.rules_set_all.to_excel(self.fail_path.joinpath('rules_set_all.xlsx'), index=False)
        self.rules_set_train.to_excel(self.fail_path.joinpath('rules_set_train.xlsx'), index=False)
        self.rules_set_test.to_excel(self.fail_path.joinpath('rules_set_test.xlsx'), index=False)

    def get_catch_df(self, rules_list=None):
        if rules_list is None:
            rules_list = self.rules_set_all['规则'].to_list()

        self.catch_df = self.rft.make_catch_df(rules_list)
        self.catch_df.to_excel(self.fail_path.joinpath('catch_df.xlsx'), index=False)

