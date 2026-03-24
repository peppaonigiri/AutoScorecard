import os
import sys
import ast

import pandas as pd
import numpy as np

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from ParameterRegulation.Optuna import run, objective_fix_sample
from sklearn.ensemble import StackingClassifier

import warnings
warnings.filterwarnings("ignore")


class TzStacking():
    """
    Stacking集成学习类，支持多模型融合和两层训练
    """
    
    def __init__(self, df, ex_lst, model_name_lst=['xgb', 'lgb'], second_model='lr', target_col='target',dep = 'label'):
        """
        初始化Stacking模型
        
        Parameters:
        -----------
        df : pd.DataFrame
            训练数据
        ex_lst : list
            需要排除的特征列表
        model_name_lst : list, default=['xgb', 'lgb']
            第一层模型名称列表
        second_model : str, default='lr'
            第二层模型名称
        target_col : str, default='target'
            目标列名称
        """
        self.df = df.copy()
        self.ex_lst = ex_lst
        self.keep_lst = [i for i in df.columns if i not in ex_lst]
        self.model_name_lst = model_name_lst
        self.second_model = second_model
        self.target_col = target_col
        self.dep = dep
        self.params_dict = {}
        self.skf = None
        self.fold_name_lst = None
        self.second_params = None
        self.proba_lst = None
        self.second_df = None
        self.second_df_stacking = None


    def model_init(self, model_name, params):
        """
        初始化模型
        
        Parameters:
        -----------
        model_name : str
            模型名称 ('xgb', 'lgb', 'lgbm', 'lr')
        params : dict
            模型参数字典
            
        Returns:
        --------
        model : sklearn/xgboost/lightgbm model
            初始化的模型对象
        """
        if model_name == 'xgb':
            model = xgb.XGBClassifier(**params)
        elif model_name == 'lr':
            model = LogisticRegression(**params)
        elif model_name in ['lgb', 'lgbm']:
            model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
        return model

    def skf_create(self, n_splits=5, random_state=2025):
        """
        创建分层K折交叉验证对象
        
        Parameters:
        -----------
        n_splits : int, default=5
            折数
        random_state : int, default=2025
            随机种子
        """
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.fold_name_lst = [f'fold{i}' for i in range(1, n_splits + 1)]
        X = self.df.loc[self.df[self.target_col] == 'train']
        y = self.df.loc[self.df[self.target_col] == 'train', self.dep]

        original_indices = X.index.values

        for fold, (train_idx, test_idx) in enumerate(self.skf.split(X, y)):
            train_orig_indices = original_indices[train_idx]
            test_orig_indices = original_indices[test_idx]
            
            self.df.loc[train_orig_indices, f'fold{fold+1}'] = 'fold_train'
            self.df.loc[test_orig_indices, f'fold{fold+1}'] = 'fold_test'

    def _normalize_param_list(self, param_value, param_name, model_count):
        """
        标准化参数列表，确保参数数量与模型数量匹配
        
        Parameters:
        -----------
        param_value : int or list
            参数值（可以是单个值或列表）
        param_name : str
            参数名称（用于错误提示）
        model_count : int
            模型数量
            
        Returns:
        --------
        list : 标准化后的参数列表
        """
        if isinstance(param_value, int):
            return [param_value] * model_count
        elif isinstance(param_value, list):
            if len(param_value) < model_count:
                print(f'{param_name}参数长度与模型数量不匹配，将默认使用第一个值进行训练！')
                return [param_value[0]] * model_count
            return param_value
        else:
            raise TypeError(f'{param_name}参数类型错误，应为int或list')

    def first_train(self, epoch=[1000], strategy=[1348], max_depth=[4]):
        """
        第一层模型训练和超参数优化
        
        Parameters:
        -----------
        epoch : int or list, default=[1000]
            训练轮数，可以是单个值或与模型数量匹配的列表
        strategy : int or list, default=[1348]
            策略参数，可以是单个值或与模型数量匹配的列表
        max_depth : int or list, default=[4]
            最大深度，可以是单个值或与模型数量匹配的列表
        """
        # 标准化参数列表
        epoch_list = self._normalize_param_list(epoch, 'epoch', len(self.model_name_lst))
        strategy_list = self._normalize_param_list(strategy, 'strategy', len(self.model_name_lst))
        max_depth_list = self._normalize_param_list(max_depth, 'max_depth', len(self.model_name_lst))
        
        file_path = None
        for model_name, epoch_val, strategy_val, max_depth_val in zip(
            self.model_name_lst, epoch_list, strategy_list, max_depth_list
        ):
            stacking_ex_lst = np.setdiff1d(self.df.columns.tolist(), self.keep_lst)
            res_opt, ress = run(
                objective_fix_sample, self.df, stacking_ex_lst, strategy_val, 
                file_path, epoch_val, model_name, max_depth=max_depth_val, 
                if_return_df=True
            )
            ress = ress.reset_index(drop=True)
            th = 0.05
            th1 = 0.1
            if 'oot_ks' in ress.columns:
                res = ress[ress['valid_ks'] > th][['train_ks', 'valid_ks', 'oot_ks']]
                res = res[res['train_ks'] > th]
                res = res[res['oot_ks'] > th]
                res['delta'] = res['train_ks'] - res['oot_ks']
                res['delta1'] = res['valid_ks'] - res['oot_ks']
                res['delta2'] = res['valid_ks'] - res['train_ks']
                res = res[abs(res['delta']) < th1]
                res = res[abs(res['delta1']) < th1]
                res = res[abs(res['delta2']) < th1]
                res.sort_values(by='oot_ks', ascending=False, inplace=True)
            else:
                res = ress[ress['valid_ks'] > th][['train_ks', 'valid_ks']]
                res = res[res['train_ks'] > th]
                res['delta2'] = res['valid_ks'] - res['train_ks']
                res = res[abs(res['delta2']) < 0.03]
                res.sort_values(by='valid_ks', ascending=False, inplace=True)
            res = res.copy()
            best_params_str = ress.iloc[res.index[0]]['model_param'].replace('true', 'True')
            self.params_dict[model_name] = ast.literal_eval(best_params_str)
    
    def second_df_create(self):
        """
        创建第二层训练数据，使用第一层模型的预测概率作为特征
        """
        if not hasattr(self, 'fold_name_lst') or self.fold_name_lst is None:
            raise ValueError("请先调用 skf_create() 方法创建交叉验证折数")
        
        for model_name in self.model_name_lst:
            if model_name not in self.params_dict:
                raise ValueError(f"模型 {model_name} 的参数未找到，请先调用 first_train() 方法")
            
            # 为每个fold训练模型并生成预测
            for fold_name in self.fold_name_lst:
                params = self.params_dict[model_name]
                model = self.model_init(model_name, params)
                
                # 获取训练和测试数据
                train_mask = self.df[fold_name] == 'fold_train'
                test_mask = self.df[fold_name] == 'fold_test'
                
                # 训练模型
                model.fit(
                    self.df[train_mask][self.keep_lst], 
                    self.df[train_mask][self.dep], 
                    sample_weight=self.df[train_mask]['weight']
                )
                
                # 预测测试集
                fold_test_proba = model.predict_proba(self.df[test_mask][self.keep_lst])[:, 1]
                self.df.loc[test_mask, f'{model_name}_proba'] = fold_test_proba
                self.df.loc[test_mask, f'{model_name}_proba_{fold_name}'] = fold_test_proba
                
                # 预测oot和valid集
                oot_valid_mask = self.df[self.target_col].isin(['oot', 'valid'])
                if oot_valid_mask.any():
                    oot_valid_proba = model.predict_proba(
                        self.df[oot_valid_mask][self.keep_lst]
                    )[:, 1]
                    self.df.loc[oot_valid_mask, f'{model_name}_proba_{fold_name}'] = oot_valid_proba

            # 计算oot和valid集的平均概率
            oot_valid_mask = self.df[self.target_col].isin(['oot', 'valid'])
            if oot_valid_mask.any():
                proba_cols = [f'{model_name}_proba_{fold_name}' for fold_name in self.fold_name_lst]
                self.df.loc[oot_valid_mask, f'{model_name}_proba'] = self.df.loc[
                    oot_valid_mask, proba_cols
                ].mean(axis=1)
            
            # 使用全部训练数据训练模型并预测
            train_mask = self.df[self.target_col] == 'train'
            predict_mask = self.df[self.target_col].isin(['train', 'oot', 'valid'])
            
            final_model = self.model_init(model_name, self.params_dict[model_name])
            final_model.fit(
                self.df[train_mask][self.keep_lst], 
                self.df[train_mask][self.dep], 
                sample_weight=self.df[train_mask]['weight']
            )
            stacking_proba = final_model.predict_proba(
                self.df[predict_mask][self.keep_lst]
            )[:, 1]
            self.df.loc[predict_mask, f'{model_name}_proba_stacking'] = stacking_proba
        
        # 创建第二层训练数据
        self.proba_lst = [f'{model_name}_proba' for model_name in self.model_name_lst]
        self.second_df = self.df[self.proba_lst + self.ex_lst].copy()
        
        proba_stacking_lst = [f'{model_name}_proba_stacking' for model_name in self.model_name_lst]
        self.second_df_stacking = self.df[proba_stacking_lst + self.ex_lst].copy()
        self.second_df_stacking.columns = [
            col.replace('_stacking', '') for col in self.second_df_stacking.columns
        ]

    def second_train(self, epoch_second=100, strategy_second=1348, max_depth_second=4,file_path = None):
        """
        第二层模型训练和超参数优化
        
        Parameters:
        -----------
        epoch_second : int, default=100
            训练轮数
        strategy_second : int, default=1348
            策略参数
        max_depth_second : int, default=4
            最大深度
        """
        self.second_df_create()
        
        if file_path is None:
            if_return_df = True
        else:
            if_return_df = False
        res_opt, ress = run(
            objective_fix_sample, self.second_df, self.ex_lst, strategy_second, 
            file_path, epoch_second, self.second_model, max_depth=max_depth_second, 
            if_return_df=if_return_df
        )
        ress = ress.reset_index(drop=True)
        th = 0.05
        th1 = 0.1
        if 'oot_ks' in ress.columns:
            res = ress[ress['valid_ks'] > th][['train_ks', 'valid_ks', 'oot_ks']]
            res = res[res['train_ks'] > th]
            res = res[res['oot_ks'] > th]
            res['delta'] = res['train_ks'] - res['oot_ks']
            res['delta1'] = res['valid_ks'] - res['oot_ks']
            res['delta2'] = res['valid_ks'] - res['train_ks']
            res = res[abs(res['delta']) < th1]
            res = res[abs(res['delta1']) < th1]
            res = res[abs(res['delta2']) < th1]
            res.sort_values(by='oot_ks', ascending=False, inplace=True)
        else:
            res = ress[ress['valid_ks'] > th][['train_ks', 'valid_ks']]
            res = res[res['train_ks'] > th]
            res['delta2'] = res['valid_ks'] - res['train_ks']
            res = res[abs(res['delta2']) < 0.03]
            res.sort_values(by='valid_ks', ascending=False, inplace=True)
        res = res.copy()
        best_params_str = ress.iloc[res.index[0]]['model_param'].replace('true', 'True')
        self.second_params = ast.literal_eval(best_params_str)

    def stacking_init(self, **kwargs):
        """
        初始化Stacking分类器
        
        Parameters:
        -----------
        **kwargs : dict
            传递给StackingClassifier的其他参数
            
        Returns:
        --------
        stacking_clf : StackingClassifier
            初始化的Stacking分类器
        """
        if self.skf is None:
            raise ValueError("请先调用 skf_create() 方法创建交叉验证对象")
        if not self.params_dict:
            raise ValueError("请先调用 first_train() 方法训练第一层模型")
        if self.second_params is None:
            raise ValueError("请先调用 second_train() 方法训练第二层模型")
        
        base_learners = [] 
        for model_name in self.model_name_lst:
            model = self.model_init(model_name, self.params_dict[model_name])
            # 使用临时变量避免修改原始model_name
            stack_model_name = 'lgbm' if model_name == 'lgb' else model_name
            base_learners.append((stack_model_name, model))
        
        second_model = self.model_init(self.second_model, self.second_params)
        stacking_clf = StackingClassifier(
            estimators=base_learners,
            final_estimator=second_model,
            cv=self.skf,
            **kwargs
        )
        return stacking_clf

if __name__ == '__main__':
    ex_lst = ['mobile','label','backDateTime','weight','target','month_time','day_time','week_time']
    data = pd.read_pickle('data/data_end_before_modeling.pkl')
    keep_lst = [i for i in data.columns if i not in ex_lst]
    
    # init
    tz_stacking = TzStacking(data, ex_lst, model_name_lst=['xgb','lgb'],second_model='lr',target_col='target')
    
    # 交叉验证生成
    tz_stacking.skf_create(n_splits=5, random_state=2025)
    
    # 第一层参数训练
    tz_stacking.first_train(epoch=[10], strategy=[1348], max_depth=[4])
    
    # 第二层参数训练
    tz_stacking.second_train(epoch_second=10, strategy_second=1348, max_depth_second=4)

    # stacking
    stacking_clf = tz_stacking.stacking_init()

    stacking_clf.fit(data[data['target'] == 'train'][keep_lst], 
        data[data['target'] == 'train']['label'], 
        sample_weight=data[data['target'] == 'train']['weight'])
    
    from ScoreCard.evaluation import ScorecardEvaluation
    se = ScorecardEvaluation('信贷')
    res_data = se.eval_dataset(data, stacking_clf, keep_lst, col_name='target')