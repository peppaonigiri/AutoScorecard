from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
from tqdm import tqdm

def n_fold_cross_importance(data, n_splits=5, keep_list=None, cut_value=0, dep='label', model_type='xgb',params=None):
    """
    交叉验证特征重要性
    :param data: pandas DataFrame
    :param n_splits: 交叉验证折数
    :param keep_list: 保留特征列表
    :param cut_value: 重要性阈值
    :param dep: Y标签列名
    :param model_type: 模型类型, 'xgb' or 'lgb'
    :return: keep_list, ipt_df_mean, 返回的ipt_df_mean是各个fold的平均重要性
    """
    if keep_list is None:
        raise ValueError('keep_list is None')
    
    if model_type == 'xgb':
        model_params = {
            'learning_rate': 0.1,
            'n_estimators': 200,
            'max_depth': 5,
            'min_child_weight': 1,
            'colsample_bytree': 0.7,
            'subsample': 0.7,
            'scale_pos_weight': 1,
            'n_jobs': -1,
            'reg_lambda': 300,
            'random_state': 2024
        }
    elif model_type == 'lgb':
        model_params = {
            'learning_rate': 0.1,
            'n_estimators': 200,
            'max_depth': 5,
            'min_child_weight': 1,
            'min_child_samples': 100,
            'num_leaves': 30,
            'colsample_bytree': 0.7,
            'subsample': 0.7,
            'scale_pos_weight': 1,
            'n_jobs': -1,
            'reg_lambda': 300,
            'verbosity': -1,
            'random_state': 2024
        }
    else:
        raise ValueError('model_type 必须为 "xgb" 或 "lgb"')
    
    if isinstance(params, dict):
        model_params.update(params)
    
    print('当前模型参数: ', model_params)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    ipt_list = []
    ipt_df_list = []

    if model_type == 'xgb':
        model = xgb.XGBClassifier(**model_params)
    elif model_type == 'lgb':
        model = lgb.LGBMClassifier(**model_params)

    for train_index, valid_index in tqdm(kf.split(data), total=n_splits, desc='交叉验证特征重要性'):
        
        model.fit(data.loc[train_index, keep_list], data.loc[train_index, dep])

        if model_type == 'xgb':
            importance = model.feature_importances_
        elif model_type == 'lgb':
            model.importance_type = 'gain'
            importance = model.feature_importances_

        temp_ipt_df = pd.DataFrame({
            'var_names': keep_list,
            'model_gain': importance
        })
        
        temp_ipt_df.sort_values(by='model_gain', ascending=False, inplace=True)

        ipt_keep = temp_ipt_df[temp_ipt_df['model_gain'] > cut_value]['var_names'].tolist()
        ipt_list.append(ipt_keep)
        temp_ipt_df.set_index('var_names', inplace=True)
        ipt_df_list.append(temp_ipt_df)


    ipt_df_mean = pd.concat(ipt_df_list,axis=1)
    ipt_df_mean.columns = [f'fold_{i}' for i in range(len(ipt_df_list))]
    ipt_df_mean['model_gain'] = ipt_df_mean.mean(axis=1)
    ipt_df_mean.reset_index(inplace=True)
    ipt_df_mean.sort_values(by='model_gain', ascending=False, inplace=True)
    ipt_df_mean = ipt_df_mean[['var_names','model_gain']]
    
    final_ipt_keep = set(ipt_list[0])
    for signal_list in ipt_list:
        # 列表求交集
        final_ipt_keep = final_ipt_keep.intersection(set(signal_list))
    final_ipt_keep = list(final_ipt_keep)

    return final_ipt_keep, ipt_df_mean