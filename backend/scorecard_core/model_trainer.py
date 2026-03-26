# -*- coding: utf-8 -*-
"""scorecard_core - 模型训练模块（XGB + Optuna）"""

import sys
import os
import json
import logging
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

logger = logging.getLogger(__name__)

# 将 scorecard 包加入 path
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_scorecard_path = os.path.join(_base_dir, 'scorecard')
if _scorecard_path not in sys.path:
    sys.path.insert(0, _scorecard_path)


# ===================== 模型参数模板 =====================

def param_xgb(trial, max_depth=6):
    import xgboost as xgb
    return {
        "verbosity": 0,
        "booster": "gbtree",
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 500.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 30, 100, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, max_depth),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 50, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, log=True),
    }


def param_lgb(trial, max_depth=6):
    return {
        "boosting_type": "gbdt",
        "objective": "binary",
        "random_state": 2022,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 10, 500, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, max_depth),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 500.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 500.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.3, 1, log=True),
    }


def param_lr(trial):
    return {
        "penalty": 'l2',
        "C": trial.suggest_float("C", 0.01, 1, log=True),
        "fit_intercept": True,
        "class_weight": "balanced",
        "max_iter": trial.suggest_int("max_iter", 100, 5000, log=True),
        "solver": 'liblinear',
    }


# 参数模板注册表
PARAM_BUILDERS = {
    'xgb': param_xgb,
    'lgb': param_lgb,
    'lr': param_lr,
}


# ===================== 策略函数 =====================

def strategy(res, strategy_type, threshold=0.03):
    """
    Optuna 统计优化策略。
    res: DataFrame 行, 应当包含各数据集统计指标。
    strategy_type: 策略编号。
    threshold: 对齐差值阈值 (0.01 - 0.1)。
    """
    return _expanded_strategy(res, strategy_type, threshold)


def _expanded_strategy(res, strategy_type, threshold=0.03):
    """
    整合后的扩展策略库。其中大部分逻辑参考自 Optuna_strategy.py
    """
    s_type = int(strategy_type)
    
    # 辅助变量，防止 KeyError
    t_ks = res['train_ks'].mean() if 'train_ks' in res.columns else 0
    v_ks = res['valid_ks'].mean() if 'valid_ks' in res.columns else 0
    o_ks = res['oot_ks'].mean() if 'oot_ks' in res.columns else v_ks
    
    t_auc = res['train_auc'].mean() if 'train_auc' in res.columns else 0
    v_auc = res['valid_auc'].mean() if 'valid_auc' in res.columns else 0
    o_auc = res['oot_auc'].mean() if 'oot_auc' in res.columns else v_auc

    # --- KS 系列 (1-19, 1348-1350) ---
    if s_type in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1348, 1349, 1350]:
        # 通用对齐判定 (Train vs Valid vs OOT)
        if s_type == 1:
            return o_ks - 0.2 * abs(t_ks - o_ks) - 0.2 * abs(v_ks - o_ks)
        elif s_type == 2:
            return t_ks - 0.2 * abs(t_ks - o_ks)
        
        # 使用自定义 threshold 的对齐策略
        curr_thresh = threshold
        if s_type == 4: curr_thresh = 0.02
        if s_type == 5: curr_thresh = 0.04
        if s_type == 1348: curr_thresh = 0.05
        if s_type == 1349: curr_thresh = 0.03
        if s_type == 1350: curr_thresh = 0.02

        if s_type in [3, 4, 5, 1348, 1349, 1350]:
            if abs(t_ks - o_ks) < curr_thresh and abs(t_ks - v_ks) < curr_thresh and abs(o_ks - v_ks) < curr_thresh:
                return o_ks
            else:
                return -(abs(t_ks - o_ks) + abs(t_ks - v_ks))
        
        elif s_type == 6:
            return o_ks if (abs(t_ks - o_ks) < 0.3 and abs(t_ks - v_ks) < 0.1) else -(abs(t_ks - o_ks) + abs(t_ks - v_ks))
        
        elif s_type == 10:
            return (v_ks + o_ks) / 2 if (abs(t_ks - v_ks) < 0.1 and abs(t_ks - o_ks) < 0.1) else -(abs(t_ks - o_ks) + abs(t_ks - v_ks))

        elif s_type in [11, 12]: # 无 OOT
            if abs(t_ks - v_ks) < curr_thresh: return v_ks
            return -abs(t_ks - v_ks)
            
        return v_ks # 兜底

    # --- Lift / Top N 系列 (201) ---
    elif s_type == 201:
        top_lift = res['oot_top_5_lift'].mean() if 'oot_top_5_lift' in res.columns else 0
        return (v_ks + o_ks) * 0.9 + top_lift * 0.1 if (abs(t_ks - v_ks) < 0.1 and abs(t_ks - o_ks) < 0.1) else -(abs(t_ks - o_ks) + abs(t_ks - v_ks))

    # --- AUC 系列 (20-29) ---
    elif s_type in [20, 21, 22, 23, 24, 25, 26, 27]:
        curr_thresh = threshold
        if s_type == 20:
            if abs(t_auc - v_auc) < 0.02: return v_auc
            return -abs(t_auc - v_auc)
        elif s_type == 26:
            if abs(t_auc - o_auc) < curr_thresh and abs(t_auc - v_auc) < curr_thresh and abs(o_auc - v_auc) < curr_thresh:
                return o_auc
            return -(abs(t_auc - o_auc) + abs(t_auc - v_auc))
        return v_auc

    # 兜底：直接取验证集 KS
    return v_ks


# ===================== 模型评估 =====================

def evaluate_model(model, datasets, ft_lst, dep='label'):
    """评估模型在各数据集上的表现"""
    metrics = {}
    for name, df in datasets.items():
        try:
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(df[ft_lst])[:, 1]
            else:
                y_pred = model.predict(df[ft_lst])

            y_true = df[dep].values
            weight = df['weight'].values if 'weight' in df.columns else None

            fpr, tpr, _ = roc_curve(y_true, y_pred, sample_weight=weight)
            ks = float(abs(fpr - tpr).max())
            auc_val = float(roc_auc_score(y_true, y_pred, sample_weight=weight))

            metrics[name + '_ks'] = ks
            metrics[name + '_auc'] = auc_val
        except Exception as e:
            logger.warning(f"评估 {name} 数据集失败: {e}")

    return metrics


def get_feature_importance(model, ft_lst):
    """获取特征重要性"""
    importance = {}
    if hasattr(model, 'feature_importances_'):
        for feat, imp in zip(ft_lst, model.feature_importances_):
            importance[feat] = float(imp)
    elif hasattr(model, 'coef_'):
        for feat, coef in zip(ft_lst, model.coef_[0]):
            importance[feat] = float(abs(coef))
    # 按重要性排序
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return importance


# ===================== 建模主流程 =====================

def create_model(model_type, params):
    """根据类型和参数创建模型实例"""
    if model_type == 'xgb':
        import xgboost as xgb
        return xgb.XGBClassifier(**params)
    elif model_type == 'lgb':
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params)
    elif model_type == 'lr':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    elif model_type == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    elif model_type == 'dt':
        from sklearn import tree
        return tree.DecisionTreeClassifier(**params)
    else:
        import xgboost as xgb
        return xgb.XGBClassifier(**params)


def run_optuna_training(db, task_id, progress_callback,
                        data_path, exclude_cols, feature_list,
                        dep, model_type, strategy_type, strategy_threshold,
                        n_trials, max_depth, project_id, model_save_dir,
                        split_ratios=None, oot_col=None, oot_start_time=None,
                        score_config=None):
    """
    Optuna 调参训练主函数（在线程中执行）

    参数:
        db: 数据库会话
        task_id: 任务 ID
        progress_callback: 进度回调函数
        data_path: 数据文件路径
        exclude_cols: 排除列
        feature_list: 特征列（可选）
        dep: 目标变量
        model_type: 模型类型
        strategy_type: Optuna 策略编号
        n_trials: 试验次数
        max_depth: 最大树深度
        project_id: 项目 ID
        model_save_dir: 模型保存目录
        split_ratios: 划分比例
        oot_col: OOT 时间列
        oot_start_time: OOT 起始时间
    """
    import optuna
    from optuna.samplers import TPESampler

    from scorecard_core.data_processor import load_data, split_dataset, add_weight_column

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 加载数据
    progress_callback(5, {'stage': '加载数据'})
    df = load_data(data_path)
    df = add_weight_column(df)

    # 强制类别变量数值化处理
    logger.info(f"转换前特征 Dtypes: \n{df.dtypes}")
    # 首先处理标签列
    if dep in df.columns and not pd.api.types.is_numeric_dtype(df[dep]):
        logger.info(f"强制转换目标列 (dep): {dep}")
        df[dep] = pd.factorize(df[dep])[0]
        
    # 处理所有非数值列（特征列）
    for col in df.columns:
        if col in ['target', 'weight'] or col == dep:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.info(f"强制转换特征列: {col}")
            # 使用 pd.factorize 转换为数值，保留 NaN (设置为 -1)
            df[col] = pd.factorize(df[col])[0].astype(float)
            # 将 -1 (pd.factorize 的 NaN 默认值) 还原为真正的 NaN，让 XGBoost 处理
            df[col] = df[col].replace(-1, np.nan)
            
    logger.info(f"转换后特征 Dtypes: \n{df.dtypes}")

    # 拆分数据集（使用增强后的分层拆分）
    datasets = split_dataset(df, dep=dep, 
                             ratios=split_ratios, 
                             oot_col=oot_col, 
                             oot_start_time=oot_start_time,
                             random_state=42)

    # 确定特征列
    if feature_list and len(feature_list) > 0:
        ft_lst = feature_list
    else:
        default_exclude = list(set(exclude_cols + [dep, 'target', 'weight']))
        ft_lst = [c for c in df.columns if c not in default_exclude]

    progress_callback(10, {'stage': '开始 Optuna 调参', 'n_features': len(ft_lst)})

    # 获取参数构建器
    param_builder = PARAM_BUILDERS.get(model_type, param_xgb)

    # 记录每次试验结果
    trial_results = []
    best_model = None
    best_metrics = None
    best_params = None
    best_value = float('-inf')

    def objective(trial):
        nonlocal best_model, best_metrics, best_params, best_value

        # 构建参数
        if model_type == 'lr':
            params = param_builder(trial)
        else:
            params = param_builder(trial, max_depth)

        # 创建并训练模型
        model = create_model(model_type, params)
        train_data = datasets.get('train', list(datasets.values())[0])

        if 'weight' in train_data.columns:
            model.fit(train_data[ft_lst], train_data[dep], sample_weight=train_data['weight'])
        else:
            model.fit(train_data[ft_lst], train_data[dep])

        # 评估
        metrics = evaluate_model(model, datasets, ft_lst, dep)
        metrics_df = pd.DataFrame(metrics, index=[0])

        # 计算策略值
        value = strategy(metrics_df, strategy_type, strategy_threshold)
        if hasattr(value, 'values'):
            value = float(value.values[0])
        else:
            value = float(value)

        trial_results.append({
            'trial': trial.number,
            'value': value,
            'params': params,
            'metrics': metrics,
        })

        # 更新最优 (始终选取 OOT KS 最高的版本作为最终输出)
        curr_oot_ks = metrics.get('oot_ks', 0)
        if curr_oot_ks > best_value:
            best_value = curr_oot_ks
            best_model = model
            best_metrics = metrics
            best_params = params

        # 报告进度
        pct = 10 + (trial.number + 1) / n_trials * 80
        progress_callback(pct, {
            'stage': f'Optuna 调参中 ({trial.number + 1}/{n_trials})',
            'current_trial': trial.number,
            'current_value': value,
            'best_value': best_value,
            'current_metrics': metrics,
        })

        return value

    # 执行 Optuna
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=n_trials)

    progress_callback(90, {'stage': '保存模型'})

    # 保存模型
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f'model_{task_id}.pkl')
    if best_model is not None:
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

    importance = get_feature_importance(best_model, ft_lst) if best_model else {}

    # --- 计算分数分布 ---
    score_dist = {}
    if best_model is not None:
        try:
            # 使用全量数据进行分数分布测算
            if hasattr(best_model, 'predict_proba'):
                probs = best_model.predict_proba(df[ft_lst])[:, 1]
            else:
                probs = best_model.predict(df[ft_lst])
            
            sc = score_config or {'mode': 'manual', 'base_score': 600, 'pdo': 20, 'base_odds': 50}
            
            if sc.get('mode') == 'auto':
                from scorecard_core.monitor_engine import auto_adjust_score
                auto_res = auto_adjust_score(probs, 
                                             min_score=sc.get('min_score', 300), 
                                             max_score=sc.get('max_score', 900))
                scores = auto_res['scores']
                # 更新 sc 供后续保存
                sc.update(auto_res['config'])
            else:
                from scorecard_core.scoring import proba2score
                scores = proba2score(probs, 
                                     pdo=sc.get('pdo', 20), 
                                     base_score=sc.get('base_score', 600), 
                                     base_odds=sc.get('base_odds', 50))
            
            # 分离好坏人分数 (0=好, 1=坏)
            good_scores = scores[df[dep] == 0]
            bad_scores = scores[df[dep] == 1]
            
            # 统一分箱
            bins = np.linspace(min(scores), max(scores), 21) # 20个区间
            hist_good, _ = np.histogram(good_scores, bins=bins)
            hist_bad, _ = np.histogram(bad_scores, bins=bins)
            
            score_dist = {
                'bins': bins.tolist(),
                'counts_good': hist_good.tolist(),
                'counts_bad': hist_bad.tolist(),
                'avg_good': float(np.mean(good_scores)) if len(good_scores) > 0 else 0,
                'avg_bad': float(np.mean(bad_scores)) if len(bad_scores) > 0 else 0,
            }
        except Exception as e:
            logger.error(f"计算分数分布失败: {e}")

    # 存储 ModelResult 到数据库
    from app.models import ModelResult
    model_result = ModelResult(
        project_id=project_id,
        task_id=task_id,
        model_type=model_type,
        model_path=model_path,
        params=best_params or {},
        metrics=best_metrics or {},
        feature_importance=importance,
        feature_list=ft_lst,
        optuna_strategy=strategy_type,
        n_trials=n_trials,
        score_config=sc,  # 保存实际使用的（可能是 auto 计算出的）配置
        score_distribution=score_dist
    )
    db.add(model_result)
    db.commit()

    progress_callback(95, {'stage': '完成'})

    return {
        'model_result_id': model_result.id,
        'best_params': best_params,
        'best_metrics': best_metrics,
        'best_value': best_value,
        'feature_importance': importance,
        'n_features': len(ft_lst),
        'total_trials': len(trial_results),
    }
