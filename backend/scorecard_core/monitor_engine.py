# -*- coding: utf-8 -*-
"""scorecard_core - 监控与模拟引擎"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from scorecard_core.strategy_engine import run_policy_flow, enrich_df_with_model_scores

logger = logging.getLogger(__name__)

from scorecard_core.scoring import proba2score

class ScoreCalculator:
    """自动评分寻参器 (移植自 auto_score.py)"""
    def __init__(self, initial_p=600, initial_q=50, initial_pdo=20, min_score=300, max_score=900):
        self.p = initial_p
        self.q = initial_q
        self.pdo = initial_pdo
        self.min_score = min_score
        self.max_score = max_score
        # 目标跨度：允许的最大跨度和最小跨度 (基于总区间)
        self.range_score_max = (max_score - min_score)
        self.range_score_min = (max_score - min_score) * 0.6

    def card_score(self, pred):
        # pred 是坏人概率
        prob = np.clip(pred, 1e-6, 1 - 1e-6)
        b = -self.pdo / np.log(2)
        a = self.p + b * np.log(self.q)
        # odds = (1-p)/p
        odds = (1 - prob) / prob
        score = a - b * np.log(odds)
        return score

    def automatic_scoring(self, prba_arr):
        scores = self.card_score(prba_arr)
        scor_min = np.min(scores)
        scor_max = np.max(scores)

        max_iterations = 300
        adj_pdo = 5
        adj_p = 50
        
        iterations = 0
        while ( (scor_max - scor_min) >= self.range_score_max or (scor_max - scor_min) <= self.range_score_min ) and iterations < max_iterations:
            if (scor_max - scor_min) >= self.range_score_max:
                self.pdo = max(1, self.pdo - adj_pdo)
            elif (scor_max - scor_min) <= self.range_score_min:
                self.pdo += adj_pdo
            
            scores = self.card_score(prba_arr)
            scor_min, scor_max = np.min(scores), np.max(scores)
            iterations += 1
            if iterations % 50 == 0: adj_pdo = max(1, adj_pdo // 2)

        iterations = 0
        while (scor_min < self.min_score or scor_max > self.max_score) and iterations < max_iterations:
            if scor_min < self.min_score:
                self.p += adj_p
            if scor_max > self.max_score:
                self.p = max(0, self.p - adj_p)

            scores = self.card_score(prba_arr)
            scor_min, scor_max = np.min(scores), np.max(scores)
            iterations += 1
            if iterations % 50 == 0: adj_p = max(1, adj_p // 2)

        return scores, float(self.p), float(self.q), float(self.pdo)

def auto_adjust_score(probs, min_score=300, max_score=900):
    """便捷入口：从概率自动推导最优评分参数"""
    calc = ScoreCalculator(min_score=min_score, max_score=max_score)
    scores, p, q, pdo = calc.automatic_scoring(probs)
    return {
        'scores': scores,
        'config': {'base_score': p, 'base_odds': q, 'pdo': pdo}
    }

def simulate_business_intake(df, n_samples=5000, drift_scale=0.03, time_col=None):
    """
    业务进件模拟器：基于历史数据生成具有统计偏移的未来数据。
    
    参数:
        df: 原始参考数据集 (通常是训练集或最新 OOT)
        n_samples: 模拟生成的样本量
        drift_scale: 特征偏移比例 (0.01~0.1)
        time_col: 时间列名，用于推演未来日期
    """
    # 1. 使用 Bootstrap 采样基础数据
    sim_df = df.sample(n=n_samples, replace=True).reset_index(drop=True)
    
    # 2. 对数值型特征增加随机漂移 (Drift)
    numeric_cols = sim_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 跳过标签和权重列
        if col in ['label', 'target', 'weight']:
            continue
            
        std_val = sim_df[col].std()
        if pd.isna(std_val) or std_val == 0:
            continue
            
        # 模拟均值偏移 (Mean Shift)
        # 随机产生一个正负 drift_scale * std 的偏移量
        shift = np.random.uniform(-drift_scale, drift_scale) * std_val
        # 增加正态分布噪声
        noise = np.random.normal(0, 0.01 * std_val, size=n_samples)
        
        sim_df[col] = sim_df[col] + shift + noise
    
    # 3. 模拟时间推演 (Time Shift)
    if time_col and time_col in sim_df.columns:
        try:
            sim_df[time_col] = pd.to_datetime(sim_df[time_col])
            max_date = sim_df[time_col].max()
            # 模拟未来 1-3 个月的随机分布
            days_shift = np.random.randint(30, 90, size=n_samples)
            sim_df[time_col] = sim_df[time_col].apply(lambda x: max_date + timedelta(days=int(np.random.randint(1, 90))))
        except Exception as e:
            logger.warning(f"时间推演模拟失败: {e}")

    return sim_df

def calculate_psi_from_dist(actual_scores, expected_dist, num_bins=10):
    """
    基于预设分箱(训练集分布)计算 PSI。
    expected_dist 包含 'bins' 和 'counts_good' + 'counts_bad'
    """
    if not expected_dist or 'bins' not in expected_dist:
        return 0.0
    
    # 1. 提取基准分箱和占比
    bins = list(expected_dist['bins'])
    # 扩展边界以覆盖所有模拟分数 (避免漏计导致的 PSI 偏低)
    if len(bins) > 1:
        bins[0] = -np.inf
        bins[-1] = np.inf
    
    # 整合好坏人为总基准计数
    expected_counts = np.array(expected_dist.get('counts_good', [])) + np.array(expected_dist.get('counts_bad', []))
    if len(expected_counts) == 0:
        return 0.0
    
    expected_percents = expected_counts / expected_counts.sum()
    
    # 2. 统计当前分数的分布 (使用相同的 bins)
    actual_counts, _ = np.histogram(actual_scores, bins=bins)
    actual_percents = actual_counts / actual_counts.sum()
    
    # 3. 避免除以零
    actual_percents[actual_percents == 0] = 0.000001
    expected_percents[expected_percents == 0] = 0.000001
    
    # 4. 计算 PSI
    psi_val = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return float(psi_val)

def run_monitoring_task(model_path, current_df, feature_list, score_config=None, score_distribution=None, batch_name="模拟监控批次"):
    """
    执行一次完整的监控流程：模拟 -> 打分 -> 计算指标
    """
    # 1. 加载模型
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        logger.error(f"加载监控模型失败: {e}")
        return None

    # 2. 预处理 (与训练保持一致)
    df_copy = current_df.copy()
    for col in feature_list:
        if col in df_copy.columns and not pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = pd.factorize(df_copy[col])[0].astype(float)
            df_copy[col] = df_copy[col].replace(-1, np.nan)

    # 3. 预测概率并转换为分数
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(df_copy[feature_list])[:, 1]
    else:
        probs = model.predict(df_copy[feature_list])
    
    sc = score_config or {}
    scores = proba2score(
        probs,
        pdo=sc.get('pdo', 20),
        base_score=sc.get('base_score', 600),
        base_odds=sc.get('base_odds', 50)
    )
    
    # 4. 基于基准分布计算 PSI
    psi_val = calculate_psi_from_dist(scores, score_distribution)
    
    # 5. 生成当前批次的分布数据 (用于 ECharts 展示)
    hist, bin_edges = np.histogram(scores, bins=10)
    score_dist = {
        'bins': bin_edges.astype(float).tolist(),
        'counts': hist.astype(int).tolist()
    }

    result = {
        'batch_name': batch_name,
        'psi': psi_val,
        'avg_score': float(np.mean(scores)),
        'score_dist': score_dist,
        'sample_size': len(scores),
        'metrics': {
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'std': float(np.std(scores))
        }
    }
    
    return result

def run_strategy_monitoring(df, active_strategies, db_session, model_result_model, batch_name="策略监控批次"):
    """
    针对策略流的监控：多策略顺序执行模拟
    """
    if df.empty or not active_strategies:
        return None

    # 1. 数据预处理 (计算模型规则所需的分数)
    # 提取所有规则中涉及的模型 ID
    all_rules = []
    for s in active_strategies:
        all_rules.extend(s.rules)
    
    # 模拟数据丰富化 (补充模型分)
    df = enrich_df_with_model_scores(df, all_rules, db_session, model_result_model)

    # 2. 运行策略流
    # 将 ORM 对象转换为 dict 以适配 engine
    strat_dicts = []
    for s in active_strategies:
        strat_dicts.append({
            "id": s.id,
            "name": s.name,
            "rules": s.rules,
            "combine_logic": s.combine_logic,
            "rule_type": s.rule_type
        })
    
    flow_res = run_policy_flow(df, strat_dicts)

    return {
        "batch_name": batch_name,
        "total_count": flow_res["total_count"],
        "pass_count": flow_res["pass_count"],
        "hit_count": flow_res["hit_count"],
        "approval_rate": flow_res["approval_rate"],
        "rule_stats": flow_res["rule_stats"],
        "created_at": datetime.now().isoformat()
    }
