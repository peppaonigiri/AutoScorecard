# -*- coding: utf-8 -*-
"""风控策略分析引擎"""

import pandas as pd
import numpy as np
import os
import pickle
from typing import List, Dict, Any

from scorecard_core.scoring import proba2score

def enrich_df_with_model_scores(df: pd.DataFrame, rules: List[Dict], db_session, model_result_model) -> pd.DataFrame:
    """
    遍历规则，识别需要模型打分的字段并实时计算填入 df。
    :param model_result_model: ModelResult ORM class (避免循环引用)
    """
    model_ids = set()
    for r in rules:
        if r.get('field', '').startswith('_model_result_'):
            try:
                mid = int(r['field'].replace('_model_result_', ''))
                model_ids.add(mid)
            except:
                pass

    if not model_ids:
        return df

    for mid in model_ids:
        if f'_model_result_{mid}' in df.columns:
            continue
            
        model_res = db_session.query(model_result_model).filter(model_result_model.id == mid).first()
        if not model_res or not model_res.model_path or not os.path.exists(model_res.model_path):
            continue
            
        try:
            with open(model_res.model_path, 'rb') as f:
                model = pickle.load(f)
            
            ft_lst = model_res.feature_list
            # 确保 df 中有这些特征
            missing = [c for c in ft_lst if c not in df.columns]
            if missing:
                continue
                
            # 预处理：与监控/训练保持一致 (处理非数值)
            sub_df = df[ft_lst].copy()
            for col in ft_lst:
                if not pd.api.types.is_numeric_dtype(sub_df[col]):
                    sub_df[col] = pd.factorize(sub_df[col])[0].astype(float)
                    sub_df[col] = sub_df[col].replace(-1, np.nan)

            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(sub_df)[:, 1]
            else:
                probs = model.predict(sub_df)

            sc = model_res.score_config or {}
            scores = proba2score(
                probs,
                pdo=sc.get('pdo', 20),
                base_score=sc.get('base_score', 600),
                base_odds=sc.get('base_odds', 50)
            )
            df[f'_model_result_{mid}'] = scores
        except Exception as e:
            print(f" enrichment 失败: {e}")
            
    return df

def run_strategy_analysis(df: pd.DataFrame, rules: List[Dict], combine_logic: str = 'and', rule_type: str = 'reject') -> Dict[str, Any]:
    """
    运行策略分析，返回各项风控指标
    :param df: 原始数据集 (需包含标签列)
    :param rules: 规则列表
    :param combine_logic: 'and', 'or'
    :param rule_type: 'reject' (拦截模式), 'approve' (通过模式)
    """
    if df.empty or not rules:
        return {}

    # 1. 构建布尔掩码
    masks = []
    for rule in rules:
        field = rule['field']
        op = rule['op']
        val = rule['val']
        
        if field not in df.columns:
            continue
            
        # 转换为数值类型进行比较（增强鲁棒性）
        try:
            col_data = pd.to_numeric(df[field], errors='coerce')
            val_num = float(val)
        except:
            col_data = df[field]
            val_num = val

        if op == '>':
            masks.append(col_data > val_num)
        elif op == '<':
            masks.append(col_data < val_num)
        elif op == '>=':
            masks.append(col_data >= val_num)
        elif op == '<=':
            masks.append(col_data <= val_num)
        elif op == '==':
            masks.append(col_data == val_num)
        elif op == '!=':
            masks.append(col_data != val_num)

    if not masks:
        return {"error": "NO_VALID_RULES"}

    # 2. 逻辑组合 (命中规则的掩码)
    if combine_logic == 'and':
        hit_mask = np.logical_and.reduce(masks)
    else:
        hit_mask = np.logical_or.reduce(masks)

    # 3. 根据规则类型确定通过与拦截集合
    if rule_type == 'approve':
        final_pass_mask = hit_mask
        final_rej_mask = ~hit_mask
    else: # reject
        final_pass_mask = ~hit_mask
        final_rej_mask = hit_mask

    # 4. 计算指标
    # 寻找标签列 (优先 label, 且必须是数值型)
    target_col = None
    candidate_cols = ['label', 'target', 'bad_ind', 'is_bad']
    for col in candidate_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            target_col = col
            break
            
    if not target_col:
        # 如果没找到，尝试找最后一个带 bad 或 label 字样的数值列
        for col in reversed(df.columns):
            if ('bad' in col.lower() or 'label' in col.lower()) and pd.api.types.is_numeric_dtype(df[col]):
                target_col = col
                break
    
    total_count = len(df)
    hit_count = int(final_rej_mask.sum())
    pass_count = int(final_pass_mask.sum())
    
    approval_rate = pass_count / total_count if total_count > 0 else 0
    rejection_rate = hit_count / total_count if total_count > 0 else 0

    results = {
        "total_count": total_count,
        "hit_count": hit_count,
        "pass_count": pass_count,
        "approval_rate": round(approval_rate, 4),
        "rejection_rate": round(rejection_rate, 4),
    }

    if target_col:
        total_bad = int(df[target_col].sum())
        hit_bad = int(df.loc[final_rej_mask, target_col].sum())
        pass_bad = int(df.loc[final_pass_mask, target_col].sum())
        
        # 坏样本捕获率 (指被拦截掉的坏比例)
        bad_capture_rate = hit_bad / total_bad if total_bad > 0 else 0
        # 通过件坏账率
        pass_bad_rate = pass_bad / pass_count if pass_count > 0 else 0
        # 总体坏账率
        base_bad_rate = total_bad / total_count if total_count > 0 else 0
        
        # Lift 计算: (拦截组的坏人浓度 / 总体坏人浓度)
        hit_bad_rate = hit_bad / hit_count if hit_count > 0 else 0
        lift = hit_bad_rate / base_bad_rate if base_bad_rate > 0 else 0

        results.update({
            "total_bad": total_bad,
            "hit_bad": hit_bad,
            "bad_capture_rate": round(bad_capture_rate, 4),
            "pass_bad_rate": round(pass_bad_rate, 4),
            "base_bad_rate": round(base_bad_rate, 4),
            "lift": round(lift, 2)
        })

    return results
def run_policy_flow(df: pd.DataFrame, strategies: List[Dict]) -> Dict[str, Any]:
    """
    运行策略流：按顺序执行多条策略，执行短路逻辑。
    :param df: 数据集
    :param strategies: 策略对象列表，已按 priority 排序
    """
    if df.empty or not strategies:
        return {"total_count": len(df), "pass_count": len(df), "hit_count": 0, "rule_stats": []}

    total_count = len(df)
    # 当前剩余待决策的样本索引 (Boolean mask)
    remaining_mask = np.ones(total_count, dtype=bool)
    
    # 决策结果：0-待定, 1-通过, -1-拒绝
    decisions = np.zeros(total_count, dtype=int)
    # 记录命中情况
    hit_by_strategy = np.zeros(total_count, dtype=int) # 存储命中的策略ID

    rule_stats = []

    for strat in strategies:
        s_id = strat.get('id', 0)
        s_name = strat.get('name', 'Unknown')
        rules = strat.get('rules', [])
        logic = strat.get('combine_logic', 'and')
        s_type = strat.get('rule_type', 'reject')

        # 0. 获取当前节点的输入样本量 (即分母：到达本节点的样本)
        node_input_indices = remaining_mask.copy()
        node_input_count = int(node_input_indices.sum())
        
        # [内部辅助逻辑：获取当前策略规则的命中掩码 (针对所有数据)]
        masks = []
        for r in rules:
            f, op, v = r['field'], r['op'], r['val']
            if f not in df.columns: continue
            try:
                col_data = pd.to_numeric(df[f], errors='coerce')
                v_num = float(v)
            except:
                col_data, v_num = df[f], v
            if op == '>': masks.append(col_data > v_num)
            elif op == '<': masks.append(col_data < v_num)
            elif op == '>=': masks.append(col_data >= v_num)
            elif op == '<=': masks.append(col_data <= v_num)
            elif op == '==': masks.append(col_data == v_num)
            elif op == '!=': masks.append(col_data != v_num)
        
        if not masks:
            # 规则无效，全部通过当前层
            rule_stats.append({
                "id": s_id, "name": s_name, "rule_type": s_type,
                "node_input_count": node_input_count,
                "intercept_count": 0, "pass_count": node_input_count,
                "node_intercept_rate": 0.0, "global_intercept_rate": 0.0,
                "node_pass_rate": 1.0, "global_pass_rate": round(node_input_count / total_count, 4)
            })
            continue
            
        # 命中规则条件的总掩码 (满足 IF 条件)
        strat_cond_mask = np.logical_and.reduce(masks) if logic == 'and' else np.logical_or.reduce(masks)
        
        # 1. 核心决策逻辑
        if s_type == 'reject':
            # 命中条件则拦截
            intercept_indices = node_input_indices & strat_cond_mask
            pass_indices = node_input_indices & (~strat_cond_mask)
            decisions[intercept_indices] = -1
        else: # approve
            # 命中条件则通过，没命中条件则拦截
            pass_indices = node_input_indices & strat_cond_mask
            intercept_indices = node_input_indices & (~strat_cond_mask)
            decisions[pass_indices] = 1

        # 更新剩余存活样本集，供后续规则执行
        remaining_mask[intercept_indices] = False
        
        # 2. 统计指标计算
        intercept_count = int(intercept_indices.sum())
        pass_count = int(pass_indices.sum())
        
        rule_stats.append({
            "id": s_id,
            "name": s_name,
            "rule_type": s_type,
            "node_input_count": node_input_count,
            "intercept_count": intercept_count,
            "pass_count": pass_count,
            # 独立比率 (分母是到达该节点的数量)
            "node_intercept_rate": round(intercept_count / node_input_count, 4) if node_input_count > 0 else 0,
            "node_pass_rate": round(pass_count / node_input_count, 4) if node_input_count > 0 else 0,
            # 全局比率 (分母是初始总样本)
            "global_intercept_rate": round(intercept_count / total_count, 4) if total_count > 0 else 0,
            "global_pass_rate": round(pass_count / total_count, 4) if total_count > 0 else 0
        })

    # 4. 汇总决策结果
    has_approve_rules = any(s.get('rule_type') == 'approve' for s in strategies)
    
    # 最终结果：
    # 如果策略流中有“通过型”规则，那么没有命中任何规则的人（0）默认应该被拒绝（-1）
    # 如果全都是“拦截型”规则，那么没有命中任何规则的人（0）默认应该通过（1）
    
    final_decisions = decisions.copy()
    if has_approve_rules:
        # 无明确通过标识的，全部记为拒绝
        final_pass_count = int((final_decisions == 1).sum())
        final_rej_count = int((final_decisions <= 0).sum())
    else:
        # 无明确拒绝标识的，全部记为通过
        final_pass_count = int((final_decisions == 0).sum())
        final_rej_count = int((final_decisions == -1).sum())
    
    return {
        "total_count": total_count,
        "pass_count": final_pass_count,
        "hit_count": final_rej_count,
        "approval_rate": round(final_pass_count / total_count, 4) if total_count > 0 else 0,
        "rule_stats": rule_stats
    }
