# -*- coding: utf-8 -*-
"""
策略置换分析（Swap In/Out）核心计算模块

逻辑来源：《100天成为风控专家》策略置换D类调优（Python实操）

核心概念：
- 置入（Swap In）：旧策略拒绝 + 新策略通过的客群（新策略"捞回"）
- 置出（Swap Out）：旧策略通过 + 新策略拒绝的客群（新策略"新增拦截"）
- 拒绝推断法：置入客群无贷后标签，用新策略分箱badrate估算
"""

import pandas as pd
import numpy as np
from typing import Any, List, Optional


def _eval_rule(series: pd.Series, op: str, val: Any) -> pd.Series:
    """将字段值与阈值按操作符比较，返回布尔 Series（True=命中策略=拒绝）"""
    op = op.strip()
    # 处理 in 操作：val 可能是列表或逗号分隔字符串
    if op == 'in':
        if isinstance(val, str):
            val = [v.strip() for v in val.split(',')]
        elif not isinstance(val, (list, tuple)):
            val = [val]
        # 同时尝试字符串匹配和数值匹配（兼容混合类型列）
        str_vals = [str(v) for v in val]
        return series.astype(str).isin(str_vals)

    # 处理数值比较：尝试解析 val 为数值
    try:
        num_val = float(val)
    except (TypeError, ValueError):
        num_val = None

    if num_val is not None and op in ['>=', '>', '<=', '<', '==', '!=']:
        # 强制将 series 转为数值（Excel 读入的字符串数字也能处理）
        num_series = pd.to_numeric(series, errors='coerce')
        if op == '>=':
            return num_series >= num_val
        elif op == '>':
            return num_series > num_val
        elif op == '<=':
            return num_series <= num_val
        elif op == '<':
            return num_series < num_val
        elif op == '==':
            return num_series == num_val
        elif op == '!=':
            return num_series != num_val

    # 回退到字符串比较
    str_series = series.astype(str)
    str_val = str(val)
    if op == '==':
        return str_series == str_val
    elif op == '!=':
        return str_series != str_val
    else:
        raise ValueError(f"不支持的操作符或值类型: op={op}, val={val}")


def eval_strategy_rules(df: pd.DataFrame, rules: List[dict], combine_logic: str = 'and', rule_type: str = 'reject') -> pd.Series:
    """评估一组策略规则，返回布尔 Series (True=命中/拦截=1)"""
    if not rules:
        return pd.Series(False, index=df.index)
        
    masks = []
    for rule in rules:
        field = rule.get('field')
        op = rule.get('op')
        val = rule.get('val')
        
        if field not in df.columns:
            masks.append(pd.Series(False, index=df.index))
            continue
            
        col_data = df[field]
        val_final = val
        try:
            val_num = float(val)
            col_num = pd.to_numeric(df[field], errors='coerce')
            if not col_num.isna().all():
                col_data = col_num
                val_final = val_num
        except:
            pass

        if pd.api.types.is_numeric_dtype(col_data) and isinstance(val_final, str):
            col_data = col_data.astype(str)
        elif not pd.api.types.is_numeric_dtype(col_data) and not isinstance(val_final, str):
            val_final = str(val_final)

        # 处理 in 操作
        if op == 'in':
            if isinstance(val_final, str):
                vals = [v.strip() for v in val_final.split(',')]
            elif not isinstance(val_final, (list, tuple)):
                vals = [val_final]
            else:
                vals = val_final
            str_vals = [str(v) for v in vals]
            masks.append(col_data.astype(str).isin(str_vals))
            continue

        if op == '>':
            masks.append(col_data > val_final)
        elif op == '<':
            masks.append(col_data < val_final)
        elif op == '>=':
            masks.append(col_data >= val_final)
        elif op == '<=':
            masks.append(col_data <= val_final)
        elif op == '==':
            masks.append(col_data == val_final)
        elif op == '!=':
            masks.append(col_data != val_final)
        else:
            masks.append(pd.Series(False, index=df.index))

    if not masks:
        return pd.Series(False, index=df.index)

    if combine_logic == 'and':
        hit_mask = np.logical_and.reduce(masks)
    else:
        hit_mask = np.logical_or.reduce(masks)

    if rule_type == 'approve':
        return pd.Series(~hit_mask, index=df.index)
    else:
        return pd.Series(hit_mask, index=df.index)


def run_swap_analysis(
    df: pd.DataFrame,
    old_col: Optional[str] = None,
    old_reject_op: Optional[str] = None,
    old_reject_val: Any = None,
    new_col: Optional[str] = None,
    new_reject_op: Optional[str] = None,
    new_reject_val: Any = None,
    label_col: str = 'label',
    new_col_bins: Optional[List[float]] = None,
    old_rules: Optional[List[dict]] = None,
    old_combine_logic: str = 'and',
    old_rule_type: str = 'reject',
    new_rules: Optional[List[dict]] = None,
    new_combine_logic: str = 'and',
    new_rule_type: str = 'reject',
) -> dict:
    """
    执行新旧策略 Swap In/Out 分析。
    """
    if old_rules is None:
        if not old_col or not old_reject_op:
            return {"error": "未提供旧策略配置或规则列表"}
        old_rules = [{"field": old_col, "op": old_reject_op, "val": old_reject_val}]
    
    if new_rules is None:
        if not new_col or not new_reject_op:
            return {"error": "未提供新策略配置或规则列表"}
        new_rules = [{"field": new_col, "op": new_reject_op, "val": new_reject_val}]

    # ── Step 0: 字段校验 ─────────────────────────────────────
    for r in old_rules + new_rules:
        f = r.get('field')
        if f and f not in df.columns:
            return {"error": f"字段 '{f}' 不在数据集中"}

    # 决定用于分箱的新策略评分字段
    bin_col = new_col
    if not bin_col:
        for r in new_rules:
            f = r.get('field', '')
            if f.startswith('_model_result_') or 'score' in f.lower():
                bin_col = f
                break
        if not bin_col and new_rules:
            bin_col = new_rules[0].get('field')

    if not bin_col or bin_col not in df.columns:
        return {"error": f"用于新策略分箱的字段 '{bin_col}' 不在数据集中"}

    total_n = len(df)
    if total_n == 0:
        return {"error": "数据集为空"}

    # ── Step 1: 生成决策标记（1=拒绝，0=通过）────────────────
    try:
        df = df.copy()
        df['_old_hit'] = eval_strategy_rules(df, old_rules, old_combine_logic, old_rule_type).astype(int)
        df['_new_hit'] = eval_strategy_rules(df, new_rules, new_combine_logic, new_rule_type).astype(int)
    except Exception as e:
        return {"error": f"规则计算失败: {str(e)}"}

    # ── Step 2: 分离通过样本（有标签）和拒绝样本（无标签）────
    df_pass = df[df[label_col].notna()].copy()   # 旧策略通过 → 有贷后标签
    df_reject = df[df[label_col].isna()].copy()  # 旧策略拒绝 → 无标签

    if len(df_pass) == 0:
        return {"error": f"标签列 '{label_col}' 中没有非空值，请确认字段名或数据结构"}

    df_pass[label_col] = df_pass[label_col].astype(int)
    ttbadrate = df_pass[label_col].mean()

    # ── Step 3: 2×2 人数矩阵（在全量数据上） ─────────────────
    cnt_matrix = pd.crosstab(df['_old_hit'], df['_new_hit'])
    # 确保行列都有0/1
    for idx in [0, 1]:
        if idx not in cnt_matrix.index:
            cnt_matrix.loc[idx] = 0
        if idx not in cnt_matrix.columns:
            cnt_matrix[idx] = 0
    cnt_matrix = cnt_matrix.sort_index(axis=0).sort_index(axis=1)

    # ── Step 4: 2×2 坏客户数矩阵（在通过样本上）──────────────
    bad_matrix = df_pass.pivot_table(
        index='_old_hit', columns='_new_hit',
        values=label_col, aggfunc='sum'
    )
    for idx in [0, 1]:
        if idx not in bad_matrix.index:
            bad_matrix.loc[idx] = np.nan
        if idx not in bad_matrix.columns:
            bad_matrix[idx] = np.nan
    bad_matrix = bad_matrix.sort_index(axis=0).sort_index(axis=1)

    # ── Step 5: 拒绝推断法（估算 置入=旧拒新通 的坏率）────────
    # 置入客群 = 旧策略拒绝(old_hit=1) + 新策略通过(new_hit=0) 且无标签
    swap_in_df = df_reject[(df_reject['_old_hit'] == 1) & (df_reject['_new_hit'] == 0)]
    n_swap_in = len(swap_in_df)

    # 在通过样本上对新策略字段分箱（确保列为数值类型）
    new_col_numeric = pd.to_numeric(df_pass[bin_col], errors='coerce')

    if new_col_bins is None:
        # 自动等频分箱（6箱）
        quantiles = new_col_numeric.quantile([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]).tolist()
        new_col_bins = sorted(list(set(float(q) for q in quantiles)))

    # 修正分箱边界（确保全为 float）
    bin_edges = [-np.inf] + sorted(set(float(b) for b in new_col_bins)) + [np.inf]

    # 通过样本各分箱 badrate
    try:
        df_pass = df_pass.copy()
        df_pass['_new_col_num'] = new_col_numeric
        df_pass['_bin'] = pd.cut(df_pass['_new_col_num'], bins=bin_edges, right=False, include_lowest=True)
        bin_stats = df_pass.groupby('_bin', observed=False).agg(
            count=(label_col, 'count'),
            bad=(label_col, 'sum')
        ).reset_index()
        bin_stats['badrate'] = bin_stats['bad'] / bin_stats['count']
        bin_stats['lift'] = bin_stats['badrate'] / ttbadrate if ttbadrate > 0 else np.nan
    except Exception as e:
        return {"error": f"分箱计算失败: {str(e)}"}

    # 置入客群各分箱人数分布
    estimated_swap_in_bad = 0.0
    if n_swap_in > 0:
        try:
            swap_in_df = swap_in_df.copy()
            swap_in_df['_new_col_num'] = pd.to_numeric(swap_in_df[bin_col], errors='coerce')
            swap_in_df['_bin'] = pd.cut(swap_in_df['_new_col_num'], bins=bin_edges, right=False, include_lowest=True)
            swap_in_bin_cnt = swap_in_df.groupby('_bin', observed=False).size().reset_index(name='reject_cnt')
            bin_stats_merged = bin_stats.merge(swap_in_bin_cnt, on='_bin', how='left')
            bin_stats_merged['reject_cnt'] = bin_stats_merged['reject_cnt'].fillna(0)
            bin_stats_merged['est_bad_cnt'] = bin_stats_merged['reject_cnt'] * bin_stats_merged['badrate']
            estimated_swap_in_bad = bin_stats_merged['est_bad_cnt'].sum()
        except Exception:
            estimated_swap_in_bad = 0.0
    else:
        bin_stats_merged = bin_stats.copy()
        bin_stats_merged['reject_cnt'] = 0
        bin_stats_merged['est_bad_cnt'] = 0.0

    est_swap_in_badrate = estimated_swap_in_bad / n_swap_in if n_swap_in > 0 else 0.0

    # ── Step 6: 填充坏客户数矩阵的 (1,0) 置入格────────────────
    # (1=旧拒绝行, 0=新通过列) → 用推断值
    bad_matrix_filled = bad_matrix.copy()
    if 1 not in bad_matrix_filled.index:
        bad_matrix_filled.loc[1, :] = np.nan
    bad_matrix_filled.at[1, 0] = estimated_swap_in_bad

    # ── Step 7: 通过率对比 ────────────────────────────────────
    old_pass_n = int((df['_old_hit'] == 0).sum())
    new_pass_n = int((df['_new_hit'] == 0).sum())
    old_pass_rate = old_pass_n / total_n
    new_pass_rate = new_pass_n / total_n

    # ── Step 8: 通过样本坏率对比（含置入后的新策略整体坏率）──
    # 旧策略通过样本坏率
    old_passthrough_bad = float(df_pass.loc[df_pass['_old_hit'] == 0, label_col].sum())
    old_passthrough_n = int((df_pass['_old_hit'] == 0).sum())
    old_badrate = old_passthrough_bad / old_passthrough_n if old_passthrough_n > 0 else 0.0

    # 新策略通过样本（含通过样本中new_hit=0的真实坏客户 + 置入客群估算）
    new_pass_real_bad = float(df_pass.loc[df_pass['_new_hit'] == 0, label_col].sum())
    new_pass_real_n = int((df_pass['_new_hit'] == 0).sum())
    # 新策略整体通过样本（含置入）
    new_total_pass_bad = new_pass_real_bad + estimated_swap_in_bad
    new_total_pass_n = new_pass_real_n + n_swap_in
    new_badrate = new_total_pass_bad / new_total_pass_n if new_total_pass_n > 0 else 0.0

    # 置出（旧通新拒）真实指标
    swap_out_df_pass = df_pass[(df_pass['_old_hit'] == 0) & (df_pass['_new_hit'] == 1)]
    n_swap_out = int(cnt_matrix.at[0, 1]) if (0 in cnt_matrix.index and 1 in cnt_matrix.columns) else 0
    swap_out_real_bad = float(swap_out_df_pass[label_col].sum())
    swap_out_n_pass = len(swap_out_df_pass)
    swap_out_badrate = swap_out_real_bad / swap_out_n_pass if swap_out_n_pass > 0 else 0.0

    # ── Step 9: 组装结果 ──────────────────────────────────────
    def safe_int(v):
        try:
            return int(v) if not np.isnan(float(v)) else None
        except Exception:
            return None

    def safe_float(v, decimals=4):
        try:
            f = float(v)
            return round(f, decimals) if not np.isnan(f) else None
        except Exception:
            return None

    # 4格矩阵：(旧通新通, 旧通新拒, 旧拒新通, 旧拒新拒)
    n_00 = safe_int(cnt_matrix.at[0, 0]) if (0 in cnt_matrix.index and 0 in cnt_matrix.columns) else 0
    n_01 = safe_int(cnt_matrix.at[0, 1]) if (0 in cnt_matrix.index and 1 in cnt_matrix.columns) else 0
    n_10 = safe_int(cnt_matrix.at[1, 0]) if (1 in cnt_matrix.index and 0 in cnt_matrix.columns) else 0
    n_11 = safe_int(cnt_matrix.at[1, 1]) if (1 in cnt_matrix.index and 1 in cnt_matrix.columns) else 0

    br_00 = safe_float(bad_matrix_filled.at[0, 0] / cnt_matrix.at[0, 0]) if n_00 else None
    br_01 = safe_float(bad_matrix_filled.at[0, 1] / cnt_matrix.at[0, 1]) if n_01 else None
    br_10 = safe_float(est_swap_in_badrate)  # 推断值
    br_11 = None  # 旧策略拒绝且新策略拒绝，无标签且已被双重拒绝

    # 推断分箱明细表
    inference_table = []
    for _, row in bin_stats_merged.iterrows():
        inference_table.append({
            "bin": str(row['_bin']),
            "pass_count": safe_int(row['count']),
            "pass_bad": safe_int(row['bad']),
            "badrate": safe_float(row['badrate'], 4),
            "lift": safe_float(row['lift'], 4),
            "reject_cnt": safe_int(row['reject_cnt']),
            "est_bad_cnt": safe_float(row['est_bad_cnt'], 2),
        })

    # 综合结论文字
    delta_pass = new_pass_rate - old_pass_rate
    delta_bad = new_badrate - old_badrate
    summary_parts = []
    if delta_pass > 0:
        summary_parts.append(f"新策略通过率高于旧策略 {delta_pass*100:.2f}%")
    else:
        summary_parts.append(f"新策略通过率低于旧策略 {abs(delta_pass)*100:.2f}%")
    if delta_bad < 0:
        summary_parts.append(f"通过样本整体坏率下降 {abs(delta_bad)*100:.2f}%（坏率改善）")
    else:
        summary_parts.append(f"通过样本整体坏率上升 {abs(delta_bad)*100:.2f}%（请谨慎评估）")
    summary_parts.append(f"置出客群坏率为 {swap_out_badrate*100:.2f}%（高坏率客户被新策略有效拦截）")
    summary_parts.append(f"置入客群坏率为 {est_swap_in_badrate*100:.2f}%（新策略捞回客群风险可控）")

    return {
        # 2×2 决策矩阵（人数）
        "decision_matrix": {
            "双通_n": n_00,
            "置出_n": n_01,
            "置入_n": n_10,
            "双拒_n": n_11,
            "total": total_n,
        },
        # 2×2 坏率矩阵
        "badrate_matrix": {
            "双通_badrate": br_00,
            "置出_badrate": br_01,
            "置入_badrate_inferred": br_10,
            "置出_real_bad_count": safe_int(swap_out_real_bad),
            "置入_est_bad_count": safe_float(estimated_swap_in_bad, 1),
        },
        # 置入指标
        "swap_in": {
            "count": n_swap_in,
            "est_bad_count": safe_float(estimated_swap_in_bad, 1),
            "est_badrate": safe_float(est_swap_in_badrate, 4),
            "label": "置入（旧拒新通）",
            "note": "基于历史数据，坏率为真实值"
        },
        # 置出指标
        "swap_out": {
            "count": n_swap_out,
            "real_bad_count": safe_int(swap_out_real_bad),
            "real_badrate": safe_float(swap_out_badrate, 4),
            "label": "置出（旧通新拒）",
            "note": "坏率为真实数据"
        },
        # 通过率对比
        "pass_rate_comparison": {
            "old_pass_rate": safe_float(old_pass_rate, 4),
            "new_pass_rate": safe_float(new_pass_rate, 4),
            "delta": safe_float(delta_pass, 4),
            "old_pass_count": old_pass_n,
            "new_pass_count": new_pass_n,
            "total": total_n,
        },
        # 整体坏率对比（通过客群）
        "overall_badrate_comparison": {
            "old_badrate": safe_float(old_badrate, 4),
            "new_badrate": safe_float(new_badrate, 4),
            "delta": safe_float(delta_bad, 4),
            "overall_badrate": safe_float(ttbadrate, 4),
        },
        # 拒绝推断分箱明细
        "rejection_inference_table": inference_table,
        # 综合结论
        "summary_text": "；".join(summary_parts),
    }
