# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import toad
import re
from datetime import datetime
import json

try:
    from scorecard_core import new_psi, feature_plot_generator, iv_report
except ImportError:
    # 兼容性导入，如果结构不同可能需要调整
    pass

def _score_ks_bucket(score_series: pd.Series, label_series: pd.Series, bucket: int = 10) -> pd.DataFrame:
    """基于分数（低分=高风险）计算 KS 分箱表，升序排列（低分在前）"""
    df = pd.DataFrame({'score': score_series.values, 'label': label_series.values}).dropna()
    total = len(df)
    if total == 0:
        return pd.DataFrame(columns=['min', 'max', 'bad_rate', 'total_prop', 'cum_bad_rate', 'cum_total_prop'])
    try:
        df['bucket'] = pd.qcut(df['score'], q=bucket, duplicates='drop')
    except Exception:
        df['bucket'] = pd.cut(df['score'], bins=bucket, duplicates='drop')
    agg = df.groupby('bucket', observed=True)['label'].agg(['count', 'sum']).reset_index()
    agg.columns = ['bucket', 'count', 'bad']
    agg = agg.sort_values('bucket').reset_index(drop=True)
    agg['min'] = agg['bucket'].apply(lambda x: x.left)
    agg['max'] = agg['bucket'].apply(lambda x: x.right)
    agg['bad_rate'] = agg['bad'] / agg['count']
    agg['total_prop'] = agg['count'] / total
    agg['cum_total_prop'] = agg['total_prop'].cumsum()
    agg['cum_bad'] = agg['bad'].cumsum()
    agg['cum_count'] = agg['count'].cumsum()
    agg['cum_bad_rate'] = agg['cum_bad'] / agg['cum_count']
    return agg[['min', 'max', 'bad_rate', 'total_prop', 'cum_bad_rate', 'cum_total_prop']]


def generate_model_report(data_end, model, keep_lst, save_dir, model_id, res_data=None, res_month=None, pdo_df=None, if_weight=True, label_col='label', target_col='target', score_config=None):
    """
    基于用户提供的逻辑生成模型报告
    :param data_end: pd.DataFrame, 包含 'proba', label_col, target_col, 'month_time', 'weight' 等字段
...
    :param if_weight: 是否使用权重
    :param label_col: 标签列名
    :param target_col: 划分列名
    """
    os.makedirs(save_dir, exist_ok=True)
    report_file = os.path.join(save_dir, f"model_report_{model_id}.xlsx")
    
    # --- PSI 计算 (月度) ---
    # 获取最小月份并将其数据分布作为 Baseline
    min_month = data_end['month_time'].min()
    data_min_month = data_end[data_end['month_time'] == min_month]
    
    # 在 概率(proba) 上计算分箱，以 train 或最小月份为基准
    # 这里以 train 的 proba 分箱为基准（更可靠）
    data_train = data_end[data_end[target_col] == 'train']
    # 强制 20 等分（或 10 等分）
    train_edges = toad.transform.Combiner().fit(data_train[['proba', label_col]], y=label_col, method='quantile', n_bins=10).export()['proba']
    # 转换为包含极值的完整边界列表，防止 pd.cut 报错且覆盖全量程
    train_proba_bins = [-np.inf] + list(train_edges) + [np.inf]
    
    # 使用最小月份的分箱占比作为原始对照
    def get_dist(df, bins):
        if df.empty:
            return np.zeros(len(bins)-1)
        return pd.cut(df['proba'], bins=bins, include_lowest=True).value_counts(normalize=True).sort_index().values

    # 计算逐月 PSI
    psi_list = []
    months = sorted(data_end['month_time'].unique())
    base_dist = get_dist(data_min_month, train_proba_bins)
    
    for m in months:
        if m == min_month:
            psi_list.append({'month': m, 'psi': 0.0, 'is_baseline': True})
            continue
        curr_dist = get_dist(data_end[data_end['month_time'] == m], train_proba_bins)
        # 简单计算 PSI (含 0 处理)
        b = np.clip(base_dist, 1e-6, 1)
        c = np.clip(curr_dist, 1e-6, 1)
        psi_val = np.sum((c - b) * np.log(c / b))
        psi_list.append({'month': m, 'psi': float(psi_val), 'is_baseline': False})
    
    psi_df_month = pd.DataFrame(psi_list)
    
    # 训练集 vs OOT 的总 PSI
    data_oot = data_end[data_end[target_col] == 'oot']
    if not data_oot.empty:
        oot_dist = get_dist(data_oot, train_proba_bins)
        b = np.clip(get_dist(data_train, train_proba_bins), 1e-6, 1)
        o = np.clip(oot_dist, 1e-6, 1)
        overall_psi = np.sum((o - b) * np.log(o / b))
    else:
        overall_psi = 0.0
    
    psi_df_tv_o = pd.DataFrame([{'metric': 'Train vs OOT PSI', 'value': overall_psi}])
    
    # 2. 数据概要 (info_df)
    df_list = []
    temp_df = pd.DataFrame({
        'dataset': 'all',
        'count': data_end[label_col].count(),
        'badrate': data_end[label_col].mean(),
        'badrate_weight': (data_end[label_col] * data_end['weight']).sum() / data_end['weight'].sum() if if_weight else 0,
        'month_min': data_end['month_time'].min(),
        'month_max': data_end['month_time'].max()
    }, index=[0])
    df_list.append(temp_df)

    def group_weight_badrate_cal(group):
        return (group[label_col] * group['weight']).sum() / group['weight'].sum()

    for col in [target_col, 'month_time']:
        grouped = data_end.groupby(col).agg({label_col: ['count', 'mean'], 'month_time': ['min', 'max']}).reset_index()
        grouped.columns = ['dataset', 'count', 'badrate', 'month_min', 'month_max']
        if if_weight:
            grouped['badrate_weight'] = (
                data_end.groupby(col)
                .apply(group_weight_badrate_cal)
                .reset_index(drop=True)
            )
        grouped.sort_values(by='dataset', ascending=True, inplace=True)
        df_list.append(grouped)
    info_df = pd.concat(df_list, axis=0, ignore_index=True)
    if not if_weight:
        info_df.drop(columns=['badrate_weight'], axis=1, inplace=True, errors='ignore')

    # 3. Lift 表 (lift_df) — 基于分数分箱（与策略引擎一致）
    tag_list = ['train', 'valid', 'oot']
    tag_list_str = tag_list
    df_list_lift = []
    use_score = 'score' in data_end.columns
    for tag_val in tag_list:
        temp_df_sub = data_end[data_end[target_col].isin([tag_val])]
        if not temp_df_sub.empty:
            if use_score:
                lift_df_diff = _score_ks_bucket(temp_df_sub['score'], temp_df_sub[label_col], bucket=10)
            else:
                score_ks = toad.metrics.KS_bucket(temp_df_sub['proba'], temp_df_sub[label_col], bucket=10)
                lift_df_diff = score_ks[['min', 'max', 'bad_rate', 'total_prop', 'cum_bad_rate', 'cum_total_prop']]
            df_list_lift.append(lift_df_diff)
        else:
            df_list_lift.append(pd.DataFrame())
    lift_df = pd.concat(df_list_lift, axis=1, keys=tag_list_str)

    # 4. IV & Importance & KS (ipt_iv_df)
    iv_calculator = iv_report.IVCalculator(data_end, label=label_col, target=target_col, keep_list=keep_lst, max_leaf_nodes=6, min_samples_leaf=0.05)
    iv_df = iv_calculator.iv_report(use_thread=True, max_workers=20)
    
    # 计算每个特征在各数据集上的 KS
    ks_results = []
    for var in keep_lst:
        var_ks = {'var_name': var}
        # 使用 iv_calculator 中的 bins 进行分箱后计算 KS
        bins = iv_calculator.bins_dict.get(var)
        for tag in ['train', 'valid', 'oot']:
            subset = data_end[data_end[target_col] == tag]
            if not subset.empty and bins is not None:
                try:
                    # 使用 toad 计算分箱后的 KS
                    c = toad.transform.Combiner()
                    c.load({var: bins})
                    binned = c.transform(subset[[var, label_col]])
                    ks = toad.metrics.KS(binned[var], binned[label_col])
                    var_ks[f'{tag}_ks'] = float(ks)
                except:
                    var_ks[f'{tag}_ks'] = 0.0
            else:
                var_ks[f'{tag}_ks'] = 0.0
        ks_results.append(var_ks)
    ks_df = pd.DataFrame(ks_results)

    iv_df.sort_values(by='train_iv', ascending=False, inplace=True)
    
    ipt_df = pd.DataFrame({'var_name': keep_lst, 'importance': model.feature_importances_})
    ipt_df.sort_values(by='importance', ascending=False, inplace=True)
    
    ipt_iv_df = pd.merge(ipt_df, iv_df, on='var_name', how='left')
    ipt_iv_df = pd.merge(ipt_iv_df, ks_df, on='var_name', how='left')
    
    # 5. 分箱详情 (iv_detail_df)
    iv_detail_df = iv_calculator.detail_report(ipt_iv_df['var_name'], 'all')
    # 不再丢弃 ks, badCumRate 等，保留 Lift 和 CumLift

    # 排序分箱
    def extract_left(bin_str):
        try:
            left = re.findall(r'[\[\(](-inf|-?\d+\.?\d*)', str(bin_str))[0]
            return float('-inf') if left == '-inf' else float(left)
        except IndexError:
            return float('inf')
    
    iv_detail_df = iv_detail_df.sort_values(
        by=['var_name', 'bin'],
        key=lambda s: s if s.name == 'var_name' else s.map(extract_left),
        ascending=[True, True]
    )

    # 6. 模型评估对比 (eval_df)
    # 如果没传入 res_data/res_month，需要自行算一下基础 KS/AUC
    if res_data is None:
        perf_list = []
        for tag in ['train', 'valid', 'oot']:
            td = data_end[data_end[target_col] == tag]
            if not td.empty:
                if td[label_col].nunique() > 1:
                    auc = toad.metrics.AUC(td['proba'], td[label_col])
                    ks = toad.metrics.KS(td['proba'], td[label_col])
                else:
                    auc = 0.5
                    ks = 0.0
                perf_list.append({'datasets': tag, 'auc': auc, 'ks': ks})
        res_data = pd.DataFrame(perf_list)
        
    if res_month is None:
        perf_list_m = []
        for m in data_end['month_time'].unique():
            td = data_end[data_end['month_time'] == m]
            if not td.empty:
                if td[label_col].nunique() > 1:
                    auc = toad.metrics.AUC(td['proba'], td[label_col])
                    ks = toad.metrics.KS(td['proba'], td[label_col])
                else:
                    auc = 0.5
                    ks = 0.0
                perf_list_m.append({'datasets': m, 'auc': auc, 'ks': ks})
        res_month = pd.DataFrame(perf_list_m)

    model_performance_ks = pd.concat([res_data[['datasets','auc','ks']], res_month[['datasets','auc','ks']]])

    def calc_lift_group_v2(group):
        res = {}
        bad_rate_all = group[label_col].mean()
        for n in [1, 2, 3, 5, 10, 20]:
            thr = group['proba'].quantile(1 - n/100)
            top_n = group[group['proba'] >= thr]
            bad_rate_n = top_n[label_col].mean()
            res[f'top_{n}_lift'] = bad_rate_n / bad_rate_all if bad_rate_all > 0 else 0
        return pd.Series(res)

    result_data = data_end.groupby(target_col).apply(calc_lift_group_v2).reset_index()
    result_month = data_end.groupby('month_time').apply(calc_lift_group_v2).reset_index()
    result_data.rename(columns={target_col: 'datasets'}, inplace=True)
    result_month.rename(columns={'month_time': 'datasets'}, inplace=True)
    model_performance_lift = pd.concat([result_data, result_month])
    eval_df = pd.merge(model_performance_ks, model_performance_lift, on='datasets', how='left')

    # 导出 Excel
    with pd.ExcelWriter(report_file, engine='openpyxl') as writer:
        info_df.to_excel(writer, sheet_name='数据概要', index=False)
        eval_df.to_excel(writer, sheet_name='模型性能', index=False)
        ipt_iv_df.to_excel(writer, sheet_name='特征重要性', index=False)
        iv_detail_df.to_excel(writer, sheet_name='特征分箱', index=False)
        lift_df.to_excel(writer, sheet_name='lift', index=True)
        psi_df_tv_o.to_excel(writer, sheet_name='psi_训练与oot', index=False)
        psi_df_month.to_excel(writer, sheet_name='psi_逐月', index=False)
        if pdo_df is not None:
            pdo_df.to_excel(writer, sheet_name='打分参数', index=False)
        
        # 特征分布报告图表
        # 为图表生成组件准备兼容的列名 (组件固化了 'target' 和 'label')
        plot_data = data_end.copy()
        # 预清理：如果数据中已存在 'target' 或 'label' 且不是我们要重命名的那一列，则剔除，防止列名重复
        if 'target' in plot_data.columns and target_col != 'target':
            plot_data = plot_data.drop(columns=['target'])
        if 'label' in plot_data.columns and label_col != 'label':
            plot_data = plot_data.drop(columns=['label'])
            
        plot_data = plot_data.rename(columns={target_col: 'target', label_col: 'label'})
        
        # 只取前 10 个最重要的特征生成图表
        top_10_features = list(ipt_iv_df['var_name'].iloc[:10])
        
        feature_plot_generator.create_feature_plot_report(
            excel_writer=writer,
            data=plot_data,
            feature_list=top_10_features,
            sheet_name='特征分箱报告'
        )

    # 返回给前端显示的数据字典 (转换为 JSON 友好格式)
    # 对于 multi-index column 的 lift_df, 转换成平坦结构
    flat_lift = lift_df.copy()
    if isinstance(flat_lift.columns, pd.MultiIndex):
        flat_lift.columns = [f"{c[0]}_{c[1]}" for c in flat_lift.columns]
    
    report_json = {
        "data_summary": info_df.to_dict(orient='records'),
        "performance_eval": eval_df.to_dict(orient='records'),
        "feature_importance": ipt_iv_df.to_dict(orient='records'),
        "bin_details": iv_detail_df.to_dict(orient='records'),
        "lift_table": flat_lift.to_dict(orient='records'),
        "psi_train_oot": psi_df_tv_o.to_dict(orient='records'),
        "psi_monthly": psi_df_month.to_dict(orient='records'),
        "file_path": report_file
    }
    
    return report_json

def clean_serializable(obj):
    """递归清理对象中的 NaN 和 Inf，转换为 None，以适配 JSON 标准"""
    if isinstance(obj, list):
        return [clean_serializable(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: clean_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj

def run_report_task(db, task_id, progress_callback, project_id, model_result_id, **kwargs):
    """
    异步任务执行函数
    """
    from app.models import Project, Dataset, ModelResult, ModelReport
    from scorecard_core.data_processor import load_data, split_dataset
    import joblib
    
    progress_callback(10, {"message": "正在加载数据与模型..."})
    
    project = db.query(Project).filter(Project.id == project_id).first()
    model_result = db.query(ModelResult).filter(ModelResult.id == model_result_id).first()
    
    if not model_result:
        raise Exception("模型结果未找到")
        
    # 加载数据集：优先从 model_result.params 取 dataset_id，精确匹配
    from app.models import Task as TaskModel
    model_params = model_result.params or {}
    dataset_id_from_params = model_params.get('dataset_id')

    # 兜底：从关联的 Task.params 取（老数据 ModelResult.params 没存 dataset_id）
    if not dataset_id_from_params and model_result.task_id:
        task_obj = db.query(TaskModel).filter(TaskModel.id == model_result.task_id).first()
        if task_obj and task_obj.params:
            dataset_id_from_params = task_obj.params.get('dataset_id')

    if dataset_id_from_params:
        dataset = db.query(Dataset).filter(
            Dataset.id == dataset_id_from_params,
            Dataset.project_id == project_id
        ).first()
        if not dataset:
            dataset = db.query(Dataset).filter(Dataset.project_id == project_id).first()
    else:
        dataset = db.query(Dataset).filter(Dataset.project_id == project_id).first()

    if not dataset:
        raise Exception(
            f"未找到关联数据集（project_id={project_id}，"
            f"dataset_id={dataset_id_from_params or '未记录'}）"
        )
        
    df = load_data(dataset.file_path)
    
    # 划分数据 (必须与建模时一致)
    split_config = model_result.params.get('split_config') or project.split_config
    datasets = split_dataset(
        df, 
        ratios=split_config.get('split_ratios', [0.6, 0.2, 0.2]),
        oot_col=split_config.get('oot_col'),
        oot_start_time=split_config.get('oot_start_time'),
        oot_pct=split_config.get('oot_pct')
    )
    
    train_df = datasets.get('train', pd.DataFrame())
    valid_df = datasets.get('valid', pd.DataFrame())
    oot_df = datasets.get('oot', pd.DataFrame())
    
    # 标记 target
    target_col = 'target_tmp'
    train_df[target_col] = 'train'
    valid_df[target_col] = 'valid'
    oot_df[target_col] = 'oot'
    data_end = pd.concat([train_df, valid_df, oot_df], axis=0).reset_index(drop=True)
    
    # 获取 label 字段
    label_col = (model_result.params or {}).get('dep', 'label')
    if label_col not in data_end.columns:
        # 兜底查找
        if 'label' in data_end.columns:
            label_col = 'label'
    
    # 加载模型
    model = joblib.load(model_result.model_path)
    
    progress_callback(30, {"message": "正在计算打分与概率..."})

    # 计算概率
    features = model_result.feature_list
    data_end['proba'] = model.predict_proba(data_end[features])[:, 1]

    # 计算分数（与策略引擎保持一致）
    sc = model_result.score_config or {}
    try:
        from scorecard_core.scoring import proba2score
        data_end['score'] = proba2score(
            data_end['proba'].values,
            pdo=sc.get('pdo', 20),
            base_score=sc.get('base_score', 600),
            base_odds=sc.get('base_odds', 50),
        )
    except Exception:
        pass  # 打分失败时降级为概率分箱
    
    # 如果有权重列
    if 'weight' not in data_end.columns:
        data_end['weight'] = 1.0
        
    # 如果有月度列
    oot_col_name = split_config.get('oot_col')
    if oot_col_name and oot_col_name in data_end.columns:
        data_end['month_time'] = pd.to_datetime(data_end[oot_col_name]).dt.to_period('M').astype(str)
    else:
        # 默认使用创建时间或固定值
        data_end['month_time'] = '2024-01'

    progress_callback(50, {"message": "完成指标计算，正在生成详细内容..."})
    
    save_dir = os.path.join(os.path.dirname(model_result.model_path), "reports")
    
    report_data = generate_model_report(
        data_end=data_end,
        model=model,
        keep_lst=features,
        save_dir=save_dir,
        model_id=model_result_id,
        res_data=None, # 内部重新计算
        res_month=None,
        pdo_df=None, # 如果有评分卡参数可以传入
        if_weight=True,
        label_col=label_col,
        target_col=target_col
    )
    
    progress_callback(90, {"message": "正在保存报告到数据库..."})
    
    # 保存结果到数据库
    report = db.query(ModelReport).filter(ModelReport.model_result_id == model_result_id).first()
    if not report:
        report = ModelReport(
            project_id=project_id,
            model_result_id=model_result_id
        )
        db.add(report)
        
    report.data_summary = clean_serializable(report_data['data_summary'])
    report.performance_eval = clean_serializable(report_data['performance_eval'])
    report.feature_importance = clean_serializable(report_data['feature_importance'])
    report.bin_details = clean_serializable(report_data['bin_details'])
    report.lift_table = clean_serializable(report_data['lift_table'])
    report.psi_train_oot = clean_serializable(report_data['psi_train_oot'])
    report.psi_monthly = clean_serializable(report_data['psi_monthly'])
    report.file_path = report_data['file_path']
    
    db.commit()
    
    return {"report_id": report.id, "file_path": report.file_path}
