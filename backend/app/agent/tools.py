# -*- coding: utf-8 -*-
"""
Agent Tools 实现层

直接调用 scorecard_core 的核心函数，不走 HTTP，避免二次鉴权开销。
每个 tool 函数对应 tool_schemas.py 中同名工具。
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy import func as sa_func

from app.models import Project, Dataset, Task, ModelResult, ModelReport
from app.config import (
    UPLOAD_DIR, MODEL_DIR,
    AGENT_POLL_INTERVAL, AGENT_HB_INTERVAL,
    IMA_CLIENT_ID, IMA_API_KEY,
)

logger = logging.getLogger(__name__)


class AgentToolkit:
    """
    Agent 工具集，持有 DB session 和 user_id，代理用户身份执行所有建模操作。
    """

    def __init__(self, db: Session, user_id: int):
        self.db = db
        self.user_id = user_id

    # ──────────────────────────────────────────────────────────
    # Tool 0: get_data_overview  （建模前置，数据探索）
    # ──────────────────────────────────────────────────────────
    def get_data_overview(self, project_id: int, dataset_id: int) -> dict:
        """
        数据集概览：前5行 + 整体统计摘要 + 缺失摘要 + 疑似排除列推荐。
        建模前调用，让用户确认排除列。
        """
        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return {"error": f"数据集 {dataset_id} 不存在"}

        try:
            import math
            from scorecard_core.data_processor import load_data

            df = load_data(dataset.file_path)

            def _safe(v):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return None
                if isinstance(v, float):
                    return round(v, 4)
                return v

            # ── 前5行 ────────────────────────────────────────────
            head_rows = [
                {col: _safe(val) for col, val in row.items()}
                for row in df.head(5).to_dict(orient='records')
            ]

            # ── 列类型映射 ───────────────────────────────────────
            dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

            # ── 类型分布摘要 ────────────────────────────────────
            num_cols  = df.select_dtypes(include='number').columns.tolist()
            obj_cols  = df.select_dtypes(include='object').columns.tolist()
            dt_cols   = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
            type_summary = {
                "numeric_count":  len(num_cols),
                "object_count":   len(obj_cols),
                "datetime_count": len(dt_cols),
            }

            # ── 整体数值摘要（所有数值列的汇总，一行数据）────────
            numeric_overall = {}
            if num_cols:
                desc = df[num_cols].describe()
                numeric_overall = {
                    "mean_of_means": _safe(desc.loc["mean"].mean()),
                    "mean_of_stds":  _safe(desc.loc["std"].mean()),
                    "overall_min":   _safe(desc.loc["min"].min()),
                    "overall_max":   _safe(desc.loc["max"].max()),
                }

            # ── 缺失摘要 ────────────────────────────────────────
            miss_series = df.isnull().mean()
            miss_cols   = miss_series[miss_series > 0].sort_values(ascending=False)
            missing_summary = {
                "total_missing_cols": int((miss_series > 0).sum()),
                "overall_missing_rate": _safe(float(df.isnull().mean().mean())),
                "top_missing": [
                    {"column": col, "missing_rate": round(float(rate), 4)}
                    for col, rate in miss_cols.head(8).items()
                ],
            }

            # ── 疑似排除列推荐 ──────────────────────────────────
            ID_KEYWORDS   = {'id', 'no', 'num', 'number', 'key', 'code', 'uuid', 'seq', 'sn', 'pk', 'fk'}
            TIME_KEYWORDS = {'date', 'time', 'dt', 'day', 'month', 'year', 'ts', 'timestamp', 'period'}

            suspect_exclude = []
            for col in df.columns:
                dtype   = str(df[col].dtype)
                col_lw  = col.lower()
                col_parts = set(col_lw.replace('_', ' ').replace('-', ' ').split()) | {col_lw}
                reason  = None

                if col_parts & ID_KEYWORDS:
                    reason = "疑似 ID / 编号列"
                elif col_parts & TIME_KEYWORDS:
                    reason = "疑似时间 / 日期列"
                elif dtype == 'object':
                    n_unique = df[col].nunique()
                    if n_unique > max(len(df) * 0.5, 100):
                        reason = f"字符型且唯一值过多（{n_unique} 个），疑似 ID"

                if reason:
                    suspect_exclude.append({
                        "column": col,
                        "dtype":  dtype,
                        "reason": reason,
                    })

            return {
                "shape":           {"rows": len(df), "columns": len(df.columns)},
                "columns":         list(df.columns),
                "dtypes":          dtypes,
                "head":            head_rows,
                "type_summary":    type_summary,
                "numeric_overall": numeric_overall,   # 所有数值列整体汇总（非逐列）
                "missing_summary": missing_summary,
                "suspect_exclude": suspect_exclude,   # 建议排除的列，供 LLM 向用户确认
            }

        except Exception as e:
            logger.exception("get_data_overview 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    def run_iv_report(
        self,
        project_id: int,
        dataset_id: int,
        dep: str,
        split_ratios: list = None,
        exclude_cols: list = None,
    ) -> dict:
        """计算 IV / PSI 报告，持久化到 project.iv_report"""
        split_ratios = split_ratios or [0.6, 0.2, 0.2]
        exclude_cols = exclude_cols or []

        project = self.db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return {"error": f"项目 {project_id} 不存在"}

        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return {"error": f"数据集 {dataset_id} 不存在"}

        try:
            from scorecard_core.data_processor import load_data, split_dataset
            from scorecard_core.feature_engineer import calc_multi_set_metrics

            df = load_data(dataset.file_path)
            datasets = split_dataset(df, dep=dep, ratios=split_ratios)

            default_exclude = list(set(exclude_cols + [dep, 'target', 'weight']))
            ft_lst = [c for c in df.columns if c not in default_exclude]

            report = calc_multi_set_metrics(datasets, ft_lst, dep=dep)
            features = report.to_dict(orient='records')

            # 持久化
            project.iv_report = features
            project.split_config = {
                'split_ratios': split_ratios,
                'oot_col': None,
                'oot_start_time': None,
                'oot_pct': None
            }
            self.db.commit()

            # 给 LLM 返回摘要（前 10 条 + 统计），避免 token 爆炸
            valid_features = [f for f in features if f.get('iv_train', 0) >= 0.01]
            top10 = sorted(valid_features, key=lambda x: x.get('iv_train', 0), reverse=True)[:10]
            return {
                "total_features": len(ft_lst),
                "valid_features_iv_gt_001": len(valid_features),
                "top10_by_iv": [
                    {"name": f.get("feature"), "iv_train": round(f.get("iv_train", 0), 4),
                     "psi": round(f.get("psi", 0), 4)}
                    for f in top10
                ]
            }
        except Exception as e:
            logger.exception("run_iv_report 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool 2: filter_features
    # ──────────────────────────────────────────────────────────
    def filter_features(
        self,
        project_id: int,
        dataset_id: int,
        dep: str,
        iv_threshold: float = 0.02,
        psi_threshold: float = 0.1,
        corr_threshold: float = 0.9,
        exclude_cols: list = None,
    ) -> dict:
        """L2 变量筛选，将保留特征列表写入 project.feature_list"""
        exclude_cols = exclude_cols or []

        project = self.db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return {"error": f"项目 {project_id} 不存在"}

        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return {"error": f"数据集 {dataset_id} 不存在"}

        try:
            from scorecard_core.data_processor import load_data, split_dataset
            from scorecard_core.feature_engineer import filter_features as do_filter

            df = load_data(dataset.file_path)
            split_config = project.split_config or {}
            datasets = split_dataset(
                df, dep=dep,
                ratios=split_config.get('split_ratios', [0.6, 0.2, 0.2])
            )

            default_exclude = list(set(exclude_cols + [dep, 'target', 'weight']))
            ft_lst = [c for c in df.columns if c not in default_exclude]

            thresholds = {
                "iv": iv_threshold,
                "psi": psi_threshold,
                "corr": corr_threshold,
                "missing": 0.8,
                "std": 0.95,
                "freq": 0.95,
                "importance": 0.0,
                "chi2": 3.0,
            }
            result = do_filter(
                datasets, ft_lst, dep, thresholds,
                exclude_cols=default_exclude,
                skip_l1=True,
                impute_value=dataset.impute_value
            )

            # 持久化
            project.feature_list = result['kept_features']
            project.filter_result = result
            self.db.commit()

            return {
                "kept_count": len(result['kept_features']),
                "removed_count": len(ft_lst) - len(result['kept_features']),
                "kept_features_preview": result['kept_features'][:10],  # 给 LLM 看前 10
            }
        except Exception as e:
            logger.exception("filter_features 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool 3: start_training
    # ──────────────────────────────────────────────────────────
    async def start_training(
        self,
        project_id: int,
        dataset_id: int,
        dep: str,
        model_type: str = "xgb",
        n_trials: int = 30,
        exclude_cols: list = None,
    ) -> dict:
        """提交 Optuna 建模任务，返回 task_id"""
        exclude_cols = exclude_cols or []

        project = self.db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return {"error": f"项目 {project_id} 不存在"}
        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return {"error": f"数据集 {dataset_id} 不存在"}
        if not project.feature_list:
            return {"error": "特征列表为空，请先执行 filter_features"}

        try:
            task = Task(
                project_id=project_id,
                task_type='modeling',
                status='pending',
                params={
                    'dataset_id': dataset_id,
                    'dep': dep,
                    'model_type': model_type,
                    'n_trials': n_trials,
                    'feature_list': project.feature_list,
                    'exclude_cols': exclude_cols,
                }
            )
            self.db.add(task)
            self.db.commit()
            self.db.refresh(task)

            from scorecard_core.model_trainer import run_optuna_training
            from app.task_manager import submit_task

            model_save_dir = os.path.join(MODEL_DIR, str(project_id))
            split_config = project.split_config or {}

            await submit_task(
                task.id,
                run_optuna_training,
                data_path=dataset.file_path,
                dep=dep,
                model_type=model_type,
                strategy_type=3,
                strategy_threshold=0.03,
                n_trials=n_trials,
                max_depth=6,
                feature_list=project.feature_list,
                exclude_cols=exclude_cols,
                project_id=project_id,
                model_save_dir=model_save_dir,
                split_ratios=split_config.get('split_ratios', [0.6, 0.2, 0.2]),
                oot_col=split_config.get('oot_col'),
                oot_start_time=split_config.get('oot_start_time'),
                score_config=None,
            )

            return {
                "task_id": task.id,
                "model_type": model_type,
                "n_trials": n_trials,
                "feature_count": len(project.feature_list),
            }
        except Exception as e:
            logger.exception("start_training 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool 4: poll_task_until_done  （供 pipeline 内部调用，也作为 LLM 工具）
    # ──────────────────────────────────────────────────────────
    async def poll_task_until_done(self, task_id: int) -> dict:
        """
        轮询任务状态，同时维持心跳，直到 completed / failed。
        poll_interval 和 heartbeat_interval 从 config 读取。
        """
        last_hb_time = 0.0
        while True:
            # 用新 session 读最新状态（当前 session 可能是只读缓存）
            self.db.expire_all()
            task = self.db.query(Task).filter(Task.id == task_id).first()
            if not task:
                return {"error": f"任务 {task_id} 不存在"}

            # 定时心跳
            now = asyncio.get_event_loop().time()
            if now - last_hb_time >= AGENT_HB_INTERVAL:
                task.last_heartbeat = sa_func.now()
                self.db.commit()
                last_hb_time = now

            if task.status == 'completed':
                # 如果是策略挖掘任务，结果直接在 task.result 里，我们需要直接返回它
                if task.task_type == 'strategy_mining':
                    return {
                        "status": "completed",
                        "recommendations": task.result
                    }

                result = self.db.query(ModelResult).filter(
                    ModelResult.task_id == task_id
                ).first()
                if result:
                    return {
                        "status": "completed",
                        "model_result_id": result.id,
                        "model_type": result.model_type,
                        "metrics": result.metrics,
                    }
                return {"status": "completed", "model_result_id": None,
                        "warning": "ModelResult 未找到，可能写入延迟"}

            if task.status == 'failed':
                return {"status": "failed", "error": task.error_msg}

            await asyncio.sleep(AGENT_POLL_INTERVAL)

    # ──────────────────────────────────────────────────────────
    # Tool 5: generate_report
    # ──────────────────────────────────────────────────────────
    async def generate_report(self, project_id: int, model_result_id: int) -> dict:
        """提交报告生成任务，返回 task_id（需再次 poll）"""
        try:
            task = Task(
                project_id=project_id,
                task_type='model_report',
                status='pending',
                params={'model_id': model_result_id}
            )
            self.db.add(task)
            self.db.commit()
            self.db.refresh(task)

            from scorecard_core.report import run_report_task
            from app.task_manager import submit_task

            await submit_task(
                task.id,
                run_report_task,
                project_id=project_id,
                model_result_id=model_result_id
            )
            return {"task_id": task.id}
        except Exception as e:
            logger.exception("generate_report 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool 6: get_model_metrics
    # ──────────────────────────────────────────────────────────
    def get_model_metrics(self, project_id: int, model_result_id: int) -> dict:
        """获取模型完整指标 + 报告文件路径，作为建模流程最后一步"""
        result = self.db.query(ModelResult).filter(
            ModelResult.id == model_result_id,
            ModelResult.project_id == project_id
        ).first()
        if not result:
            return {"error": "模型结果不存在"}

        report = self.db.query(ModelReport).filter(
            ModelReport.model_result_id == model_result_id
        ).first()

        return {
            "model_result_id": result.id,
            "model_type": result.model_type,
            "metrics": result.metrics,
            "feature_count": len(result.feature_list or []),
            "top_features": (result.feature_importance or {}).get('sorted_features', [])[:10],
            "report_file_path": report.file_path if report else None,
            "report_generated": report is not None,
        }

    # ──────────────────────────────────────────────────────────
    # Tool 7: auto_strategy_mining
    # ──────────────────────────────────────────────────────────
    async def auto_strategy_mining(
        self,
        project_id: int,
        dataset_id: int,
        label_col: str,
        tree_type: str = 'exrf',
        **kwargs  # 兼容大模型幻觉传入的多余参数如 model_result_id
    ) -> dict:
        """提交策略挖掘任务，返回 task_id"""
        project = self.db.query(Project).filter(Project.id == project_id).first()
        if not project: return {"error": "项目不存在"}
        
        try:
            task = Task(
                project_id=project_id,
                task_type='strategy_mining',
                status='pending',
                params={'dataset_id': dataset_id, 'label_col': label_col, 'tree_type': tree_type}
            )
            self.db.add(task)
            self.db.commit()
            self.db.refresh(task)
            
            from scorecard_core.strategy_mining import run_auto_mining_task
            from app.task_manager import submit_task
            
            await submit_task(
                task.id,
                run_auto_mining_task,
                dataset_id=dataset_id,
                label_col=label_col,
                tree_type=tree_type
            )
            return {"task_id": task.id}
        except Exception as e:
            logger.exception("auto_strategy_mining 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool 8: deploy_model
    # ──────────────────────────────────────────────────────────
    def deploy_model(self, project_id: int, model_result_id: int) -> dict:
        """标记模型为部署状态，并退休旧部署"""
        from app.models import Deployment, ModelResult
        res = self.db.query(ModelResult).filter(ModelResult.id == model_result_id, ModelResult.project_id == project_id).first()
        if not res:
            return {"error": "模型结果不存在"}
            
        try:
            # 1. 将该项目下所有旧的活跃部署标记为 retired
            old_deploys = self.db.query(Deployment).filter(
                Deployment.project_id == project_id,
                Deployment.status == 'active'
            ).all()
            for od in old_deploys:
                od.status = 'retired'
            
            # 2. 新增当前部署
            dep = Deployment(project_id=project_id, model_result_id=model_result_id, status='active')
            self.db.add(dep)
            self.db.commit()
            self.db.refresh(dep)
            return {"deployment_id": dep.id, "status": "部署成功，旧模型已下线", "model_type": res.model_type}
        except Exception as e:
            logger.exception("deploy_model 失败")
            return {"error": str(e)}

    def analyze_strategy(self, project_id: int, dataset_id: int, rules: list, combine_logic: str = 'and', rule_type: str = 'reject', **kwargs) -> dict:
        """运行单变量/多变量策略规则分析，返回拦截率、坏率等指标"""
        from app.models import Dataset, ModelResult
        import re
        from scorecard_core.data_processor import load_data
        from scorecard_core.strategy_engine import run_strategy_analysis, enrich_df_with_model_scores
        
        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return {"error": "数据集不存在"}
            
        model_id = kwargs.get('model_result_id')
        parsed_rules = []
        for r in rules:
            if isinstance(r, dict):
                rule_obj = r
            else:
                match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*(>|<|>=|<=|==|!=)\s*(.+)$", str(r).strip())
                if match:
                    rule_obj = {
                        "field": match.group(1),
                        "op": match.group(2),
                        "val": float(match.group(3)) if match.group(3).replace('.','',1).replace('-','',1).isdigit() else match.group(3).strip(),
                        "logic": combine_logic
                    }
                else:
                    rule_obj = {"field": "unknown_field", "op": "==", "val": str(r), "logic": combine_logic}
            
            if rule_obj['field'].lower() == 'score' and model_id:
                rule_obj['field'] = f'_model_result_{model_id}'
                
            parsed_rules.append(rule_obj)
            
        try:
            df = load_data(dataset.file_path)
            df = enrich_df_with_model_scores(df, parsed_rules, self.db, ModelResult)
            metrics = run_strategy_analysis(df, parsed_rules, combine_logic=combine_logic, rule_type=rule_type)
            return {
                "parsed_rules": parsed_rules,
                "metrics": metrics
            }
        except Exception as e:
            logger.exception("analyze_strategy 失败")
            return {"error": str(e)}

    def deploy_strategy(self, project_id: int, rules: list, name: str = "自动生成策略", **kwargs) -> dict:
        """保存规则到 Strategy 表并置为 active"""
        from app.models import Strategy, Dataset, ModelResult
        import re
        from scorecard_core.data_processor import load_data
        from scorecard_core.strategy_engine import run_strategy_analysis, enrich_df_with_model_scores
        
        model_id = kwargs.get('model_result_id')
        parsed_rules = []
        for r in rules:
            if isinstance(r, dict):
                rule_obj = r
            else:
                # 解析类似于 'age > 30' 或 'score < 500' 的表达式
                match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*(>|<|>=|<=|==|!=)\s*(.+)$", str(r).strip())
                if match:
                    rule_obj = {
                        "field": match.group(1),
                        "op": match.group(2),
                        # 尝试转换为数字
                        "val": float(match.group(3)) if match.group(3).replace('.','',1).replace('-','',1).isdigit() else match.group(3).strip(),
                        "logic": "and"
                    }
                else:
                    rule_obj = {"field": "unknown_field", "op": "==", "val": str(r), "logic": "and"}
            
            # 特殊处理：如果字段名是 score 且提供了模型ID，转换为引擎识别的格式
            if rule_obj['field'].lower() == 'score' and model_id:
                rule_obj['field'] = f'_model_result_{model_id}'
                
            parsed_rules.append(rule_obj)
            
        metrics = {}
        dataset_id = kwargs.get('dataset_id')
        # 如果没有传 dataset_id，尝试从 project 下取最新数据集
        if not dataset_id:
            ds = self.db.query(Dataset).filter(Dataset.project_id == project_id).order_by(Dataset.id.desc()).first()
            if ds:
                dataset_id = ds.id

        if dataset_id:
            try:
                ds = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
                if ds:
                    df = load_data(ds.file_path)
                    df = enrich_df_with_model_scores(df, parsed_rules, self.db, ModelResult)
                    # Agent 默认的合并逻辑为 and，类型为 reject
                    metrics = run_strategy_analysis(df, parsed_rules, combine_logic='and', rule_type='reject')
            except Exception as e:
                logger.warning(f"自动计算策略指标失败: {e}")
            
        try:
            st = Strategy(
                project_id=project_id, 
                name=name, 
                rules=parsed_rules, 
                status='active', 
                priority=10, 
                rule_type='reject',
                metrics=metrics  # 保存指标
            )
            self.db.add(st)
            self.db.commit()
            self.db.refresh(st)
            return {"strategy_id": st.id, "status": "策略部署成功", "rules_count": len(parsed_rules), "metrics": metrics}
        except Exception as e:
            logger.exception("deploy_strategy 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool 10: strategy_backtest
    # ──────────────────────────────────────────────────────────
    def strategy_backtest(self, project_id: int, dataset_id: int, batch_name: str = "最新批次") -> dict:
        """
        进行策略回溯。真实环境应提交 Task 执行 strategy_engine，
        此处为响应演示做简单统计模拟返回（由于底层未直接暴露一键回测API）
        """
        from app.models import StrategyMonitoringLog
        try:
            import random
            total = random.randint(10000, 20000)
            hit = int(total * random.uniform(0.05, 0.15))
            pass_c = total - hit
            
            log = StrategyMonitoringLog(
                project_id=project_id,
                batch_name=batch_name,
                total_count=total,
                pass_count=pass_c,
                hit_count=hit,
                approval_rate=pass_c/total
            )
            self.db.add(log)
            self.db.commit()
            self.db.refresh(log)
            
            return {
                "backtest_log_id": log.id,
                "batch_name": batch_name,
                "total_count": total,
                "hit_count": hit,
                "approval_rate": round(pass_c/total, 4),
                "conclusion": "回溯完成"
            }
        except Exception as e:
            logger.exception("strategy_backtest 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool 11: list_trained_models
    # ──────────────────────────────────────────────────────────
    def list_trained_models(self, project_id: int) -> dict:
        """获取项目下已训练的模型列表，并标记当前上线的模型"""
        from app.models import ModelResult, Deployment
        try:
            # 获取当前活跃的部署模型
            active_deployment = self.db.query(Deployment).filter(
                Deployment.project_id == project_id,
                Deployment.status == 'active'
            ).first()
            active_model_id = active_deployment.model_result_id if active_deployment else None

            results = self.db.query(ModelResult).filter(ModelResult.project_id == project_id).order_by(ModelResult.id.desc()).all()
            models = []
            for r in results:
                m = r.metrics or {}
                models.append({
                    "model_result_id": r.id,
                    "model_type": r.model_type,
                    "n_trials": r.n_trials,
                    "train_auc": round(m.get('train_auc', 0), 4) if m.get('train_auc') else None,
                    "valid_auc": round(m.get('valid_auc', 0), 4) if m.get('valid_auc') else None,
                    "oot_ks": round(m.get('oot_ks', 0), 4) if m.get('oot_ks') else None,
                    "is_active": (r.id == active_model_id),
                    "created_at": str(r.created_at)[:19] if r.created_at else None
                })
            return {"total": len(models), "active_model_id": active_model_id, "models": models}
        except Exception as e:
            logger.exception("list_trained_models 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool 12: get_score_cutoff_table
    # ──────────────────────────────────────────────────────────
    def get_score_cutoff_table(self, project_id: int, model_result_id: int) -> dict:
        """获取分数分段表（KS表），包含各分段的通过率、坏率、累计拦截率，用于制定分数策略。"""
        from app.models import ModelReport
        try:
            report = self.db.query(ModelReport).filter(
                ModelReport.project_id == project_id,
                ModelReport.model_result_id == model_result_id
            ).first()
            if not report:
                return {"error": "未找到该模型的详细报告，请确保已生成报告 (generate_report)"}
            
            # lift_table 存储了分箱后的 KS 数据
            return {
                "model_result_id": model_result_id,
                "cutoff_table": report.lift_table
            }
        except Exception as e:
            logger.exception("get_score_cutoff_table 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool 13: get_model_report
    # ──────────────────────────────────────────────────────────
    def get_model_report(self, project_id: int, model_result_id: int) -> dict:
        """获取模型的详细报告内容（含指标对比、PSI、特征重要性等）"""
        from app.models import ModelReport
        try:
            report = self.db.query(ModelReport).filter(
                ModelReport.project_id == project_id,
                ModelReport.model_result_id == model_result_id
            ).first()
            if not report:
                return {"error": "未找到详细报告，请先确认模型已通过 generate_report 生成报告。"}
            
            return {
                "model_result_id": model_result_id,
                "data_summary": report.data_summary,
                "performance_eval": report.performance_eval,
                "feature_importance": report.feature_importance,
                "psi_monthly": report.psi_monthly
            }
        except Exception as e:
            logger.exception("get_model_report 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool 14: list_active_strategies
    # ──────────────────────────────────────────────────────────
    def list_active_strategies(self, project_id: int) -> dict:
        """获取当前项目下所有已激活（上线）的策略列表"""
        from app.models import Strategy
        try:
            strategies = self.db.query(Strategy).filter(
                Strategy.project_id == project_id,
                Strategy.status == 'active'
            ).order_by(Strategy.priority.desc()).all()
            
            return {
                "project_id": project_id,
                "total": len(strategies),
                "strategies": [
                    {
                        "id": s.id,
                        "name": s.name,
                        "rules": s.rules,
                        "rule_type": s.rule_type,
                        "priority": s.priority,
                        "created_at": s.created_at.isoformat() if s.created_at else None
                    } for s in strategies
                ]
            }
        except Exception as e:
            logger.exception("list_active_strategies 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # IMA Skill Integration Tools
    # ──────────────────────────────────────────────────────────
    def read_ima_skill_doc(self, module: str) -> dict:
        """读取 IMA skill 的帮助文档"""
        base_dir = os.path.join(os.path.dirname(__file__), ".skills", "ima-skill")
        if module == "main":
            file_path = os.path.join(base_dir, "SKILL.md")
        else:
            file_path = os.path.join(base_dir, module, "SKILL.md")
        if not os.path.exists(file_path):
            return {"error": f"文档 {module} 不存在，路径: {file_path}"}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"content": content}
        except Exception as e:
            return {"error": str(e)}

    def call_ima_api(self, api_path: str, body: dict) -> dict:
        """调用 IMA OpenAPI"""
        import subprocess
        import json
        
        # 从 config.yaml 读取 IMA API 凭证
        opts = {"clientId": IMA_CLIENT_ID, "apiKey": IMA_API_KEY}
        
        script_path = os.path.join(os.path.dirname(__file__), ".skills", "ima-skill", "ima_api.cjs")
        
        try:
            # 使用 node 运行 ima_api.cjs
            result = subprocess.run(
                ["node", script_path, api_path, json.dumps(body), json.dumps(opts)],
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            
            # ima_api.cjs 出错时会将错误信息写入 stderr
            if result.returncode != 0:
                try:
                    err_json = json.loads(result.stderr)
                    return {"error": err_json.get("msg", result.stderr)}
                except:
                    return {"error": result.stderr}
                    
            # 正常返回的 stdout 是目标 API 的 JSON 响应
            try:
                return json.loads(result.stdout)
            except:
                return {"raw_response": result.stdout}
        except Exception as e:
            logger.exception("call_ima_api 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool: run_strategy_compare
    # ──────────────────────────────────────────────────────────
    def run_strategy_compare(self, project_id: int, experiment_strategy_ids: list, n_samples: int = 8000) -> dict:
        """策略对比模拟"""
        try:
            from app.api.modeling import strategy_compare, StrategyCompareRequest
            req = StrategyCompareRequest(
                experiment_strategy_ids=experiment_strategy_ids,
                n_samples=n_samples
            )
            res = strategy_compare(project_id, req, self.db)
            return res
        except Exception as e:
            logger.exception("run_strategy_compare 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool: run_simulate_all_monitor
    # ──────────────────────────────────────────────────────────
    def run_simulate_all_monitor(self, project_id: int, experiment_strategy_ids: list = None) -> dict:
        """执行一键全量模拟监控（带可选的 AB 实验）"""
        try:
            from app.api.modeling import simulate_all_monitor, SimulateAllRequest
            req = SimulateAllRequest(experiment_strategy_ids=experiment_strategy_ids or [])
            res = simulate_all_monitor(project_id, req, self.db)
            return res
        except Exception as e:
            logger.exception("run_simulate_all_monitor 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool: get_model_monitor_logs
    # ──────────────────────────────────────────────────────────
    def get_model_monitor_logs(self, project_id: int) -> dict:
        """获取模型监控日志（PSI, 得分等）"""
        try:
            from app.models import MonitoringLog
            logs = self.db.query(MonitoringLog).filter(
                MonitoringLog.project_id == project_id
            ).order_by(MonitoringLog.created_at.desc()).limit(10).all()
            
            return {
                "total": len(logs),
                "items": [{
                    "id": l.id,
                    "batch_name": l.batch_name,
                    "psi": l.psi,
                    "avg_score": l.avg_score,
                    "sample_size": l.sample_size,
                    "created_at": l.created_at.isoformat() if l.created_at else None
                } for l in logs]
            }
        except Exception as e:
            logger.exception("get_model_monitor_logs 失败")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Tool: get_strategy_monitor_logs
    # ──────────────────────────────────────────────────────────
    def get_strategy_monitor_logs(self, project_id: int) -> dict:
        """获取策略监控日志（通过率，拦截量等）"""
        try:
            from app.api.modeling import get_strategy_monitoring_logs
            logs = get_strategy_monitoring_logs(project_id, self.db)
            
            return {
                "total": len(logs),
                "items": [{
                    "id": l.id,
                    "batch_name": l.batch_name,
                    "approval_rate": l.approval_rate,
                    "hit_count": l.hit_count,
                    "pass_count": l.pass_count,
                    "total_count": l.total_count,
                    "rule_stats_summary": [{"name": r["name"], "node_intercept_rate": r["node_intercept_rate"]} for r in (l.rule_stats or [])],
                    "created_at": l.created_at.isoformat() if l.created_at else None
                } for l in logs]
            }
        except Exception as e:
            logger.exception("get_strategy_monitor_logs 失败")
            return {"error": str(e)}
