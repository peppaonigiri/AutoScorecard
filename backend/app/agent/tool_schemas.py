# -*- coding: utf-8 -*-
"""
Tool Schemas - 第一阶段建模 Agent 工具定义

供 DeepSeek/OpenAI Function Calling 使用的 JSON Schema。
第一阶段（建模）：6 个工具，项目和数据集已由用户预先准备好。

预留第二阶段（报告评价+模型优化）工具，暂不开放，后续添加。
"""

# ============================================================
# 第一阶段：建模 Pipeline 工具
# ============================================================

PHASE1_TOOLS = [
    # ── 数据探索工具（建模前置步骤）──────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_data_overview",
            "description": (
                "获取数据集的基本概览，包括：前5行数据、各列数据类型、描述性统计（均值/标准差/分位数等）"
                "以及各列缺失率。"
                "【建模前必须调用此工具】：用于让用户确认哪些列需要排除（如 ID 列、时间列、泄露列），"
                "以及了解数据基本情况。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "integer",
                        "description": "建模项目 ID"
                    },
                    "dataset_id": {
                        "type": "integer",
                        "description": "数据集 ID"
                    }
                },
                "required": ["project_id", "dataset_id"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "run_iv_report",
            "description": (
                "计算指定数据集所有特征的 IV（信息价值）和 PSI（稳定性指数）。"
                "返回每个特征的统计摘要，LLM 需根据此结果决定后续筛选阈值。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "integer",
                        "description": "建模项目 ID"
                    },
                    "dataset_id": {
                        "type": "integer",
                        "description": "要分析的数据集 ID"
                    },
                    "dep": {
                        "type": "string",
                        "description": "标签列名，通常为 'label' 或 'bad_flag'"
                    },
                    "split_ratios": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "训练/验证/测试比例，如 [0.6, 0.2, 0.2]。默认 [0.6, 0.2, 0.2]"
                    },
                    "exclude_cols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "需要排除的列名（如 ID 列、时间列）。默认 []"
                    }
                },
                "required": ["project_id", "dataset_id", "dep"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_features",
            "description": (
                "基于 IV、PSI、相关性等阈值执行 L2 变量筛选，返回最终入模特征列表。"
                "筛选结果会自动持久化到项目中，后续建模直接使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "integer",
                        "description": "建模项目 ID"
                    },
                    "dataset_id": {
                        "type": "integer",
                        "description": "数据集 ID"
                    },
                    "dep": {
                        "type": "string",
                        "description": "标签列名"
                    },
                    "iv_threshold": {
                        "type": "number",
                        "description": "IV 最低阈值（低于此值的特征被剔除）。建议 0.02，特征多时可提高到 0.03"
                    },
                    "psi_threshold": {
                        "type": "number",
                        "description": "PSI 最高阈值（高于此值的特征被剔除，表示不稳定）。建议 0.1"
                    },
                    "corr_threshold": {
                        "type": "number",
                        "description": "相关性阈值（高相关特征对中保留 IV 更高的）。建议 0.9"
                    },
                    "exclude_cols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "排除列，同 iv_report 保持一致"
                    }
                },
                "required": ["project_id", "dataset_id", "dep"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_training",
            "description": (
                "启动 Optuna 自动调参建模任务（异步），立即返回 task_id。"
                "建模特征列表使用上一步 filter_features 保存的结果。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "integer",
                        "description": "建模项目 ID"
                    },
                    "dataset_id": {
                        "type": "integer",
                        "description": "数据集 ID"
                    },
                    "dep": {
                        "type": "string",
                        "description": "标签列名"
                    },
                    "model_type": {
                        "type": "string",
                        "enum": ["xgb", "lgb", "lr", "rf"],
                        "description": "模型类型：xgb=XGBoost, lgb=LightGBM, lr=逻辑回归, rf=随机森林"
                    },
                    "n_trials": {
                        "type": "integer",
                        "description": "Optuna 搜索次数，建议 20-100。次数越多精度越高但耗时越长"
                    },
                    "exclude_cols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "排除列"
                    }
                },
                "required": ["project_id", "dataset_id", "dep", "model_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "poll_task_until_done",
            "description": (
                "持续轮询异步任务状态，直到任务完成或失败。"
                "轮询期间自动维持心跳，防止任务因心跳超时被系统终止。"
                "返回最终状态、model_result_id 和模型指标。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "由 start_training 或 generate_report 返回的任务 ID"
                    }
                },
                "required": ["task_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "触发模型报告生成任务（异步）。"
                "报告包含 KS/AUC 曲线、特征重要性、分箱分布等，输出为 Excel 文件。"
                "返回 task_id，需再次调用 poll_task_until_done 等待报告生成完成。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "integer",
                        "description": "建模项目 ID"
                    },
                    "model_result_id": {
                        "type": "integer",
                        "description": "由 poll_task_until_done 返回的 model_result_id"
                    }
                },
                "required": ["project_id", "model_result_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_metrics",
            "description": (
                "获取指定模型的完整评估指标（AUC、KS、各数据集表现等）和报告文件路径。"
                "作为最后一步调用，用于生成建模总结。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "integer",
                        "description": "建模项目 ID"
                    },
                    "model_result_id": {
                        "type": "integer",
                        "description": "模型结果 ID"
                    }
                },
                "required": ["project_id", "model_result_id"]
            }
        }
    },
]

# ============================================================
# 第二阶段（预留）：报告评价 + 模型优化工具
# ============================================================

PHASE2_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "auto_strategy_mining",
            "description": "基于决策树自动从数据中挖掘出可用的风控规则策略，返回建议的规则列表。可以在模型训练之后调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "建模项目 ID"},
                    "dataset_id": {"type": "integer", "description": "数据集 ID"},
                    "label_col": {"type": "string", "description": "标签列，通常为 label 或 bad_flag"},
                    "tree_type": {"type": "string", "description": "树算法类型", "enum": ["exrf", "rf", "dt"], "default": "exrf"}
                },
                "required": ["project_id", "dataset_id", "label_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deploy_model",
            "description": "将选定的最优模型标记为上线部署状态，生成部署记录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "model_result_id": {"type": "integer", "description": "需要上线的模型结果 ID"}
                },
                "required": ["project_id", "model_result_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deploy_strategy",
            "description": "将挖掘出的规则或分数阈值合并为一个风控策略并上线。支持分数策略：若 field 为 'score' 且传入了 model_result_id，系统会自动关联该模型的得分。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "name": {"type": "string", "description": "策略名称"},
                    "rules": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "规则列表，如 ['age > 30', 'score < 450']"
                    },
                    "model_result_id": {"type": "integer", "description": "可选。如果要制定分数策略，请提供对应的模型 ID"}
                },
                "required": ["project_id", "rules"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "strategy_backtest",
            "description": "策略/模型上线后，对指定的数据集进行回溯（Backtest），评估上线后的拦截率、坏率、PSI 等监控指标。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "dataset_id": {"type": "integer", "description": "用于回溯的数据集 ID"},
                    "batch_name": {"type": "string", "description": "回溯批次名称"}
                },
                "required": ["project_id", "dataset_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_trained_models",
            "description": "获取当前项目下所有已训练完成的模型列表（包含模型ID、类型、AUC等核心指标），用于选择模型进行上线或策略挖掘。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"}
                },
                "required": ["project_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_score_cutoff_table",
            "description": "获取模型的分数分段评估表（KS表），包含每个分段的坏率、样本占比、累计拦截率等指标。用于制定基于分数的准入策略（如：拦截分数 < 500 的人群）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "model_result_id": {"type": "integer"}
                },
                "required": ["project_id", "model_result_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_report",
            "description": "获取指定模型的详细评估报告。包含数据概要、训练/验证/测试集的 AUC/KS 对比、特征重要性排行以及月度 PSI 指标。用于深度分析模型质量。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "model_result_id": {"type": "integer"}
                },
                "required": ["project_id", "model_result_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_active_strategies",
            "description": "获取当前项目下所有已激活（上线）的策略列表，包含策略名称、规则内容、优先级等。用于查看当前的决策逻辑。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"}
                },
                "required": ["project_id"]
            }
        }
    }
]

# 当前激活的工具集
ACTIVE_TOOLS = PHASE1_TOOLS + PHASE2_TOOLS
