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
            "name": "handle_missing_values",
            "description": (
                "对数据集的数值列进行缺失值填充，生成一个新的已填充数据集快照，并返回新的 dataset_id。"
                "【调用时机】：在 get_data_overview 发现缺失率较高（如 > 5%）时，建议在 run_iv_report 之前调用。"
                "填充后必须使用返回的 new_dataset_id 替换原 dataset_id 进行后续建模操作。"
                "常用填充值：-999（表示缺失标记，适合树模型）、0（适合已知零值场景）。"
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
                        "description": "待填充的数据集 ID"
                    },
                    "fill_value": {
                        "type": "number",
                        "description": "缺失值填充数值，默认 -999。树模型推荐 -999，线性模型可考虑均值（需手动指定）"
                    },
                    "exclude_cols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "不参与填充的列名（如标签列、ID 列）。默认 []"
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
            "name": "analyze_strategy",
            "description": (
                "进行策略规则的单变量或多变量分析与测试，在实际部署前评估规则的拦截率、通过率、坏率等表现。"
                "【重要】rules 列表中每个元素必须是单条规则（如 'age < 20'），不要将多条规则用 and/or 拼成一个字符串。"
                "多条规则之间的逻辑关系通过 combine_logic 参数控制。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "dataset_id": {"type": "integer", "description": "用于测试规则的数据集 ID"},
                    "rules": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "规则列表，每条为单一条件，如 ['FT2_0697_m12_to_m24 <= -995.5873', 'FT2_0656_m24 <= -227.6464']"
                    },
                    "combine_logic": {"type": "string", "enum": ["and", "or"], "default": "and", "description": "多条规则的合并逻辑，默认 and"},
                    "rule_type": {"type": "string", "enum": ["reject", "pass", "review"], "default": "reject"},
                    "model_result_id": {"type": "integer", "description": "可选。如果测试分数策略（如 score < 450），需传入对应模型 ID"}
                },
                "required": ["project_id", "dataset_id", "rules"]
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
            "description": (
                "将挖掘出的规则或分数阈值合并为一个风控策略并上线。"
                "【重要】rules 列表中每个元素必须是单条规则（如 'age > 30'），不要将多条规则用 and/or 拼成一个字符串。"
                "多条规则之间的逻辑关系通过 combine_logic 参数控制（默认 and）。"
                "支持分数策略：若 field 为 'score' 且传入了 model_result_id，系统会自动关联该模型的得分。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "name": {"type": "string", "description": "策略名称"},
                    "rules": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "规则列表，每条为单一条件，如 ['FT2_0697_m12_to_m24 <= -995.5873', 'FT2_0656_m24 <= -227.6464']"
                    },
                    "combine_logic": {
                        "type": "string",
                        "enum": ["and", "or"],
                        "description": "多条规则的合并逻辑，默认 and（所有条件同时满足才拦截）"
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
    },
    {
        "type": "function",
        "function": {
            "name": "read_ima_skill_doc",
            "description": "读取 IMA 知识库/笔记技能的使用说明文档。如果用户要求操作知识库、笔记、存入文档等，请先调用此工具了解 API 用法。module 可填 'main', 'notes', 或 'knowledge-base'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "module": {"type": "string", "enum": ["main", "notes", "knowledge-base"]}
                },
                "required": ["module"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_ima_api",
            "description": "调用 IMA OpenAPI 进行知识库/笔记操作。使用前必须先阅读对应的 SKILL.md 文档了解 api_path 和 body 的具体结构。",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_path": {"type": "string", "description": "API 路径，例如 'openapi/list_docs'"},
                    "body": {"type": "object", "description": "请求体 JSON 对象"}
                },
                "required": ["api_path", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_strategy_compare",
            "description": "策略对比模拟：在同一批模拟进件数据上，分别跑当前上线策略（基准组A）和指定的实验策略（实验组B），对比通过率、拦截率等指标差异。用于回答'如果换成策略X会怎样'这类问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "项目 ID"},
                    "experiment_strategy_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "实验组策略 ID 列表（可从 list_all_strategies 获取）"
                    },
                    "n_samples": {"type": "integer", "description": "模拟样本量，默认 8000"}
                },
                "required": ["project_id", "experiment_strategy_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_simulate_all_monitor",
            "description": "执行一键全量模拟监控任务（触发后端 simulate_all 接口）。此工具将基于历史数据随机偏移生成一批进件数据，并执行当前上线模型及上线策略的打分与拦截，还可附带指定实验策略进行对比实验。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "项目 ID"},
                    "experiment_strategy_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "可选的实验组策略 ID 列表，用于顺带进行 AB 实验对比"
                    }
                },
                "required": ["project_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_monitor_logs",
            "description": "获取模型监控日志列表。用于查看各批次模拟数据的模型 PSI（稳定性）、平均得分、得分分布等。PSI > 0.1 表示需要注意，> 0.25 表示模型已衰退危险。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "项目 ID"}
                },
                "required": ["project_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_strategy_monitor_logs",
            "description": "获取策略流监控日志列表。用于查看各批次模拟数据经过线上策略引擎后的表现，包括：整体通过率、拦截量、各独立规则的拦截率与通过率等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "项目 ID"}
                },
                "required": ["project_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_swap_analysis",
            "description": (
                "策略置换分析（Swap In/Out）：对比新旧两套策略的效果差异。\n"
                "置入（Swap In）：旧策略拒绝+新策略通过的客群，即新策略捞回的客群。\n"
                "置出（Swap Out）：旧策略通过+新策略拒绝的客群，即新策略新增拦截的客群。\n"
                "输出 2x2 决策矩阵、置入客群估算坏率（拒绝推断法）、置出客群真实坏率、通过率和逾期率对比。\n"
                "【分数策略用法】：若新/旧策略是基于模型分数的（如'拦截538分以下'），\n"
                "  需将 new_col 设为 '_model_result_{model_result_id}'（如 '_model_result_26'），\n"
                "  同时传入 model_result_id，后端会自动对数据集进行实时打分并填充该列。\n"
                "【推荐用法】：若已知策略 ID，优先传入 old_strategy_id / new_strategy_id，\n"
                "  后端会自动从策略库读取 rules（含 model_result_id 关联），无需手动指定规则字段。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "项目 ID"},
                    "dataset_id": {"type": "integer", "description": "数据集 ID，需同时包含新旧策略字段和逾期标签"},
                    "label_col": {"type": "string", "description": "逾期标签列名，旧策略拒绝客户该列为 NaN"},
                    "old_strategy_id": {
                        "type": "integer",
                        "description": "【推荐】旧策略的 strategy_id（从 list_project_strategies 获取）。传入后无需再指定 old_col/old_reject_op/old_reject_val"
                    },
                    "new_strategy_id": {
                        "type": "integer",
                        "description": "【推荐】新策略的 strategy_id（从 list_project_strategies 获取）。传入后无需再指定 new_col/new_reject_op/new_reject_val"
                    },
                    "model_result_id": {
                        "type": "integer",
                        "description": "【分数策略必填】当新策略或旧策略基于模型分数时（field='score' 或 field='_model_result_X'），传入对应的模型 ID。后端将自动对数据集打分并填充分数列。"
                    },
                    "old_col": {"type": "string", "description": "旧策略字段名。若旧策略为分数策略，填 '_model_result_{model_result_id}'，如 '_model_result_26'"},
                    "old_reject_op": {"type": "string", "enum": [">=", ">", "<=", "<", "==", "in"], "description": "旧策略拒绝操作符"},
                    "old_reject_val": {"description": "旧策略拒绝阈值，数值或逗号分隔字符串如 'D,E'"},
                    "new_col": {"type": "string", "description": "新策略字段名。若新策略为分数策略，填 '_model_result_{model_result_id}'，如 '_model_result_26'"},
                    "new_reject_op": {"type": "string", "enum": [">=", ">", "<=", "<", "==", "in"], "description": "新策略拒绝操作符"},
                    "new_reject_val": {"description": "新策略拒绝阈值（如 538）"},
                    "new_col_bins": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "新策略分箱节点，用于拒绝推断法估算置入客群坏率。不传则自动等频分箱"
                    }
                },
                "required": ["project_id", "dataset_id", "label_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_project_strategies",
            "description": "查看项目下历史保存的所有策略方案（包括草稿状态和已上线状态、优先级、规则列表以及历史预估指标等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "项目 ID"}
                },
                "required": ["project_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_strategy_status",
            "description": "操作特定策略的上线或下架。上线状态设为 'active'，下架/草稿状态设为 'draft'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "项目 ID"},
                    "strategy_id": {"type": "integer", "description": "策略 ID"},
                    "status": {
                        "type": "string",
                        "enum": ["active", "draft"],
                        "description": "目标状态：'active' 表示上线，'draft' 表示下架（重置为草稿）"
                    }
                },
                "required": ["project_id", "strategy_id", "status"]
            }
        }
    }
]

# 当前激活的工具集
ACTIVE_TOOLS = PHASE1_TOOLS + PHASE2_TOOLS
