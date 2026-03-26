# AutoModeling Platform (自动化评分卡建模平台)

🚀 **AutoModeling Platform** 是一款面向风控金融场景的、端到端自动化评分卡建模与策略分析平台。它整合了传统评分卡理论与现代机器学习技术，旨在提供从原始数据清洗、变量筛选、Optuna 模型调优到策略回测与线上监测的全生命周期建模工具。

---

## ✨ 核心亮点 (Key Highlights)

-   **一致性评分引擎**：系统内置统一的 Proba-to-Score 映射模型，确保从训练、模拟到线上监测的各环境下，分值计算逻辑完全对齐，消除分值偏差风险。
-   **系统级资源安全**：内置**任务心跳自毁机制**。当浏览器刷新或断开连接时，后台高消耗任务（如寻参调优、报表生成）会在 60s 内自动终止并释放 CPU 资源，有效防止服务器空转。
-   **无头绘图优化**：采用 Matplotlib `Agg` 非交互式后端，支持在无桌面环境服务器中稳定输出 Excel 报表与可视化图表，极大提升了生产环境下的绘图可靠性。

---

## 🛠️ 环境准备与安装 (Setup)

### 1. 后端环境 (Backend Setup)
-   **Python 推荐**: Python 3.8 或更高版本（推荐使用 Conda 虚拟环境 `p_3_8_fb`）。
-   **安装依赖**:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```
-   **初始化**: 根目录下的 `config.yaml` 存储核心配置（如数据库、心跳开关）。首次运行前请执行 `init_scorecard.sql` 初始化表结构。

### 2. 前端环境 (Frontend Setup)
-   **Node.js**: 建议使用 LTR 版本（16.x 或更高）。
-   **安装依赖**:
    ```bash
    cd frontend
    npm install
    ```

---

## 🚀 启动与运行 (Running)

-   **一键联测**: 双击点击根目录下的 `start_project.bat` 脚本（Windows 环境专用）。
-   **手动分块启动**:
    -   **API 后端**: `cd backend && python app/main.py` (默认端口 8081)。
    -   **UI 前端**: `cd frontend && npm run dev` (Vite 调试模式)。

---

## 📖 核心业务指南 (Case Flow)

1.  **数据资产上传**：上传 CSV 数据集。系统自动执行数据清洗与 **L1 基础分析**（去除缺失率/方差不足特征）。
2.  **多集变量分析**：灵活定义划分比例，对比训练集/验证集与测试集 (OOT) 的 IV、PSI 稳定性指标。
3.  **变量筛选 L2**：基于 IV、相关性、PSI 阈值执行自动变量剔除，锁定入模最佳特征池。
4.  **智能建模调优**：集成 **Optuna** 深度寻优算法，基于贝叶斯策略自动调参，并一键完成评分卡转换（支持 PDO/基准分调整）。
5.  **专业模型报告**：后台渲染 KS/AUC 曲线图与分箱分布图，支持导出内嵌特征分析详情的 Excel 企业级报告。
6.  **策略挖掘回测**：基于决策树自动挖掘拦截规则，通过历史样本回测评估“通过率 vs 坏账率”的平衡。
7.  **联机模拟监测**：基于 Bootstrap 生成模拟流量，动态监控指标漂移情况，实现线上稳定性实时预警。

---

## ⚙️ 全局配置项 (`config.yaml`)

| 配置模块 | 变量名 | 注解 |
| :--- | :--- | :--- |
| **database** | `password` | 数据库密码。为空时自动切换至本地 SQLite 文件驱动模式。 |
| **server** | `heartbeat_enabled` | **心跳开关**。控制关闭网页是否自动终断耗时后台任务。 |
| **server** | `heartbeat_timeout` | **超时阔值**。心跳续约最大时间（建议 60-120 秒）。 |
| **modeling** | `n_trials` | Optuna 进行参数寻优尝试的次数，值越大结果精度越高。 |

---

## 🏗️ 1. 层次化系统架构图 (System Architecture)

该图展示了前后端物理隔离下的功能划分，特别强调了 `TaskManager` 的后台防护机制，以及核心算法引擎（`scorecard_core`）承担的职责流转。

```mermaid
graph TB
    subgraph Frontend [前端 UI 层 (React + AntD + Vite)]
        UI_Auth[权限与工作台模块]
        UI_Data[数据资产与初筛可视化]
        UI_Feat[特征分箱与PSI/IV大盘]
        UI_Model[模型调优设置与进度监控]
        UI_Report[模型报告下载与可视化评估]
        UI_Strategy[策略编排挖掘与回测推演]
        UI_Monitor[上线部署与分布偏移监测]
    end

    subgraph Backend_API [后端网关暴露层 (FastAPI)]
        API_Auth([Auth.py])
        API_Project([Project.py])
        API_Dataset([Dataset.py])
        API_Feature([Feature.py])
        API_Modeling([Modeling.py])
        API_Strategy([Strategy.py])
    end

    subgraph Backend_Task [异步调度与保障层 (TaskManager)]
        Task_Queue[SQLite/PG 状态调度]
        Heartbeat[任务心跳保护与进程树销毁]
    end

    subgraph Backend_Core [核心算法引擎 (scorecard_core)]
        Core_Data[[data_processor: 异常值与同值率初筛]]
        Core_Feat[[feature_engineer: WOE分箱与PSI剔除]]
        Core_Train[[model_trainer: Optuna寻参转评分卡]]
        Core_Report[[report: 评估曲线与报告生成]]
        Core_Mining[[strategy_mining: 决策树基发掘]]
        Core_Engine[[strategy_engine: 规则推演拦截库]]
        Core_Monitor[[monitor_engine: 打分分布偏移估算]]
    end

    subgraph Database [持久层 (PostgreSQL / SQLite)]
        DB_Users[(Users)]
        DB_Projects[(Projects/Datasets)]
        DB_Tasks[(Tasks / ModelResults)]
        DB_Deploy[(Deployments/Logs)]
        DB_Strategy[(Strategies)]
    end

    subgraph Storage [持久文件 (Storage)]
        S_CSV(样本 CSV)
        S_BIN(Pickle 模型)
        S_EXCEL(离线报告图表)
    end

    Frontend == 携带 JWT ===> Backend_API
    Backend_API --> Task_Queue
    Backend_API --> Core_Data
    Backend_API --> Core_Feat
    API_Strategy --> Core_Mining
    API_Strategy --> Core_Engine
    Task_Queue -. 唤起子进程 .-> Core_Train
    Heartbeat -. 中断拦截 .-> Core_Train
    Core_Train --> Core_Report
    Backend_API <--> Database
    Core_Data --> S_CSV
    Core_Train --> S_BIN
    Core_Report --> S_EXCEL
    Core_Data ..> DB_Projects
    Core_Train ..> DB_Tasks
    Core_Monitor ..> DB_Deploy
    Core_Engine ..> DB_Strategy
```

---

## 🔄 2. 核心业务流程与时序交互图 (User Flow & Interactions)

该图重点演示了“特征挖掘 - 调参算力防线 - 策略验证”的核心生命周期进度，包含后台心跳自毁防联断机制：

```mermaid
sequenceDiagram
    autonumber
    actor U as 风控建模人员 (User)
    participant F as 前端界面 (React)
    participant B as 后端网关 (FastAPI)
    participant C as 核心算法 (scorecard_core)
    participant TM as 任务管理引擎 (TaskManager)
    participant DB as 数据库 (PostgreSQL)

    %% 阶段1
    rect rgb(230, 245, 255)
    Note right of U: 一、数据资产接入与L1初步处理
    U->>F: 上传CSV进件数据集文件（对照文档等）
    F->>B: POST /api/datasets/upload (多模表单流)
    B->>C: data_processor.process_l1()
    C-->>B: 解析Type、同值率和缺失筛选
    B->>DB: 写入 Datasets 的 meta信息与统计缓存
    B-->>F: 返回上传进度、表头映射与初筛报告
    end

    %% 阶段2
    rect rgb(240, 255, 240)
    Note right of U: 二、特征工程与L2深度筛选 (人工+自动)
    U->>F: 圈定目标集与OOT集，设置 IV 与 PSI 剔除阈值
    F->>B: POST /api/feature/select
    B->>C: feature_engineer.binning_and_filter()
    C-->>B: 执行WOE分箱映射、相关性矩阵计算与剔除
    B->>DB: 更新 Project.feature_list 固定入模特征阵列
    B-->>F: 回传特征看板进行预览评估
    end

    %% 阶段3
    rect rgb(255, 245, 230)
    Note right of U: 三、后台并发建模与【心跳保活销毁机制】
    U->>F: 触发 Optuna 模型训练(设最大尝试与PDO)
    F->>B: POST /api/modeling/submit_optuna
    B->>DB: 新增 Tasks 记录 (Status=pending)
    B->>TM: 发起异步进程分离 task_manager.run_task(task_id)
    TM->>C: Python 子进程拉起 model_trainer.run_optuna_training()
    
    par [前端心跳轮询保活]
        loop 2s 轮询监控
            F->>B: POST /api/modeling/heartbeat
            B->>DB: 刷新 Task.last_heartbeat (延续寿命)
            B-->>F: 返回当前进度 progress%
        end
    and [TaskManager 猎杀者判定]
        loop 10s 死循环检测
            TM->>DB: 检查超时情况 (now() - last_heartbeat) 
            opt 发现断联断网 (差值 > 60秒 取消任务)
                TM-->>TM: 触发强杀机制 os.kill() 或 SIGTERM
                TM->>DB: 更新 Task 状态为 failed (超时截杀)
            end
        end
    end
    
    C-->>TM: 训练完毕导出参数，并将逻辑回归折现成标准风控制表
    TM->>DB: 特征贡献/AUC写入 ModelResults 与 score_distribution
    TM->>DB: 更新 Tasks 状态为 completed
    end

    %% 阶段4
    rect rgb(245, 235, 255)
    Note right of U: 四、指标报告生成与策略发掘模拟
    U->>F: 审查在线曲线或提出验证规则
    F->>B: GET /api/modeling/{id}/report
    B->>C: report.py 生成图表序列及本地化 Excel 写入
    B-->>F: 在线呈现 KS/AUC / Lift 数据视面
    
    U->>F: "设定实验规则" 用于进件拦截测算
    F->>B: POST /api/strategy/backtest
    B->>C: strategy_engine.vectorized_match() (底层矢量急速排查)
    C-->>B: 获取现有客群的历史审批率(Approval Rate)与业务提升度(Lift)
    B->>DB: 回放指标写入 Strategy Monitoring Log
    B-->>F: 渲染展示通过率、坏账率分布
    end
```

---

## 🗄️ 3. 稳态数据库实体关系图 (ER Diagram)

此图体现了各个组件产生的持久化资产结构体系与级联销毁关系网：

```mermaid
erDiagram
    Users {
        int id PK
        string username "账户名称"
        string hashed_password "令牌"
        int is_admin "管理级标签"
    }

    Projects {
        int id PK
        int owner_id FK
        string name "空间节点名"
        string status "执行位态"
        json feature_list "有效特征保留底表"
        json split_config "OOT切分策略配置"
    }

    Datasets {
        int id PK
        int project_id FK
        string file_path "物理存储路径"
        json l1_results "异常及同值过滤字典"
        json stats_cache "描述性统计总览"
    }

    Tasks {
        int id PK
        int project_id FK
        string task_type "特征过滤与Optuna特征"
        string status "存活周期(pending/running)"
        datetime last_heartbeat "心跳信号保活戳"
    }

    ModelResults {
        int id PK
        int project_id FK
        int task_id FK "绑定的异步产出流"
        string model_path "Pickle文件持久化指针"
        json params "模型反解超参最优阵列"
        json metrics "入参及AUC评估分数"
        json score_distribution "风控直方分布图"
    }

    Strategy {
        int id PK
        int project_id FK
        string rule_type "通过阈/反欺诈拦截阈"
        json rules "向量推理条件规则集"
        json metrics "拦截率及通过成效指标"
    }

    StrategyMonitoringLog {
        int id PK
        int project_id FK
        string batch_name "监控时段次"
        json rule_stats "触发热点详情"
    }

    ModelReports {
        int id PK
        int model_result_id FK
        json performance_eval "报告前端切片映射"
    }

    Users ||--o{ Projects : "所有者(级联删除)"
    Projects ||--o{ Datasets : "输入资源"
    Projects ||--o{ Tasks : "下推训练流"
    Projects ||--o{ ModelResults : "容纳调优模型"
    Projects ||--o{ Strategy : "配置阻断器"
    Projects ||--o{ StrategyMonitoringLog: "沉淀监控批次"
    Tasks ||--o| ModelResults : "孵化"
    ModelResults ||--o| ModelReports : "可视化展现"
```

---

祝您建模体验愉快！如果有任何建议，请联系系统管理员或查阅 `/backend/scorecard_core/` 实现。
