# AutoModeling Platform (自动化评分卡建模平台)

🚀 **AutoModeling Platform** 是一款面向风控场景的端到端自动化评分卡解决方案。它集成了数据上传、特征筛选、Optuna 模型调优、策略演习及线上监控等核心功能。

---

## 🛠️ 1. 开发环境集成指南

如果您是刚刚通过 Git Clone 检出项目，请务必执行以下步骤以初始化您的本地环境。

### A. 后端环境 (Backend Setup)
1. **运行环境**: Python 版本建议 **不低于 3.8**（推荐使用您本地的 `3.8` 环境）。
2. **进入目录**: `cd backend`
3. **安装依赖**: 
   *由于 Python 三方库已被 Git 忽略，必须执行依赖安装：*
   ```bash
   pip install -r requirements.txt
   ```
3. **数据库初始化**:
   - 检查根目录 `config.yaml` 的 `database` 配置。
   - 运行项目根目录下的 `init_scorecard.sql` 脚本，创建核心业务表。

### B. 前端环境 (Frontend Setup)
1. **进入目录**: `cd frontend`
2. **一键安装清单**:
   *这也是恢复被忽略包依赖的关键步骤：*
   ```bash
   npm install
   ```

---

## 🚀 2. 快速启动 (Running)

- **一键运行**: 
  在配置好 Python 环境后，您可以直接双击根目录下的 `start_project.bat` 脚本同时调起前、后端。
  
- **手动启动**:
  - **后端**: 在 `backend/` 目录下执行 `python app/main.py`（默认运行在 8081 端口）。
  - **前端**: 在 `frontend/` 目录下执行 `npm run dev`（Vite 调试模式，默认 5173 端口）。

---

## 📂 3. 核心目录结构
- `/backend`: 基于 FastAPI 的逻辑后端及核心算法。
- `/frontend`: 基于 React + Ant Design 的现代化 UI 界面。
- `/storage`: **[本地持久化目录]** 存放训练出的 `.pkl` 模型、`.csv` 原始数据集及生成的报告。
  - *注意：此目录内容受 .gitignore 保护，不参与版本管理。*

---

## 🏗️ 4. 系统交互流程图 (System Architecture)

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户 (Browser)
    participant Front as 前端 (React + AntD)
    participant API as 后端 (FastAPI)
    participant Task as 异步引擎 (TaskManager)
    participant Core as 算法库 (scorecard_core)
    participant DB as 数据库 (PostgreSQL)
    participant Disk as 物理存储 (Storage)

    Note over User, Disk: 场景 1: 登录与权限控制 (Auth)
    User->>Front: 输入账号/密码
    Front->>API: POST /auth/login
    API->>DB: 查询用户信息
    DB-->>API: 返回 HashedPassword
    API->>API: Bcrypt 验证并签发 JWT
    API-->>Front: 返回 Token

    Note over User, Disk: 场景 2: 数据资产管理 (Dataset)
    User->>Front: 上传数据文件 (CSV)
    Front->>API: POST /datasets/upload (带 Token)
    API->>Disk: 写入 /storage/uploads/
    API->>Core: 计算基础统计 & L1 初筛 (缺失率/方差)
    API->>DB: 保存数据元信息与初筛结果
    API-->>Front: 列表展示数据集详情

    Note over User, Disk: 场景 3: 自动化建模流程 (Modeling)
    User->>Front: 设置参数并启动建模
    Front->>API: POST /modeling/submit
    API->>Task: 注册异步任务 (Pending)
    API-->>Front: 返回 task_id
    Front->>API: 轮询查询任务进度
    
    activate Task
    Task->>Core: 调用 run_optuna_training
    Core->>Core: 数据拆分 -> Optuna 调参 -> 评分卡转换
    Core->>Disk: 保存 model.pkl
    Task->>DB: 写入 ModelResult (KS/AUC/特征重要性)
    Task->>DB: 更新 Task 状态 (Completed)
    deactivate Task

    Note over User, Disk: 场景 4: 结果分析与可视化 (Result)
    User->>Front: 查看模型报告
    Front->>API: GET /model_results/{id}
    API->>DB: 读取指标与分箱分布
    API-->>Front: 渲染 ECharts 可视化图表 (KS/AUC/Score Dist)

    Note over User, Disk: 场景 5: 风控策略编排与历史回测 (Strategy & Backtest)
    User->>Front: 1. 可视化编排规则 (如: score < 550 OR multi_loan > 5)
    Front->>API: 2. POST /strategies/save
    API->>DB: 3. 存储规则 JSONB 及其优先级
    
    User->>Front: 4. 选择历史数据集进行“回测分析” (Backtest)
    Front->>API: 5. POST /strategies/backtest {dataset_id, strategy_id}
    API->>Disk: 6. 加载含有真实 Label 的历史样本
    API->>Core: 7. 执行向量化规则匹配 (Vectorized Match)
    Core->>Core: 8. 计算业务指标 (误伤率/捕获率/坏账抵御能力)
    API->>DB: 9. 固化回测报告快照
    API-->>Front: 10. 给回测结果看板, 展示“通过率 vs 坏账率”权衡曲线

    Note over User, Disk: 场景 6: 联机模拟监测与稳定性预警 (Monitor)
    User->>Front: 点击“联合模拟监测” (部署模型 + 激活策略)
    Front->>API: POST /monitor/simulate_all
    API->>Core: 1. 基于 Bootstrap Drift 生成模拟进件
    API->>Core: 2. 预测概率 -> 映射评分 -> 策略流实时拦截
    API->>DB: 3. 记录全量监测报告 (PSI / 拦截强度分布)
    API-->>Front: 4. 多维指标看板, 当 PSI > 0.1 时触发预警状态
```

祝您建模体验愉快！如果有任何问题，请随时在文档反馈。
