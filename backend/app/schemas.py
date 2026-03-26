# -*- coding: utf-8 -*-
"""Pydantic 请求/响应模型"""

from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime


# ==========================================
# User 相关设计
# ==========================================

class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    is_admin: int
    is_active: int
    created_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class TokenData(BaseModel):
    username: Optional[str] = None

class PasswordChange(BaseModel):
    old_password: str
    new_password: str

class RoleChange(BaseModel):
    is_admin: int

class AdminPasswordReset(BaseModel):
    new_password: str

# ==========================================
# Project 相关
# ==========================================

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = ''

class ProjectResponse(BaseModel):
    id: int
    name: str
    description: str
    status: str
    owner_id: Optional[int] = None
    owner_name: Optional[str] = None
    is_public: int = 0
    feature_list: List[str] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ProjectVisibilityUpdate(BaseModel):
    is_public: int

class ProjectListResponse(BaseModel):
    total: int
    items: List[ProjectResponse]


# ========== 数据集 ==========

class DatasetResponse(BaseModel):
    id: int
    project_id: int
    name: str
    file_path: str
    file_size: int
    n_rows: int
    n_cols: int
    columns_info: Dict[str, Any]
    stats_cache: Optional[Dict[str, Any]] = None
    l1_results: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        orm_mode = True

class DataPreviewResponse(BaseModel):
    columns: List[str]
    dtypes: Dict[str, str]
    data: List[Dict[str, Any]]
    total_rows: int

class DataStatsResponse(BaseModel):
    n_rows: int
    n_cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_rates: Dict[str, float]
    numeric_stats: Dict[str, Dict[str, Any]]  # {col: {mean, std, min, max, ...}}
    label_distribution: Optional[Dict[str, int]] = None

class BinningExplorerRequest(BaseModel):
    variable: str
    label_col: str = "label"
    method: str = "decision_tree" # 'quantile', 'decision_tree', 'chi'
    n_bins: int = 10
    min_samples_leaf: float = 0.05
    max_leaf_nodes: int = 10

class AutoMiningRequest(BaseModel):
    dataset_id: int
    label_col: str = "label"
    max_vars: int = 1
    min_lift: float = 1.2
    min_bad_rate: float = 0.05


# ========== 特征工程 ==========

class FilterThresholds(BaseModel):
    """变量筛选阈值（前端可调）"""
    missing: float = 0.8
    std: float = 0.95
    freq: float = 0.95
    iv: float = 0.02
    corr: float = 0.9
    psi: float = 0.1
    importance: float = 0.0
    chi2: float = 3.0

class FeatureFilterRequest(BaseModel):
    dataset_id: int
    dep: str = 'label'
    exclude_cols: List[str] = []
    thresholds: FilterThresholds = FilterThresholds()
    skip_l1: bool = True
    # 划分参数
    split_ratios: Optional[List[float]] = [0.6, 0.2, 0.2]
    oot_col: Optional[str] = None
    oot_start_time: Optional[str] = None
    oot_pct: Optional[float] = None  # 如 0.1 表示最后10%

class IVReportRequest(BaseModel):
    dataset_id: int
    dep: str = 'label'
    exclude_cols: List[str] = []
    # 划分参数
    split_ratios: Optional[List[float]] = [0.6, 0.2, 0.2]
    oot_col: Optional[str] = None
    oot_start_time: Optional[str] = None
    oot_pct: Optional[float] = None

class FeatureReportResponse(BaseModel):
    features: List[Dict[str, Any]]  # [{name, iv, ks, psi, missing_rate, ...}]
    total_features: int
    filtered_features: int


class ScoreConfig(BaseModel):
    mode: str = 'manual' # 'manual' or 'auto'
    base_score: float = 600.0
    pdo: float = 20.0
    base_odds: float = 50.0
    min_score: float = 300.0
    max_score: float = 900.0

# ========== 建模 ==========

class ModelingRequest(BaseModel):
    """建模请求（前端可调参数）"""
    dataset_id: int
    dep: str = 'label'
    exclude_cols: List[str] = []
    feature_list: Optional[List[str]] = None  # 为空则使用全部非排除列
    model_type: str = 'xgb'  # xgb / lgb / lr / dt / rf / exrf / catboost
    strategy_type: int = 3   # Optuna 策略编号
    n_trials: int = 100
    max_depth: int = 6
    strategy_threshold: float = 0.03
    score_config: ScoreConfig = ScoreConfig()
    # 划分参数
    split_ratios: Optional[List[float]] = None
    oot_col: Optional[str] = None
    oot_start_time: Optional[str] = None
    oot_pct: Optional[float] = None

class ModelingResponse(BaseModel):
    task_id: int
    status: str


# ========== 任务 ==========

class TaskResponse(BaseModel):
    id: int
    project_id: int
    task_type: str
    status: str
    progress: float
    params: Dict[str, Any]
    result: Any
    error_msg: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


# ========== 模型结果 ==========

class ModelResultResponse(BaseModel):
    id: int
    project_id: int
    task_id: Optional[int]
    model_type: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    feature_importance: Dict[str, Any]
    feature_list: List[str]
    optuna_strategy: Optional[int]
    n_trials: int
    score_config: Optional[Dict[str, Any]] = None
    score_distribution: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        orm_mode = True


# ========== 上线与监控 ==========

class DeploymentRequest(BaseModel):
    model_result_id: int

class DeploymentResponse(BaseModel):
    id: int
    project_id: int
    model_result_id: int
    status: str
    deployed_at: datetime
    class Config:
        orm_mode = True

class SimulationRequest(BaseModel):
    deployment_id: int
    n_samples: int = 5000
    drift_scale: float = 0.03
    batch_name: Optional[str] = None

class MonitoringLogResponse(BaseModel):
    id: int
    deployment_id: int
    batch_name: str
    sample_size: int
    psi: float
    avg_score: float
    score_dist: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: datetime
    class Config:
        orm_mode = True

# ========== 策略相关 ==========

class Rule(BaseModel):
    field: str
    op: str  # >, <, >=, <=, ==, !=
    val: Any
    logic: str = 'and'  # and, or

class StrategyAnalyzeRequest(BaseModel):
    project_id: int
    dataset_id: int
    rules: List[Rule]
    combine_logic: str = 'and' # and: 满足所有, or: 满足任一
    rule_type: str = 'reject'  # reject: 拦截, approve: 通过

class StrategyCreate(BaseModel):
    project_id: int
    name: str
    description: Optional[str] = None
    rules: List[Rule]
    combine_logic: str = 'and'
    rule_type: str = 'reject'
    metrics: Dict[str, Any] = {}

class StrategyResponse(BaseModel):
    id: int
    project_id: int
    name: str
    description: Optional[str]
    status: str
    priority: int
    combine_logic: str
    rule_type: str
    rules: List[Rule]
    metrics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    class Config:
        orm_mode = True
class StrategyReorderItem(BaseModel):
    id: int
    priority: int

class StrategyReorderRequest(BaseModel):
    items: List[StrategyReorderItem]

class StrategyStatusUpdateRequest(BaseModel):
    status: str
