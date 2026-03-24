# -*- coding: utf-8 -*-
"""ORM 模型定义"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class User(Base):
    """用户"""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    is_admin = Column(Integer, default=0)  # 0: 普通用户, 1: 管理员 (SQLite boolean workaround)
    is_active = Column(Integer, default=1) # 0: 停用, 1: 启用
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Project(Base):
    """项目"""
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, default='')
    status = Column(String(50), default='created')  # created / processing / completed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关联
    datasets = relationship('Dataset', back_populates='project', cascade='all, delete-orphan')
    tasks = relationship('Task', back_populates='project', cascade='all, delete-orphan')
    model_results = relationship('ModelResult', back_populates='project', cascade='all, delete-orphan')

    # 持久化状态
    feature_list = Column(JSON, default=list)  # 最近一次成功筛选或选择的特征列表
    split_config = Column(JSON, default=dict)  # 数据集划分配置 {ratios, oot_col, oot_start_time}


class Dataset(Base):
    """数据集"""
    __tablename__ = 'datasets'

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    name = Column(String(200), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, default=0)
    n_rows = Column(Integer, default=0)
    n_cols = Column(Integer, default=0)
    columns_info = Column(JSON, default=dict)  # {col_name: dtype, ...}
    stats_cache = Column(JSON, default=dict)   # 缓存的基础统计信息
    l1_results = Column(JSON, default=dict)    # L1 初筛结果 {kept, dropped_info}
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关联
    project = relationship('Project', back_populates='datasets')


class Task(Base):
    """异步任务"""
    __tablename__ = 'tasks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    task_type = Column(String(50), nullable=False)  # feature_report / filter / modeling
    status = Column(String(50), default='pending')  # pending / running / completed / failed
    progress = Column(Float, default=0.0)  # 0-100
    params = Column(JSON, default=dict)  # 任务参数
    result = Column(JSON, default=dict)  # 任务结果
    error_msg = Column(Text, default='')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关联
    project = relationship('Project', back_populates='tasks')


class ModelResult(Base):
    """模型结果"""
    __tablename__ = 'model_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    task_id = Column(Integer, ForeignKey('tasks.id'), nullable=True)
    model_type = Column(String(50), nullable=False)
    model_path = Column(String(500), default='')  # 模型文件路径
    params = Column(JSON, default=dict)  # 最优参数
    metrics = Column(JSON, default=dict)  # {train_ks, valid_ks, oot_ks, train_auc, ...}
    feature_importance = Column(JSON, default=dict)  # {feature_name: importance, ...}
    feature_list = Column(JSON, default=list)  # 使用的特征列表
    optuna_strategy = Column(Integer, nullable=True)  # Optuna 策略类型
    n_trials = Column(Integer, default=0)
    
    # 评分卡相关配置与分布汇总
    score_config = Column(JSON, default=dict)        # {base_score, pdo, base_odds}
    score_distribution = Column(JSON, default=dict)  # {bins, counts_good, counts_bad}
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关联
    project = relationship('Project', back_populates='model_results')
    task = relationship('Task')


class Deployment(Base):
    """模型上线部署记录"""
    __tablename__ = 'deployments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    model_result_id = Column(Integer, ForeignKey('model_results.id'), nullable=False)
    status = Column(String(50), default='active')  # active / retired
    deployed_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关联
    project = relationship('Project')
    model_result = relationship('ModelResult')


class MonitoringLog(Base):
    """监控日志（包含模拟或真实打分统计）"""
    __tablename__ = 'monitoring_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    deployment_id = Column(Integer, ForeignKey('deployments.id'), nullable=False)
    batch_name = Column(String(100))  # 如 "模拟进件 2024-04"
    sample_size = Column(Integer, default=0)
    psi = Column(Float, default=0.0)
    avg_score = Column(Float, default=0.0)
    score_dist = Column(JSON, default=dict)  # 分数分布直方图数据
    metrics = Column(JSON, default=dict)    # 其他评估指标
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关联
    project = relationship('Project')
    deployment = relationship('Deployment')


class Strategy(Base):
    """风控策略配置"""
    __tablename__ = 'strategies'

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, default='')
    # 状态控制与排序
    status = Column(String(50), default='draft')  # draft / active
    priority = Column(Integer, default=0)
    combine_logic = Column(String(20), default='and')  # and / or
    rule_type = Column(String(20), default='reject')  # reject / approve
    
    # 规则列表 JSON: [{"field": "age", "op": ">", "val": 18, "logic": "and"}, ...]
    rules = Column(JSON, default=list)
    # 策略预估指标 JSON: {"approval_rate": 0.8, "lift": 1.2, ...}
    metrics = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    project = relationship("Project")

class StrategyMonitoringLog(Base):
    """策略监控日志"""
    __tablename__ = 'strategy_monitoring_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    batch_name = Column(String(100))
    total_count = Column(Integer, default=0)
    pass_count = Column(Integer, default=0)
    hit_count = Column(Integer, default=0)
    approval_rate = Column(Float, default=0.0)
    rule_stats = Column(JSON, default=list) # 详情统计 [{"id", "name", "hit_rate", ...}]
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("Project")
