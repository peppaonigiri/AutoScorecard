# -*- coding: utf-8 -*-
"""应用配置"""

import os
import yaml

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载 config.yaml
_config_path = os.path.join(BASE_DIR, 'config.yaml')
with open(_config_path, 'r', encoding='utf-8') as f:
    _cfg = yaml.safe_load(f)

# 数据库配置（临时使用 SQLite）
SQLITE_DB_PATH = _cfg['database'].get('sqlite_db', './scorecard.db')
if not os.path.isabs(SQLITE_DB_PATH):
    SQLITE_DB_PATH = os.path.join(BASE_DIR, SQLITE_DB_PATH)
DATABASE_URL = f"sqlite:///{SQLITE_DB_PATH}"

# 服务器配置
SERVER_HOST = _cfg['server']['host']
SERVER_PORT = _cfg['server']['port']
SERVER_DEBUG = _cfg['server']['debug']

# 存储配置
UPLOAD_DIR = os.path.join(BASE_DIR, _cfg['storage']['upload_dir'])
MODEL_DIR = os.path.join(BASE_DIR, _cfg['storage']['model_dir'])
REPORT_DIR = os.path.join(BASE_DIR, _cfg['storage']['report_dir'])

# 确保存储目录存在
for d in [UPLOAD_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# 建模默认参数
DEFAULT_DEP = _cfg['modeling']['default_dep']
DEFAULT_MAX_DEPTH = _cfg['modeling']['default_max_depth']
DEFAULT_N_TRIALS = _cfg['modeling']['default_n_trials']
DEFAULT_MODEL_TYPE = _cfg['modeling']['default_model_type']

# scorecard 包路径（原始工具包）
SCORECARD_PATH = os.path.join(BASE_DIR, 'scorecard')
