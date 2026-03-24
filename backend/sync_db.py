import sys
import os

# 确保项目根目录在 sys.path 中
_base_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_base_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 导入 Base, engine 和 models 确保 tables 注册
from app.database import Base, engine
from app.models import Project, Dataset, Task, ModelResult, Deployment, MonitoringLog

print("正在同步数据库表结构...")
try:
    Base.metadata.create_all(bind=engine)
    print("数据库表结构同步成功。")
except Exception as e:
    print(f"同步失败: {e}")
