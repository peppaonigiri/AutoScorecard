import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from sqlalchemy import text
from app.database import engine

def run():
    print("Migrating database for AB Experiment (Strategy Compare)...")
    try:
        # 使用 begin() 以适配 SQLAlchemy 1.x 的事务块
        with engine.begin() as conn:
            conn.execute(text('ALTER TABLE strategy_monitoring_logs ADD COLUMN IF NOT EXISTS compare_result JSONB DEFAULT NULL'))
        print("OK: Added compare_result JSONB to strategy_monitoring_logs")
    except Exception as e:
        print(f"Error during migration: {e}")
        
if __name__ == "__main__":
    run()
