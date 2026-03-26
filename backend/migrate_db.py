# -*- coding: utf-8 -*-
from app.database import engine
from sqlalchemy import text

def migrate():
    with engine.connect() as conn:
        print("Starting migration...")
        # 尝试逐条执行并提交 (SQLAlchemy 1.4+ 自动处理事务或需要显式 commit)
        for col_def in [
            "ALTER TABLE projects ADD COLUMN owner_id INTEGER;",
            "ALTER TABLE projects ADD COLUMN is_public INTEGER DEFAULT 0;"
        ]:
            try:
                conn.execute(text(col_def))
                print(f"Executed: {col_def}")
            except Exception as e:
                print(f"Skipped (already exists?): {col_def} - Error: {e}")
        
        # 针对 SQLite/PG 通用手动提交（1.4 版本）
        try:
            conn.execute(text("COMMIT;"))
        except:
            pass
        print("Migration process finished.")

if __name__ == "__main__":
    migrate()
