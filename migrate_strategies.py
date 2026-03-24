# -*- coding: utf-8 -*-
import sqlite3
import os

db_path = 'scorecard.db'

def migrate():
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    columns_to_add = [
        ("status", "VARCHAR(50) DEFAULT 'draft'"),
        ("priority", "INTEGER DEFAULT 0"),
        ("combine_logic", "VARCHAR(20) DEFAULT 'and'"),
        ("rule_type", "VARCHAR(20) DEFAULT 'reject'"),
        ("updated_at", "DATETIME DEFAULT '2026-03-20 11:00:00'")
    ]

    for col_name, col_type in columns_to_add:
        try:
            cursor.execute(f"ALTER TABLE strategies ADD COLUMN {col_name} {col_type}")
            print(f"Added column {col_name} to strategies table.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                print(f"Column {col_name} already exists.")
            else:
                print(f"Error adding column {col_name}: {e}")

    # 创建策略监控日志表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS strategy_monitoring_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        batch_name VARCHAR(100),
        total_count INTEGER DEFAULT 0,
        pass_count INTEGER DEFAULT 0,
        hit_count INTEGER DEFAULT 0,
        approval_rate FLOAT DEFAULT 0.0,
        rule_stats JSON DEFAULT '[]',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(project_id) REFERENCES projects(id)
    )
    """)
    print("Strategy monitoring logs table verified/created.")

    conn.commit()
    conn.close()
    print("Migration completed.")

if __name__ == "__main__":
    migrate()
