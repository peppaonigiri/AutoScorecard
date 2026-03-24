import sqlite3
import os

db_path = 'd:/work/98.selftest/AutoModeling/scorecard.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE strategies ADD COLUMN description TEXT DEFAULT ''")
    conn.commit()
    print("Column added successfully")
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
