import sqlite3
import os

db_path = r'd:\work\98.selftest\AutoModeling\scorecard.db'

def migrate():
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        print("Adding score_config column...")
        cursor.execute("ALTER TABLE model_results ADD COLUMN score_config JSON DEFAULT '{}'")
        print("Adding score_distribution column...")
        cursor.execute("ALTER TABLE model_results ADD COLUMN score_distribution JSON DEFAULT '{}'")
        conn.commit()
        print("Migration successful.")
    except sqlite3.OperationalError as e:
        if 'duplicate column name' in str(e).lower():
            print("Columns already exist.")
        else:
            print(f"OperationalError: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    migrate()
