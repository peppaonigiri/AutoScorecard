import os
import sys

# Ensure backend directory is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.database import SessionLocal
from app.models import Dataset, ModelResult
from app.config import UPLOAD_DIR, MODEL_DIR, REPORT_DIR

def get_db_referenced_files():
    db = SessionLocal()
    try:
        datasets = db.query(Dataset.file_path).all()
        models = db.query(ModelResult.model_path).all()
        
        # Extract file paths, filter out None or empty values, and normalize
        referenced_files = set()
        for (fp,) in datasets:
            if fp:
                referenced_files.add(os.path.normpath(os.path.abspath(fp)).lower())
        for (mp,) in models:
            if mp:
                referenced_files.add(os.path.normpath(os.path.abspath(mp)).lower())
                
        return referenced_files
    finally:
        db.close()

def get_storage_files(dirs):
    all_files = set()
    for d in dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                all_files.add(os.path.normpath(os.path.abspath(os.path.join(root, f))).lower())
    return all_files

if __name__ == "__main__":
    referenced_files = get_db_referenced_files()
    storage_dirs = [UPLOAD_DIR, MODEL_DIR, REPORT_DIR]
    storage_files = get_storage_files(storage_dirs)
    
    unreferenced_files = storage_files - referenced_files
    
    print(f"Total files referenced by DB: {len(referenced_files)}")
    print(f"Total files in Storage: {len(storage_files)}")
    print(f"Deleting {len(unreferenced_files)} unreferenced files...")
    
    deleted_count = 0
    for f in unreferenced_files:
        try:
            if os.path.exists(f):
                os.remove(f)
                deleted_count += 1
                print(f"Deleted: {f}")
        except Exception as e:
            print(f"Failed to delete {f}: {e}")
            
    print(f"Cleanup complete. Deleted {deleted_count} files.")
