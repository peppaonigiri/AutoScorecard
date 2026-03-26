# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app.api.project import list_projects
from app.models import User
import traceback

print("--- 后端函数内诊断开始 ---")
db = SessionLocal()
try:
    user = db.query(User).filter(User.username == 'root').first()
    print(f"用户: {user.username}")
    
    print("正在调用 list_projects...")
    res = list_projects(db=db, current_user=user)
    print(f"调用成功! 类型: {type(res)}")
    print(f"结果: total={res.total}, 数量={len(res.items)}")
    
except Exception:
    print("捕获到异常 Traceback:")
    traceback.print_exc()
finally:
    db.close()
print("--- 后端函数内诊断结束 ---")
