# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app.models import Project, User
from app.schemas import ProjectResponse
import traceback

print("--- 诊断开始 ---")
db = SessionLocal()
try:
    p = db.query(Project).first()
    if p:
        print(f"数据加载成功: id={p.id}, name={p.name}")
        pr = ProjectResponse.from_orm(p)
        print("Pydantic 序列化成功:", pr.dict())
    else:
        print("数据库无项目数据")
        
    u = db.query(User).first()
    if u:
        print(f"用户加载成功: id={u.id}, username={u.username}")
    else:
        print("数据库无用户数据")

except Exception:
    print("发生错误:")
    traceback.print_exc()
finally:
    db.close()
print("--- 诊断结束 ---")
