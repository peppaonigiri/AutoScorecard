# -*- coding: utf-8 -*-
"""
PostgreSQL 数据库连接测试脚本
使用 config.yaml 中预留的参数进行连接验证
"""

import yaml
import os
import sys
from sqlalchemy import create_engine, text

# 1. 定位并加载配置文件 (寻找根目录下的 config.yaml)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR)
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')

def test_postgresql_connection():
    print(f"🔹 寻找配置文件路径: {CONFIG_PATH}")
    
    try:
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"未找到 {CONFIG_PATH}")

        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        # 提取配置参数 (根据之前的配置约定)
        # 这里我们手动写上您在 config.yaml 中注释的部分 
        # 如果后续您在配置文件中取消了注释，也可以改成直接读取 cfg['database']
        db_user = "postgres"
        db_password = "123456"
        db_host = "localhost"
        db_port = 5432
        db_name = "scorecard"
        
        # 构建连接字符串 (建议安装 psycopg2-binary)
        # 格式: postgresql://user:password@host:port/database
        connection_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        print(f"🚀 正在尝试建立连接: {db_host}:{db_port}/{db_name} ...")
        
        # 2. 创建引擎并执行简单查询
        engine = create_engine(connection_url, connect_args={'connect_timeout': 5})
        
        with engine.connect() as conn:
            # 执行一个典型的 SQL 语句查看版本
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()
            print("\n" + "="*40)
            print("✅ 数据库成功连接！")
            print(f"🔹 数据库版本: {version[0]}")
            print("="*40)
            
    except FileNotFoundError as e:
        print(f"❌ 错误: {str(e)}")
    except Exception as e:
        print("\n" + "="*40)
        print("❌ 数据库连接失败!")
        print(f"⚠️ 报错详情: {str(e)}")
        print("="*40)
        print("\n💡 排错指南:")
        print("1. 请确认 PostgreSQL 数据库服务已启动。")
        print("2. 检查数据库 'scorecard' 是否已创建。")
        print("3. 请确保已安装数据库驱动: 'pip install psycopg2-binary'")
        print("4. 确认防火墙规则已放行 5432 端口。")

if __name__ == "__main__":
    test_postgresql_connection()
