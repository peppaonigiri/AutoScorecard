# -*- coding: utf-8 -*-
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.schemas import UserCreate, UserResponse, PasswordChange, RoleChange, AdminPasswordReset
from app.api.auth import get_password_hash, get_current_user, get_current_admin_user, verify_password
import os
import shutil
import pandas as pd
from app.config import UPLOAD_DIR

router = APIRouter(prefix="/api/v1/users", tags=["users"])

@router.post("/register", response_model=UserResponse)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == user_in.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user_in.password)
    new_user = User(
        username=user_in.username,
        hashed_password=hashed_password,
        is_admin=0,  # 强制规定页面注册的都是普通用户
        is_active=1
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # ---------------------------------------------------------
    # 自动创建默认项目 example 并初始化示例数据
    # ---------------------------------------------------------
    try:
        from app.models import Project, Dataset
        from scorecard_core.data_processor import calculate_dataset_summary
        
        # 1. 创建项目
        example_project = Project(
            name="example",
            description="系统自动创建的示例项目",
            owner_id=new_user.id
        )
        db.add(example_project)
        db.commit()
        db.refresh(example_project)
        
        # 2. 准备示例数据文件
        # 公共数据存放路径
        common_data_path = os.path.join(os.path.dirname(UPLOAD_DIR), 'common', 'example_data.csv')
        
        if os.path.exists(common_data_path):
            # 为该项目创建专属存储目录
            proj_data_dir = os.path.join(UPLOAD_DIR, str(example_project.id))
            os.makedirs(proj_data_dir, exist_ok=True)
            
            target_file_path = os.path.join(proj_data_dir, 'example_data.csv')
            shutil.copy2(common_data_path, target_file_path)
            
            # 3. 解析并计算指标 (模拟上传过程)
            df = pd.read_csv(target_file_path)
            stats, l1_res = calculate_dataset_summary(df)
            
            example_dataset = Dataset(
                project_id=example_project.id,
                name="example_data.csv",
                file_path=target_file_path,
                file_size=os.path.getsize(target_file_path),
                n_rows=len(df),
                n_cols=len(df.columns),
                columns_info=stats['dtypes'],
                stats_cache=stats,
                l1_results=l1_res
            )
            db.add(example_dataset)
            db.commit()
            
    except Exception as e:
        # 即使自动创建失败，也不应阻断用户注册流程，记录错误即可
        import logging
        logging.error(f"子流程错误: 自动创建默认项目失败: {e}")

    return new_user

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

@router.put("/me/password")
def update_password(pw_change: PasswordChange, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(pw_change.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Old password incorrect")
    
    current_user.hashed_password = get_password_hash(pw_change.new_password)
    db.add(current_user)
    db.commit()
    return {"success": True}

@router.get("/all", response_model=List[UserResponse])
def get_all_users(current_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

@router.put("/{user_id}/role")
def update_user_role(user_id: int, role_change: RoleChange, current_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    if target_user.username == "root":
        raise HTTPException(status_code=403, detail="Cannot modify root user's role")
        
    target_user.is_admin = role_change.is_admin
    db.add(target_user)
    db.commit()
    return {"success": True, "is_admin": target_user.is_admin}

@router.put("/{user_id}/reset-password")
def reset_user_password(user_id: int, pw_reset: AdminPasswordReset, current_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """管理员重置用户密码"""
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
        
    target_user.hashed_password = get_password_hash(pw_reset.new_password)
    db.add(target_user)
    db.commit()
    return {"success": True, "detail": "Password reset successfully"}

@router.get("/{user_id}/projects")
def get_user_projects(user_id: int, current_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """获取指定用户的所有项目（管理员用）"""
    from app.models import Project
    projects = db.query(Project).filter(Project.owner_id == user_id).order_by(Project.created_at.desc()).all()
    return projects

@router.delete("/{user_id}")
def delete_user(user_id: int, current_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """管理员删除用户（级联删除所有项目及物理文件）"""
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    if target_user.username == "root":
        raise HTTPException(status_code=403, detail="Cannot delete root user")
    
    # 物理文件清理：获取该用户的所有项目ID
    # 由于设置了级联删除，我们需要在从DB删除User之前手动清理文件
    from app.models import Project
    projects = db.query(Project).filter(Project.owner_id == user_id).all()
    
    for proj in projects:
        proj_dir = os.path.join(UPLOAD_DIR, str(proj.id))
        if os.path.exists(proj_dir):
            try:
                shutil.rmtree(proj_dir)
            except Exception as e:
                import logging
                logging.error(f"清理用户项目目录失败 {proj_dir}: {e}")

    # 从数据库删除 (触发 ORM 级联关系)
    db.delete(target_user)
    db.commit()
    
    return {"success": True, "detail": f"User {user_id} and all related data deleted"}
