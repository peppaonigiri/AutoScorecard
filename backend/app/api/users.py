# -*- coding: utf-8 -*-
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.schemas import UserCreate, UserResponse, PasswordChange, RoleChange, AdminPasswordReset
from app.api.auth import get_password_hash, get_current_user, get_current_admin_user, verify_password

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
