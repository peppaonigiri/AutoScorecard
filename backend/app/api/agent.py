# -*- coding: utf-8 -*-
"""
Agent API 路由

POST /api/agent/run
  接收：project_id, dataset_id, dep, 以及用户自然语言指令
  返回：SSE 流（text/event-stream）

每个 SSE 帧是一个 JSON 对象：
  {"type": "tool_start",  "step": N, "tool": "run_iv_report", "args": {...}}
  {"type": "tool_done",   "step": N, "tool": "run_iv_report", "result": {...}}
  {"type": "llm_thought", "content": "LLM 的分析文字"}
  {"type": "finished",    "summary": "最终建模汇总"}
  {"type": "error",       "message": "..."}
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from app.database import get_db
from app.api.auth import get_current_user
from app.agent.pipeline import AutoModelingAgent
from app.config import AGENT_ENABLED

router = APIRouter(prefix="/api/agent", tags=["LLM Agent"])


class AgentChatRequest(BaseModel):
    """多轮对话请求体：前端持有并传入完整对话历史"""
    project_id: int = Field(..., description="已创建的项目 ID")
    dataset_id: int = Field(..., description="已上传的数据集 ID")
    dep: str = Field(default="label", description="标签列名")
    model_type: str = Field(default="xgb", description="模型类型")
    n_trials: int = Field(default=30, description="Optuna 搜索次数")
    exclude_cols: List[str] = Field(default_factory=list)
    # 多轮对话历史（由前端维护，每次请求传入全量）
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="对话历史，格式: [{role: 'user'|'assistant', content: '...'}]"
    )
    # 本轮新消息
    user_message: str = Field(..., description="用户本轮输入")


@router.post("/chat")
async def agent_chat(
    req: AgentChatRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    多轮对话端点。
    前端维护完整 messages 历史，每次请求传入 messages + user_message。
    以 SSE 流返回当前轮次的工具调用进度和最终回复。

    SSE 帧格式（同 /run）：
      {"type": "tool_start",  "step": N, "tool": "...", "args": {...}}
      {"type": "tool_done",   "step": N, "tool": "...", "result": {...}}
      {"type": "llm_thought", "content": "..."}
      {"type": "finished",    "summary": "...", "assistant_message": "完整回复文本"}
      {"type": "error",       "message": "..."}
    """
    if not AGENT_ENABLED:
        import json
        async def _err():
            yield f"data: {json.dumps({'type':'error','message':'Agent 未启用'})}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    # 移除首轮 user_message 注入前缀的做法，改为将当前最新状态作为 context 传入
    full_user_message = req.user_message

    agent = AutoModelingAgent(db=db, user_id=current_user.id)

    return StreamingResponse(
        agent.run_stream(
            user_message=full_user_message,
            history=req.messages,
            context={
                "project_id": req.project_id,
                "dataset_id": req.dataset_id,
                "dep": req.dep,
                "model_type": req.model_type,
                "n_trials": req.n_trials,
                "exclude_cols": req.exclude_cols or [],
            }
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
