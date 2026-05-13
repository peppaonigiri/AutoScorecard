# -*- coding: utf-8 -*-
"""
AutoModeling Agent Pipeline

核心 Agentic Loop：
- 接收用户自然语言指令
- 驱动 DeepSeek 通过 Function Calling 依次调用工具
- 以 SSE 格式实时向调用方推流每一步进度
"""

import json
import logging
import inspect
from typing import AsyncGenerator

from openai import AsyncOpenAI

from app.agent.tools import AgentToolkit
from app.agent.tool_schemas import ACTIVE_TOOLS
from app.agent.prompt import SYSTEM_PROMPT
from app.config import AGENT_API_KEY, AGENT_BASE_URL, AGENT_MODEL, AGENT_TIMEOUT

logger = logging.getLogger(__name__)


# ── SSE 帧格式工具函数 ────────────────────────────────────────────

def _sse(data: dict) -> str:
    """将 dict 序列化为 SSE data 帧"""
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


# ── Agent 主类 ────────────────────────────────────────────────────

class AutoModelingAgent:
    """
    第一阶段建模 Agent。
    用户已完成项目创建和数据上传，Agent 负责接管：
      IV分析 → 变量筛选 → Optuna训练 → 等待完成 → 生成报告 → 返回指标摘要
    """

    def __init__(self, db, user_id: int):
        self.toolkit = AgentToolkit(db=db, user_id=user_id)

        import httpx
        # 根本原因：系统代理的 SSL 握手失败（httpx 默认读取系统代理）
        # 通过显式设置 proxies=None 直连 DeepSeek API，绕过有问题的系统代理
        _http_client = httpx.AsyncClient(
            verify=False,
            trust_env=False,  # 禁用系统代理，直连
        )

        self.llm = AsyncOpenAI(
            api_key=AGENT_API_KEY,
            base_url=AGENT_BASE_URL,
            timeout=AGENT_TIMEOUT,
            http_client=_http_client,
        )

    # ── Tool 分发器 ───────────────────────────────────────────────
    async def _dispatch(self, tool_name: str, args: dict) -> str:
        """根据 LLM 的 tool_call 名称，路由到 AgentToolkit 对应方法"""
        fn = getattr(self.toolkit, tool_name, None)
        if fn is None:
            result = {"error": f"未知工具: {tool_name}"}
        else:
            try:
                # 支持同步和异步 tool 方法
                if inspect.iscoroutinefunction(fn):
                    result = await fn(**args)
                else:
                    result = fn(**args)
            except Exception as e:
                logger.exception(f"工具 {tool_name} 执行异常")
                result = {"error": str(e)}

        return json.dumps(result, ensure_ascii=False, default=str)

    # ── 主流：SSE 推流生成器 ─────────────────────────────────────
    async def run_stream(
        self,
        user_message: str,
        history: list = None,       # 前端传入的历史 [{role, content}, ...]
        context: dict = None,       # 当前最新的状态(如 project_id)
    ) -> AsyncGenerator[str, None]:
        """
        多轮对话 Agentic Loop。
        - history：本轮之前的对话历史（前端维护，每次全量传入）
        - user_message：本轮用户输入（已含上下文前缀）

        SSE 帧格式：
          {"type": "tool_start",  "step": N, "tool": "...", "args": {...}}
          {"type": "tool_done",   "step": N, "tool": "...", "result": {...}}
          {"type": "llm_thought", "content": "..."}
          {"type": "finished",    "summary": "...", "assistant_message": "完整回复"}
          {"type": "error",       "message": "..."}
        """
        # ── 构建完整 messages 上下文 ──────────────────────────────
        system_content = SYSTEM_PROMPT
        if context:
            system_content += (
                f"\n\n[当前系统强制注入的上下文（每一轮都有效）]\n"
                f"- 项目 ID={context.get('project_id')}\n"
                f"- 数据集 ID={context.get('dataset_id')}\n"
                f"- 标签列={context.get('dep')}\n"
                f"- 期望模型={context.get('model_type')}\n"
                f"- Optuna次数={context.get('n_trials')}\n"
                f"- 全局排除列={context.get('exclude_cols')}\n"
                f"⚠️ 注意：调用任何工具时，请**必须**使用以上确切的 project_id 和 dataset_id，绝对不可自行编造！"
            )

        messages = [{"role": "system", "content": system_content}]

        # 追加历史对话（前端传入）
        for h in (history or []):
            role = h.get("role", "user")
            content = h.get("content", "")
            # 只追加 user / assistant 两种角色（过滤掉 tool 等内部角色）
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        # 追加本轮用户消息
        messages.append({"role": "user", "content": user_message})

        step = 0
        max_loops = 20  # 防止无限循环的安全阀

        for _ in range(max_loops):
            # ── 调用 LLM ──────────────────────────────────────────
            try:
                response = await self.llm.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=messages,
                    tools=ACTIVE_TOOLS,
                    tool_choice="auto",
                )
            except Exception as e:
                yield _sse({"type": "error", "message": f"LLM 调用失败: {e}"})
                return

            msg = response.choices[0].message

            # ── 情况 A：LLM 决定调用工具 ─────────────────────────
            if msg.tool_calls:
                # openai SDK 返回的 msg 对象可直接追加到 messages
                messages.append(msg)

                for tc in msg.tool_calls:
                    step += 1
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    # 推流：工具开始
                    yield _sse({
                        "type": "tool_start",
                        "step": step,
                        "tool": tool_name,
                        "args": args,
                    })

                    # 执行工具
                    result_str = await self._dispatch(tool_name, args)
                    result_obj = json.loads(result_str)

                    # 推流：工具完成
                    yield _sse({
                        "type": "tool_done",
                        "step": step,
                        "tool": tool_name,
                        "result": result_obj,
                        "has_error": "error" in result_obj,
                    })

                    # 如果工具报错，停止流程
                    if "error" in result_obj:
                        yield _sse({
                            "type": "error",
                            "message": f"步骤 {step} ({tool_name}) 失败: {result_obj['error']}"
                        })
                        return

                    # 将工具结果追加给 LLM
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

            # ── 情况 B：LLM 认为任务完成，输出最终文本 ──────────
            else:
                final_text = msg.content or "好的，本轮任务已完成。"

                yield _sse({
                    "type": "finished",
                    "summary": final_text,
                    "assistant_message": final_text,  # 前端存入对话历史用
                })
                return

        # 超过最大循环次数（极少见）
        yield _sse({"type": "error", "message": "Agent 超过最大循环次数，请检查工具调用逻辑"})
