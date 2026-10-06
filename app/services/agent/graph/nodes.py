import json
import uuid
import logging
from dataclasses import replace, is_dataclass
from typing import Any, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.llm.base import LLMClient
from app.services.agent.tools import execute_tool
from app.services.agent.memory.short_term import add_chat_message
from app.services.llm.llm_types import LLMPurpose, TextDelta, ToolCall, Usage
from app.schemas import (
    Node, 
    MessageRole,
    MessageType, 
    ThoughtStep, 
    NodeTransition, 
    ToolCallDetail
)

logger = logging.getLogger(__name__)

def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Retrieve value from dict or dataclass object safely."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _update_state(state: Any, **updates) -> Any:
    """Immutably update state dict or dataclass object."""
    if isinstance(state, dict):
        new_state = dict(state)
        new_state.update(updates)
        return new_state
    if is_dataclass(state):
        return replace(state, **updates)
    # Fallback if state is an object with __dict__
    new_state = dict(getattr(state, "__dict__", {}))
    new_state.update(updates)
    return new_state


async def think_node(
    state: Any, 
    llm: LLMClient,
    session_id: uuid.UUID, 
) -> AsyncGenerator[str | NodeTransition, None]:
    """Execute the LLM reasoning (think) turn in the agent workflow.

    Streams response chunks from the Groq model and determines whether the agent
    decides to call external tools or complete the final answer turn.

    Args:
        state (Any): Current agent graph state containing messages and turn counts.
        db (AsyncSession): Active database session.
        session_id (uuid.UUID): Target chat session UUID.

    Yields:
        AsyncGenerator[str | NodeTransition, None]: SSE stream formatted event chunks
        or NodeTransition signal indicating next node (EXECUTE or END).

    Raises:
        RuntimeError: Raised when LLM generation fails.
    """
    messages = _get_val(state, "messages", [])
    loop_count = _get_val(state, "loop_count", 0)
    logger.debug(f"[Session {session_id}] Entering think_node, turn={loop_count + 1}")

    text, tool_calls, usage = "", [], None
    async for ev in llm.stream(messages, LLMPurpose.AGENT):
        if isinstance(ev, TextDelta):
            text += ev.text
            yield f"event: answer\ndata: {json.dumps({'text': ev.text})}\n\n"
        elif isinstance(ev, ToolCall):
            tool_calls.append(ev)
        elif isinstance(ev, Usage):
            usage = ev

    if usage is None:
        logger.warning(f"[Session {session_id}] Token usage statistics were None for turn {loop_count + 1}")

    role: str = "assistant"

    new_state = _update_state(
        state,
        last_turn_usage=usage,
        loop_count=loop_count + 1
    )

    if not tool_calls:
        logger.info(f"[Session {session_id}] think_node finished reasoning without calling tools.")
        if not text.strip():
            text = (
                "I apologize, but I did not receive a suitable response from the model. "
                "Could you please try asking the question again?"
            )

            yield f"event: answer\ndata: {json.dumps({'text': text})}\n\n"

        message = {
            "role": role,
            "content": text
        }

        final_step = ThoughtStep(
            loop_index=_get_val(new_state, "loop_count", 1) - 1,
            prompt_tokens=usage.prompt_tokens if usage is not None else 0,
            completion_tokens=usage.completion_tokens if usage is not None else 0,
            token=usage.total if usage is not None else 0,
            thought=text,
            tool_calls=[]
        )

        existing_steps = _get_val(new_state, "thought_steps", []) or []
        final_state = _update_state(
            new_state,
            messages=messages + [message],
            thought_steps=existing_steps + [final_step],
            final_response=text or "",
            next_node=Node.END
        )

        yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"
        yield NodeTransition(state=final_state, next_node=Node.END)
        return

    tool_calls_payload = []
    for tc in tool_calls:
        tool_calls_payload.append({
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": tc.arguments
            }
        })

    logger.info(f"[Session {session_id}] think_node selected {len(tool_calls_payload)} tool call(s).")
    message = {
        "role": role,
        "content": text,
        "tool_calls": tool_calls_payload
    }

    next_state = _update_state(
        new_state,
        messages=messages + [message],
        next_node=Node.EXECUTE
    )

    yield NodeTransition(state=next_state, next_node=Node.EXECUTE)
    return

async def execute_node(
    state: Any, 
    db: AsyncSession, 
    session_id: uuid.UUID, 
    document_id: uuid.UUID | None
) -> AsyncGenerator[str | NodeTransition, None]:
    """Execute requested tool calls from the last reasoning turn.

    Dispatches tool calls, records executed thought steps, updates active document context,
    and transitions back to the THINK node for follow-up reasoning.

    Args:
        state (Any): Current agent graph state containing tool call requests.
        db (AsyncSession): Active database session for tool operations.
        session_id (uuid.UUID): Target chat session UUID.

    Yields:
        AsyncGenerator[str | NodeTransition, None]: SSE stream formatted event chunks
        or NodeTransition signal back to THINK node.
    """
    messages = _get_val(state, "messages", [])
    tc_messages = messages[-1] if messages else {}
    tool_calls_detail: list[ToolCallDetail] = []
    new_messages: list[dict] = list(messages)

    tool_calls = tc_messages.get("tool_calls", []) if isinstance(tc_messages, dict) else (getattr(tc_messages, "tool_calls", []) or [])
    logger.debug(f"[Session {session_id}] Entering execute_node, executing {len(tool_calls)} tool call(s).")

    await add_chat_message(
        db=db,
        session_id=session_id,
        document_id=document_id,
        role=MessageRole.ASSISTANT,
        type=MessageType.TOOL_CALL,
        content={"tool_info": tool_calls}
    )

    for tc in tool_calls:
        tc_id = tc["id"] if isinstance(tc, dict) else tc.id
        func = tc["function"] if isinstance(tc, dict) else tc.function
        tool_name = func["name"] if isinstance(func, dict) else func.name
        args_str = func["arguments"] if isinstance(func, dict) else func.arguments

        try:
            tool_input = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            tool_input = {}

        logger.info(f"[Session {session_id}] Executing tool '{tool_name}' with args: {tool_input}")

        result = await execute_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            db=db
        )

        tool_calls_detail.append(ToolCallDetail(
            id=tc_id,
            name=tool_name,
            arguments=tool_input,
            result=str(result)
        ))

        await add_chat_message(
            db=db,
            session_id=session_id,
            document_id=document_id,
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content={"tool_result": str(result)}
        )

        new_messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": str(result)
        })

    thought_content = tc_messages.get("content") if isinstance(tc_messages, dict) else getattr(tc_messages, "content", "")

    loop_count = _get_val(state, "loop_count", 1)
    last_turn_usage = _get_val(state, "last_turn_usage", None)
    existing_steps = _get_val(state, "thought_steps", []) or []

    new_thought_step = ThoughtStep(
        loop_index=loop_count - 1,
        prompt_tokens=last_turn_usage.prompt_tokens if last_turn_usage is not None else 0,
        completion_tokens=last_turn_usage.completion_tokens if last_turn_usage is not None else 0,
        token=last_turn_usage.total if last_turn_usage is not None else 0,
        thought=thought_content,
        tool_calls=tool_calls_detail
    )

    await add_chat_message(
        db=db,
        session_id=session_id,
        document_id=document_id,
        role=MessageRole.ASSISTANT,
        type=MessageType.THINKING,
        content={"thought": new_thought_step.model_dump()}
    )

    new_state = _update_state(
        state,
        messages=new_messages,
        thought_steps=existing_steps + [new_thought_step],
        final_response=None,
        next_node=Node.THINK
    )

    yield f"event: thought\ndata: {new_thought_step.model_dump_json()}\n\n"
    yield NodeTransition(state=new_state, next_node=Node.THINK)
    return