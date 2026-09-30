import json
import uuid
import logging
from dataclasses import replace, is_dataclass
from typing import Any, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.agent.memory.short_term import add_chat_message
from app.services.client import GROQ_MODEL, groq_client
from app.services.agent.tools import TOOLS, execute_tool
from app.schemas import (
    MessageRole,
    MessageType, 
    Node, 
    NodeTransition, 
    ThoughtStep, 
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
    db: AsyncSession, 
    session_id: uuid.UUID, 
    document_id: uuid.UUID | None
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

    accumulated_tc: dict = {}
    accumulated_content: str = ""
    role: str = "assistant"
    calling_tools: bool = False
    turn_tokens: int = 0
    try:
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS,
            stream=True
        )
    except Exception as e:
        logger.error(f"[Session {session_id}] LLM generation failed: {e}")
        raise RuntimeError(f"Generate failed: {e}")

    async for chunk in response:
        delta = chunk.choices[0].delta

        if getattr(chunk, "usage", None) is not None:
            turn_tokens = chunk.usage.total_tokens

        if getattr(delta, "content", None) is not None:
            piece_content = delta.content
            accumulated_content += piece_content

            yield f"event: answer\ndata: {json.dumps({'text': piece_content})}\n\n"

        if getattr(delta, "tool_calls", None) is not None:
            calling_tools = True
            for tc in delta.tool_calls:
                index = tc.index

                if index not in accumulated_tc:
                    accumulated_tc[index] = {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": ""
                    }

                if tc.function.arguments:
                    accumulated_tc[index]["arguments"] += tc.function.arguments

    new_state = _update_state(
        state,
        last_turn_tokens=turn_tokens,
        loop_count=loop_count + 1
    )

    if not calling_tools:
        logger.info(f"[Session {session_id}] think_node finished reasoning without calling tools.")
        if not accumulated_content.strip():
            accumulated_content = (
                "I apologize, but I did not receive a suitable response from the model. "
                "Could you please try asking the question again?"
            )

            yield f"event: answer\ndata: {json.dumps({'text': accumulated_content})}\n\n"

        message = {
            "role": role,
            "content": accumulated_content
        }

        final_step = ThoughtStep(
            loop_index=_get_val(new_state, "loop_count", 1) - 1,
            token=turn_tokens,
            thought=accumulated_content,
            tool_calls=[]
        )

        existing_steps = _get_val(new_state, "thought_steps", []) or []
        final_state = _update_state(
            new_state,
            messages=messages + [message],
            thought_steps=existing_steps + [final_step],
            final_response=accumulated_content or "",
            next_node=Node.END
        )

        yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"
        yield NodeTransition(state=final_state, next_node=Node.END)
        return

    final_tool_calls = []
    tool_calls_payload = []
    for idx, tc in accumulated_tc.items():
        try:
            parsed_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
        except json.JSONDecodeError:
            parsed_args = {}

        final_tool_calls.append({
            "id": tc["id"],
            "name": tc["name"],
            "arguments": parsed_args
        })

        tool_calls_payload.append({
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": tc["arguments"]
            }
        })

    logger.info(f"[Session {session_id}] think_node selected {len(final_tool_calls)} tool call(s).")
    message = {
        "role": role,
        "content": accumulated_content,
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
    current_doc_ids = set(_get_val(state, "doc_ids_used", []))
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
        
        tool_input = json.loads(args_str) if isinstance(args_str, str) else args_str
        doc_id_param = tool_input.get("document_id")

        logger.info(f"[Session {session_id}] Executing tool '{tool_name}' with args: {tool_input}")

        result = await execute_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            db=db
        )

        if doc_id_param and not str(result).startswith("Error:"):
            try:
                doc_uuid = uuid.UUID(doc_id_param) if isinstance(doc_id_param, str) else doc_id_param
                current_doc_ids.add(doc_uuid)
            except Exception:
                pass

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
    last_turn_tokens = _get_val(state, "last_turn_tokens", 0)
    existing_steps = _get_val(state, "thought_steps", []) or []

    new_thought_step = ThoughtStep(
        loop_index=loop_count - 1,
        token=last_turn_tokens,
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
        doc_ids_used=list(current_doc_ids),
        thought_steps=existing_steps + [new_thought_step],
        final_response=None,
        next_node=Node.THINK
    )

    yield f"event: thought\ndata: {new_thought_step.model_dump_json()}\n\n"
    yield NodeTransition(state=new_state, next_node=Node.THINK)
    return