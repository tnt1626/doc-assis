import json
from collections.abc import AsyncGenerator
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.memory import add_chat_message
from app.services.client import GROQ_MODEL, groq_client
from app.services.agent.tools import TOOLS, execute_tool
from app.schemas import (
    AgentState,
    MessageRole,
    MessageType, 
    Node, 
    NodeTransition, 
    ThoughtStep, 
    ToolCallDetail
)

async def think_node(state: AgentState, db: AsyncSession, session_id: uuid.UUID) -> AsyncGenerator[str | NodeTransition, None, None]:
    """Execute LLM reasoning turn to determine whether to call tools or finish answering.

    Args:
        state (AgentState): Current graph state containing conversation messages and step history.
        db (AsyncSession): Database session.
        session_id (uuid.UUID): Target chat session UUID.

    Yields:
        AsyncGenerator[str | NodeTransition, None]: SSE stream tokens or NodeTransition signal.
    """
    accumulated_tc: dict = {}
    accumulated_content: str = ""
    role: str = "assistant"
    calling_tools: bool = False
    turn_tokens: int = 0
    try:
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=state.messages,
            tools=TOOLS,
            stream=True
        )
    except Exception as e:
        raise RuntimeError(f"Generate failed: {e}")

    async for chunk in response:
        delta = chunk.choices[0].delta

        if getattr(delta, "role", None) is not None:
            role = delta.role

        if getattr(chunk, "usage", None) is not None:
            turn_tokens = chunk.usage.total_tokens

        if getattr(delta, "content", None) is not None:
            piece_content = delta.content
            accumulated_content += piece_content

            yield f"event: answer\ndata: {json.dumps({"text": piece_content})}\n\n"

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


    new_state = AgentState(
        question=state.question,
        messages=state.messages,
        thought_steps=state.thought_steps,
        last_turn_tokens=turn_tokens,
        loop_count=state.loop_count + 1,
        final_response=state.final_response
    )

    if not calling_tools:
        message = {
            "role": role,
            "content": accumulated_content
        }

        final_step = ThoughtStep(
            loop_index=new_state.loop_count - 1,
            token=turn_tokens,
            thought=accumulated_content,
            tool_calls=[]
        )

        final_state = AgentState(
            question=new_state.question,
            messages=new_state.messages + [message],
            thought_steps=new_state.thought_steps + [final_step],
            last_turn_tokens=new_state.last_turn_tokens,
            loop_count=new_state.loop_count,
            final_response=accumulated_content or "",
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

    message = {
        "role": role,
        "content": accumulated_content,
        "tool_calls": tool_calls_payload
    }

    new_state.messages = new_state.messages + [message]

    yield NodeTransition(state=new_state, next_node=Node.EXECUTE)
    return


async def execute_node(state: AgentState, db: AsyncSession, session_id: uuid.UUID) -> AsyncGenerator[str | NodeTransition, None, None]:
    """Execute requested tool calls from the last thought turn and record thought steps.

    Args:
        state (AgentState): Current graph state containing the tool call message.
        db (AsyncSession): Database session required for tool executions and memory logging.
        session_id (uuid.UUID): Target chat session UUID.

    Yields:
        AsyncGenerator[str | NodeTransition, None]: SSE event stream tokens or NodeTransition signal.
    """
    tc_messages = state.messages[-1]
    tool_calls_detail: list[ToolCallDetail] = []
    new_messages: list[dict] = list(state.messages)

    tool_calls = tc_messages.get("tool_calls", []) if isinstance(tc_messages, dict) else (tc_messages.tool_calls or [])

    await add_chat_message(
        db=db,
        session_id=session_id,
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
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content={"tool_result": str(result)}
        )

        new_messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": str(result)
        })

    thought_content = tc_messages.get("content") if isinstance(tc_messages, dict) else tc_messages.content

    new_thought_step = ThoughtStep(
        loop_index=state.loop_count - 1,
        token=state.last_turn_tokens,
        thought=thought_content,
        tool_calls=tool_calls_detail
    )

    await add_chat_message(
        db=db,
        session_id=session_id,
        role=MessageRole.ASSISTANT,
        type=MessageType.THINKING,
        content={"thought": new_thought_step.model_dump()}
    )

    new_state = AgentState(
        question=state.question,
        messages=new_messages,
        thought_steps=state.thought_steps + [new_thought_step],
        loop_count=state.loop_count,
        final_response=None
    )

    yield f"event: thought\ndata: {new_thought_step.model_dump_json()}\n\n"
    yield NodeTransition(state=new_state, next_node=Node.THINK)
    return