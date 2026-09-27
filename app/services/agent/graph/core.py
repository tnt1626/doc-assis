import uuid
import json
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from langgraph.graph import StateGraph, START, END
from app.models import ChatHistory
from app.services.agent.graph.state import AgentGraphState
from app.services.agent.graph.nodes import execute_node, think_node
from app.schemas import MessageRole, MessageType, NodeTransition, Node
from app.services.agent.memory.short_term import add_chat_message, get_session_messages

logger = logging.getLogger(__name__)

def route_next(state: AgentGraphState) -> str:
    """Determine the next state transition in LangGraph."""
    if state.get("next_node") == "execute":
        return "execute"
    return END

def create_agent_state_graph():
    """Build and compile the LangGraph StateGraph workflow."""
    workflow = StateGraph(AgentGraphState)
    workflow.add_node("think", think_node)
    workflow.add_node("execute", execute_node)
    
    workflow.add_edge(START, "think")
    workflow.add_conditional_edges(
        "think",
        route_next,
        {"execute": "execute", END: END}
    )
    workflow.add_edge("execute", "think")
    
    return workflow.compile()

class AgentGraph:
    """LangGraph-powered Agent runner controlling state transitions between THINK and EXECUTE nodes."""

    def __init__(self, max_loops: int = 8):
        """Initialize the LangGraph Agent executor.

        Args:
            max_loops (int, optional): Maximum loop iterations permitted. Defaults to 8.
        """
        self.max_loops = max_loops
        self.graph = create_agent_state_graph()

    async def run(
        self,
        question: str,
        session_id: uuid.UUID,
        document_id: uuid.UUID | None,
        db: AsyncSession,
        limit: int = 20
    ) -> AsyncGenerator[str, None]:
        """Execute the agent graph state machine asynchronously.

        Args:
            question (str): User query/question.
            session_id (uuid.UUID): Target session UUID.
            document_id (uuid.UUID | None): Optional specific document ID context.
            db (AsyncSession): Database session for tool execution and memory storage.
            limit (int, optional): Maximum historical messages to retrieve. Defaults to 20.

        Yields:
            str: SSE formatted event strings.
        """
        historical_records = await get_session_messages(
            db=db,
            session_id=session_id,
            limit=limit
        )

        await add_chat_message(
            db=db,
            session_id=session_id,
            role=MessageRole.USER,
            type=MessageType.MESSAGE,
            content={"text": question}
        )

        initial_messages = self._build_init_message(
            question=question,
            chat_history=historical_records,
            document_id=document_id
        )

        state: AgentGraphState = {
            "question": question,
            "session_id": session_id,
            "document_id": document_id,
            "messages": initial_messages,
            "loop_count": 0,
            "last_turn_tokens": 0,
            "thought_steps": [],
            "final_response": None,
            "next_node": "think",
            "db": db,
        }

        current_node = Node.THINK

        while current_node != Node.END:
            if state.get("loop_count", 0) >= self.max_loops:
                yield f"event: error\ndata: {json.dumps({'detail': f'Reached maximum tool loops ({self.max_loops}) without answer.'})}\n\n"
                yield f"event: done\ndata: {json.dumps({'status': 'max_loops_exceeded'})}\n\n"
                return

            if current_node == Node.THINK:
                async for item in think_node(state, db, session_id):
                    if isinstance(item, str):
                        yield item
                    elif isinstance(item, NodeTransition):
                        state = item.state if isinstance(item.state, dict) else state
                        current_node = item.next_node

            elif current_node == Node.EXECUTE:
                async for item in execute_node(state, db, session_id):
                    if isinstance(item, str):
                        yield item
                    elif isinstance(item, NodeTransition):
                        state = item.state if isinstance(item.state, dict) else state
                        current_node = item.next_node

        final_response = state.get("final_response")
        if final_response:
            await add_chat_message(
                db=db,
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                type=MessageType.MESSAGE,
                content={"text": final_response},
                token_count=state.get("last_turn_tokens", 0)
            )

        return

    def _build_init_message(
        self,
        question: str,
        chat_history: list[ChatHistory] | None,
        document_id: uuid.UUID | None
    ) -> list[dict]:
        """Construct system prompt and initial message array for the agent graph state.

        Args:
            question (str): User question.
            chat_history (list[ChatHistory] | None): Preceding chat messages retrieved from database.
            document_id (uuid.UUID | None): Target document UUID if scoped.

        Returns:
            list[dict]: List of formatted message dictionaries.
        """
        system_parts = [
            "You are an intelligent assistant that can search for and read document content.",
            "When the user asks about document content, use the tool to search before answering.",
            "Respond in English, concisely, and base your answer on the information found.",
            "If no relevant information is found, state that clearly.",
        ]
    
        if document_id:
            system_parts.append(
                f"\nThe user is asking about the document with ID: {document_id}. "
                f"Prioritize using the search_document tool with this document_id."
            )
        
        system_prompt = "\n".join(system_parts)
    
        messages = [
            {"role": "system", "content": system_prompt},
        ]
    
        if chat_history:
            for chat in chat_history:
                if chat.type == MessageType.MESSAGE or chat.type == "message":
                    text_content = chat.content.get("text", "") if isinstance(chat.content, dict) else str(chat.content)
                    if text_content:
                        messages.append({"role": chat.role, "content": text_content})

        messages.append({"role": "user", "content": question})

        return messages