import uuid
import json
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from langgraph.graph import StateGraph, START, END
from app.models import ChatHistory
from app.config import AGENT_DIR, SOUL_FILE
from app.services.client import GROQ_SMALL_MODEL, groq_client
from app.services.agent.memory.long_term import Memory
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
        memory = Memory(
            db=db,
            client=groq_client,
            small_model=GROQ_SMALL_MODEL,
            agent_dir=AGENT_DIR
        )

        mem_context = await memory.before_run(
            message=question,
            doc_id=document_id
        )

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
            mem_context=mem_context,
            document_id=document_id
        )


        state: AgentGraphState = {
            "question"          : question,
            "session_id"        : session_id,
            "document_id"       : document_id,
            "doc_ids_used"      : [document_id] if document_id else [],
            "messages"          : initial_messages,
            "loop_count"        : 0,
            "last_turn_tokens"  : 0,
            "thought_steps"     : [],
            "final_response"    : None,
            "next_node"         : "think",
            "db"                : db,
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

        final_doc_ids = state.get("doc_ids_used", [])
        await memory.after_run(
            session_id=session_id,
            doc_ids_used=final_doc_ids
        )

        return

    def _build_init_message(
        self,
        question: str,
        chat_history: list[ChatHistory] | None,
        mem_context: dict,
        document_id: uuid.UUID | None
    ) -> list[dict[str, str]]:
        """Construct system prompt and initial message array for the agent graph state.

        Args:
            question (str): User question.
            chat_history (list[ChatHistory] | None): Preceding chat messages retrieved from database.
            document_id (uuid.UUID | None): Target document UUID if scoped.

        Returns:
            list[dict]: List of formatted message dictionaries.
        """
        personality = self._get_personality()
        sections = [personality]
        user_profile = mem_context.get("user_profile")
        doc_memory = mem_context.get("doc_memory")

        if user_profile:
            sections.append(user_profile)

        if doc_memory:
            sections.append(doc_memory)

        if document_id:
            sections.append((
                "## Active document\n"
                f"Document ID: {document_id}\n"
                "\n"
                "When the user asks questions without specifying a document, "
                "assume they are asking about this document.\n"
                "\n"
                "When calling search_document, use this document_id by default "
                "unless the user explicitly asks to search across all documents."
            ))
        
        system_prompt = "\n\n".join(sections)
    
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        last_conversation = self._build_last_conversation(chat_history)
        messages.extend(last_conversation)

        messages.append({"role": "user", "content": question})

        return messages


    def _build_last_conversation(self, chat_history: list[ChatHistory] | None) -> list[dict[str, str]]:
        """Build last conversation from chat history as a template that can feed to LLM"""
        if not chat_history:
            return []

        conversation = []
        for chat in chat_history:
            if chat.type == MessageType.MESSAGE or chat.type == "message":
                text_content = chat.content.get("text", "") if isinstance(chat.content, dict) else str(chat.content)
                if text_content:
                    conversation.append({
                        "role": chat.role,
                        "content": text_content
                    })

        return conversation


    def _get_personality(self) -> str:
        """Return content in SOULD.md as agent personality"""
        return (AGENT_DIR / SOUL_FILE).read_text(encoding='utf-8')