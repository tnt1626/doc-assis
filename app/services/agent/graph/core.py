import uuid
import json
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import ChatHistory
from app.services.memory import add_chat_message, get_session_messages
from app.services.agent.graph.nodes import execute_node, think_node
from app.schemas import (
    AgentState,
    MessageRole,
    MessageType, 
    Node, 
    NodeTransition
)

class AgentGraph:
    """Graph-based Agent runner controlling state transitions between THINK and EXECUTE nodes."""

    def __init__(self, max_loops: int = 8):
        """Initialize the AgentGraph executor.

        Args:
            max_loops (int, optional): Maximum loop iterations permitted. Defaults to 8.
        """
        self.max_loops = max_loops

    async def run(
        self,
        question: str,
        session_id: uuid.UUID,
        document_id: uuid.UUID | None,
        db: AsyncSession,
        limit: int = 20
    ):
        """Execute the agent graph state machine asynchronously.

        Args:
            question (str): User query/question.
            chat_history (list[ChatMessage] | None): Conversation history messages.
            document_id (uuid.UUID | None): Optional specific document ID context.
            db (AsyncSession): Database session for tool execution.

        Returns:
            AgentResponse: Final answer and tracked thought steps.
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

        state = AgentState(
            question=question,
            messages=initial_messages
        )

        current_node = Node.THINK

        while current_node != Node.END:
            if state.loop_count >= self.max_loops:
                yield f"event: error\ndata: {json.dumps({'detail': f'Reached maximum tool loops ({self.max_loops}) without answer.'})}\n\n"
                yield f"event: done\ndata: {json.dumps({'status': 'max_loops_exceeded'})}\n\n"
                return

            if current_node == Node.THINK:
                async for item in think_node(state, db, session_id):
                    if isinstance(item, str):
                        yield item
                    elif isinstance(item, NodeTransition):
                        state, current_node = item.state, item.next_node

            elif current_node == Node.EXECUTE:
                async for item in execute_node(state, db, session_id):
                    if isinstance(item, str):
                        yield item
                    elif isinstance(item, NodeTransition):
                        state, current_node = item.state, item.next_node

        if state.final_response:
            await add_chat_message(
                db=db,
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                type=MessageType.MESSAGE,
                content={"text": state.final_response},
                token_count=state.last_turn_tokens
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
            chat_history (list[ChatMessage] | None): Preceding chat messages.
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

async def main():
    from app.database import SessionLocal
    
    graph = AgentGraph()
    async with SessionLocal() as session:
        async for chunk in graph.run(
            question="What documents are available?",
            chat_history=None,
            document_id=None,
            db=session
        ):
            print(chunk, end="", flush=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())