import uuid
import logging
from datetime import datetime
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import ChatHistory, Session
from app.schemas import MessageRole, MessageType
from app.services.agent.memory.long_term import Memory

logger = logging.getLogger(__name__)

DEFAULT_TITLES = {"New Conversation", "New Chat", "Cuộc trò chuyện mới"}

async def get_session(db: AsyncSession, session_id: uuid.UUID) -> Session | None:
    """Retrieve chat session by ID."""
    session = await db.scalar(select(Session).where(Session.id == session_id))
    return session


async def create_session(db: AsyncSession, title: str) -> Session:
    """Create a new chat session."""
    try:
        session = Session(title=title, is_auto_titled=False)
        db.add(session)
        await db.commit()
        await db.refresh(session)
        return session
    except Exception as e:
        await db.rollback()
        raise e


async def get_session_messages(db: AsyncSession, session_id: uuid.UUID, limit: int = 20) -> list[ChatHistory]:
    """Retrieve recent user/assistant chat messages for a session in chronological order.

    Filters history entries to include only standard chat messages (excluding raw tool call records)
    and sorts them from oldest to newest for LLM conversation context.

    Args:
        db (AsyncSession): Active database session.
        session_id (uuid.UUID): Target chat session UUID.
        limit (int, optional): Maximum number of recent messages to fetch. Defaults to 20.

    Returns:
        list[ChatHistory]: Chronologically ordered list of ChatHistory instances.

    Raises:
        Exception: Rolls back database transaction and re-raises any database error.
    """
    try:
        logger.debug(f"[Session {session_id}] Fetching recent {limit} chat messages.")
        chat_histories = (
            await db.scalars(
                select(ChatHistory)
                .where(ChatHistory.session_id == session_id)
                .where(ChatHistory.type == "message")
                .order_by(ChatHistory.created_at.desc())
                .limit(limit)
            )
        ).all()

        return list(reversed(chat_histories))
    except Exception as e:
        logger.error(f"[Session {session_id}] Failed to retrieve session messages: {e}")
        await db.rollback()
        raise e


async def _maybe_auto_title(
    session: Session,
    user_text: str,
    update_values: dict
):
    """Automatically generate session title from initial user message."""
    if session.title in DEFAULT_TITLES and user_text:
        clean_title = user_text.strip().replace("\n", " ")
        auto_title = clean_title[:35] + ("..." if len(clean_title) > 35 else "")
        update_values["title"] = auto_title
        update_values["is_auto_titled"] = True


async def add_chat_message(
    db: AsyncSession,
    session_id: uuid.UUID,
    document_id: uuid.UUID | None,
    role: MessageRole,
    type: MessageType,
    content: dict,
    token_count: int = 0
) -> ChatHistory:
    """Add a new chat record to session history and update session metadata.

    Args:
        db (AsyncSession): Active database session.
        session_id (uuid.UUID): Target chat session UUID.
        role (MessageRole): Sender role (USER, ASSISTANT, TOOL).
        type (MessageType): Message category (MESSAGE, THINKING, TOOL_CALL, TOOL_RESULT).
        content (dict): Message content dictionary payload.
        token_count (int, optional): Associated token count. Defaults to 0.

    Returns:
        ChatHistory: Persisted ChatHistory model instance.

    Raises:
        Exception: Rolls back database transaction and re-raises any database error.
    """
    try:
        logger.debug(f"[Session {session_id}] Adding chat record: role={role}, type={type}")
        chat_history = ChatHistory(
            session_id=session_id,
            document_id=document_id,
            role=role,
            type=type,
            content=content,
            token_count=token_count
        )

        db.add(chat_history)

        session = await get_session(db, session_id)
        if session:
            update_values = {
                "message_count": Session.message_count + 1,
                "updated_at": datetime.now()
            }

            if role == MessageRole.USER:
                user_text = content.get("text", "") if isinstance(content, dict) else str(content)
                await _maybe_auto_title(session, user_text, update_values)

            await db.execute(
                update(Session)
                .where(Session.id == session.id)
                .values(**update_values)
            )
        
        await db.commit()
        await db.refresh(chat_history)

        return chat_history
    except Exception as e:
        logger.error(f"[Session {session_id}] Failed to add chat message: {e}")
        await db.rollback()
        raise e


async def update_session_title(db: AsyncSession, session_id: uuid.UUID, title: str) -> Session | None:
    """Update the title of a specific chat session."""
    try:
        session = await get_session(db, session_id)
        if session:
            session.title = title
            session.is_auto_titled = True
            session.updated_at = datetime.now()
            await db.commit()
            await db.refresh(session)
            return session
        return None
    except Exception as e:
        await db.rollback()
        raise e


async def delete_session(db: AsyncSession, session_id: uuid.UUID, memory: Memory) -> bool:
    """Delete a chat session by ID with consolidation."""
    await memory.force_consolidation(session_id)

    try:
        session = await get_session(db, session_id)
        if session:
            await db.delete(session)
            await db.commit()
            return True
        return False
    except Exception as e:
        await db.rollback()
        raise e