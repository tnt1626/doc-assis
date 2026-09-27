import uuid
from datetime import datetime
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import ChatHistory, Session
from app.schemas import MessageRole, MessageType


DEFAULT_TITLES = {"New Conversation", "New Chat", "Cuộc trò chuyện mới"}

async def get_session(db: AsyncSession, session_id: uuid.UUID) -> Session | None:
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
    """Retrieve the recent chat messages for a specific session ordered by creation time."""
    try:
        chat_histories = (
            await db.scalars(
                select(ChatHistory)
                .where(ChatHistory.session_id == session_id)
                .order_by(ChatHistory.created_at.desc())
                .limit(limit)
            )
        ).all()

        return list(chat_histories)
    except Exception as e:
        await db.rollback()
        raise e


async def _maybe_auto_title(
    session: Session,
    user_text: str,
    update_values: dict
):
    """Auto title for a new session"""
    if session.title in DEFAULT_TITLES and user_text:
        clean_title = user_text.strip().replace("\n", " ")
        auto_title = clean_title[:35] + ("..." if len(clean_title) > 35 else "")
        update_values["title"] = auto_title
        update_values["is_auto_titled"] = True


async def add_chat_message(
    db: AsyncSession,
    session_id: uuid.UUID,
    role: MessageRole,
    type: MessageType,
    content: dict,
    token_count: int = 0
) -> ChatHistory:
    """Add a new chat message to a session and update session metadata."""
    try:
        chat_history = ChatHistory(
            session_id=session_id,
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


async def delete_session(db: AsyncSession, session_id: uuid.UUID) -> bool:
    """Delete a chat session by ID."""
    try:
        session = await get_session(db, session_id)
        if session:
            session.is_deleted = True
            await db.commit()
            return True
        return False
    except Exception as e:
        await db.rollback()
        raise e