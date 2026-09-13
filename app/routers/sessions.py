import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Depends, HTTPException
from app.models import Session
from app.database import get_db
from app.schemas import ChatMessage, SessionCreate, SessionResponse
from app.services import memory


session_router = APIRouter(prefix="/session")

@session_router.post("/", response_model=SessionResponse)
async def create_session(
    payload: SessionCreate, 
    db: AsyncSession = Depends(get_db)
):
    """Create a new chat session.

    Args:
        payload (SessionCreate): Payload containing session title.
        db (AsyncSession): Database session.

    Returns:
        SessionResponse: Newly created session object.
    """
    try:
        new_session = await memory.create_session(
            db=db,
            title=payload.title
        )
        return new_session
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@session_router.get("/", response_model=list[SessionResponse])
async def get_sessions(
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve recent chat sessions sorted by updated timestamp in descending order.

    Args:
        limit (int, optional): Maximum number of sessions to retrieve. Defaults to 20.
        db (AsyncSession): Database session.

    Returns:
        list[SessionResponse]: List of session objects.
    """
    sessions = (
        await db.scalars(
            select(Session)
            .order_by(Session.updated_at.desc())
            .limit(limit)
        )
    ).all()

    return list(sessions)

@session_router.get("/{session_id}/messages/", response_model=list[ChatMessage])
async def get_messages(
    session_id: uuid.UUID,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve chat message history for a given session.

    Args:
        session_id (uuid.UUID): Target session UUID.
        limit (int, optional): Maximum number of messages to fetch. Defaults to 20.
        db (AsyncSession): Database session.

    Returns:
        list[ChatMessage]: List of historical chat messages.
    """
    messages = await memory.get_session_messages(
        db=db,
        session_id=session_id,
        limit=limit
    )

    return messages

@session_router.delete("/{session_id}/")
async def delete_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Delete a chat session and all associated messages.

    Args:
        session_id (uuid.UUID): UUID of session to delete.
        db (AsyncSession): Database session.

    Returns:
        dict: Deletion status object.
    """
    deleted = await memory.delete_session(
        db=db,
        session_id=session_id
    )
    return {
        "deleted": deleted
    }