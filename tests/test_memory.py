import uuid
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from app.models import ChatHistory
from app.schemas import MessageRole, MessageType
from app.services.agent.memory.short_term import (
    get_session_messages,
    add_chat_message,
    create_session,
    DEFAULT_TITLES,
)
from app.services.agent.memory.long_term import Memory


@pytest.mark.asyncio
async def test_get_session_messages_ordering_and_filtering():
    """Verify that get_session_messages filters by type='message' and returns in chronological order."""
    db = AsyncMock()
    session_id = uuid.uuid4()

    msg1 = MagicMock(spec=ChatHistory)
    msg1.type = "message"
    msg1.created_at = 100

    msg2 = MagicMock(spec=ChatHistory)
    msg2.type = "message"
    msg2.created_at = 200

    scalars_result = MagicMock()
    # Mocking order_by(desc) result from DB query: msg2 (newer), msg1 (older)
    scalars_result.all.return_value = [msg2, msg1]
    db.scalars.return_value = scalars_result

    result = await get_session_messages(db, session_id, limit=10)

    # Output should be reversed (chronological: oldest to newest)
    assert result == [msg1, msg2]


@pytest.mark.asyncio
async def test_memory_init_and_local_mem(tmp_path: Path):
    """Test Memory initialization, agent_dir creation, and default prompt files."""
    db = AsyncMock()
    llm = MagicMock()
    agent_dir = tmp_path / ".agent"

    memory = Memory(
        db=db,
        llm=llm,
        agent_dir=agent_dir
    )

    assert (agent_dir / "USER.md").exists()
    assert (agent_dir / "SOUL.md").exists()
    assert "No information about this user yet." in (agent_dir / "USER.md").read_text()
    assert "You are a research assistant" in (agent_dir / "SOUL.md").read_text()


@pytest.mark.asyncio
async def test_memory_before_run(tmp_path: Path):
    """Test Memory.before_run fetching user_profile and per_doc_memory."""
    db = AsyncMock()
    llm = MagicMock()
    agent_dir = tmp_path / ".agent"

    memory = Memory(
        db=db,
        llm=llm,
        agent_dir=agent_dir
    )

    # Set custom user profile
    (agent_dir / "USER.md").write_text("User prefers Python code examples.")

    doc_id = uuid.uuid4()
    mock_doc_mem = MagicMock()
    mock_doc_mem.content = "This document is about Machine Learning."
    db.scalar.return_value = mock_doc_mem

    res = await memory.before_run(
        message="Explain paper",
        doc_id=doc_id
    )

    assert "User prefers Python code examples." in res["user_profile"]
    assert "This document is about Machine Learning." in res["doc_memory"]
