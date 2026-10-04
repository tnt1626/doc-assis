import uuid
from typing import TypedDict, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas import ThoughtStep
from app.services.llm.llm_types import Usage

class AgentGraphState(TypedDict, total=False):
    """LangGraph State definition for Document QA Agent."""
    question        : str
    session_id      : uuid.UUID
    document_id     : Optional[uuid.UUID]
    messages        : List[dict]
    loop_count      : int
    last_turn_usage : Usage | None
    thought_steps   : List[ThoughtStep]
    final_response  : Optional[str]
    next_node       : str
    db              : AsyncSession
