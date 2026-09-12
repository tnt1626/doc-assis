import uuid
from enum import Enum
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import Integer, DateTime, Text, UUID, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector


class FileStatus(str, Enum):
    """Status enum for tracking document processing pipeline."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Base(DeclarativeBase):
    pass

class Document(Base):
    """SQLAlchemy model representing an uploaded document and its processing metadata."""
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    status: Mapped[FileStatus] = mapped_column(Text, nullable=False, default=FileStatus.PENDING)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID, ForeignKey("documents.id"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Session(Base):
    """SQLAlchemy model representing a chat session."""
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    messages: Mapped[list["ChatHistory"]] = relationship("ChatHistory", back_populates="session", cascade="all, delete-orphan")


class ChatHistory(Base):
    """SQLAlchemy model representing individual messages within a chat session."""
    __tablename__ = "chat_history"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[dict] = mapped_column(JSONB, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped["Session"] = relationship("Session", back_populates="messages")