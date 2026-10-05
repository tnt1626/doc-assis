import uuid
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks
from app import config
from app.services.rag.generator import generate
from app.services.rag.embedder import embed_batch
from app.models import Document, Chunk, FileStatus
from app.services.rag.chunker import chunk_by_sentences
from app.services.extractor import extract_document_content
from app.database import get_db, SessionLocal, update_doc_status
from app.services.llm.clients import OllamaConnectionError, OllamaModelNotFound
from app.schemas import DocumentUploadResponse, DocumentListItem, QueryRequest, QueryResponse

CONTENT_TYPES = [
    "text/html",
    "text/css",
    "text/csv",
    "text/xml",
    "text/plain",
    "text/markdown",
    "application/json",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/docx",
    "application/msword",
]

logger = logging.getLogger(__name__)

doc_router = APIRouter(prefix="/documents", tags=["documents"])

async def upload(doc_id: uuid.UUID, content: str):
    """Background task to chunk document content, compute embeddings in batches, and store chunks in the database.

    Args:
        doc_id (uuid.UUID): ID of the document being processed.
        content (str): Full text content of the document.
    """
    async with SessionLocal() as session:
        await update_doc_status(doc_id, FileStatus.PROCESSING, session)
        try:
            chunks = chunk_by_sentences(content, config.MAX_CHARS, config.OVERLAP_SENTENCES, config.MIN_CHARS)
            embeds = await embed_batch(chunks, config.BATCH_SIZE)

            for idx, embed in enumerate(embeds):
                session.add(Chunk(
                    document_id=doc_id,
                    content=chunks[idx],
                    embedding=embed,
                    chunk_index=idx
                ))

            await session.commit()
            await update_doc_status(doc_id, FileStatus.COMPLETED, session)
        except Exception as e:
            await session.rollback()
            await update_doc_status(doc_id, FileStatus.FAILED, session)
            logger.error(f"Failed to process document {doc_id}: {e}")


@doc_router.post("/", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a text document and process it for Q&A."""
    filename_lower = (file.filename or "").lower()
    is_supported_type = (
        file.content_type in CONTENT_TYPES or
        filename_lower.endswith((".pdf", ".docx", ".doc", ".txt", ".md", ".json", ".csv", ".html"))
    )
    if not is_supported_type:
        raise HTTPException(
            status_code=400,
            detail=f"Format as {file.content_type} is not supported"
        )
    
    content = await file.read()
    is_docling_file = (
        file.content_type in [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/docx",
            "application/msword",
        ] or filename_lower.endswith((".pdf", ".docx", ".doc"))
    )

    if is_docling_file:
        try:
            text = extract_document_content(content, file.filename or "document.pdf")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        try:
            text = content.decode("utf-8")
            if not text.strip():
                raise HTTPException(status_code=400, detail="The text file is empty.")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="The text file must be in UTF-8 encoding format.")

    try:
        doc = Document(filename=file.filename, content=text)
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        logger.info(f"Saved metadata for document '{file.filename}' (ID: {doc.id}). Starting background chunking & embedding.")
    except Exception:
        await db.rollback()
        logger.exception(f"Failed to save metadata for document '{file.filename}'")
        raise HTTPException(status_code=500, detail="Failed to save document")

    background_tasks.add_task(
        upload,
        doc_id=doc.id,
        content=text,
    )
    return DocumentUploadResponse.model_validate(doc)


@doc_router.post("/{document_id}/query", response_model=QueryResponse)
async def query_document(
    request: QueryRequest,
    document_id: uuid.UUID,
    top_k: int = 5,
    db: AsyncSession = Depends(get_db),
):
    """Query a document using natural language and get an AI-generated answer."""
    logger.info(f"Querying document {document_id} with question: '{request.question}'")
    doc = await db.scalar(select(Document).where(Document.id == document_id))
    if not doc:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    if doc.status != FileStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Document is not ready for querying. Current status: {doc.status}"
        )

    try:
        response = await generate(
            question=request.question,
            document_id=document_id,
            top_k=top_k,
            db=db
        )
        logger.info(f"Successfully generated answer for document {document_id}")

        return QueryResponse(**response)
    except OllamaConnectionError as e:
        logger.error(f"Ollama connection error for document {document_id}: {e}")
        raise HTTPException(status_code=503, detail="AI service unavailable")
    except OllamaModelNotFound:
        raise HTTPException(status_code=500, detail="Model not configured")
    

@doc_router.get("/", response_model=list[DocumentListItem])
async def list_documents(
    offset: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """Query all documents"""
    items = (
        await db.scalars(
            select(Document)
            .offset(offset)
            .limit(limit)
        )
    ).all()

    items = [DocumentListItem.model_validate(item) for item in items]
    return items

@doc_router.delete("/{id}/", response_model=dict)
async def delete_document(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete document with id if existed"""
    item = (
        await db.scalar(
            select(Document)
            .where(Document.id == id)
        )
    )

    if not item:
        raise HTTPException(
            status_code=404,
            detail="Not found document"
        )

    await db.delete(item)
    await db.commit()

    logger.info(f"Document with ID {id} has been permanently deleted from database.")
    return {
        "deleted": True
    }