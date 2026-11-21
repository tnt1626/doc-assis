import uuid
from typing import Optional
from sqlalchemy import select

from mcp.server.mcpserver import MCPServer
from app.config import PAGE_SIZE, CRAG_ENABLED
from app.database import SessionLocal
from app.models import Chunk, Document
from app.services.rag.retriever import retrieve_vec
from app.services.rag.crag_engine import crag_engine

mcp_server = MCPServer(
    name="document-qa-server",
    title="Document QA MCP Server",
    version="0.1.0",
    description="MCP Server providing document search, retrieval, listing, and summarization capabilities"
)

def _is_valid_uuid(id_str: str) -> bool:
    try:
        uuid.UUID(id_str)
        return True
    except (ValueError, TypeError):
        return False

@mcp_server.tool()
async def search_document(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 5
) -> str:
    """Search for relevant text chunks in a specific document or across all documents based on the query.

    Args:
        query: The query or search term to look for in the document.
        document_id: Optional UUID of the document to search within.
        top_k: The number of relevant text chunks to return. Defaults to 5.
    """
    async with SessionLocal() as db:
        doc_uuid = None
        doc = None

        if document_id:
            if not _is_valid_uuid(document_id):
                return f"Error: '{document_id}' is not a valid UUID."

            doc_uuid = uuid.UUID(document_id)
            doc = await db.scalar(select(Document).where(Document.id == doc_uuid))
            if not doc:
                return f"Error: Document with ID '{document_id}' not found."

        rows = await retrieve_vec(query, doc_uuid, top_k, db)
        if not rows:
            return "No relevant text chunks found for this query."

        if CRAG_ENABLED:
            crag_res = crag_engine.process(query, rows)
            if not crag_res.has_evidence:
                return "No relevant verified evidence found for this query in the specified document(s)."

            header = f"Evaluated retrieval confidence: {crag_res.action.value}."
            if doc_uuid and doc:
                header += f" Refined evidence from '{doc.filename}':"
            else:
                header += " Refined evidence:"
            return f"{header}\n\n{crag_res.context}"

        result_lines = []
        if doc_uuid and doc:
            result_lines.append(f"Found {len(rows)} relevant chunks in '{doc.filename}':\n")
            for i, (chunk, _, distance) in enumerate(rows, 1):
                result_lines.append(f"[Chunk {i} | Distance: {distance:.4f}]\n{chunk.content}\n")
        else:
            result_lines.append(f"Found {len(rows)} relevant chunks across all documents:\n")
            for i, (chunk, filename, distance) in enumerate(rows, 1):
                result_lines.append(
                    f"[Chunk {i}] (From Document: '{filename}' | ID: {chunk.document_id} | Distance: {distance:.4f})\n"
                    f"{chunk.content}\n"
                )

        return "\n".join(result_lines)

@mcp_server.tool()
async def list_documents(limit: int = 20) -> str:
    """List all documents that have been uploaded to the system.

    Args:
        limit: The maximum number of documents to return. Defaults to 20.
    """
    async with SessionLocal() as db:
        docs = (
            await db.scalars(
                select(Document)
                .order_by(Document.created_at.desc())
                .limit(limit)
            )
        ).all()

        if not docs:
            return "No documents have been uploaded to the system yet."

        lines = [f"The system has {len(docs)} documents:\n"]
        for doc in docs:
            lines.append(
                f"- ID: {doc.id}\n"
                f"  Filename: {doc.filename}\n"
                f"  Uploaded at: {doc.created_at.strftime('%Y-%m-%d %H:%M')}\n"
            )
        return "\n".join(lines)

@mcp_server.tool()
async def get_full_document(document_id: str, page: int = 1) -> str:
    """Retrieve the full content of a document page by page.

    Args:
        document_id: The UUID of the document to retrieve.
        page: Page number to read (each page is ~4000 chars). Defaults to 1.
    """
    async with SessionLocal() as db:
        if not _is_valid_uuid(document_id):
            return f"Error: '{document_id}' is not a valid UUID."

        doc_uuid = uuid.UUID(document_id)
        doc = await db.scalar(select(Document).where(Document.id == doc_uuid))
        if not doc:
            return f"Error: Document with ID '{document_id}' not found."

        start = (page - 1) * PAGE_SIZE
        end = start + PAGE_SIZE
        content_chunk = doc.content[start:end] if doc.content else ""
        total_pages = (len(doc.content) + PAGE_SIZE - 1) // PAGE_SIZE if doc.content else 1

        return (
            f"=== Document '{doc.filename}' (Page {page}/{total_pages}) ===\n\n"
            f"{content_chunk}\n\n"
            f"[System Note: Page {page} of {total_pages}. "
            f"{'Call get_full_document with page=' + str(page+1) + ' to read more.' if page < total_pages else 'End of document.'}]"
        )

@mcp_server.tool()
async def summarize_document(document_id: str) -> str:
    """Summarize a document including status, filename, created_at, updated_at and first chunk of content.

    Args:
        document_id: The UUID of the document to get summarization.
    """
    async with SessionLocal() as db:
        if not _is_valid_uuid(document_id):
            return f"Error: '{document_id}' is not a valid UUID."

        doc_uuid = uuid.UUID(document_id)
        doc = await db.scalar(select(Document).where(Document.id == doc_uuid))
        if not doc:
            return f"Error: Document with ID '{document_id}' not found."

        first_chunk = await db.scalar(
            select(Chunk)
            .where(
                Chunk.document_id == doc_uuid,
                Chunk.chunk_index == 0
            )
        )
        if not first_chunk:
            return f"Error: Content of document with ID '{doc_uuid}' has not been chunked yet"

        return (
            f"[Summarization of document with id={doc_uuid}]\n"
            f"Status: {doc.status}\n"
            f"Filename: {doc.filename}\n"
            f"Created at: {doc.created_at}\n"
            f"Updated at: {doc.updated_at}\n"
            f"Content of first chunk:\n{first_chunk.content}."
        )

@mcp_server.resource("document://{document_id}")
async def get_document_resource(document_id: str) -> str:
    """Read a document as an MCP Resource.

    Args:
        document_id: The UUID of the target document.
    """
    async with SessionLocal() as db:
        if not _is_valid_uuid(document_id):
            return f"Error: Invalid UUID '{document_id}'"
        doc_uuid = uuid.UUID(document_id)
        doc = await db.scalar(select(Document).where(Document.id == doc_uuid))
        if not doc:
            return f"Document '{document_id}' not found."
        return f"# {doc.filename}\n\n{doc.content}"
