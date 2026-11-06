import uuid
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import THRESHOLD, CRAG_ENABLED
from app.models import Chunk, Document
from app.services.rag.embedder import embed_text

async def retrieve_hybrid(
    question: str,
    document_id: uuid.UUID | None,
    top_k: int,
    db: AsyncSession,
    rrf_k: int = 60
) -> list[tuple[Chunk, str, float]]:
    """Retrieve top-k relevant document chunks using Hybrid Search (Dense Vector + PostgreSQL tsvector Full-Text Search)
    fused with Reciprocal Rank Fusion (RRF).

    Args:
        question (str): User query string.
        document_id (uuid.UUID | None): Optional document ID filter. If None, searches across all chunks.
        top_k (int): Number of top matching chunks to retrieve.
        db (AsyncSession): Database session.
        rrf_k (int, optional): RRF constant parameter. Defaults to 60.

    Returns:
        list[tuple[Chunk, str, float]]: List of (Chunk, filename, distance/score) tuples.
    """
    # 1. Vector Search Candidates
    embed_vec = await embed_text(question)
    distance_col = Chunk.embedding.cosine_distance(embed_vec).label("distance")
    
    vec_query = (
        select(Chunk, Document.filename, distance_col)
        .join(Document, Document.id == Chunk.document_id)
        .where(distance_col < THRESHOLD)
    )
    if document_id:
        vec_query = vec_query.where(Chunk.document_id == document_id)

    vec_rows = (
        await db.execute(
            vec_query.order_by(distance_col).limit(top_k * 3)
        )
    ).all()

    # 2. Keyword Search Candidates (PostgreSQL tsvector)
    ts_query_expr = func.plainto_tsquery('simple', question)
    ts_vector_expr = func.to_tsvector('simple', Chunk.content)
    rank_col = func.ts_rank(ts_vector_expr, ts_query_expr).label("rank")

    kw_query = (
        select(Chunk, Document.filename, rank_col)
        .join(Document, Document.id == Chunk.document_id)
        .where(ts_vector_expr.op('@@')(ts_query_expr))
    )
    if document_id:
        kw_query = kw_query.where(Chunk.document_id == document_id)

    kw_rows = (
        await db.execute(
            kw_query.order_by(rank_col.desc()).limit(top_k * 3)
        )
    ).all()

    # 3. Reciprocal Rank Fusion (RRF)
    scores: dict[uuid.UUID, float] = {}
    chunk_map: dict[uuid.UUID, tuple[Chunk, str, float]] = {}

    for rank, row in enumerate(vec_rows, start=1):
        cid = row.Chunk.id
        chunk_map[cid] = (row.Chunk, row.filename, float(row.distance))
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    for rank, row in enumerate(kw_rows, start=1):
        cid = row.Chunk.id
        if cid not in chunk_map:
            chunk_map[cid] = (row.Chunk, row.filename, 1.0)
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

    results: list[tuple[Chunk, str, float]] = []
    for cid in sorted_cids:
        chunk, filename, dist = chunk_map[cid]
        results.append((chunk, filename, dist))

    # Fallback to standard vector query if no hybrid matches meet distance threshold
    # Note: When CRAG is enabled, we avoid blindly injecting distant chunks into the generator.
    if not results and not CRAG_ENABLED:
        fallback_query = (
            select(Chunk, Document.filename, distance_col)
            .join(Document, Document.id == Chunk.document_id)
        )
        if document_id:
            fallback_query = fallback_query.where(Chunk.document_id == document_id)
        fallback_rows = (
            await db.execute(fallback_query.order_by(distance_col).limit(top_k))
        ).all()
        for r in fallback_rows:
            results.append((r.Chunk, r.filename, float(r.distance)))

    return results

async def retrieve_vec(question: str, document_id: uuid.UUID | None, top_k: int, db: AsyncSession):
    """Retrieve top-k relevant document chunks using hybrid search."""
    return await retrieve_hybrid(question, document_id, top_k, db)