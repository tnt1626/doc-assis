import logging
from dataclasses import dataclass
from typing import Optional, Any
from app.config import (
    CRAG_STRIP_THRESHOLD,
    CRAG_PRESERVE_NEIGHBORS,
    CRAG_MAX_REFINED_CHARS,
)
from app.services.rag.chunker import split_into_sentences
from app.services.rag.evaluator import retrieval_evaluator

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeStrip:
    """A fine-grained knowledge strip with provenance metadata."""
    text: str
    doc_id: Optional[str]
    filename: str
    chunk_id: Optional[str]
    strip_index: int
    char_start: int
    char_end: int
    score: float = 0.0


@dataclass
class RefinementResult:
    """Result of Stage B knowledge refinement."""
    refined_context: str
    selected_strips: list[KnowledgeStrip]
    total_strips_evaluated: int
    has_evidence: bool
    original_char_count: int
    refined_char_count: int


def decompose_chunk(
    chunk_content: str,
    doc_id: Optional[str],
    filename: str,
    chunk_id: Optional[str],
    sentences_per_strip: int = 2
) -> list[KnowledgeStrip]:
    """Decompose a raw document chunk into small knowledge strips (1-2 sentences)."""
    sentences = split_into_sentences(chunk_content)
    if not sentences:
        return []

    strips: list[KnowledgeStrip] = []
    current_char = 0
    strip_idx = 0

    for i in range(0, len(sentences), sentences_per_strip):
        group = sentences[i : i + sentences_per_strip]
        strip_text = " ".join(group).strip()
        start_char = chunk_content.find(group[0], current_char)
        if start_char == -1:
            start_char = current_char
        end_char = start_char + len(strip_text)
        current_char = end_char

        strips.append(
            KnowledgeStrip(
                text=strip_text,
                doc_id=doc_id,
                filename=filename,
                chunk_id=chunk_id,
                strip_index=strip_idx,
                char_start=start_char,
                char_end=end_char,
                score=0.0
            )
        )
        strip_idx += 1

    return strips


class KnowledgeRefiner:
    """Stage B: Decompose-then-Recompose Knowledge Refiner.
    
    Filters out noisy, irrelevant sentences from retrieved chunks while preserving
    provenance and context flow.
    """

    def __init__(self, evaluator=retrieval_evaluator):
        self.evaluator = evaluator

    def refine(
        self,
        query: str,
        retrieved_items: list[tuple[Any, str, float]],
        strip_threshold: float = CRAG_STRIP_THRESHOLD,
        preserve_neighbors: bool = CRAG_PRESERVE_NEIGHBORS,
        max_chars: int = CRAG_MAX_REFINED_CHARS,
    ) -> RefinementResult:
        """Execute Stage B Knowledge Refinement on retrieved chunks.
        
        Args:
            query: User's query.
            retrieved_items: List of (chunk_obj, filename, score/dist) tuples.
            strip_threshold: Minimum cross-encoder score to keep a strip.
            preserve_neighbors: Whether to retain adjacent strips for context.
            max_chars: Budget cap for final refined context.
            
        Returns:
            RefinementResult containing refined text, strips, and metadata.
        """
        if not retrieved_items:
            return RefinementResult(
                refined_context="",
                selected_strips=[],
                total_strips_evaluated=0,
                has_evidence=False,
                original_char_count=0,
                refined_char_count=0
            )

        all_strips: list[KnowledgeStrip] = []
        original_char_count = 0

        # 1. Decompose all chunks into strips
        for item in retrieved_items:
            chunk = item[0]
            filename = item[1]
            content = getattr(chunk, "content", str(chunk))
            chunk_id = str(getattr(chunk, "id", ""))
            doc_id = str(getattr(chunk, "document_id", ""))
            original_char_count += len(content)

            chunk_strips = decompose_chunk(
                chunk_content=content,
                doc_id=doc_id,
                filename=filename,
                chunk_id=chunk_id
            )
            all_strips.extend(chunk_strips)

        if not all_strips:
            return RefinementResult(
                refined_context="",
                selected_strips=[],
                total_strips_evaluated=0,
                has_evidence=False,
                original_char_count=original_char_count,
                refined_char_count=0
            )

        # 2. Batch score strips with CPU cross-encoder
        strip_texts = [s.text for s in all_strips]
        scores = self.evaluator.evaluate_strips(query, strip_texts)
        for s, score in zip(all_strips, scores):
            s.score = score

        # 3. Filter relevant strips
        retained_indices = set()
        for idx, s in enumerate(all_strips):
            if s.score >= strip_threshold:
                retained_indices.add(idx)
                if preserve_neighbors:
                    # Keep previous and next strip if from same chunk
                    if idx > 0 and all_strips[idx - 1].chunk_id == s.chunk_id:
                        retained_indices.add(idx - 1)
                    if idx + 1 < len(all_strips) and all_strips[idx + 1].chunk_id == s.chunk_id:
                        retained_indices.add(idx + 1)

        if not retained_indices:
            logger.info("Stage B Refinement: No strips met CRAG_STRIP_THRESHOLD. Yielding NO_EVIDENCE.")
            return RefinementResult(
                refined_context="",
                selected_strips=[],
                total_strips_evaluated=len(all_strips),
                has_evidence=False,
                original_char_count=original_char_count,
                refined_char_count=0
            )

        # 4. Recompose in original appearance order
        sorted_indices = sorted(retained_indices)
        selected_strips: list[KnowledgeStrip] = []
        current_chars = 0
        context_blocks: list[str] = []

        for idx in sorted_indices:
            strip = all_strips[idx]
            block = f"[{strip.filename}]: {strip.text}"
            block_len = len(block) + 1
            if current_chars + block_len > max_chars:
                logger.debug(f"Reached max_chars budget ({max_chars}). Stopping recomposition.")
                break
            selected_strips.append(strip)
            context_blocks.append(block)
            current_chars += block_len

        refined_context = "\n".join(context_blocks)
        refined_char_count = len(refined_context)

        logger.debug(
            f"Stage B Refinement: {len(all_strips)} strips -> {len(selected_strips)} selected "
            f"({original_char_count} -> {refined_char_count} chars)"
        )

        return RefinementResult(
            refined_context=refined_context,
            selected_strips=selected_strips,
            total_strips_evaluated=len(all_strips),
            has_evidence=len(selected_strips) > 0,
            original_char_count=original_char_count,
            refined_char_count=refined_char_count
        )


# Global singleton instance
knowledge_refiner = KnowledgeRefiner()

