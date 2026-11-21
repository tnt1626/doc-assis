import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Any
from app.config import (
    CRAG_ENABLED,
    CRAG_RETRIEVAL_SCOPE,
    CRAG_EVALUATOR_MODEL,
    CRAG_TELEMETRY_ENABLED,
)
from app.services.rag.evaluator import ActionType, retrieval_evaluator
from app.services.rag.refiner import knowledge_refiner, RefinementResult
from app.services.rag.web_search import external_search_client

logger = logging.getLogger(__name__)


@dataclass
class CRAGTelemetry:
    """Experiment telemetry tracking metrics without sensitive document data."""
    rag_mode: str
    scope: str
    evaluator_model: str
    action_triggered: str
    document_max_score: float
    retrieved_chunks_count: int
    refined_strips_count: int
    original_context_chars: int
    refined_context_chars: int
    evaluator_latency_ms: float
    refiner_latency_ms: float
    web_latency_ms: float
    total_crag_latency_ms: float
    web_search_used: bool
    evidence_status: str


@dataclass
class CRAGResult:
    """Output of CRAG processing pipeline."""
    action: ActionType
    has_evidence: bool
    context: str
    sources: list[str]
    telemetry: Optional[CRAGTelemetry] = None


class CRAGEngine:
    """Corrective Retrieval Augmented Generation (CRAG) Engine.
    
    Coordinates Stage A (Document-Level Evaluation) and Stage B (Knowledge Refinement)
    along with safe fallback logic.
    """

    def __init__(
        self,
        evaluator=retrieval_evaluator,
        refiner=knowledge_refiner,
        search_client=external_search_client
    ):
        self.evaluator = evaluator
        self.refiner = refiner
        self.search_client = search_client

    def process(
        self,
        query: str,
        retrieved_items: list[tuple[Any, str, float]]
    ) -> CRAGResult:
        """Process retrieved chunks through CRAG pipeline.
        
        Args:
            query: User natural language question.
            retrieved_items: List of (Chunk, filename, distance/score) tuples.
            
        Returns:
            CRAGResult with verified context and metadata.
        """
        start_total = time.perf_counter()

        # If CRAG is disabled, bypass and construct raw context (Vanilla RAG behavior)
        if not CRAG_ENABLED:
            raw_blocks = []
            sources = []
            for item in retrieved_items:
                chunk = item[0]
                content = getattr(chunk, "content", str(chunk))
                raw_blocks.append(f"Content: {content}. Distance: {item[2]}")
                sources.append(content)
            
            return CRAGResult(
                action=ActionType.CORRECT if retrieved_items else ActionType.INCORRECT,
                has_evidence=len(retrieved_items) > 0,
                context="\n".join(raw_blocks),
                sources=sources,
                telemetry=None
            )

        # -------------------------------------------------------------
        # STAGE A: Document-Level Retrieval Evaluation
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        doc_texts = [
            getattr(item[0], "content", str(item[0]))
            for item in retrieved_items
        ]
        action, max_score, doc_scores = self.evaluator.evaluate_documents(query, doc_texts)
        eval_ms = (time.perf_counter() - t0) * 1000.0

        refine_ms = 0.0
        web_ms = 0.0
        web_search_used = False
        final_context_blocks: list[str] = []
        sources: list[str] = []
        selected_strips_count = 0
        original_chars = sum(len(t) for t in doc_texts)
        refined_chars = 0

        # -------------------------------------------------------------
        # STAGE B: Knowledge Refinement & Routing based on Action
        # -------------------------------------------------------------
        if action == ActionType.CORRECT:
            # High confidence in retrieved documents -> Refine internal knowledge
            t_refine = time.perf_counter()
            refinement = self.refiner.refine(query, retrieved_items)
            refine_ms = (time.perf_counter() - t_refine) * 1000.0

            if refinement.has_evidence:
                final_context_blocks.append(refinement.refined_context)
                sources.extend([s.text for s in refinement.selected_strips])
                selected_strips_count = len(refinement.selected_strips)
                refined_chars = refinement.refined_char_count

        elif action == ActionType.AMBIGUOUS:
            # Medium confidence -> Combine internal refinement + optional web search
            t_refine = time.perf_counter()
            refinement = self.refiner.refine(query, retrieved_items)
            refine_ms = (time.perf_counter() - t_refine) * 1000.0

            if refinement.has_evidence:
                final_context_blocks.append(
                    "--- INTERNAL EVIDENCE (DOCUMENTS) ---\n" + refinement.refined_context
                )
                sources.extend([s.text for s in refinement.selected_strips])
                selected_strips_count += len(refinement.selected_strips)
                refined_chars += refinement.refined_char_count

            # If WEB_ALLOWED, complement with web search
            if self.search_client.is_web_allowed():
                t_web = time.perf_counter()
                web_refinement = self.search_client.search_and_refine(query)
                web_ms = (time.perf_counter() - t_web) * 1000.0
                web_search_used = True

                if web_refinement.has_evidence:
                    final_context_blocks.append(web_refinement.refined_context)
                    sources.extend([s.text for s in web_refinement.selected_strips])
                    selected_strips_count += len(web_refinement.selected_strips)
                    refined_chars += web_refinement.refined_char_count

        elif action == ActionType.INCORRECT:
            # Low confidence in retrieved documents -> Discard retrieved chunks
            logger.info("CRAG Action INCORRECT: Discarding irrelevant retrieved documents.")
            if self.search_client.is_web_allowed():
                t_web = time.perf_counter()
                web_refinement = self.search_client.search_and_refine(query)
                web_ms = (time.perf_counter() - t_web) * 1000.0
                web_search_used = True

                if web_refinement.has_evidence:
                    final_context_blocks.append(web_refinement.refined_context)
                    sources.extend([s.text for s in web_refinement.selected_strips])
                    selected_strips_count += len(web_refinement.selected_strips)
                    refined_chars += web_refinement.refined_char_count

        total_crag_ms = (time.perf_counter() - start_total) * 1000.0
        final_context = "\n\n".join(final_context_blocks).strip()
        has_evidence = bool(final_context)

        telemetry = None
        if CRAG_TELEMETRY_ENABLED:
            telemetry = CRAGTelemetry(
                rag_mode="crag",
                scope=CRAG_RETRIEVAL_SCOPE,
                evaluator_model=CRAG_EVALUATOR_MODEL,
                action_triggered=action.value,
                document_max_score=round(max_score, 4),
                retrieved_chunks_count=len(retrieved_items),
                refined_strips_count=selected_strips_count,
                original_context_chars=original_chars,
                refined_context_chars=refined_chars,
                evaluator_latency_ms=round(eval_ms, 2),
                refiner_latency_ms=round(refine_ms, 2),
                web_latency_ms=round(web_ms, 2),
                total_crag_latency_ms=round(total_crag_ms, 2),
                web_search_used=web_search_used,
                evidence_status="VALID_EVIDENCE" if has_evidence else "NO_EVIDENCE"
            )
            logger.info(f"CRAG Telemetry: {asdict(telemetry)}")

        return CRAGResult(
            action=action,
            has_evidence=has_evidence,
            context=final_context,
            sources=sources,
            telemetry=telemetry
        )


# Global singleton instance
crag_engine = CRAGEngine()

