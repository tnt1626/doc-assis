import re
import logging
from typing import Optional
from app.config import (
    CRAG_RETRIEVAL_SCOPE,
    CRAG_WEB_TIMEOUT,
    CRAG_WEB_MAX_RESULTS,
)
from app.services.rag.refiner import KnowledgeRefiner, RefinementResult

logger = logging.getLogger(__name__)


def rewrite_query_for_search(question: str) -> str:
    """Extract search engine keywords from a natural language question.
    
    Fast zero-overhead local keyword extraction removing question filler words.
    """
    cleaned = re.sub(r"[?!.,;:\"'()\[\]{}]", " ", question).strip()
    stopwords = {
        "what", "is", "the", "are", "how", "to", "in", "of", "and", "a", "an",
        "for", "which", "who", "whom", "where", "when", "why", "can", "could",
        "should", "would", "does", "do", "did", "tell", "me", "about", "give",
        "là", "gì", "như", "thế", "nào", "ở", "đâu", "khi", "tại", "sao",
        "hãy", "cho", "tôi", "biết", "về", "những", "cái", "các", "của"
    }
    words = cleaned.split()
    keywords = [w for w in words if w.lower() not in stopwords]
    return " ".join(keywords) if keywords else cleaned


class ExternalSearchClient:
    """Client for optional external search fallback in CRAG."""

    def __init__(self, refiner: Optional[KnowledgeRefiner] = None):
        self.refiner = refiner or KnowledgeRefiner()

    def is_web_allowed(self) -> bool:
        """Check if web search is permitted by system configuration."""
        return CRAG_RETRIEVAL_SCOPE.strip().upper() == "WEB_ALLOWED"

    def search(self, query: str) -> list[dict[str, str]]:
        """Perform zero-key web search using duckduckgo-search with strict timeout."""
        if not self.is_web_allowed():
            logger.debug("Web search skipped: CRAG_RETRIEVAL_SCOPE is not 'WEB_ALLOWED'.")
            return []

        search_query = rewrite_query_for_search(query)
        logger.info(f"Initiating web search with query: '{search_query}' (original: '{query}')")

        try:
            from duckduckgo_search import DDGS
            with DDGS(timeout=CRAG_WEB_TIMEOUT) as ddgs:
                results = list(
                    ddgs.text(
                        keywords=search_query,
                        max_results=CRAG_WEB_MAX_RESULTS
                    )
                )
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "body": r.get("body", "")
                    }
                    for r in results if r.get("body")
                ]
        except Exception as e:
            logger.warning(f"Web search attempt failed or timed out: {e}. Falling back safely.")
            return []

    def search_and_refine(self, query: str) -> RefinementResult:
        """Search external web and refine retrieved snippets using Stage B refiner."""
        search_results = self.search(query)
        if not search_results:
            return RefinementResult(
                refined_context="",
                selected_strips=[],
                total_strips_evaluated=0,
                has_evidence=False,
                original_char_count=0,
                refined_char_count=0
            )

        # Convert web results to items expected by KnowledgeRefiner: (mock_chunk, filename, score)
        class _WebChunk:
            def __init__(self, body: str, url: str):
                self.content = body
                self.id = url
                self.document_id = "web"

        items = [
            (_WebChunk(r["body"], r["url"]), f"Web: {r['title']}", 1.0)
            for r in search_results
        ]

        refinement = self.refiner.refine(query, items)
        if refinement.has_evidence:
            refinement.refined_context = (
                "--- EXTERNAL EVIDENCE (WEB SEARCH) ---\n" + refinement.refined_context
            )
        return refinement


# Global singleton instance
external_search_client = ExternalSearchClient()

