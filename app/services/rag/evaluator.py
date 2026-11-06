import logging
from enum import Enum
from typing import Optional
import numpy as np

from app.config import (
    CRAG_EVALUATOR_MODEL,
    CRAG_EVALUATOR_DEVICE,
    CRAG_EVALUATOR_BATCH_SIZE,
    CRAG_EVALUATOR_MAX_LENGTH,
    CRAG_THRESHOLD_LOWER,
    CRAG_THRESHOLD_UPPER,
)

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """CRAG Confidence action triggered from Stage A evaluation."""
    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Transform raw logits into a bounded [0.0, 1.0] ranking score.
    
    Note: Transformed score is a semantic ranking score, NOT a calibrated
    posterior probability that the document is correct or sufficient.
    """
    return 1.0 / (1.0 + np.exp(-x))


class RetrievalEvaluator:
    """Lightweight CPU-only Cross-Encoder Evaluator for CRAG Stage A and Stage B."""

    _instance: Optional["RetrievalEvaluator"] = None
    _model = None

    def __new__(cls) -> "RetrievalEvaluator":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self):
        """Lazy-load the cross-encoder model strictly on CPU."""
        if self._model is None:
            logger.info(
                f"Initializing CPU CrossEncoder: model='{CRAG_EVALUATOR_MODEL}', "
                f"device='{CRAG_EVALUATOR_DEVICE}', max_length={CRAG_EVALUATOR_MAX_LENGTH}"
            )
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                model_name_or_path=CRAG_EVALUATOR_MODEL,
                device=CRAG_EVALUATOR_DEVICE,
                max_length=CRAG_EVALUATOR_MAX_LENGTH,
            )
        return self._model

    def predict_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute transformed relevance scores in [0.0, 1.0] for (query, text) pairs in batches."""
        if not pairs:
            return []

        try:
            # sentence-transformers predict returns ndarray of raw logits
            raw_logits = self.model.predict(
                pairs,
                batch_size=CRAG_EVALUATOR_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # Transform logits to [0, 1] range using sigmoid
            transformed = _sigmoid(raw_logits)
            if isinstance(transformed, np.ndarray):
                return [float(s) for s in transformed.tolist()]
            return [float(transformed)]
        except Exception as e:
            logger.error(f"Error evaluating text pairs with CrossEncoder: {e}")
            # Safe fallback if evaluation fails
            return [0.0] * len(pairs)

    def evaluate_documents(
        self,
        query: str,
        documents: list[str]
    ) -> tuple[ActionType, float, list[float]]:
        """Stage A: Document-level evaluation.
        
        Evaluates retrieved chunks/documents against the query.
        Uses max aggregation: S_doc = max(scores).
        
        Limitations: High S_doc indicates existence of at least 1 relevant chunk,
        not necessarily answer sufficiency.
        
        Returns:
            tuple[ActionType, float, list[float]]: (action, max_score, individual_scores)
        """
        if not documents:
            return ActionType.INCORRECT, 0.0, []

        pairs = [(query, doc) for doc in documents]
        scores = self.predict_scores(pairs)
        max_score = max(scores) if scores else 0.0

        if max_score >= CRAG_THRESHOLD_UPPER:
            action = ActionType.CORRECT
        elif max_score < CRAG_THRESHOLD_LOWER:
            action = ActionType.INCORRECT
        else:
            action = ActionType.AMBIGUOUS

        logger.debug(
            f"Stage A Eval: {len(documents)} docs, max_score={max_score:.4f}, action={action.value}"
        )
        return action, float(max_score), scores

    def evaluate_strips(
        self,
        query: str,
        strips: list[str]
    ) -> list[float]:
        """Stage B: Evaluate fine-grained knowledge strips using the same CPU evaluator."""
        if not strips:
            return []
        pairs = [(query, strip) for strip in strips]
        return self.predict_scores(pairs)


# Global singleton instance
retrieval_evaluator = RetrievalEvaluator()

