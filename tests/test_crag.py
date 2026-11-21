import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from app.config import (
    CRAG_THRESHOLD_LOWER,
    CRAG_THRESHOLD_UPPER,
    CRAG_STRIP_THRESHOLD,
)
from app.services.rag.evaluator import (
    RetrievalEvaluator,
    ActionType,
    _sigmoid,
)
from app.services.rag.refiner import (
    KnowledgeRefiner,
    decompose_chunk,
    KnowledgeStrip,
)
from app.services.rag.web_search import (
    rewrite_query_for_search,
    ExternalSearchClient,
)
from app.services.rag.crag_engine import CRAGEngine


@dataclass
class MockChunk:
    id: str
    document_id: str
    content: str


def test_sigmoid_transformation():
    # Test sigmoid properties
    assert _sigmoid(0.0) == 0.5
    assert _sigmoid(100.0) > 0.99
    assert _sigmoid(-100.0) < 0.01


def test_evaluator_device_and_stage_a():
    evaluator = RetrievalEvaluator()
    # Check evaluator device configuration
    assert evaluator._instance is not None

    # Mock model predict to return controlled logits
    with patch.object(evaluator, "predict_scores") as mock_predict:
        # High score -> CORRECT
        mock_predict.return_value = [0.85, 0.40]
        action, max_s, scores = evaluator.evaluate_documents("query", ["doc1", "doc2"])
        assert action == ActionType.CORRECT
        assert max_s == 0.85
        assert scores == [0.85, 0.40]

        # Medium score -> AMBIGUOUS
        mock_predict.return_value = [0.55, 0.20]
        action, max_s, scores = evaluator.evaluate_documents("query", ["doc1", "doc2"])
        assert action == ActionType.AMBIGUOUS
        assert max_s == 0.55

        # Low score -> INCORRECT
        mock_predict.return_value = [0.15, 0.22]
        action, max_s, scores = evaluator.evaluate_documents("query", ["doc1", "doc2"])
        assert action == ActionType.INCORRECT
        assert max_s == 0.22

        # Empty documents
        action, max_s, scores = evaluator.evaluate_documents("query", [])
        assert action == ActionType.INCORRECT
        assert max_s == 0.0
        assert scores == []


def test_decompose_chunk():
    text = "The Python programming language was created by Guido van Rossum. It was released in 1991. It emphasizes readability."
    strips = decompose_chunk(
        chunk_content=text,
        doc_id="doc123",
        filename="python_guide.txt",
        chunk_id="chk1",
        sentences_per_strip=2
    )
    assert len(strips) >= 2
    assert strips[0].doc_id == "doc123"
    assert strips[0].filename == "python_guide.txt"
    assert strips[0].chunk_id == "chk1"
    assert "Guido van Rossum" in strips[0].text


def test_knowledge_refiner_filtering_and_provenance():
    mock_evaluator = MagicMock()
    refiner = KnowledgeRefiner(evaluator=mock_evaluator)

    chunk1 = MockChunk(
        id="c1",
        document_id="d1",
        content="Antigravity is an agentic AI coding assistant. It supports pairing with users. The sky is bright blue on Tuesdays."
    )
    retrieved_items = [(chunk1, "antigravity.md", 0.12)]

    # Mock evaluate_strips: first strip high score, second strip low score
    mock_evaluator.evaluate_strips.return_value = [0.85, 0.10]

    result = refiner.refine(
        query="What is Antigravity?",
        retrieved_items=retrieved_items,
        strip_threshold=0.40,
        preserve_neighbors=False
    )

    assert result.has_evidence is True
    assert len(result.selected_strips) == 1
    assert "Antigravity is an agentic AI" in result.refined_context
    assert "The sky is bright blue" not in result.refined_context
    assert "[antigravity.md]:" in result.refined_context
    assert result.selected_strips[0].score == 0.85


def test_knowledge_refiner_no_evidence():
    mock_evaluator = MagicMock()
    refiner = KnowledgeRefiner(evaluator=mock_evaluator)

    chunk1 = MockChunk(
        id="c1",
        document_id="d1",
        content="Apples are fruit. Bananas are yellow."
    )
    retrieved_items = [(chunk1, "fruit.txt", 0.55)]

    # Both strips score very low
    mock_evaluator.evaluate_strips.return_value = [0.05, 0.12]

    result = refiner.refine(
        query="How to write a compiler in Rust?",
        retrieved_items=retrieved_items,
        strip_threshold=0.40
    )

    assert result.has_evidence is False
    assert result.refined_context == ""
    assert result.selected_strips == []


def test_rewrite_query_for_search():
    assert rewrite_query_for_search("What is the capital of France?") == "capital France"
    assert rewrite_query_for_search("Thủ đô của nước Pháp là gì?") == "Thủ đô nước Pháp"


def test_crag_engine_correct_action_flow():
    mock_eval = MagicMock()
    mock_refiner = MagicMock()
    mock_search = MagicMock()
    mock_search.is_web_allowed.return_value = False

    engine = CRAGEngine(
        evaluator=mock_eval,
        refiner=mock_refiner,
        search_client=mock_search
    )

    chunk1 = MockChunk(id="c1", document_id="d1", content="Exact matching content.")
    retrieved_items = [(chunk1, "doc.txt", 0.1)]

    # Stage A gives CORRECT
    mock_eval.evaluate_documents.return_value = (ActionType.CORRECT, 0.92, [0.92])

    from app.services.rag.refiner import RefinementResult, KnowledgeStrip
    mock_refiner.refine.return_value = RefinementResult(
        refined_context="[doc.txt]: Exact matching content.",
        selected_strips=[
            KnowledgeStrip(
                text="Exact matching content.",
                doc_id="d1",
                filename="doc.txt",
                chunk_id="c1",
                strip_index=0,
                char_start=0,
                char_end=22,
                score=0.92
            )
        ],
        total_strips_evaluated=1,
        has_evidence=True,
        original_char_count=22,
        refined_char_count=22
    )

    res = engine.process("query", retrieved_items)

    assert res.action == ActionType.CORRECT
    assert res.has_evidence is True
    assert "[doc.txt]: Exact matching content." in res.context
    assert res.telemetry is not None
    assert res.telemetry.action_triggered == "CORRECT"
    assert res.telemetry.evidence_status == "VALID_EVIDENCE"


def test_crag_engine_incorrect_document_only_flow():
    mock_eval = MagicMock()
    mock_refiner = MagicMock()
    mock_search = MagicMock()
    # DOCUMENT_ONLY: web is not allowed
    mock_search.is_web_allowed.return_value = False

    engine = CRAGEngine(
        evaluator=mock_eval,
        refiner=mock_refiner,
        search_client=mock_search
    )

    chunk1 = MockChunk(id="c1", document_id="d1", content="Irrelevant text.")
    retrieved_items = [(chunk1, "doc.txt", 0.9)]

    # Stage A gives INCORRECT
    mock_eval.evaluate_documents.return_value = (ActionType.INCORRECT, 0.15, [0.15])

    res = engine.process("query", retrieved_items)

    assert res.action == ActionType.INCORRECT
    assert res.has_evidence is False
    assert res.context == ""
    assert res.telemetry.evidence_status == "NO_EVIDENCE"
    assert res.telemetry.web_search_used is False

