from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    BatchSearchRequest,
    BatchSearchResponse,
    BatchTiming,
    DocumentInfo,
    DocumentsResponse,
    HealthResponse,
    ReadinessChecks,
    ReadyResponse,
    ResultScores,
    SearchConfig,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchTiming,
    StatsResponse,
)

class TestSearchRequestValidation:

    def test_defaults(self):
        r = SearchRequest(query="тест")
        assert r.candidate_top_k == 30
        assert r.final_top_k == 5
        assert r.use_reranker is True
        assert r.context_selection == "anchor_page"
        assert r.filters is None
        assert r.include_text is True
        assert r.include_embedding_text is False

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SearchRequest(query="")
        assert "min_length" in str(exc.value) or "at least" in str(exc.value).lower()

    def test_whitespace_only_query_accepted_at_schema_level(self):
        # Pydantic принимает, а endpoint делает .strip() и отдаёт 400
        r = SearchRequest(query="   ")
        assert r.query == "   "

    def test_query_too_long_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="x" * 4001)

    def test_query_max_length_accepted(self):
        r = SearchRequest(query="x" * 4000)
        assert len(r.query) == 4000

    def test_final_top_k_greater_than_candidate_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SearchRequest(query="q", candidate_top_k=10, final_top_k=15)
        assert "final_top_k" in str(exc.value)

    def test_equal_top_k_accepted(self):
        r = SearchRequest(query="q", candidate_top_k=10, final_top_k=10)
        assert r.final_top_k == r.candidate_top_k

    def test_candidate_top_k_out_of_range(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="q", candidate_top_k=0)
        with pytest.raises(ValidationError):
            SearchRequest(query="q", candidate_top_k=101)

    def test_final_top_k_out_of_range(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="q", final_top_k=0)
        with pytest.raises(ValidationError):
            SearchRequest(query="q", candidate_top_k=100, final_top_k=21)

    def test_context_selection_valid_values(self):
        for mode in ("none", "anchor_page", "anchor_section", "anchor_document"):
            r = SearchRequest(query="q", context_selection=mode)
            assert r.context_selection == mode

    def test_context_selection_invalid_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="q", context_selection="unknown_mode")

    def test_filters_optional(self):
        r = SearchRequest(query="q", filters={"document_id": "kr155_2", "label": None})
        assert r.filters is not None
        assert r.filters.document_id == "kr155_2"
        assert r.filters.label is None

    def test_full_valid_request(self):
        r = SearchRequest(
            query="Пациент 58 лет, боль за грудиной",
            candidate_top_k=30,
            final_top_k=5,
            use_reranker=True,
            context_selection="anchor_page",
            filters={"document_id": "kr155_2", "label": "diagnosis", "specialty": "cardiology"},
            include_text=True,
            include_embedding_text=False,
        )
        assert r.filters.specialty == "cardiology"

class TestSearchFilters:

    def test_all_none_default(self):
        f = SearchFilters()
        assert f.document_id is None
        assert f.label is None
        assert f.specialty is None

    def test_partial_filter(self):
        f = SearchFilters(label="diagnosis")
        assert f.label == "diagnosis"
        assert f.document_id is None


class TestBatchSearchRequestValidation:

    def test_empty_queries_rejected(self):
        with pytest.raises(ValidationError):
            BatchSearchRequest(queries=[])

    def test_single_query_accepted(self):
        r = BatchSearchRequest(queries=["тест"])
        assert len(r.queries) == 1

    def test_cross_validation_final_gt_candidate(self):
        with pytest.raises(ValidationError):
            BatchSearchRequest(queries=["q"], candidate_top_k=5, final_top_k=10)

    def test_valid_batch(self):
        r = BatchSearchRequest(
            queries=["q1", "q2", "q3"],
            candidate_top_k=30,
            final_top_k=5,
        )
        assert len(r.queries) == 3

class TestResponseSchemas:

    def test_health_response(self):
        h = HealthResponse(
            status="ok",
            service="rag-retrieval-api",
            loaded=True,
            embedding_model_key="e5_large",
            reranker_key="bge_reranker_v2_m3",
            num_chunks=3702,
            device="cpu",
            uptime_sec=42.5,
        )
        assert h.status == "ok"
        assert h.num_chunks == 3702

    def test_ready_response(self):
        r = ReadyResponse(
            ready=True,
            checks=ReadinessChecks(
                embeddings_loaded=True,
                metadata_loaded=True,
                embedding_model_loaded=True,
                reranker_loaded=True,
            ),
        )
        assert r.ready is True

    def test_search_result_with_all_fields(self):
        res = SearchResult(
            rank=1,
            chunk_id="chunk-001",
            document_id="kr155_2",
            document_title="КР155",
            section_id="2",
            section_title="Диагностика",
            label="diagnosis",
            specialty="cardiology",
            page_start=12,
            page_end=13,
            source="docs/kr155_2.pdf",
            text="Текст.",
            embedding_text=None,
            scores=ResultScores(dense_score=0.9, reranker_score=3.5, final_score=3.5),
            context_source="anchor",
        )
        assert res.rank == 1
        assert res.scores.dense_score == 0.9
        assert res.scores.reranker_score == 3.5

    def test_search_result_reranker_score_optional(self):
        res = SearchResult(
            rank=1,
            chunk_id="chunk-001",
            scores=ResultScores(dense_score=0.85, reranker_score=None, final_score=0.85),
        )
        assert res.scores.reranker_score is None

    def test_stats_response(self):
        s = StatsResponse(
            num_chunks=3702,
            num_documents=50,
            labels={"diagnosis": 1200, "treatment": 900},
            specialties={"cardiology": 300},
            embedding_dim=1024,
        )
        assert s.num_chunks == 3702
        assert s.labels["diagnosis"] == 1200

    def test_documents_response(self):
        r = DocumentsResponse(
            documents=[
                DocumentInfo(
                    document_id="kr155_2",
                    document_title="КР155",
                    num_chunks=42,
                    labels=["diagnosis", "treatment"],
                    page_min=1,
                    page_max=45,
                )
            ],
            total_documents=1,
        )
        assert r.total_documents == 1
        assert r.documents[0].labels == ["diagnosis", "treatment"]
