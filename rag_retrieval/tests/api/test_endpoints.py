from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.main import app

@pytest.fixture(autouse=True)
def mock_service(mock_retrieval_service):
    """Патчим глобальный _service для всех тестов в этом файле."""
    with patch.object(api_main, "_service", mock_retrieval_service), \
         patch.object(api_main, "_api_config", {
             "api": {"title": "Test", "version": "0.1.0"},
             "performance": {"max_query_length_chars": 4000, "max_batch_size": 3},
             "retrieval": {},
             "response": {},
             "logging": {},
         }):
        yield mock_retrieval_service


@pytest.fixture()
def client():
    return TestClient(app)

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_expected_fields(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["loaded"] is True
        assert data["num_chunks"] == 3702
        assert data["device"] == "cpu"
        assert "uptime_sec" in data

    def test_health_when_service_none(self, client):
        with patch.object(api_main, "_service", None):
            resp = client.get("/health")
            assert resp.status_code in (200, 503)


class TestReadyEndpoint:

    def test_ready_returns_200_when_ready(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200

    def test_ready_body(self, client):
        data = client.get("/ready").json()
        assert data["ready"] is True
        checks = data["checks"]
        assert checks["embeddings_loaded"] is True
        assert checks["embedding_model_loaded"] is True
        assert checks["reranker_loaded"] is True

    def test_ready_returns_503_when_not_ready(self, client, mock_service):
        mock_service.get_ready.return_value = {
            "ready": False,
            "checks": {
                "embeddings_loaded": True,
                "metadata_loaded": True,
                "embedding_model_loaded": False,
                "reranker_loaded": False,
            },
        }
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["ready"] is False

class TestSearchEndpoint:

    def _default_payload(self, **overrides):
        base = {
            "query": "Пациент 58 лет, боль за грудиной",
            "candidate_top_k": 30,
            "final_top_k": 5,
            "use_reranker": True,
            "context_selection": "anchor_page",
            "include_text": True,
        }
        base.update(overrides)
        return base

    def test_search_returns_200(self, client):
        resp = client.post("/search", json=self._default_payload())
        assert resp.status_code == 200

    def test_search_response_structure(self, client):
        data = client.post("/search", json=self._default_payload()).json()
        assert "query" in data
        assert "results" in data
        assert "timing" in data
        assert "trace_id" in data
        assert "config" in data

    def test_search_result_fields(self, client):
        data = client.post("/search", json=self._default_payload()).json()
        r = data["results"][0]
        for field in ("rank", "chunk_id", "document_id", "page_start", "page_end", "scores"):
            assert field in r, f"Missing field: {field}"

    def test_search_scores_fields(self, client):
        data = client.post("/search", json=self._default_payload()).json()
        scores = data["results"][0]["scores"]
        assert "dense_score" in scores
        assert "reranker_score" in scores
        assert "final_score" in scores

    def test_search_timing_fields(self, client):
        data = client.post("/search", json=self._default_payload()).json()
        timing = data["timing"]
        for key in (
            "query_embedding_time_ms",
            "dense_retrieval_time_ms",
            "reranking_time_ms",
            "context_selection_time_ms",
            "total_time_ms",
        ):
            assert key in timing

    def test_search_calls_service(self, client, mock_service):
        client.post("/search", json=self._default_payload())
        mock_service.search.assert_called_once()

    def test_search_passes_query_to_service(self, client, mock_service):
        client.post("/search", json=self._default_payload(query="ИБС стенокардия"))
        call_kwargs = mock_service.search.call_args.kwargs
        assert call_kwargs["query"] == "ИБС стенокардия"

    def test_search_passes_top_k_params(self, client, mock_service):
        client.post(
            "/search",
            json=self._default_payload(candidate_top_k=20, final_top_k=3),
        )
        kwargs = mock_service.search.call_args.kwargs
        assert kwargs["candidate_top_k"] == 20
        assert kwargs["final_top_k"] == 3

    def test_search_passes_context_selection(self, client, mock_service):
        client.post("/search", json=self._default_payload(context_selection="anchor_section"))
        kwargs = mock_service.search.call_args.kwargs
        assert kwargs["context_selection"] == "anchor_section"

    def test_search_passes_filters_to_service(self, client, mock_service):
        payload = self._default_payload()
        payload["filters"] = {"document_id": "kr155_2", "label": "diagnosis"}
        client.post("/search", json=payload)
        kwargs = mock_service.search.call_args.kwargs
        assert kwargs["filters"]["document_id"] == "kr155_2"
        assert kwargs["filters"]["label"] == "diagnosis"

    def test_search_empty_query_returns_400(self, client):
        resp = client.post("/search", json={"query": ""})
        assert resp.status_code == 422

    def test_search_whitespace_query_returns_400(self, client):
        resp = client.post("/search", json={"query": "   "})
        assert resp.status_code == 400

    def test_search_missing_query_returns_422(self, client):
        resp = client.post("/search", json={"candidate_top_k": 10})
        assert resp.status_code == 422

    def test_search_invalid_context_selection_returns_422(self, client):
        resp = client.post(
            "/search",
            json=self._default_payload(context_selection="bad_mode"),
        )
        assert resp.status_code == 422

    def test_search_final_top_k_gt_candidate_returns_422(self, client):
        resp = client.post(
            "/search",
            json=self._default_payload(candidate_top_k=10, final_top_k=15),
        )
        assert resp.status_code == 422

    def test_search_service_runtime_error_returns_503(self, client, mock_service):
        mock_service.search.side_effect = RuntimeError("reranker is not loaded")
        resp = client.post("/search", json=self._default_payload())
        assert resp.status_code == 503

    def test_search_service_exception_returns_500(self, client, mock_service):
        mock_service.search.side_effect = Exception("unexpected failure")
        resp = client.post("/search", json=self._default_payload())
        assert resp.status_code == 500

    def test_search_no_filters(self, client, mock_service):
        client.post("/search", json=self._default_payload())
        kwargs = mock_service.search.call_args.kwargs
        assert kwargs["filters"] is None

    def test_search_query_length_limit(self, client):
        resp = client.post("/search", json={"query": "x" * 4001})
        assert resp.status_code == 422

    def test_search_service_not_loaded_returns_503(self, client):
        with patch.object(api_main, "_service", None):
            resp = client.post("/search", json={"query": "тест"})
            assert resp.status_code == 503


class TestBatchSearchEndpoint:

    def _payload(self, queries, **overrides):
        base = {
            "queries": queries,
            "candidate_top_k": 10,
            "final_top_k": 3,
            "use_reranker": False,
            "context_selection": "none",
            "include_text": False,
        }
        base.update(overrides)
        return base

    def test_batch_returns_200(self, client):
        resp = client.post("/batch_search", json=self._payload(["q1"]))
        assert resp.status_code == 200

    def test_batch_response_structure(self, client):
        data = client.post("/batch_search", json=self._payload(["q1"])).json()
        assert "items" in data
        assert "batch_timing" in data
        assert "total_time_ms" in data["batch_timing"]
        assert "avg_time_ms" in data["batch_timing"]

    def test_batch_calls_service(self, client, mock_service):
        client.post("/batch_search", json=self._payload(["q1", "q2"]))
        mock_service.batch_search.assert_called_once()

    def test_batch_passes_queries_stripped(self, client, mock_service):
        client.post("/batch_search", json=self._payload(["q1", "q2"]))
        kwargs = mock_service.batch_search.call_args.kwargs
        assert kwargs["queries"] == ["q1", "q2"]

    def test_batch_exceeds_limit_returns_400(self, client):
        resp = client.post(
            "/batch_search",
            json=self._payload(["q"] * 10),
        )
        assert resp.status_code == 400

    def test_batch_empty_queries_returns_422(self, client):
        resp = client.post("/batch_search", json=self._payload([]))
        assert resp.status_code == 422

    def test_batch_empty_query_in_list_returns_400(self, client):
        resp = client.post("/batch_search", json=self._payload(["q1", "   "]))
        assert resp.status_code == 400

    def test_batch_final_gt_candidate_returns_422(self, client):
        resp = client.post(
            "/batch_search",
            json=self._payload(["q1"], candidate_top_k=5, final_top_k=10),
        )
        assert resp.status_code == 422

    def test_batch_runtime_error_returns_503(self, client, mock_service):
        mock_service.batch_search.side_effect = RuntimeError("reranker not loaded")
        resp = client.post("/batch_search", json=self._payload(["q1"]))
        assert resp.status_code == 503

class TestStatsEndpoint:

    def test_stats_returns_200(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200

    def test_stats_body(self, client):
        data = client.get("/stats").json()
        assert data["num_chunks"] == 3
        assert data["num_documents"] == 2
        assert "labels" in data
        assert "specialties" in data
        assert data["embedding_dim"] == 1024

    def test_stats_calls_service(self, client, mock_service):
        client.get("/stats")
        mock_service.stats.assert_called_once()


class TestDocumentsEndpoint:

    def test_documents_returns_200(self, client):
        resp = client.get("/documents")
        assert resp.status_code == 200

    def test_documents_body(self, client):
        data = client.get("/documents").json()
        assert "documents" in data
        assert "total_documents" in data
        assert data["total_documents"] == len(data["documents"])

    def test_documents_fields(self, client):
        data = client.get("/documents").json()
        doc = data["documents"][0]
        for field in ("document_id", "num_chunks", "labels"):
            assert field in doc

    def test_documents_calls_service(self, client, mock_service):
        client.get("/documents")
        mock_service.documents.assert_called_once()

class TestConfigEndpoint:

    def test_config_returns_200(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200

    def test_config_returns_dict(self, client):
        data = client.get("/config").json()
        assert isinstance(data, dict)

    def test_config_contains_api_section(self, client):
        data = client.get("/config").json()
        assert "api" in data


class TestResponseHeaders:

    def test_health_has_trace_id_header(self, client):
        resp = client.get("/health")
        assert "x-trace-id" in resp.headers

    def test_search_has_trace_id_header(self, client):
        resp = client.post(
            "/search",
            json={"query": "тест", "candidate_top_k": 10, "final_top_k": 5},
        )
        assert "x-trace-id" in resp.headers
