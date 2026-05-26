from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class SearchFilters(BaseModel):
    document_id: Optional[str] = None
    label: Optional[str] = None
    specialty: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    candidate_top_k: int = Field(30, ge=1, le=100)
    final_top_k: int = Field(5, ge=1, le=20)
    use_reranker: bool = True
    context_selection: Literal["none", "anchor_page", "anchor_section", "anchor_document"] = "anchor_page"
    filters: Optional[SearchFilters] = None
    include_text: bool = True
    include_embedding_text: bool = False

    @model_validator(mode="after")
    def check_top_k_order(self) -> "SearchRequest":
        if self.final_top_k > self.candidate_top_k:
            raise ValueError(
                f"final_top_k ({self.final_top_k}) must be <= candidate_top_k ({self.candidate_top_k})"
            )
        return self


class ResultScores(BaseModel):
    dense_score: float
    reranker_score: Optional[float] = None
    final_score: float


class SearchResult(BaseModel):
    rank: int
    chunk_id: str
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    label: Optional[str] = None
    specialty: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source: Optional[str] = None
    text: Optional[str] = None
    embedding_text: Optional[str] = None
    scores: ResultScores
    context_source: Optional[str] = None


class SearchConfig(BaseModel):
    candidate_top_k: int
    final_top_k: int
    use_reranker: bool
    context_selection: str
    filters: Dict[str, Any]


class SearchTiming(BaseModel):
    query_embedding_time_ms: float
    dense_retrieval_time_ms: float
    reranking_time_ms: float
    context_selection_time_ms: float
    total_time_ms: float


class SearchResponse(BaseModel):
    query: str
    config: SearchConfig
    results: List[SearchResult]
    timing: SearchTiming
    trace_id: str


class BatchSearchRequest(BaseModel):
    queries: List[str] = Field(..., min_length=1)
    candidate_top_k: int = Field(30, ge=1, le=100)
    final_top_k: int = Field(5, ge=1, le=20)
    use_reranker: bool = True
    context_selection: Literal["none", "anchor_page", "anchor_section", "anchor_document"] = "anchor_page"
    filters: Optional[SearchFilters] = None
    include_text: bool = True
    include_embedding_text: bool = False

    @model_validator(mode="after")
    def check_constraints(self) -> "BatchSearchRequest":
        if self.final_top_k > self.candidate_top_k:
            raise ValueError(
                f"final_top_k ({self.final_top_k}) must be <= candidate_top_k ({self.candidate_top_k})"
            )
        return self


class BatchTiming(BaseModel):
    total_time_ms: float
    avg_time_ms: float


class BatchSearchResponse(BaseModel):
    items: List[SearchResponse]
    batch_timing: BatchTiming


class ReadinessChecks(BaseModel):
    embeddings_loaded: bool
    metadata_loaded: bool
    embedding_model_loaded: bool
    reranker_loaded: bool


class HealthResponse(BaseModel):
    status: str
    service: str
    loaded: bool
    embedding_model_key: str
    reranker_key: str
    num_chunks: int
    device: str
    uptime_sec: float


class ReadyResponse(BaseModel):
    ready: bool
    checks: ReadinessChecks


class DocumentInfo(BaseModel):
    document_id: str
    document_title: Optional[str] = None
    num_chunks: int
    labels: List[str]
    page_min: Optional[int] = None
    page_max: Optional[int] = None


class DocumentsResponse(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int


class StatsResponse(BaseModel):
    num_chunks: int
    num_documents: int
    labels: Dict[str, int]
    specialties: Dict[str, int]
    embedding_dim: Optional[int] = None
