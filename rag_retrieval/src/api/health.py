from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from .retrieval_service import RetrievalService


def build_health_response(service: "RetrievalService") -> Dict[str, Any]:
    return service.get_health()


def build_ready_response(service: "RetrievalService") -> Dict[str, Any]:
    return service.get_ready()
