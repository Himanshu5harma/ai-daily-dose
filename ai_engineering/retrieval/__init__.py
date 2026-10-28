"""Policy-aware retrieval: filter candidates by access, region and version, then rank."""

from .policy import (
    AccessLevel,
    CandidateDocument,
    DocumentMetadata,
    UserContext,
    filter_eligible,
    is_eligible,
    rank_top_k,
    retrieve,
)

__all__ = [
    "AccessLevel",
    "CandidateDocument",
    "DocumentMetadata",
    "UserContext",
    "filter_eligible",
    "is_eligible",
    "rank_top_k",
    "retrieve",
]
