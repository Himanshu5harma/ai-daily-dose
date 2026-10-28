"""Policy filtering and ranking for retrieval candidates.

Filtering always runs before ranking, so ineligible documents never take part in
top-k selection.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

GLOBAL_REGION = "global"
DEFAULT_TOP_K = 3


class AccessLevel(str, Enum):
    """Clearance levels; a user can read documents at or below their own level."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    @property
    def rank(self) -> int:
        return _ACCESS_RANK[self]


_ACCESS_RANK = {level: rank for rank, level in enumerate(AccessLevel)}


class StrictModel(BaseModel):
    # Unknown fields are rejected so a misspelled policy field cannot be silently ignored.
    model_config = ConfigDict(extra="forbid")


class UserContext(StrictModel):
    role: str = Field(min_length=1)
    access_level: AccessLevel
    region: str = Field(min_length=1)
    allowed_product_versions: list[str]


class DocumentMetadata(StrictModel):
    access_level: AccessLevel
    region: str = Field(min_length=1)
    product_version: str = Field(min_length=1)


class CandidateDocument(StrictModel):
    id: str = Field(min_length=1)
    content: str
    score: float = Field(allow_inf_nan=False)
    metadata: DocumentMetadata


def is_authorized(user: UserContext, doc: CandidateDocument) -> bool:
    return user.access_level.rank >= doc.metadata.access_level.rank


def is_region_allowed(user: UserContext, doc: CandidateDocument) -> bool:
    doc_region = doc.metadata.region.casefold()
    return doc_region == GLOBAL_REGION or doc_region == user.region.casefold()


def is_version_allowed(user: UserContext, doc: CandidateDocument) -> bool:
    return doc.metadata.product_version in user.allowed_product_versions


def is_eligible(user: UserContext, doc: CandidateDocument) -> bool:
    return (
        is_authorized(user, doc) and is_region_allowed(user, doc) and is_version_allowed(user, doc)
    )


def filter_eligible(
    user: UserContext, candidates: Iterable[CandidateDocument]
) -> list[CandidateDocument]:
    return [doc for doc in candidates if is_eligible(user, doc)]


def rank_top_k(documents: Iterable[CandidateDocument], k: int) -> list[CandidateDocument]:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    # Ties are broken by id so results are deterministic.
    return sorted(documents, key=lambda doc: (-doc.score, doc.id))[:k]


def retrieve(
    user: UserContext, candidates: Iterable[CandidateDocument], k: int = DEFAULT_TOP_K
) -> list[CandidateDocument]:
    return rank_top_k(filter_eligible(user, candidates), k)
