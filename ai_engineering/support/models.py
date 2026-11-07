"""Contracts at the assistant's trust boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class Principal:
    """Constructed by authentication, never from a chat request or model output."""

    tenant_id: str
    user_id: str
    scopes: frozenset[str] = frozenset()


class Document(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=16000)


class Message(StrictModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = Field(default=None, max_length=100)


class ToolCall(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    arguments: dict = Field(default_factory=dict)


class Answer(StrictModel):
    text: str = Field(min_length=1, max_length=16000)
    citations: list[str] = Field(default_factory=list, max_length=10)


class Observation(StrictModel):
    tool: str
    result: dict


@dataclass(frozen=True)
class ModelContext:
    instructions: str
    message: str
    history: tuple[Message, ...]
    untrusted_documents: tuple[Document, ...]
    untrusted_tool_results: tuple[Observation, ...]
    allowed_tools: tuple[str, ...]


class Model(Protocol):
    async def __call__(
        self, context: ModelContext, *, timeout_seconds: float
    ) -> ToolCall | Answer: ...


class ChatResponse(StrictModel):
    conversation_id: str
    status: Literal[
        "completed",
        "retrieval_unavailable",
        "no_documents",
        "tool_unavailable",
        "not_authorized",
        "budget_exhausted",
        "model_unavailable",
    ]
    answer: str
    citations: list[str] = Field(default_factory=list)
    tool_calls: int
    model_steps: int


class Limits(StrictModel):
    max_tool_calls: int = Field(default=6, ge=1, le=100, strict=True)
    max_model_steps: int = Field(default=4, ge=1, le=20, strict=True)
    max_attempts: int = Field(default=2, ge=1, le=5, strict=True)
    total_timeout_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
    per_tool_timeout_seconds: float = Field(default=2, gt=0, allow_inf_nan=False)
    model_timeout_seconds: float = Field(default=3, gt=0, allow_inf_nan=False)
    retry_delay_seconds: float = Field(default=0.1, ge=0, allow_inf_nan=False)
