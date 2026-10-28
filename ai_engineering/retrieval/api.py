"""FastAPI app exposing policy-aware retrieval.

Run locally with an ASGI server, e.g. `uvicorn ai_engineering.retrieval.api:app --reload`.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field

from .policy import (
    DEFAULT_TOP_K,
    CandidateDocument,
    DocumentMetadata,
    StrictModel,
    UserContext,
    retrieve,
)

MAX_CANDIDATES = 1000


class RetrieveRequest(StrictModel):
    query: str = Field(min_length=1)
    user_context: UserContext
    candidate_documents: list[CandidateDocument] = Field(max_length=MAX_CANDIDATES)


class RetrievedDocument(StrictModel):
    id: str
    rank: int
    score: float
    content: str
    metadata: DocumentMetadata


class RetrieveResponse(StrictModel):
    query: str
    top_k: int
    count: int
    results: list[RetrievedDocument]


app = FastAPI(title="Policy-aware retrieval")


@app.exception_handler(RequestValidationError)
def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Rejected input values are not echoed back: they may hold document content or
    # non-JSON values such as NaN.
    errors = [
        {"loc": list(err["loc"]), "type": err["type"], "msg": err["msg"]} for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/retrieve", response_model=RetrieveResponse, status_code=status.HTTP_200_OK)
def retrieve_documents(request: RetrieveRequest) -> RetrieveResponse:
    selected = retrieve(request.user_context, request.candidate_documents, DEFAULT_TOP_K)
    results = [
        RetrievedDocument(
            id=doc.id, rank=rank, score=doc.score, content=doc.content, metadata=doc.metadata
        )
        for rank, doc in enumerate(selected, start=1)
    ]
    return RetrieveResponse(
        query=request.query, top_k=DEFAULT_TOP_K, count=len(results), results=results
    )
