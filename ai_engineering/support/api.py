"""Authenticated support API; identity never comes from the request body."""

from collections.abc import Mapping
from hashlib import sha256
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .models import ChatRequest, ChatResponse, Message, Principal
from .runtime import SupportAssistant
from .store import ConversationNotFound


def create_app(assistant: SupportAssistant, tokens: Mapping[str, Principal]) -> FastAPI:
    """Local token authentication. Production can replace this dependency with OIDC.

    Tokens are opaque secrets provisioned by server code, not unsigned identity
    claims. Only their digests are retained in this authentication lookup.
    """
    identities = {sha256(token.encode()).digest(): principal for token, principal in tokens.items()}
    if any(not token for token in tokens):
        raise ValueError("authentication tokens must not be empty")
    app = FastAPI(title="Tenant-aware support assistant")
    bearer = HTTPBearer(auto_error=False)

    def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> Principal:
        principal = None
        if credentials is not None:
            principal = identities.get(sha256(credentials.credentials.encode()).digest())
        if principal is None:
            raise HTTPException(
                401, "authentication required", headers={"WWW-Authenticate": "Bearer"}
            )
        return principal

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": list(error["loc"]), "type": error["type"], "msg": error["msg"]}
                    for error in exc.errors()
                ]
            },
        )

    @app.exception_handler(ConversationNotFound)
    async def missing_conversation(_, exc: ConversationNotFound):
        return JSONResponse(status_code=404, content={"detail": "conversation not found"})

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, principal: Annotated[Principal, Depends(authenticate)]):
        return await assistant.chat(principal, request)

    @app.get("/conversations/{conversation_id}", response_model=list[Message])
    def history(conversation_id: str, principal: Annotated[Principal, Depends(authenticate)]):
        return assistant.store.history(principal, conversation_id)

    return app
