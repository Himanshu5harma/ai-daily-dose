"""Run with uvicorn examples.support_assistant:create_demo_app --factory."""

import os
from contextlib import asynccontextmanager

from ai_engineering.support.api import create_app
from ai_engineering.support.demo import DemoModel
from ai_engineering.support.models import Document, Principal
from ai_engineering.support.openai_model import OpenAIModel
from ai_engineering.support.runtime import SupportAssistant
from ai_engineering.support.store import SQLiteStore
from ai_engineering.support.tools import (
    TicketArguments,
    TicketConnection,
    TicketStatusAdapter,
    Tool,
    ToolGateway,
)


def create_demo_app():
    token = os.environ["SUPPORT_DEMO_TOKEN"]  # No built-in default credential.
    principal = Principal("demo-company", "demo-user", frozenset({"tickets:read"}))
    store = SQLiteStore(os.environ.get("SUPPORT_DB", "support-demo.sqlite3"))
    store.provision_document(
        principal.tenant_id,
        Document(id="refund-policy", title="Refund policy", content="Refunds require a receipt."),
        readers=frozenset({principal.user_id}),
    )
    tools = ()
    if os.environ.get("SUPPORT_TICKET_URL"):
        adapter = TicketStatusAdapter(
            {
                (principal.tenant_id, principal.user_id): TicketConnection(
                    os.environ["SUPPORT_TICKET_URL"],
                    os.environ["SUPPORT_TICKET_TOKEN"],
                ),
            }
        )
        tools = (
            Tool(
                "get_ticket_status",
                "tickets:read",
                frozenset({principal.tenant_id}),
                TicketArguments,
                adapter,
            ),
        )
    model = DemoModel()
    if os.environ.get("SUPPORT_OPENAI_MODEL"):
        model = OpenAIModel(
            api_key=os.environ["OPENAI_API_KEY"], model=os.environ["SUPPORT_OPENAI_MODEL"]
        )
    assistant = SupportAssistant(store, model, ToolGateway(tools))
    app = create_app(assistant, {token: principal})

    @asynccontextmanager
    async def lifespan(_):
        try:
            yield
        finally:
            store.close()

    app.router.lifespan_context = lifespan
    return app
