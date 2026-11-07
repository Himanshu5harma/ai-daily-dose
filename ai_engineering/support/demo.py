"""Offline extractive planner implementing the same contract as a hosted model."""

import re

from .models import Answer, ModelContext, ToolCall


class DemoModel:
    async def __call__(self, context: ModelContext, *, timeout_seconds: float) -> Answer | ToolCall:
        # Only the user's current message chooses an action; documents cannot
        # introduce a tool instruction into this deterministic demonstration.
        ticket = re.search(r"\bticket\s+([A-Za-z0-9_-]{1,64})\b", context.message, re.I)
        if ticket and not context.untrusted_tool_results:
            if "get_ticket_status" not in context.allowed_tools:
                return Answer(text="Ticket lookup is not enabled for your account.")
            return ToolCall(name="get_ticket_status", arguments={"ticket_id": ticket[1]})
        if context.untrusted_tool_results:
            result = context.untrusted_tool_results[-1].result
            return Answer(text=f"Ticket {result['id']} is {result['status']}.")
        if not context.untrusted_documents:
            return Answer(text="I couldn't find relevant company documents.")
        return Answer(
            text="\n\n".join(f"{doc.title}: {doc.content}" for doc in context.untrusted_documents),
            citations=[doc.id for doc in context.untrusted_documents],
        )
