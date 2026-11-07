"""Stateless Responses API adapter; application code executes all proposed tools.

See https://developers.openai.com/api/docs/guides/function-calling . Each step
is an independent decision over the supplied context, without provider-side
conversation IDs or automatic retries that could escape the request budget.
"""

import json

import httpx

from .models import Answer, ModelContext, ToolCall
from .tools import TicketArguments, ToolUnavailable


class OpenAIModel:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not api_key or not model:
            raise ValueError("API key and model must be configured")
        self._api_key = api_key
        self._model = model
        self._transport = transport

    async def __call__(self, context: ModelContext, *, timeout_seconds: float) -> Answer | ToolCall:
        tools = [
            {
                "type": "function",
                "name": "respond",
                "strict": True,
                "description": "Return the answer with authorized document IDs as citations.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "citations": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text", "citations"],
                },
            }
        ]
        if "get_ticket_status" in context.allowed_tools:
            tools.append(
                {
                    "type": "function",
                    "name": "get_ticket_status",
                    "strict": True,
                    "description": "Read the status of the ticket requested by the user.",
                    "parameters": TicketArguments.model_json_schema(),
                }
            )
        payload = {
            "message": context.message,
            "untrusted_history": [message.model_dump() for message in context.history],
            "untrusted_documents": [doc.model_dump() for doc in context.untrusted_documents],
            "untrusted_tool_results": [
                item.model_dump() for item in context.untrusted_tool_results
            ],
        }
        async with httpx.AsyncClient(
            transport=self._transport,
            follow_redirects=False,
            trust_env=False,
            timeout=timeout_seconds,
        ) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "store": False,
                    "instructions": context.instructions,
                    "input": [{"role": "user", "content": json.dumps(payload)}],
                    "tools": tools,
                    "tool_choice": "required",
                    "parallel_tool_calls": False,
                    "max_output_tokens": 2048,
                },
            ) as response:
                if response.status_code != 200:
                    raise ToolUnavailable("model service unavailable")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 1_048_576:
                        raise ToolUnavailable("model response too large")
        result = json.loads(body)
        if result.get("status") != "completed":
            raise ToolUnavailable("model response incomplete")
        calls = [item for item in result.get("output", []) if item.get("type") == "function_call"]
        if len(calls) != 1:
            raise ToolUnavailable("expected one model decision")
        call = calls[0]
        arguments = json.loads(call["arguments"])
        if call["name"] == "respond":
            return Answer.model_validate(arguments)
        return ToolCall(name=call["name"], arguments=arguments)
