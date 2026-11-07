"""Deny-by-default tool gateway and a read-only external ticket adapter."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

import httpx
from pydantic import Field, ValidationError

from .models import Principal, StrictModel, ToolCall


class ToolDenied(PermissionError):
    pass


class ToolUnavailable(RuntimeError):
    pass


class TransientFailure(ToolUnavailable):
    """Only failures explicitly classified as transient are retried."""


ToolHandler = Callable[[Principal, StrictModel, float], Awaitable[dict]]


@dataclass(frozen=True)
class Tool:
    name: str
    required_scope: str
    tenant_ids: frozenset[str]
    arguments_schema: type[StrictModel]
    handler: ToolHandler


class ToolGateway:
    def __init__(
        self,
        tools: tuple[Tool, ...] = (),
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("tool names must be unique")
        if failure_threshold < 1 or cooldown_seconds <= 0 or not isfinite(cooldown_seconds):
            raise ValueError("circuit breaker limits must be positive")
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._failures: dict[tuple[str, str, str], tuple[int, float]] = {}
        self._probes: set[tuple[str, str, str]] = set()

    def allowed_names(self, principal: Principal) -> tuple[str, ...]:
        return tuple(
            name
            for name, tool in self._tools.items()
            if principal.tenant_id in tool.tenant_ids and tool.required_scope in principal.scopes
        )

    def authorize(self, principal: Principal, call: ToolCall) -> tuple[Tool, StrictModel]:
        if call.name not in self.allowed_names(principal):
            raise ToolDenied("tool is not authorized")
        tool = self._tools[call.name]
        try:
            arguments = tool.arguments_schema.model_validate(call.arguments)
        except ValidationError:
            raise ToolDenied("invalid tool arguments") from None
        return tool, arguments

    def record_failure(self, principal: Principal, name: str) -> None:
        key = (principal.tenant_id, principal.user_id, name)
        failures = self._failures.get(key, (0, 0))[0] + 1
        self._failures[key] = (
            failures,
            self._clock() + self._cooldown if failures >= self._failure_threshold else 0,
        )

    async def execute(self, principal: Principal, call: ToolCall, timeout: float) -> dict:
        tool, arguments = self.authorize(principal, call)
        key = (principal.tenant_id, principal.user_id, tool.name)
        _, open_until = self._failures.get(key, (0, 0))
        if open_until > self._clock() or key in self._probes:
            raise ToolUnavailable("tool circuit is open")
        if open_until:
            self._failures.pop(key, None)
            self._probes.add(key)
        state_before_call = self._failures.get(key)
        started = self._clock()
        try:
            result = await tool.handler(principal, arguments, timeout)
            if self._clock() - started >= timeout:
                raise asyncio.TimeoutError
        except (TimeoutError, asyncio.TimeoutError):
            self.record_failure(principal, tool.name)
            # Distinguish a handler timeout from cancellation by the outer deadline.
            raise TransientFailure("tool timed out") from None
        except TransientFailure:
            self.record_failure(principal, tool.name)
            raise
        finally:
            if open_until:
                self._probes.discard(key)
        # A late success must not close a circuit opened by another concurrent call.
        if self._failures.get(key) == state_before_call:
            self._failures.pop(key, None)
        return result


class TicketArguments(StrictModel):
    ticket_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")


class TicketResult(StrictModel):
    id: str
    status: Literal["open", "pending", "resolved", "closed"]


@dataclass(frozen=True)
class TicketConnection:
    base_url: str
    token: str = field(repr=False)

    def __post_init__(self):
        url = httpx.URL(self.base_url)
        if (
            url.scheme != "https"
            or not url.host
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("ticket base URL must be a server-configured HTTPS URL")


class TicketStatusAdapter:
    """Connections must use tenant- AND user-scoped upstream authorization.

    No arbitrary URL, headers, credentials, or tenant override can come from the
    model. Redirects are disabled to avoid forwarding requests to another host.
    """

    def __init__(
        self,
        connections: Mapping[tuple[str, str], TicketConnection],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._connections = dict(connections)
        self._transport = transport

    async def __call__(
        self, principal: Principal, arguments: TicketArguments, timeout_seconds: float
    ) -> dict:
        connection = self._connections.get((principal.tenant_id, principal.user_id))
        if connection is None:
            raise ToolDenied("ticket integration is not authorized")
        url = f"{connection.base_url.rstrip('/')}/tickets/{arguments.ticket_id}"
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                follow_redirects=False,
                trust_env=False,
                timeout=timeout_seconds,
            ) as client:
                # Bound response size; return a narrow schema instead of arbitrary text/HTML.
                async with client.stream(
                    "GET",
                    url,
                    headers={"Authorization": f"Bearer {connection.token}"},
                ) as response:
                    if response.status_code in (401, 403, 404):
                        raise ToolDenied("ticket is not accessible")
                    if response.status_code == 429 or response.status_code >= 500:
                        raise TransientFailure("ticket service temporarily unavailable")
                    if response.status_code != 200:
                        raise ToolUnavailable("unexpected ticket service response")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > 16_384:
                            raise ToolUnavailable("ticket service response too large")
            result = TicketResult.model_validate_json(body)
            if result.id != arguments.ticket_id:
                raise ToolUnavailable("ticket response did not match request")
            return result.model_dump()
        except (httpx.TimeoutException, httpx.TransportError):
            raise TransientFailure("ticket service connection failed") from None
        except ValidationError:
            raise ToolUnavailable("invalid ticket service response") from None
