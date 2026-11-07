# Tenant-aware support assistant

A runnable reference implementation of authenticated document search and a bounded
external-tool workflow. The existing `/retrieve` exercise remains a separate app;
the assistant searches its own server-provisioned documents and never accepts user
identity or document permissions from a chat payload.

## Run locally

From the repository root, using the virtual environment described in the main README:

```bash
python -m pip install -e '.[dev]'
export SUPPORT_DEMO_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m uvicorn examples.support_assistant:create_demo_app --factory
```

In another terminal with the same token configured:

```bash
curl http://127.0.0.1:8000/chat \
  -H "Authorization: Bearer $SUPPORT_DEMO_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the refund policy?"}'
```

The default planner is deterministic and returns excerpts from the seeded refund
document. It requires no provider account. SQLite history is stored in
`support-demo.sqlite3`; set `SUPPORT_DB` to choose a different path. Include the
returned `conversation_id` in the next request to continue a conversation.
`GET /conversations/{conversation_id}` returns its most recent 20 messages and
requires the same tenant and owner identity.

## Optional hosted model and external ticket service

To use the Responses API adapter, explicitly set `SUPPORT_OPENAI_MODEL` to a model
available to your account and `OPENAI_API_KEY` to your API key before starting the
server. This sends the authorized context to the provider and incurs provider usage.
The adapter uses independent, stateless decisions with `store: false`, strict function
schemas, no automatic retries, and no hosted tools. `store: false` does not by itself
establish a zero-retention agreement with a provider. The application executes all
tool proposals through its own gateway. See the official
[function calling documentation](https://developers.openai.com/api/docs/guides/function-calling).

To enable ticket lookup, set `SUPPORT_TICKET_URL` to a trusted HTTPS API base URL
and `SUPPORT_TICKET_TOKEN` to a token scoped to the demo tenant AND demo user.
The adapter sends `GET {base_url}/tickets/{ticket_id}` with bearer authentication.
The expected JSON response is exactly:

```json
{"id": "123", "status": "open"}
```

Supported statuses are `open`, `pending`, `resolved`, and `closed`. The adapter
rejects redirects, unexpected fields, mismatched IDs, and responses over 16 KiB.
A request such as `{"message":"Check ticket 123"}` works without matching policy
documents. Provision connections under `(tenant_id, user_id)` in server code;
an administrator token with unrestricted resource access is unsuitable.

## Enforcement points

| Concern | Implementation |
| --- | --- |
| Identity | Opaque bearer tokens map to server-owned `Principal` objects. Identity fields in request bodies are rejected. |
| Document isolation | SQLite joins enforce tenant and explicit user read permissions before ranking and model context construction. |
| Conversation isolation | Every read and write checks both tenant and conversation owner; inaccessible IDs return 404. |
| Tool authorization | Gateway checks registered name, tenant allowlist, required scope, and argument schema on each attempt. |
| Shared budget | One monotonic deadline covers retrieval, model decisions, tool attempts, and retry delays. |
| Loop control | Six tool attempts including retrieval/retries, four model steps, and repeated identical calls stop execution by default. |
| Per-attempt timeout | Two seconds per tool and three per model step, capped to remaining request time; late results are rejected. |
| Recovery | Retry only timeouts and explicitly transient failures, at most twice per operation; authorization denials are never retried. |
| Circuit breaker | Repeated transient failures or tool timeouts open a circuit, scoped to tenant/user/tool, for 30 seconds. A single recovery probe is permitted. |
| Retrieval fallback | Return an explicit unavailable response without invoking the model; empty results permit authorized tools but cannot produce an unsupported policy answer. |
| Untrusted content | Documents, history, and results are labeled as data; tool permissions are enforced independently of model behavior. |
| Credentials | Server-configured connections inject credentials; model context and persisted chat omit them. |

`Limits` configures budgets in trusted server code. Requests cannot raise them.
Async network operations are cancelled on timeout; SQLite search runs in a worker
with its own deadline, timed lock acquisition, and SQL progress interruption.
Logs contain final status and counters, without raw chat text, tokens, or tool errors.

## Scope and operational limits

This is a local reference implementation, not a deployed enterprise service. Token
provisioning is static; a production identity adapter should validate OIDC tokens,
current tenant membership, revocation, and user scopes. SQLite uses application
checks rather than database row security; production storage should add database
isolation, encrypted storage, retention controls, and distributed coordination.
The circuit breaker is per process, and conversations do not serialize concurrent
turns from the same user. Metadata/history writes are synchronous local SQLite
operations; this is not a hard end-to-end wall-clock guarantee under database stalls.

Search ranks word overlap over up to 1,000 authorized documents and returns three.
Provision bounded document chunks rather than whole books. History access means
access to the user's previously stored messages; changing document permissions does
not retroactively erase excerpts already saved in history.

Only a read-only ticket tool is shipped. Adding writes requires a separate approval
boundary, idempotency keys, and reconciliation after uncertain outcomes. Human
handoff is suggested in fallback text; the demo does not create or send a support
ticket. A system prompt and valid citation IDs cannot guarantee factual grounding or
eliminate prompt injection. Render text safely in a client and restrict network
egress in deployment. See [OWASP's guidance on excessive agency](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/).

## Verify

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The tests cover tenant and owner isolation, identity spoofing, malicious tool
proposals, deadlines, cancellation, capped delays, retries, loop limits, circuit
recovery, persistence, and mocked ticket/model HTTP contracts. They make no paid
model calls or live external-tool requests. Clock and retry sleep are injected for
budget tests; cancellation tests use events without elapsed-time sleeps.
