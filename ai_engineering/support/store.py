"""Small persistent store with tenant and owner checks on every conversation access."""

from __future__ import annotations

import asyncio
import re
import sqlite3
import threading
import time
from uuid import uuid4

from .models import Document, Message, Principal


class ConversationNotFound(LookupError):
    pass


class SQLiteStore:
    """Local reference storage. Provision documents through trusted server code only."""

    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        self._db.executescript("""
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL
                    REFERENCES conversations(id),
                role TEXT NOT NULL, content TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                tenant_id TEXT NOT NULL, id TEXT NOT NULL,
                title TEXT NOT NULL, content TEXT NOT NULL,
                PRIMARY KEY (tenant_id, id)
            );
            CREATE TABLE IF NOT EXISTS document_readers (
                tenant_id TEXT NOT NULL, document_id TEXT NOT NULL, user_id TEXT NOT NULL,
                PRIMARY KEY (tenant_id, document_id, user_id),
                FOREIGN KEY (tenant_id, document_id) REFERENCES documents(tenant_id, id)
            );
        """)

    def close(self) -> None:
        self._db.close()

    def provision_document(
        self, tenant_id: str, document: Document, *, readers: frozenset[str]
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?) "
                "ON CONFLICT(tenant_id, id) DO UPDATE SET title=excluded.title, "
                "content=excluded.content",
                (tenant_id, document.id, document.title, document.content),
            )
            self._db.execute(
                "DELETE FROM document_readers WHERE tenant_id=? AND document_id=?",
                (tenant_id, document.id),
            )
            self._db.executemany(
                "INSERT INTO document_readers VALUES (?, ?, ?)",
                [(tenant_id, document.id, user_id) for user_id in readers],
            )

    async def search(
        self, principal: Principal, query: str, *, timeout_seconds: float
    ) -> tuple[Document, ...]:
        # Async timeout/cancellation must remain responsive while SQLite or scoring runs.
        deadline = time.monotonic() + timeout_seconds
        return await asyncio.to_thread(self._search, principal, query, deadline)

    def _search(self, principal: Principal, query: str, deadline: float) -> tuple[Document, ...]:
        def remaining() -> float:
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("document search deadline exhausted")
            return seconds

        # Permissions constrain candidates before ranking or model context construction.
        if not self._lock.acquire(timeout=remaining()):
            raise TimeoutError("document search lock timed out")
        try:
            self._db.execute(f"PRAGMA busy_timeout = {max(1, int(remaining() * 1000))}")
            self._db.set_progress_handler(lambda: time.monotonic() >= deadline, 100)
            rows = self._db.execute(
                "SELECT d.id, d.title, d.content FROM documents d "
                "JOIN document_readers r ON d.tenant_id=r.tenant_id AND d.id=r.document_id "
                "WHERE d.tenant_id=? AND r.user_id=? ORDER BY d.id LIMIT 1000",
                (principal.tenant_id, principal.user_id),
            ).fetchall()
        except sqlite3.OperationalError:
            remaining()  # Convert a deadline interruption into a retryable timeout.
            raise
        finally:
            self._db.set_progress_handler(None, 0)
            self._db.execute("PRAGMA busy_timeout = 5000")
            self._lock.release()
        remaining()
        terms = set(re.findall(r"\w+", query.casefold()))
        scored = []
        for doc_id, title, content in rows:
            remaining()
            scored.append(
                (
                    len(terms & set(re.findall(r"\w+", f"{title} {content}".casefold()))),
                    Document(id=doc_id, title=title, content=content),
                )
            )
        remaining()
        return tuple(
            doc for score, doc in sorted(scored, key=lambda x: (-x[0], x[1].id)) if score > 0
        )[:3]

    def open_conversation(self, principal: Principal, conversation_id: str | None) -> str:
        with self._lock, self._db:
            if conversation_id is None:
                conversation_id = str(uuid4())
                self._db.execute(
                    "INSERT INTO conversations VALUES (?, ?, ?)",
                    (conversation_id, principal.tenant_id, principal.user_id),
                )
            self._require_owner(principal, conversation_id)
        return conversation_id

    def _require_owner(self, principal: Principal, conversation_id: str) -> None:
        row = self._db.execute(
            "SELECT 1 FROM conversations WHERE id=? AND tenant_id=? AND user_id=?",
            (conversation_id, principal.tenant_id, principal.user_id),
        ).fetchone()
        if row is None:
            raise ConversationNotFound("conversation not found")

    def history(self, principal: Principal, conversation_id: str) -> tuple[Message, ...]:
        with self._lock:
            self._require_owner(principal, conversation_id)
            rows = self._db.execute(
                "SELECT role, content FROM messages WHERE conversation_id=? "
                "ORDER BY id DESC LIMIT 20",
                (conversation_id,),
            ).fetchall()
        return tuple(Message(role=role, content=content) for role, content in reversed(rows))

    def save_turn(
        self, principal: Principal, conversation_id: str, question: str, answer: str
    ) -> None:
        with self._lock, self._db:
            self._require_owner(principal, conversation_id)
            self._db.executemany(
                "INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)",
                [(conversation_id, "user", question), (conversation_id, "assistant", answer)],
            )
