"""Retry retrieval calls while sharing one overall time budget."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RetrievalTimeoutError(TimeoutError):
    """Raised when a retrieval operation exhausts its total time budget."""


def retrieve_with_retry(
    retrieve: Callable[..., T],
    *,
    timeout_seconds: float = 5.0,
    total_timeout_seconds: float = 5.0,
    max_retries: int = 2,
    retry_delay_seconds: float = 0.1,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``retrieve`` with bounded retries under a single deadline.

    The callback receives ``timeout_seconds`` (capped to the remaining total
    budget). ``max_retries`` counts retries after the initial attempt.
    Exceptions from the callback are retried until that limit is reached.
    The callback must enforce its supplied timeout; this synchronous wrapper
    cannot interrupt it. Results arriving at or after the deadline are rejected.
    """
    if timeout_seconds <= 0 or total_timeout_seconds <= 0:
        raise ValueError("timeouts must be greater than zero")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative")

    deadline = clock() + total_timeout_seconds
    retries = 0
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise RetrievalTimeoutError("retrieval exceeded its total time budget")

        try:
            result = retrieve(timeout_seconds=min(timeout_seconds, remaining))
        except Exception:
            if deadline - clock() <= 0:
                raise RetrievalTimeoutError(
                    "retrieval exceeded its total time budget"
                ) from None
            if retries >= max_retries:
                raise
        else:
            if deadline - clock() <= 0:
                raise RetrievalTimeoutError("retrieval exceeded its total time budget")
            return result

        remaining = deadline - clock()
        if remaining <= 0:
            raise RetrievalTimeoutError("retrieval exceeded its total time budget")
        delay = min(retry_delay_seconds, remaining)
        if delay:
            sleep(delay)
        retries += 1
        if deadline - clock() <= 0:
            raise RetrievalTimeoutError("retrieval exceeded its total time budget")
