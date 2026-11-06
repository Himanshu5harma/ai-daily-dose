import pytest

from ai_engineering.services.retrieval_retry import (
    RetrievalTimeoutError,
    retrieve_with_retry,
)


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_fast_success_returns_normally_and_passes_budget():
    fake = FakeTime()
    seen_timeouts = []

    result = retrieve_with_retry(
        lambda *, timeout_seconds: seen_timeouts.append(timeout_seconds) or "ok",
        timeout_seconds=4,
        total_timeout_seconds=5,
        clock=fake.clock,
        sleep=fake.sleep,
    )

    assert result == "ok"
    assert seen_timeouts == [4]


def test_remaining_budget_caps_attempt_timeout():
    fake = FakeTime()
    seen_timeouts = []
    calls = 0

    def retrieve(*, timeout_seconds):
        nonlocal calls
        calls += 1
        seen_timeouts.append(timeout_seconds)
        fake.now += 1
        if calls == 1:
            raise RuntimeError("temporary failure")
        return "ok"

    assert retrieve_with_retry(
        retrieve,
        timeout_seconds=10,
        total_timeout_seconds=3,
        retry_delay_seconds=0.5,
        clock=fake.clock,
        sleep=fake.sleep,
    ) == "ok"
    assert seen_timeouts == [3, 1.5]
    assert fake.sleeps == [0.5]


def test_exhausted_deadline_does_not_start_another_attempt():
    fake = FakeTime()
    calls = 0

    def retrieve(*, timeout_seconds):
        nonlocal calls
        calls += 1
        fake.now += timeout_seconds
        raise RuntimeError("slow failure")

    with pytest.raises(RetrievalTimeoutError):
        retrieve_with_retry(
            retrieve,
            timeout_seconds=10,
            total_timeout_seconds=2,
            max_retries=5,
            clock=fake.clock,
            sleep=fake.sleep,
        )
    assert calls == 1


def test_retry_sleep_is_capped_and_no_attempt_starts_at_deadline():
    fake = FakeTime()
    calls = []

    def retrieve(*, timeout_seconds):
        calls.append(fake.now)
        fake.now += 1
        raise RuntimeError("temporary failure")

    with pytest.raises(RetrievalTimeoutError):
        retrieve_with_retry(
            retrieve,
            total_timeout_seconds=3,
            retry_delay_seconds=5,
            clock=fake.clock,
            sleep=fake.sleep,
        )

    assert calls == [0]
    assert fake.sleeps == [2]
    assert fake.now == 3


def test_repeated_failures_share_one_deadline():
    fake = FakeTime()
    calls = []

    def retrieve(*, timeout_seconds):
        calls.append(fake.now)
        fake.now += 0.5
        raise RuntimeError("temporary failure")

    with pytest.raises(RetrievalTimeoutError):
        retrieve_with_retry(
            retrieve,
            total_timeout_seconds=2,
            max_retries=10,
            retry_delay_seconds=0.5,
            clock=fake.clock,
            sleep=fake.sleep,
        )

    assert calls == [0, 1]
    assert fake.sleeps == [0.5, 0.5]


@pytest.mark.parametrize("elapsed", [5, 6])
def test_success_at_or_after_deadline_is_rejected(elapsed):
    fake = FakeTime()

    def retrieve(*, timeout_seconds):
        fake.now += elapsed
        return "late result"

    with pytest.raises(RetrievalTimeoutError):
        retrieve_with_retry(retrieve, clock=fake.clock, sleep=fake.sleep)
    assert fake.sleeps == []


def test_retry_limit_preserves_original_failure_before_deadline():
    fake = FakeTime()
    failure = RuntimeError("retrieval failed")
    calls = []

    def retrieve(*, timeout_seconds):
        calls.append(fake.now)
        raise failure

    with pytest.raises(RuntimeError) as exc_info:
        retrieve_with_retry(
            retrieve, max_retries=1, clock=fake.clock, sleep=fake.sleep
        )
    assert exc_info.value is failure
    assert len(calls) == 2
    assert fake.sleeps == [0.1]
