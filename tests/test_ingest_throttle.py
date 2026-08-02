"""Tests for the ThrottledClient rate limiter (#415 Phase 2)."""

from __future__ import annotations

import httpx
import pytest

from moncpipelib.ingest._throttle import ThrottledClient


class FakeClock:
    """Deterministic monotonic clock; ``sleep`` advances it and records calls."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_client(handler: httpx.MockTransport | None = None) -> httpx.Client:
    transport = handler or httpx.MockTransport(lambda _request: httpx.Response(200, json={"ok": 1}))
    return httpx.Client(transport=transport)


class TestThrottledClient:
    def test_first_request_does_not_sleep(self) -> None:
        fake = FakeClock()
        with make_client() as client:
            throttled = ThrottledClient.from_rate_limit(
                client, rate_limit_rps=5, clock=fake.clock, sleep=fake.sleep
            )
            throttled.get("https://example.test/a")
        assert fake.sleeps == []

    def test_back_to_back_requests_sleep_residual_interval(self) -> None:
        fake = FakeClock()
        with make_client() as client:
            throttled = ThrottledClient.from_rate_limit(
                client, rate_limit_rps=5, clock=fake.clock, sleep=fake.sleep
            )
            throttled.get("https://example.test/a")
            fake.advance(0.05)  # 50ms elapsed; interval is 200ms
            throttled.get("https://example.test/b")
        assert len(fake.sleeps) == 1
        assert fake.sleeps[0] == pytest.approx(0.15)

    def test_spaced_requests_do_not_sleep(self) -> None:
        fake = FakeClock()
        with make_client() as client:
            throttled = ThrottledClient.from_rate_limit(
                client, rate_limit_rps=5, clock=fake.clock, sleep=fake.sleep
            )
            throttled.get("https://example.test/a")
            fake.advance(0.5)  # well past the 200ms interval
            throttled.get("https://example.test/b")
        assert fake.sleeps == []

    def test_sequence_of_requests_paces_each_gap(self) -> None:
        fake = FakeClock()
        with make_client() as client:
            throttled = ThrottledClient.from_rate_limit(
                client, rate_limit_rps=10, clock=fake.clock, sleep=fake.sleep
            )
            for _ in range(4):
                throttled.get("https://example.test/x")
        # first is free; each subsequent back-to-back call sleeps the full 100ms
        assert fake.sleeps == pytest.approx([0.1, 0.1, 0.1])

    def test_params_are_forwarded(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={})

        fake = FakeClock()
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            throttled = ThrottledClient(
                client, min_interval_s=0, clock=fake.clock, sleep=fake.sleep
            )
            throttled.get("https://example.test/q", params={"classId": "N0000175605", "page": 2})
        assert seen[0].url.params["classId"] == "N0000175605"
        assert seen[0].url.params["page"] == "2"

    def test_raise_for_status_propagates(self) -> None:
        fake = FakeClock()
        transport = httpx.MockTransport(lambda _request: httpx.Response(503))
        with httpx.Client(transport=transport) as client:
            throttled = ThrottledClient(
                client, min_interval_s=0, clock=fake.clock, sleep=fake.sleep
            )
            with pytest.raises(httpx.HTTPStatusError):
                throttled.get("https://example.test/down")

    def test_rate_limit_must_be_positive(self) -> None:
        with make_client() as client:
            with pytest.raises(ValueError, match="rate_limit_rps must be > 0"):
                ThrottledClient.from_rate_limit(client, rate_limit_rps=0)
            with pytest.raises(ValueError, match="rate_limit_rps must be > 0"):
                ThrottledClient.from_rate_limit(client, rate_limit_rps=-3)

    def test_min_interval_must_be_non_negative(self) -> None:
        with make_client() as client, pytest.raises(ValueError, match="min_interval_s"):
            ThrottledClient(client, min_interval_s=-0.1)

    def test_surface_is_get_only(self) -> None:
        with make_client() as client:
            throttled = ThrottledClient(client, min_interval_s=0)
            assert not hasattr(throttled, "post")
            assert not hasattr(throttled, "put")
            assert not hasattr(throttled, "delete")
