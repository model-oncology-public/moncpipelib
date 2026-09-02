"""Tests for the CrawlPlan protocol + registry (#415 Phase 4)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

from moncpipelib.ingest._throttle import ThrottledClient
from moncpipelib.ingest.crawl_plans import (
    CRAWL_PLANS,
    CrawlPlan,
    CrawlRecord,
    get_crawl_plan,
    register_crawl_plan,
)
from moncpipelib.ingest.types import IngestContext


class _StubPlan:
    name: ClassVar[str] = "stub_plan"

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def crawl(
        self,
        client: ThrottledClient,
        api_key: str | None,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> Iterator[CrawlRecord]:
        del client, api_key, config, ctx
        yield CrawlRecord(filename="a.ndjson", record={"k": 1})


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Snapshot + restore the registry so tests don't leak stubs."""
    before = dict(CRAWL_PLANS)
    try:
        yield
    finally:
        CRAWL_PLANS.clear()
        CRAWL_PLANS.update(before)


def test_register_and_lookup_round_trip() -> None:
    plan = _StubPlan()
    register_crawl_plan(plan)
    assert get_crawl_plan("stub_plan") is plan


def test_unknown_plan_raises_with_known_names() -> None:
    register_crawl_plan(_StubPlan())
    with pytest.raises(KeyError, match=r"Unknown crawl plan 'nope'.*stub_plan"):
        get_crawl_plan("nope")


def test_no_builtin_plans_ship_with_moncpipelib() -> None:
    """Per-source plans live with consumers (data-platform); the
    library registry starts empty."""
    assert CRAWL_PLANS == {}


def test_reregistration_overwrites() -> None:
    first = _StubPlan()
    second = _StubPlan()
    register_crawl_plan(first)
    register_crawl_plan(second)
    assert get_crawl_plan("stub_plan") is second


def test_stub_satisfies_runtime_checkable_protocol() -> None:
    assert isinstance(_StubPlan(), CrawlPlan)


def test_crawl_record_is_frozen() -> None:
    record = CrawlRecord(filename="a.ndjson", record={"k": 1})
    with pytest.raises(AttributeError):
        record.filename = "b.ndjson"  # type: ignore[misc]
