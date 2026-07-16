"""Crawl plan Protocol and registry for the ``api_crawl`` ingest pattern (#415).

Crawl plans are the bridge between an ``api_crawl`` ingest contract and
a query API that has **no bulk download** -- the dataset only exists as
the union of many small JSON responses (e.g. RxClass: enumerate ~16k
classes, then one ``classMembers`` call per class).  Each plan knows
how to walk its specific API and yield the assembled records.

Design (per the #415 migration plan, decision D1): the plan is
**imperative**, not declarative.  It receives a throttled, redacting
client and yields :class:`CrawlRecord`s one at a time -- fan-out
(enumerate then per-key calls) and pagination (cursor from the previous
response) both collapse to "yield as you go".  The pattern -- not the
plan -- owns rate-limit pacing, NDJSON assembly with hash-in-the-same-
pass, blob upload, and the manifest.

Plan-author contract:

- **Deterministic ordering** (D4): enumerate and emit in a stable order
  (sort your keys).  The landed blob's sha256 is the idempotency key;
  nondeterministic ordering defeats hash-compare skips on re-runs.
- **Client-only HTTP**: every request goes through the provided
  :class:`~moncpipelib.ingest._throttle.ThrottledClient`.  Constructing
  your own ``httpx.Client`` bypasses both the rate budget and the
  transport-log redaction -- a SOC 2 / HITRUST audit violation (see
  ``SECURITY.md``).
- **No import-time network** -- same rule as patterns and resolvers;
  ``Definitions(...)`` construction must stay network-free.
- **No secrets / PHI in records**: yielded records land in durable
  blobs.  api_keys, signed URLs, and anything beyond the source's
  declared sensitivity classification must never appear in a record.
- **validate_config** runs at contract-load time -- including in CI.
  It MUST be deterministic, MUST NOT make network calls, MUST NOT
  perform filesystem I/O, and MUST reject unknown keys (the same ADR-2
  contract as :meth:`ReleaseResolver.validate_config`).

Like resolvers, plans are stateless singletons: one instance per
registered :attr:`CrawlPlan.name`, parameterless ``__init__``, all
per-call state through method arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from moncpipelib.ingest._throttle import ThrottledClient
    from moncpipelib.ingest.types import IngestContext


@dataclass(frozen=True)
class CrawlRecord:
    """One assembled record destined for one NDJSON blob.

    Attributes:
        filename: Target blob filename within the partition prefix
            (e.g. ``"has_epc.ndjson"``).  Plans may spread records
            across a **small** number of filenames (one open tempfile
            handle each for the duration of the crawl); the filename
            set must be bounded by a handful, not by the key space.
            Like resolver filename hints, plan-authored filenames are
            NOT sanitized -- a malformed name fails loudly at upload
            time rather than being silently rewritten.
        record: One JSON-serializable mapping.  Serialized by the
            pattern as a single NDJSON line with ``sort_keys=True``
            (deterministic bytes, per D4).  Redaction contract above:
            no secrets, no PHI beyond the source's classification.
    """

    filename: str
    record: Mapping[str, Any]


@runtime_checkable
class CrawlPlan(Protocol):
    """Protocol every crawl plan implements.

    Stateless; one instance per registered name.  See the module
    docstring for the plan-author contract (determinism, client-only
    HTTP, no import-time network, record redaction, ADR-2 validation).
    """

    name: ClassVar[str]

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Return a list of validation error strings (empty if valid).

        Called by ``_validate_api_crawl_block`` at contract-load time
        with the contents of ``ingest.api_crawl.crawl_config``.  Plans
        should validate required keys, value types, plan-specific
        format rules, AND reject unknown keys (per ADR-2).

        Network calls are forbidden; filesystem I/O is forbidden; the
        function must be deterministic and fast (target < 1ms).
        """
        ...

    def crawl(
        self,
        client: ThrottledClient,
        api_key: str | None,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> Iterator[CrawlRecord]:
        """Walk the upstream API and yield assembled records.

        Runs at materialize time only (never at discovery or import
        time).  ``client`` is the rate-limited, redacting HTTP surface
        -- the ONLY sanctioned way to reach the network.  ``api_key``
        is ``None`` for credential-less contracts (public APIs like
        RxClass); plans that require authentication should raise
        :class:`~moncpipelib.ingest.exceptions.IngestResolutionError`
        when it is ``None``.

        Yield records incrementally -- never accumulate the full
        assembly in memory (the I/O-at-Boundaries invariant; the
        pattern streams each record to disk as it arrives).  Any raise
        aborts the partition: the dispatcher will not write a manifest
        and the next run re-crawls from scratch (D5).
        """
        ...


CRAWL_PLANS: dict[str, CrawlPlan] = {}


def register_crawl_plan(plan: CrawlPlan) -> None:
    """Register ``plan`` under its :attr:`CrawlPlan.name`.

    Subsequent calls with the same name overwrite the previous entry --
    useful for testing with a stub, never intended for production.

    Per-source plans live with their consumers (data-platform) and are
    registered from consumer code, following the resolver-registry
    precedent; moncpipelib ships no builtin plans today.
    """
    CRAWL_PLANS[plan.name] = plan


def get_crawl_plan(name: str) -> CrawlPlan:
    """Look up a registered crawl plan by name.

    Raises:
        KeyError: If no plan with that name is registered.  The message
            lists the known plans, so a YAML typo surfaces at
            contract-load time with a useful suggestion.
    """
    try:
        return CRAWL_PLANS[name]
    except KeyError as e:
        known = sorted(CRAWL_PLANS)
        raise KeyError(f"Unknown crawl plan {name!r}. Known crawl plans: {known}") from e


__all__ = [
    "CRAWL_PLANS",
    "CrawlPlan",
    "CrawlRecord",
    "get_crawl_plan",
    "register_crawl_plan",
]
