"""``api_crawl`` ingest pattern (#415).

For sources with **no bulk download** -- the dataset only exists as the
union of many small JSON responses from a query API (first consumer:
RxClass via RxNav, ~16k enumerated classes -> one ``classMembers`` call
each).  A registered :class:`~moncpipelib.ingest.crawl_plans.CrawlPlan`
walks the API and yields records; this pattern owns everything at the
boundary:

- **Discovery** delegates to a registered
  :class:`~moncpipelib.ingest.resolvers.ReleaseResolver` exactly as
  ``api_resolver`` does (shared via ``_resolver_discovery``); crawl
  resolvers implement ``resolve_url`` to raise, since there is no
  download URL.
- **Rate limiting**: all plan HTTP goes through a
  :class:`~moncpipelib.ingest._throttle.ThrottledClient` built from the
  contract's required ``rate_limit_rps`` (RxNav caps at 20 req/s/IP).
  The wrapped client comes from :func:`build_redacting_client`, so
  transport logs never carry query strings or headers.
- **Streaming assembly**: each record is serialized as one NDJSON line
  (``sort_keys=True`` -- deterministic bytes, decision D4) into a
  per-filename :func:`~moncpipelib.ingest._hashing.hashing_tempfile`.
  Peak memory is bounded by one record + one response, never the
  assembled payload (I/O-at-Boundaries invariant).
- **Idempotency + manifest** are inherited: uploads go through the
  shared ``hash_compare_and_upload`` and the dispatcher writes
  ``_manifest.json``.  Note the sha-skip is best-effort here -- a live
  API only reproduces bytes when the upstream is unchanged AND the plan
  emits deterministically (a documented plan requirement).
- **Failure semantics** (decision D5): any plan exception propagates
  uncaught -- no manifest is written and the next run re-crawls from
  scratch.  Checkpoint/resume is explicit future work.

Never touches the network at import time (Protocol requirement).
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, ClassVar

from moncpipelib.ingest._hashing import HashingTempfileWriter, hashing_tempfile
from moncpipelib.ingest._http import build_redacting_client
from moncpipelib.ingest._throttle import ThrottledClient
from moncpipelib.ingest.crawl_plans import get_crawl_plan
from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.patterns._resolver_discovery import (
    discover_partitions_via_resolver,
    fetch_api_key,
    partition_metadata_via_resolver,
)
from moncpipelib.ingest.patterns._upload import hash_compare_and_upload
from moncpipelib.ingest.prefix import render_prefix
from moncpipelib.ingest.types import IngestResult, PartitionSpec

if TYPE_CHECKING:
    from moncpipelib.contracts.models import IngestContract
    from moncpipelib.ingest.types import IngestContext
    from moncpipelib.resources.blob import BlobStorageResource

_DEFAULT_TIMEOUT_S: float = 60.0
"""Default per-request timeout.  Crawl responses are small JSON
documents -- deliberately far below api_resolver's 3600s archive-
download default."""

_DEFAULT_CONNECT_TIMEOUT_S: float = 30.0
_DEFAULT_RETRIES: int = 3


class ApiCrawlPattern:
    """REST-crawl ingest pattern (see module docstring).

    The contract's ``api_crawl`` block names a registered
    :class:`CrawlPlan` (``crawl_plan`` + ``crawl_config``), a registered
    :class:`ReleaseResolver` for period discovery (``resolver`` +
    ``resolver_config`` + ``partition``), an optional ``credential``
    block, an optional ``fetch`` block, and the required
    ``rate_limit_rps``.
    """

    name: ClassVar[str] = "api_crawl"

    def discover_partitions(
        self,
        contract: IngestContract,
        ctx: IngestContext,
    ) -> list[PartitionSpec]:
        """Resolver-driven discovery, shared with ``api_resolver``.

        For version-driven crawl sources the resolver reads a cheap
        version endpoint (e.g. RxClass ``/version``) and emits a new
        partition key when a tracked source version changes; version-
        less sub-sources use a time-based fallback inside the
        resolver's ``current_release``.  Discovery stays cheap -- the
        crawl itself runs only at materialize time.
        """
        cfg = self._read_pattern_config(contract)
        return discover_partitions_via_resolver(cfg, ctx, pattern_name="ApiCrawlPattern")

    def materialize_partition(
        self,
        contract: IngestContract,
        partition_spec: PartitionSpec,
        blob: BlobStorageResource,
        ctx: IngestContext,
    ) -> list[IngestResult]:
        """Execute the crawl plan and land the assembled NDJSON blob(s).

        Records are streamed to per-filename hashing tempfiles as they
        are yielded; on crawl completion each file goes through the
        shared hash-compare upload.  An empty crawl raises
        :class:`IngestResolutionError` -- zero records from a live API
        is an upstream failure signal, not an empty partition.
        """
        cfg = self._read_pattern_config(contract)
        api_key = fetch_api_key(ctx, cfg, pattern_name="ApiCrawlPattern")
        plan = get_crawl_plan(cfg["crawl_plan"])
        crawl_config = cfg.get("crawl_config") or {}
        prefix = render_prefix(contract.prefix_template, partition_spec.key, contract)

        fetch_cfg: dict[str, Any] = cfg.get("fetch", {}) or {}
        retries = int(fetch_cfg.get("retries", _DEFAULT_RETRIES))
        timeout_s = float(fetch_cfg.get("timeout_s", _DEFAULT_TIMEOUT_S))
        connect_timeout_s = float(fetch_cfg.get("connect_timeout_s", _DEFAULT_CONNECT_TIMEOUT_S))
        ua_cfg = fetch_cfg.get("user_agent")
        user_agent = str(ua_cfg) if ua_cfg is not None else None

        results: list[IngestResult] = []
        with (
            build_redacting_client(
                timeout_s=timeout_s,
                connect_timeout_s=connect_timeout_s,
                retries=retries,
                follow_redirects=True,
                user_agent=user_agent,
            ) as client,
            ExitStack() as stack,
        ):
            throttled = ThrottledClient.from_rate_limit(
                client, rate_limit_rps=float(cfg["rate_limit_rps"])
            )
            # Insertion-ordered: upload order follows first-yield order,
            # keeping the IngestResult list deterministic given a
            # deterministic plan (D4).
            writers: dict[str, HashingTempfileWriter] = {}
            record_count = 0
            for crawl_record in plan.crawl(throttled, api_key, crawl_config, ctx):
                writer = writers.get(crawl_record.filename)
                if writer is None:
                    writer = stack.enter_context(hashing_tempfile(suffix=".ndjson"))
                    writers[crawl_record.filename] = writer
                # sort_keys => deterministic bytes for identical records;
                # a non-JSON-serializable value fails loudly here (plans
                # own producing JSON-safe records).
                line = json.dumps(dict(crawl_record.record), sort_keys=True, separators=(",", ":"))
                writer.write(line.encode("utf-8") + b"\n")
                record_count += 1

            if record_count == 0:
                raise IngestResolutionError(
                    f"crawl plan {cfg['crawl_plan']!r} yielded zero records for "
                    f"partition {partition_spec.key!r}.  An empty crawl from a "
                    "live API signals an upstream or plan failure; refusing to "
                    "land an empty partition."
                )

            for filename, writer in writers.items():
                writer.close()
                results.append(
                    hash_compare_and_upload(
                        blob,
                        contract.sensitivity,
                        prefix,
                        filename,
                        writer.path,
                        writer.sha256_hexdigest(),
                        writer.size_bytes,
                    )
                )
        return results

    def partition_metadata(
        self,
        contract: IngestContract,
        partition_key: str,
        ctx: IngestContext,
    ) -> dict[str, Any]:
        """Return the resolver's release dict for ``partition_key`` (per #256).

        Same manifest-fields mechanism as ``api_resolver`` -- the
        dispatcher re-asks the pattern at manifest-write time; shared
        implementation in ``_resolver_discovery``.
        """
        cfg = self._read_pattern_config(contract)
        return partition_metadata_via_resolver(
            cfg, partition_key, ctx, pattern_name="ApiCrawlPattern"
        )

    def manifest_resolver_block(self, contract: IngestContract) -> dict[str, Any]:
        """Return the manifest's audit block: crawl plan + resolver (per #415).

        Names both extension points that produced the partition -- the
        crawl plan that assembled the payload and the resolver that
        keyed the period.  Redaction contract: persisted durably in
        ``_manifest.json`` -- no api_keys, signed URLs, or PHI.
        """
        cfg = contract.pattern_config
        return {
            "name": str(cfg.get("crawl_plan", "unknown")),
            "config": {
                "resolver": cfg.get("resolver"),
                "resolver_config": dict(cfg.get("resolver_config") or {}),
                "crawl_config": dict(cfg.get("crawl_config") or {}),
            },
        }

    @staticmethod
    def _read_pattern_config(contract: IngestContract) -> dict[str, Any]:
        cfg = contract.pattern_config
        if not cfg:
            raise IngestResolutionError(
                f"Contract {contract.source_name!r} has empty api_crawl config"
            )
        return cfg
