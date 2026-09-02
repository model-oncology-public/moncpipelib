"""``browser_export`` ingest pattern (#463).

For sources with **no addressable URL** -- the file only materializes as
the side effect of a headless browser clicking through a UI (e.g. a
DevExpress-generated report page that regenerates its export server-side
and streams the result to whichever tab happens to click "Export"). A
registered :class:`~moncpipelib.ingest.export_plans.ExportPlan` drives a
sanctioned :class:`~moncpipelib.ingest._browser.BrowserSession` through the
source's click-path and yields the files it downloads; this pattern owns
everything at the boundary:

- **Discovery** is pure clock arithmetic via
  :func:`~moncpipelib.ingest._cadence.cadence_boundary` -- one partition per
  cadence period, forward-only (no backfill: there is no URL with which to
  request a past period). Never touches a browser or imports playwright
  (D2), so a process that only discovers partitions never needs the
  ``browser`` extra installed.
- **Materialization** opens a :func:`~moncpipelib.ingest._browser.browser_session`
  (which runs the D2 chromium probe as its first act, before any page is
  loaded), drives the named export plan, and for each yielded
  :class:`~moncpipelib.ingest.export_plans.ExportedFile` hashes it, applies
  the optional ``validate_content`` byte check, resolves its landing
  filename, and hands it to the shared ``hash_compare_and_upload``.
- **Idempotency** is hash-compare only -- there is no ``idempotency`` config
  key (D8): a daily-regenerated export differs byte-wise even when
  semantically unchanged, so an inert "choice" would be a lie.
- **Untrusted filenames** (D8): the browser session already guards against
  local-disk traversal (the on-disk filename is session-generated, never
  derived from ``suggested_filename``); this pattern additionally sanitizes
  ``suggested_filename`` before it becomes a *blob path component*, and
  refuses to ever land a file under the reserved manifest name.
- **Failure semantics** (D7): any raise -- from the export plan, the
  zero-byte guard, ``validate_content``, or a duplicate-filename collision
  -- propagates uncaught. No manifest is written; the next run re-drives
  the browser from scratch. This is the same partial-write recovery mode
  every other pattern relies on (:func:`~moncpipelib.ingest.dispatcher.materialize_with_manifest`).

**Pass count over the payload, stated honestly** (D9's scoped exception to
the I/O-at-Boundaries "hash in the same pass as the write" rule): playwright
-- not this library -- performs the write for each download, so there is no
write pass here to piggyback a hash on. After the browser session has
produced a file on disk, this pattern reads it back in one chunked pass to
compute its sha256 + size (:func:`_hash_file`), then ``hash_compare_and_upload``
performs a second, independent streamed pass -- but only on a sha256
mismatch, i.e. zero passes when the content is unchanged. Peak Python heap
is bounded by the read chunk size regardless of file size. This is **one or
two bounded-chunk read passes**, never a single combined pass, and this
module must not be described otherwise. Transient on-disk footprint while a
:class:`~moncpipelib.ingest._browser.BrowserSession` is open is documented
there as ~2x a download's own size (playwright's own retained copy plus
``save_as``'s copy), not 1x -- this pattern does not change that.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, ClassVar

from moncpipelib.ingest._browser import (
    DEFAULT_DOWNLOAD_TIMEOUT_S,
    DEFAULT_HEADLESS,
    DEFAULT_NAVIGATION_TIMEOUT_S,
    browser_session,
)
from moncpipelib.ingest._cadence import cadence_boundary
from moncpipelib.ingest._hostnames import (
    malformed_allowed_hosts_reason,
    malformed_allowlist_host_reason,
)
from moncpipelib.ingest.dispatcher import _MANIFEST_FILENAME
from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.export_plans import get_export_plan
from moncpipelib.ingest.filenames import sanitize_blob_filename
from moncpipelib.ingest.patterns._upload import hash_compare_and_upload
from moncpipelib.ingest.prefix import render_payload_filename, render_prefix
from moncpipelib.ingest.types import IngestResult, PartitionSpec

if TYPE_CHECKING:
    from pathlib import Path

    from moncpipelib.contracts.models import IngestContract
    from moncpipelib.ingest.export_plans import ExportedFile
    from moncpipelib.ingest.types import IngestContext
    from moncpipelib.resources.blob import BlobStorageResource

_DEFAULT_MAX_FIRST_BYTES_CHECK: int = 256
"""Must equal ``http_urls._DEFAULT_MAX_FIRST_BYTES_CHECK`` -- pinned by
:func:`test_default_max_first_bytes_check_matches_http_urls`. Deliberately
duplicated rather than imported: the two patterns' ``validate_content``
implementations are independent (browser_export has no ``content_type_in``
support), and importing a private constant across pattern modules would
imply a coupling that does not exist."""

_DEFAULT_PUBLICATION_LAG_HOURS: int = 0
_HASH_CHUNK_BYTES: int = 1024 * 1024
_FIRST_BYTES_WINDOW: int = 65_536
"""Leading window read from a landed file for the ``validate_content`` byte
check -- bounded independent of file size, matching ``http_urls``' pattern
of only ever inspecting a fixed-size head."""

_ERR_SENSITIVITY_NOT_PUBLIC = (
    "browser_export: refusing to materialize a non-public-sensitivity contract"
)
_ERR_ZERO_BYTES = "browser_export: download produced zero bytes"
_ERR_CONTENT_REJECTED = "browser_export: download rejected by validate_content"
_ERR_STALE_PARTITION = "browser_export: refusing to materialize a stale partition key"
_ERR_ALLOWED_HOST_INVALID = (
    "browser_export: refusing to materialize with an invalid allowed_hosts entry"
)
_ERR_USER_AGENT_INVALID = "browser_export: refusing to materialize with an invalid fetch.user_agent"
_ERR_DUPLICATE_FILENAME = "browser_export: duplicate landed filename"
_ERR_NO_FILENAME = "browser_export: no usable filename"
_ERR_RESERVED_FILENAME = "browser_export: refusing reserved filename"

_DOT_ONLY_NAMES: frozenset[str] = frozenset({".", ".."})
"""``sanitize_blob_filename`` strips path separators and unsafe characters
but does not reject a name that survives sanitization as a bare
directory-reference component -- ``sanitize_blob_filename("..") == ".."``
verbatim. ``_resolve_filename`` rejects both explicitly rather than
landing ``<prefix>/..`` as a blob path."""


def _hash_file(path: Path) -> tuple[str, int]:
    """Compute sha256 + size for ``path`` in one chunked read pass.

    ``path`` is a file the :class:`~moncpipelib.ingest._browser.BrowserSession`
    already wrote via ``Download.save_as`` -- this is a *read-back*, not a
    write pass (see the module docstring's D9 note): there is no write this
    library performs that a hash could piggyback on. Never uses ``stat()``
    for size -- that would be a second source of truth, and the same class
    of bug the extractor's streaming hash helper (``_hash_stream_to_tempfile``)
    exists to avoid.
    """
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(_HASH_CHUNK_BYTES), b""):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _rejects_first_bytes(head: bytes, cfg: dict[str, Any]) -> bool:
    """Replicate ``http_urls._build_validator``'s reject branch exactly.

    Deliberately NOT a call to ``_build_validator`` (D7): its
    ``content_type_in`` branch rejects when the observed type is ``None``,
    which is every browser download -- reusing it would silently reject
    100% of ``browser_export`` traffic the moment a contract set
    ``content_type_in``. This pattern's loader validation forbids that key
    outright (see ``contracts/loader.py``), so only the byte-match half is
    needed here.
    """
    needles = [s.encode("utf-8").lower() for s in (cfg.get("reject_first_bytes_match") or [])]
    if not needles:
        return False
    limit = int(cfg.get("max_first_bytes_check", _DEFAULT_MAX_FIRST_BYTES_CHECK))
    stripped = head.lstrip()[:limit].lower()
    return any(stripped.startswith(n) for n in needles)


class BrowserExportPattern:
    """Headless-browser-driven ingest pattern (see module docstring)."""

    name: ClassVar[str] = "browser_export"

    def discover_partitions(
        self,
        contract: IngestContract,
        ctx: IngestContext,
    ) -> list[PartitionSpec]:
        """Return the single current cadence-boundary partition.

        Pure clock arithmetic -- touches no browser and imports no
        playwright (D2), so a process that only discovers partitions
        (e.g. a sensor tick) needs no ``browser`` extra. Forward-only;
        there is no URL with which to request a past period, so this
        always returns exactly one spec for the current boundary.
        """
        del ctx  # browser_export needs no secrets/log during discovery
        cfg = self._read_pattern_config(contract)
        partition_key = self._current_boundary_key(cfg)
        return [PartitionSpec(key=partition_key, metadata=self._metadata_for(cfg, partition_key))]

    def materialize_partition(
        self,
        contract: IngestContract,
        partition_spec: PartitionSpec,
        blob: BlobStorageResource,
        ctx: IngestContext,
    ) -> list[IngestResult]:
        """Drive the export plan through a sanctioned browser session and land its files.

        Ordering (D2, D7, D8, D9):

        0. The ``sensitivity`` backstop runs before anything else, including
           the stale-key guard (#464/#468 review-gate finding 3).
           ``validate_ingest_contract_schema`` already enforces
           ``sensitivity: public`` for ``browser_export`` at contract LOAD
           time -- but ``IngestContract`` is a plain dataclass, so any
           caller constructing one directly (bypassing the loader, as
           ``resolve_source_for_partition``-adjacent tooling or a hand-built
           test contract could) skips that check entirely. This is the same
           class of gap the repo's mode-scoped-flag invariant names: a
           control checked only by the loader needs a write/materialize-time
           backstop because a value arriving by another path skips the
           loader's static check.
        1. The stale-key guard then runs before anything else that can
           fail -- including before the chromium probe -- so a stale
           partition on a browser-free machine reports the real problem,
           and a working machine never launches chromium for a partition
           that will be refused regardless.
        2. The ``allowed_hosts`` + ``fetch.user_agent`` backstop (#464/#468
           round 3; unified onto both of the loader's checks per the #463
           egress-allowlist reframe migration doc's Step 1b) runs next,
           before ``browser_session`` opens: the identical hole the
           ``sensitivity`` backstop above closes -- ``validate_ingest_contract_schema``
           enforces both at contract load, but a hand-built ``IngestContract``
           skips that entirely. Each ``allowed_hosts`` entry is re-checked
           through the same ``moncpipelib.ingest._hostnames`` shape checks
           (list-level and per-entry) the loader uses -- there is exactly
           one implementation of each check, so this cannot drift from
           what the loader enforces; a present ``fetch.user_agent`` is re-checked for the
           same non-empty-printable-ASCII shape the loader requires (its
           *presence* is not re-enforced here -- an absent value degrades
           to chromium's default UA, a governance gap, rather than a
           header-injection-shaped one: a malformed value (e.g. embedded
           CRLF) would reach an HTTP header raw, which an absent value
           cannot).
        3. ``browser_session(...)`` opens (running the D2 probe as its
           first act) and drives ``plan.export(...)``.
        4. Each yielded file is hashed, zero-byte-checked, optionally
           content-validated, filename-resolved, duplicate-checked, and
           uploaded via the shared hash-compare helper.

        **The stale-key guard's rollover window, stated honestly (D4a.1).**
        The guard recomputes the boundary from the clock at materialize
        time and compares it to ``partition_spec.key`` -- it has no notion
        of "close enough". A Dagster retry that lands just after a cadence
        boundary rollover is therefore rejected as stale even though it is
        the *same* logical run: a key discovered at 23:55 in ``anchor_tz``
        and retried at 00:05 fails permanently, with a message that
        describes the situation as "stale" rather than "a rollover raced
        your retry". This is a deliberately accepted tradeoff, not a bug --
        the alternative (some window of "close enough" tolerance) would
        reintroduce exactly the silent-wrong-key risk this guard exists to
        close, and browser_export has no URL with which to safely re-request
        the same period the retry meant to target. Operationally, this
        means a retry policy with backoff crossing a cadence boundary will
        eventually fail a partition that would have succeeded on the first
        attempt; that failure is expected and requires a fresh partition
        run, not a retry of the same key.

        Any raise anywhere in this sequence propagates uncaught -- no
        manifest is written, and the next run re-drives the browser from
        scratch (the same partial-write recovery mode every pattern
        relies on).
        """
        if contract.sensitivity != "public":
            raise IngestResolutionError(
                f"{_ERR_SENSITIVITY_NOT_PUBLIC}: got sensitivity={contract.sensitivity!r}. "
                "browser_export drives a headless browser executing arbitrary "
                "third-party JavaScript and is restricted to 'public' sensitivity "
                "(see SECURITY.md, 'Browser-Driven Ingest Exports'). "
                "validate_ingest_contract_schema enforces this at contract load, "
                "but IngestContract is a plain dataclass any caller can construct "
                "directly -- this is the materialize-time backstop for that path."
            )

        cfg = self._read_pattern_config(contract)

        expected = self._current_boundary_key(cfg)
        if partition_spec.key != expected:
            raise IngestResolutionError(
                f"{_ERR_STALE_PARTITION}: got {partition_spec.key!r}, current "
                f"cadence boundary is {expected!r}. browser_export has no URL "
                "with which to request a past period, so materializing a stale "
                "key would land today's bytes under yesterday's key with an "
                "action='uploaded' manifest and no error."
            )

        prefix = render_prefix(contract.prefix_template, partition_spec.key, contract)
        browser_cfg: dict[str, Any] = cfg.get("browser") or {}
        fetch_cfg: dict[str, Any] = cfg.get("fetch") or {}
        validate_content_cfg: dict[str, Any] | None = cfg.get("validate_content")
        export_config: dict[str, Any] = cfg.get("export_config") or {}

        headless = bool(browser_cfg.get("headless", DEFAULT_HEADLESS))
        navigation_timeout_s = float(
            browser_cfg.get("navigation_timeout_s", DEFAULT_NAVIGATION_TIMEOUT_S)
        )
        download_timeout_s = float(
            browser_cfg.get("download_timeout_s", DEFAULT_DOWNLOAD_TIMEOUT_S)
        )
        ua_cfg = fetch_cfg.get("user_agent")
        user_agent = str(ua_cfg) if ua_cfg is not None else None

        try:
            allowed_hosts_cfg = cfg["allowed_hosts"]
        except KeyError as exc:
            raise IngestResolutionError(
                "browser_export: contract pattern_config missing 'allowed_hosts'"
            ) from exc

        # allowed_hosts backstop (#464/#468 round 3, unified per the #463
        # egress-allowlist reframe migration doc's Step 1b; round 6 drops
        # the deny-list entirely, see docs/migrations/20260807_463-egress-allowlist-reframe.md):
        # the loader enforces list shape (non-empty list) and per-entry
        # shape (bare hostname; no scheme, port, path, wildcard,
        # percent-encoding, whitespace, malformed DNS label, or non-ASCII
        # character) at contract LOAD time -- but IngestContract is a
        # plain dataclass, so a caller constructing one directly (the
        # identical bypass the sensitivity backstop above closes) skips
        # that check entirely. Both checks are re-run here from the
        # single shared `moncpipelib.ingest._hostnames` implementation the
        # loader uses, so this backstop cannot enforce a weaker or
        # different check than the loader does. #464/#468 round 4 (J6):
        # the list-level check below used to be a hand-rolled
        # `isinstance(..., list)` guard with no emptiness check, so
        # `allowed_hosts: []` passed it, ran the per-entry loop zero
        # times, and reached `browser_session` with a runtime allowlist
        # matching nothing -- fails CLOSED, but surfaces as a 60s
        # navigation timeout naming the wrong cause rather than this
        # diagnostic. `malformed_allowed_hosts_reason` also type-guards
        # `allowed_hosts_cfg` itself before it is iterated: a string would
        # otherwise iterate character by character, and a non-iterable
        # value would raise a raw `TypeError` rather than
        # `IngestResolutionError`. Runs before the export-plan lookup
        # below (and therefore before a browser can ever open) even
        # though it depends on nothing from that lookup, so a malformed
        # `allowed_hosts` list is refused regardless of deploy skew in
        # the export-plan registry.
        list_shape_reason = malformed_allowed_hosts_reason(allowed_hosts_cfg)
        if list_shape_reason is not None:
            raise IngestResolutionError(
                f"{_ERR_ALLOWED_HOST_INVALID}: 'allowed_hosts' {list_shape_reason}, "
                f"got {type(allowed_hosts_cfg).__name__}. "
                "validate_ingest_contract_schema enforces this at contract "
                "load, but IngestContract is a plain dataclass any caller "
                "can construct directly -- this is the materialize-time "
                "backstop for that path."
            )
        for h in allowed_hosts_cfg:
            shape_reason = malformed_allowlist_host_reason(h)
            if shape_reason is not None:
                raise IngestResolutionError(
                    f"{_ERR_ALLOWED_HOST_INVALID}: {h!r} {shape_reason}. "
                    "validate_ingest_contract_schema enforces this at contract "
                    "load, but IngestContract is a plain dataclass any caller "
                    "can construct directly -- this is the materialize-time "
                    "backstop for that path."
                )

        # fetch.user_agent backstop (#464/#468 round 3): validate_ingest_contract_schema
        # requires a present `fetch.user_agent` to be a non-empty, printable-ASCII
        # string (httpx encodes header values as strict ASCII, and a real
        # transport rejects control characters -- e.g. CRLF -- only at request
        # time). Only the FORMAT is re-checked here, not presence: an absent
        # value degrades to chromium's default UA, which is a governance gap,
        # not the header-injection-shaped hole a malformed value would be.
        if user_agent is not None and not (
            user_agent.strip() and user_agent.isascii() and user_agent.isprintable()
        ):
            raise IngestResolutionError(
                f"{_ERR_USER_AGENT_INVALID}: fetch.user_agent {user_agent!r} must "
                "be a non-empty, printable-ASCII string. "
                "validate_ingest_contract_schema enforces this at contract "
                "load, but IngestContract is a plain dataclass any caller "
                "can construct directly -- this is the materialize-time "
                "backstop for that path."
            )

        # Both dict-key lookups below are load-time-guaranteed by the
        # loader for a contract that went through it -- but `get_export_plan`
        # is NOT: the realistic trigger is deploy skew, where the process
        # that materializes registers a different export-plan set than
        # the process that validated the contract at load time. A raw
        # `KeyError` from either would violate the "parsers/patterns fail
        # loudly with IngestResolutionError, never a raw exception a
        # sensor body doesn't recognise" invariant.
        try:
            plan_name = cfg["export_plan"]
        except KeyError as exc:
            raise IngestResolutionError(
                "browser_export: contract pattern_config missing 'export_plan'"
            ) from exc
        try:
            plan = get_export_plan(str(plan_name))
        except KeyError as exc:
            raise IngestResolutionError(f"browser_export: {exc}") from exc

        results: list[IngestResult] = []
        seen: dict[str, str] = {}
        with browser_session(
            allowed_hosts=[str(h) for h in allowed_hosts_cfg],
            ctx=ctx,
            user_agent=user_agent,
            headless=headless,
            navigation_timeout_s=navigation_timeout_s,
            download_timeout_s=download_timeout_s,
        ) as session:
            for exported in plan.export(session, partition_spec.key, export_config, ctx):
                session.require_contains(exported.path)
                sha256, size_bytes = _hash_file(exported.path)
                if size_bytes == 0:
                    raise IngestResolutionError(
                        f"{_ERR_ZERO_BYTES} for partition {partition_spec.key!r}"
                    )
                if validate_content_cfg is not None:
                    with exported.path.open("rb") as fp:
                        head = fp.read(min(_FIRST_BYTES_WINDOW, size_bytes))
                    if _rejects_first_bytes(head, validate_content_cfg):
                        limit = int(
                            validate_content_cfg.get(
                                "max_first_bytes_check", _DEFAULT_MAX_FIRST_BYTES_CHECK
                            )
                        )
                        raise IngestResolutionError(
                            f"{_ERR_CONTENT_REJECTED} for partition "
                            f"{partition_spec.key!r}: first_bytes="
                            f"{head.lstrip()[:limit]!r}"
                        )
                filename = self._resolve_filename(contract, partition_spec.key, exported)
                if filename in seen:
                    raise IngestResolutionError(
                        f"{_ERR_DUPLICATE_FILENAME} {filename!r} for partition "
                        f"{partition_spec.key!r}: both {seen[filename]!r} and "
                        f"{exported.suggested_filename!r} resolved to it"
                    )
                seen[filename] = exported.suggested_filename
                results.append(
                    hash_compare_and_upload(
                        blob,
                        contract.sensitivity,
                        prefix,
                        filename,
                        exported.path,
                        sha256,
                        size_bytes,
                    )
                )
        return results

    def partition_metadata(
        self,
        contract: IngestContract,
        partition_key: str,
        ctx: IngestContext,
    ) -> dict[str, Any]:
        """Return the manifest ``fields`` block for ``partition_key`` (per #256).

        Echoes the passed-in ``partition_key`` verbatim -- never
        recomputes the boundary from the clock, since the dispatcher calls
        this *after* materialize and a run crossing midnight must not
        describe a different day than the partition it belongs to. Does
        NOT re-run the D4a.1 stale-key guard: ``materialize_partition``
        already did, and re-checking here would fail a materialization
        that succeeded legitimately just before a boundary rollover.
        """
        del ctx
        cfg = self._read_pattern_config(contract)
        return self._metadata_for(cfg, partition_key)

    def manifest_resolver_block(self, contract: IngestContract) -> dict[str, Any]:
        """Return the manifest ``resolver`` audit block: export plan + allowlist (C4).

        Records the ``allowed_hosts`` the contract declared for the
        partition, in its own durable audit record -- what the contract
        author declared, not evidence that any network boundary enforced
        it (``allowed_hosts`` is a contract-authoring mistake-catcher, not
        a security control; see SECURITY.md). Redaction contract, same as
        every other pattern's resolver block: persisted durably in
        ``_manifest.json`` and MUST NOT contain API keys, signed URLs, or
        PHI -- ``export_config`` is contract-authored and must uphold that
        contract itself.
        """
        cfg = contract.pattern_config
        return {
            "name": str(cfg.get("export_plan", "unknown")),
            "config": {
                "export_config": dict(cfg.get("export_config") or {}),
                "allowed_hosts": list(cfg.get("allowed_hosts") or []),
            },
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _read_pattern_config(contract: IngestContract) -> dict[str, Any]:
        cfg = contract.pattern_config
        if not cfg:
            raise IngestResolutionError(
                f"Contract {contract.source_name!r} has empty browser_export config"
            )
        return cfg

    @staticmethod
    def _current_boundary_key(cfg: dict[str, Any]) -> str:
        """Compute the current cadence-boundary partition key.

        Any :class:`ValueError` escaping :func:`cadence_boundary`, or a
        :class:`KeyError` from a missing ``partition`` block, is converted
        to :class:`IngestResolutionError` here so a pattern never leaks a
        raw exception into a sensor body. In practice the loader validates
        the presence of ``partition`` plus its cadence, tz, and lag at
        contract load, so this conversion is a belt, not the primary guard.
        """
        try:
            p = cfg["partition"]
        except KeyError as exc:
            raise IngestResolutionError(
                "browser_export: contract pattern_config missing 'partition'"
            ) from exc
        try:
            boundary = cadence_boundary(
                cadence=str(p["cadence"]),
                anchor_tz=str(p["anchor_tz"]),
                lag_hours=int(p.get("publication_lag_hours", _DEFAULT_PUBLICATION_LAG_HOURS)),
            )
        except ValueError as exc:
            raise IngestResolutionError(f"browser_export: {exc}") from exc
        return boundary.isoformat()

    @staticmethod
    def _metadata_for(cfg: dict[str, Any], partition_key: str) -> dict[str, Any]:
        p = cfg.get("partition") or {}
        return {
            "partition_key": partition_key,
            "snapshot_date": partition_key,
            "cadence": str(p.get("cadence", "")),
            "anchor_tz": str(p.get("anchor_tz", "")),
            "publication_lag_hours": int(
                p.get("publication_lag_hours", _DEFAULT_PUBLICATION_LAG_HOURS)
            ),
        }

    @staticmethod
    def _resolve_filename(
        contract: IngestContract,
        partition_key: str,
        exported: ExportedFile,
    ) -> str:
        """Resolve the blob filename for one exported file (D8).

        Two-level chain, both terminating in the reserved-name check:
        ``payload_filename_template`` (authored, used verbatim -- a
        malformed authored name fails loudly at upload rather than being
        silently rewritten), else ``sanitize_blob_filename(suggested_filename)``
        (untrusted, sanitized). Deliberately does NOT call
        ``resolve_payload_filename``: that helper's ``url`` argument is
        required and its terminal error talks about "the URL has no usable
        basename" -- a URL-less pattern passing ``url=""`` would emit that
        exact fiction D8 exists to avoid.
        """
        if contract.payload_filename_template:
            resolved = render_payload_filename(
                contract.payload_filename_template, partition_key, contract
            )
        else:
            sanitized = sanitize_blob_filename(exported.suggested_filename)
            if sanitized is None:
                raise IngestResolutionError(
                    f"{_ERR_NO_FILENAME} for partition {partition_key!r}: the "
                    f"export's suggested_filename {exported.suggested_filename!r} "
                    "sanitized to nothing. Set 'ingest.payload_filename_template' "
                    "on the contract."
                )
            resolved = sanitized
        if resolved in _DOT_ONLY_NAMES:
            raise IngestResolutionError(
                f"{_ERR_NO_FILENAME} for partition {partition_key!r}: the resolved "
                f"filename {resolved!r} is a bare directory-reference component, "
                "not a usable blob filename -- 'sanitize_blob_filename' strips "
                "path separators and unsafe characters but does not reject a "
                "name that survives as '.' or '..'. Set "
                "'ingest.payload_filename_template' on the contract."
            )
        if resolved == _MANIFEST_FILENAME:
            raise IngestResolutionError(
                f"{_ERR_RESERVED_FILENAME} {_MANIFEST_FILENAME!r} for partition "
                f"{partition_key!r} -- that name is reserved for the partition's "
                "own manifest, and writing it would forge the partition's audit "
                "record."
            )
        return resolved
