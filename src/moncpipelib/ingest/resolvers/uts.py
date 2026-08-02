"""UTS Release API resolver for UMLS Metathesaurus + RxNorm.

The NLM UTS Release API
(`https://uts-ws.nlm.nih.gov/releases <https://uts-ws.nlm.nih.gov/releases>`_)
exposes both a current-release endpoint AND a list-historical endpoint
per release type.  This resolver queries the current endpoint via
:meth:`UtsReleaseResolver.current_release` and the list endpoint via
:meth:`UtsReleaseResolver.historical_release` (per #228), and produces
the authenticated download URL for the ``api_resolver`` materializer.

Two release types are supported, distinguished by the contract's
``resolver_config.release_type``:

- ``umls-full-release`` -- UMLS Metathesaurus full release (zip-of-zips
  per ADR-1's worked example).
- ``rxnorm-full-monthly-release`` -- RxNorm full monthly release.

Lifted from
``data-platform/pipelines/external-loads/src/external_loads/_uts_download.py``
and wrapped in the :class:`~moncpipelib.ingest.resolvers.ReleaseResolver`
Protocol per ADR-2.

Audit / compliance:

- The releases endpoint is unauthenticated; the api_key is only
  injected at :meth:`UtsReleaseResolver.resolve_url` time.
- All HTTP I/O uses
  :func:`moncpipelib.ingest._http.build_redacting_client`, so the
  api_key (which lives inside the URL query string) cannot appear in
  the audit log via the redacted transport hooks.  Callers that
  forward the URL further must avoid logging it from application code.
"""

from __future__ import annotations

from datetime import date
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlencode, urlparse

import ijson

from moncpipelib.ingest._http import build_redacting_client
from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.resolvers import ResolvedDownload

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moncpipelib.ingest.types import IngestContext


class _IterBytesReader:
    """Adapt an iterator of bytes chunks into a file-like ``read(n)`` interface.

    ``ijson.items`` heuristically dispatches by source type: it uses a
    chunked-read path when the source has ``read()`` (file-like) and a
    raw-event-stream path when the source only has ``__iter__``.  The
    iterator path expects each yielded value to be a parser EVENT
    tuple, not a raw bytes chunk -- feeding it ``response.iter_bytes()``
    directly raises ``ValueError("too many values to unpack")``.

    This adapter wraps an iterator of bytes into a minimal file-like
    so ``ijson.items`` takes the chunked-read path.  Memory footprint
    is bounded by the SDK's chunk size + ijson's read buffer (default
    64 KiB), regardless of total response size.
    """

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks: Iterator[bytes] = iter(chunks)
        self._leftover: bytes = b""
        self._exhausted = False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            # Drain the rest -- caller asked for everything.  ijson
            # never does this in the chunked-read path, but support it
            # for completeness so the file-like contract holds.
            rest = self._leftover
            self._leftover = b""
            for chunk in self._chunks:
                rest += chunk
            self._exhausted = True
            return rest

        while not self._exhausted and len(self._leftover) < size:
            try:
                self._leftover += next(self._chunks)
            except StopIteration:
                self._exhausted = True
                break

        out, self._leftover = self._leftover[:size], self._leftover[size:]
        return out


_RELEASES_URL = "https://uts-ws.nlm.nih.gov/releases"
_DOWNLOAD_URL = "https://uts-ws.nlm.nih.gov/download"

_KNOWN_RELEASE_TYPES: frozenset[str] = frozenset(
    {"umls-full-release", "rxnorm-full-monthly-release"}
)


class UtsReleaseResolver:
    """Release resolver for the NLM UTS Release API.

    Stateless singleton -- ``__init__`` is parameterless.  Implements
    :class:`~moncpipelib.ingest.resolvers.ReleaseResolver`.
    """

    name: ClassVar[str] = "uts_release"

    discovery_requires_auth: ClassVar[bool] = False
    """The UTS ``/releases`` endpoint is public; only ``/download``
    (resolved by :meth:`resolve_url`) requires the api_key.  Opting out
    of the discovery-time fetch lets ``build_discovery_sensor`` run on
    daemon pods without Azure workload-identity federation -- only the
    materialization step pods need Key Vault access.  Tracked via #253.
    """

    KNOWN_UTS_CONFIG_FIELDS: ClassVar[frozenset[str]] = frozenset({"release_type", "start_date"})

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Validate ``ingest.api_resolver.resolver_config`` for UTS.

        Per ADR-2: deterministic; no network; rejects unknown keys.
        Per #228: optional ``start_date`` bounds historical discovery.
        """
        errors: list[str] = []

        if "release_type" not in config:
            errors.append("release_type: required")
        else:
            value = config["release_type"]
            if not isinstance(value, str):
                errors.append("release_type: must be a string")
            elif value not in _KNOWN_RELEASE_TYPES:
                known = sorted(_KNOWN_RELEASE_TYPES)
                errors.append(f"release_type: must be one of {known}")

        if "start_date" in config:
            value = config["start_date"]
            if isinstance(value, date):
                pass  # date or datetime objects accepted
            elif isinstance(value, str):
                try:
                    date.fromisoformat(value)
                except ValueError:
                    errors.append("start_date: must be an ISO date string (YYYY-MM-DD)")
            else:
                errors.append("start_date: must be an ISO date string or date")

        for key in config:
            if key not in self.KNOWN_UTS_CONFIG_FIELDS:
                errors.append(f"{key}: unknown field")

        return errors

    def current_release(
        self,
        api_key: str | None,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> dict[str, Any]:
        """Return metadata for the current release of the configured type.

        The UTS releases endpoint is unauthenticated, so ``api_key`` is
        accepted but unused at this call site (Protocol uniformity).

        Returns a mapping with at least:

        - ``"partition_key"``: the release version (used as the
          dynamic partition key).
        - ``"release_version"``: same value, kept verbatim in the
          manifest's ``fields`` block so consumers using
          ``FromIngestTemplate.effective_from_field = "release_version"``
          can hydrate from it.
        - ``"download_url"``: the unauthenticated upstream URL.  The
          authenticated proxy URL is produced by
          :meth:`resolve_url` from this value plus ``api_key``.

        Raises:
            ValueError: When the UTS API returns an empty release list
                for the configured ``release_type``.
            httpx.HTTPStatusError: On a 4xx / 5xx response.
        """
        del api_key  # releases endpoint is unauthenticated
        del ctx  # no ctx fields needed at this site (yet)
        release_type = str(config["release_type"])

        with build_redacting_client() as client:
            response = client.get(
                _RELEASES_URL,
                params={"releaseType": release_type, "current": "true"},
            )
            response.raise_for_status()
            payload = response.json()

        if isinstance(payload, list):
            if not payload:
                raise ValueError(f"No current release found for release_type={release_type!r}")
            release = payload[0]
        else:
            release = payload
        if not release:
            raise ValueError(f"No current release found for release_type={release_type!r}")

        version = str(release["releaseVersion"])
        download_url = str(release["downloadUrl"])
        return {
            "partition_key": version,
            "release_version": version,
            "download_url": download_url,
        }

    def historical_release(
        self,
        api_key: str | None,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> list[dict[str, Any]]:
        """Return the full historical release list for the configured type.

        Hits ``/releases?releaseType=X`` (without ``current=true``) so
        UTS returns every release it still hosts.  Each entry maps to
        the same shape :meth:`current_release` returns, plus
        ``release_date`` (used for ``start_date`` filtering when
        configured).

        Per #228: bounds live in ``config``.  When
        ``config["start_date"]`` is set, releases earlier than that
        date are filtered out **during** iteration -- the unfiltered
        list never lands on the heap (#247 / Migration 012 Phase F).
        Lets a contract that started in 2024 avoid registering 2019
        releases UTS still happens to host.

        Streaming: the response body is parsed via
        :mod:`ijson.items` over the chunked HTTP stream so a release
        history of arbitrary size keeps peak heap bounded by one entry
        rather than the full array.  In practice UTS hosts <~1k
        releases per type, but future-proof the pattern in case the
        endpoint ever paginates differently.

        The releases endpoint itself is unauthenticated; ``api_key`` is
        accepted for Protocol uniformity but unused.

        Per #256: the result is memoized on ``ctx`` for the lifetime of
        the context.  Within a single materialization, both
        :meth:`resolve_url` and
        :class:`~moncpipelib.ingest.patterns.api_resolver.ApiResolverPattern.partition_metadata`
        invoke this method; the cache collapses them to one HTTP fetch.

        Raises:
            httpx.HTTPStatusError: On a 4xx / 5xx response.
        """
        del api_key  # releases endpoint is unauthenticated
        release_type = str(config["release_type"])
        start_date_filter = self._coerce_start_date(config.get("start_date"))

        cache_key = (
            "uts_release.historical",
            release_type,
            start_date_filter.isoformat() if start_date_filter is not None else None,
        )
        return ctx.get_or_compute(
            cache_key,
            lambda: self._historical_release_uncached(release_type, start_date_filter),
        )

    @staticmethod
    def _historical_release_uncached(
        release_type: str,
        start_date_filter: date | None,
    ) -> list[dict[str, Any]]:
        """Issue the actual HTTP fetch for ``historical_release``.

        Factored out from :meth:`historical_release` so the cache wrapper
        can hand a thunk to :meth:`IngestContext.get_or_compute` without
        re-entering the cache check.
        """
        releases: list[dict[str, Any]] = []
        with (
            build_redacting_client() as client,
            client.stream(
                "GET",
                _RELEASES_URL,
                params={"releaseType": release_type, "current": "false"},
            ) as response,
        ):
            response.raise_for_status()
            # Wrap the chunked HTTP body in a file-like adapter so
            # ``ijson.items`` takes its chunked-read path (the iterator
            # path expects parser events, not raw bytes).
            for entry in ijson.items(_IterBytesReader(response.iter_bytes()), "item"):
                if not entry:
                    continue
                version = str(entry["releaseVersion"])
                download_url = str(entry["downloadUrl"])
                release_date_raw = entry.get("releaseDate")
                release_date_str = str(release_date_raw) if release_date_raw is not None else None

                if start_date_filter is not None and release_date_str is not None:
                    try:
                        release_date_obj = date.fromisoformat(release_date_str)
                    except ValueError:
                        # Unparseable upstream date: include the release
                        # rather than silently dropping it.  Operators see
                        # the raw value in the partition_spec metadata.
                        release_date_obj = None
                    if release_date_obj is not None and release_date_obj < start_date_filter:
                        continue

                releases.append(
                    {
                        "partition_key": version,
                        "release_version": version,
                        "download_url": download_url,
                        "release_date": release_date_str,
                    }
                )
        return releases

    @staticmethod
    def _coerce_start_date(value: Any) -> date | None:
        """Parse the optional ``start_date`` config value.

        Accepts ``date``, ``datetime``, ISO string, or ``None``.
        ``validate_config`` already enforced format at contract-load
        time, so we only need to handle the runtime shape.
        """
        if value is None:
            return None
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))

    def resolve_url(
        self,
        api_key: str | None,
        partition_key: str,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> ResolvedDownload:
        """Return the authenticated UTS download URL for ``partition_key``.

        Per #228: looks up ``partition_key`` against
        :meth:`historical_release` first; falls back to
        :meth:`current_release` if the resolver doesn't surface it
        (defensive -- the historical endpoint should include current).
        Materialization for historical partitions works end-to-end.

        The returned :class:`ResolvedDownload`'s ``url`` embeds the
        ``api_key`` as a query parameter. Materializers MUST fetch it
        with :func:`~moncpipelib.ingest._http.build_redacting_client`
        so the api_key is not echoed into transport-layer logs.

        Per #270, the ``filename`` field is the basename of the
        upstream UMLS / RxNorm download URL (e.g.
        ``"umls-2026AB-full.zip"``) -- a stable semantic name the UTS
        API surfaces even when the materializer fetches via the proxied
        download endpoint with the api_key in the query string. This
        hint is propagated into the non-archive payload filename chain;
        for archive contracts (``extract: ["zip", "zip"]`` per the
        UMLS shape) it is unused because extraction yields per-member
        names instead.

        Raises:
            IngestResolutionError: When ``api_key is None`` (the UTS
                download endpoint requires authentication), or when no
                release matches ``partition_key`` (UTS has dropped that
                historical release).
        """
        if api_key is None:
            raise IngestResolutionError(
                "UtsReleaseResolver.resolve_url requires an api_key; "
                "the contract must declare a 'credential' block."
            )
        candidates = self.historical_release(api_key, config, ctx)
        if not candidates:
            candidates = [self.current_release(api_key, config, ctx)]

        match = next((r for r in candidates if r["partition_key"] == partition_key), None)
        if match is None:
            available = sorted(str(r["partition_key"]) for r in candidates)
            raise IngestResolutionError(
                f"UTS does not currently host release {partition_key!r}; "
                f"available releases: {available}"
            )
        release = match
        download_url = release["download_url"]
        params = urlencode({"url": download_url, "apiKey": api_key})
        # Use the UPSTREAM download_url's basename as the filename hint,
        # not the proxied UTS endpoint URL -- the proxied URL has no
        # semantic filename in its path.
        upstream_basename = PurePosixPath(urlparse(str(download_url)).path).name
        return ResolvedDownload(
            url=f"{_DOWNLOAD_URL}?{params}",
            filename=upstream_basename or None,
        )
