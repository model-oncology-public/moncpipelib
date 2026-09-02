"""Universal blob-landing ingest boundary.

See ``docs/migrations/20260423_209-ingest-boundary-phase1.md`` for
the Phase 1 scope and rationale.
"""

from __future__ import annotations

from moncpipelib.ingest._browser import BrowserSession

# `browser_session` (the session factory) and `ensure_chromium_available`
# are deliberately NOT exported here: plan authors need the BrowserSession
# *type* to annotate their `export` signature, but they never construct a
# session themselves -- the `browser_export` pattern (Step 3) is the only
# sanctioned caller. This is a signal, not a wall: nothing stops
# `from moncpipelib.ingest._browser import browser_session` directly.
from moncpipelib.ingest._throttle import ThrottledClient
from moncpipelib.ingest.crawl_plans import (
    CRAWL_PLANS,
    CrawlPlan,
    CrawlRecord,
    get_crawl_plan,
    register_crawl_plan,
)
from moncpipelib.ingest.dispatcher import materialize_with_manifest
from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.export_plans import (
    EXPORT_PLANS,
    ExportedFile,
    ExportPlan,
    get_export_plan,
    register_export_plan,
)
from moncpipelib.ingest.filenames import sanitize_blob_filename
from moncpipelib.ingest.manifest import (
    KNOWN_MAX_VERSION,
    IngestManifest,
    ManifestFileEntry,
)
from moncpipelib.ingest.partition_reader import (
    ManifestFieldError,
    download_partition_parts_with_manifest,
    read_partition_with_manifest,
)
from moncpipelib.ingest.patterns import (
    BUILTIN_PATTERN_NAMES,
    INGEST_PATTERNS,
    IngestPattern,
    get_pattern,
    register_pattern,
)
from moncpipelib.ingest.patterns.api_crawl import ApiCrawlPattern
from moncpipelib.ingest.patterns.api_resolver import ApiResolverPattern
from moncpipelib.ingest.patterns.blob_mirror import BlobMirrorPattern
from moncpipelib.ingest.patterns.browser_export import BrowserExportPattern
from moncpipelib.ingest.patterns.http_urls import HttpUrlsPattern
from moncpipelib.ingest.prefix import render_payload_filename, render_prefix
from moncpipelib.ingest.resolver import resolve_source_for_partition
from moncpipelib.ingest.resolvers import (
    RESOLVERS,
    ReleaseResolver,
    ResolvedDownload,
    get_resolver,
    register_resolver,
)
from moncpipelib.ingest.resolvers.uts import UtsReleaseResolver
from moncpipelib.ingest.sensors import build_discovery_sensor
from moncpipelib.ingest.streaming import StreamTooLargeError, drain_to_bytes
from moncpipelib.ingest.types import (
    BlobRef,
    IngestContext,
    IngestResult,
    PartitionSpec,
    RawUrl,
)

__all__ = [
    "BUILTIN_PATTERN_NAMES",
    "CRAWL_PLANS",
    "EXPORT_PLANS",
    "INGEST_PATTERNS",
    "KNOWN_MAX_VERSION",
    "RESOLVERS",
    "ApiCrawlPattern",
    "ApiResolverPattern",
    "BlobMirrorPattern",
    "BlobRef",
    "BrowserExportPattern",
    "BrowserSession",
    "CrawlPlan",
    "CrawlRecord",
    "ExportPlan",
    "ExportedFile",
    "HttpUrlsPattern",
    "IngestContext",
    "IngestManifest",
    "IngestPattern",
    "IngestResolutionError",
    "IngestResult",
    "ManifestFieldError",
    "ManifestFileEntry",
    "PartitionSpec",
    "RawUrl",
    "ReleaseResolver",
    "ResolvedDownload",
    "StreamTooLargeError",
    "ThrottledClient",
    "UtsReleaseResolver",
    "build_discovery_sensor",
    "download_partition_parts_with_manifest",
    "drain_to_bytes",
    "get_crawl_plan",
    "get_export_plan",
    "get_pattern",
    "get_resolver",
    "materialize_with_manifest",
    "read_partition_with_manifest",
    "register_crawl_plan",
    "register_export_plan",
    "register_pattern",
    "register_resolver",
    "render_payload_filename",
    "render_prefix",
    "resolve_source_for_partition",
    "sanitize_blob_filename",
]
