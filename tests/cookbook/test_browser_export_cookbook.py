"""Cookbook test for the ``browser_export`` ingest pattern (#463).

Demonstrates landing a source with **no addressable URL** -- the file only
materializes as the side effect of a headless browser clicking through a
UI (e.g. a report page that regenerates its export server-side and streams
the result to whichever tab happens to click "Export"):

- Register a custom :class:`ExportPlan` that drives a
  :class:`~moncpipelib.ingest.BrowserSession`-shaped object through
  ``navigate`` -> ``select_option`` -> ``click_and_await_download``,
  locating every control by ARIA role + accessible name (D6) -- the names
  live in ``export_config``, not in library code.
- Construct a ``browser_export`` :class:`IngestContract` matching the
  design doc's draft contract shape.
- Monkeypatch the pattern module's ``browser_session`` factory to a stub
  that yields a fake session and writes a small JSON payload, so the
  example runs in CI with **no browser binary and no network**. This is a
  test seam only -- production code never injects a session; the pattern
  always opens a real, sanctioned one.
- Call :func:`materialize_with_manifest` and observe the landed blob path
  and the manifest's ``fields["snapshot_date"]`` and ``resolver["name"]``.

Code between ``# --- cookbook:start ---`` / ``# --- cookbook:end ---``
is extracted into ``docs/cookbook.md`` by the cookbook plugin.
"""

from __future__ import annotations

from typing import Any

import pytest

from moncpipelib.ingest.export_plans import EXPORT_PLANS


@pytest.fixture
def cookbook_export_plan_registrations() -> Any:
    """Restore the export-plan registry after the example."""
    plans_before = dict(EXPORT_PLANS)
    yield
    EXPORT_PLANS.clear()
    EXPORT_PLANS.update(plans_before)


@pytest.mark.cookbook(
    title="Land a browser-only export with browser_export + an ExportPlan",
    description=(
        "Some public sources have no addressable download URL -- the file "
        "only materializes as the side effect of a headless browser "
        "clicking through a UI (first consumer: HRSA's 340B OPAIS Covered "
        "Entity report). ``browser_export`` drives a registered "
        ":class:`ExportPlan` through a sanctioned "
        ":class:`~moncpipelib.ingest.BrowserSession` -- the ONLY browsing "
        "surface a plan may use -- and lands whatever files it downloads. "
        "playwright is an optional ``browser`` dependency extra, never a "
        "core one (D1): only the environment that actually drives a "
        "browser needs to install it. Plans live in consumer "
        "code and are registered like resolvers and crawl plans; "
        "moncpipelib ships none itself (D3). ``allowed_hosts`` is "
        "required and governs navigation and subresource loads, but "
        "**not** a ``blob:`` download, which issues no network request -- "
        "so payload trust for a blob-served export rests on the page "
        "origin plus byte-level content validation, not the allowlist "
        "(D5). Contracts are restricted to ``sensitivity: public`` in "
        "code, because a headless browser executing arbitrary "
        "third-party JavaScript must not be pointed at a confidential or "
        "PHI source without a separate review (D10). This example "
        "registers a stub plan and monkeypatches the session factory so "
        "it runs in CI with no browser binary and no network; real "
        "pipelines register a production plan against the live page."
    ),
    category="ingest",
)
def test_cookbook_browser_export_roundtrip(
    cookbook_export_plan_registrations: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    del cookbook_export_plan_registrations
    # --- cookbook:start ---
    import json
    import logging
    from collections.abc import Iterator
    from contextlib import contextmanager
    from typing import ClassVar

    from freezegun import freeze_time

    from moncpipelib.contracts.models import IngestContract
    from moncpipelib.ingest import (
        BrowserExportPattern,
        ExportedFile,
        IngestContext,
        IngestManifest,
        materialize_with_manifest,
        register_export_plan,
    )
    from moncpipelib.ingest.patterns import browser_export

    # --- 1. Register an export plan ---
    # The plan is the per-source extension point (registered from consumer
    # code, like resolvers and crawl plans). It receives a BrowserSession --
    # the ONLY sanctioned way to reach the driving page -- and drives it by
    # ARIA role + accessible name (D6), never by CSS selector or DOM
    # position, so the plan survives a markup redesign that leaves the
    # visible control names untouched.
    class _HrsaCoveredEntityPlan:
        name: ClassVar[str] = "_cookbook_hrsa_covered_entity"

        def validate_config(self, config: dict[str, Any]) -> list[str]:
            unknown = set(config) - {"report_name"}
            return [f"unknown field {k!r}" for k in sorted(unknown)]

        def export(
            self,
            session: Any,
            partition_key: str,
            config: dict[str, Any],
            ctx: Any,
        ) -> Iterator[ExportedFile]:
            del partition_key, ctx  # this driving source exposes no as-of date to cross-check
            session.navigate("https://340bopais.hrsa.gov/reports")
            session.select_option(role="combobox", name="Report", value=config["report_name"])
            yield session.click_and_await_download(role="button", name="Export")

    register_export_plan(_HrsaCoveredEntityPlan())

    # --- 2. Declare the browser_export ingest contract ---
    # In production this is a *.ingest.yaml loaded via load_ingest_contract;
    # `sensitivity: public` and `compliance_review` are both required at
    # load time for this pattern (D10), regardless of the source's actual
    # classification, because a headless browser in the ingest runtime is a
    # high-risk surface on its own. `fetch.user_agent` is likewise
    # load-required for this pattern: a descriptive UA is a good-citizen
    # control when the browser is driving a .gov host, and making it
    # required forces a contract author to consciously supply one rather
    # than silently shipping chromium's default `HeadlessChrome/...`.
    ingest = IngestContract(
        source_id="66666666-6666-6666-6666-666666666666",
        source_name="cookbook-hrsa-covered-entity",
        sensitivity="public",
        pattern="browser_export",
        prefix_template="hrsa/340b-opais/{partition_key}",
        extract=(),
        strip_extensions=(),
        pattern_config={
            "export_plan": "_cookbook_hrsa_covered_entity",
            "export_config": {"report_name": "Covered Entity Daily"},
            "allowed_hosts": ["340bopais.hrsa.gov"],
            "partition": {
                "mode": "dynamic",
                "cadence": "daily",
                "anchor_tz": "America/New_York",
            },
            "fetch": {"user_agent": "example.com/340b-ingest (contact: data-platform)"},
        },
        data_owner="data-platform",
        compliance_review="SECURITY.md#browser-driven-ingest-exports",
    )

    # --- 3. Test seam: stub the browser session (no real browser in CI) ---
    # Production code NEVER does this -- `browser_export.browser_session`
    # is the pattern's only caller, and it always opens a real, sanctioned
    # session with a navigation/subresource host allowlist (a
    # contract-authoring mistake-catcher, not a security boundary -- see
    # SECURITY.md), a per-run tempdir, and no page/browser handle
    # exposed. This monkeypatch exists solely so the example runs without
    # a chromium binary or network access.
    class _StubBrowserSession:
        def __init__(self, download_dir: Any) -> None:
            self._download_dir = download_dir

        def navigate(self, url: str, **kwargs: Any) -> None:
            del url, kwargs

        def select_option(self, **kwargs: Any) -> None:
            del kwargs

        def click_and_await_download(self, **kwargs: Any) -> ExportedFile:
            del kwargs
            path = self._download_dir / "0000.download"
            path.write_bytes(json.dumps({"covered_entity_count": 42}).encode())
            return ExportedFile(path=path, suggested_filename="CoveredEntity.json")

        def require_contains(self, path: Any) -> None:
            del path

    @contextmanager
    def _fake_browser_session(**kwargs: Any) -> Iterator[_StubBrowserSession]:
        del kwargs
        yield _StubBrowserSession(tmp_path)

    monkeypatch.setattr(browser_export, "browser_session", _fake_browser_session)

    # --- 4. In-memory blob stand-in ---
    class _InMemoryBlob:
        def __init__(self) -> None:
            self.store: dict[str, tuple[bytes, str]] = {}

        def read_sha256_metadata(self, sensitivity: str, path: str) -> str | None:
            del sensitivity
            entry = self.store.get(path)
            return entry[1] if entry else None

        def upload(self, sensitivity: str, path: str, data: Any, sha256: str) -> None:
            del sensitivity
            body = data if isinstance(data, bytes) else data.read()
            self.store[path] = (body, sha256)

    blob = _InMemoryBlob()
    ctx = IngestContext(log=logging.getLogger("browser-export-cookbook"))

    # --- 5. Discover the partition, then materialize against the stub ---
    pattern = BrowserExportPattern()
    with freeze_time("2026-08-06 12:00:00"):  # 08:00 EDT, no cross-midnight edge
        [partition_spec] = pattern.discover_partitions(ingest, ctx)
        assert partition_spec.key == "2026-08-06"

        results = materialize_with_manifest(pattern, ingest, partition_spec, blob, ctx)  # type: ignore[arg-type]

    [landed] = results
    assert landed.action == "uploaded"
    assert landed.path == "hrsa/340b-opais/2026-08-06/CoveredEntity.json"

    payload = json.loads(blob.store[landed.path][0])
    assert payload == {"covered_entity_count": 42}

    # The manifest's audit block names the export plan, and fields carry
    # the cadence-derived snapshot date (FromIngestTemplate consumers read
    # this via effective_from_field).
    from io import BytesIO

    manifest = IngestManifest.read_from(
        BytesIO(blob.store["hrsa/340b-opais/2026-08-06/_manifest.json"][0])
    )
    assert manifest.resolver["name"] == "_cookbook_hrsa_covered_entity"
    assert manifest.fields["snapshot_date"] == "2026-08-06"
    # --- cookbook:end ---
