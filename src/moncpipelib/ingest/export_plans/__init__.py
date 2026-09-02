"""Export plan Protocol and registry for the ``browser_export`` ingest pattern (#463).

Export plans are the bridge between a ``browser_export`` ingest contract and
a driving source with **no addressable URL** -- the file only materializes as
the side effect of a browser session clicking through a UI (e.g. a
DevExpress-generated report page that regenerates its export server-side and
streams the result to whichever tab happens to click "Export"). Each plan
knows the specific click-path for its source and yields the files that
navigation produces.

Design (per the #463 design doc, decisions D1/D5/D6): the plan is
**imperative**, not declarative -- the same shape as :mod:`crawl_plans`.  It
receives a sanctioned :class:`~moncpipelib.ingest._browser.BrowserSession`
and yields :class:`ExportedFile`\\ s.  The plan drives ``navigate`` /
``expect_control`` / ``click`` / ``select_option`` / ``click_and_await_download``;
the pattern -- not the plan -- owns host-allowlist enforcement, the per-run
tempdir, hashing, blob upload, and the manifest.

Plan-author contract:

- **Session-only browsing**: every page interaction goes through the
  provided :class:`~moncpipelib.ingest._browser.BrowserSession`.
  Constructing a browser, a raw ``playwright`` object, or an ``httpx.Client``
  inside plan code bypasses the host allowlist, the User-Agent, and the
  tempdir cleanup guarantees -- the same class of audit violation as
  bypassing ``ThrottledClient`` (see ``SECURITY.md``).
- **No import-time network, no import-time playwright** -- same rule as
  patterns, resolvers, and crawl plans; ``Definitions(...)`` construction
  must stay both network-free and playwright-free (the latter is the D1
  invariant this whole pattern exists to protect).
- **ADR-2 validate_config**: runs at contract-load time, including in CI.
  It MUST be deterministic, MUST NOT make network calls, and MUST NOT
  perform filesystem I/O -- a chromium/dependency probe here is forbidden
  (see :func:`~moncpipelib.ingest._browser.ensure_chromium_available`'s
  docstring for why: a probe at load time would make every CI import of a
  contract module depend on a browser binary being installed).  It MUST
  reject unknown keys, per the same ADR-2 contract as
  :meth:`~moncpipelib.ingest.crawl_plans.CrawlPlan.validate_config`.
- **Generation-date cross-check (D4a.3)**: where the driving export exposes
  its own generation/as-of date, the plan receives ``partition_key`` and is
  **obliged** to compare it against that date and raise
  :class:`~moncpipelib.ingest.exceptions.IngestResolutionError` on a
  mismatch.  This is stated as a plan obligation, not a library guarantee --
  whether a given driving source exposes such a date at all is unverified
  and source-specific.
- **``ExportedFile.path`` is session-owned**: valid only for the lifetime of
  the enclosing ``with browser_session(...)`` block.  Plans must not copy,
  move, retain, or reopen the path after yielding it -- the session unlinks
  the per-run tempdir (including every downloaded file under it) when the
  context manager exits, on both success and failure.
- **``multi_file`` must be declared ``True``** by any plan whose ``export``
  can yield more than one file.  Under-declaring produces silent overwrites:
  the loader rejects ``ingest.payload_filename_template`` for a
  ``multi_file`` plan (one template renders the same filename for every
  file), so an un-declared multi-file plan combined with a template
  silently clobbers all but the last file (see :class:`ExportPlan.multi_file`).

Like crawl plans and resolvers, export plans are stateless singletons: one
instance per registered :attr:`ExportPlan.name`, parameterless ``__init__``,
all per-call state through method arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moncpipelib.ingest._browser import BrowserSession
    from moncpipelib.ingest.types import IngestContext


@dataclass(frozen=True)
class ExportedFile:
    """One file a :meth:`ExportPlan.export` call has downloaded via the browser.

    Attributes:
        path: Absolute path to the downloaded file inside the
            :class:`~moncpipelib.ingest._browser.BrowserSession`'s private
            per-run tempdir.  **The session owns this file**, not the plan
            and not the pattern: it is unlinked when the session's context
            manager exits, on both success and failure.  Valid only inside
            the enclosing ``with browser_session(...)`` block.  The filename
            component is session-generated (a zero-padded counter), never
            derived from ``suggested_filename`` -- see
            :meth:`~moncpipelib.ingest._browser.BrowserSession.click_and_await_download`
            for the local-disk-traversal rationale.
        suggested_filename: **Untrusted**, verbatim from playwright's
            ``Download.suggested_filename`` (set by the remote site's
            ``Content-Disposition`` header or its own JavaScript).  Never
            usable as a path component without going through
            :func:`~moncpipelib.ingest.filenames.sanitize_blob_filename`
            first.  May be an empty string.

    Deliberately carries nothing else:

    - No ``size_bytes`` / ``sha256``: the pattern computes both in one
        chunked pass over ``path``.  Carrying them here would create a
        second source of truth a plan could populate incorrectly.
    - No ``download_url``: playwright's ``Download.url`` is a ``blob:`` URL
        for the driving source -- page-internal, useless for audit, and a
        standing temptation to log a URL into a sink ``ingest/_http.py``
        declares off-limits.
    - No plan-authored filename override: the naming chain is fixed at two
        levels (contract ``payload_filename_template``, else
        ``sanitize_blob_filename(suggested_filename)``).  A frozen dataclass
        with no positional-arg consumers outside the library, so a future
        field with a default is a non-breaking addition if one is ever
        needed.
    """

    path: Path
    suggested_filename: str


@runtime_checkable
class ExportPlan(Protocol):
    """Protocol every browser export plan implements.

    Stateless; one instance per registered name.  See the module docstring
    for the plan-author contract (session-only browsing, no import-time
    playwright, ADR-2 validation, the generation-date obligation, and the
    ``multi_file`` declaration rule).

    ``multi_file`` (a ``bool``, defaulting to ``False``; see the module
    docstring) is deliberately **not** declared as a member of this
    Protocol, even though the design calls it out as a ``ClassVar``.
    Verified empirically: ``typing.Protocol``'s ``__protocol_attrs__``
    collection (and therefore ``@runtime_checkable``'s ``isinstance``
    check) includes *any* name that appears in the class body -- annotated
    or not, defaulted or not -- not only names lacking a default as one
    might assume.  Declaring ``multi_file`` here would make it a *required*
    attribute for ``isinstance(plan, ExportPlan)``, which is exactly
    backwards: a duck-typed plan that never fans out (the common case)
    would then fail the protocol check for omitting an attribute whose
    whole point is an implicit default.  It is therefore purely a
    documented convention, read via ``getattr(plan, "multi_file", False)``
    -- never plain attribute access -- by the contract loader (Step 3).
    """

    name: ClassVar[str]

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Return a list of validation error strings (empty if valid).

        Called by the ``browser_export`` block validator at contract-load
        time with the contents of ``ingest.browser_export.export_config``.
        Plans should validate required keys, value types, plan-specific
        format rules, AND reject unknown keys (per ADR-2).

        Network calls are forbidden; filesystem I/O is forbidden -- and so
        is a chromium/playwright dependency probe (see
        :func:`~moncpipelib.ingest._browser.ensure_chromium_available`).
        The function must be deterministic and fast (target < 1ms).
        """
        ...

    def export(
        self,
        session: BrowserSession,
        partition_key: str,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> Iterator[ExportedFile]:
        """Drive ``session`` through the click-path and yield downloaded files.

        Runs at materialize time only (never at discovery or import time).
        ``session`` is the sanctioned browsing surface -- the ONLY way to
        reach the driving source.  Constructing a separate browser or HTTP
        client bypasses the host allowlist and tempdir cleanup guarantees.

        Per D4a.3, where the driving export exposes its own generation
        date, the plan is obliged to compare it against ``partition_key``
        and raise :class:`~moncpipelib.ingest.exceptions.IngestResolutionError`
        on a mismatch -- a stale server-side cache producing yesterday's
        file under today's partition key is a silent correctness gap
        otherwise.

        Yield files as they are produced.  Any raise aborts the partition:
        the dispatcher will not write a manifest and the next run re-drives
        the browser from scratch.
        """
        ...


EXPORT_PLANS: dict[str, ExportPlan] = {}


def register_export_plan(plan: ExportPlan) -> None:
    """Register ``plan`` under its :attr:`ExportPlan.name`.

    Subsequent calls with the same name overwrite the previous entry --
    useful for testing with a stub, never intended for production.

    Per-source plans live with their consumers (data-platform) and are
    registered from consumer code, following the resolver-registry and
    crawl-plan-registry precedent; moncpipelib ships no builtin plans today.
    """
    EXPORT_PLANS[plan.name] = plan


def get_export_plan(name: str) -> ExportPlan:
    """Look up a registered export plan by name.

    Raises:
        KeyError: If no plan with that name is registered.  The message
            lists the known plans, so a YAML typo surfaces at
            contract-load time with a useful suggestion.
    """
    try:
        return EXPORT_PLANS[name]
    except KeyError as e:
        known = sorted(EXPORT_PLANS)
        raise KeyError(f"Unknown export plan {name!r}. Known export plans: {known}") from e


__all__ = [
    "EXPORT_PLANS",
    "ExportedFile",
    "ExportPlan",
    "get_export_plan",
    "register_export_plan",
]
