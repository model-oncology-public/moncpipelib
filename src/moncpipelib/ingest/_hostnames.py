"""Hostname normalization and authoring-shape validation for ``browser_export``'s ``allowed_hosts``.

Two jobs, deliberately small:

1. :func:`normalize_hostname` -- the ONE normalization applied to BOTH
   sides of every runtime allowlist comparison in ``ingest/_browser.py``
   (the enforced-set construction and the HTTP and WebSocket route
   handlers -- its only call sites). A single implementation, because
   rounds 1 and 2 of this guard normalized in two places, differently,
   and drifted (see ``docs/migrations/20260807_463-egress-allowlist-reframe.md``).
   Contract-load validation and the materialize-time backstop do NOT
   call it -- they run job 2's shape screen on the as-authored entry
   (see :func:`malformed_allowlist_host_reason` for the exact transforms
   its sub-checks apply to their own working copies).
2. :func:`malformed_allowed_hosts_reason` / :func:`malformed_allowlist_host_reason`
   -- a well-formedness screen over what a contract author wrote: is this a
   non-empty list of bare hostnames, each shaped so it could ever match a
   request a real Chromium makes. A malformed entry (a port, a wildcard, an
   empty DNS label, a non-punycode Unicode label) would otherwise validate
   clean and then match nothing, surfacing as a navigation timeout naming the
   wrong cause.

This module passes NO judgment on what a well-shaped name names. An entry
naming loopback, an IP address, an in-cluster service, or a cloud metadata
endpoint validates clean here: no name inspection can decide where a name
resolves, and earlier revisions of this module that tried (reserved-name
sets, encoded-address detection, an internet-addressability rule) were
removed deliberately -- see the round-6 section of
``docs/migrations/20260807_463-egress-allowlist-reframe.md``. The egress
boundary for ``browser_export`` is deployment-provided network policy, not
anything in this module (``SECURITY.md``, "Browser-Driven Ingest Exports").
"""

from __future__ import annotations

_UNICODE_LABEL_SEPARATOR_TRANSLATION = str.maketrans(
    {
        "。": ".",  # IDEOGRAPHIC FULL STOP
        "．": ".",  # FULLWIDTH FULL STOP
        "｡": ".",  # HALFWIDTH IDEOGRAPHIC FULL STOP
    }
)
"""Label separators a real Chromium (per WHATWG URL / UTS #46) treats as
equivalent to U+002E FULL STOP when splitting a hostname into labels --
Python's own ``str.split(".")`` does not. Applied in two places, each
directly: inside :func:`normalize_hostname`, so the runtime comparison
sees the same label boundaries Chromium will actually use, and inside
the shape screen (:func:`_label_shape_defect` and the non-ASCII check
in :func:`malformed_allowlist_host_reason`), so a disguised separator
cannot hide a label-shape defect -- ``localhost。localdomain`` validated
clean as a single dotless label before this translation existed, while
Chromium requested the literal ASCII ``localhost.localdomain`` on the
wire."""


def normalize_hostname(host: str) -> str:
    """Return ``host`` lowercased, with Chromium's Unicode label separators mapped to ``.``, and trailing ``.`` stripped to a fixed point.

    The ONE normalization used everywhere a hostname is compared at
    runtime for the ``browser_export`` allowlist: the configured-allowlist
    set built by ``browser_session``, and both runtime comparison handlers
    (HTTP route + WebSocket route) in this package's ``_browser.py`` --
    its only call sites. There is exactly one implementation so those
    call sites cannot drift out of sync with each other the way round 1
    and round 2 did.

    A trailing ``.`` is the valid DNS root label (RFC 1034 section 3.1):
    both ``urlsplit`` and a real Chromium preserve it verbatim on the wire,
    and a name with a trailing dot resolves identically to the same name
    without it -- so an authored entry ``example.gov.`` must match a
    request for ``example.gov``, and vice versa. Stripped to a FIXED
    POINT, not merely once, so ``example.gov..`` also compares equal to
    the canonical form Chromium puts on the wire (pinned by the binding
    test's ``example.gov..`` corpus entry).

    :data:`_UNICODE_LABEL_SEPARATOR_TRANSLATION` is applied before the
    trailing-dot strip, so a hostname ending in one of those separators is
    also stripped to a fixed point exactly like a literal trailing ``.``.

    Idempotent: ``normalize_hostname(normalize_hostname(x)) ==
    normalize_hostname(x)`` for any ``x``, since the result never ends in
    ``.``, contains no character the translation table maps away, and
    lowercasing an already-lowercased string is a no-op.

    An input of ``"."`` (or ``".."``, ...) normalizes to ``""``. Such values
    never reach the runtime's allowed set: ``malformed_allowlist_host_reason``
    rejects them at contract load and at the materialize-time backstop
    (an empty-label shape defect).
    """
    normalized = host.lower().translate(_UNICODE_LABEL_SEPARATOR_TRANSLATION)
    while normalized.endswith("."):
        normalized = normalized[:-1]
    return normalized


def _label_shape_defect(value: str) -> str | None:
    """Return why ``value``'s DNS-label shape is malformed as raw-authored input, else ``None``.

    Applied BEFORE the rest of normalization (lowercasing; the trailing-dot
    strip to a fixed point) -- so a leading or internal empty label (a
    leading ``.``, or a doubled ``..`` anywhere) is caught HERE.

    :data:`_UNICODE_LABEL_SEPARATOR_TRANSLATION` IS applied here, before
    splitting into labels (#464/#468 round 4, K1) -- without it, this
    function split on a literal ``.`` in the RAW value, so a disguised
    separator was invisible to it even though :func:`normalize_hostname`
    maps it to ``.`` everywhere else in this module: ``。metadata`` (a
    disguised leading dot), ``340bopais。.hrsa.gov`` (a disguised internal
    empty label), and ``hrsa。-gov`` (a disguised leading-hyphen label) all
    validated clean here before this fix, exactly the "normalize once,
    then check everything" lesson round 3 already learned for the other
    checks in this module.

    Trailing ``.`` is stripped to a FIXED POINT, after that translation,
    before splitting into labels -- matching :func:`normalize_hostname`'s
    own trailing-dot handling -- so ``340bopais.hrsa.gov..`` (accepted
    elsewhere in this module) is not misread here as ending in an empty
    label; only a LEADING or INTERNAL empty label is rejected.

    A LEADING empty label gets its own, more actionable message
    (#464/#468 round 4, K9): ``.hrsa.gov``'s most likely real intent is
    curl/NO_PROXY-style subdomain-wildcard syntax (".example.com" meaning
    "example.com and every subdomain"), which this allowlist does not
    support -- "contains an empty DNS label" alone does not tell the
    author what to do instead, so the message says so directly: declare
    each subdomain explicitly.
    """
    trimmed = value.translate(_UNICODE_LABEL_SEPARATOR_TRANSLATION).rstrip(".")
    labels = trimmed.split(".")
    if labels and labels[0] == "":
        return (
            "starts with an empty DNS label (a leading '.') -- this "
            "allowlist has no subdomain-wildcard syntax (curl/NO_PROXY's "
            "'.example.com', meaning 'example.com and every subdomain', is "
            "not supported here); declare each subdomain explicitly"
        )
    if any(label == "" for label in labels):
        return "contains an empty DNS label (an internal '..')"
    if any(label.startswith("-") or label.endswith("-") for label in labels):
        return "contains a DNS label starting or ending with '-'"
    return None


def malformed_allowlist_host_reason(value: object) -> str | None:
    """Return why ``value`` is not shaped like a bare ``allowed_hosts`` entry, else ``None``.

    Pure shape pre-screen over ``value`` as authored: it never calls
    :func:`normalize_hostname` and never lowercases, but its sub-checks
    do transform their own working copies -- the label-shape and
    non-ASCII checks apply :data:`_UNICODE_LABEL_SEPARATOR_TRANSLATION`
    so a disguised separator cannot hide a defect, and the label-shape
    check strips trailing dots to a fixed point before splitting (its
    own ``rstrip(".")``), which is what lets a root-label entry like
    ``340bopais.hrsa.gov..`` validate instead of being misread as ending
    in empty labels. It makes no judgment about whether a well-shaped
    string is safe: this module makes no such judgment (see the module
    docstring). This is the loader's ONLY per-entry
    ``allowed_hosts`` check (``contracts/loader.py`` runs this on every
    entry); extracted here, per
    ``docs/migrations/20260807_463-egress-allowlist-reframe.md`` Step 1b,
    so the materialize-time backstop in
    ``ingest/patterns/browser_export.py`` runs the SAME check the loader
    does -- the inconsistency that let a hand-built contract's
    ``"evil.com:8080"`` or ``"*.hrsa.gov"`` entry reach a live browser
    session unrejected.

    Rejects: not a string; empty; leading or trailing whitespace; an
    internal ``/``, ``:``, ``*``, or ``%`` (scheme, port, path, wildcard
    syntax, or a percent-encoding artifact); any whitespace character
    anywhere in the string; a DNS-label shape defect per
    :func:`_label_shape_defect` (#464/#468 round 4, J8) -- an empty label
    (a leading ``.`` or an internal ``..``) or a label starting/ending
    with ``-``; and a non-ASCII character surviving the Unicode
    label-separator translation. Each of ``.hrsa.gov``,
    ``340bopais..hrsa.gov``, ``-.hrsa.gov``, and
    ``340bopais.hrsa.gov%20`` validated clean at BOTH ``allowed_hosts``
    checks before this fix and became a permanently dead allowlist entry
    that blocks every request with no diagnostic -- under the
    mistake-catcher reclassification, a leading or doubled dot is the
    single most representative mistake this guard exists to catch.

    The returned message names ``%`` explicitly (#464/#468 round 4, K9):
    for a guard reframed as a diagnostic, the message IS the product --
    an author who writes ``hrsa.gov%20`` and gets back a message that
    omits ``%`` from the list of rejected characters has to guess which
    of the OTHER three candidates (scheme, port, path) is actually theirs.

    The non-ASCII check (relocated here in round 6 from the removed
    ``disallowed_allowlist_host_reason``) runs on the TRANSLATED value:
    the three Unicode label separators are permitted (they normalize to
    ``.``); any other non-ASCII character is rejected. Chromium requests
    the IDNA/punycode form of a Unicode hostname (``münchen.de`` is
    requested as ``xn--mnchen-3ya.de``); this module does not perform
    that encoding, so an entry must already be supplied in punycode --
    without this check, a non-punycode entry would validate clean and
    then never match any request a real Chromium makes, becoming a dead
    entry with no diagnostic, exactly the failure mode this function
    exists to catch. This check does NOT claim to catch anything beyond
    that dead-entry case.
    """
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(c in value for c in "/:*%")
        or any(c.isspace() for c in value)
    ):
        return (
            "must be a bare hostname (no scheme, port, path, wildcard, "
            "percent-encoding, or whitespace)"
        )
    label_defect = _label_shape_defect(value)
    if label_defect is not None:
        return label_defect
    translated = value.translate(_UNICODE_LABEL_SEPARATOR_TRANSLATION)
    if not translated.isascii():
        return (
            "contains non-ASCII characters (allowed_hosts must be the "
            "punycode form Chromium sends on the wire, e.g. 'xn--...', not "
            "the Unicode label)"
        )
    return None


def malformed_allowed_hosts_reason(value: object) -> str | None:
    """Return why ``value`` is not shaped like a valid ``allowed_hosts`` list, else ``None``.

    The list-level counterpart to :func:`malformed_allowlist_host_reason`
    (which validates one ENTRY): ``value`` itself must be a list, and
    non-empty. #464/#468 round 4 (J6): the loader already rejected a
    non-list or empty ``allowed_hosts`` at contract load -- but the
    materialize-time backstop in ``ingest/patterns/browser_export.py``
    hand-rolled its OWN, weaker type guard (``isinstance(value, list)``
    with no emptiness check), so a hand-built contract's
    ``allowed_hosts: []`` passed the backstop, ran its per-entry loop
    zero times, and reached :func:`~moncpipelib.ingest._browser.browser_session`
    with a runtime allowlist that matches nothing. That fails CLOSED --
    ``_install_host_allowlist``'s enforced set is empty, so every request
    is aborted -- but surfaces as a 60-second navigation timeout naming
    the wrong cause rather than this load/materialize-time diagnostic.
    Moved here, per the same single-source discipline as
    :func:`malformed_allowlist_host_reason`, so both the loader and the
    backstop enforce it identically instead of drifting the way the
    per-entry checks did across rounds 1 and 2.
    """
    if not isinstance(value, list) or not value:
        return "must be a non-empty list of hostnames"
    return None
