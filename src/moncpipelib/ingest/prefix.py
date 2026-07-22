"""Template rendering for the ingest boundary.

Two surfaces share one bounded-placeholder template engine:

- :func:`render_prefix` -- renders ``IngestContract.prefix_template`` to
  produce the per-partition blob prefix (e.g.
  ``"cms_asp/{partition_key}"`` -> ``"cms_asp/2025_q1"``).
- :func:`render_payload_filename` -- renders
  ``IngestContract.payload_filename_template`` (#270) to produce the
  blob filename for non-archive (``extract: []``) payloads (e.g.
  ``"{source_name}_{partition_key}.csv"`` ->
  ``"seer_cpc_smvl_V2024B.csv"``).

Both deliberately reuse the same :data:`_ALLOWED_PLACEHOLDERS` set: a
landing path's prefix and filename are both audit-visible artifacts, and
constraining them to the same bounded vocabulary keeps blob paths in the
UI mapping 1:1 to what is in the contract. No ``strftime`` codes, no
custom date parsing -- a partition key is used verbatim.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moncpipelib.contracts.models import IngestContract


# Permitted placeholder set. Extending this requires a conscious choice;
# every new placeholder becomes part of the landing path contract.
_ALLOWED_PLACEHOLDERS: frozenset[str] = frozenset({"partition_key", "source_name"})

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _render_template(
    template: str,
    partition_key: str,
    contract: IngestContract,
    *,
    template_field_name: str,
) -> str:
    """Render an ingest template against the bounded placeholder set.

    Shared by :func:`render_prefix` and :func:`render_payload_filename`.

    Args:
        template: The raw template, e.g. ``"cms_asp/{partition_key}"``.
        partition_key: Used verbatim; no date parsing.
        contract: Provides bounded fields like ``source_name``.
        template_field_name: For error messages -- ``"prefix template"``
            or ``"payload filename template"`` -- so the rejection cites
            the offending field.

    Raises:
        ValueError: If the template references a placeholder outside
            :data:`_ALLOWED_PLACEHOLDERS`.
    """
    bindings = {
        "partition_key": partition_key,
        "source_name": contract.source_name,
    }

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in _ALLOWED_PLACEHOLDERS:
            raise ValueError(
                f"Unknown placeholder '{{{name}}}' in {template_field_name} "
                f"{template!r}. Allowed: {sorted(_ALLOWED_PLACEHOLDERS)}"
            )
        return bindings[name]

    return _PLACEHOLDER_RE.sub(_sub, template)


def render_prefix(
    template: str,
    partition_key: str,
    contract: IngestContract,
) -> str:
    """Render an ingest prefix template for a given partition.

    Args:
        template: The raw prefix template, e.g. ``"cms_asp/{partition_key}"``.
        partition_key: The partition key for this materialization. Used
            verbatim; no date parsing or formatting is performed.
        contract: The ingest contract, used to pull bounded fields such
            as ``source_name``.

    Returns:
        The rendered prefix string.

    Raises:
        ValueError: If the template references a placeholder outside
            :data:`_ALLOWED_PLACEHOLDERS`.
    """
    return _render_template(
        template, partition_key, contract, template_field_name="prefix template"
    )


def render_payload_filename(
    template: str,
    partition_key: str,
    contract: IngestContract,
) -> str:
    """Render an ingest payload-filename template for a given partition.

    Per #270, ``IngestContract.payload_filename_template`` is the
    highest-precedence input to the non-archive payload filename
    derivation chain (template -> resolver hint -> Content-Disposition
    -> URL basename -> raise). Authored templates are NOT passed through
    :func:`~moncpipelib.ingest.filenames.sanitize_blob_filename`: a
    malformed authored name should fail loudly at upload time rather
    than be silently rewritten.

    Args:
        template: The raw filename template, e.g.
            ``"{source_name}_{partition_key}.csv"``.
        partition_key: The partition key for this materialization. Used
            verbatim.
        contract: The ingest contract, used to pull ``source_name``.

    Returns:
        The rendered filename string.

    Raises:
        ValueError: If the template references a placeholder outside
            :data:`_ALLOWED_PLACEHOLDERS`.
    """
    return _render_template(
        template,
        partition_key,
        contract,
        template_field_name="payload filename template",
    )
