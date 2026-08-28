"""Stable content hashes for ``DataContract`` instances.

Migration 019 (#308) Phase 3 adds two SHA256-based fingerprint columns
to ``lineage.pipeline_registry``:

- ``contract_hash`` -- digest over the contract's full semantic content
  (schema, SLA, tags, ownership, expectations, etc.). Used for "did the
  contract content change between this run and the last" change-detection
  queries against the registry.
- ``schema_fingerprint`` -- digest over the column schema's identity
  fields only. Lets consumers answer "did the table shape change"
  cheaply without re-reading the full contract.

Both hashes are deterministic: keys are sorted, lists of dicts are
canonicalised, enum / date / UUID values are coerced to their string
form. Re-hashing the same parsed contract always produces the same
digest; the hashes are insensitive to YAML key-order or formatting.

The hashes deliberately exclude the ``contract_hash`` and
``schema_fingerprint`` fields themselves so re-population by
``load_contract`` does not feed back into the computation.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from moncpipelib.contracts.models import DataContract


_HASH_EXCLUDED_FIELDS = frozenset({"contract_hash", "schema_fingerprint"})


def _json_default(obj: Any) -> Any:
    """Last-resort serializer for ``json.dumps``.

    Handles types that ``dataclasses.asdict`` leaves alone but ``json``
    cannot natively serialise (datetime / UUID / Decimal / set / bytes).
    Enum values are normalised to their ``.value`` form so two contracts
    that disagree only on whether a field is the enum member or its
    string value still hash identically.
    """
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(
            (_json_default(x) if not isinstance(x, (str, int, float, bool)) else x) for x in obj
        )
    if isinstance(obj, bytes):
        return obj.hex()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _to_canonical_payload(contract: DataContract) -> dict[str, Any]:
    """Return a deterministic dict shape suitable for hashing.

    Drops the two fingerprint fields themselves so the hash is stable
    across re-population by ``load_contract``.
    """
    payload = dataclasses.asdict(contract)
    for field_name in _HASH_EXCLUDED_FIELDS:
        payload.pop(field_name, None)
    # phi participates only when it diverges from pii: phi defaults to the
    # pii value (#391), and hashing the mirrored default would flip every
    # pre-phi contract_hash on upgrade.
    schema = payload.get("schema")
    if isinstance(schema, dict):
        for col in schema.get("columns", []):
            if isinstance(col, dict) and col.get("phi") == col.get("pii"):
                col.pop("phi", None)
    return payload


def compute_contract_hash(contract: DataContract) -> str:
    """Compute a stable SHA256 digest over the contract's semantic content.

    Sensitive to: schema changes, SLA changes, ownership changes, tag
    changes, expectation changes, parameter changes, data-source changes.

    Insensitive to: YAML key order, dict iteration order, formatting,
    and the ``contract_hash`` / ``schema_fingerprint`` fields themselves.

    Args:
        contract: Parsed contract.

    Returns:
        Hex-encoded SHA256 digest (64 characters).
    """
    payload = _to_canonical_payload(contract)
    serialised = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def compute_schema_fingerprint(contract: DataContract) -> str:
    """Compute a stable SHA256 digest over the column schema's identity fields.

    Sensitive to: column add / remove / rename, type changes, nullable
    flips, PII flag flips, primary-key flips, managed flag flips, and PHI
    flag flips (only when phi diverges from pii -- phi defaults to the pii
    value, so pre-phi fingerprints stay stable).

    Insensitive to: column description, column tests, column order
    (columns are sorted by name before hashing), SLA, tags, ownership.

    Args:
        contract: Parsed contract.

    Returns:
        Hex-encoded SHA256 digest (64 characters).
    """
    cols: list[dict[str, Any]] = []
    for c in contract.schema.columns:
        entry: dict[str, Any] = {
            "name": c.name,
            "type": c.type.value if isinstance(c.type, enum.Enum) else str(c.type),
            "nullable": c.nullable,
            "pii": c.pii,
            "primary_key": c.primary_key,
            "managed": c.managed,
        }
        if c.phi != c.pii:
            entry["phi"] = c.phi
        cols.append(entry)
    cols.sort(key=lambda col: str(col["name"]))
    serialised = json.dumps(cols, sort_keys=True)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def derive_data_classification(contract: DataContract) -> str:
    """Roll up column-level PII flags to a single classification string.

    Migration 019 (#308) Phase 3 design decision: ``data_classification``
    is derived at upsert time from column-level PII flags rather than
    added as a top-level contract field. Avoids a contract-spec change
    for an aggregation that is already deterministic from existing
    fields.

    Returns:
        ``"PHI"`` if any non-managed column has ``phi=True``, ``"none"``
        otherwise. ``phi`` defaults to the column's ``pii`` value (#391),
        so contracts that never annotate ``phi`` classify exactly as
        before; explicitly clearing every column (``phi: false``) is the
        only way a PII-bearing contract can classify as ``"none"``.
    """
    for col in contract.schema.columns:
        if col.managed:
            continue
        if col.phi:
            return "PHI"
    return "none"
