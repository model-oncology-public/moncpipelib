"""Composable tag builder for Dagster job and op tags.

Provides ``ContractTags`` (immutable, derived from data contracts) and
``RunTags`` (mutable builder) for constructing ``dict[str, str]`` tag dicts
compatible with Dagster's ``define_asset_job(tags=...)`` and
``@asset(op_tags=...)``.

HIPAA note: Auto-derived tags intentionally exclude PII column names, contact
emails, and other potentially PHI-adjacent metadata. Only structural metadata
(layer, team name, pipeline_id) is surfaced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moncpipelib.contracts.models import DataContract

logger = logging.getLogger(__name__)

TAG_NAMESPACE = "moncpipelib"


@dataclass(frozen=True, slots=True)
class ContractTags:
    """Immutable tag snapshot derived from one or more data contracts.

    Auto-derives safe metadata tags from contract fields. Does NOT include
    PII column names or contact information (HIPAA compliance).
    """

    _tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        """Return a copy of the tags dict."""
        return dict(self._tags)

    @classmethod
    def from_contract(cls, contract: DataContract) -> ContractTags:
        """Build tags from a single data contract.

        Auto-derives:
            - ``moncpipelib/layer``: data layer (bronze/silver/gold)
            - ``moncpipelib/owner``: team name (not contact info)
            - ``moncpipelib/pipeline_id``: stable pipeline UUID
            - ``moncpipelib/has_sla``: whether an SLA is defined
            - ``moncpipelib/has_pii``: whether PII columns exist (boolean only)
            - ``moncpipelib/source_system``: lineage source system (if configured)

        User-defined tags from the contract YAML ``tags`` section are merged in.
        A warning is logged if user tags use the reserved ``moncpipelib/`` namespace.
        """
        tags: dict[str, str] = {}
        tags[f"{TAG_NAMESPACE}/layer"] = contract.layer
        if contract.owner is not None:
            tags[f"{TAG_NAMESPACE}/owner"] = contract.owner.team
        tags[f"{TAG_NAMESPACE}/pipeline_id"] = contract.pipeline_id
        tags[f"{TAG_NAMESPACE}/has_sla"] = str(contract.sla is not None).lower()
        tags[f"{TAG_NAMESPACE}/has_pii"] = str(bool(contract.get_pii_columns())).lower()
        if contract.lineage and contract.lineage.source_system:
            tags[f"{TAG_NAMESPACE}/source_system"] = contract.lineage.source_system

        # Merge user-defined tags from contract YAML
        for k, v in contract.tags.items():
            if k.startswith(f"{TAG_NAMESPACE}/"):
                logger.warning(
                    "Contract '%s' defines tag '%s' in reserved namespace '%s/'. "
                    "It will be included but may conflict with auto-derived tags.",
                    contract.asset,
                    k,
                    TAG_NAMESPACE,
                )
            tags[k] = v

        return cls(_tags=tags)

    @classmethod
    def from_assets(
        cls,
        contracts_dir: str | Path,
        assets: list[str],
        search_paths: list[Path | str] | None = None,
    ) -> ContractTags:
        """Aggregate tags across multiple asset contracts.

        Scalar fields are comma-joined (sorted unique values). Boolean fields
        use logical OR. User-defined tags are merged; conflicting values raise
        ``ValueError``.
        """
        from moncpipelib.contracts.loader import load_contract_for_asset

        resolved_paths: list[Path | str] = search_paths or [Path(contracts_dir)]
        contracts: list[DataContract] = []
        for asset_name in assets:
            contract = load_contract_for_asset(asset_name, search_paths=resolved_paths)
            if contract is not None:
                contracts.append(contract)

        if not contracts:
            return cls(_tags={})

        # Aggregate scalar fields
        layers = sorted({c.layer for c in contracts})
        owners = sorted({c.owner.team for c in contracts if c.owner})
        pipeline_ids = sorted({c.pipeline_id for c in contracts})
        has_sla = any(c.sla is not None for c in contracts)
        has_pii = any(bool(c.get_pii_columns()) for c in contracts)
        source_systems = sorted(
            {c.lineage.source_system for c in contracts if c.lineage and c.lineage.source_system}
        )

        tags: dict[str, str] = {
            f"{TAG_NAMESPACE}/layer": ",".join(layers),
            f"{TAG_NAMESPACE}/pipeline_id": ",".join(pipeline_ids),
            f"{TAG_NAMESPACE}/has_sla": str(has_sla).lower(),
            f"{TAG_NAMESPACE}/has_pii": str(has_pii).lower(),
        }
        if owners:
            tags[f"{TAG_NAMESPACE}/owner"] = ",".join(owners)
        if source_systems:
            tags[f"{TAG_NAMESPACE}/source_system"] = ",".join(source_systems)

        # Merge user-defined tags; conflict = ValueError
        user_tags: dict[str, str] = {}
        for c in contracts:
            for k, v in c.tags.items():
                if k in user_tags and user_tags[k] != v:
                    raise ValueError(
                        f"Conflicting user tag '{k}': contract '{c.asset}' defines "
                        f"'{v}' but another contract defines '{user_tags[k]}'"
                    )
                user_tags[k] = v
        tags.update(user_tags)

        return cls(_tags=tags)


class RunTags:
    """Mutable tag builder that composes contract tags with runtime tags.

    Usage::

        run_tags = RunTags()
        run_tags.add_contract_tags(ContractTags.from_contract(contract))
        run_tags.add_tags({"image/version": "1.2.3"})
        run_tags.add_tag("env", "production")
        job = define_asset_job(name="my_job", tags=run_tags.to_dict())
    """

    def __init__(self) -> None:
        self._tags: dict[str, str] = {}

    def add_contract_tags(self, contract_tags: ContractTags) -> RunTags:
        """Merge in contract-derived tags. Returns self for chaining."""
        self._tags.update(contract_tags.to_dict())
        return self

    def add_tags(self, tags: dict[str, str]) -> RunTags:
        """Merge in arbitrary runtime tags. Returns self for chaining."""
        for k, v in tags.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError(
                    f"Tags must be dict[str, str], got key={type(k).__name__}, "
                    f"value={type(v).__name__}"
                )
        self._tags.update(tags)
        return self

    def add_tag(self, key: str, value: str) -> RunTags:
        """Add a single tag. Returns self for chaining."""
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError(
                f"Tag key and value must be str, got key={type(key).__name__}, "
                f"value={type(value).__name__}"
            )
        self._tags[key] = value
        return self

    def remove_tag(self, key: str) -> RunTags:
        """Remove a tag by key. Raises ``KeyError`` if not present. Returns self for chaining."""
        del self._tags[key]
        return self

    def to_dict(self) -> dict[str, str]:
        """Return a copy of all composed tags."""
        return dict(self._tags)

    def __getitem__(self, key: str) -> str:
        return self._tags[key]

    def __len__(self) -> int:
        return len(self._tags)

    def __contains__(self, key: object) -> bool:
        return key in self._tags

    def __repr__(self) -> str:
        return f"RunTags({self._tags!r})"
