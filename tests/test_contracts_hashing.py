"""Tests for ``moncpipelib.contracts.hashing``.

Migration 019 (#308) Phase 3 introduces stable content fingerprints
for ``DataContract``. These tests pin two contracts of behaviour:

1. Determinism: re-hashing the same parsed contract always produces the
   same digest; dict key order in the input must NOT affect output.
2. Sensitivity: a semantically meaningful change to the contract must
   produce a different ``contract_hash``; a schema-only change must
   produce a different ``schema_fingerprint``; the two hashes are
   orthogonal in their sensitivity bands.
"""

from __future__ import annotations

import copy
from pathlib import Path

from moncpipelib.contracts.hashing import (
    compute_contract_hash,
    compute_schema_fingerprint,
    derive_data_classification,
)
from moncpipelib.contracts.models import (
    SLA,
    Column,
    ColumnType,
    DataContract,
    Owner,
    Schema,
)


def _make_contract(
    *,
    description: str | None = "demo",
    sla_hours: int | None = 24,
    owner_team: str = "data_platform",
    tags: dict[str, str] | None = None,
    extra_columns: list[Column] | None = None,
) -> DataContract:
    """Build a small but realistic ``DataContract`` for hashing tests."""
    cols: list[Column] = [
        Column(name="id", type=ColumnType.INTEGER, nullable=False, pii=False, primary_key=True),
        Column(name="patient_name", type=ColumnType.STRING, nullable=False, pii=True),
        Column(name="created_at", type=ColumnType.DATETIME, nullable=False, pii=False),
    ]
    if extra_columns:
        cols.extend(extra_columns)
    return DataContract(
        version="1.0",
        pipeline_id="11111111-2222-3333-4444-555555555555",
        asset="bronze__demo",
        layer="bronze",
        schema=Schema(columns=cols),
        description=description,
        owner=Owner(team=owner_team),
        sla=SLA(freshness_hours=sla_hours) if sla_hours is not None else None,
        tags=tags or {},
    )


class TestComputeContractHash:
    """Determinism + sensitivity for ``compute_contract_hash``."""

    def test_same_contract_same_hash(self) -> None:
        c1 = _make_contract()
        c2 = _make_contract()
        assert compute_contract_hash(c1) == compute_contract_hash(c2)

    def test_returns_hex_sha256_digest(self) -> None:
        digest = compute_contract_hash(_make_contract())
        assert isinstance(digest, str)
        assert len(digest) == 64
        int(digest, 16)  # asserts the digest is valid hex

    def test_excludes_contract_hash_field_itself(self) -> None:
        """Pre-populating ``contract_hash`` on the contract must not
        change the digest -- otherwise re-hashing after the loader stamps
        the field would produce a different digest."""
        c = _make_contract()
        baseline = compute_contract_hash(c)
        c.contract_hash = "already_populated"
        c.schema_fingerprint = "already_populated"
        assert compute_contract_hash(c) == baseline

    def test_sensitive_to_column_added(self) -> None:
        c1 = _make_contract()
        c2 = _make_contract(
            extra_columns=[Column(name="dob", type=ColumnType.DATE, nullable=True, pii=True)]
        )
        assert compute_contract_hash(c1) != compute_contract_hash(c2)

    def test_sensitive_to_sla_change(self) -> None:
        c1 = _make_contract(sla_hours=24)
        c2 = _make_contract(sla_hours=48)
        assert compute_contract_hash(c1) != compute_contract_hash(c2)

    def test_sensitive_to_owner_change(self) -> None:
        c1 = _make_contract(owner_team="data_platform")
        c2 = _make_contract(owner_team="analytics")
        assert compute_contract_hash(c1) != compute_contract_hash(c2)

    def test_sensitive_to_tags_change(self) -> None:
        c1 = _make_contract(tags={"domain": "claims"})
        c2 = _make_contract(tags={"domain": "providers"})
        assert compute_contract_hash(c1) != compute_contract_hash(c2)

    def test_insensitive_to_tag_key_order(self) -> None:
        """The hash must be insensitive to dict iteration order -- two
        contracts whose ``tags`` differ only by insertion order produce
        the same digest."""
        c1 = _make_contract(tags={"a": "1", "b": "2", "c": "3"})
        c2 = _make_contract(tags={"c": "3", "a": "1", "b": "2"})
        assert compute_contract_hash(c1) == compute_contract_hash(c2)

    def test_idempotent_under_deepcopy(self) -> None:
        """``copy.deepcopy`` produces a structurally identical contract;
        the hash must remain stable."""
        c1 = _make_contract()
        c2 = copy.deepcopy(c1)
        assert compute_contract_hash(c1) == compute_contract_hash(c2)

    def test_insensitive_to_explicit_phi_matching_pii(self) -> None:
        """An explicit ``phi`` equal to ``pii`` hashes like the unset
        (mirrored) default, so pre-phi contract hashes stay stable (#391)."""
        c1 = _make_contract()
        c2 = _make_contract()
        c2.schema.columns[1] = Column(
            name="patient_name", type=ColumnType.STRING, nullable=False, pii=True, phi=True
        )
        assert compute_contract_hash(c1) == compute_contract_hash(c2)

    def test_sensitive_to_phi_divergence(self) -> None:
        c1 = _make_contract()
        c2 = _make_contract()
        c2.schema.columns[1] = Column(
            name="patient_name", type=ColumnType.STRING, nullable=False, pii=True, phi=False
        )
        assert compute_contract_hash(c1) != compute_contract_hash(c2)


class TestComputeSchemaFingerprint:
    """Schema fingerprint is sensitive to schema identity changes only."""

    def test_same_schema_same_fingerprint(self) -> None:
        c1 = _make_contract()
        c2 = _make_contract()
        assert compute_schema_fingerprint(c1) == compute_schema_fingerprint(c2)

    def test_returns_hex_sha256_digest(self) -> None:
        digest = compute_schema_fingerprint(_make_contract())
        assert isinstance(digest, str)
        assert len(digest) == 64
        int(digest, 16)

    def test_sensitive_to_column_add(self) -> None:
        c1 = _make_contract()
        c2 = _make_contract(
            extra_columns=[Column(name="email", type=ColumnType.STRING, nullable=True, pii=True)]
        )
        assert compute_schema_fingerprint(c1) != compute_schema_fingerprint(c2)

    def test_sensitive_to_type_change(self) -> None:
        c1 = _make_contract()
        c2 = _make_contract()
        c2.schema.columns[0].type = ColumnType.STRING  # was INTEGER
        assert compute_schema_fingerprint(c1) != compute_schema_fingerprint(c2)

    def test_sensitive_to_nullable_flip(self) -> None:
        c1 = _make_contract()
        c2 = _make_contract()
        c2.schema.columns[0].nullable = not c2.schema.columns[0].nullable
        assert compute_schema_fingerprint(c1) != compute_schema_fingerprint(c2)

    def test_sensitive_to_pii_flip(self) -> None:
        c1 = _make_contract()
        c2 = _make_contract()
        c2.schema.columns[1].pii = False  # was True
        assert compute_schema_fingerprint(c1) != compute_schema_fingerprint(c2)

    def test_insensitive_to_description_change(self) -> None:
        """``description`` is part of ``contract_hash`` but NOT
        ``schema_fingerprint`` -- two contracts with different
        descriptions hash to the same schema fingerprint."""
        c1 = _make_contract(description="original")
        c2 = _make_contract(description="rewritten for clarity")
        assert compute_schema_fingerprint(c1) == compute_schema_fingerprint(c2)

    def test_insensitive_to_sla_change(self) -> None:
        c1 = _make_contract(sla_hours=24)
        c2 = _make_contract(sla_hours=48)
        assert compute_schema_fingerprint(c1) == compute_schema_fingerprint(c2)

    def test_insensitive_to_tags_change(self) -> None:
        c1 = _make_contract(tags={"a": "1"})
        c2 = _make_contract(tags={"b": "2"})
        assert compute_schema_fingerprint(c1) == compute_schema_fingerprint(c2)

    def test_insensitive_to_column_order(self) -> None:
        """Columns are sorted by name before hashing so a reordering in
        the YAML produces the same fingerprint."""
        c1 = _make_contract()
        c2 = _make_contract()
        c2.schema.columns = list(reversed(c2.schema.columns))
        assert compute_schema_fingerprint(c1) == compute_schema_fingerprint(c2)

    def test_insensitive_to_explicit_phi_matching_pii(self) -> None:
        """An explicit ``phi`` equal to ``pii`` fingerprints like the unset
        (mirrored) default, so pre-phi fingerprints stay stable (#391)."""
        c1 = _make_contract()
        c2 = _make_contract()
        c2.schema.columns[1] = Column(
            name="patient_name", type=ColumnType.STRING, nullable=False, pii=True, phi=True
        )
        assert compute_schema_fingerprint(c1) == compute_schema_fingerprint(c2)

    def test_sensitive_to_phi_divergence(self) -> None:
        c1 = _make_contract()
        c2 = _make_contract()
        c2.schema.columns[1] = Column(
            name="patient_name", type=ColumnType.STRING, nullable=False, pii=True, phi=False
        )
        assert compute_schema_fingerprint(c1) != compute_schema_fingerprint(c2)


class TestHashOrthogonality:
    """``contract_hash`` and ``schema_fingerprint`` are orthogonal in their
    sensitivity bands. Pinned so a future refactor doesn't collapse them.
    """

    def test_description_changes_contract_hash_not_fingerprint(self) -> None:
        c1 = _make_contract(description="original")
        c2 = _make_contract(description="rewritten")
        assert compute_contract_hash(c1) != compute_contract_hash(c2)
        assert compute_schema_fingerprint(c1) == compute_schema_fingerprint(c2)

    def test_column_type_change_changes_both(self) -> None:
        c1 = _make_contract()
        c2 = _make_contract()
        c2.schema.columns[0].type = ColumnType.STRING
        assert compute_contract_hash(c1) != compute_contract_hash(c2)
        assert compute_schema_fingerprint(c1) != compute_schema_fingerprint(c2)


class TestDeriveDataClassification:
    """``derive_data_classification`` rolls up column-level PII flags."""

    def test_phi_when_any_non_managed_column_is_pii(self) -> None:
        c = _make_contract()  # has patient_name (pii=True)
        assert derive_data_classification(c) == "PHI"

    def test_none_when_no_pii_columns(self) -> None:
        c = DataContract(
            version="1.0",
            pipeline_id="x",
            asset="x",
            layer="bronze",
            schema=Schema(
                columns=[
                    Column(name="id", type=ColumnType.INTEGER, nullable=False, pii=False),
                    Column(name="counter", type=ColumnType.INTEGER, nullable=False, pii=False),
                ]
            ),
        )
        assert derive_data_classification(c) == "none"

    def test_managed_pii_columns_excluded(self) -> None:
        """Managed columns (auto-injected by moncpipelib) carrying
        ``pii=True`` as metadata-of-metadata must NOT promote the rollup
        to ``PHI``. Only application-facing columns should contribute."""
        c = DataContract(
            version="1.0",
            pipeline_id="x",
            asset="x",
            layer="bronze",
            schema=Schema(
                columns=[
                    Column(name="id", type=ColumnType.INTEGER, nullable=False, pii=False),
                    Column(
                        name="_lineage_id",
                        type=ColumnType.UUID,
                        nullable=False,
                        pii=True,
                        managed=True,
                    ),
                ]
            ),
        )
        assert derive_data_classification(c) == "none"

    def test_empty_schema_returns_none(self) -> None:
        c = DataContract(
            version="1.0",
            pipeline_id="x",
            asset="x",
            layer="bronze",
            schema=Schema(columns=[]),
        )
        assert derive_data_classification(c) == "none"

    def test_none_when_all_columns_cleared_of_phi(self) -> None:
        """Explicitly clearing every column (``phi: false``) is the only
        way a PII-bearing contract classifies as ``none`` (#391)."""
        c = DataContract(
            version="1.0",
            pipeline_id="x",
            asset="x",
            layer="bronze",
            schema=Schema(
                columns=[
                    Column(
                        name="provider_npi",
                        type=ColumnType.STRING,
                        nullable=False,
                        pii=True,
                        phi=False,
                    ),
                    Column(name="id", type=ColumnType.INTEGER, nullable=False, pii=False),
                ]
            ),
        )
        assert derive_data_classification(c) == "none"

    def test_phi_when_phi_true_on_non_pii_column(self) -> None:
        """A ``phi: true`` annotation promotes the rollup even when
        ``pii`` is false (de-identification reversed, for example)."""
        c = DataContract(
            version="1.0",
            pipeline_id="x",
            asset="x",
            layer="bronze",
            schema=Schema(
                columns=[
                    Column(
                        name="lab_value",
                        type=ColumnType.STRING,
                        nullable=False,
                        pii=False,
                        phi=True,
                    ),
                ]
            ),
        )
        assert derive_data_classification(c) == "PHI"


class TestLoaderPopulatesFingerprints:
    """Integration with ``load_contract`` -- both fingerprint fields are
    populated after a YAML round-trip."""

    def test_load_contract_populates_both_fingerprints(self, tmp_path: Path) -> None:
        """A YAML-loaded contract has ``contract_hash`` /
        ``schema_fingerprint`` populated as 64-char hex strings."""
        import textwrap

        from moncpipelib.contracts.loader import load_contract

        contract_path = tmp_path / "demo.contract.yaml"
        contract_path.write_text(
            textwrap.dedent(
                """
                version: "1.0"
                pipeline_id: "11111111-2222-3333-4444-555555555555"
                asset: "bronze__demo"
                layer: "bronze"
                schema:
                  columns:
                    - name: id
                      type: integer
                      nullable: false
                      primary_key: true
                      pii: false
                    - name: patient_name
                      type: string
                      nullable: false
                      pii: true
                """
            ).strip()
        )

        contract = load_contract(contract_path)

        assert isinstance(contract.contract_hash, str)
        assert len(contract.contract_hash) == 64
        int(contract.contract_hash, 16)

        assert isinstance(contract.schema_fingerprint, str)
        assert len(contract.schema_fingerprint) == 64
        int(contract.schema_fingerprint, 16)

    def test_load_contract_hashes_are_stable_across_reloads(self, tmp_path: Path) -> None:
        """Loading the same file twice must produce the same hash --
        no time-based or run-id leakage into the fingerprint."""
        import textwrap

        from moncpipelib.contracts.loader import load_contract

        contract_path = tmp_path / "demo2.contract.yaml"
        contract_path.write_text(
            textwrap.dedent(
                """
                version: "1.0"
                pipeline_id: "22222222-2222-2222-2222-222222222222"
                asset: "bronze__demo2"
                layer: "bronze"
                schema:
                  columns:
                    - name: id
                      type: integer
                      nullable: false
                      pii: false
                """
            ).strip()
        )

        c1 = load_contract(contract_path)
        c2 = load_contract(contract_path)
        assert c1.contract_hash == c2.contract_hash
        assert c1.schema_fingerprint == c2.schema_fingerprint
