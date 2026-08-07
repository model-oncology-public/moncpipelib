"""Tests for moncpipelib.tags (ContractTags + RunTags)."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from moncpipelib.contracts.models import (
    SLA,
    Column,
    ColumnType,
    DataContract,
    LineageConfig,
    Owner,
    Schema,
)
from moncpipelib.tags import TAG_NAMESPACE, ContractTags, RunTags


def _make_contract(
    *,
    asset: str = "test_asset",
    layer: str = "silver",
    pipeline_id: str = "550e8400-e29b-41d4-a716-446655440000",
    owner: Owner | None = None,
    sla: SLA | None = None,
    lineage: LineageConfig | None = None,
    pii_columns: bool = False,
    tags: dict[str, str] | None = None,
) -> DataContract:
    columns = [
        Column(name="id", type=ColumnType.INTEGER, nullable=False, primary_key=True, pii=False),
    ]
    if pii_columns:
        columns.append(
            Column(
                name="patient_name",
                type=ColumnType.STRING,
                nullable=True,
                pii=True,
            )
        )
    return DataContract(
        version="1.0",
        pipeline_id=pipeline_id,
        asset=asset,
        layer=layer,
        schema=Schema(columns=columns),
        owner=owner,
        sla=sla,
        lineage=lineage,
        tags=tags or {},
    )


# ---------------------------------------------------------------------------
# ContractTags.from_contract
# ---------------------------------------------------------------------------


class TestContractTagsFromContract:
    def test_all_fields(self) -> None:
        contract = _make_contract(
            owner=Owner(team="data-eng"),
            sla=SLA(freshness_hours=24),
            lineage=LineageConfig(source_system="sftp"),
            pii_columns=True,
        )
        tags = ContractTags.from_contract(contract).to_dict()

        assert tags[f"{TAG_NAMESPACE}/layer"] == "silver"
        assert tags[f"{TAG_NAMESPACE}/owner"] == "data-eng"
        assert tags[f"{TAG_NAMESPACE}/pipeline_id"] == contract.pipeline_id
        assert tags[f"{TAG_NAMESPACE}/has_sla"] == "true"
        assert tags[f"{TAG_NAMESPACE}/has_pii"] == "true"
        assert tags[f"{TAG_NAMESPACE}/source_system"] == "sftp"

    def test_minimal(self) -> None:
        contract = _make_contract()
        tags = ContractTags.from_contract(contract).to_dict()

        assert tags[f"{TAG_NAMESPACE}/layer"] == "silver"
        assert tags[f"{TAG_NAMESPACE}/pipeline_id"] == contract.pipeline_id
        assert tags[f"{TAG_NAMESPACE}/has_sla"] == "false"
        assert tags[f"{TAG_NAMESPACE}/has_pii"] == "false"
        assert f"{TAG_NAMESPACE}/owner" not in tags
        assert f"{TAG_NAMESPACE}/source_system" not in tags

    def test_includes_user_tags(self) -> None:
        contract = _make_contract(tags={"team/priority": "high", "env": "prod"})
        tags = ContractTags.from_contract(contract).to_dict()

        assert tags["team/priority"] == "high"
        assert tags["env"] == "prod"

    def test_reserved_namespace_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        contract = _make_contract(
            tags={f"{TAG_NAMESPACE}/custom": "override"},
        )
        with caplog.at_level(logging.WARNING, logger="moncpipelib.tags"):
            ContractTags.from_contract(contract)

        assert any("reserved namespace" in r.message for r in caplog.records)

    def test_no_pii_column_names(self) -> None:
        contract = _make_contract(pii_columns=True)
        tags = ContractTags.from_contract(contract).to_dict()

        all_values = " ".join(tags.values())
        assert "patient_name" not in all_values

    def test_to_dict_returns_copy(self) -> None:
        ct = ContractTags.from_contract(_make_contract())
        d1 = ct.to_dict()
        d1["injected"] = "value"
        assert "injected" not in ct.to_dict()


# ---------------------------------------------------------------------------
# ContractTags.from_assets
# ---------------------------------------------------------------------------


class TestContractTagsFromAssets:
    def test_aggregation(self, tmp_path: Path) -> None:
        """Multiple contracts with different layers get comma-joined."""
        c1 = _make_contract(
            asset="a1", layer="bronze", pipeline_id="00000000-0000-0000-0000-000000000001"
        )
        c2 = _make_contract(
            asset="a2",
            layer="silver",
            pipeline_id="00000000-0000-0000-0000-000000000002",
            owner=Owner(team="analytics"),
        )

        def mock_load(asset_name: str, **_kwargs: object) -> DataContract | None:
            return {"a1": c1, "a2": c2}.get(asset_name)

        with patch("moncpipelib.contracts.loader.load_contract_for_asset", side_effect=mock_load):
            tags = ContractTags.from_assets(tmp_path, ["a1", "a2"]).to_dict()

        assert tags[f"{TAG_NAMESPACE}/layer"] == "bronze,silver"
        assert tags[f"{TAG_NAMESPACE}/owner"] == "analytics"

    def test_boolean_or(self, tmp_path: Path) -> None:
        c1 = _make_contract(asset="a1", sla=SLA(freshness_hours=24))
        c2 = _make_contract(asset="a2")

        def mock_load(asset_name: str, **_kwargs: object) -> DataContract | None:
            return {"a1": c1, "a2": c2}.get(asset_name)

        with patch("moncpipelib.contracts.loader.load_contract_for_asset", side_effect=mock_load):
            tags = ContractTags.from_assets(tmp_path, ["a1", "a2"]).to_dict()

        assert tags[f"{TAG_NAMESPACE}/has_sla"] == "true"

    def test_conflict_raises(self, tmp_path: Path) -> None:
        c1 = _make_contract(asset="a1", tags={"priority": "high"})
        c2 = _make_contract(asset="a2", tags={"priority": "low"})

        def mock_load(asset_name: str, **_kwargs: object) -> DataContract | None:
            return {"a1": c1, "a2": c2}.get(asset_name)

        with (
            patch("moncpipelib.contracts.loader.load_contract_for_asset", side_effect=mock_load),
            pytest.raises(ValueError, match="Conflicting user tag"),
        ):
            ContractTags.from_assets(tmp_path, ["a1", "a2"])

    def test_empty(self, tmp_path: Path) -> None:
        with patch("moncpipelib.contracts.loader.load_contract_for_asset", return_value=None):
            tags = ContractTags.from_assets(tmp_path, ["missing"]).to_dict()

        assert tags == {}


# ---------------------------------------------------------------------------
# RunTags
# ---------------------------------------------------------------------------


class TestRunTags:
    def test_add_contract_tags(self) -> None:
        ct = ContractTags.from_contract(_make_contract(layer="gold"))
        rt = RunTags()
        rt.add_contract_tags(ct)
        assert rt.to_dict()[f"{TAG_NAMESPACE}/layer"] == "gold"

    def test_add_tags_dict(self) -> None:
        rt = RunTags()
        rt.add_tags({"image/version": "1.2.3", "env": "staging"})
        d = rt.to_dict()
        assert d["image/version"] == "1.2.3"
        assert d["env"] == "staging"

    def test_add_tag_single(self) -> None:
        rt = RunTags()
        rt.add_tag("key", "value")
        assert rt.to_dict()["key"] == "value"

    def test_add_tags_type_error(self) -> None:
        rt = RunTags()
        with pytest.raises(TypeError, match="dict\\[str, str\\]"):
            rt.add_tags({"key": 123})  # type: ignore[dict-item]

    def test_add_tag_type_error(self) -> None:
        rt = RunTags()
        with pytest.raises(TypeError, match="str"):
            rt.add_tag("key", 123)  # type: ignore[arg-type]

    def test_chaining(self) -> None:
        rt = RunTags()
        result = rt.add_tag("a", "1").add_tags({"b": "2"}).add_tag("c", "3")
        assert result is rt
        assert len(rt) == 3

    def test_override_order(self) -> None:
        rt = RunTags()
        rt.add_tag("k", "first")
        rt.add_tag("k", "second")
        assert rt.to_dict()["k"] == "second"

    def test_to_dict_returns_copy(self) -> None:
        rt = RunTags()
        rt.add_tag("k", "v")
        d = rt.to_dict()
        d["injected"] = "value"
        assert "injected" not in rt.to_dict()

    def test_contains_and_len(self) -> None:
        rt = RunTags()
        rt.add_tag("present", "yes")
        assert "present" in rt
        assert "absent" not in rt
        assert len(rt) == 1

    def test_remove_tag(self) -> None:
        rt = RunTags().add_tag("a", "1").add_tag("b", "2")
        result = rt.remove_tag("a")
        assert result is rt
        assert "a" not in rt
        assert rt.to_dict() == {"b": "2"}

    def test_remove_tag_missing_raises(self) -> None:
        rt = RunTags()
        with pytest.raises(KeyError):
            rt.remove_tag("nonexistent")

    def test_getitem(self) -> None:
        rt = RunTags().add_tag("key", "value")
        assert rt["key"] == "value"

    def test_getitem_missing_raises(self) -> None:
        rt = RunTags()
        with pytest.raises(KeyError):
            rt["missing"]  # noqa: B018

    def test_repr(self) -> None:
        rt = RunTags().add_tag("env", "prod")
        r = repr(rt)
        assert r == "RunTags({'env': 'prod'})"
