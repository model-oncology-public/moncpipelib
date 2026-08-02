"""Tests for contract loader."""

import logging
from pathlib import Path

import pytest

from moncpipelib.config import LineageDefaults
from moncpipelib.contracts import (
    ColumnType,
    ContractNotFoundError,
    ContractValidationError,
    DataContract,
    Severity,
    load_contract,
    load_contract_for_asset,
    validate_contract_schema,
)
from moncpipelib.contracts.loader import (
    _FIELD_TO_PATHS,
    KNOWN_COLUMN_FIELDS,
    KNOWN_COLUMN_TEST_PARAMS,
    KNOWN_COLUMN_TEST_TYPES,
    KNOWN_EXPECTATION_PARAMS,
    KNOWN_EXPECTATION_TYPES,
    KNOWN_OWNER_FIELDS,
    KNOWN_SCHEMA_FIELDS,
    KNOWN_SLA_FIELDS,
    KNOWN_TESTING_FIELDS,
    KNOWN_TOP_LEVEL_FIELDS,
    KNOWN_UPSTREAM_FIELDS,
    _build_contract_index,
    _clear_contract_index_cache,
)


@pytest.fixture(autouse=True)
def _clear_index_cache():
    """Clear the contract index cache before and after each test."""
    _clear_contract_index_cache()
    yield
    _clear_contract_index_cache()


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
INVALID_DIR = FIXTURES_DIR / "invalid_contracts"


class TestLoadContract:
    """Tests for load_contract function."""

    def test_load_valid_contract(self):
        """Test loading a valid contract YAML file."""
        contract = load_contract(FIXTURES_DIR / "valid_contract.yaml")

        assert isinstance(contract, DataContract)
        assert contract.version == "1.0"
        assert contract.asset == "claims_bronze"
        assert contract.layer == "bronze"
        assert contract.description is not None
        assert "Raw claims data" in contract.description

    def test_load_minimal_contract(self):
        """Test loading a minimal contract with only required fields."""
        contract = load_contract(FIXTURES_DIR / "minimal_contract.yaml")

        assert contract.version == "1.0"
        assert contract.asset == "simple_asset"
        assert contract.layer == "bronze"
        assert contract.description is None
        assert contract.owner is None
        assert len(contract.schema.columns) == 2

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(ContractNotFoundError) as exc:
            load_contract(FIXTURES_DIR / "nonexistent.yaml")
        assert "not found" in str(exc.value).lower()

    def test_load_invalid_yaml_syntax(self, tmp_path):
        """Test loading a file with invalid YAML syntax."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("version: 1.0\nasset: {unclosed")

        with pytest.raises(ContractValidationError) as exc:
            load_contract(invalid_yaml)
        assert "Invalid YAML" in str(exc.value)

    def test_load_empty_file(self, tmp_path):
        """Test loading an empty YAML file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        with pytest.raises(ContractValidationError) as exc:
            load_contract(empty_file)
        assert "Empty contract file" in str(exc.value)


class TestContractValidation:
    """Tests for contract YAML validation."""

    def test_missing_required_fields(self):
        """Test validation catches missing required fields."""
        data = {"version": "1.0"}  # Missing asset, layer, schema
        errors = validate_contract_schema(data)
        assert any("Missing required fields" in e for e in errors)

    def test_unsupported_version(self):
        """Test validation catches unsupported version."""
        data = {
            "version": "2.0",
            "pipeline_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "asset": "test",
            "layer": "bronze",
            "schema": {"columns": [{"name": "id", "type": "integer", "nullable": False}]},
        }
        errors = validate_contract_schema(data)
        assert any("Unsupported contract version" in e for e in errors)

    def test_empty_columns(self):
        """Test validation catches empty columns list."""
        with pytest.raises(ContractValidationError) as exc:
            load_contract(INVALID_DIR / "empty_columns.yaml")
        assert "columns" in str(exc.value).lower()

    def test_invalid_column_type(self):
        """Test validation catches invalid column type."""
        with pytest.raises(ContractValidationError) as exc:
            load_contract(INVALID_DIR / "invalid_column_type.yaml")
        assert "invalid type" in str(exc.value).lower()

    @pytest.mark.parametrize("col_type", ["json", "jsonb"])
    def test_json_column_types_accepted(self, col_type: str, tmp_path: Path) -> None:
        """Loader accepts json and jsonb as valid column types."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text(f"""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: bronze
schema:
  columns:
    - name: payload
      type: {col_type}
      nullable: true
      pii: false
""")
        contract = load_contract(contract_yaml)
        col = contract.get_column("payload")
        assert col is not None
        assert col.type.value == col_type

    def test_missing_version(self):
        """Test validation catches missing version field."""
        with pytest.raises(ContractValidationError) as exc:
            load_contract(INVALID_DIR / "missing_version.yaml")
        assert "missing required fields" in str(exc.value).lower()

    def test_missing_schema(self):
        """Test validation catches missing schema field."""
        with pytest.raises(ContractValidationError) as exc:
            load_contract(INVALID_DIR / "missing_schema.yaml")
        assert "missing required fields" in str(exc.value).lower()


class TestContractParsing:
    """Tests for contract data parsing."""

    @pytest.fixture
    def full_contract(self):
        """Load the full valid contract for parsing tests."""
        return load_contract(FIXTURES_DIR / "valid_contract.yaml")

    def test_owner_parsing(self, full_contract):
        """Test owner metadata is parsed correctly."""
        owner = full_contract.owner
        assert owner is not None
        assert owner.team == "data-engineering"
        assert owner.contact == "data-platform@modeloncology.com"
        assert owner.slack_channel == "#data-platform-alerts"

    def test_column_parsing(self, full_contract):
        """Test columns are parsed correctly."""
        columns = full_contract.schema.columns
        assert len(columns) == 8

        # Check claim_id column
        claim_id = full_contract.get_column("claim_id")
        assert claim_id is not None
        assert claim_id.type == ColumnType.STRING
        assert claim_id.nullable is False
        assert claim_id.primary_key is True
        assert len(claim_id.tests) == 2  # not_null, unique

    def test_column_tests_parsing(self, full_contract):
        """Test column tests are parsed correctly."""
        amount = full_contract.get_column("amount")
        assert amount is not None
        assert len(amount.tests) == 2

        # Check test with severity
        greater_than = amount.tests[0]
        assert greater_than.test_type == "greater_than"
        assert greater_than.parameters.get("value") == 0
        assert greater_than.severity == Severity.WARN

        less_than = amount.tests[1]
        assert less_than.test_type == "less_than"
        assert less_than.parameters.get("value") == 1000000
        assert less_than.severity == Severity.ERROR

    def test_test_with_when_condition(self, full_contract):
        """Test parsing tests with when condition."""
        diag_code = full_contract.get_column("diagnosis_code")
        assert diag_code is not None
        assert len(diag_code.tests) == 1

        pattern_test = diag_code.tests[0]
        assert pattern_test.test_type == "pattern"
        assert pattern_test.when == "not_null"
        assert pattern_test.severity == Severity.WARN

    def test_accepted_values_parsing(self, full_contract):
        """Test parsing accepted_values test."""
        status = full_contract.get_column("status")
        assert status is not None

        # Find the accepted_values test
        av_test = None
        for test in status.tests:
            if test.test_type == "accepted_values":
                av_test = test
                break

        assert av_test is not None
        assert av_test.parameters.get("values") == ["pending", "approved", "denied", "appealed"]

    def test_managed_columns(self, full_contract):
        """Test managed columns are parsed correctly."""
        lineage_id = full_contract.get_column(LineageDefaults.ID_COLUMN)
        assert lineage_id is not None
        assert lineage_id.managed is True
        assert lineage_id.type == ColumnType.UUID

    def test_expectations_parsing(self, full_contract):
        """Test table expectations are parsed correctly."""
        expectations = full_contract.expectations
        assert len(expectations) == 4

        # Find row_count expectation
        row_count = None
        for exp in expectations:
            if exp.expectation_type == "row_count":
                row_count = exp
                break

        assert row_count is not None
        assert row_count.parameters.get("min") == 1
        assert row_count.parameters.get("max") == 10000000
        assert row_count.severity == Severity.ERROR

    def test_upstream_parsing(self, full_contract):
        """Test upstream dependencies are parsed correctly."""
        upstream = full_contract.upstream
        assert len(upstream) == 1

        sftp_source = upstream[0]
        assert sftp_source.name == "sftp_claims_file"
        assert sftp_source.type == "external"
        assert sftp_source.system == "sftp"
        assert sftp_source.description is not None

    def test_sla_parsing(self, full_contract):
        """Test SLA metadata is parsed correctly."""
        sla = full_contract.sla
        assert sla is not None
        assert sla.freshness_hours == 24
        assert sla.update_frequency == "daily"
        assert sla.availability_percent == 99.9


class TestLoadContractForAsset:
    """Tests for load_contract_for_asset function."""

    def test_find_contract_in_search_paths(self, tmp_path):
        """Test finding contract in provided search paths."""
        # Create a test contract
        contract_dir = tmp_path / "contracts"
        contract_dir.mkdir()
        contract_file = contract_dir / "test_asset.contract.yaml"
        contract_file.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: bronze
schema:
  columns:
    - name: id
      type: integer
      nullable: false
""")

        contract = load_contract_for_asset(
            "test_asset",
            layer="bronze",
            search_paths=[contract_dir],
        )

        assert contract is not None
        assert contract.asset == "test_asset"

    def test_contract_not_found(self, tmp_path):
        """Test returns None when contract not found."""
        contract = load_contract_for_asset(
            "nonexistent_asset",
            layer="bronze",
            search_paths=[tmp_path],
        )
        assert contract is None

    def test_search_priority(self, tmp_path):
        """Test search paths are checked in order."""
        # Create two directories with different contracts
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        # Contract in dir1
        (dir1 / "priority_asset.contract.yaml").write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: priority_asset
layer: bronze
description: "From dir1"
schema:
  columns:
    - name: id
      type: integer
      nullable: false
""")

        # Different contract in dir2
        (dir2 / "priority_asset.contract.yaml").write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: priority_asset
layer: bronze
description: "From dir2"
schema:
  columns:
    - name: id
      type: integer
      nullable: false
""")

        # dir1 is first in search paths, so it should be found first
        contract = load_contract_for_asset(
            "priority_asset",
            layer="bronze",
            search_paths=[dir1, dir2],
        )

        assert contract is not None
        assert contract.description == "From dir1"


_MINIMAL_CONTRACT_YAML = """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: {asset}
layer: bronze
schema:
  columns:
    - name: id
      type: integer
      nullable: false
"""


class TestContractDiscoveryDeprecation:
    """Tests for deprecation warnings on implicit contract discovery."""

    def test_cwd_fallback_emits_deprecation_warning(self, tmp_path, monkeypatch):
        """Warn when contract is resolved via cwd fallback."""
        contract_file = tmp_path / "cwd_asset.contract.yaml"
        contract_file.write_text(_MINIMAL_CONTRACT_YAML.format(asset="cwd_asset"))
        monkeypatch.chdir(tmp_path)
        with pytest.warns(DeprecationWarning, match="current working directory"):
            contract = load_contract_for_asset("cwd_asset", caller_file="/nonexistent/path.py")
        assert contract is not None

    def test_layer_dir_fallback_emits_deprecation_warning(self, tmp_path, monkeypatch):
        """Warn when contract is resolved via assets/{layer}/ fallback."""
        layer_dir = tmp_path / "assets" / "bronze"
        layer_dir.mkdir(parents=True)
        contract_file = layer_dir / "layer_asset.contract.yaml"
        contract_file.write_text(_MINIMAL_CONTRACT_YAML.format(asset="layer_asset"))
        monkeypatch.chdir(tmp_path)
        with pytest.warns(DeprecationWarning, match="assets/bronze/"):
            contract = load_contract_for_asset(
                "layer_asset", layer="bronze", caller_file="/nonexistent/path.py"
            )
        assert contract is not None

    def test_explicit_search_paths_no_warning(self, tmp_path):
        """No deprecation warning when explicit search_paths is provided."""
        import warnings

        contract_file = tmp_path / "explicit_asset.contract.yaml"
        contract_file.write_text(_MINIMAL_CONTRACT_YAML.format(asset="explicit_asset"))
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            contract = load_contract_for_asset("explicit_asset", search_paths=[tmp_path])
        assert contract is not None

    def test_no_warning_when_nothing_found(self, tmp_path, monkeypatch):
        """No deprecation warning when no contract is found anywhere."""
        import warnings

        monkeypatch.chdir(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = load_contract_for_asset(
                "nonexistent_asset", caller_file="/nonexistent/path.py"
            )
        assert result is None


class TestContentBasedContractDiscovery:
    """Tests for content-based contract discovery via search_paths."""

    def test_matches_by_asset_field_not_filename(self, tmp_path):
        """Contract matched by its YAML asset field, not filename."""
        # File named differently than the asset field
        contract_file = tmp_path / "gold_synthetic_bridge_business_hours.contract.yaml"
        contract_file.write_text(_MINIMAL_CONTRACT_YAML.format(asset="bridge_business_hours_gold"))

        contract = load_contract_for_asset(
            "bridge_business_hours_gold",
            search_paths=[tmp_path],
        )

        assert contract is not None
        assert contract.asset == "bridge_business_hours_gold"

    def test_recursive_discovery_in_subdirectories(self, tmp_path):
        """Contracts in nested subdirectories are discovered recursively."""
        nested_dir = tmp_path / "defs" / "gold" / "dim"
        nested_dir.mkdir(parents=True)
        contract_file = nested_dir / "dim_patient.contract.yaml"
        contract_file.write_text(_MINIMAL_CONTRACT_YAML.format(asset="dim_patient"))

        contract = load_contract_for_asset(
            "dim_patient",
            search_paths=[tmp_path],
        )

        assert contract is not None
        assert contract.asset == "dim_patient"

    def test_search_paths_does_not_fallback_to_deprecated(self, tmp_path, monkeypatch):
        """When search_paths is provided, deprecated strategies are not used."""
        import warnings

        # Put a contract in CWD that the deprecated strategy would find
        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        (cwd_dir / "cwd_only_asset.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(asset="cwd_only_asset")
        )
        monkeypatch.chdir(cwd_dir)

        # search_paths points to an empty directory -- should NOT fall through
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = load_contract_for_asset(
                "cwd_only_asset",
                search_paths=[empty_dir],
            )

        assert result is None

    def test_duplicate_asset_earlier_search_path_shadows(self, tmp_path, caplog):
        """A same-asset duplicate in a LATER search path is shadowed with a warning."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "a.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(asset="dup_asset").replace(
                "layer: bronze", "layer: bronze\ndescription: first"
            )
        )
        (dir2 / "b.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(asset="dup_asset").replace(
                "layer: bronze", "layer: bronze\ndescription: second"
            )
        )

        with caplog.at_level(logging.WARNING):
            contract = load_contract_for_asset(
                "dup_asset",
                search_paths=[dir1, dir2],
            )

        assert contract is not None
        assert contract.description == "first"
        assert "shadowed" in caplog.text

    def test_duplicate_asset_same_search_path_raises(self, tmp_path):
        """A same-asset/same-layer duplicate WITHIN one search path is a hard error (#405)."""
        (tmp_path / "a.contract.yaml").write_text(_MINIMAL_CONTRACT_YAML.format(asset="dup_asset"))
        (tmp_path / "b.contract.yaml").write_text(_MINIMAL_CONTRACT_YAML.format(asset="dup_asset"))

        with pytest.raises(
            ContractValidationError,
            match=r"Duplicate contract for asset 'dup_asset'.*cannot be disambiguated by sink",
        ):
            load_contract_for_asset("dup_asset", search_paths=[tmp_path])

    def test_overlapping_search_paths_not_duplicates(self, tmp_path):
        """The same file discovered via overlapping search paths is one contract."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "a.contract.yaml").write_text(_MINIMAL_CONTRACT_YAML.format(asset="one_asset"))

        contract = load_contract_for_asset("one_asset", search_paths=[tmp_path, subdir])

        assert contract is not None
        assert contract.asset == "one_asset"

    def test_build_contract_index_caches_results(self, tmp_path):
        """Index is built once and cached for the same search paths."""
        contract_file = tmp_path / "cached.contract.yaml"
        contract_file.write_text(_MINIMAL_CONTRACT_YAML.format(asset="cached_asset"))

        index1 = _build_contract_index([tmp_path])
        assert "cached_asset" in index1.by_asset

        # Remove the file -- cached index should still have it
        contract_file.unlink()
        index2 = _build_contract_index([tmp_path])
        assert index2 is index1  # Same object from cache

    def test_malformed_yaml_skipped(self, tmp_path, caplog):
        """Malformed YAML files are skipped with a warning."""
        (tmp_path / "bad.contract.yaml").write_text("{{invalid yaml")
        (tmp_path / "good.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(asset="good_asset")
        )

        with caplog.at_level(logging.WARNING):
            contract = load_contract_for_asset(
                "good_asset",
                search_paths=[tmp_path],
            )

        assert contract is not None
        assert "Failed to read contract file" in caplog.text

    def test_no_asset_field_skipped(self, tmp_path):
        """YAML files without an asset field are silently skipped."""
        (tmp_path / "no_asset.contract.yaml").write_text("version: '1.0'\n")
        (tmp_path / "has_asset.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(asset="has_asset")
        )

        contract = load_contract_for_asset(
            "has_asset",
            search_paths=[tmp_path],
        )

        assert contract is not None
        assert contract.asset == "has_asset"

    def test_multiple_search_paths_merged(self, tmp_path):
        """Contracts from multiple search paths are all indexed."""
        dir_silver = tmp_path / "silver"
        dir_gold = tmp_path / "gold"
        dir_silver.mkdir()
        dir_gold.mkdir()

        (dir_silver / "claims.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(asset="claims_silver")
        )
        (dir_gold / "dim_patient.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(asset="dim_patient")
        )

        # Both should be found from the combined search paths
        c1 = load_contract_for_asset("claims_silver", search_paths=[dir_silver, dir_gold])
        c2 = load_contract_for_asset("dim_patient", search_paths=[dir_silver, dir_gold])

        assert c1 is not None
        assert c1.asset == "claims_silver"
        assert c2 is not None
        assert c2.asset == "dim_patient"

    def test_mismatch_logs_available_assets_and_close_matches(self, tmp_path, caplog):
        """When asset not found but contracts exist, warn with available names."""
        (tmp_path / "gold_synthetic_dim_date.contract.yaml").write_text(
            _MINIMAL_CONTRACT_YAML.format(asset="gold_synthetic_dim_date")
        )

        with caplog.at_level(logging.WARNING):
            result = load_contract_for_asset(
                "dim_date_gold",
                search_paths=[tmp_path],
            )

        assert result is None
        assert "No contract found for asset 'dim_date_gold'" in caplog.text
        assert "gold_synthetic_dim_date" in caplog.text
        assert "Close matches:" in caplog.text

    def test_no_warning_when_search_path_empty(self, tmp_path, caplog):
        """No warning when search path has no contracts at all."""
        with caplog.at_level(logging.WARNING):
            result = load_contract_for_asset(
                "nonexistent",
                search_paths=[tmp_path],
            )

        assert result is None
        assert "No contract found" not in caplog.text


_LAYERED_CONTRACT_YAML = """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: {asset}
layer: {layer}
schema:
  columns:
    - name: id
      type: integer
      nullable: false
"""


class TestLayerAwareContractIndex:
    """Tests for layer-aware contract discovery and disambiguation."""

    def test_same_asset_different_layers_coexist(self, tmp_path):
        """Two contracts with same asset name but different layers both load."""
        (tmp_path / "bronze.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="fda_ndc_directory", layer="bronze")
        )
        (tmp_path / "silver.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="fda_ndc_directory", layer="silver")
        )

        bronze = load_contract_for_asset(
            "fda_ndc_directory", layer="bronze", search_paths=[tmp_path]
        )
        silver = load_contract_for_asset(
            "fda_ndc_directory", layer="silver", search_paths=[tmp_path]
        )

        assert bronze is not None
        assert bronze.layer == "bronze"
        assert silver is not None
        assert silver.layer == "silver"

    def test_slash_name_matches_last_component(self, tmp_path):
        """Slash-separated asset name falls back to last component."""
        (tmp_path / "ndc.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="fda_ndc_directory", layer="silver")
        )

        # Simulates AssetKey(["reference_silver", "fda_ndc_directory"]).to_user_string()
        result = load_contract_for_asset(
            "reference_silver/fda_ndc_directory", search_paths=[tmp_path]
        )

        assert result is not None
        assert result.asset == "fda_ndc_directory"

    def test_slash_name_disambiguated_by_layer(self, tmp_path):
        """Slash name + layer= selects the correct contract."""
        (tmp_path / "bronze.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="fda_ndc_directory", layer="bronze")
        )
        (tmp_path / "silver.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="fda_ndc_directory", layer="silver")
        )

        result = load_contract_for_asset(
            "reference_silver/fda_ndc_directory",
            layer="silver",
            search_paths=[tmp_path],
        )

        assert result is not None
        assert result.layer == "silver"

    def test_exact_match_beats_component_match(self, tmp_path):
        """A contract with asset: 'a/b' wins over asset: 'b' when queried as 'a/b'."""
        (tmp_path / "full.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="ns/widget", layer="silver")
        )
        (tmp_path / "short.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="widget", layer="silver")
        )

        result = load_contract_for_asset("ns/widget", search_paths=[tmp_path])

        assert result is not None
        # The exact match "ns/widget" should win, not the short "widget"
        assert result.asset == "ns/widget"

    def test_ambiguous_without_layer_raises(self, tmp_path):
        """Multiple same-name contracts without layer= raises an error."""
        (tmp_path / "bronze.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="shared_asset", layer="bronze")
        )
        (tmp_path / "silver.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="shared_asset", layer="silver")
        )

        with pytest.raises(ContractValidationError, match="no layer was provided"):
            load_contract_for_asset("shared_asset", search_paths=[tmp_path])

    def test_ambiguous_with_invalid_layer_raises(self, tmp_path):
        """Multiple same-name contracts with non-matching layer= raises an error."""
        (tmp_path / "bronze.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="shared_asset", layer="bronze")
        )
        (tmp_path / "silver.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="shared_asset", layer="silver")
        )

        with pytest.raises(ContractValidationError, match="did not resolve to a unique match"):
            load_contract_for_asset("shared_asset", layer="gold", search_paths=[tmp_path])

    def test_invalid_layer_in_contract_raises(self, tmp_path):
        """Contract with a layer not in VALID_LAYERS raises at index build time."""
        (tmp_path / "bad.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="test_asset", layer="reference_silver")
        )

        with pytest.raises(
            ContractValidationError,
            match=r"Invalid layer 'reference_silver'.*Valid layers are:",
        ):
            load_contract_for_asset("test_asset", search_paths=[tmp_path])

    def test_same_asset_same_layer_across_paths_shadows(self, tmp_path, caplog):
        """A same-asset/same-layer duplicate across search paths shadows with a warning."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "a.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="dup_asset", layer="bronze")
        )
        (dir2 / "b.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="dup_asset", layer="bronze")
        )

        with caplog.at_level(logging.WARNING):
            _clear_contract_index_cache()
            index = _build_contract_index([dir1, dir2])

        assert "shadowed" in caplog.text
        assert len(index.by_asset["dup_asset"]) == 1

    def test_same_asset_same_layer_same_path_raises(self, tmp_path):
        """A same-asset/same-layer duplicate within one search path raises (#405)."""
        (tmp_path / "a.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="dup_asset", layer="bronze")
        )
        (tmp_path / "b.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="dup_asset", layer="bronze")
        )

        with pytest.raises(
            ContractValidationError,
            match=r"Duplicate contract for asset 'dup_asset'.*cannot be disambiguated",
        ):
            _build_contract_index([tmp_path])


_SINKED_CONTRACT_YAML = """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: {asset}
layer: {layer}
description: "{description}"
schema:
  columns:
    - name: id
      type: integer
      nullable: false
      pii: false
sinks:
  - type: table
    schema: {sink_schema}
    table: {sink_table}
    mode: full_refresh
"""


class TestSinkQualifiedResolution:
    """Regression tests for #405: contract identity is sink-qualified.

    Two contracts may legitimately declare the same ``asset`` and ``layer``
    when their sinks land in different schemas (e.g. synthetic_gold.dim_provider
    and reference_gold.dim_provider). The index must keep both addressable and
    never silently resolve a schema-qualified lookup to the wrong one.
    """

    def _write_dim_provider_pair(self, tmp_path):
        (tmp_path / "synthetic_dim_provider.contract.yaml").write_text(
            _SINKED_CONTRACT_YAML.format(
                asset="dim_provider",
                layer="gold",
                description="synthetic",
                sink_schema="synthetic_gold",
                sink_table="dim_provider",
            )
        )
        (tmp_path / "reference_dim_provider.contract.yaml").write_text(
            _SINKED_CONTRACT_YAML.format(
                asset="dim_provider",
                layer="gold",
                description="reference",
                sink_schema="reference_gold",
                sink_table="dim_provider",
            )
        )

    def test_same_asset_same_layer_distinct_sinks_coexist(self, tmp_path):
        """Sink-disjoint same-asset/same-layer contracts both enter the index."""
        self._write_dim_provider_pair(tmp_path)

        index = _build_contract_index([tmp_path])

        assert len(index.by_asset["dim_provider"]) == 2
        assert "synthetic_gold/dim_provider" in index.by_sink
        assert "reference_gold/dim_provider" in index.by_sink

    def test_sink_qualified_lookup_resolves_correct_contract(self, tmp_path):
        """AssetKey-style 'schema/table' names resolve to the matching sink's contract."""
        self._write_dim_provider_pair(tmp_path)

        reference = load_contract_for_asset("reference_gold/dim_provider", search_paths=[tmp_path])
        synthetic = load_contract_for_asset("synthetic_gold/dim_provider", search_paths=[tmp_path])

        assert reference is not None and reference.description == "reference"
        assert synthetic is not None and synthetic.description == "synthetic"

    def test_sink_qualified_lookup_with_layer_hint(self, tmp_path):
        """A layer hint does not interfere with sink-qualified resolution."""
        self._write_dim_provider_pair(tmp_path)

        reference = load_contract_for_asset(
            "reference_gold/dim_provider", layer="gold", search_paths=[tmp_path]
        )

        assert reference is not None and reference.description == "reference"

    def test_bare_name_lookup_with_same_layer_duplicates_raises(self, tmp_path):
        """A bare-name lookup that cannot disambiguate raises instead of guessing."""
        self._write_dim_provider_pair(tmp_path)

        with pytest.raises(
            ContractValidationError,
            match=r"Multiple contracts found for asset 'dim_provider'.*sink-qualified",
        ):
            load_contract_for_asset("dim_provider", layer="gold", search_paths=[tmp_path])

        with pytest.raises(
            ContractValidationError,
            match=r"Multiple contracts found for asset 'dim_provider'",
        ):
            load_contract_for_asset("dim_provider", search_paths=[tmp_path])

    def test_extra_leading_key_components_resolve_via_last_two(self, tmp_path):
        """Asset keys with extra prefix components match on their last two."""
        self._write_dim_provider_pair(tmp_path)

        reference = load_contract_for_asset(
            "pg_monc/reference_gold/dim_provider", search_paths=[tmp_path]
        )

        assert reference is not None and reference.description == "reference"

    def test_overlapping_sinks_same_layer_raises_at_build(self, tmp_path):
        """Same asset+layer with the SAME sink cannot be disambiguated -- build error."""
        (tmp_path / "a.contract.yaml").write_text(
            _SINKED_CONTRACT_YAML.format(
                asset="dim_provider",
                layer="gold",
                description="a",
                sink_schema="reference_gold",
                sink_table="dim_provider",
            )
        )
        (tmp_path / "b.contract.yaml").write_text(
            _SINKED_CONTRACT_YAML.format(
                asset="dim_provider",
                layer="gold",
                description="b",
                sink_schema="reference_gold",
                sink_table="dim_provider",
            )
        )

        with pytest.raises(
            ContractValidationError,
            match=r"Duplicate contract for asset 'dim_provider'.*cannot be disambiguated",
        ):
            _build_contract_index([tmp_path])

    def test_sinkless_duplicate_same_layer_raises_at_build(self, tmp_path):
        """Same asset+layer where one contract has no sink identity -- build error."""
        (tmp_path / "a.contract.yaml").write_text(
            _SINKED_CONTRACT_YAML.format(
                asset="dim_provider",
                layer="gold",
                description="a",
                sink_schema="reference_gold",
                sink_table="dim_provider",
            )
        )
        (tmp_path / "b.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="dim_provider", layer="gold")
        )

        with pytest.raises(
            ContractValidationError,
            match=r"Duplicate contract for asset 'dim_provider'.*cannot be disambiguated",
        ):
            _build_contract_index([tmp_path])

    def test_different_layers_still_disambiguate_without_sinks(self, tmp_path):
        """Pre-#405 layer disambiguation continues to work unchanged."""
        (tmp_path / "a.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="fda_ndc_directory", layer="bronze")
        )
        (tmp_path / "b.contract.yaml").write_text(
            _LAYERED_CONTRACT_YAML.format(asset="fda_ndc_directory", layer="silver")
        )

        bronze = load_contract_for_asset(
            "fda_ndc_directory", layer="bronze", search_paths=[tmp_path]
        )

        assert bronze is not None and bronze.layer == "bronze"


class TestLoaderNewSections:
    """Tests for parsing sources, sinks, and testing sections."""

    def test_parse_sources(self, tmp_path):
        """Test loading contract with sources section."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: silver
schema:
  columns:
    - name: id
      type: integer
      nullable: false
sources:
  - type: table
    schema: bronze
    table: raw_data
    description: Raw data source
sinks:
  - type: table
    schema: silver
    table: clean_data
""")
        contract = load_contract(contract_yaml)
        assert len(contract.sources) == 1
        assert contract.sources[0]["type"] == "table"
        assert contract.sources[0]["schema"] == "bronze"
        assert contract.sources[0]["table"] == "raw_data"

    def test_parse_sinks(self, tmp_path):
        """Test loading contract with sinks section."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: silver
schema:
  columns:
    - name: id
      type: integer
      nullable: false
sinks:
  - type: table
    database: analytics
    schema: synthetic_silver
    table: fda_ndc_package
    mode: full_refresh
""")
        contract = load_contract(contract_yaml)
        assert len(contract.sinks) == 1
        assert contract.sinks[0]["schema"] == "synthetic_silver"
        assert contract.sinks[0]["mode"] == "full_refresh"

    def test_parse_testing_config(self, tmp_path):
        """Test loading contract with testing section."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: silver
schema:
  columns:
    - name: id
      type: integer
      nullable: false
testing:
  enabled: true
  source_row_limit: 500
  source_where_clause: "created_date >= CURRENT_DATE - INTERVAL '7 days'"
  expected_min_rows: 1
  expected_max_rows: 1000
  timeout_seconds: 120
""")
        contract = load_contract(contract_yaml)
        assert contract.testing is not None
        assert contract.testing.enabled is True
        assert contract.testing.source_row_limit == 500
        assert contract.testing.source_where_clause is not None
        assert "CURRENT_DATE" in contract.testing.source_where_clause
        assert contract.testing.expected_min_rows == 1
        assert contract.testing.expected_max_rows == 1000
        assert contract.testing.timeout_seconds == 120

    def test_testing_config_defaults(self, tmp_path):
        """Test testing section with minimal fields uses defaults."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: silver
schema:
  columns:
    - name: id
      type: integer
      nullable: false
testing:
  enabled: false
""")
        contract = load_contract(contract_yaml)
        assert contract.testing is not None
        assert contract.testing.enabled is False
        assert contract.testing.source_row_limit == 1000  # Default
        assert contract.testing.timeout_seconds == 300  # Default

    def test_no_testing_section(self, tmp_path):
        """Test contract without testing section has None."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: silver
schema:
  columns:
    - name: id
      type: integer
      nullable: false
""")
        contract = load_contract(contract_yaml)
        assert contract.testing is None
        assert contract.sources == []
        assert contract.sinks == []

    def test_parse_lineage_config(self, tmp_path):
        """Test loading contract with lineage section."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: bronze
schema:
  columns:
    - name: id
      type: integer
      nullable: false
lineage:
  enabled: true
  source_system: openfda
  transformation_type: ingest
""")
        contract = load_contract(contract_yaml)
        assert contract.lineage is not None
        assert contract.lineage.enabled is True
        assert contract.lineage.source_system == "openfda"
        assert contract.lineage.transformation_type == "ingest"

    def test_lineage_config_defaults(self, tmp_path):
        """Test lineage section with minimal fields uses defaults."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: bronze
schema:
  columns:
    - name: id
      type: integer
      nullable: false
lineage:
  enabled: false
""")
        contract = load_contract(contract_yaml)
        assert contract.lineage is not None
        assert contract.lineage.enabled is False
        assert contract.lineage.source_system is None
        assert contract.lineage.transformation_type is None

    def test_no_lineage_section(self, tmp_path):
        """Test contract without lineage section has None."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: bronze
schema:
  columns:
    - name: id
      type: integer
      nullable: false
""")
        contract = load_contract(contract_yaml)
        assert contract.lineage is None

    def test_multiple_sources(self, tmp_path):
        """Test loading contract with multiple sources."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: test_asset
layer: silver
schema:
  columns:
    - name: id
      type: integer
      nullable: false
sources:
  - type: table
    schema: bronze
    table: source_a
  - type: table
    database: external_db
    schema: raw
    table: source_b
  - type: external
    system: api
    description: REST API source
""")
        contract = load_contract(contract_yaml)
        assert len(contract.sources) == 3

        # Verify get_source_tables only returns table types
        table_refs = contract.get_source_tables()
        assert len(table_refs) == 2
        assert table_refs[0].schema == "bronze"
        assert table_refs[1].database == "external_db"


class TestUnknownFieldValidation:
    """Tests for unknown-field detection and did-you-mean suggestions."""

    def _base(self) -> dict:  # type: ignore[type-arg]
        """Minimal valid contract dict for mutation in tests."""
        return {
            "version": "1.0",
            "pipeline_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "asset": "test",
            "layer": "bronze",
            "schema": {"columns": [{"name": "id", "type": "integer", "nullable": False}]},
        }

    # --- top-level ---

    def test_unknown_top_level_field_raises(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.yaml"
        contract_yaml.write_text(
            "version: '1.0'\npipeline_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'\nasset: t\nlayer: bronze\nschema:\n  columns:\n"
            "    - name: id\n      type: integer\n      nullable: false\n"
            "typo_owner: team-x\n"
        )
        with pytest.raises(ContractValidationError, match="unknown field 'typo_owner'"):
            load_contract(contract_yaml)

    def test_unknown_top_level_suggests_did_you_mean(self) -> None:
        data = self._base()
        data["owmer"] = {"team": "x"}  # typo for "owner"
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "owmer" in e]
        assert unknown, "Expected an error for unknown field 'owmer'"
        assert "Did you mean 'owner'?" in unknown[0]

    def test_unknown_top_level_no_suggestion_for_garbage(self) -> None:
        data = self._base()
        data["zzzzz_not_a_real_key"] = "whatever"
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "zzzzz_not_a_real_key" in e]
        assert unknown
        assert "Did you mean" not in unknown[0]

    def test_known_top_level_fields_are_allowed(self) -> None:
        """Verify every documented top-level field passes without unknown-field error."""
        data = self._base()
        data["description"] = "desc"
        data["owner"] = {"team": "eng"}
        data["expectations"] = []
        data["upstream"] = []
        data["sla"] = {"freshness_hours": 24}
        data["sources"] = []
        data["sinks"] = []
        data["testing"] = {"enabled": True}
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "unknown field" in e]
        assert not unknown, f"Unexpected unknown-field errors: {unknown}"

    # --- schema level ---

    def test_unknown_schema_field_raises(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.yaml"
        contract_yaml.write_text(
            "version: '1.0'\npipeline_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'\nasset: t\nlayer: bronze\nschema:\n"
            "  columns:\n    - name: id\n      type: integer\n      nullable: false\n"
            "  extra_schema_key: oops\n"
        )
        with pytest.raises(ContractValidationError, match="unknown field 'extra_schema_key'"):
            load_contract(contract_yaml)

    # --- column level ---

    def test_unknown_column_field_raises(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.yaml"
        contract_yaml.write_text(
            "version: '1.0'\npipeline_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'\nasset: t\nlayer: bronze\nschema:\n  columns:\n"
            "    - name: id\n      type: integer\n      nullable: false\n"
            "      uniq_constraint: true\n"
        )
        with pytest.raises(ContractValidationError, match="unknown field 'uniq_constraint'"):
            load_contract(contract_yaml)

    def test_unknown_column_field_suggests_did_you_mean(self) -> None:
        data = self._base()
        data["schema"]["columns"][0]["primaary_key"] = True  # typo
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "primaary_key" in e]
        assert unknown
        assert "Did you mean 'primary_key'?" in unknown[0]

    # --- owner level ---

    def test_unknown_owner_field_raises(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.yaml"
        contract_yaml.write_text(
            "version: '1.0'\npipeline_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'\nasset: t\nlayer: bronze\nschema:\n  columns:\n"
            "    - name: id\n      type: integer\n      nullable: false\n"
            "owner:\n  team: eng\n  slakk_channel: '#foo'\n"
        )
        with pytest.raises(ContractValidationError, match="unknown field 'slakk_channel'"):
            load_contract(contract_yaml)

    def test_unknown_owner_field_suggests_did_you_mean(self) -> None:
        data = self._base()
        data["owner"] = {"team": "eng", "slakk_channel": "#foo"}
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "slakk_channel" in e]
        assert unknown
        assert "Did you mean 'slack_channel'?" in unknown[0]

    # --- upstream level ---

    def test_unknown_upstream_field_raises(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.yaml"
        contract_yaml.write_text(
            "version: '1.0'\npipeline_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'\nasset: t\nlayer: bronze\nschema:\n  columns:\n"
            "    - name: id\n      type: integer\n      nullable: false\n"
            "upstream:\n  - name: src\n    type: external\n    systm: sftp\n"
        )
        with pytest.raises(ContractValidationError, match="unknown field 'systm'"):
            load_contract(contract_yaml)

    def test_unknown_upstream_field_suggests_did_you_mean(self) -> None:
        data = self._base()
        data["upstream"] = [{"name": "src", "type": "external", "systm": "sftp"}]
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "systm" in e]
        assert unknown
        assert "Did you mean 'system'?" in unknown[0]

    # --- sla level ---

    def test_unknown_sla_field_raises(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.yaml"
        contract_yaml.write_text(
            "version: '1.0'\npipeline_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'\nasset: t\nlayer: bronze\nschema:\n  columns:\n"
            "    - name: id\n      type: integer\n      nullable: false\n"
            "sla:\n  freshnesss_hours: 24\n"
        )
        with pytest.raises(ContractValidationError, match="unknown field 'freshnesss_hours'"):
            load_contract(contract_yaml)

    def test_unknown_sla_field_suggests_did_you_mean(self) -> None:
        data = self._base()
        data["sla"] = {"freshnesss_hours": 24}
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "freshnesss_hours" in e]
        assert unknown
        assert "Did you mean 'freshness_hours'?" in unknown[0]

    # --- testing level ---

    def test_unknown_testing_field_raises(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.yaml"
        contract_yaml.write_text(
            "version: '1.0'\npipeline_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'\nasset: t\nlayer: bronze\nschema:\n  columns:\n"
            "    - name: id\n      type: integer\n      nullable: false\n"
            "testing:\n  enabbled: true\n"
        )
        with pytest.raises(ContractValidationError, match="unknown field 'enabbled'"):
            load_contract(contract_yaml)

    def test_unknown_testing_field_suggests_did_you_mean(self) -> None:
        data = self._base()
        data["testing"] = {"enabbled": True}
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "enabbled" in e]
        assert unknown
        assert "Did you mean 'enabled'?" in unknown[0]

    # --- lineage level ---

    def test_unknown_lineage_field_raises(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.yaml"
        contract_yaml.write_text(
            "version: '1.0'\npipeline_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'\nasset: t\nlayer: bronze\nschema:\n  columns:\n"
            "    - name: id\n      type: integer\n      nullable: false\n"
            "lineage:\n  source_systm: openfda\n"
        )
        with pytest.raises(ContractValidationError, match="unknown field 'source_systm'"):
            load_contract(contract_yaml)

    def test_unknown_lineage_field_suggests_did_you_mean(self) -> None:
        data = self._base()
        data["lineage"] = {"source_systm": "openfda"}
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "source_systm" in e]
        assert unknown
        assert "Did you mean 'source_system'?" in unknown[0]

    # --- test types ---

    def test_unknown_column_test_type_raises(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.yaml"
        contract_yaml.write_text(
            "version: '1.0'\npipeline_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'\nasset: t\nlayer: bronze\nschema:\n  columns:\n"
            "    - name: id\n      type: integer\n      nullable: false\n"
            "      tests:\n        - greter_than: 0\n"
        )
        with pytest.raises(ContractValidationError, match="unknown test type 'greter_than'"):
            load_contract(contract_yaml)

    def test_unknown_column_test_type_suggests_did_you_mean(self) -> None:
        data = self._base()
        data["schema"]["columns"][0]["tests"] = [{"greter_than": 0}]
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "greter_than" in e]
        assert unknown
        assert "Did you mean 'greater_than'?" in unknown[0]

    # --- expectation types ---

    def test_unknown_expectation_type_raises(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.yaml"
        contract_yaml.write_text(
            "version: '1.0'\npipeline_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'\nasset: t\nlayer: bronze\nschema:\n  columns:\n"
            "    - name: id\n      type: integer\n      nullable: false\n"
            "expectations:\n  - row_cout: {min: 1}\n"
        )
        with pytest.raises(ContractValidationError, match="unknown expectation type 'row_cout'"):
            load_contract(contract_yaml)

    def test_unknown_expectation_type_suggests_did_you_mean(self) -> None:
        data = self._base()
        data["expectations"] = [{"row_cout": {"min": 1}}]
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "row_cout" in e]
        assert unknown
        assert "Did you mean 'row_count'?" in unknown[0]

    # --- known field constants are exportable ---

    def test_known_field_sets_are_complete(self) -> None:
        """Smoke test that the exported constant sets contain expected members."""
        assert "version" in KNOWN_TOP_LEVEL_FIELDS
        assert "testing" in KNOWN_TOP_LEVEL_FIELDS
        assert "columns" in KNOWN_SCHEMA_FIELDS
        assert "primary_key" in KNOWN_COLUMN_FIELDS
        assert "slack_channel" in KNOWN_OWNER_FIELDS
        assert "system" in KNOWN_UPSTREAM_FIELDS
        assert "freshness_hours" in KNOWN_SLA_FIELDS
        assert "source_row_limit" in KNOWN_TESTING_FIELDS
        assert "accepted_values" in KNOWN_COLUMN_TEST_TYPES
        assert "row_count" in KNOWN_EXPECTATION_TYPES

    # --- cross-level path hints ---

    def test_path_hint_for_field_at_wrong_level_single_path(self) -> None:
        """A field unique to one section suggests 'did you mean section.field?'"""
        data = self._base()
        data["source_row_limit"] = 500  # belongs under testing:
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "source_row_limit" in e]
        assert unknown
        assert "Did you mean 'testing.source_row_limit'?" in unknown[0]

    def test_path_hint_for_field_at_wrong_level_multi_path(self) -> None:
        """A field valid in multiple sections lists all valid paths."""
        data = self._base()
        # 'type' is valid in schema.columns[*] and upstream[*], not at top level
        data["type"] = "integer"
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "'type'" in e and "unknown field" in e]
        assert unknown
        # should mention both valid paths
        assert "schema.columns[*].type" in unknown[0]
        assert "upstream[*].type" in unknown[0]

    def test_path_hint_owner_field_at_top_level(self) -> None:
        """owner-only field placed at top level gets path hint."""
        data = self._base()
        data["slack_channel"] = "#eng"  # belongs under owner:
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "slack_channel" in e]
        assert unknown
        assert "Did you mean 'owner.slack_channel'?" in unknown[0]

    def test_path_hint_sla_field_at_top_level(self) -> None:
        """sla-only field placed at top level gets path hint."""
        data = self._base()
        data["freshness_hours"] = 24  # belongs under sla:
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "freshness_hours" in e]
        assert unknown
        assert "Did you mean 'sla.freshness_hours'?" in unknown[0]

    def test_same_level_typo_takes_priority_over_path_hint(self) -> None:
        """Same-level fuzzy suggestion wins over cross-level path hint."""
        data = self._base()
        # 'owmer' is a typo for 'owner' (same level); it is NOT a known sub-section field
        data["owmer"] = {"team": "x"}
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "owmer" in e]
        assert unknown
        # same-level suggestion should appear, not a path hint
        assert "Did you mean 'owner'?" in unknown[0]
        assert "This field is valid under" not in unknown[0]

    def test_field_to_paths_lookup_table(self) -> None:
        """_FIELD_TO_PATHS contains expected entries for all sections."""
        assert _FIELD_TO_PATHS["source_row_limit"] == ["testing.source_row_limit"]
        assert _FIELD_TO_PATHS["freshness_hours"] == ["sla.freshness_hours"]
        assert _FIELD_TO_PATHS["slack_channel"] == ["owner.slack_channel"]
        # 'description' is valid at top-level, column, and upstream
        assert "description" in _FIELD_TO_PATHS["description"]
        assert "schema.columns[*].description" in _FIELD_TO_PATHS["description"]
        assert "upstream[*].description" in _FIELD_TO_PATHS["description"]


class TestExpectationParamValidation:
    """Per-type parameter-key validation for expectations and column tests (#394)."""

    def _base(self) -> dict:  # type: ignore[type-arg]
        """Minimal valid contract dict for mutation in tests."""
        return {
            "version": "1.0",
            "pipeline_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "asset": "test",
            "layer": "bronze",
            "schema": {"columns": [{"name": "id", "type": "integer", "nullable": False}]},
        }

    # --- table expectations ---

    def test_nested_severity_in_expectation_raises(self) -> None:
        """The exact #394 repro: severity nested inside row_count parameters."""
        data = self._base()
        data["expectations"] = [{"row_count": {"min": 100000, "max": 500000, "severity": "warn"}}]
        errors = validate_contract_schema(data)
        nested = [e for e in errors if "'severity' must be a sibling of 'row_count'" in e]
        assert nested
        assert "silently ignored" in nested[0]

    def test_sibling_severity_in_expectation_ok(self) -> None:
        data = self._base()
        data["expectations"] = [{"row_count": {"min": 100000, "max": 500000}, "severity": "warn"}]
        assert validate_contract_schema(data) == []

    def test_sibling_severity_parses_to_warn(self, tmp_path: Path) -> None:
        """End-to-end: sibling severity actually lands on the parsed expectation."""
        contract_yaml = tmp_path / "c.yaml"
        contract_yaml.write_text(
            "version: '1.0'\npipeline_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'\n"
            "asset: t\nlayer: bronze\nschema:\n  columns:\n"
            "    - name: id\n      type: integer\n      nullable: false\n"
            "expectations:\n"
            "  - row_count:\n      min: 1\n      max: 10\n    severity: warn\n"
        )
        contract = load_contract(contract_yaml)
        exp = contract.expectations[0]
        assert exp.severity == Severity.WARN
        assert "severity" not in exp.parameters

    def test_nested_severity_raises_via_load_contract(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.yaml"
        contract_yaml.write_text(
            "version: '1.0'\npipeline_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'\n"
            "asset: t\nlayer: bronze\nschema:\n  columns:\n"
            "    - name: id\n      type: integer\n      nullable: false\n"
            "expectations:\n"
            "  - row_count:\n      min: 1\n      max: 10\n      severity: warn\n"
        )
        with pytest.raises(ContractValidationError, match="must be a sibling"):
            load_contract(contract_yaml)

    def test_unknown_expectation_param_suggests_did_you_mean(self) -> None:
        data = self._base()
        data["expectations"] = [{"row_count": {"minn": 5}}]
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "unknown parameter 'minn'" in e]
        assert unknown
        assert "Did you mean 'min'?" in unknown[0]
        assert "Valid parameters: max, min" in unknown[0]

    def test_nested_when_in_expectation_raises(self) -> None:
        data = self._base()
        data["expectations"] = [{"row_count": {"min": 1, "when": "not_null"}}]
        errors = validate_contract_schema(data)
        assert any("'when' is not supported on table expectations" in e for e in errors)

    def test_expectation_bare_value_raises(self) -> None:
        """A scalar value parses to {'value': ...} which no expectation reads."""
        data = self._base()
        data["expectations"] = [{"row_count": 100}]
        errors = validate_contract_schema(data)
        assert any("takes a mapping of parameters" in e for e in errors)

    def test_bare_expectation_type_ok(self) -> None:
        """history_completeness takes no parameters; a bare key is valid."""
        data = self._base()
        data["expectations"] = [{"history_completeness": None}]
        assert validate_contract_schema(data) == []

    def test_all_valid_expectation_params_accepted(self) -> None:
        data = self._base()
        data["schema"]["columns"].append(
            {"name": "updated_at", "type": "datetime", "nullable": False}
        )
        data["expectations"] = [
            {"row_count": {"min": 1, "max": 10}},
            {"freshness": {"column": "updated_at", "max_age_hours": 48}},
            {"null_percentage": {"column": "id", "max_percent": 10}},
            {"unique_combination": {"columns": ["id"]}},
        ]
        assert validate_contract_schema(data) == []

    # --- column tests ---

    def test_nested_severity_in_column_test_raises(self) -> None:
        data = self._base()
        data["schema"]["columns"][0]["tests"] = [
            {"between": {"min": 1, "max": 10, "severity": "warn"}}
        ]
        errors = validate_contract_schema(data)
        assert any("'severity' must be a sibling of 'between'" in e for e in errors)

    def test_nested_when_in_column_test_raises(self) -> None:
        data = self._base()
        data["schema"]["columns"][0]["tests"] = [
            {"accepted_values": {"values": [1, 2], "when": "not_null"}}
        ]
        errors = validate_contract_schema(data)
        assert any("'when' must be a sibling of 'accepted_values'" in e for e in errors)

    def test_sibling_modifiers_in_column_test_ok(self) -> None:
        data = self._base()
        data["schema"]["columns"][0]["tests"] = [
            {"between": {"min": 1, "max": 10}, "severity": "warn", "when": "not_null"}
        ]
        assert validate_contract_schema(data) == []

    def test_scalar_shorthand_column_test_ok(self) -> None:
        data = self._base()
        data["schema"]["columns"][0]["tests"] = [
            {"greater_than": 0, "severity": "warn"},
            {"within_days": 365},
        ]
        assert validate_contract_schema(data) == []

    def test_unknown_column_test_param_suggests_did_you_mean(self) -> None:
        data = self._base()
        data["schema"]["columns"][0]["tests"] = [{"accepted_values": {"vals": [1, 2]}}]
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "unknown parameter 'vals'" in e]
        assert unknown
        assert "Did you mean 'values'?" in unknown[0]

    def test_param_registries_cover_all_known_types(self) -> None:
        """Every known type must have a parameter registry entry."""
        assert set(KNOWN_EXPECTATION_PARAMS) == set(KNOWN_EXPECTATION_TYPES)
        assert set(KNOWN_COLUMN_TEST_PARAMS) == set(KNOWN_COLUMN_TEST_TYPES)


class TestSourceSinkValidation:
    """Tests for per-type validation of sources and sinks entries."""

    def _base(self) -> dict[str, object]:
        return {
            "version": "1.0",
            "pipeline_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "asset": "test_asset",
            "layer": "bronze",
        }

    # --- type field ---

    def test_missing_type_in_source(self) -> None:
        data = self._base()
        data["sources"] = [{"schema": "public", "table": "users"}]
        errors = validate_contract_schema(data)
        assert any("Source 0" in e and "'type' is required" in e for e in errors)

    def test_missing_type_in_sink(self) -> None:
        data = self._base()
        data["sinks"] = [{"schema": "public", "table": "output"}]
        errors = validate_contract_schema(data)
        assert any("Sink 0" in e and "'type' is required" in e for e in errors)

    def test_unknown_type_with_suggestion(self) -> None:
        data = self._base()
        data["sources"] = [{"type": "tablr", "schema": "public", "table": "users"}]
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "Source 0" in e and "unknown type" in e]
        assert unknown
        assert "Did you mean 'table'?" in unknown[0]

    def test_unknown_type_no_suggestion(self) -> None:
        data = self._base()
        data["sources"] = [{"type": "kafka"}]
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "Source 0" in e and "unknown type" in e]
        assert unknown
        assert "Did you mean" not in unknown[0]

    # --- table type: required fields ---

    def test_table_source_missing_schema(self) -> None:
        data = self._base()
        data["sources"] = [{"type": "table", "table": "users"}]
        errors = validate_contract_schema(data)
        assert any("Source 0" in e and "'schema' is required" in e for e in errors)

    def test_table_source_missing_table(self) -> None:
        data = self._base()
        data["sources"] = [{"type": "table", "schema": "public"}]
        errors = validate_contract_schema(data)
        assert any("Source 0" in e and "'table' is required" in e for e in errors)

    def test_table_sink_missing_schema(self) -> None:
        data = self._base()
        data["sinks"] = [{"type": "table", "table": "output"}]
        errors = validate_contract_schema(data)
        assert any("Sink 0" in e and "'schema' is required" in e for e in errors)

    # --- table type: unknown fields ---

    def test_mode_rejected_in_table_source(self) -> None:
        """mode is valid for sinks but not sources."""
        data = self._base()
        data["sources"] = [
            {"type": "table", "schema": "public", "table": "users", "mode": "append"}
        ]
        errors = validate_contract_schema(data)
        assert any("Source 0" in e and "unknown field 'mode'" in e for e in errors)

    def test_unknown_field_in_table_source(self) -> None:
        data = self._base()
        data["sources"] = [
            {"type": "table", "schema": "public", "table": "users", "frobulate": "x"}
        ]
        errors = validate_contract_schema(data)
        assert any("Source 0" in e and "unknown field 'frobulate'" in e for e in errors)

    def test_unknown_field_in_table_sink(self) -> None:
        data = self._base()
        data["sinks"] = [{"type": "table", "schema": "public", "table": "output", "frobulate": "x"}]
        errors = validate_contract_schema(data)
        assert any("Sink 0" in e and "unknown field 'frobulate'" in e for e in errors)

    # --- table type: valid entries pass ---

    def test_valid_table_source(self) -> None:
        data = self._base()
        data["sources"] = [{"type": "table", "schema": "public", "table": "users"}]
        errors = validate_contract_schema(data)
        source_errors = [e for e in errors if "Source" in e]
        assert not source_errors

    def test_valid_table_source_with_optional_fields(self) -> None:
        data = self._base()
        data["sources"] = [
            {
                "type": "table",
                "schema": "public",
                "table": "users",
                "database": "mydb",
                "description": "User records",
            }
        ]
        errors = validate_contract_schema(data)
        source_errors = [e for e in errors if "Source" in e]
        assert not source_errors

    def test_valid_table_sink_with_mode(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "reporting",
                "table": "summary",
                "mode": "full_refresh",
            }
        ]
        errors = validate_contract_schema(data)
        sink_errors = [e for e in errors if "Sink" in e]
        assert not sink_errors

    def test_mode_valid_in_sink_not_error(self) -> None:
        """mode should not be flagged as unknown in sinks."""
        data = self._base()
        data["sinks"] = [{"type": "table", "schema": "public", "table": "out", "mode": "append"}]
        errors = validate_contract_schema(data)
        assert not any("unknown field 'mode'" in e for e in errors)

    # --- external type: permissive ---

    def test_valid_external_source(self) -> None:
        data = self._base()
        data["sources"] = [{"type": "external", "system": "salesforce"}]
        errors = validate_contract_schema(data)
        source_errors = [e for e in errors if "Source" in e]
        assert not source_errors

    def test_external_source_arbitrary_fields_pass(self) -> None:
        """External entries are permissive — arbitrary extra fields are allowed."""
        data = self._base()
        data["sources"] = [
            {
                "type": "external",
                "system": "custom_api",
                "endpoint": "/v2/records",
                "auth": "bearer",
            }
        ]
        errors = validate_contract_schema(data)
        source_errors = [e for e in errors if "Source" in e]
        assert not source_errors

    def test_external_sink_arbitrary_fields_pass(self) -> None:
        data = self._base()
        data["sinks"] = [{"type": "external", "system": "s3", "bucket": "my-bucket"}]
        errors = validate_contract_schema(data)
        sink_errors = [e for e in errors if "Sink" in e]
        assert not sink_errors

    def test_external_source_with_only_type(self) -> None:
        """External with only type field is valid."""
        data = self._base()
        data["sources"] = [{"type": "external"}]
        errors = validate_contract_schema(data)
        source_errors = [e for e in errors if "Source" in e]
        assert not source_errors

    # --- non-dict entries ---

    def test_source_entry_not_a_dict(self) -> None:
        data = self._base()
        data["sources"] = ["not_a_dict"]
        errors = validate_contract_schema(data)
        assert any("Source 0" in e and "must be an object" in e for e in errors)

    def test_sink_entry_not_a_dict(self) -> None:
        data = self._base()
        data["sinks"] = [42]
        errors = validate_contract_schema(data)
        assert any("Sink 0" in e and "must be an object" in e for e in errors)

    # --- sources/sinks not a list ---

    def test_sources_not_a_list(self) -> None:
        data = self._base()
        data["sources"] = {"type": "table"}
        errors = validate_contract_schema(data)
        assert any("'sources' must be a list" in e for e in errors)

    def test_sinks_not_a_list(self) -> None:
        data = self._base()
        data["sinks"] = "table"
        errors = validate_contract_schema(data)
        assert any("'sinks' must be a list" in e for e in errors)

    # --- multiple entries ---

    def test_errors_reported_per_entry_index(self) -> None:
        """Errors for each entry use the correct 0-based index."""
        data = self._base()
        data["sources"] = [
            {"type": "table", "schema": "public", "table": "users"},
            {"type": "tablr"},  # typo
        ]
        errors = validate_contract_schema(data)
        assert not any("Source 0" in e for e in errors)
        assert any("Source 1" in e and "unknown type" in e for e in errors)

    # --- sink mode validation ---

    def test_sink_mode_invalid_value(self) -> None:
        data = self._base()
        data["sinks"] = [{"type": "table", "schema": "public", "table": "out", "mode": "foobar"}]
        errors = validate_contract_schema(data)
        assert any("Sink 0" in e and "unknown mode 'foobar'" in e for e in errors)

    def test_sink_mode_valid_values(self) -> None:
        for mode in ("full_refresh", "upsert", "append", "scd2"):
            data = self._base()
            data["sinks"] = [{"type": "table", "schema": "public", "table": "out", "mode": mode}]
            errors = validate_contract_schema(data)
            sink_errors = [e for e in errors if "Sink" in e]
            assert not sink_errors, f"mode '{mode}' should be valid but got: {sink_errors}"

    # --- business_key type validation ---

    def test_sink_business_key_string_valid(self) -> None:
        data = self._base()
        data["schema"] = {"columns": [{"name": "id", "type": "text"}]}
        data["sinks"] = [
            {"type": "table", "schema": "public", "table": "out", "business_key": "id"}
        ]
        errors = validate_contract_schema(data)
        sink_errors = [e for e in errors if "Sink" in e]
        assert not sink_errors

    def test_sink_business_key_list_valid(self) -> None:
        data = self._base()
        data["schema"] = {
            "columns": [
                {"name": "id", "type": "text"},
                {"name": "name", "type": "text"},
            ]
        }
        data["sinks"] = [
            {
                "type": "table",
                "schema": "public",
                "table": "out",
                "business_key": ["id", "name"],
            }
        ]
        errors = validate_contract_schema(data)
        sink_errors = [e for e in errors if "Sink" in e]
        assert not sink_errors

    def test_sink_business_key_invalid_type(self) -> None:
        data = self._base()
        data["sinks"] = [{"type": "table", "schema": "public", "table": "out", "business_key": 123}]
        errors = validate_contract_schema(data)
        assert any("Sink 0" in e and "'business_key' must be a string or list" in e for e in errors)

    def test_sink_business_key_list_invalid_item(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "public",
                "table": "out",
                "business_key": ["id", 123],
            }
        ]
        errors = validate_contract_schema(data)
        assert any("Sink 0" in e and "'business_key[1]' must be a string" in e for e in errors)

    # --- tracked_columns type validation ---

    def test_sink_tracked_columns_string_valid(self) -> None:
        data = self._base()
        data["schema"] = {"columns": [{"name": "status", "type": "text"}]}
        data["sinks"] = [
            {"type": "table", "schema": "public", "table": "out", "tracked_columns": "status"}
        ]
        errors = validate_contract_schema(data)
        sink_errors = [e for e in errors if "Sink" in e]
        assert not sink_errors

    def test_sink_tracked_columns_list_valid(self) -> None:
        data = self._base()
        data["schema"] = {
            "columns": [
                {"name": "status", "type": "text"},
                {"name": "amount", "type": "numeric"},
            ]
        }
        data["sinks"] = [
            {
                "type": "table",
                "schema": "public",
                "table": "out",
                "tracked_columns": ["status", "amount"],
            }
        ]
        errors = validate_contract_schema(data)
        sink_errors = [e for e in errors if "Sink" in e]
        assert not sink_errors

    def test_sink_tracked_columns_invalid_type(self) -> None:
        data = self._base()
        data["sinks"] = [
            {"type": "table", "schema": "public", "table": "out", "tracked_columns": 123}
        ]
        errors = validate_contract_schema(data)
        assert any(
            "Sink 0" in e and "'tracked_columns' must be a string or list" in e for e in errors
        )

    # --- detect_deletes type validation ---

    def test_sink_detect_deletes_bool_valid(self) -> None:
        data = self._base()
        data["sinks"] = [
            {"type": "table", "schema": "public", "table": "out", "detect_deletes": True}
        ]
        errors = validate_contract_schema(data)
        sink_errors = [e for e in errors if "Sink" in e]
        assert not sink_errors

    def test_sink_detect_deletes_invalid_type(self) -> None:
        data = self._base()
        data["sinks"] = [
            {"type": "table", "schema": "public", "table": "out", "detect_deletes": "yes"}
        ]
        errors = validate_contract_schema(data)
        assert any("Sink 0" in e and "'detect_deletes' must be a boolean" in e for e in errors)

    # --- full_refresh_method validation (#4) ---

    def test_sink_full_refresh_method_valid(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "public",
                "table": "out",
                "mode": "full_refresh",
                "full_refresh_method": "delete",
            }
        ]
        errors = validate_contract_schema(data)
        assert not [e for e in errors if "Sink" in e]

    def test_sink_full_refresh_method_unknown_value_rejected(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "public",
                "table": "out",
                "mode": "full_refresh",
                "full_refresh_method": "vaccum",
            }
        ]
        errors = validate_contract_schema(data)
        assert any("Sink 0" in e and "'full_refresh_method' must be one of" in e for e in errors)

    def test_sink_full_refresh_method_invalid_type_rejected(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "public",
                "table": "out",
                "mode": "full_refresh",
                "full_refresh_method": True,
            }
        ]
        errors = validate_contract_schema(data)
        assert any("Sink 0" in e and "'full_refresh_method' must be a string" in e for e in errors)

    def test_sink_full_refresh_method_on_non_full_refresh_mode_rejected(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "public",
                "table": "out",
                "mode": "upsert",
                "primary_key": ["id"],
                "full_refresh_method": "delete",
            }
        ]
        errors = validate_contract_schema(data)
        assert any(
            "Sink 0" in e and "'full_refresh_method' is only valid with mode 'full_refresh'" in e
            for e in errors
        )

    # --- cross-validation: sink column references against schema ---

    def test_sink_business_key_references_valid_column(self) -> None:
        data = self._base()
        data["schema"] = {
            "columns": [
                {"name": "id", "type": "text"},
                {"name": "name", "type": "text"},
            ]
        }
        data["sinks"] = [
            {
                "type": "table",
                "schema": "public",
                "table": "out",
                "business_key": ["id", "name"],
            }
        ]
        errors = validate_contract_schema(data)
        ref_errors = [e for e in errors if "references column" in e]
        assert not ref_errors

    def test_sink_business_key_references_unknown_column(self) -> None:
        data = self._base()
        data["schema"] = {"columns": [{"name": "id", "type": "text"}]}
        data["sinks"] = [
            {
                "type": "table",
                "schema": "public",
                "table": "out",
                "business_key": "no_such_col",
            }
        ]
        errors = validate_contract_schema(data)
        assert any(
            "Sink 0" in e and "'business_key' references column 'no_such_col'" in e for e in errors
        )

    def test_sink_tracked_columns_references_unknown_column(self) -> None:
        data = self._base()
        data["schema"] = {"columns": [{"name": "status", "type": "text"}]}
        data["sinks"] = [
            {
                "type": "table",
                "schema": "public",
                "table": "out",
                "tracked_columns": ["status", "missing_col"],
            }
        ]
        errors = validate_contract_schema(data)
        assert any(
            "Sink 0" in e and "'tracked_columns' references column 'missing_col'" in e
            for e in errors
        )

    def test_sink_tracked_columns_mixed_valid_invalid(self) -> None:
        data = self._base()
        data["schema"] = {
            "columns": [
                {"name": "status", "type": "text"},
                {"name": "amount", "type": "numeric"},
            ]
        }
        data["sinks"] = [
            {
                "type": "table",
                "schema": "public",
                "table": "out",
                "tracked_columns": ["status", "bad_col", "amount"],
            }
        ]
        errors = validate_contract_schema(data)
        ref_errors = [e for e in errors if "references column" in e]
        assert len(ref_errors) == 1
        assert "'bad_col'" in ref_errors[0]


class TestPiiFieldValidation:
    """Tests for PII field in column definitions."""

    def _base(self) -> dict:  # type: ignore[type-arg]
        return {
            "version": "1.0",
            "pipeline_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "asset": "test",
            "layer": "bronze",
            "schema": {"columns": [{"name": "id", "type": "integer", "nullable": False}]},
        }

    def test_pii_field_accepted(self, tmp_path: Path) -> None:
        """Test that pii: false is accepted in column YAML."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: pii_test
layer: bronze
schema:
  columns:
    - name: event_type
      type: string
      nullable: false
      pii: false
    - name: patient_id
      type: string
      nullable: false
      pii: true
""")
        contract = load_contract(contract_yaml)
        event_type = contract.get_column("event_type")
        patient_id = contract.get_column("patient_id")
        assert event_type is not None
        assert event_type.pii is False
        assert patient_id is not None
        assert patient_id.pii is True

    def test_pii_field_invalid_type(self) -> None:
        """Test that pii: 'yes' produces a validation error."""
        data = self._base()
        data["schema"]["columns"][0]["pii"] = "yes"
        errors = validate_contract_schema(data)
        pii_errors = [e for e in errors if "'pii' must be a boolean" in e]
        assert pii_errors

    def test_pii_field_default_true(self, tmp_path: Path) -> None:
        """Test that omitting pii defaults to True (safe by default)."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: default_test
layer: bronze
schema:
  columns:
    - name: id
      type: integer
      nullable: false
""")
        contract = load_contract(contract_yaml)
        col = contract.get_column("id")
        assert col is not None
        assert col.pii is True

    def test_pii_field_not_flagged_as_unknown(self) -> None:
        """Test that 'pii' is in known column fields and does not error."""
        data = self._base()
        data["schema"]["columns"][0]["pii"] = False
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "unknown field 'pii'" in e]
        assert not unknown

    def test_phi_field_accepted(self, tmp_path: Path) -> None:
        """Test that explicit phi annotations are parsed (#391)."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: phi_test
layer: bronze
schema:
  columns:
    - name: provider_npi
      type: string
      nullable: false
      pii: true
      phi: false
    - name: patient_id
      type: string
      nullable: false
      pii: true
      phi: true
""")
        contract = load_contract(contract_yaml)
        provider = contract.get_column("provider_npi")
        patient = contract.get_column("patient_id")
        assert provider is not None
        assert provider.pii is True
        assert provider.phi is False
        assert patient is not None
        assert patient.phi is True

    def test_phi_field_defaults_to_pii_value(self, tmp_path: Path) -> None:
        """Test that omitting phi mirrors the pii value in both directions (#391)."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: phi_default_test
layer: bronze
schema:
  columns:
    - name: patient_id
      type: string
      nullable: false
      pii: true
    - name: event_type
      type: string
      nullable: false
      pii: false
    - name: unannotated
      type: string
      nullable: false
""")
        contract = load_contract(contract_yaml)
        patient = contract.get_column("patient_id")
        event = contract.get_column("event_type")
        unannotated = contract.get_column("unannotated")
        assert patient is not None
        assert patient.phi is True
        assert event is not None
        assert event.phi is False
        assert unannotated is not None
        assert unannotated.phi is True  # pii defaults true, phi mirrors it

    def test_phi_field_invalid_type(self) -> None:
        """Test that phi: 'yes' produces a validation error."""
        data = self._base()
        data["schema"]["columns"][0]["phi"] = "yes"
        errors = validate_contract_schema(data)
        phi_errors = [e for e in errors if "'phi' must be a boolean" in e]
        assert phi_errors

    def test_phi_field_not_flagged_as_unknown(self) -> None:
        """Test that 'phi' is in known column fields and does not error."""
        data = self._base()
        data["schema"]["columns"][0]["phi"] = False
        errors = validate_contract_schema(data)
        unknown = [e for e in errors if "unknown field 'phi'" in e]
        assert not unknown

    def test_unannotated_pii_columns_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that loading a contract with unannotated pii columns emits a warning."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: warn_test
layer: bronze
schema:
  columns:
    - name: patient_id
      type: string
      nullable: false
    - name: event_type
      type: string
      nullable: false
      pii: false
""")
        with caplog.at_level(logging.WARNING, logger="moncpipelib.contracts.loader"):
            contract = load_contract(contract_yaml)

        assert contract.asset == "warn_test"
        # patient_id has no explicit pii, should be warned about
        assert any("patient_id" in record.message for record in caplog.records)
        # event_type has explicit pii: false, should NOT be warned about
        assert not any("event_type" in record.message for record in caplog.records)

    def test_no_warning_when_all_columns_annotated(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test no warning when all columns explicitly set pii."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: clean_test
layer: bronze
schema:
  columns:
    - name: patient_id
      type: string
      nullable: false
      pii: true
    - name: event_type
      type: string
      nullable: false
      pii: false
""")
        with caplog.at_level(logging.WARNING, logger="moncpipelib.contracts.loader"):
            load_contract(contract_yaml)

        pii_warnings = [r for r in caplog.records if "no explicit 'pii' annotation" in r.message]
        assert not pii_warnings

    def test_managed_columns_excluded_from_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test managed columns are excluded from pii annotation warning."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text("""
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: managed_test
layer: bronze
schema:
  columns:
    - name: event_type
      type: string
      nullable: false
      pii: false
    - name: _moncpipelib_lineage_id
      type: uuid
      nullable: false
      managed: true
""")
        with caplog.at_level(logging.WARNING, logger="moncpipelib.contracts.loader"):
            load_contract(contract_yaml)

        pii_warnings = [r for r in caplog.records if "no explicit 'pii' annotation" in r.message]
        assert not pii_warnings


class TestTagsField:
    """Tests for the optional tags field in contract YAML."""

    def _base(self) -> dict:  # type: ignore[type-arg]
        return {
            "version": "1.0",
            "pipeline_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "asset": "test",
            "layer": "bronze",
            "schema": {"columns": [{"name": "id", "type": "integer", "nullable": False}]},
        }

    def test_valid_contract_with_tags(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.contract.yaml"
        contract_yaml.write_text(
            'version: "1.0"\n'
            'pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"\n'
            "asset: tagged\n"
            "layer: bronze\n"
            "schema:\n"
            "  columns:\n"
            "    - name: id\n"
            "      type: integer\n"
            "      nullable: false\n"
            "tags:\n"
            "  team/priority: high\n"
            '  oncall/pager: "true"\n'
        )
        contract = load_contract(contract_yaml)
        assert contract.tags == {"team/priority": "high", "oncall/pager": "true"}

    def test_tags_non_string_value_rejected(self) -> None:
        data = self._base()
        data["tags"] = {"key": 123}
        errors = validate_contract_schema(data)
        tag_errors = [e for e in errors if "tags" in e]
        assert tag_errors
        assert any("string" in e for e in tag_errors)

    def test_tags_non_dict_rejected(self) -> None:
        data = self._base()
        data["tags"] = ["a", "b"]
        errors = validate_contract_schema(data)
        tag_errors = [e for e in errors if "tags" in e]
        assert tag_errors

    def test_tags_default_empty_dict(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.contract.yaml"
        contract_yaml.write_text(
            'version: "1.0"\n'
            'pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"\n'
            "asset: no_tags\n"
            "layer: bronze\n"
            "schema:\n"
            "  columns:\n"
            "    - name: id\n"
            "      type: integer\n"
            "      nullable: false\n"
        )
        contract = load_contract(contract_yaml)
        assert contract.tags == {}

    def test_parameters_parsed(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.contract.yaml"
        contract_yaml.write_text(
            'version: "1.0"\n'
            'pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"\n'
            "asset: parameterized\n"
            "layer: silver\n"
            "schema:\n"
            "  columns:\n"
            "    - name: id\n"
            "      type: integer\n"
            "      nullable: false\n"
            "parameters:\n"
            "  days_of_tolerance: 30\n"
            "  include_expired: false\n"
            "  allowed_codes:\n"
            "    - A01\n"
            "    - B02\n"
        )
        contract = load_contract(contract_yaml)
        assert contract.parameters == {
            "days_of_tolerance": 30,
            "include_expired": False,
            "allowed_codes": ["A01", "B02"],
        }

    def test_parameters_default_empty_dict(self, tmp_path: Path) -> None:
        contract_yaml = tmp_path / "c.contract.yaml"
        contract_yaml.write_text(
            'version: "1.0"\n'
            'pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"\n'
            "asset: no_params\n"
            "layer: bronze\n"
            "schema:\n"
            "  columns:\n"
            "    - name: id\n"
            "      type: integer\n"
            "      nullable: false\n"
        )
        contract = load_contract(contract_yaml)
        assert contract.parameters == {}


# ---------------------------------------------------------------------------
# TestGuardrailValidation (#401)
# ---------------------------------------------------------------------------


class TestGuardrailValidation:
    """Load-time guard rails from #401: nullable PKs, upsert/partition coherence,
    detect_deletes scoping, sink primary_key cross-referencing, and the
    sequence_column field the strict validator wrongly rejected."""

    def _base(self, **schema_overrides: object) -> dict[str, object]:
        columns = schema_overrides.pop(
            "columns",
            [
                {"name": "id", "type": "string", "nullable": False, "pii": False},
                {"name": "from_date", "type": "date", "nullable": True, "pii": False},
                {"name": "val", "type": "string", "nullable": True, "pii": False},
            ],
        )
        return {
            "version": "1.0",
            "pipeline_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "asset": "test_asset",
            "layer": "silver",
            "schema": {"columns": columns},
        }

    # --- sequence_column is a legal sink field (spec'd since #109) ---

    def test_sequence_column_accepted_in_sink(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "scd2",
                "business_key": ["id"],
                "sequence_column": "seq_id",
            }
        ]
        errors = validate_contract_schema(data)
        assert not any("sequence_column" in e for e in errors)

    def test_sequence_column_null_accepted(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "scd2",
                "business_key": ["id"],
                "sequence_column": None,
            }
        ]
        errors = validate_contract_schema(data)
        assert not any("sequence_column" in e for e in errors)

    def test_sequence_column_non_string_rejected(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "scd2",
                "business_key": ["id"],
                "sequence_column": 7,
            }
        ]
        errors = validate_contract_schema(data)
        assert any("'sequence_column' must be a string or null" in e for e in errors)

    # --- nullable primary keys (#401 item 3) ---

    def test_nullable_schema_level_primary_key_rejected(self) -> None:
        data = self._base(
            columns=[
                {
                    "name": "id",
                    "type": "string",
                    "nullable": True,
                    "primary_key": True,
                    "pii": False,
                },
            ]
        )
        errors = validate_contract_schema(data)
        assert any(
            "'primary_key: true' cannot be combined with 'nullable: true'" in e for e in errors
        )

    def test_non_nullable_schema_level_primary_key_accepted(self) -> None:
        data = self._base(
            columns=[
                {
                    "name": "id",
                    "type": "string",
                    "nullable": False,
                    "primary_key": True,
                    "pii": False,
                },
            ]
        )
        assert validate_contract_schema(data) == []

    def test_nullable_sink_primary_key_member_rejected_for_upsert(self) -> None:
        """The dim_hcpcs failure: upsert pk includes a nullable date column."""
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "upsert",
                "primary_key": ["id", "from_date"],
            }
        ]
        errors = validate_contract_schema(data)
        assert any("['from_date']" in e and "nullable" in e for e in errors)

    def test_nullable_sink_primary_key_ignored_for_non_upsert(self) -> None:
        """Nullable pk members only block upsert sinks (NULL-vs-ON CONFLICT)."""
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "append",
                "primary_key": ["id", "from_date"],
            }
        ]
        errors = validate_contract_schema(data)
        assert not any("nullable" in e for e in errors)

    # --- sink primary_key cross-reference (#401 adjacent) ---

    def test_sink_primary_key_ghost_column_rejected(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "upsert",
                "primary_key": ["id", "ghost_col"],
            }
        ]
        errors = validate_contract_schema(data)
        assert any(
            "'primary_key' references column 'ghost_col'" in e and "not defined" in e
            for e in errors
        )

    def test_partition_column_not_cross_referenced(self) -> None:
        """partition_column is injected at write time; it is conventionally
        absent from schema.columns and must not be flagged as a ghost ref."""
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "append",
                "partition_column": "load_period",
            }
        ]
        assert validate_contract_schema(data) == []

    # --- upsert partition guard, static form (#401 item 2, guard 2) ---

    def test_upsert_partition_column_not_in_sink_pk_rejected(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "upsert",
                "primary_key": ["id"],
                "partition_column": "load_period",
            }
        ]
        errors = validate_contract_schema(data)
        assert any("does not include it" in e and "load_period" in e for e in errors)

    def test_upsert_partition_column_in_sink_pk_accepted(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "upsert",
                "primary_key": ["id", "load_period"],
                "partition_column": "load_period",
            }
        ]
        # 'load_period' is not a schema column (it is injected at write time),
        # but as the sink's declared partition_column it is exempt from the
        # primary_key ghost-ref rule -- guard 2 requires it in the pk.
        assert validate_contract_schema(data) == []

    def test_upsert_partition_guard_uses_schema_pk_fallback(self) -> None:
        """Without a sink-level primary_key the schema flags are the
        effective conflict key (mirrors reconcile_primary_key)."""
        data = self._base(
            columns=[
                {
                    "name": "id",
                    "type": "string",
                    "nullable": False,
                    "primary_key": True,
                    "pii": False,
                },
            ]
        )
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "upsert",
                "partition_column": "load_period",
            }
        ]
        errors = validate_contract_schema(data)
        assert any("does not include it" in e and "['id']" in e for e in errors)

    # --- detect_deletes requires scd2 (#401 adjacent) ---

    def test_detect_deletes_on_non_scd2_mode_rejected(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "full_refresh",
                "detect_deletes": True,
            }
        ]
        errors = validate_contract_schema(data)
        assert any("'detect_deletes' is only valid with mode 'scd2'" in e for e in errors)

    def test_detect_deletes_on_scd2_accepted(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "scd2",
                "business_key": ["id"],
                "detect_deletes": True,
            }
        ]
        errors = validate_contract_schema(data)
        assert not any("detect_deletes" in e for e in errors)

    def test_detect_deletes_without_declared_mode_accepted(self) -> None:
        """Mode may live in asset metadata; without a declared sink mode the
        write-time guard remains the backstop."""
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "detect_deletes": True,
            }
        ]
        errors = validate_contract_schema(data)
        assert not any("detect_deletes" in e for e in errors)

    # --- skip_unchanged requires upsert (mirror issue
    # model-oncology-public/moncpipelib#3) ---

    def test_skip_unchanged_on_non_upsert_mode_rejected(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "full_refresh",
                "skip_unchanged": True,
            }
        ]
        errors = validate_contract_schema(data)
        assert any("'skip_unchanged' is only valid with mode 'upsert'" in e for e in errors)

    def test_skip_unchanged_on_upsert_accepted(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "upsert",
                "primary_key": ["id"],
                "skip_unchanged": True,
            }
        ]
        errors = validate_contract_schema(data)
        assert not any("skip_unchanged" in e for e in errors)

    def test_skip_unchanged_non_boolean_rejected(self) -> None:
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "upsert",
                "primary_key": ["id"],
                "skip_unchanged": "yes",
            }
        ]
        errors = validate_contract_schema(data)
        assert any("'skip_unchanged' must be a boolean" in e for e in errors)

    def test_skip_unchanged_false_on_non_upsert_accepted(self) -> None:
        """An explicit False is inert everywhere and must not be flagged."""
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "mode": "scd2",
                "business_key": ["id"],
                "skip_unchanged": False,
            }
        ]
        errors = validate_contract_schema(data)
        assert not any("skip_unchanged" in e for e in errors)

    def test_skip_unchanged_without_declared_mode_accepted(self) -> None:
        """Mode may live in asset metadata; without a declared sink mode the
        write-time guard (validate_write_config) remains the backstop."""
        data = self._base()
        data["sinks"] = [
            {
                "type": "table",
                "schema": "silver",
                "table": "t",
                "skip_unchanged": True,
            }
        ]
        errors = validate_contract_schema(data)
        assert not any("skip_unchanged" in e for e in errors)


# ---------------------------------------------------------------------------
# TestPartitionedSinkGuards (#401 item 2, guard 1 static form)
# ---------------------------------------------------------------------------


class TestPartitionedSinkGuards:
    """load_contract rejects destructive sink modes without partition_column
    when the resolved data source declares partitioning."""

    CONTRACT_HEADER = (
        'version: "1.0"\n'
        'pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"\n'
        "asset: fda_ndc_directory\n"
        "layer: bronze\n"
        "data_source: fda.source.yaml\n"
        "schema:\n"
        "  columns:\n"
        "    - name: ndc\n"
        "      type: string\n"
        "      nullable: false\n"
        "      pii: false\n"
    )

    PARTITIONED_SOURCE = (
        'source_id: "b2c3d4e5-f6a7-8901-bcde-f23456789012"\n'
        "source_name: fda_ndc_directory\n"
        "periods:\n"
        "  - source: https://example.test/2025.zip\n"
        "    effective_from: 2025-01-01\n"
        "    effective_to: 2026-01-01\n"
        "    partition_key: p2025\n"
        "  - source: https://example.test/2026.zip\n"
        "    effective_from: 2026-01-01\n"
        "    partition_key: p2026\n"
    )

    UNPARTITIONED_SOURCE = (
        'source_id: "b2c3d4e5-f6a7-8901-bcde-f23456789012"\n'
        "source_name: fda_ndc_directory\n"
        "periods:\n"
        "  - source: https://example.test/all.zip\n"
        "    effective_from: 2025-01-01\n"
    )

    def _write(self, tmp_path: Path, contract_body: str, source_body: str) -> Path:
        (tmp_path / "fda.source.yaml").write_text(source_body)
        contract_path = tmp_path / "fda.contract.yaml"
        contract_path.write_text(contract_body)
        return contract_path

    def test_partitioned_full_refresh_without_partition_column_rejected(
        self, tmp_path: Path
    ) -> None:
        """The data-platform PF-4 failure: dynamically partitioned bronze with
        mode full_refresh and no partition_column, statically rejected."""
        contract = self.CONTRACT_HEADER + (
            "sinks:\n"
            "  - type: table\n"
            "    schema: synthetic_bronze\n"
            "    table: fda_ndc_directory\n"
            "    mode: full_refresh\n"
        )
        path = self._write(tmp_path, contract, self.PARTITIONED_SOURCE)
        with pytest.raises(ContractValidationError) as exc:
            load_contract(path)
        assert "partitioned" in str(exc.value)
        assert "partition_column" in str(exc.value)

    def test_partitioned_scd2_without_partition_column_rejected(self, tmp_path: Path) -> None:
        contract = self.CONTRACT_HEADER + (
            "sinks:\n"
            "  - type: table\n"
            "    schema: silver\n"
            "    table: fda_ndc_directory\n"
            "    mode: scd2\n"
            "    business_key: [ndc]\n"
        )
        path = self._write(tmp_path, contract, self.PARTITIONED_SOURCE)
        with pytest.raises(ContractValidationError) as exc:
            load_contract(path)
        assert "partition_column" in str(exc.value)

    def test_partitioned_full_refresh_with_partition_column_accepted(self, tmp_path: Path) -> None:
        contract = self.CONTRACT_HEADER + (
            "sinks:\n"
            "  - type: table\n"
            "    schema: synthetic_bronze\n"
            "    table: fda_ndc_directory\n"
            "    mode: full_refresh\n"
            "    partition_column: load_period\n"
        )
        path = self._write(tmp_path, contract, self.PARTITIONED_SOURCE)
        loaded = load_contract(path)
        assert loaded.data_source is not None

    def test_unpartitioned_source_full_refresh_accepted(self, tmp_path: Path) -> None:
        """No period carries a partition_key -> not partitioned -> no guard."""
        contract = self.CONTRACT_HEADER + (
            "sinks:\n"
            "  - type: table\n"
            "    schema: synthetic_bronze\n"
            "    table: fda_ndc_directory\n"
            "    mode: full_refresh\n"
        )
        path = self._write(tmp_path, contract, self.UNPARTITIONED_SOURCE)
        loaded = load_contract(path)
        assert loaded.data_source is not None

    def test_partitioned_append_without_partition_column_accepted(self, tmp_path: Path) -> None:
        """Append is non-destructive; the guard covers full_refresh/scd2 only."""
        contract = self.CONTRACT_HEADER + (
            "sinks:\n"
            "  - type: table\n"
            "    schema: synthetic_bronze\n"
            "    table: fda_ndc_directory\n"
            "    mode: append\n"
        )
        path = self._write(tmp_path, contract, self.PARTITIONED_SOURCE)
        loaded = load_contract(path)
        assert loaded.data_source is not None

    def test_undeclared_mode_without_partition_column_accepted(self, tmp_path: Path) -> None:
        """Mode from asset metadata is not statically knowable; the write-time
        guard remains the backstop."""
        contract = self.CONTRACT_HEADER + (
            "sinks:\n  - type: table\n    schema: synthetic_bronze\n    table: fda_ndc_directory\n"
        )
        path = self._write(tmp_path, contract, self.PARTITIONED_SOURCE)
        loaded = load_contract(path)
        assert loaded.data_source is not None
