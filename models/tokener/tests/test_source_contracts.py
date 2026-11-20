import json

import pytest
import yaml

from models.llm._wire import TRANSPORT_FIELDS
from scripts import source_contracts
from scripts.source_contracts import apply_source_contracts, load_source_contracts
from scripts.sync_catalog import LLM_DIR, SOURCE_ROOT, CatalogError, sync_files


def test_source_rules_and_provider_mapping_are_reproducible_and_drift_is_detected(
    tmp_path, monkeypatch
):
    contracts = load_source_contracts(SOURCE_ROOT)
    files = {path.name: path.read_text() for path in LLM_DIR.glob("*.yaml")}
    generated = apply_source_contracts(files.copy(), contracts)
    assert generated == {name: (LLM_DIR / name).read_text() for name in generated}
    monkeypatch.setitem(source_contracts.SOURCE_PROVIDERS, "openai", "openai")
    openai_contracts = load_source_contracts(SOURCE_ROOT)
    assert openai_contracts
    applied = apply_source_contracts(
        {"gpt-4o-mini.yaml": files["gpt-4o-mini.yaml"]}, openai_contracts
    )
    for name, body in applied.items():
        if name.startswith("_"):
            continue
        rules = yaml.safe_load(body)["parameter_rules"]
        assert not TRANSPORT_FIELDS & {rule["name"] for rule in rules}
    sync_files(generated, check=False, output_dir=tmp_path)
    assert sync_files(generated, check=True, output_dir=tmp_path)
    target = tmp_path / "gpt-4o-mini.yaml"
    target.write_text(target.read_text().replace("default: 8192", "default: 4096"))
    assert not sync_files(generated, check=True, output_dir=tmp_path)
    target.write_text(generated[target.name])
    (tmp_path / "_provider_models.json").write_text(json.dumps({"glm-5": "openai"}))
    assert not sync_files(generated, check=True, output_dir=tmp_path)


def test_missing_source_provider_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setitem(source_contracts.SOURCE_PROVIDERS, "openai", "openai")
    with pytest.raises(CatalogError):
        load_source_contracts(tmp_path)
