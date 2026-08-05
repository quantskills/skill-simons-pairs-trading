from __future__ import annotations

import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from research_contract import CODE_VERSION  # noqa: E402


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_public_docs_describe_adjusted_fdr_formation_only_workflow():
    combined = "\n".join([
        _read("SKILL.md"),
        _read("README.md"),
        _read("references/methodology.md"),
    ])

    assert "get_stock_daily_pre" in combined
    assert "前复权" in combined
    assert "Benjamini" in combined
    assert "形成期" in combined
    assert "max_pairs_per_symbol" in combined
    assert "run_id" in combined
    assert "price_panel_sha256" in combined
    assert "回测后使用每一对" not in combined
    assert "references/pca_reduce.py" not in combined
    assert "references\\pca_reduce.py" not in combined


def test_obsolete_post_backtest_pca_selector_is_not_shipped():
    assert not (ROOT / "references" / "pca_reduce.py").exists()


def test_public_package_does_not_ship_stale_generated_outputs():
    assert not (ROOT / "outputs").exists()


def test_power_shell_examples_use_consistent_date_indexes_and_continuations():
    example = _read("examples/01_daily_signal.md")

    assert '$env:SIMONS_TODAY="20260710"' in example
    assert "--indexes 000300.SH 000905.SH" in example
    assert "20260712" not in example
    assert " \\\n" not in example
    assert "SHORT_SPREAD" not in example
    assert "LONG_SPREAD" not in example
    assert "NO_TRADE" in example


def test_output_documentation_uses_parameterized_run_directory():
    combined = _read("SKILL.md") + "\n" + _read("README.md")

    assert "backtest_YYYYMMDD_<config_id前12位>" in combined
    assert "signals_YYYYMMDD_<config_id前12位>.csv" in combined
    assert "formation_representatives.csv" in combined
    assert "formation_collinearity.json" in combined


def test_package_and_ci_versions_match_the_research_contract():
    project = tomllib.loads(_read("pyproject.toml"))
    workflow = _read(".github/workflows/ci.yml")

    assert project["project"]["version"] == CODE_VERSION
    assert project["project"]["requires-python"] == ">=3.11,<3.13"
    assert "--require-hashes" in workflow
