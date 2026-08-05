from __future__ import annotations

import subprocess
import sys
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "scripts" / "cli.py"
sys.path.insert(0, str(ROOT / "scripts"))

import cli as simons_cli  # noqa: E402


PAIR_SYMBOLS = [
    "600000.SH", "600001.SH", "000001.SZ", "000002.SZ"
]
PAIR_NAMES = [
    f"{PAIR_SYMBOLS[0]}~{PAIR_SYMBOLS[1]}",
    f"{PAIR_SYMBOLS[2]}~{PAIR_SYMBOLS[3]}",
]


@pytest.mark.parametrize(
    "bad_pair",
    [
        "../escape,000001.SZ",
        r"..\escape,000001.SZ",
        "600519.SH,C:/temp/escape",
        "600519.SH,000001.SZ/../../escape",
    ],
)
def test_diagnose_rejects_path_like_symbols_before_login(
        bad_pair, monkeypatch):
    login_calls = []
    monkeypatch.setattr(
        simons_cli, "ensure_login", lambda: login_calls.append(True)
    )
    monkeypatch.setattr(
        simons_cli,
        "build_price_panel",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("非法代码不应触发数据访问")
        ),
    )

    with pytest.raises(ValueError, match="证券代码"):
        simons_cli.cmd_diagnose(SimpleNamespace(
            pair=bad_pair,
            lookback_days=500,
            price_basis="pre_adjusted",
            z_window=60,
        ))

    assert login_calls == []


def test_default_output_root_is_anchored_to_skill_not_current_directory(
        monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SIMONS_OUT", raising=False)

    assert simons_cli._default_output_root() == \
        ROOT / "outputs" / "simons_pairs"


def test_installed_console_output_defaults_to_current_directory(
        monkeypatch, tmp_path):
    installed_root = tmp_path / "site-packages"
    working = tmp_path / "working"
    installed_root.mkdir()
    working.mkdir()
    monkeypatch.setattr(simons_cli, "SKILL_ROOT", installed_root)
    monkeypatch.chdir(working)
    monkeypatch.delenv("SIMONS_OUT", raising=False)

    assert simons_cli._default_output_root() == \
        working / "outputs" / "simons_pairs"


def test_all_help_commands_render_without_crashing():
    commands = [
        ["--help"],
        ["signal", "--help"],
        ["backtest", "--help"],
        ["diagnose", "--help"],
        ["health-check", "--help"],
        ["check-login", "--help"],
    ]
    for args in commands:
        result = subprocess.run(
            [sys.executable, str(CLI), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_check_login_never_prints_token(monkeypatch, capsys):
    token = "sensitive-token-value"
    monkeypatch.setattr(simons_cli, "ensure_login", lambda: token)

    simons_cli.cmd_check_login(object())

    output = capsys.readouterr().out
    assert token not in output
    assert token[:8] not in output
    assert "OK" in output


def test_health_check_reports_safely(monkeypatch, capsys):
    token = "must-never-appear"
    monkeypatch.setenv("PANDADATA_TOKEN", token)
    monkeypatch.setattr(simons_cli, "_load_dotenv", lambda candidates=None: None)
    monkeypatch.setattr(
        simons_cli,
        "startup_health_check",
        lambda **kwargs: {
            "status": "ok",
            "python": "3.11.9",
            "endpoint": "https://provider.example.test/api",
            "sdk": "compatible",
            "auth_mode": "token",
            "tls_version": "TLSv1.3",
            "certificate_not_after": "2030-12-31",
        },
    )

    rc = simons_cli.cmd_health_check(
        SimpleNamespace(timeout=3.0, transport_only=False)
    )

    output = capsys.readouterr().out
    assert rc == 0
    assert "OK" in output
    assert token not in output


def test_main_runs_login_before_data_command(monkeypatch):
    calls = []
    monkeypatch.setattr(
        simons_cli, "ensure_login", lambda: calls.append("login") or "ok"
    )
    monkeypatch.setattr(
        simons_cli,
        "cmd_signal",
        lambda args: calls.append("command") or 0,
    )

    rc = simons_cli.main(["signal"])

    assert rc == 0
    assert calls == ["login", "command"]


def test_main_fails_closed_before_data_access(monkeypatch, capsys):
    secret = "must-never-appear"
    monkeypatch.setattr(
        simons_cli,
        "ensure_login",
        lambda: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    monkeypatch.setattr(
        simons_cli,
        "cmd_signal",
        lambda args: pytest.fail("data command must not run"),
    )

    rc = simons_cli.main(["signal"])

    output = capsys.readouterr()
    assert rc != 0
    assert secret not in output.out + output.err


def test_check_login_failure_is_sanitized(monkeypatch, capsys):
    secret = "do-not-print-this"
    monkeypatch.setattr(
        simons_cli, "ensure_login",
        lambda: (_ for _ in ()).throw(RuntimeError(f"failure {secret}")),
    )

    rc = simons_cli.cmd_check_login(object())

    output = capsys.readouterr().out
    assert rc != 0
    assert secret not in output
    assert "FAIL" in output


def test_backtest_uses_latest_available_historical_start_date_for_universe(
        monkeypatch, tmp_path):
    observed = {}
    monkeypatch.setattr(simons_cli, "ensure_login", lambda: "ok")
    monkeypatch.setattr(simons_cli, "_today_str", lambda: "20260710")
    periods = simons_cli._research_periods(
        today="20260710", years=5, formation_days=252
    )
    requested_data_start = periods["data_period"][0]
    first_fallback = (
        pd.Timestamp(requested_data_start) - pd.Timedelta(days=1)
    ).strftime("%Y%m%d")
    second_fallback = (
        pd.Timestamp(requested_data_start) - pd.Timedelta(days=2)
    ).strftime("%Y%m%d")

    def fake_universe(indexes, date):
        observed.setdefault("dates", []).append(date)
        if date in {requested_data_start, first_fallback}:
            return pd.DataFrame()
        return pd.DataFrame({"symbol": ["A", "B"]})

    monkeypatch.setattr(simons_cli, "get_universe", fake_universe)
    monkeypatch.setattr(
        simons_cli, "build_price_panel",
        lambda symbols, start, end, **kwargs: pd.DataFrame(
            {"A": 1.0, "B": 1.0},
            index=pd.bdate_range(
                pd.Timestamp(start) - pd.Timedelta(days=10),
                pd.Timestamp(end),
            ),
        ),
    )
    monkeypatch.setattr(
        simons_cli, "get_industry_map",
        lambda *args, **kwargs: pd.DataFrame(
            {"symbol": ["A", "B"], "industry": ["x", "x"]}
        ),
    )
    monkeypatch.setattr(
        simons_cli,
        "select_pairs_for_backtest",
        lambda *args, **kwargs: (
            pd.DataFrame(),
            {
                "selection_method": "formation_leg_return_pca_hclust_v1",
                "n_candidates": 0,
                "n_representatives": 0,
                "condition_number": None,
                "effective_rank": 0.0,
                "max_abs_pair_correlation": 0.0,
                "pc_explained_variance_ratio": [],
            },
        ),
    )

    def fake_outdir(sub=""):
        path = tmp_path / sub
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(simons_cli, "_outdir", fake_outdir)
    args = SimpleNamespace(
        indexes=["000300.SH"], years=5, corr=0.8, pvalue=0.05,
        hl_min=2.0, hl_max=60.0, top_n=40, formation=252,
        reestimate_days=60, z_entry=2.0, z_exit=0.5,
        z_stop=4.0, max_hold=60, z_window=60, pair_stop_loss=0.05,
        cost_bps=7.5, method="static",
    )

    simons_cli.cmd_backtest(args)

    assert observed["dates"] == [
        requested_data_start, first_fallback, second_fallback
    ]
    report_path = next(tmp_path.rglob("backtest_20260710_*/report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["strategy_gate"] == "NO_TRADE_INSUFFICIENT_SAMPLE"
    assert report["universe_date"] == second_fallback
    assert report["evaluation_period"] == ["20210711", "20260710"]
    assert report["performance"]["n_days"] >= 5 * 250
    assert report["selection_diagnostics"]["industry_data_quality"] == {
        "requested": 2,
        "resolved": 2,
        "unknown": 0,
        "coverage": 1.0,
        "minimum": 0.95,
    }


def _full_gate_config():
    return {
        "formation_days": 252,
        "reestimate_days": 60,
        "corr_threshold": 0.80,
        "pvalue_cutoff": 0.05,
        "fdr_alpha": 0.05,
        "half_life_min": 2.0,
        "half_life_max": 60.0,
        "method": "static",
        "kalman_delta": 1e-4,
        "kalman_r": 1e-3,
        "z_window": 60,
        "z_entry": 2.0,
        "z_exit": 0.5,
        "z_stop": 4.0,
        "max_hold_days": 60,
        "pair_stop_loss": 0.05,
        "cost_bps_one_side": 7.5,
        "short_borrow_bps_annual": 800.0,
        "price_basis": "pre_adjusted",
        "price_source_method": "get_stock_daily_pre",
        "selection_method": "formation_leg_return_pca_hclust_v1",
        "dedup_clusters": 20,
        "max_pairs_per_symbol": 1,
        "top_n": 40,
        "evaluation_years": 5,
        "signal_lookback_days": 400,
    }


def _valid_report(strategy_gate="RESEARCH_PASS"):
    indexes = ["000300.SH", "000905.SH"]
    config = _full_gate_config()
    periods = {
        "data_period": ["20200722", "20260710"],
        "formation_period": ["20200722", "20210709"],
        "evaluation_period": ["20210711", "20260710"],
    }
    fingerprints = {
        "price_panel_sha256": "1" * 64,
        "universe_sha256": "2" * 64,
        "industry_map_sha256": "3" * 64,
    }
    performance = {
        "sharpe": 0.3, "annual_ret": 0.02, "annual_vol": 0.10,
        "max_dd": -0.08, "calmar": 0.25, "n_days": 1250,
    }
    trades = {
        "n_trades": 150, "win_rate": 0.52, "avg_ret": 0.001,
        "avg_hold": 10.0, "profit_factor": 1.1,
    }
    if strategy_gate == "RESEARCH_PASS":
        performance.update({"sharpe": 1.0, "annual_ret": 0.05})
    elif strategy_gate == "NO_TRADE_NEGATIVE_EDGE":
        performance.update({"sharpe": -0.1, "annual_ret": 0.02})
    elif strategy_gate == "NO_TRADE_INSUFFICIENT_SAMPLE":
        performance.update({"sharpe": 1.0, "annual_ret": 0.05})
        trades["n_trades"] = 40
    signal_snapshot = {
        "lookback_days": 400,
        "price_basis": "pre_adjusted",
        "symbols": PAIR_SYMBOLS,
        "price_panel_sha256": "4" * 64,
        "industry_map_sha256": "5" * 64,
    }
    identity = simons_cli._make_run_identity(
        indexes=indexes,
        config_summary=config,
        periods=periods,
        data_fingerprints=fingerprints,
        representative_pairs=PAIR_NAMES,
        research_results={
            "strategy_gate": strategy_gate,
            "performance": performance,
            "trades": trades,
        },
        signal_snapshot=signal_snapshot,
    )
    return {
        **identity,
        "period": ["20210711", "20260710"],
        "periods": periods,
        "data_period": periods["data_period"],
        "formation_period": periods["formation_period"],
        "evaluation_period": periods["evaluation_period"],
        "indexes": indexes,
        "config": dict(config),
        "config_summary": dict(config),
        "data_fingerprints": fingerprints,
        "representative_pairs": PAIR_NAMES,
        "strategy_gate": strategy_gate,
        "performance": performance,
        "trades": trades,
        "signal_snapshot": signal_snapshot,
    }


def test_report_pairs_validate_codes_uniqueness_and_limit():
    expected = [
        (PAIR_SYMBOLS[0], PAIR_SYMBOLS[1]),
        (PAIR_SYMBOLS[2], PAIR_SYMBOLS[3]),
    ]
    assert simons_cli._validated_report_pairs(
        PAIR_NAMES, limit=2
    ) == expected
    assert simons_cli._validated_report_pairs(
        ["../../escape~000001.SZ"], limit=2
    ) is None
    assert simons_cli._validated_report_pairs(
        [PAIR_NAMES[0], PAIR_NAMES[0]], limit=2
    ) is None
    assert simons_cli._validated_report_pairs(
        PAIR_NAMES, limit=1
    ) is None


def test_signal_spread_respects_kalman_method(monkeypatch):
    dates = pd.bdate_range("2025-01-01", periods=80)
    panel = pd.DataFrame({
        PAIR_SYMBOLS[0]: np.linspace(10, 12, len(dates)),
        PAIR_SYMBOLS[1]: np.linspace(8, 10, len(dates)),
    }, index=dates)
    expected = pd.Series(
        np.linspace(-0.2, 0.3, len(dates)), index=dates
    )
    observed = {}

    def fake_kalman(price_a, price_b, *, delta, r_var):
        observed.update({"delta": delta, "r_var": r_var})
        return pd.DataFrame({"spread": expected})

    monkeypatch.setattr(
        simons_cli, "kalman_dynamic_beta", fake_kalman
    )
    monkeypatch.setattr(
        simons_cli,
        "static_spread",
        lambda *args, **kwargs: pytest.fail(
            "Kalman 信号不得退回静态价差"
        ),
    )
    spread = simons_cli._signal_spread(
        panel,
        SimpleNamespace(
            a=PAIR_SYMBOLS[0],
            b=PAIR_SYMBOLS[1],
            alpha=0.0,
            beta=1.0,
        ),
        {
            "method": "kalman",
            "kalman_delta": 2e-4,
            "kalman_r": 3e-3,
        },
    )

    pd.testing.assert_series_equal(spread, expected, check_names=False)
    assert observed == {"delta": 2e-4, "r_var": 3e-3}


def test_signal_gate_requires_matching_research_pass_report(tmp_path):
    assert simons_cli._load_signal_gate("20260710", tmp_path / "missing.json") \
        == "NO_TRADE_MISSING_BACKTEST"

    report = tmp_path / "report.json"
    valid = _valid_report("NO_TRADE_NEGATIVE_EDGE")
    report.write_text(json.dumps(valid), encoding="utf-8")
    assert simons_cli._load_signal_gate(
        "20260710", report,
        expected_indexes=valid["indexes"],
        expected_config=valid["config_summary"],
    ) \
        == "NO_TRADE_NEGATIVE_EDGE"

    stale = deepcopy(valid)
    stale["evaluation_period"][-1] = "20260709"
    stale["periods"]["evaluation_period"][-1] = "20260709"
    report.write_text(json.dumps(stale), encoding="utf-8")
    assert simons_cli._load_signal_gate(
        "20260710", report,
        expected_indexes=valid["indexes"],
        expected_config=valid["config_summary"],
    ) \
        == "NO_TRADE_STALE_BACKTEST"

    report.write_text(json.dumps(valid), encoding="utf-8")
    assert simons_cli._load_signal_gate(
        "20260710", report, expected_indexes=["000905.SH"],
        expected_config=valid["config_summary"],
    ) == "NO_TRADE_CONFIG_MISMATCH"


def test_gate_rejects_each_missing_or_changed_critical_parameter(tmp_path):
    valid = _valid_report()
    report_path = tmp_path / "report.json"
    for key in simons_cli.REQUIRED_GATE_CONFIG_KEYS:
        tampered = deepcopy(valid)
        tampered["config_summary"].pop(key)
        tampered["config"].pop(key)
        report_path.write_text(json.dumps(tampered), encoding="utf-8")

        assert simons_cli._load_signal_gate(
            "20260710",
            report_path,
            expected_indexes=valid["indexes"],
            expected_config=valid["config_summary"],
        ) == "NO_TRADE_CONFIG_MISMATCH", key


def test_gate_rejects_tampered_data_fingerprint_and_price_basis(tmp_path):
    valid = _valid_report()
    report_path = tmp_path / "report.json"
    tampered = deepcopy(valid)
    tampered["data_fingerprints"]["price_panel_sha256"] = "9" * 64
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert simons_cli._load_signal_gate(
        "20260710", report_path,
        expected_indexes=valid["indexes"],
        expected_config=valid["config_summary"],
    ) == "NO_TRADE_INVALID_BACKTEST"

    valid["config_summary"]["price_basis"] = "post_adjusted"
    valid["config"]["price_basis"] = "post_adjusted"
    report_path.write_text(json.dumps(valid), encoding="utf-8")
    assert simons_cli._load_signal_gate(
        "20260710", report_path,
        expected_indexes=valid["indexes"],
        expected_config=_full_gate_config(),
    ) == "NO_TRADE_CONFIG_MISMATCH"


def test_gate_rejects_tampered_strategy_outcome_or_performance(tmp_path):
    valid = _valid_report("NO_TRADE_WEAK_EDGE")
    path = tmp_path / "report.json"

    changed_gate = deepcopy(valid)
    changed_gate["strategy_gate"] = "RESEARCH_PASS"
    path.write_text(json.dumps(changed_gate), encoding="utf-8")
    assert simons_cli._load_signal_gate(
        "20260710",
        path,
        expected_indexes=valid["indexes"],
        expected_config=valid["config_summary"],
    ) == "NO_TRADE_INVALID_BACKTEST"

    rehashed_gate = deepcopy(valid)
    rehashed_gate["strategy_gate"] = "RESEARCH_PASS"
    rehashed_gate.update(simons_cli._make_run_identity(
        indexes=rehashed_gate["indexes"],
        config_summary=rehashed_gate["config_summary"],
        periods=rehashed_gate["periods"],
        data_fingerprints=rehashed_gate["data_fingerprints"],
        representative_pairs=rehashed_gate["representative_pairs"],
        research_results={
            "strategy_gate": rehashed_gate["strategy_gate"],
            "performance": rehashed_gate["performance"],
            "trades": rehashed_gate["trades"],
        },
        signal_snapshot=rehashed_gate["signal_snapshot"],
    ))
    path.write_text(json.dumps(rehashed_gate), encoding="utf-8")
    assert simons_cli._load_signal_gate(
        "20260710",
        path,
        expected_indexes=valid["indexes"],
        expected_config=valid["config_summary"],
    ) == "NO_TRADE_INVALID_BACKTEST"

    changed_performance = deepcopy(valid)
    changed_performance["performance"]["sharpe"] = 99.0
    path.write_text(json.dumps(changed_performance), encoding="utf-8")
    assert simons_cli._load_signal_gate(
        "20260710",
        path,
        expected_indexes=valid["indexes"],
        expected_config=valid["config_summary"],
    ) == "NO_TRADE_INVALID_BACKTEST"


def test_shared_signal_window_fingerprint_detects_price_or_industry_revision():
    dates = pd.bdate_range("2025-01-01", "2026-07-10")
    panel = pd.DataFrame({
        "A": np.linspace(10, 20, len(dates)),
        "B": np.linspace(8, 18, len(dates)),
    }, index=dates)
    industry = pd.DataFrame({
        "symbol": ["A", "B"], "industry": ["bank", "bank"]
    })
    snapshot = simons_cli._make_signal_snapshot(
        panel=panel,
        industry_map=industry,
        symbols=["A", "B"],
        end_date="20260710",
        lookback_days=400,
        price_basis="pre_adjusted",
    )

    assert simons_cli._signal_snapshot_matches(
        snapshot,
        panel=panel,
        industry_map=industry,
        end_date="20260710",
    )
    revised_prices = panel.copy()
    revised_prices.iloc[-1, 0] += 0.01
    assert not simons_cli._signal_snapshot_matches(
        snapshot,
        panel=revised_prices,
        industry_map=industry,
        end_date="20260710",
    )
    assert not simons_cli._signal_snapshot_matches(
        snapshot,
        panel=panel.drop(columns=["B"]),
        industry_map=industry,
        end_date="20260710",
    )
    revised_industry = industry.copy()
    revised_industry.loc[0, "industry"] = "nonbank"
    assert not simons_cli._signal_snapshot_matches(
        snapshot,
        panel=panel,
        industry_map=revised_industry,
        end_date="20260710",
    )


def test_incomplete_signal_snapshot_is_recorded_but_never_allows_direction():
    dates = pd.bdate_range("2025-01-01", "2026-07-10")
    panel = pd.DataFrame({
        "A": np.linspace(10, 20, len(dates)),
    }, index=dates)
    industry = pd.DataFrame({
        "symbol": ["A", "B"], "industry": ["bank", "bank"]
    })

    snapshot = simons_cli._make_signal_snapshot(
        panel=panel,
        industry_map=industry,
        symbols=["A", "B"],
        end_date="20260710",
        lookback_days=400,
        price_basis="pre_adjusted",
    )

    assert snapshot["data_complete"] is False
    assert snapshot["missing_price_symbols"] == ["B"]
    assert not simons_cli._signal_snapshot_matches(
        snapshot,
        panel=panel,
        industry_map=industry,
        end_date="20260710",
    )


def test_industry_quality_fails_closed_below_95_percent_coverage():
    symbols = [f"S{i:02d}" for i in range(20)]
    acceptable = pd.DataFrame({
        "symbol": symbols,
        "industry": ["bank"] * 19 + ["UNKNOWN"],
    })
    insufficient = acceptable.copy()
    insufficient.loc[18, "industry"] = "UNKNOWN"

    quality = simons_cli._require_industry_coverage(
        acceptable, symbols, minimum=0.95
    )

    assert quality == {
        "requested": 20,
        "resolved": 19,
        "unknown": 1,
        "coverage": 0.95,
        "minimum": 0.95,
    }
    with pytest.raises(RuntimeError, match="行业数据覆盖率"):
        simons_cli._require_industry_coverage(
            insufficient, symbols, minimum=0.95
        )


def test_research_period_adds_warmup_before_full_evaluation_window():
    periods = simons_cli._research_periods(
        today="20260710", years=5, formation_days=252
    )

    evaluation_start = pd.Timestamp(periods["evaluation_period"][0])
    evaluation_end = pd.Timestamp(periods["evaluation_period"][1])
    data_start = pd.Timestamp(periods["data_period"][0])
    assert (evaluation_end - evaluation_start).days == 5 * 365
    assert (evaluation_start - data_start).days >= 400


def test_research_gate_rejects_less_than_five_year_evaluation(tmp_path):
    short = _valid_report()
    short["config_summary"]["evaluation_years"] = 4
    short["config"]["evaluation_years"] = 4
    short["periods"]["evaluation_period"] = ["20220711", "20260710"]
    short["evaluation_period"] = ["20220711", "20260710"]
    short["period"] = ["20220711", "20260710"]
    short.update(simons_cli._make_run_identity(
        indexes=short["indexes"],
        config_summary=short["config_summary"],
        periods=short["periods"],
        data_fingerprints=short["data_fingerprints"],
        representative_pairs=short["representative_pairs"],
        research_results={
            "strategy_gate": short["strategy_gate"],
            "performance": short["performance"],
            "trades": short["trades"],
        },
        signal_snapshot=short["signal_snapshot"],
    ))
    path = tmp_path / "report.json"
    path.write_text(json.dumps(short), encoding="utf-8")

    assert simons_cli._load_signal_gate(
        "20260710",
        path,
        expected_indexes=short["indexes"],
        expected_config=short["config_summary"],
    ) == "NO_TRADE_CONFIG_MISMATCH"


def test_prepare_research_panel_uses_last_warmup_rows_and_full_evaluation():
    dates = pd.bdate_range("2020-01-01", "2026-07-10")
    panel = pd.DataFrame(
        {"A": np.arange(len(dates)) + 1.0, "B": np.arange(len(dates)) + 2.0},
        index=dates,
    )

    research, formation, evaluation = simons_cli._prepare_research_panel(
        panel, evaluation_start="20210711", formation_days=252
    )

    assert len(formation) == 252
    assert formation.index.max() < pd.Timestamp("20210711")
    assert evaluation.index.min() >= pd.Timestamp("20210711")
    assert research.index[252] == evaluation.index[0]


def test_config_hash_prevents_different_runs_from_overwriting_each_other():
    base = _full_gate_config()
    changed = {**base, "z_entry": 2.5}
    first = simons_cli._config_id(["000300.SH", "000905.SH"], base)
    second = simons_cli._config_id(["000300.SH", "000905.SH"], changed)

    assert first != second
    assert simons_cli._backtest_subdir("20260710", first) != \
        simons_cli._backtest_subdir("20260710", second)


def test_symbol_overlap_limit_is_part_of_gate_and_run_identity():
    assert "max_pairs_per_symbol" in simons_cli.REQUIRED_GATE_CONFIG_KEYS
    base = _full_gate_config()
    relaxed = {**base, "max_pairs_per_symbol": 2}

    assert simons_cli._config_id(simons_cli.DEFAULT_INDEXES, base) != \
        simons_cli._config_id(simons_cli.DEFAULT_INDEXES, relaxed)


def test_unused_legacy_trading_window_is_not_a_public_or_gated_parameter():
    parser = simons_cli._build_parser()
    subparsers = next(
        action for action in parser._actions
        if isinstance(action, __import__("argparse")._SubParsersAction)
    )
    backtest_help = subparsers.choices["backtest"].format_help()

    assert "--trading" not in backtest_help
    assert "trading_days" not in simons_cli.REQUIRED_GATE_CONFIG_KEYS


def test_backtest_and_signal_parser_defaults_use_same_dual_index_pool():
    parser = simons_cli._build_parser()

    backtest_args = parser.parse_args(["backtest"])
    signal_args = parser.parse_args(["signal"])

    assert backtest_args.indexes == simons_cli.DEFAULT_INDEXES
    assert signal_args.indexes == simons_cli.DEFAULT_INDEXES


def test_no_trade_signal_output_never_exposes_long_or_short_direction(
        monkeypatch, tmp_path, capsys):
    report = _valid_report("NO_TRADE_WEAK_EDGE")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    dates = pd.bdate_range("2025-01-01", periods=280)
    panel = pd.DataFrame({
        PAIR_SYMBOLS[0]: np.linspace(10, 20, len(dates)),
        PAIR_SYMBOLS[1]: np.linspace(8, 18, len(dates)),
        PAIR_SYMBOLS[2]: np.linspace(12, 22, len(dates)),
        PAIR_SYMBOLS[3]: np.linspace(9, 19, len(dates)),
    }, index=dates)
    panel.attrs.update({
        "price_basis": "pre_adjusted",
        "source_method": "get_stock_daily_pre",
    })
    monkeypatch.setattr(simons_cli, "ensure_login", lambda: "ok")
    monkeypatch.setattr(simons_cli, "_today_str", lambda: "20260710")
    monkeypatch.setattr(
        simons_cli, "get_universe",
        lambda *args, **kwargs: pd.DataFrame({
            "symbol": PAIR_SYMBOLS[:2]
        }),
    )
    monkeypatch.setattr(simons_cli, "build_price_panel", lambda *a, **k: panel)
    monkeypatch.setattr(
        simons_cli,
        "get_industry_map",
        lambda *a, **k: pd.DataFrame({
            "symbol": PAIR_SYMBOLS,
            "industry": ["x", "x", "y", "y"],
        }),
    )
    monkeypatch.setattr(
        simons_cli,
        "screen_cointegrated",
        lambda *a, **k: pd.DataFrame([{
            "a": PAIR_SYMBOLS[0], "b": PAIR_SYMBOLS[1],
            "beta": 1.0, "alpha": 0.0,
            "pvalue": 0.01, "pvalue_fdr": 0.02, "fdr_pass": True,
            "half_life": 10.0,
        }]),
    )
    monkeypatch.setattr(
        simons_cli,
        "rolling_zscore",
        lambda series, window: pd.Series(
            np.r_[np.zeros(len(series) - 1), 3.0], index=series.index
        ),
    )
    monkeypatch.setattr(
        simons_cli, "_signal_snapshot_matches", lambda *a, **k: True
    )
    monkeypatch.setattr(simons_cli, "_outdir", lambda sub="": tmp_path / sub)
    cfg = _full_gate_config()
    args = SimpleNamespace(
        indexes=report["indexes"],
        lookback_days=400,
        corr=cfg["corr_threshold"],
        pvalue=cfg["pvalue_cutoff"],
        fdr=cfg["fdr_alpha"],
        hl_min=cfg["half_life_min"],
        hl_max=cfg["half_life_max"],
        formation=cfg["formation_days"],
        reestimate_days=cfg["reestimate_days"],
        z_window=cfg["z_window"],
        z_entry=cfg["z_entry"],
        z_exit=cfg["z_exit"],
        z_stop=cfg["z_stop"],
        max_hold=cfg["max_hold_days"],
        pair_stop_loss=cfg["pair_stop_loss"],
        cost_bps=cfg["cost_bps_one_side"],
        method=cfg["method"],
        price_basis=cfg["price_basis"],
        dedup_clusters=cfg["dedup_clusters"],
        top_n=cfg["top_n"],
        years=cfg["evaluation_years"],
        gate_report=str(report_path),
    )

    simons_cli.cmd_signal(args)

    stdout = capsys.readouterr().out
    output_files = list(tmp_path.rglob("signals_*.csv"))
    assert output_files
    csv_text = output_files[0].read_text(encoding="utf-8-sig")
    assert "LONG" not in stdout + csv_text
    assert "SHORT" not in stdout + csv_text
    output = pd.read_csv(output_files[0])
    assert set(output["signal"]) == {"NO_TRADE"}
    assert set(output["gate_report_run_id"]) == {report["run_id"]}
    assert "NO_TRADE_WEAK_EDGE" in stdout.splitlines()[0]


def test_signal_data_revision_downgrades_research_pass_and_hides_direction(
        monkeypatch, tmp_path, capsys):
    report = _valid_report("RESEARCH_PASS")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    dates = pd.bdate_range("2025-01-01", periods=280)
    panel = pd.DataFrame({
        PAIR_SYMBOLS[0]: np.linspace(10, 20, len(dates)),
        PAIR_SYMBOLS[1]: np.linspace(8, 18, len(dates)),
        PAIR_SYMBOLS[2]: np.linspace(12, 22, len(dates)),
        PAIR_SYMBOLS[3]: np.linspace(9, 19, len(dates)),
    }, index=dates)
    industry = pd.DataFrame({
        "symbol": PAIR_SYMBOLS,
        "industry": ["x", "x", "y", "y"],
    })
    monkeypatch.setattr(simons_cli, "_today_str", lambda: "20260710")
    monkeypatch.setattr(simons_cli, "ensure_login", lambda: "ok")
    monkeypatch.setattr(simons_cli, "build_price_panel", lambda *a, **k: panel)
    monkeypatch.setattr(simons_cli, "get_industry_map", lambda *a, **k: industry)
    monkeypatch.setattr(
        simons_cli, "_signal_snapshot_matches", lambda *a, **k: False
    )
    monkeypatch.setattr(
        simons_cli,
        "screen_cointegrated",
        lambda *a, **k: pd.DataFrame([{
            "a": PAIR_SYMBOLS[0], "b": PAIR_SYMBOLS[1],
            "beta": 1.0, "alpha": 0.0,
            "pvalue": 0.01, "pvalue_fdr": 0.02, "fdr_pass": True,
            "half_life": 10.0,
        }]),
    )
    monkeypatch.setattr(
        simons_cli,
        "rolling_zscore",
        lambda series, window: pd.Series(
            np.r_[np.zeros(len(series) - 1), 3.0], index=series.index
        ),
    )

    def fake_outdir(sub=""):
        path = tmp_path / sub
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(simons_cli, "_outdir", fake_outdir)
    args = SimpleNamespace(
        indexes=report["indexes"], years=5, lookback_days=400,
        corr=0.8, pvalue=0.05, fdr=0.05, hl_min=2.0, hl_max=60.0,
        top_n=40, dedup_clusters=20, max_pairs_per_symbol=1,
        formation=252, reestimate_days=60, z_window=60,
        z_entry=2.0, z_exit=0.5, z_stop=4.0, max_hold=60,
        pair_stop_loss=0.05, cost_bps=7.5, short_borrow_bps=800.0,
        method="static", price_basis="pre_adjusted",
        gate_report=str(report_path),
    )

    simons_cli.cmd_signal(args)

    output_text = capsys.readouterr().out
    csv_text = next(tmp_path.rglob("signals_*.csv")).read_text(
        encoding="utf-8-sig"
    )
    assert output_text.splitlines()[0].startswith("NO_TRADE_DATA_MISMATCH")
    assert "LONG" not in output_text + csv_text
    assert "SHORT" not in output_text + csv_text


def test_signal_missing_representative_price_fails_closed_without_screening(
        monkeypatch, tmp_path, capsys):
    report = _valid_report("RESEARCH_PASS")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    dates = pd.bdate_range("2025-01-01", periods=280)
    panel = pd.DataFrame({
        PAIR_SYMBOLS[0]: np.linspace(10, 20, len(dates)),
        PAIR_SYMBOLS[2]: np.linspace(12, 22, len(dates)),
        PAIR_SYMBOLS[3]: np.linspace(9, 19, len(dates)),
    }, index=dates)
    industry = pd.DataFrame({
        "symbol": PAIR_SYMBOLS,
        "industry": ["x", "x", "y", "y"],
    })
    monkeypatch.setattr(simons_cli, "_today_str", lambda: "20260710")
    monkeypatch.setattr(simons_cli, "ensure_login", lambda: "ok")
    monkeypatch.setattr(simons_cli, "build_price_panel", lambda *a, **k: panel)
    monkeypatch.setattr(simons_cli, "get_industry_map", lambda *a, **k: industry)
    monkeypatch.setattr(
        simons_cli,
        "screen_cointegrated",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("数据不匹配时不得继续筛选")
        ),
    )

    def fake_outdir(sub=""):
        path = tmp_path / sub
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(simons_cli, "_outdir", fake_outdir)
    args = SimpleNamespace(
        indexes=report["indexes"], years=5, lookback_days=400,
        corr=0.8, pvalue=0.05, fdr=0.05, hl_min=2.0, hl_max=60.0,
        top_n=40, dedup_clusters=20, max_pairs_per_symbol=1,
        formation=252, reestimate_days=60, z_window=60,
        z_entry=2.0, z_exit=0.5, z_stop=4.0, max_hold=60,
        pair_stop_loss=0.05, cost_bps=7.5, short_borrow_bps=800.0,
        method="static", price_basis="pre_adjusted",
        gate_report=str(report_path),
    )

    assert simons_cli.cmd_signal(args) == 0

    output_text = capsys.readouterr().out
    output = pd.read_csv(next(tmp_path.rglob("signals_*.csv")))
    assert output.empty
    assert output_text.splitlines()[0].startswith("NO_TRADE_DATA_MISMATCH")


def test_signal_only_evaluates_pairs_bound_to_report_representatives(
        monkeypatch, tmp_path):
    report = _valid_report("RESEARCH_PASS")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    dates = pd.bdate_range("2025-01-01", periods=280)
    panel = pd.DataFrame({
        name: np.linspace(10 + offset, 20 + offset, len(dates))
        for offset, name in enumerate(PAIR_SYMBOLS)
    }, index=dates)
    industry = pd.DataFrame({
        "symbol": PAIR_SYMBOLS,
        "industry": ["x", "x", "y", "y"],
    })
    observed = {}
    monkeypatch.setattr(simons_cli, "_today_str", lambda: "20260710")
    monkeypatch.setattr(simons_cli, "ensure_login", lambda: "ok")
    monkeypatch.setattr(simons_cli, "build_price_panel", lambda *a, **k: panel)
    monkeypatch.setattr(simons_cli, "get_industry_map", lambda *a, **k: industry)
    monkeypatch.setattr(
        simons_cli, "_signal_snapshot_matches", lambda *a, **k: True
    )

    def fake_screen(_panel, pairs, **kwargs):
        observed["pairs"] = list(pairs)
        return pd.DataFrame()

    monkeypatch.setattr(simons_cli, "screen_cointegrated", fake_screen)

    def fake_outdir(sub=""):
        path = tmp_path / sub
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(simons_cli, "_outdir", fake_outdir)
    args = SimpleNamespace(
        indexes=report["indexes"], years=5, lookback_days=400,
        corr=0.8, pvalue=0.05, fdr=0.05, hl_min=2.0, hl_max=60.0,
        top_n=40, dedup_clusters=20, max_pairs_per_symbol=1,
        formation=252, reestimate_days=60, z_window=60,
        z_entry=2.0, z_exit=0.5, z_stop=4.0, max_hold=60,
        pair_stop_loss=0.05, cost_bps=7.5, short_borrow_bps=800.0,
        method="static", price_basis="pre_adjusted",
        gate_report=str(report_path),
    )

    simons_cli.cmd_signal(args)

    assert observed["pairs"] == [
        (PAIR_SYMBOLS[0], PAIR_SYMBOLS[1]),
        (PAIR_SYMBOLS[2], PAIR_SYMBOLS[3]),
    ]
