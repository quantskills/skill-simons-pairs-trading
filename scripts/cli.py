"""Offline-reproducible command line workflow for A-share pairs research."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_layer import (  # noqa: E402
    PRICE_BASIS_TO_METHOD,
    _load_dotenv,
    build_price_panel,
    ensure_login,
    get_industry_map,
    get_universe,
    normalize_price_basis,
)
from pandadata_security import (  # noqa: E402
    PandaDataSecurityError,
    startup_health_check,
)
from stats_core import (  # noqa: E402
    SELECTION_METHOD,
    engle_granger,
    kalman_dynamic_beta,
    rolling_zscore,
    screen_cointegrated,
    select_pairs_for_backtest,
    static_spread,
)
from backtest import (  # noqa: E402
    BacktestConfig,
    Trade,
    backtest_single_pair,
    performance_metrics,
    plot_equity,
    plot_pair_diagnostic,
    strategy_gate,
    trades_stats,
)
from research_contract import (  # noqa: E402
    REQUIRED_GATE_CONFIG_KEYS,
    config_id as _contract_config_id,
    dataframe_sha256,
    identity_is_valid,
    make_run_identity as _contract_make_run_identity,
)


DEFAULT_INDEXES = ["000300.SH", "000905.SH"]
SKILL_ROOT = Path(__file__).resolve().parent.parent
_A_SHARE_SYMBOL = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
MIN_INDUSTRY_COVERAGE = 0.95


def _default_output_root() -> Path:
    configured = os.environ.get("SIMONS_OUT")
    if configured:
        return Path(configured).expanduser()
    if (SKILL_ROOT / "SKILL.md").is_file():
        return SKILL_ROOT / "outputs" / "simons_pairs"
    return Path.cwd() / "outputs" / "simons_pairs"


def _outdir(sub: str = "") -> Path:
    root = _default_output_root()
    path = root / sub if sub else root
    path.mkdir(parents=True, exist_ok=True)
    return path


def _today_str() -> str:
    value = os.environ.get("SIMONS_TODAY")
    if value:
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError as exc:
            raise ValueError("SIMONS_TODAY 必须为 YYYYMMDD") from exc
        return value
    return datetime.now().strftime("%Y%m%d")


def _period(days_back: int) -> tuple[str, str]:
    if int(days_back) <= 0:
        raise ValueError("回看天数必须为正整数")
    end = datetime.strptime(_today_str(), "%Y%m%d")
    start = end - timedelta(days=int(days_back))
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _research_periods(
        *, today: str, years: int, formation_days: int) -> dict[str, list[str]]:
    if int(years) <= 0:
        raise ValueError("years 必须为正整数")
    if int(formation_days) <= 0:
        raise ValueError("formation_days 必须为正整数")
    end = datetime.strptime(today, "%Y%m%d")
    evaluation_start = end - timedelta(days=int(years) * 365)
    warmup_calendar_days = max(
        400, int(math.ceil(int(formation_days) * 1.65))
    )
    data_start = evaluation_start - timedelta(days=warmup_calendar_days)
    return {
        "data_period": [
            data_start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        ],
        "formation_period": [
            data_start.strftime("%Y%m%d"),
            (evaluation_start - timedelta(days=1)).strftime("%Y%m%d"),
        ],
        "evaluation_period": [
            evaluation_start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        ],
    }


def _prepare_research_panel(
        panel: pd.DataFrame,
        *,
        evaluation_start: str,
        formation_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if panel.empty:
        raise RuntimeError("复权价格面板为空")
    try:
        boundary = pd.Timestamp(
            datetime.strptime(evaluation_start, "%Y%m%d")
        )
    except ValueError as exc:
        raise ValueError("evaluation_start 必须为 YYYYMMDD") from exc
    ordered = panel.copy()
    ordered.index = pd.to_datetime(ordered.index)
    ordered = ordered[~ordered.index.duplicated(keep="last")].sort_index()
    formation_available = ordered.loc[ordered.index < boundary]
    evaluation = ordered.loc[ordered.index >= boundary]
    if len(formation_available) < int(formation_days):
        raise RuntimeError(
            f"形成期需要 {formation_days} 个交易日，实际仅 "
            f"{len(formation_available)} 个"
        )
    if evaluation.empty:
        raise RuntimeError("评估期没有可用交易日")
    formation = formation_available.tail(int(formation_days))
    research = pd.concat([formation, evaluation])
    for key, value in panel.attrs.items():
        research.attrs[key] = value
        formation.attrs[key] = value
        evaluation.attrs[key] = value
    return research, formation, evaluation


def _historical_universe(
        indexes: Sequence[str],
        requested_date: str,
        max_lookback: int = 14,
) -> tuple[pd.DataFrame, str]:
    base = datetime.strptime(requested_date, "%Y%m%d")
    for offset in range(max_lookback + 1):
        date = (base - timedelta(days=offset)).strftime("%Y%m%d")
        universe = get_universe(list(indexes), date=date)
        if not universe.empty:
            return universe, date
    raise RuntimeError(
        f"{requested_date} 及此前 {max_lookback} 天没有可用历史指数成分"
    )


def _config_summary_from_args(args: argparse.Namespace) -> dict[str, Any]:
    price_basis = normalize_price_basis(getattr(
        args, "price_basis", "pre_adjusted"
    ))
    summary = {
        "formation_days": int(getattr(args, "formation", 252)),
        "reestimate_days": int(getattr(args, "reestimate_days", 60)),
        "corr_threshold": float(getattr(args, "corr", 0.80)),
        "pvalue_cutoff": float(getattr(args, "pvalue", 0.05)),
        "fdr_alpha": float(getattr(args, "fdr", 0.05)),
        "half_life_min": float(getattr(args, "hl_min", 2.0)),
        "half_life_max": float(getattr(args, "hl_max", 60.0)),
        "method": str(getattr(args, "method", "static")),
        "kalman_delta": float(
            getattr(args, "kalman_delta", 1e-4)
        ),
        "kalman_r": float(getattr(args, "kalman_r", 1e-3)),
        "z_window": int(getattr(args, "z_window", 60)),
        "z_entry": float(getattr(args, "z_entry", 2.0)),
        "z_exit": float(getattr(args, "z_exit", 0.5)),
        "z_stop": float(getattr(args, "z_stop", 4.0)),
        "max_hold_days": int(getattr(args, "max_hold", 60)),
        "pair_stop_loss": float(getattr(args, "pair_stop_loss", 0.05)),
        "cost_bps_one_side": float(getattr(args, "cost_bps", 7.5)),
        "short_borrow_bps_annual": float(
            getattr(args, "short_borrow_bps", 800.0)
        ),
        "price_basis": price_basis,
        "price_source_method": PRICE_BASIS_TO_METHOD[price_basis],
        "selection_method": SELECTION_METHOD,
        "dedup_clusters": int(getattr(args, "dedup_clusters", 20)),
        "max_pairs_per_symbol": int(
            getattr(args, "max_pairs_per_symbol", 1)
        ),
        "top_n": int(getattr(args, "top_n", 40)),
        "evaluation_years": int(getattr(args, "years", 5)),
        "signal_lookback_days": int(
            getattr(args, "lookback_days", 400)
        ),
    }
    _validate_research_config(summary)
    return summary


def _validate_research_config(config: Mapping[str, Any]) -> None:
    BacktestConfig(
        formation_days=int(config["formation_days"]),
        reestimate_days=int(config["reestimate_days"]),
        pvalue_cutoff=float(config["pvalue_cutoff"]),
        fdr_alpha=float(config["fdr_alpha"]),
        half_life_min=float(config["half_life_min"]),
        half_life_max=float(config["half_life_max"]),
        z_entry=float(config["z_entry"]),
        z_exit=float(config["z_exit"]),
        z_stop=float(config["z_stop"]),
        max_hold_days=int(config["max_hold_days"]),
        pair_stop_loss=float(config["pair_stop_loss"]),
        z_window=int(config["z_window"]),
        cost_bps_one_side=float(config["cost_bps_one_side"]),
        short_borrow_bps_annual=float(config["short_borrow_bps_annual"]),
        method=str(config["method"]),
        kalman_delta=float(config["kalman_delta"]),
        kalman_r=float(config["kalman_r"]),
    )
    if not 0.0 <= float(config["corr_threshold"]) <= 1.0:
        raise ValueError("corr_threshold 必须在 [0, 1] 内")
    for key in (
        "dedup_clusters", "max_pairs_per_symbol", "top_n",
        "evaluation_years", "signal_lookback_days",
    ):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if int(config["evaluation_years"]) < 5:
        raise ValueError("evaluation_years 至少为 5 年")


def _config_id(indexes: Sequence[str], config_summary: Mapping[str, Any]) -> str:
    return _contract_config_id(indexes, config_summary)


def _make_run_identity(**kwargs) -> dict[str, str]:
    return _contract_make_run_identity(**kwargs)


def _backtest_subdir(today: str, config_id: str) -> str:
    return f"backtest_{today}_{config_id[:12]}"


def _signal_filename(today: str, config_id: str) -> str:
    return f"signals_{today}_{config_id[:12]}.csv"


def _signal_industry_frame(
        industry_map: pd.DataFrame,
        symbols: Sequence[str]) -> pd.DataFrame:
    required = {"symbol", "industry"}
    if not required.issubset(industry_map.columns):
        raise ValueError("行业数据必须包含 symbol 和 industry")
    wanted = sorted({str(symbol) for symbol in symbols})
    frame = industry_map[["symbol", "industry"]].copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["industry"] = frame["industry"].astype(str)
    frame = frame.loc[frame["symbol"].isin(wanted)]
    frame = frame.drop_duplicates("symbol", keep="last")
    frame = frame.set_index("symbol").reindex(wanted).reset_index()
    return frame


def _industry_coverage_summary(
        industry_map: pd.DataFrame,
        symbols: Sequence[str],
        *,
        minimum: float = MIN_INDUSTRY_COVERAGE,
) -> dict[str, Any]:
    if not 0.0 < float(minimum) <= 1.0:
        raise ValueError("行业数据最低覆盖率必须在 (0, 1] 之间")
    wanted = sorted({str(symbol) for symbol in symbols})
    if not wanted:
        raise ValueError("行业覆盖率检查至少需要一只股票")
    frame = _signal_industry_frame(industry_map, wanted)
    values = frame["industry"].fillna("").astype(str).str.strip()
    resolved_mask = values.ne("") & values.str.upper().ne("UNKNOWN")
    resolved = int(resolved_mask.sum())
    requested = len(wanted)
    return {
        "requested": requested,
        "resolved": resolved,
        "unknown": requested - resolved,
        "coverage": resolved / requested,
        "minimum": float(minimum),
    }


def _require_industry_coverage(
        industry_map: pd.DataFrame,
        symbols: Sequence[str],
        *,
        minimum: float = MIN_INDUSTRY_COVERAGE,
) -> dict[str, Any]:
    quality = _industry_coverage_summary(
        industry_map, symbols, minimum=minimum
    )
    if quality["coverage"] + 1e-12 < quality["minimum"]:
        raise RuntimeError(
            "行业数据覆盖率不足："
            f"{quality['resolved']}/{quality['requested']} "
            f"({quality['coverage']:.2%})，最低要求 "
            f"{quality['minimum']:.2%}；已停止回测以避免失真"
        )
    return quality


def _make_signal_snapshot(
        *,
        panel: pd.DataFrame,
        industry_map: pd.DataFrame,
        symbols: Sequence[str],
        end_date: str,
        lookback_days: int,
        price_basis: str,
) -> dict[str, Any]:
    if int(lookback_days) <= 0:
        raise ValueError("lookback_days 必须为正整数")
    basis = normalize_price_basis(price_basis)
    end = pd.Timestamp(datetime.strptime(end_date, "%Y%m%d"))
    start = end - pd.Timedelta(days=int(lookback_days))
    wanted = sorted({str(symbol) for symbol in symbols})
    prices = panel.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.loc[(prices.index >= start) & (prices.index <= end)]
    prices.columns = [str(column) for column in prices.columns]
    absent = set(wanted) - set(prices.columns)
    prices = prices.reindex(columns=wanted).sort_index()
    observation_counts = {
        symbol: int(prices[symbol].notna().sum())
        for symbol in wanted
    }
    all_empty = {
        symbol for symbol, count in observation_counts.items() if count == 0
    }
    missing = sorted(absent | all_empty)
    industries = _signal_industry_frame(industry_map, wanted)
    return {
        "lookback_days": int(lookback_days),
        "window_start": start.strftime("%Y%m%d"),
        "window_end": end.strftime("%Y%m%d"),
        "price_basis": basis,
        "symbols": wanted,
        "data_complete": not missing,
        "missing_price_symbols": missing,
        "price_observation_counts": observation_counts,
        "price_panel_sha256": dataframe_sha256(prices),
        "industry_map_sha256": dataframe_sha256(industries),
        "n_price_rows": int(len(prices)),
    }


def _signal_snapshot_matches(
        expected: Mapping[str, Any],
        *,
        panel: pd.DataFrame,
        industry_map: pd.DataFrame,
        end_date: str,
) -> bool:
    if expected.get("data_complete") is not True:
        return False
    try:
        actual = _make_signal_snapshot(
            panel=panel,
            industry_map=industry_map,
            symbols=expected["symbols"],
            end_date=end_date,
            lookback_days=int(expected["lookback_days"]),
            price_basis=str(expected["price_basis"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return actual.get("data_complete") is True and actual == dict(expected)


def _load_json_report(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return bool(np.isclose(
                float(actual), float(expected), rtol=0.0, atol=1e-12
            ))
        except (TypeError, ValueError):
            return False
    return actual == expected


def _load_signal_gate(
        today: str,
        report_path=None,
        expected_indexes=None,
        expected_config=None,
) -> str:
    if report_path is None:
        if expected_indexes is None or expected_config is None:
            return "NO_TRADE_MISSING_BACKTEST"
        cid = _config_id(expected_indexes, expected_config)
        path = _outdir(_backtest_subdir(today, cid)) / "report.json"
    else:
        path = Path(report_path)
    if not path.exists():
        return "NO_TRADE_MISSING_BACKTEST"
    report = _load_json_report(path)
    if report is None:
        return "NO_TRADE_INVALID_BACKTEST"

    periods = report.get("periods", {})
    evaluation_period = periods.get(
        "evaluation_period", report.get("evaluation_period", [])
    )
    if not isinstance(evaluation_period, list) or len(evaluation_period) != 2:
        return "NO_TRADE_INVALID_BACKTEST"
    if evaluation_period[-1] != today:
        return "NO_TRADE_STALE_BACKTEST"

    if expected_indexes is not None and sorted(report.get("indexes", [])) != \
            sorted(expected_indexes):
        return "NO_TRADE_CONFIG_MISMATCH"
    actual_config = report.get("config_summary")
    if not isinstance(actual_config, dict):
        return "NO_TRADE_CONFIG_MISMATCH"
    if any(key not in actual_config for key in REQUIRED_GATE_CONFIG_KEYS):
        return "NO_TRADE_CONFIG_MISMATCH"
    if expected_config is not None:
        if any(key not in expected_config for key in REQUIRED_GATE_CONFIG_KEYS):
            return "NO_TRADE_CONFIG_MISMATCH"
        if any(
            not _values_match(actual_config.get(key), expected_config[key])
            for key in REQUIRED_GATE_CONFIG_KEYS
        ):
            return "NO_TRADE_CONFIG_MISMATCH"

    try:
        evaluation_start = datetime.strptime(evaluation_period[0], "%Y%m%d")
        evaluation_end = datetime.strptime(evaluation_period[1], "%Y%m%d")
        expected_years = int(actual_config["evaluation_years"])
    except (TypeError, ValueError, KeyError):
        return "NO_TRADE_INVALID_BACKTEST"
    if expected_years < 5:
        return "NO_TRADE_CONFIG_MISMATCH"
    if (evaluation_end - evaluation_start).days < expected_years * 365:
        return "NO_TRADE_CONFIG_MISMATCH"
    if float(actual_config["cost_bps_one_side"]) < 7.5:
        return "NO_TRADE_CONFIG_MISMATCH"
    if actual_config["price_basis"] not in PRICE_BASIS_TO_METHOD:
        return "NO_TRADE_CONFIG_MISMATCH"
    if PRICE_BASIS_TO_METHOD[actual_config["price_basis"]] != \
            actual_config["price_source_method"]:
        return "NO_TRADE_CONFIG_MISMATCH"
    if not identity_is_valid(report):
        return "NO_TRADE_INVALID_BACKTEST"

    gate = str(report.get("strategy_gate", ""))
    allowed = {
        "RESEARCH_PASS",
        "NO_TRADE_NEGATIVE_EDGE",
        "NO_TRADE_INSUFFICIENT_SAMPLE",
        "NO_TRADE_INVALID_METRICS",
        "NO_TRADE_WEAK_EDGE",
    }
    if gate not in allowed:
        return "NO_TRADE_INVALID_BACKTEST"
    try:
        computed_gate = strategy_gate(report["performance"], report["trades"])
    except (KeyError, TypeError, ValueError):
        return "NO_TRADE_INVALID_BACKTEST"
    return gate if gate == computed_gate else "NO_TRADE_INVALID_BACKTEST"


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _safe_json(dict(value)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def _formation_diagnostics_markdown(diagnostics: Mapping[str, Any]) -> str:
    ratios = diagnostics.get("pc_explained_variance_ratio", [])
    ratio_text = ", ".join(f"{float(x):.2%}" for x in ratios[:5]) or "无"
    condition = diagnostics.get("condition_number")
    condition_text = "不可计算" if condition is None else f"{float(condition):.4g}"
    industry_quality = diagnostics.get("industry_data_quality", {})
    industry_text = (
        f"{int(industry_quality.get('resolved', 0))}/"
        f"{int(industry_quality.get('requested', 0))} "
        f"({float(industry_quality.get('coverage', 0.0)):.2%})"
    )
    return "\n".join([
        "# 形成期共线诊断",
        "",
        f"- 方法：{diagnostics.get('selection_method', SELECTION_METHOD)}",
        f"- 行业数据覆盖：{industry_text}",
        f"- 候选配对：{diagnostics.get('n_candidates', 0)}",
        f"- 代表配对：{diagnostics.get('n_representatives', 0)}",
        f"- 条件数：{condition_text}",
        f"- 有效秩：{float(diagnostics.get('effective_rank', 0.0)):.3f}",
        f"- 最大绝对相关："
        f"{float(diagnostics.get('max_abs_pair_correlation', 0.0)):.3f}",
        f"- 单股最大重复次数：{int(diagnostics.get('max_symbol_reuse', 0))}",
        f"- 有效独立对数："
        f"{float(diagnostics.get('effective_independent_pairs', 0.0)):.2f}",
        f"- 前五主成分解释率：{ratio_text}",
        "",
        "仅使用形成期的两腿收益代理完成聚类，评估期数据未参与选对。",
    ]) + "\n"


def _build_report(
        *,
        config_summary: Mapping[str, Any],
        periods: Mapping[str, Sequence[str]],
        observed_periods: Mapping[str, Sequence[str]],
        universe_date: str,
        indexes: Sequence[str],
        coint_df: pd.DataFrame,
        selection_diagnostics: Mapping[str, Any],
        perf: Mapping[str, Any],
        tstats: Mapping[str, Any],
        gate: str,
        data_fingerprints: Mapping[str, str],
        signal_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    representative_pairs = (
        coint_df["a"].astype(str) + "~" + coint_df["b"].astype(str)
    ).tolist() if not coint_df.empty else []
    identity = _make_run_identity(
        indexes=indexes,
        config_summary=config_summary,
        periods=periods,
        data_fingerprints=data_fingerprints,
        representative_pairs=representative_pairs,
        research_results={
            "strategy_gate": gate,
            "performance": dict(perf),
            "trades": dict(tstats),
        },
        signal_snapshot=signal_snapshot,
    )
    return {
        **identity,
        "period": list(periods["evaluation_period"]),
        "periods": {key: list(value) for key, value in periods.items()},
        "data_period": list(periods["data_period"]),
        "formation_period": list(periods["formation_period"]),
        "evaluation_period": list(periods["evaluation_period"]),
        "observed_periods": {
            key: list(value) for key, value in observed_periods.items()
        },
        "universe_date": universe_date,
        "indexes": list(indexes),
        "config": dict(config_summary),
        "config_summary": dict(config_summary),
        "data_fingerprints": dict(data_fingerprints),
        "signal_snapshot": dict(signal_snapshot),
        "representative_pairs": representative_pairs,
        "n_pairs": int(len(coint_df)),
        "selection_diagnostics": dict(selection_diagnostics),
        "performance": dict(perf),
        "trades": dict(tstats),
        "strategy_gate": gate,
        "research_status": (
            "RESEARCH_ONLY" if gate == "RESEARCH_PASS" else "NO_TRADE"
        ),
        "limitations": [
            "A股个股做空和融券券源可得性未验证；结果始终只用于研究。",
            "已计入可配置融券年费，但实际费率、召回和强平规则未完整模拟。",
            "T+1、涨跌停、停牌和成交冲击未完整模拟。",
            "行业接口没有历史日期参数，历史研究使用查询时行业分类。",
            "股指期货不能精确替代个股空腿，未把它视为等价执行方案。",
            "供应商可能修订历史复权数据；数据指纹用于识别具体输入版本。",
        ],
    }


def _fmt_metric(value: Any, pattern: str = ".2f") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return format(number, pattern) if np.isfinite(number) else "N/A"


def _render_report(report: Mapping[str, Any], coint_df: pd.DataFrame) -> str:
    perf = report["performance"]
    trade = report["trades"]
    status = report["research_status"]
    gate = report["strategy_gate"]
    heading = f"# {status} — A股配对研究报告"
    markdown = [
        heading,
        "",
        f"> 门控结论：**{gate}**。"
        "即使为 RESEARCH_ONLY，也不表示可以实盘。",
        "",
        f"- Run ID：`{report['run_id']}`",
        f"- Config ID：`{report['config_id']}`",
        f"- 结果摘要：`{report['results_sha256']}`",
        f"- 共享信号窗口摘要：`{report['signal_snapshot_sha256']}`",
        f"- 信号窗口价格完整："
        f"{'是' if report['signal_snapshot'].get('data_complete') else '否'}",
        f"- 信号窗口缺失股票："
        f"{', '.join(report['signal_snapshot'].get('missing_price_symbols', [])) or '无'}",
        f"- 代码版本：`{report['code_version']}`",
        f"- 指数池：{'+'.join(report['indexes'])}",
        f"- 数据区间：{' → '.join(report['data_period'])}",
        f"- 形成期：{' → '.join(report['formation_period'])}",
        f"- 样本外评估期：{' → '.join(report['evaluation_period'])}",
        f"- 历史成分日期：{report['universe_date']}",
        f"- 价格口径：{report['config_summary']['price_basis']} "
        f"({report['config_summary']['price_source_method']})",
        f"- FDR：Benjamini–Hochberg "
        f"q≤{report['config_summary']['fdr_alpha']}",
        f"- 形成期去重：{report['config_summary']['selection_method']}",
        f"- 价格面板指纹："
        f"`{report['data_fingerprints']['price_panel_sha256']}`",
        f"- 历史股票池指纹："
        f"`{report['data_fingerprints']['universe_sha256']}`",
        f"- 行业映射指纹："
        f"`{report['data_fingerprints']['industry_map_sha256']}`",
        f"- 行业数据覆盖："
        f"{report['selection_diagnostics']['industry_data_quality']['resolved']}/"
        f"{report['selection_diagnostics']['industry_data_quality']['requested']} "
        f"({report['selection_diagnostics']['industry_data_quality']['coverage']:.2%})",
        "",
        "## 绩效",
        "",
        "| 夏普 | 年化收益 | 年化波动 | 最大回撤 | 交易日 |",
        "|---:|---:|---:|---:|---:|",
        f"| {_fmt_metric(perf.get('sharpe'))} | "
        f"{_fmt_metric(perf.get('annual_ret'), '.2%')} | "
        f"{_fmt_metric(perf.get('annual_vol'), '.2%')} | "
        f"{_fmt_metric(perf.get('max_dd'), '.2%')} | "
        f"{int(perf.get('n_days', 0))} |",
        "",
        "## 交易与成本",
        "",
        f"- 交易笔数：{int(trade.get('n_trades', 0))}",
        f"- 胜率：{_fmt_metric(trade.get('win_rate'), '.1%')}",
        f"- 平均净收益：{_fmt_metric(trade.get('avg_ret'), '.4f')}",
        f"- 平均持有日：{_fmt_metric(trade.get('avg_hold'), '.1f')}",
        f"- 单边换手成本："
        f"{report['config_summary']['cost_bps_one_side']} bp",
        f"- 融券年费假设："
        f"{report['config_summary']['short_borrow_bps_annual']} bp",
        "",
        "## 重要限制",
        "",
    ]
    markdown.extend(f"- {item}" for item in report["limitations"])
    if int(perf.get("n_days", 0)) > 0:
        markdown.extend(["", "## 净值曲线", "", "![equity](equity.png)"])
    if not coint_df.empty:
        markdown.extend([
            "",
            "## 形成期代表配对",
            "",
            coint_df.head(20).to_markdown(index=False),
        ])
    return "\n".join(markdown) + "\n"


def _validated_report_pairs(
        values: Any, *, limit: int
) -> list[tuple[str, str]] | None:
    """Validate and bound report-controlled symbols before any data call."""
    if (
        not isinstance(values, list)
        or int(limit) <= 0
        or len(values) > int(limit)
    ):
        return None
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        if not isinstance(value, str):
            return None
        parts = value.split("~")
        if (
            len(parts) != 2
            or parts[0] == parts[1]
            or not all(_A_SHARE_SYMBOL.fullmatch(item) for item in parts)
        ):
            return None
        canonical = tuple(sorted((parts[0], parts[1])))
        if canonical in seen:
            return None
        seen.add(canonical)
        pairs.append((parts[0], parts[1]))
    return pairs


def _signal_spread(
        panel: pd.DataFrame,
        row: Any,
        config: Mapping[str, Any],
) -> pd.Series:
    """Use the same spread model selected for the bound backtest."""
    if config["method"] == "kalman":
        dynamic = kalman_dynamic_beta(
            panel[row.a],
            panel[row.b],
            delta=float(config["kalman_delta"]),
            r_var=float(config["kalman_r"]),
        )
        if dynamic.empty:
            return pd.Series(dtype=float)
        return dynamic["spread"]
    return static_spread(
        panel[row.a],
        panel[row.b],
        float(row.alpha),
        float(row.beta),
    )


def cmd_signal(args) -> int:
    config = _config_summary_from_args(args)
    today = _today_str()
    cid = _config_id(args.indexes, config)
    report_path = Path(args.gate_report) if getattr(args, "gate_report", None) \
        else _outdir(_backtest_subdir(today, cid)) / "report.json"
    gate = _load_signal_gate(
        today,
        report_path,
        expected_indexes=args.indexes,
        expected_config=config,
    )
    report = _load_json_report(report_path) or {}
    verified_report = identity_is_valid(report)
    report_run_id = str(report.get("run_id", "")) if verified_report else ""
    validated_pairs = (
        _validated_report_pairs(
            report.get("representative_pairs", []),
            limit=config["top_n"],
        )
        if verified_report else []
    )
    if validated_pairs is None:
        gate = "NO_TRADE_INVALID_BACKTEST"
        pairs: list[tuple[str, str]] = []
    else:
        pairs = validated_pairs
    bound_pair_names = [f"{a}~{b}" for a, b in pairs]

    panel = pd.DataFrame()
    industry = pd.DataFrame(columns=["symbol", "industry"])
    data_match = False
    screened = pd.DataFrame()
    if verified_report and pairs:
        symbols = sorted({symbol for pair in pairs for symbol in pair})
        start, end = _period(config["signal_lookback_days"])
        ensure_login()
        panel = build_price_panel(
            symbols,
            start,
            end,
            field="close",
            price_basis=config["price_basis"],
        )
        industry = get_industry_map(symbols, level="L1")
        industry_quality = _industry_coverage_summary(industry, symbols)
        if industry_quality["coverage"] >= industry_quality["minimum"]:
            data_match = _signal_snapshot_matches(
                report.get("signal_snapshot", {}),
                panel=panel,
                industry_map=industry,
                end_date=today,
            )
        if not data_match:
            gate = "NO_TRADE_DATA_MISMATCH"
        else:
            screened = screen_cointegrated(
                panel,
                pairs,
                pvalue_cutoff=config["pvalue_cutoff"],
                fdr_alpha=config["fdr_alpha"],
                half_life_range=(
                    config["half_life_min"], config["half_life_max"]
                ),
            )
            if not screened.empty:
                bound_set = set(bound_pair_names)
                names = (
                    screened["a"].astype(str) + "~"
                    + screened["b"].astype(str)
                )
                screened = screened.loc[
                    names.isin(bound_set)
                ].reset_index(drop=True)
    elif verified_report and gate == "RESEARCH_PASS":
        gate = "NO_TRADE_NO_BOUND_PAIRS"

    can_show_direction = gate == "RESEARCH_PASS" and data_match

    rows: list[dict[str, Any]] = []
    for row in screened.itertuples(index=False):
        spread = _signal_spread(panel, row, config)
        z = rolling_zscore(spread, config["z_window"]).dropna()
        if z.empty:
            continue
        current_z = float(z.iloc[-1])
        base = {
            **row._asdict(),
            "current_z": current_z,
            "spread_last_date": str(pd.Timestamp(z.index[-1]).date()),
            "strategy_gate": gate,
            "research_status": (
                "RESEARCH_ONLY" if can_show_direction else "NO_TRADE"
            ),
            "gate_report_run_id": report_run_id,
            "gate_report_verified": bool(verified_report),
            "signal_data_match": bool(data_match),
            "price_basis": config["price_basis"],
            "config_id": cid,
        }
        if can_show_direction:
            if current_z > config["z_entry"]:
                base["signal"] = "SHORT_SPREAD"
            elif current_z < -config["z_entry"]:
                base["signal"] = "LONG_SPREAD"
            else:
                base["signal"] = "WATCH"
        else:
            base["signal"] = "NO_TRADE"
        rows.append(base)

    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.sort_values(
            "current_z", key=lambda values: -values.abs()
        )
    else:
        output = pd.DataFrame(columns=[
            "a", "b", "pvalue", "pvalue_fdr", "current_z", "signal",
            "strategy_gate", "research_status", "gate_report_run_id",
            "gate_report_verified", "signal_data_match", "price_basis",
            "config_id",
        ])
    output_path = _outdir() / _signal_filename(today, cid)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(
        f"{'RESEARCH_ONLY' if can_show_direction else gate} | "
        f"研究门控={gate}"
    )
    if can_show_direction:
        active = output.loc[output["signal"].isin(
            ["LONG_SPREAD", "SHORT_SPREAD"]
        )]
        print(f"理论研究方向：{len(active)} 条；仍不可直接实盘")
        if not active.empty:
            print(active[[
                "a", "b", "pvalue", "pvalue_fdr", "half_life",
                "current_z", "signal",
            ]].to_string(index=False))
    else:
        print(f"方向已隐藏；保存 {len(output)} 条无方向诊断记录")
    print(f"输出：{output_path}")
    return 0


def cmd_backtest(args) -> int:
    config = _config_summary_from_args(args)
    today = _today_str()
    requested_periods = _research_periods(
        today=today,
        years=config["evaluation_years"],
        formation_days=config["formation_days"],
    )
    data_start, end = requested_periods["data_period"]
    evaluation_start = requested_periods["evaluation_period"][0]

    ensure_login()
    universe, universe_date = _historical_universe(args.indexes, data_start)
    symbols = sorted(set(universe["symbol"].astype(str)))
    if len(symbols) < 2:
        raise RuntimeError("历史指数池不足两只股票")
    panel = build_price_panel(
        symbols,
        data_start,
        end,
        field="close",
        price_basis=config["price_basis"],
    )
    industry = get_industry_map(symbols, level="L1")
    industry_quality = _require_industry_coverage(industry, symbols)
    research_panel, formation_panel, evaluation_panel = \
        _prepare_research_panel(
            panel,
            evaluation_start=evaluation_start,
            formation_days=config["formation_days"],
        )
    requested_periods["formation_period"] = [
        pd.Timestamp(formation_panel.index.min()).strftime("%Y%m%d"),
        pd.Timestamp(formation_panel.index.max()).strftime("%Y%m%d"),
    ]
    observed_periods = {
        "data_period": [
            pd.Timestamp(research_panel.index.min()).strftime("%Y%m%d"),
            pd.Timestamp(research_panel.index.max()).strftime("%Y%m%d"),
        ],
        "formation_period": list(requested_periods["formation_period"]),
        "evaluation_period": [
            pd.Timestamp(evaluation_panel.index.min()).strftime("%Y%m%d"),
            pd.Timestamp(evaluation_panel.index.max()).strftime("%Y%m%d"),
        ],
    }

    selected, selection_diagnostics = select_pairs_for_backtest(
        research_panel,
        industry,
        formation_days=config["formation_days"],
        corr_threshold=config["corr_threshold"],
        pvalue_cutoff=config["pvalue_cutoff"],
        fdr_alpha=config["fdr_alpha"],
        half_life_range=(
            config["half_life_min"], config["half_life_max"]
        ),
        top_n=config["top_n"],
        dedup_clusters=config["dedup_clusters"],
        max_pairs_per_symbol=config["max_pairs_per_symbol"],
        return_diagnostics=True,
    )
    selection_diagnostics = {
        **selection_diagnostics,
        "industry_data_quality": industry_quality,
    }
    cfg = BacktestConfig(
        formation_days=config["formation_days"],
        reestimate_days=config["reestimate_days"],
        pvalue_cutoff=config["pvalue_cutoff"],
        fdr_alpha=config["fdr_alpha"],
        half_life_min=config["half_life_min"],
        half_life_max=config["half_life_max"],
        z_entry=config["z_entry"],
        z_exit=config["z_exit"],
        z_stop=config["z_stop"],
        max_hold_days=config["max_hold_days"],
        pair_stop_loss=config["pair_stop_loss"],
        z_window=config["z_window"],
        cost_bps_one_side=config["cost_bps_one_side"],
        short_borrow_bps_annual=config["short_borrow_bps_annual"],
        method=config["method"],
        kalman_delta=config["kalman_delta"],
        kalman_r=config["kalman_r"],
    )

    all_trades: list[Trade] = []
    pair_daily: dict[str, pd.Series] = {}
    for row in selected.itertuples(index=False):
        trades, daily = backtest_single_pair(
            research_panel, (row.a, row.b), cfg
        )
        all_trades.extend(trades)
        pair_daily[f"{row.a}~{row.b}"] = daily["ret"].reindex(
            evaluation_panel.index, fill_value=0.0
        )
    if pair_daily:
        pair_daily_frame = pd.DataFrame(pair_daily).fillna(0.0)
        combined = pair_daily_frame.mean(axis=1)
    else:
        pair_daily_frame = pd.DataFrame(index=evaluation_panel.index)
        combined = pd.Series(
            0.0, index=evaluation_panel.index, name="daily_ret"
        )
    perf = performance_metrics(combined)
    tstats = trades_stats(all_trades)
    gate = strategy_gate(perf, tstats)

    fingerprints = {
        "price_panel_sha256": dataframe_sha256(research_panel),
        "universe_sha256": dataframe_sha256(
            universe.sort_values(list(universe.columns)).reset_index(drop=True)
        ),
        "industry_map_sha256": dataframe_sha256(
            industry.sort_values(list(industry.columns)).reset_index(drop=True)
        ),
    }
    representative_symbols = sorted(set(
        selected.get("a", pd.Series(dtype=str)).astype(str).tolist()
        + selected.get("b", pd.Series(dtype=str)).astype(str).tolist()
    ))
    signal_snapshot = _make_signal_snapshot(
        panel=research_panel,
        industry_map=industry,
        symbols=representative_symbols,
        end_date=end,
        lookback_days=config["signal_lookback_days"],
        price_basis=config["price_basis"],
    )
    report = _build_report(
        config_summary=config,
        periods=requested_periods,
        observed_periods=observed_periods,
        universe_date=universe_date,
        indexes=args.indexes,
        coint_df=selected,
        selection_diagnostics=selection_diagnostics,
        perf=perf,
        tstats=tstats,
        gate=gate,
        data_fingerprints=fingerprints,
        signal_snapshot=signal_snapshot,
    )
    outdir = _outdir(_backtest_subdir(today, report["config_id"]))

    trade_columns = [item.name for item in fields(Trade)]
    trades_frame = pd.DataFrame(
        [trade.__dict__ for trade in all_trades], columns=trade_columns
    )
    selected.to_csv(
        outdir / "formation_representatives.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected.to_csv(outdir / "pairs.csv", index=False, encoding="utf-8-sig")
    trades_frame.to_csv(
        outdir / "trades.csv", index=False, encoding="utf-8-sig"
    )
    combined.rename("daily_ret").to_frame().to_csv(
        outdir / "daily_returns.csv", encoding="utf-8-sig"
    )
    pair_daily_frame.to_csv(
        outdir / "pair_daily_returns.csv", encoding="utf-8-sig"
    )
    _write_json(
        outdir / "formation_collinearity.json", selection_diagnostics
    )
    (outdir / "formation_collinearity.md").write_text(
        _formation_diagnostics_markdown(selection_diagnostics),
        encoding="utf-8",
    )
    _write_json(outdir / "report.json", report)
    (outdir / "report.md").write_text(
        _render_report(report, selected), encoding="utf-8"
    )
    plot_equity(
        combined,
        str(outdir / "equity.png"),
        title=f"A-share Pairs Research — {config['method']}",
    )

    print(
        f"{report['research_status']} | 研究门控={gate} | "
        f"run_id={report['run_id']}"
    )
    print(f"代表配对={len(selected)}，交易={tstats['n_trades']}")
    print(f"输出：{outdir}")
    return 0


def cmd_diagnose(args) -> int:
    pieces = [item.strip().upper() for item in args.pair.split(",")]
    if len(pieces) != 2 or not all(pieces) or pieces[0] == pieces[1]:
        raise ValueError("--pair 必须是两个不同代码，以逗号分隔")
    if not all(_A_SHARE_SYMBOL.fullmatch(item) for item in pieces):
        raise ValueError(
            "证券代码必须使用 6 位数字加 .SH、.SZ 或 .BJ"
        )
    ensure_login()
    start, end = _period(args.lookback_days)
    basis = normalize_price_basis(args.price_basis)
    panel = build_price_panel(
        pieces, start, end, field="close", price_basis=basis
    )
    result = engle_granger(
        panel[pieces[0]], panel[pieces[1]], pieces[0], pieces[1]
    )
    if result is None:
        print("NO_TRADE | 样本不足或关系无效")
        return 0
    print(
        f"RESEARCH_ONLY | p={result.pvalue:.4f}, beta={result.beta:.4f}, "
        f"half_life={result.half_life:.1f}, price_basis={basis}"
    )
    output = _outdir() / f"diagnose_{pieces[0]}_{pieces[1]}_{basis}.png"
    plot_pair_diagnostic(
        panel,
        (pieces[0], pieces[1]),
        result.alpha,
        result.beta,
        args.z_window,
        str(output),
    )
    print(f"输出：{output}")
    return 0


def cmd_health_check(args) -> int:
    """Check runtime, SDK, provider HTTPS, and credential readiness."""
    _load_dotenv()
    try:
        report = startup_health_check(
            timeout=args.timeout,
            require_credentials=not args.transport_only,
        )
    except PandaDataSecurityError as exc:
        print(f"[FAIL] 启动健康检查未通过：{exc}")
        return 1
    except Exception as exc:
        print(f"[FAIL] 启动健康检查失败（{type(exc).__name__}）")
        return 1

    print("OK  启动健康检查通过")
    print(f"Python：{report['python']}")
    print(f"HTTPS 端点：{report['endpoint']}")
    print(f"TLS：{report['tls_version']}")
    print(f"SDK：{report['sdk']}")
    print(f"凭证方式：{report['auth_mode']}")
    return 0


def cmd_check_login(_args) -> int:
    try:
        ensure_login()
    except Exception as exc:
        print(f"[FAIL] pandadata 登录失败（{type(exc).__name__}）")
        return 1
    print("OK  pandadata 登录成功")
    return 0


def _add_research_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--indexes", nargs="+", default=DEFAULT_INDEXES)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--corr", type=float, default=0.80)
    parser.add_argument("--pvalue", type=float, default=0.05)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--hl-min", type=float, default=2.0)
    parser.add_argument("--hl-max", type=float, default=60.0)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--dedup-clusters", type=int, default=20)
    parser.add_argument("--max-pairs-per-symbol", type=int, default=1)
    parser.add_argument("--formation", type=int, default=252)
    parser.add_argument("--reestimate-days", type=int, default=60)
    parser.add_argument("--z-window", type=int, default=60)
    parser.add_argument("--z-entry", type=float, default=2.0)
    parser.add_argument("--z-exit", type=float, default=0.5)
    parser.add_argument("--z-stop", type=float, default=4.0)
    parser.add_argument("--max-hold", type=int, default=60)
    parser.add_argument("--pair-stop-loss", type=float, default=0.05)
    parser.add_argument("--cost-bps", type=float, default=7.5)
    parser.add_argument("--short-borrow-bps", type=float, default=800.0)
    parser.add_argument(
        "--method", choices=["static", "kalman"], default="static"
    )
    parser.add_argument("--kalman-delta", type=float, default=1e-4)
    parser.add_argument("--kalman-r", type=float, default=1e-3)
    parser.add_argument(
        "--price-basis",
        choices=["pre_adjusted", "post_adjusted"],
        default="pre_adjusted",
    )


def _health_timeout(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.1 <= parsed <= 30.0:
        raise argparse.ArgumentTypeError("value must be between 0.1 and 30")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simons-pairs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    health = sub.add_parser(
        "health-check",
        help="检查 Python、SDK、服务方 HTTPS 和凭证配置",
    )
    health.add_argument("--timeout", type=_health_timeout, default=5.0)
    health.add_argument(
        "--transport-only",
        action="store_true",
        help="只检查运行环境与 HTTPS，不要求已配置凭证",
    )
    health.set_defaults(func=cmd_health_check)

    signal = sub.add_parser("signal", help="生成受同日回测门控的研究信号")
    _add_research_args(signal)
    signal.add_argument("--lookback-days", type=int, default=400)
    signal.add_argument("--gate-report")
    signal.set_defaults(func=cmd_signal)

    backtest = sub.add_parser("backtest", help="运行完整样本外滚动研究")
    _add_research_args(backtest)
    backtest.set_defaults(func=cmd_backtest)

    diagnose = sub.add_parser("diagnose", help="生成单对研究诊断图")
    diagnose.add_argument("--pair", required=True)
    diagnose.add_argument("--lookback-days", type=int, default=500)
    diagnose.add_argument("--z-window", type=int, default=60)
    diagnose.add_argument(
        "--price-basis",
        choices=["pre_adjusted", "post_adjusted"],
        default="pre_adjusted",
    )
    diagnose.set_defaults(func=cmd_diagnose)

    check = sub.add_parser("check-login", help="只检查登录，不显示令牌")
    check.set_defaults(func=cmd_check_login)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd not in {"health-check", "check-login"}:
            ensure_login()
        return int(args.func(args) or 0)
    except (KeyboardInterrupt, SystemExit):
        raise
    except ValueError as exc:
        print(f"[ERROR] 参数无效：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"[ERROR] 研究任务失败（{type(exc).__name__}）。"
            "请检查数据可用性和输出目录。",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
