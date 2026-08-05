"""Deterministic research-run identity and data fingerprint helpers."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CODE_VERSION = "1.2.0"

REQUIRED_GATE_CONFIG_KEYS = (
    "formation_days",
    "reestimate_days",
    "corr_threshold",
    "pvalue_cutoff",
    "fdr_alpha",
    "half_life_min",
    "half_life_max",
    "method",
    "kalman_delta",
    "kalman_r",
    "z_window",
    "z_entry",
    "z_exit",
    "z_stop",
    "max_hold_days",
    "pair_stop_loss",
    "cost_bps_one_side",
    "short_borrow_bps_annual",
    "price_basis",
    "price_source_method",
    "selection_method",
    "dedup_clusters",
    "max_pairs_per_symbol",
    "top_n",
    "evaluation_years",
    "signal_lookback_days",
)

REQUIRED_DATA_FINGERPRINT_KEYS = (
    "price_panel_sha256",
    "universe_sha256",
    "industry_map_sha256",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash a canonical table representation, including index and column names."""
    canonical = frame.copy()
    canonical = canonical.sort_index()
    canonical = canonical.reindex(sorted(map(str, canonical.columns)), axis=1)
    canonical.columns = [str(column) for column in canonical.columns]
    text = canonical.to_csv(
        index=True,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="<NA>",
        date_format="%Y-%m-%d",
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_indexes(indexes: Iterable[str]) -> list[str]:
    normalized = sorted({str(index).strip() for index in indexes if str(index).strip()})
    if not normalized:
        raise ValueError("指数池不能为空")
    return normalized


def validate_config_summary(config_summary: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [
        key for key in REQUIRED_GATE_CONFIG_KEYS if key not in config_summary
    ]
    if missing:
        raise ValueError(f"门控配置缺少字段: {missing}")
    return {key: _jsonable(config_summary[key]) for key in REQUIRED_GATE_CONFIG_KEYS}


def config_id(indexes: Sequence[str], config_summary: Mapping[str, Any]) -> str:
    config = validate_config_summary(config_summary)
    return stable_json_sha256({
        "code_version": CODE_VERSION,
        "indexes": normalize_indexes(indexes),
        "config_summary": config,
    })


def make_run_identity(
        *,
        indexes: Sequence[str],
        config_summary: Mapping[str, Any],
        periods: Mapping[str, Sequence[str]],
        data_fingerprints: Mapping[str, str],
        representative_pairs: Sequence[str],
        research_results: Mapping[str, Any],
        signal_snapshot: Mapping[str, Any]) -> Dict[str, str]:
    config = validate_config_summary(config_summary)
    missing_fingerprints = [
        key for key in REQUIRED_DATA_FINGERPRINT_KEYS
        if key not in data_fingerprints
    ]
    if missing_fingerprints:
        raise ValueError(f"数据指纹缺少字段: {missing_fingerprints}")
    fingerprints = {
        key: str(data_fingerprints[key])
        for key in REQUIRED_DATA_FINGERPRINT_KEYS
    }
    if any(
        len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower())
        for value in fingerprints.values()
    ):
        raise ValueError("数据指纹必须是 64 位 SHA256")
    normalized_pairs = sorted({str(pair) for pair in representative_pairs})
    pair_sha = stable_json_sha256(normalized_pairs)
    results = {
        "strategy_gate": research_results.get("strategy_gate"),
        "performance": _jsonable(research_results.get("performance", {})),
        "trades": _jsonable(research_results.get("trades", {})),
    }
    results_sha = stable_json_sha256(results)
    signal_snapshot_sha = stable_json_sha256(_jsonable(signal_snapshot))
    cid = config_id(indexes, config)
    payload = {
        "code_version": CODE_VERSION,
        "config_id": cid,
        "indexes": normalize_indexes(indexes),
        "config_summary": config,
        "periods": _jsonable(periods),
        "data_fingerprints": fingerprints,
        "representative_pairs_sha256": pair_sha,
        "results_sha256": results_sha,
        "signal_snapshot_sha256": signal_snapshot_sha,
    }
    return {
        "code_version": CODE_VERSION,
        "config_id": cid,
        "representative_pairs_sha256": pair_sha,
        "results_sha256": results_sha,
        "signal_snapshot_sha256": signal_snapshot_sha,
        "run_id": stable_json_sha256(payload),
    }


def identity_is_valid(report: Mapping[str, Any]) -> bool:
    try:
        expected = make_run_identity(
            indexes=report["indexes"],
            config_summary=report["config_summary"],
            periods=report["periods"],
            data_fingerprints=report["data_fingerprints"],
            representative_pairs=report["representative_pairs"],
            research_results={
                "strategy_gate": report["strategy_gate"],
                "performance": report["performance"],
                "trades": report["trades"],
            },
            signal_snapshot=report["signal_snapshot"],
        )
    except (KeyError, TypeError, ValueError):
        return False
    return all(report.get(key) == value for key, value in expected.items())
