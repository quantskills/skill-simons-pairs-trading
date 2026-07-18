"""
合成数据端到端测试。
不依赖 pandadata 凭证——用可复现的 OU + GBM 生成价格序列。
覆盖：EG 协整识别、OU 半衰期、Kalman 动态 β、回测无前视偏差。
"""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

# 允许 tests/ 直接导 scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import stats_core
import backtest
from stats_core import (engle_granger, kalman_dynamic_beta,
                        rolling_zscore, static_spread)
from backtest import (BacktestConfig, backtest_single_pair,
                      performance_metrics, trades_stats,
                      _pnl_from_spread, strategy_gate)


# ---------- 合成数据工厂 ----------

def _make_cointegrated_pair(seed: int = 42, n: int = 800,
                            true_beta: float = 1.1, true_alpha: float = 0.5,
                            theta: float = 0.05, sigma_eps: float = 0.02
                            ) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    b_ret = rng.normal(0.0003, 0.015, n)
    b_price = 100 * np.exp(np.cumsum(b_ret))
    # spread 是 OU
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] - theta * s[t - 1] + rng.normal(0, sigma_eps)
    a_price = np.exp(np.log(b_price) * true_beta + true_alpha + s)
    return pd.DataFrame({"A": a_price, "B": b_price}, index=idx)


def _make_random_pair(seed: int = 7, n: int = 800) -> pd.DataFrame:
    """两条独立随机游走，不应通过协整。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    a = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.018, n)))
    return pd.DataFrame({"A": a, "B": b}, index=idx)


# ---------- 单元测试 ----------

def test_engle_granger_detects_true_cointegration():
    df = _make_cointegrated_pair()
    r = engle_granger(df["A"], df["B"], "A", "B")
    assert r is not None
    assert r.pvalue < 0.05, f"应识别协整，实际 p={r.pvalue}"
    # β 估计应接近真值 1.1
    assert 0.9 < r.beta < 1.3, f"β 估计偏差过大: {r.beta}"
    # OU 半衰期应为正数且合理
    assert 2 < r.half_life < 100, f"半衰期异常: {r.half_life}"


def test_engle_granger_rejects_random_walks():
    df = _make_random_pair()
    r = engle_granger(df["A"], df["B"], "A", "B")
    # 独立随机游走应有较大 p 值（不显著协整）
    if r is not None:
        assert r.pvalue > 0.05, f"不应识别为协整，实际 p={r.pvalue}"


def test_kalman_beta_tracks_true_value():
    df = _make_cointegrated_pair(seed=1)
    out = kalman_dynamic_beta(df["A"], df["B"])
    assert not out.empty
    # 稳定后的 β 应在真值附近
    beta_tail = out["beta"].iloc[-100:].mean()
    assert 0.9 < beta_tail < 1.3, f"Kalman β 收敛不合理: {beta_tail}"


def test_rolling_zscore_wellformed():
    df = _make_cointegrated_pair(seed=2)
    r = engle_granger(df["A"], df["B"], "A", "B")
    s = static_spread(df["A"], df["B"], r.alpha, r.beta)
    z = rolling_zscore(s, 60).dropna()
    # z 应大致零均值单位方差
    assert abs(z.mean()) < 0.5, f"z 均值偏差过大: {z.mean()}"
    assert 0.5 < z.std() < 2.0, f"z 标准差异常: {z.std()}"


def test_backtest_no_lookahead_and_realistic_metrics():
    """回测应产生真实的绩效指标——Sharpe 不应异常高（>3 通常是 bug）。"""
    df = _make_cointegrated_pair(seed=42)
    cfg = BacktestConfig(formation_days=252,
                          cost_bps_one_side=7.5, method="static")
    trades, daily = backtest_single_pair(df, ("A", "B"), cfg)
    perf = performance_metrics(daily["ret"])

    # 至少发生交易
    assert len(trades) > 0, "合成 OU spread 应触发多次交易"
    # Sharpe 合理区间——修复日度记账 bug 后应 <2.5
    assert -1 < perf["sharpe"] < 3.0, \
        f"Sharpe 异常，怀疑 look-ahead 或日度记账 bug: {perf['sharpe']}"
    # 波动应显著非零（否则说明 P&L 只在几天有非零值）
    assert perf["annual_vol"] > 0.02, \
        f"年化波动过低，怀疑 P&L 稀疏化: {perf['annual_vol']}"


def test_backtest_costs_are_applied():
    """开高成本时净收益应显著低于零成本。"""
    df = _make_cointegrated_pair(seed=3)
    cfg_free = BacktestConfig(formation_days=252,
                                cost_bps_one_side=0.0, method="static")
    cfg_high = BacktestConfig(formation_days=252,
                                cost_bps_one_side=50.0, method="static")
    _, d_free = backtest_single_pair(df, ("A", "B"), cfg_free)
    _, d_high = backtest_single_pair(df, ("A", "B"), cfg_high)
    assert d_free["ret"].sum() > d_high["ret"].sum(), "高成本应侵蚀收益"


def test_spread_pnl_is_normalized_by_gross_exposure():
    spread = pd.Series([0.00, 0.09])
    result = _pnl_from_spread(
        spread, side=1, open_i=0, close_i=1, gross_exposure=3.0
    )
    assert result == pytest.approx(0.03)


def test_leg_pnl_uses_entry_beta_and_gross_exposure():
    result = backtest._pnl_from_legs(
        log_return_a=0.03,
        log_return_b=0.01,
        side=1,
        beta_at_entry=2.0,
    )
    assert result == pytest.approx((0.03 - 2.0 * 0.01) / 3.0)


def test_trades_stats_basic():
    df = _make_cointegrated_pair(seed=5)
    cfg = BacktestConfig(formation_days=252, method="static")
    trades, _ = backtest_single_pair(df, ("A", "B"), cfg)
    if trades:
        st = trades_stats(trades)
        assert 0 <= st["win_rate"] <= 1
        assert st["n_trades"] == len(trades)


@pytest.mark.parametrize(
    ("perf", "stats", "expected"),
    [
        ({"sharpe": -0.1, "annual_ret": 0.02}, {"n_trades": 200},
         "NO_TRADE_NEGATIVE_EDGE"),
        ({"sharpe": 1.0, "annual_ret": 0.05}, {"n_trades": 40},
         "NO_TRADE_INSUFFICIENT_SAMPLE"),
        ({"sharpe": 1.0, "annual_ret": 0.05}, {"n_trades": 150},
         "RESEARCH_PASS"),
        ({"sharpe": 0.3, "annual_ret": 0.02}, {"n_trades": 150},
         "NO_TRADE_WEAK_EDGE"),
        ({"sharpe": 0.8, "annual_ret": 0.005}, {"n_trades": 150},
         "NO_TRADE_WEAK_EDGE"),
        ({"sharpe": float("nan"), "annual_ret": 0.05}, {"n_trades": 150},
         "NO_TRADE_INVALID_METRICS"),
        ({"sharpe": 1.0, "annual_ret": float("inf")}, {"n_trades": 150},
         "NO_TRADE_INVALID_METRICS"),
        ({"sharpe": 0.0, "annual_ret": 0.0}, {"n_trades": 0},
         "NO_TRADE_INSUFFICIENT_SAMPLE"),
    ],
)
def test_strategy_gate_blocks_unreliable_live_use(perf, stats, expected):
    assert strategy_gate(perf, stats) == expected


def _make_adverse_pair(seed: int = 17, n: int = 430) -> pd.DataFrame:
    """形成期平稳，交易期先触发做空 spread，再继续不利扩张。"""
    df = _make_cointegrated_pair(seed=seed, n=n)
    shock = np.zeros(n)
    shock[252:] = np.linspace(0.20, 0.55, n - 252)
    df["A"] = df["A"] * np.exp(shock)
    return df


def test_pair_drawdown_stop_exits_at_five_percent_loss():
    df = _make_adverse_pair()
    cfg = BacktestConfig(
        formation_days=252,
        reestimate_days=60,
        z_entry=1.5,
        z_exit=0.0,
        z_stop=99.0,
        max_hold_days=1000,
        pair_stop_loss=0.05,
        method="static",
    )

    trades, _ = backtest_single_pair(df, ("A", "B"), cfg)

    stopped = [t for t in trades if t.close_reason == "pair_drawdown_stop"]
    assert stopped, "单对亏损达到 5% 时必须强制平仓"
    assert stopped[0].max_drawdown_from_peak <= -0.05
    assert stopped[0].peak_net_pnl >= stopped[0].ret_net


def test_pair_relationship_is_reestimated_at_most_every_sixty_days():
    df = _make_adverse_pair(seed=23, n=500)
    cfg = BacktestConfig(
        formation_days=252,
        reestimate_days=60,
        z_entry=1.5,
        z_exit=0.0,
        z_stop=99.0,
        max_hold_days=1000,
        pair_stop_loss=99.0,
        method="static",
    )

    trades, _ = backtest_single_pair(df, ("A", "B"), cfg)

    period_ends = [t for t in trades if t.close_reason == "period_end"]
    assert period_ends, "滚动窗口结束时应平仓并重新估计"
    assert max(t.hold_days for t in period_ends) < 60


def test_each_rolling_window_rechecks_pvalue_and_half_life(monkeypatch):
    df = _make_cointegrated_pair(seed=29, n=400)
    monkeypatch.setattr(
        backtest,
        "engle_granger",
        lambda *args, **kwargs: SimpleNamespace(
            pvalue=0.01, half_life=200.0, alpha=0.0, beta=1.0
        ),
    )
    cfg = BacktestConfig(
        formation_days=252,
        reestimate_days=60,
        pvalue_cutoff=0.05,
        half_life_min=2.0,
        half_life_max=60.0,
    )

    trades, daily = backtest_single_pair(df, ("A", "B"), cfg)

    assert trades == []
    assert daily["ret"].abs().sum() == 0


def test_backtest_pair_selection_uses_formation_data_only():
    base = _make_cointegrated_pair(seed=31, n=520).rename(
        columns={"A": "A", "B": "B"}
    )
    rng = np.random.default_rng(31)
    base["C"] = 80 * np.exp(np.cumsum(rng.normal(0.0001, 0.02, len(base))))
    industry = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "industry": ["same", "same", "same"],
    })

    selected_before = stats_core.select_pairs_for_backtest(
        base, industry, formation_days=252, corr_threshold=0.6,
        pvalue_cutoff=0.10, half_life_range=(1, 100), top_n=10,
    )
    changed_future = base.copy()
    changed_future.loc[changed_future.index[252]:, "C"] = \
        changed_future.loc[changed_future.index[252]:, "A"]
    selected_after = stats_core.select_pairs_for_backtest(
        changed_future, industry, formation_days=252, corr_threshold=0.6,
        pvalue_cutoff=0.10, half_life_range=(1, 100), top_n=10,
    )

    pd.testing.assert_frame_equal(selected_before, selected_after)


def test_benjamini_hochberg_adjustment_is_monotone_and_preserves_order():
    adjusted, rejected = stats_core.benjamini_hochberg(
        np.array([0.01, 0.04, 0.20]), alpha=0.05
    )

    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.20])
    assert rejected.tolist() == [True, False, False]


def test_cointegration_screen_applies_fdr_and_keeps_raw_pvalue(monkeypatch):
    panel = pd.DataFrame(
        {
            "A": np.arange(1.0, 121.0),
            "B": np.arange(2.0, 122.0),
            "C": np.arange(3.0, 123.0),
            "D": np.arange(4.0, 124.0),
        }
    )
    raw = {("A", "B"): 0.01, ("A", "C"): 0.04, ("A", "D"): 0.20}

    def fake_eg(_pa, _pb, a, b):
        return SimpleNamespace(
            a=a, b=b, pvalue=raw[(a, b)], beta=1.0, alpha=0.0,
            adf_p_resid=raw[(a, b)], half_life=10.0,
            spread_std=0.1, n_obs=120,
        )

    monkeypatch.setattr(stats_core, "engle_granger", fake_eg)
    selected = stats_core.screen_cointegrated(
        panel,
        [("A", "B"), ("A", "C"), ("A", "D")],
        pvalue_cutoff=0.05,
        fdr_alpha=0.05,
        half_life_range=(2.0, 60.0),
    )

    assert selected[["a", "b"]].values.tolist() == [["A", "B"]]
    assert selected.loc[0, "pvalue"] == pytest.approx(0.01)
    assert selected.loc[0, "pvalue_fdr"] == pytest.approx(0.03)
    assert bool(selected.loc[0, "fdr_pass"])


def test_prefilter_never_pairs_different_or_unknown_industries():
    prices = pd.DataFrame({
        "A": np.linspace(10, 30, 120),
        "B": np.linspace(11, 31, 120),
        "C": np.linspace(12, 32, 120),
    })
    industries = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "industry": ["bank", "energy", "UNKNOWN"],
    })

    assert stats_core.prefilter_pairs(
        prices, industries, corr_threshold=0.5, min_obs=100
    ) == []


def test_formation_only_dedup_returns_representatives_and_diagnostics():
    rng = np.random.default_rng(91)
    dates = pd.bdate_range("2024-01-01", periods=260)
    common = rng.normal(0, 0.01, len(dates))
    prices = pd.DataFrame(index=dates)
    for name, noise in {
        "A": common,
        "B": common + rng.normal(0, 0.001, len(dates)),
        "C": -common,
        "D": -common + rng.normal(0, 0.001, len(dates)),
        "E": rng.normal(0, 0.01, len(dates)),
        "F": rng.normal(0, 0.01, len(dates)),
    }.items():
        prices[name] = 100 * np.exp(np.cumsum(noise))
    candidates = pd.DataFrame([
        {"a": "A", "b": "B", "beta": 1.0, "pvalue": 0.01,
         "pvalue_fdr": 0.03},
        {"a": "A", "b": "C", "beta": 1.0, "pvalue": 0.02,
         "pvalue_fdr": 0.03},
        {"a": "D", "b": "E", "beta": 1.0, "pvalue": 0.03,
         "pvalue_fdr": 0.03},
        {"a": "E", "b": "F", "beta": 1.0, "pvalue": 0.04,
         "pvalue_fdr": 0.04},
    ])

    representatives, diagnostics = stats_core.deduplicate_formation_pairs(
        prices, candidates, max_clusters=2, max_pairs_per_symbol=2
    )

    assert len(representatives) == 2
    assert representatives["cluster"].nunique() == 2
    assert representatives["cluster_size"].sum() == len(candidates)
    assert set(representatives["selection_method"]) == {
        "formation_leg_return_pca_hclust_v1"
    }
    assert diagnostics["n_candidates"] == 4
    assert diagnostics["n_representatives"] == 2
    assert diagnostics["formation_start"] == "2024-01-01"
    assert diagnostics["formation_end"] == "2024-12-27"
    assert "condition_number" in diagnostics
    assert diagnostics["pc_explained_variance_ratio"]


def test_rolling_window_last_day_cannot_open_a_silently_dropped_position(
        monkeypatch):
    df = _make_cointegrated_pair(seed=99, n=313)
    monkeypatch.setattr(
        backtest,
        "engle_granger",
        lambda *args, **kwargs: SimpleNamespace(
            pvalue=0.01, half_life=10.0, alpha=0.0, beta=1.0
        ),
    )
    monkeypatch.setattr(
        backtest,
        "static_spread",
        lambda a, b, alpha, beta: pd.Series(0.0, index=a.index),
    )

    def last_day_only(series, window):
        z = pd.Series(0.0, index=series.index)
        if len(z) == 312:
            z.iloc[-1] = 3.0
        return z

    monkeypatch.setattr(backtest, "rolling_zscore", last_day_only)
    cfg = BacktestConfig(
        formation_days=252, reestimate_days=60, z_entry=2.0,
        z_exit=0.5, z_stop=4.0,
    )

    trades, daily = backtest_single_pair(df, ("A", "B"), cfg)

    assert trades == []
    assert "position" in daily.columns
    assert int(daily.loc[df.index[311], "position"]) == 0
    assert int(daily.loc[df.index[312], "position"]) == 0


def test_trade_and_daily_outputs_reconcile_asset_legs_and_costs():
    df = _make_cointegrated_pair(seed=103, n=600)
    cfg = BacktestConfig(
        formation_days=252,
        reestimate_days=60,
        z_entry=1.5,
        cost_bps_one_side=7.5,
    )

    trades, daily = backtest_single_pair(df, ("A", "B"), cfg)

    assert trades
    for trade in trades:
        assert trade.ret_gross == pytest.approx(
            trade.leg_a_ret + trade.leg_b_ret
        )
        assert trade.ret_net == pytest.approx(
            trade.ret_gross - trade.transaction_cost - trade.borrow_cost
        )
        assert trade.transaction_cost == pytest.approx(0.0015)
    assert {"leg_a_ret", "leg_b_ret", "transaction_cost", "borrow_cost", "ret"} \
        .issubset(daily.columns)
    pd.testing.assert_series_equal(
        daily["ret"],
        daily["leg_a_ret"] + daily["leg_b_ret"]
        - daily["transaction_cost"] - daily["borrow_cost"],
        check_names=False,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"formation_days": 0},
        {"reestimate_days": 0},
        {"pvalue_cutoff": 1.0},
        {"fdr_alpha": 0.0},
        {"half_life_min": 20.0, "half_life_max": 10.0},
        {"z_exit": 2.0, "z_entry": 2.0},
        {"z_entry": 4.0, "z_stop": 4.0},
        {"max_hold_days": 0},
        {"pair_stop_loss": -0.05},
        {"pair_stop_loss": np.nan},
        {"pair_stop_loss": np.inf},
        {"cost_bps_one_side": -1.0},
        {"cost_bps_one_side": np.nan},
        {"cost_bps_one_side": np.inf},
        {"short_borrow_bps_annual": -1.0},
        {"short_borrow_bps_annual": np.nan},
        {"short_borrow_bps_annual": np.inf},
        {"kalman_delta": np.nan},
        {"kalman_r": np.inf},
        {"method": "unknown"},
    ],
)
def test_backtest_config_rejects_unsafe_or_incoherent_parameters(kwargs):
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)


def test_engle_granger_rejects_nonpositive_prices_safely():
    dates = pd.bdate_range("2024-01-01", periods=120)
    a = pd.Series(np.linspace(10, 20, len(dates)), index=dates)
    b = pd.Series(np.linspace(8, 18, len(dates)), index=dates)
    a.iloc[50] = 0.0

    assert engle_granger(a, b, "A", "B") is None


def test_five_percent_stop_uses_drawdown_from_trade_pnl_peak(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=260)
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=dates)
    prices.loc[dates[253], "A"] = 100.0 * np.exp(0.12)
    monkeypatch.setattr(
        backtest,
        "engle_granger",
        lambda *args, **kwargs: SimpleNamespace(
            pvalue=0.01, half_life=10.0, alpha=0.0, beta=1.0
        ),
    )
    monkeypatch.setattr(
        backtest,
        "static_spread",
        lambda a, b, alpha, beta: pd.Series(0.0, index=a.index),
    )

    def controlled_z(series, window):
        z = pd.Series(0.0, index=series.index)
        z.loc[dates[252:255]] = [-3.0, -2.0, -2.0]
        return z

    monkeypatch.setattr(backtest, "rolling_zscore", controlled_z)
    cfg = BacktestConfig(
        formation_days=252,
        reestimate_days=60,
        z_window=2,
        z_entry=1.0,
        z_exit=0.0,
        z_stop=99.0,
        max_hold_days=100,
        pair_stop_loss=0.05,
        cost_bps_one_side=0.0,
        short_borrow_bps_annual=0.0,
    )

    trades, _ = backtest_single_pair(prices, ("A", "B"), cfg)

    assert trades
    assert trades[0].close_reason == "pair_drawdown_stop"
    assert trades[0].ret_gross == pytest.approx(0.0, abs=1e-12)
    assert trades[0].peak_net_pnl == pytest.approx(0.06)
    assert trades[0].max_drawdown_from_peak <= -0.05


def test_short_borrow_carry_is_charged_while_position_is_open():
    df = _make_cointegrated_pair(seed=103, n=600)
    free = BacktestConfig(
        formation_days=252, reestimate_days=60, z_entry=1.5,
        short_borrow_bps_annual=0.0,
    )
    costly = BacktestConfig(
        formation_days=252, reestimate_days=60, z_entry=1.5,
        short_borrow_bps_annual=800.0,
    )

    _, daily_free = backtest_single_pair(df, ("A", "B"), free)
    trades_costly, daily_costly = backtest_single_pair(df, ("A", "B"), costly)

    assert trades_costly
    assert daily_costly["borrow_cost"].sum() > 0
    assert daily_costly["ret"].sum() < daily_free["ret"].sum()
    assert sum(trade.borrow_cost for trade in trades_costly) == pytest.approx(
        daily_costly["borrow_cost"].sum()
    )


def test_overlap_limiter_greedily_keeps_lowest_q_pair_per_symbol():
    candidates = pd.DataFrame([
        {"a": "A", "b": "B", "pvalue": 0.01, "pvalue_fdr": 0.02},
        {"a": "A", "b": "C", "pvalue": 0.02, "pvalue_fdr": 0.03},
        {"a": "D", "b": "E", "pvalue": 0.03, "pvalue_fdr": 0.04},
    ])

    selected, diagnostics = stats_core.limit_symbol_overlap(
        candidates, max_pairs_per_symbol=1
    )

    assert (
        selected["a"].astype(str) + "~" + selected["b"].astype(str)
    ).tolist() == ["A~B", "D~E"]
    assert diagnostics["symbol_pair_counts"] == {
        "A": 1, "B": 1, "D": 1, "E": 1
    }
    assert diagnostics["max_symbol_reuse"] == 1
    assert diagnostics["effective_independent_pairs"] == pytest.approx(2.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
