"""
回测引擎：形成期/交易期滚动 + 事件驱动开平仓 + 交易成本 + 绩效指标 + 图表。
入场：|z| > z_entry
离场：|z| < z_exit
止损：|z| > z_stop 或超过 max_hold 天
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stats_core import (engle_granger, kalman_dynamic_beta,
                        rolling_zscore, static_spread)


@dataclass
class BacktestConfig:
    formation_days: int = 252        # 形成期 12M
    reestimate_days: int = 60        # 每 60 个交易日重估协整关系
    pvalue_cutoff: float = 0.05
    fdr_alpha: float = 0.05
    half_life_min: float = 2.0
    half_life_max: float = 60.0
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 4.0
    max_hold_days: int = 60
    pair_stop_loss: float = 0.05     # 单对累计亏损 5% 强平
    z_window: int = 60               # 交易期内滚动 z-score 窗口
    cost_bps_one_side: float = 7.5   # 单边 7.5bp，开平合计 15bp
    short_borrow_bps_annual: float = 800.0  # 融券年费假设 8%
    method: str = "static"           # "static" 用形成期 α/β；"kalman" 每日动态
    kalman_delta: float = 1e-4
    kalman_r: float = 1e-3

    def __post_init__(self) -> None:
        integer_fields = {
            "formation_days": self.formation_days,
            "reestimate_days": self.reestimate_days,
            "max_hold_days": self.max_hold_days,
            "z_window": self.z_window,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} 必须为正整数")
        finite_fields = {
            "pvalue_cutoff": self.pvalue_cutoff,
            "fdr_alpha": self.fdr_alpha,
            "half_life_min": self.half_life_min,
            "half_life_max": self.half_life_max,
            "z_entry": self.z_entry,
            "z_exit": self.z_exit,
            "z_stop": self.z_stop,
            "pair_stop_loss": self.pair_stop_loss,
            "cost_bps_one_side": self.cost_bps_one_side,
            "short_borrow_bps_annual": self.short_borrow_bps_annual,
            "kalman_delta": self.kalman_delta,
            "kalman_r": self.kalman_r,
        }
        for name, value in finite_fields.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} 必须为有限数")
        if not 0.0 < float(self.pvalue_cutoff) < 1.0:
            raise ValueError("pvalue_cutoff 必须在 (0, 1) 内")
        if not 0.0 < float(self.fdr_alpha) < 1.0:
            raise ValueError("fdr_alpha 必须在 (0, 1) 内")
        if not (0.0 < float(self.half_life_min)
                < float(self.half_life_max)):
            raise ValueError("半衰期必须满足 0 < min < max")
        if not (0.0 <= float(self.z_exit) < float(self.z_entry)
                < float(self.z_stop)):
            raise ValueError("z 阈值必须满足 0 <= exit < entry < stop")
        if float(self.pair_stop_loss) <= 0.0:
            raise ValueError("pair_stop_loss 必须为正数")
        if float(self.cost_bps_one_side) < 0.0:
            raise ValueError("cost_bps_one_side 不得为负数")
        if float(self.short_borrow_bps_annual) < 0.0:
            raise ValueError("short_borrow_bps_annual 不得为负数")
        if self.method not in {"static", "kalman"}:
            raise ValueError("method 只允许 static 或 kalman")
        if float(self.kalman_delta) <= 0.0 or float(self.kalman_r) <= 0.0:
            raise ValueError("Kalman 方差参数必须为正数")


@dataclass
class Trade:
    pair: str
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    side: int                 # +1 = long a / short b（spread 负偏）；-1 反之
    beta_at_open: float
    z_open: float
    z_close: float
    leg_a_ret: float
    leg_b_ret: float
    transaction_cost: float
    borrow_cost: float
    peak_net_pnl: float
    max_drawdown_from_peak: float
    ret_gross: float
    ret_net: float
    hold_days: int
    close_reason: str


def _pnl_from_spread(spread: pd.Series, side: int,
                     open_i: int, close_i: int,
                     gross_exposure: float = 1.0) -> float:
    """
    仓位固定 1 单位 long/short spread；side=+1 表示做多 spread（预期回归上行）。
    收益 ≈ side * (spread_close - spread_open)。仅用于交易级汇总。
    日度 mark-to-market 在主循环里按 side * Δspread 单独累计（见 daily 累加）。
    """
    scale = max(abs(float(gross_exposure)), 1e-12)
    return float(side * (spread.iloc[close_i] - spread.iloc[open_i]) / scale)


def _pnl_from_legs(log_return_a: float, log_return_b: float,
                   side: int, beta_at_entry: float) -> float:
    """按入场时固定对冲比率计算两条真实资产腿的归一化收益。"""
    gross_exposure = 1.0 + abs(float(beta_at_entry))
    return float(
        side * (log_return_a - beta_at_entry * log_return_b)
        / gross_exposure
    )


def _leg_pnl_from_returns(log_return_a: float, log_return_b: float,
                          side: int,
                          beta_at_entry: float) -> Tuple[float, float]:
    """Return separately attributable normalized P&L for the two asset legs."""
    gross_exposure = 1.0 + abs(float(beta_at_entry))
    leg_a = side * float(log_return_a) / gross_exposure
    leg_b = -side * float(beta_at_entry) * float(log_return_b) / gross_exposure
    return float(leg_a), float(leg_b)


def backtest_single_pair(price_panel: pd.DataFrame,
                         pair: Tuple[str, str],
                         cfg: BacktestConfig) -> Tuple[List[Trade], pd.DataFrame]:
    """
    单对滚动窗口回测。
    返回 (trades, daily_pnl_df[date, ret])。
    """
    a, b = pair
    if a == b or a not in price_panel.columns or b not in price_panel.columns:
        raise ValueError("pair 必须包含价格面板中的两个不同代码")
    df = price_panel[[a, b]].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if (df <= 0).any().any():
        raise ValueError("回测价格必须全部为正数")
    trades: List[Trade] = []
    daily = pd.DataFrame(
        0.0,
        index=df.index,
        columns=[
            "leg_a_ret", "leg_b_ret", "transaction_cost",
            "borrow_cost", "ret",
        ],
    )
    daily["position"] = 0

    total = len(df)
    step = cfg.reestimate_days
    start = cfg.formation_days
    while start < total:
        form = df.iloc[start - cfg.formation_days:start]
        trade = df.iloc[start:min(start + step, total)]
        if trade.empty:
            break

        r = engle_granger(form[a], form[b], a, b)
        relationship_valid = (
            r is not None
            and r.pvalue < cfg.pvalue_cutoff
            and np.isfinite(r.half_life)
            and cfg.half_life_min <= r.half_life <= cfg.half_life_max
        )
        if not relationship_valid:
            start += step
            continue

        if cfg.method == "kalman":
            kal = kalman_dynamic_beta(
                pd.concat([form[a], trade[a]]),
                pd.concat([form[b], trade[b]]),
                delta=cfg.kalman_delta,
                r_var=cfg.kalman_r,
            )
            if kal.empty or not trade.index.isin(kal.index).all():
                start += step
                continue
            spread_all = kal["spread"]
            beta_series = kal["beta"].loc[trade.index]
        else:
            spread_all = static_spread(
                pd.concat([form[a], trade[a]]),
                pd.concat([form[b], trade[b]]),
                r.alpha,
                r.beta,
            )
            beta_series = pd.Series(r.beta, index=trade.index)

        z = rolling_zscore(spread_all, window=cfg.z_window).loc[trade.index]
        pos = 0
        entry_i = None
        entry_z = None
        entry_beta = None
        trade_gross = 0.0
        trade_leg_a = 0.0
        trade_leg_b = 0.0
        trade_borrow_cost = 0.0
        trade_peak_net = 0.0
        trade_max_drawdown = 0.0
        prev_log_a = None
        prev_log_b = None

        for i, (dt, zi) in enumerate(z.items()):
            log_a = float(np.log(trade[a].iloc[i]))
            log_b = float(np.log(trade[b].iloc[i]))
            if pos != 0 and prev_log_a is not None and prev_log_b is not None:
                leg_a_day, leg_b_day = _leg_pnl_from_returns(
                    log_a - prev_log_a,
                    log_b - prev_log_b,
                    pos,
                    float(entry_beta),
                )
                daily.loc[dt, "leg_a_ret"] += leg_a_day
                daily.loc[dt, "leg_b_ret"] += leg_b_day
                trade_leg_a += leg_a_day
                trade_leg_b += leg_b_day
                trade_gross += leg_a_day + leg_b_day
                gross_exposure = 1.0 + abs(float(entry_beta))
                short_weight = (
                    abs(float(entry_beta)) / gross_exposure
                    if pos > 0 else 1.0 / gross_exposure
                )
                borrow_day = (
                    float(cfg.short_borrow_bps_annual) / 1e4 / 252.0
                    * short_weight
                )
                daily.loc[dt, "borrow_cost"] += borrow_day
                trade_borrow_cost += borrow_day
                estimated_roundtrip_cost = 2.0 * (
                    cfg.cost_bps_one_side / 1e4
                )
                current_net = (
                    trade_gross - trade_borrow_cost
                    - estimated_roundtrip_cost
                )
                trade_peak_net = max(trade_peak_net, current_net)
                trade_max_drawdown = min(
                    trade_max_drawdown, current_net - trade_peak_net
                )
                prev_log_a, prev_log_b = log_a, log_b

            is_window_end = i == len(z) - 1
            if pos == 0:
                if not np.isfinite(zi) or is_window_end:
                    daily.loc[dt, "position"] = 0
                    continue
                if zi > cfg.z_entry:
                    pos, entry_i, entry_z = -1, i, zi
                elif zi < -cfg.z_entry:
                    pos, entry_i, entry_z = +1, i, zi
                if pos != 0:
                    entry_beta = float(beta_series.iloc[i])
                    if not np.isfinite(entry_beta):
                        pos, entry_i, entry_z, entry_beta = 0, None, None, None
                    else:
                        trade_gross = 0.0
                        trade_leg_a = 0.0
                        trade_leg_b = 0.0
                        trade_borrow_cost = 0.0
                        trade_peak_net = 0.0
                        trade_max_drawdown = 0.0
                        prev_log_a, prev_log_b = log_a, log_b
            else:
                hold = i - int(entry_i)
                exit_now, reason = False, ""
                current_net = (
                    trade_gross - trade_borrow_cost
                    - 2.0 * (cfg.cost_bps_one_side / 1e4)
                )
                drawdown_from_peak = current_net - trade_peak_net
                if is_window_end:
                    exit_now, reason = True, "period_end"
                elif drawdown_from_peak <= -abs(cfg.pair_stop_loss):
                    exit_now, reason = True, "pair_drawdown_stop"
                elif np.isfinite(zi) and abs(zi) < cfg.z_exit:
                    exit_now, reason = True, "mean_revert"
                elif np.isfinite(zi) and abs(zi) > cfg.z_stop:
                    exit_now, reason = True, "stop_loss"
                elif hold >= cfg.max_hold_days:
                    exit_now, reason = True, "time_stop"
                if exit_now:
                    cost = 2.0 * (cfg.cost_bps_one_side / 1e4)
                    trades.append(Trade(
                        pair=f"{a}~{b}",
                        open_date=trade.index[int(entry_i)],
                        close_date=dt,
                        side=pos,
                        beta_at_open=float(entry_beta),
                        z_open=float(entry_z),
                        z_close=float(zi) if np.isfinite(zi) else np.nan,
                        leg_a_ret=float(trade_leg_a),
                        leg_b_ret=float(trade_leg_b),
                        transaction_cost=float(cost),
                        borrow_cost=float(trade_borrow_cost),
                        peak_net_pnl=float(trade_peak_net),
                        max_drawdown_from_peak=float(trade_max_drawdown),
                        ret_gross=float(trade_gross),
                        ret_net=float(
                            trade_gross - cost - trade_borrow_cost
                        ),
                        hold_days=int(hold),
                        close_reason=reason,
                    ))
                    daily.loc[dt, "transaction_cost"] += cost
                    pos, entry_i, entry_z, entry_beta = 0, None, None, None
                    trade_gross = 0.0
                    trade_leg_a = 0.0
                    trade_leg_b = 0.0
                    trade_borrow_cost = 0.0
                    trade_peak_net = 0.0
                    trade_max_drawdown = 0.0
                    prev_log_a, prev_log_b = None, None
            daily.loc[dt, "position"] = int(pos)
        start += step

    daily["ret"] = (
        daily["leg_a_ret"] + daily["leg_b_ret"]
        - daily["transaction_cost"] - daily["borrow_cost"]
    )
    return trades, daily


def performance_metrics(daily_ret: pd.Series) -> Dict[str, float]:
    r = daily_ret.dropna()
    if len(r) == 0 or r.std() == 0:
        return {"sharpe": 0, "annual_ret": 0, "annual_vol": 0,
                "max_dd": 0, "calmar": 0, "n_days": len(r)}
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    calmar = ann_ret / abs(dd) if dd < 0 else np.inf
    return {"sharpe": float(sharpe),
            "annual_ret": float(ann_ret),
            "annual_vol": float(ann_vol),
            "max_dd": float(dd),
            "calmar": float(calmar),
            "n_days": int(len(r))}


def trades_stats(trades: List[Trade]) -> Dict[str, float]:
    if not trades:
        return {"n_trades": 0, "win_rate": 0, "avg_ret": 0,
                "avg_hold": 0, "profit_factor": 0}
    df = pd.DataFrame([asdict(t) for t in trades])
    wins = df["ret_net"] > 0
    pf_num = df.loc[wins, "ret_net"].sum()
    pf_den = -df.loc[~wins, "ret_net"].sum()
    return {"n_trades": int(len(df)),
            "win_rate": float(wins.mean()),
            "avg_ret": float(df["ret_net"].mean()),
            "avg_hold": float(df["hold_days"].mean()),
            "profit_factor": float(pf_num / pf_den) if pf_den > 0 else np.inf}


def strategy_gate(performance: Dict[str, float],
                  trade_stats: Dict[str, float],
                  min_trades: int = 100,
                  min_sharpe: float = 0.5,
                  min_annual_ret: float = 0.01) -> str:
    """Return a conservative research gate; never implies live-trading approval."""
    n_trades = int(trade_stats.get("n_trades", 0))
    if n_trades <= 0:
        return "NO_TRADE_INSUFFICIENT_SAMPLE"
    sharpe = float(performance.get("sharpe", np.nan))
    annual_ret = float(performance.get("annual_ret", np.nan))
    if not np.isfinite(sharpe) or not np.isfinite(annual_ret):
        return "NO_TRADE_INVALID_METRICS"
    if sharpe <= 0.0 or annual_ret <= 0.0:
        return "NO_TRADE_NEGATIVE_EDGE"
    if n_trades < int(min_trades):
        return "NO_TRADE_INSUFFICIENT_SAMPLE"
    if sharpe < float(min_sharpe) or annual_ret < float(min_annual_ret):
        return "NO_TRADE_WEAK_EDGE"
    return "RESEARCH_PASS"


def plot_equity(daily_ret: pd.Series, out_path: str, title: str = "") -> None:
    r = daily_ret.dropna()
    if len(r) == 0:
        return
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5),
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(cum.index, cum.values, lw=1.4, color="#c8102e")
    ax1.set_title(title or "Equity Curve", fontsize=13)
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(alpha=0.3)
    ax2.fill_between(dd.index, dd.values, 0, color="#c8102e", alpha=0.3)
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pair_diagnostic(price_panel: pd.DataFrame,
                         pair: Tuple[str, str],
                         alpha: float, beta: float,
                         z_window: int, out_path: str) -> None:
    a, b = pair
    df = price_panel[[a, b]].dropna()
    spread = np.log(df[a]) - alpha - beta * np.log(df[b])
    z = rolling_zscore(spread, z_window)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5))
    ax1.plot(spread.index, spread.values, lw=1, color="#333")
    ax1.axhline(spread.mean(), color="#c8102e", ls="--", lw=1)
    ax1.set_title(f"Log Spread: {a} - {beta:.3f}*{b}")
    ax1.grid(alpha=0.3)
    ax2.plot(z.index, z.values, lw=1, color="#1f4788")
    for lv, ls in [(2, "--"), (-2, "--"), (0.5, ":"), (-0.5, ":")]:
        ax2.axhline(lv, color="grey", ls=ls, lw=0.8)
    ax2.set_ylabel("Rolling z-score")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
