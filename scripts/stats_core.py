"""
统计核心：Engle-Granger 协整 + OU 半衰期 + Kalman 动态 β + 行业聚类预筛。
所有函数纯计算，输入 pandas，输出 pandas / dict。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, List, Dict, Tuple
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


SELECTION_METHOD = "formation_leg_return_pca_hclust_v1"


# ============ 预筛：行业内 + 相关性 ============

def prefilter_pairs(price_panel: pd.DataFrame,
                    industry_map: pd.DataFrame,
                    corr_threshold: float = 0.80,
                    min_obs: int = 200) -> List[Tuple[str, str]]:
    """
    只保留同行业、log-价相关性 > 阈值的对。
    这是 Simons 风格的第一层漏斗——先把 O(N²) 缩到几百对再做协整。
    """
    ind = dict(zip(industry_map["symbol"], industry_map["industry"]))
    prices = price_panel.dropna(how="all")
    prices = prices.loc[:, prices.count() >= min_obs]
    log_p = np.log(prices.clip(lower=1e-9))
    corr = log_p.corr()

    pairs = []
    cols = list(prices.columns)
    for a, b in combinations(cols, 2):
        ia, ib = ind.get(a, "UA"), ind.get(b, "UB")
        if ia != ib or ia in ("UNKNOWN", "UA", "UB"):
            continue
        if corr.at[a, b] < corr_threshold:
            continue
        pairs.append((a, b))
    return pairs


# ============ Engle-Granger 协整 + β + 残差 ============

@dataclass
class CointResult:
    a: str
    b: str
    pvalue: float
    beta: float          # a = alpha + beta * b + eps
    alpha: float
    adf_p_resid: float   # 残差 ADF p 值
    half_life: float     # OU 半衰期（交易日）
    spread_std: float
    n_obs: int


def _ou_half_life(spread: pd.Series) -> float:
    """
    OU: d(spread) = -theta * (spread - mu) dt + sigma dW
    半衰期 = ln(2) / theta。用 AR(1) 拟合：Δs_t = a + b * s_{t-1} + e，theta = -b。
    """
    s = spread.dropna().values
    if len(s) < 30:
        return np.nan
    ds = np.diff(s)
    x = sm.add_constant(s[:-1])
    try:
        res = sm.OLS(ds, x).fit()
        b = res.params[1]
        if b >= 0:
            return np.inf  # 不回归
        return float(np.log(2) / -b)
    except Exception:
        return np.nan


def engle_granger(price_a: pd.Series, price_b: pd.Series,
                  name_a: str, name_b: str) -> Optional[CointResult]:
    """
    对 (log price_a, log price_b) 做 EG 协整：
    1. coint() 两阶段检验 p 值
    2. OLS 拟合 β 与 α
    3. 残差 ADF
    4. OU 半衰期
    """
    df = pd.concat([price_a, price_b], axis=1, keys=["a", "b"])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 100:
        return None
    if (df <= 0).any().any():
        return None
    la, lb = np.log(df["a"]), np.log(df["b"])
    try:
        _t, pv, _c = coint(la, lb, trend="c", autolag="AIC")
    except Exception:
        return None
    try:
        X = sm.add_constant(lb)
        ols = sm.OLS(la, X).fit()
        alpha, beta = float(ols.params.iloc[0]), float(ols.params.iloc[1])
    except Exception:
        return None
    if not np.isfinite([pv, alpha, beta]).all():
        return None
    resid = la - alpha - beta * lb
    try:
        adf_p = float(adfuller(resid, autolag="AIC")[1])
    except Exception:
        adf_p = np.nan
    hl = _ou_half_life(resid)
    return CointResult(
        a=name_a, b=name_b, pvalue=float(pv),
        beta=beta, alpha=alpha,
        adf_p_resid=adf_p, half_life=hl,
        spread_std=float(resid.std()),
        n_obs=int(len(df)),
    )


def screen_cointegrated(price_panel: pd.DataFrame,
                        pairs: List[Tuple[str, str]],
                        pvalue_cutoff: float = 0.05,
                        fdr_alpha: float = 0.05,
                        half_life_range: Tuple[float, float] = (2.0, 60.0)
                        ) -> pd.DataFrame:
    """Screen pairs with raw EG p-values plus Benjamini-Hochberg FDR."""
    if not 0.0 < float(pvalue_cutoff) < 1.0:
        raise ValueError("pvalue_cutoff 必须在 (0, 1) 内")
    if not 0.0 < float(fdr_alpha) < 1.0:
        raise ValueError("fdr_alpha 必须在 (0, 1) 内")
    rows = []
    for a, b in pairs:
        r = engle_granger(price_panel[a], price_panel[b], a, b)
        if r is None:
            continue
        rows.append(r.__dict__)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    adjusted, rejected = benjamini_hochberg(
        df["pvalue"].to_numpy(dtype=float), alpha=fdr_alpha
    )
    df["pvalue_fdr"] = adjusted
    df["fdr_pass"] = rejected
    lo, hi = half_life_range
    keep = (df["pvalue"] < pvalue_cutoff) & \
           df["fdr_pass"] & \
           (df["half_life"].between(lo, hi))
    return df[keep].sort_values(
        ["pvalue_fdr", "pvalue", "a", "b"]
    ).reset_index(drop=True)


def benjamini_hochberg(
        pvalues: np.ndarray,
        alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Return BH-adjusted p-values and rejection decisions in input order."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1:
        raise ValueError("pvalues 必须是一维数组")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha 必须在 (0, 1) 内")
    if values.size == 0:
        return values.copy(), np.zeros(0, dtype=bool)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("pvalues 必须是 [0, 1] 内的有限数")

    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    m = len(ranked)
    raw_adjusted = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    adjusted = np.empty_like(monotone)
    adjusted[order] = monotone
    return adjusted, adjusted <= float(alpha)


def _formation_pair_returns(
        price_panel: pd.DataFrame,
        candidates: pd.DataFrame) -> pd.DataFrame:
    """Build formation-only, gross-exposure-normalized leg return proxies."""
    columns: Dict[str, pd.Series] = {}
    for row in candidates.itertuples(index=False):
        a, b = str(row.a), str(row.b)
        beta = float(row.beta)
        pair_name = f"{a}~{b}"
        prices = price_panel[[a, b]].apply(
            pd.to_numeric, errors="coerce"
        ).where(lambda x: x > 0)
        leg_returns = np.log(prices).diff()
        columns[pair_name] = (
            leg_returns[a] - beta * leg_returns[b]
        ) / (1.0 + abs(beta))
    if not columns:
        return pd.DataFrame(index=price_panel.index)
    return pd.DataFrame(columns).replace([np.inf, -np.inf], np.nan)


def limit_symbol_overlap(
        candidates: pd.DataFrame,
        max_pairs_per_symbol: int = 1) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Greedily keep the lowest-q pairs subject to a per-symbol reuse cap."""
    if int(max_pairs_per_symbol) <= 0:
        raise ValueError("max_pairs_per_symbol 必须为正整数")
    if candidates.empty:
        return candidates.copy(), {
            "symbol_pair_counts": {},
            "max_symbol_reuse": 0,
            "effective_independent_pairs": 0.0,
            "pairs_rejected_for_symbol_overlap": 0,
        }
    required = {"a", "b", "pvalue", "pvalue_fdr"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"重叠限制缺少字段: {sorted(missing)}")
    ordered = candidates.copy()
    ordered["_pair_sort"] = (
        ordered["a"].astype(str) + "~" + ordered["b"].astype(str)
    )
    ordered = ordered.sort_values(
        ["pvalue_fdr", "pvalue", "_pair_sort"], kind="mergesort"
    )
    counts: Dict[str, int] = {}
    kept_indices: List[int] = []
    rejected = 0
    for index, row in ordered.iterrows():
        a, b = str(row["a"]), str(row["b"])
        if (
            counts.get(a, 0) >= int(max_pairs_per_symbol)
            or counts.get(b, 0) >= int(max_pairs_per_symbol)
        ):
            rejected += 1
            continue
        kept_indices.append(index)
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1
    selected = ordered.loc[kept_indices].drop(
        columns=["_pair_sort"]
    ).reset_index(drop=True)
    unique_symbols = len(counts)
    diagnostics: Dict[str, object] = {
        "symbol_pair_counts": dict(sorted(counts.items())),
        "max_symbol_reuse": max(counts.values(), default=0),
        "effective_independent_pairs": float(
            min(len(selected), unique_symbols / 2.0)
        ),
        "pairs_rejected_for_symbol_overlap": int(rejected),
    }
    return selected, diagnostics


def deduplicate_formation_pairs(
        formation_panel: pd.DataFrame,
        candidates: pd.DataFrame,
        max_clusters: int = 20,
        max_pairs_per_symbol: int = 1) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Cluster formation-period pair leg returns and keep one pair per cluster."""
    if int(max_clusters) <= 0:
        raise ValueError("max_clusters 必须为正整数")
    if candidates.empty:
        return candidates.copy(), {
            "selection_method": SELECTION_METHOD,
            "n_candidates": 0,
            "n_representatives": 0,
            "formation_start": None,
            "formation_end": None,
            "condition_number": None,
            "effective_rank": 0.0,
            "max_abs_pair_correlation": 0.0,
            "pc_explained_variance_ratio": [],
            "symbol_pair_counts": {},
            "max_symbol_reuse": 0,
            "effective_independent_pairs": 0.0,
            "pairs_rejected_for_symbol_overlap": 0,
        }
    required = {"a", "b", "beta", "pvalue", "pvalue_fdr"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"候选配对缺少字段: {sorted(missing)}")

    pair_returns = _formation_pair_returns(formation_panel, candidates)
    usable = pair_returns.dropna(how="all")
    if usable.empty:
        raise ValueError("形成期无法计算候选配对收益")
    standardized = (
        (usable - usable.mean()) / usable.std(ddof=1).replace(0, np.nan)
    ).fillna(0.0)
    matrix = standardized.to_numpy(dtype=float)
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    eigenvalues = singular ** 2
    total_eigen = float(eigenvalues.sum())
    ratios = (
        (eigenvalues / total_eigen).tolist()
        if total_eigen > 0 else [0.0 for _ in eigenvalues]
    )
    positive = singular[singular > np.finfo(float).eps]
    condition_number = (
        float(positive.max() / positive.min()) if len(positive) else None
    )
    positive_ratios = np.asarray([x for x in ratios if x > 0], dtype=float)
    effective_rank = (
        float(np.exp(-(positive_ratios * np.log(positive_ratios)).sum()))
        if len(positive_ratios) else 0.0
    )

    corr = usable.corr(min_periods=20).fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    n_candidates = len(candidates)
    n_clusters = min(int(max_clusters), n_candidates)
    if n_candidates == 1:
        labels = np.array([1], dtype=int)
    else:
        distance = (1.0 - corr.abs()).clip(0.0, 1.0)
        distance = (distance + distance.T) / 2.0
        np.fill_diagonal(distance.values, 0.0)
        tree = linkage(
            squareform(distance.to_numpy(), checks=True), method="average"
        )
        labels = fcluster(tree, t=n_clusters, criterion="maxclust")

    working = candidates.copy().reset_index(drop=True)
    working["pair"] = working["a"].astype(str) + "~" + working["b"].astype(str)
    label_map = dict(zip(pair_returns.columns, labels))
    working["cluster"] = working["pair"].map(label_map).astype(int)
    representatives = []
    for cluster, group in working.groupby("cluster", sort=True):
        ordered = group.sort_values(
            ["pvalue_fdr", "pvalue", "pair"], kind="mergesort"
        )
        representative = ordered.iloc[0].copy()
        representative["cluster_size"] = int(len(group))
        representative["cluster_members"] = ";".join(
            sorted(group["pair"].astype(str))
        )
        representative["selection_method"] = SELECTION_METHOD
        representatives.append(representative)
    cluster_representatives = pd.DataFrame(representatives).sort_values(
        ["pvalue_fdr", "pvalue", "pair"], kind="mergesort"
    ).reset_index(drop=True)
    selected, overlap_diagnostics = limit_symbol_overlap(
        cluster_representatives,
        max_pairs_per_symbol=max_pairs_per_symbol,
    )

    off_diagonal = corr.abs().to_numpy(dtype=float).copy()
    np.fill_diagonal(off_diagonal, np.nan)
    max_abs_corr = (
        float(np.nanmax(off_diagonal)) if n_candidates > 1 else 0.0
    )
    diagnostics: Dict[str, object] = {
        "selection_method": SELECTION_METHOD,
        "n_candidates": int(n_candidates),
        "n_cluster_representatives": int(len(cluster_representatives)),
        "n_representatives": int(len(selected)),
        "n_observations": int(len(usable)),
        "formation_start": (
            str(pd.Timestamp(formation_panel.index.min()).date())
            if len(formation_panel.index) else None
        ),
        "formation_end": (
            str(pd.Timestamp(formation_panel.index.max()).date())
            if len(formation_panel.index) else None
        ),
        "condition_number": condition_number,
        "effective_rank": effective_rank,
        "max_abs_pair_correlation": max_abs_corr,
        "pc_explained_variance_ratio": [float(x) for x in ratios],
        "clusters_requested": int(max_clusters),
        "clusters_realized": int(cluster_representatives["cluster"].nunique()),
        "max_pairs_per_symbol": int(max_pairs_per_symbol),
    }
    diagnostics.update(overlap_diagnostics)
    return selected, diagnostics


def select_pairs_for_backtest(
        price_panel: pd.DataFrame,
        industry_map: pd.DataFrame,
        formation_days: int = 252,
        corr_threshold: float = 0.80,
        pvalue_cutoff: float = 0.05,
        fdr_alpha: float = 0.05,
        half_life_range: Tuple[float, float] = (2.0, 60.0),
        top_n: int = 40,
        dedup_clusters: int = 20,
        max_pairs_per_symbol: int = 1,
        return_diagnostics: bool = False):
    """只用最初形成期筛选回测候选，避免未来数据参与选对。"""
    if formation_days <= 0 or len(price_panel) < formation_days:
        empty = pd.DataFrame()
        diagnostics = {
            "selection_method": SELECTION_METHOD,
            "n_candidates": 0,
            "n_representatives": 0,
            "reason": "insufficient_formation_data",
        }
        return (empty, diagnostics) if return_diagnostics else empty
    formation = price_panel.iloc[:formation_days]
    pairs = prefilter_pairs(
        formation,
        industry_map,
        corr_threshold=corr_threshold,
        min_obs=max(100, int(formation_days * 0.8)),
    )
    selected = screen_cointegrated(
        formation,
        pairs,
        pvalue_cutoff=pvalue_cutoff,
        fdr_alpha=fdr_alpha,
        half_life_range=half_life_range,
    )
    representatives, diagnostics = deduplicate_formation_pairs(
        formation,
        selected,
        max_clusters=dedup_clusters,
        max_pairs_per_symbol=max_pairs_per_symbol,
    )
    result = representatives.head(max(0, int(top_n))).reset_index(drop=True)
    diagnostics["n_selected"] = int(len(result))
    return (result, diagnostics) if return_diagnostics else result


# ============ Kalman 动态 β ============

def kalman_dynamic_beta(price_a: pd.Series, price_b: pd.Series,
                        delta: float = 1e-4, r_var: float = 1e-3
                        ) -> pd.DataFrame:
    """
    状态方程: [alpha_t, beta_t]' = [alpha_{t-1}, beta_{t-1}]' + w_t,  Q = delta * I
    观测方程: log_a_t = alpha_t + beta_t * log_b_t + v_t,  R = r_var
    返回 index=date，columns=[alpha, beta, spread]。
    """
    df = pd.concat([np.log(price_a), np.log(price_b)],
                   axis=1, keys=["a", "b"]).dropna()
    n = len(df)
    if n < 30:
        return pd.DataFrame()

    x = np.array([0.0, 1.0])   # [alpha, beta]
    P = np.eye(2)
    Q = np.eye(2) * delta
    R = r_var

    out = np.zeros((n, 3))     # alpha, beta, spread
    y_arr = df["a"].values
    xb_arr = df["b"].values

    for t in range(n):
        # 预测
        P = P + Q
        H = np.array([1.0, xb_arr[t]])
        y_pred = H @ x
        S = H @ P @ H + R
        K = (P @ H) / S
        innov = y_arr[t] - y_pred
        x = x + K * innov
        P = P - np.outer(K, H) @ P
        out[t] = [x[0], x[1], innov]

    return pd.DataFrame(out, index=df.index, columns=["alpha", "beta", "spread"])


# ============ z-score ============

def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - mu) / sd


def static_spread(price_a: pd.Series, price_b: pd.Series,
                  alpha: float, beta: float) -> pd.Series:
    return np.log(price_a) - alpha - beta * np.log(price_b)
