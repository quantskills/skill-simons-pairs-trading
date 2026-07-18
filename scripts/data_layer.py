"""
数据层：panda_data 客户端封装 + 磁盘缓存
- 凭证：优先进程环境变量，其次用户级 ~/.pandadata.env
- 缓存：Parquet 落盘到 outputs/cache/，命中免调 API
"""
from __future__ import annotations
import functools
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import pandas as pd

from pandadata_security import (
    PandaDataSecurityError,
    establish_session,
)

try:
    import panda_data as pdd
except ImportError:
    sys.stderr.write("请先 pip install --upgrade panda_data\n")
    raise

_TOKEN = None
CACHE_DIR = Path(os.environ.get("SIMONS_CACHE", Path.home() / ".newmax" / "simons_cache"))


_CREDENTIAL_KEYS = {
    "PANDADATA_USER", "PANDADATA_MOBILE", "PANDADATA_PASSWORD",
    "PANDADATA_TOKEN", "PANDADATA_BASE_URL",
}

PRICE_BASIS_TO_METHOD = {
    "pre_adjusted": "get_stock_daily_pre",
    "post_adjusted": "get_stock_daily_post",
}


def _env_candidates() -> List[Path]:
    """Only use the dedicated user-level credential file by default."""
    return [Path.home() / ".pandadata.env"]


def _load_dotenv(candidates: Optional[List[Path]] = None) -> None:
    """读取专用凭证文件，只接受 pandadata 凭证键。"""
    for candidate in candidates or _env_candidates():
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip()
                if key in _CREDENTIAL_KEYS:
                    os.environ.setdefault(
                        key, v.strip().strip('"').strip("'")
                    )
            return


def ensure_login() -> str:
    """幂等登录。多次调用只登一次。"""
    global _TOKEN
    if _TOKEN:
        return _TOKEN
    _load_dotenv()
    user = os.environ.get("PANDADATA_USER") or os.environ.get("PANDADATA_MOBILE")
    pwd = os.environ.get("PANDADATA_PASSWORD")
    supplied_token = os.environ.get("PANDADATA_TOKEN", "").strip()
    if not supplied_token and (not user or not pwd):
        raise RuntimeError(
            "缺少 pandadata 临时令牌或账号密码。"
        )
    last_error = None
    for attempt in range(3):
        try:
            _TOKEN = establish_session(
                pdd,
                username=user or "",
                password=pwd or "",
                token=supplied_token,
            )
            return _TOKEN
        except PandaDataSecurityError:
            raise
        except Exception as exc:  # SDK/network errors must not expose request data
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError(
        f"pandadata 登录失败（{type(last_error).__name__}）"
    ) from None


def _cache_key(name: str, **kwargs) -> Path:
    raw = name + "|" + "|".join(f"{k}={kwargs[k]}" for k in sorted(kwargs))
    h = hashlib.md5(
        raw.encode(), usedforsecurity=False
    ).hexdigest()[:16]
    return CACHE_DIR / f"{name}_{h}.parquet"


def _cached(name: str, ttl_days: int = 7):
    """磁盘缓存装饰器。参数原样透传，结果以 parquet 存。"""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(**kwargs):
            path = _cache_key(name, **kwargs)
            if path.exists():
                age = (time.time() - path.stat().st_mtime) / 86400
                if age < ttl_days:
                    try:
                        return pd.read_parquet(path)
                    except Exception:
                        sys.stderr.write(
                            f"[WARN] 缓存读取失败，重新请求：{path.name}\n"
                        )
            df = fn(**kwargs)
            if df is not None and len(df) > 0:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(path)
                except Exception:
                    sys.stderr.write(
                        f"[WARN] 缓存写入失败：{path.name}\n"
                    )
            return df
        return inner
    return wrap


@_cached("index_weights", ttl_days=30)
def get_index_weights(index_symbol: str, date: str) -> pd.DataFrame:
    """指数成分与权重。date 用 YYYYMMDD。"""
    ensure_login()
    df = pdd.get_index_weights(index_symbol=index_symbol,
                                start_date=date, end_date=date)
    return df


def normalize_price_basis(price_basis: str = "pre_adjusted") -> str:
    """Return the canonical adjusted-price basis; raw prices are forbidden."""
    aliases = {
        "pre": "pre_adjusted",
        "qfq": "pre_adjusted",
        "post": "post_adjusted",
        "hfq": "post_adjusted",
    }
    canonical = aliases.get(str(price_basis).strip().lower(),
                            str(price_basis).strip().lower())
    if canonical not in PRICE_BASIS_TO_METHOD:
        raise ValueError(
            "协整研究只允许前复权(pre_adjusted)或后复权(post_adjusted)价格"
        )
    return canonical


@_cached("stock_daily_adjusted", ttl_days=1)
def get_stock_daily_adjusted(
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        price_basis: str = "pre_adjusted") -> pd.DataFrame:
    """Fetch adjusted A-share daily bars; never call the unadjusted endpoint."""
    price_basis = normalize_price_basis(price_basis)
    ensure_login()
    if isinstance(symbols, str):
        symbols = [symbols]
    symbols = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    if not symbols:
        raise ValueError("symbols 不能为空")

    from datetime import datetime, timedelta
    try:
        sd = datetime.strptime(start_date, "%Y%m%d")
        ed = datetime.strptime(end_date, "%Y%m%d")
    except ValueError as exc:
        raise ValueError("start_date/end_date 必须为 YYYYMMDD") from exc
    if sd > ed:
        raise ValueError("start_date 不得晚于 end_date")

    fetcher = getattr(pdd, PRICE_BASIS_TO_METHOD[price_basis], None)
    if not callable(fetcher):
        raise RuntimeError(
            f"当前 panda_data 不支持 {PRICE_BASIS_TO_METHOD[price_basis]}"
        )

    # panda_data limits one request to less than five years.
    parts = []
    cur = sd
    while cur <= ed:
        try:
            four_years_later = cur.replace(year=cur.year + 4)
        except ValueError:
            four_years_later = cur.replace(year=cur.year + 4, day=28)
        nxt = min(four_years_later - timedelta(days=1), ed)
        df = fetcher(
            symbol=symbols,
            start_date=cur.strftime("%Y%m%d"),
            end_date=nxt.strftime("%Y%m%d"),
            fields=fields,
        )
        if df is not None and len(df) > 0:
            parts.append(df)
        cur = nxt + timedelta(days=1)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates()


@_cached("stock_industry", ttl_days=30)
def get_stock_industry(stock_symbol: str, level: str = "L1") -> pd.DataFrame:
    ensure_login()
    return pdd.get_stock_industry(stock_symbol=stock_symbol, level=level)


def get_universe(index_codes: List[str], date: str) -> pd.DataFrame:
    """合并多个指数的最新成分股（去重）。返回 [symbol, index_symbol]。"""
    parts = []
    for idx in index_codes:
        df = get_index_weights(index_symbol=idx, date=date)
        if df is None or len(df) == 0:
            continue
        # 兼容不同字段名
        sym_col = next((c for c in df.columns if c.lower() in
                        ("stock_symbol", "symbol", "con_code", "code")), None)
        if sym_col is None:
            raise RuntimeError(f"未找到股票代码列，实际列={list(df.columns)}")
        sub = df[[sym_col]].rename(columns={sym_col: "symbol"})
        sub["index_symbol"] = idx
        parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=["symbol", "index_symbol"])
    out = pd.concat(parts, ignore_index=True).drop_duplicates()
    return out


def get_industry_map(symbols: List[str], level: str = "L1",
                     max_workers: Optional[int] = None) -> pd.DataFrame:
    """批量取行业分类，返回 [symbol, industry]。"""
    workers = max_workers or int(os.environ.get("SIMONS_INDUSTRY_WORKERS", "8"))
    workers = max(1, min(16, workers, len(symbols) or 1))

    def _load_one(s: str) -> dict:
        for attempt in range(3):
            try:
                df = get_stock_industry(stock_symbol=s, level=level)
                if df is None or len(df) == 0:
                    return {"symbol": s, "industry": "UNKNOWN"}
                ind_col = next(
                    (
                        c for c in df.columns
                        if "industry" in c.lower() or "name" in c.lower()
                    ),
                    df.columns[-1],
                )
                value = str(df[ind_col].iloc[0]).strip()
                return {
                    "symbol": s,
                    "industry": value if value else "UNKNOWN",
                }
            except Exception:
                if attempt < 2:
                    time.sleep(0.1 * (2 ** attempt))
        return {"symbol": s, "industry": "UNKNOWN"}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_load_one, symbols))
    return pd.DataFrame(rows)


def build_price_panel(symbols: List[str], start: str, end: str,
                      field: str = "close",
                      price_basis: str = "pre_adjusted") -> pd.DataFrame:
    """Build a positive adjusted-close panel for cointegration research."""
    price_basis = normalize_price_basis(price_basis)
    if str(field).lower() != "close":
        raise ValueError("协整价格面板只允许使用复权 close")
    df = get_stock_daily_adjusted(
        symbols=symbols,
        start_date=start,
        end_date=end,
        fields=[field],
        price_basis=price_basis,
    )
    if df is None or len(df) == 0:
        panel = pd.DataFrame()
        panel.attrs["price_basis"] = price_basis
        panel.attrs["source_method"] = PRICE_BASIS_TO_METHOD[price_basis]
        return panel
    date_col = next(
        (c for c in df.columns if c.lower() in ("trade_date", "date")), None
    )
    sym_col = next(
        (c for c in df.columns
         if c.lower() in ("symbol", "code", "stock_symbol")), None
    )
    px_col = next(
        (c for c in df.columns if c.lower() == str(field).lower()), None
    )
    if None in (date_col, sym_col, px_col):
        raise RuntimeError(
            "复权日线缺少 trade_date/symbol/close 必需字段"
        )
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col].astype(str))
    df[px_col] = pd.to_numeric(df[px_col], errors="coerce")
    df.loc[df[px_col] <= 0, px_col] = np.nan
    panel = df.pivot_table(index=date_col, columns=sym_col, values=px_col, aggfunc="last")
    panel = panel.sort_index()
    panel.attrs["price_basis"] = price_basis
    panel.attrs["source_method"] = PRICE_BASIS_TO_METHOD[price_basis]
    return panel


def print_login_status() -> int:
    try:
        ensure_login()
    except Exception as exc:
        print(f"[FAIL] pandadata 登录失败（{type(exc).__name__}）")
        return 1
    print("OK  pandadata 登录成功")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-login", action="store_true")
    args = ap.parse_args()
    if args.check_login:
        raise SystemExit(print_login_status())
