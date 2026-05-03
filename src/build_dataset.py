"""
Build earnings_events.parquet from raw data.

Run:
    python src/build_dataset.py

Reads from data/raw/ and writes data/processed/earnings_events.parquet.

Key design choices:
  - ret_3d and mkt_ret_3d are both FORWARD-looking (post-event prices).
    Do NOT recompute mkt_ret_3d from pct_change(3) in model code — that is
    backward-looking and produces an incorrect abnormal return.
  - pre_event_ret is NOT added here; it is added in features.py as a feature.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

HORIZONS = (1, 3, 5)
MOM_DAYS = 20


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_prices(symbol: str) -> pd.DataFrame | None:
    path = RAW / "daily_prices" / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")


def load_sp500() -> pd.DataFrame | None:
    path = RAW / "benchmarks" / "sp500_index.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if "adjclose" not in df.columns and "close" in df.columns:
        df["adjclose"] = df["close"]
    return df[["date", "adjclose"]]


def load_fundamentals(symbol: str) -> pd.DataFrame:
    """Merge income statement, balance sheet, and cash flow for one symbol."""
    dfs = []
    for subdir in ("income_statement", "balance_sheet", "cash_flow"):
        p = RAW / "quarterly_financials" / subdir / f"{symbol}.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        return pd.DataFrame()
    base = dfs[0]
    for other in dfs[1:]:
        merge_keys = [c for c in ["symbol", "year", "quarter"] if c in base.columns and c in other.columns]
        dup_cols = [c for c in other.columns if c in base.columns and c not in merge_keys]
        other = other.drop(columns=dup_cols, errors="ignore")
        base = base.merge(other, on=merge_keys, how="outer")
    return base


def load_news(symbol: str) -> pd.DataFrame | None:
    path = RAW / "news" / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "publishedDate" in df.columns:
        df["date"] = pd.to_datetime(df["publishedDate"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        return None
    return df.dropna(subset=["date"])


# ---------------------------------------------------------------------------
# Return computation — FORWARD-LOOKING
# ---------------------------------------------------------------------------

def _forward_returns(prices: pd.DataFrame, anchor: pd.Timestamp, horizons=HORIZONS) -> dict:
    """
    Compute forward returns relative to the last close BEFORE the event.
    p0 = last close before anchor date.
    ret_h = (post_event_close[h] - p0) / p0
    """
    pre = prices[prices["date"] < anchor].tail(1)
    post = prices[prices["date"] >= anchor]
    if pre.empty or len(post) < max(horizons):
        return {h: np.nan for h in horizons}
    p0 = pre.iloc[-1]["adjclose"]
    return {h: (post.iloc[h - 1]["adjclose"] - p0) / p0 for h in horizons}


def compute_event_returns(row, prices: pd.DataFrame) -> pd.Series:
    r = _forward_returns(prices, row.date)
    return pd.Series([r[1], r[3], r[5]], index=["ret_1d", "ret_3d", "ret_5d"])


def compute_mkt_returns(row, sp500: pd.DataFrame) -> pd.Series:
    """Forward S&P 500 return over the same windows as the stock return."""
    r = _forward_returns(sp500, row.date)
    return pd.Series([r[1], r[3], r[5]], index=["mkt_ret_1d", "mkt_ret_3d", "mkt_ret_5d"])


def compute_pre_event_features(row, prices: pd.DataFrame) -> pd.Series:
    hist = prices[prices["date"] < row.date].tail(MOM_DAYS + 1)
    if len(hist) < MOM_DAYS + 1:
        return pd.Series([np.nan, np.nan], index=["mom_20d", "vol_20d"])
    rets = hist["adjclose"].pct_change().dropna()
    mom = (hist["adjclose"].iloc[-1] - hist["adjclose"].iloc[0]) / hist["adjclose"].iloc[0]
    return pd.Series([mom, rets.tail(MOM_DAYS).std()], index=["mom_20d", "vol_20d"])


# ---------------------------------------------------------------------------
# News features
# ---------------------------------------------------------------------------

def news_features(row, news: pd.DataFrame | None) -> pd.Series:
    if news is None:
        return pd.Series([0, 0.0], index=["news_count_7d", "news_sentiment_7d"])
    window = news[(news["date"] >= row.date - pd.Timedelta(days=7)) & (news["date"] < row.date)]
    count = len(window)
    sent_col = "sentiment_score" if "sentiment_score" in news.columns else "sentiment"
    sentiment = window[sent_col].mean() if count > 0 and sent_col in window.columns else 0.0
    return pd.Series([count, float(sentiment)], index=["news_count_7d", "news_sentiment_7d"])


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build():
    earnings = pd.read_parquet(RAW / "earnings_calls" / "0000.parquet")
    earnings["date"] = pd.to_datetime(earnings["date"])
    earnings["symbol"] = earnings["symbol"].astype(str)
    earnings = (
        earnings
        .sort_values("date")
        .drop_duplicates(subset=["symbol", "year", "quarter"])
        .reset_index(drop=True)
    )
    print(f"Earnings events: {earnings.shape}")

    sp500 = load_sp500()
    if sp500 is None:
        print("[WARN] S&P 500 benchmark not found — mkt_ret_* will be NaN")

    ret_rows, mkt_rows, pre_rows, news_rows = [], [], [], []

    for _, row in tqdm(earnings.iterrows(), total=len(earnings), desc="events"):
        prices = load_prices(row.symbol)

        if prices is not None:
            ret_rows.append(compute_event_returns(row, prices))
            pre_rows.append(compute_pre_event_features(row, prices))
        else:
            ret_rows.append(pd.Series([np.nan] * 3, index=["ret_1d", "ret_3d", "ret_5d"]))
            pre_rows.append(pd.Series([np.nan, np.nan], index=["mom_20d", "vol_20d"]))

        if sp500 is not None:
            mkt_rows.append(compute_mkt_returns(row, sp500))
        else:
            mkt_rows.append(pd.Series([np.nan] * 3, index=["mkt_ret_1d", "mkt_ret_3d", "mkt_ret_5d"]))

        n = load_news(row.symbol)
        news_rows.append(news_features(row, n))

    earnings = pd.concat([
        earnings,
        pd.DataFrame(ret_rows),
        pd.DataFrame(mkt_rows),
        pd.DataFrame(pre_rows),
        pd.DataFrame(news_rows),
    ], axis=1)

    # Merge quarterly fundamentals
    symbols = earnings["symbol"].unique()
    fund_frames = []
    for sym in tqdm(symbols, desc="fundamentals"):
        fund_frames.append(load_fundamentals(sym))

    if fund_frames:
        fundamentals = pd.concat([f for f in fund_frames if not f.empty], ignore_index=True)
        merge_keys = [c for c in ["symbol", "year", "quarter"] if c in fundamentals.columns]
        dup_cols = [c for c in fundamentals.columns if c in earnings.columns and c not in merge_keys]
        fundamentals = fundamentals.drop(columns=dup_cols, errors="ignore")
        earnings = earnings.merge(fundamentals, on=merge_keys, how="left")

    earnings = earnings.replace([np.inf, -np.inf], np.nan)
    earnings = earnings.dropna(subset=["ret_1d"]).reset_index(drop=True)

    out = PROCESSED / "earnings_events.parquet"
    earnings.to_parquet(out, index=False)
    print(f"\nWrote {earnings.shape} → {out}")
    print(f"Symbols: {earnings['symbol'].nunique()}")
    print(f"Years: {int(earnings['year'].min())}–{int(earnings['year'].max())}")


if __name__ == "__main__":
    build()
