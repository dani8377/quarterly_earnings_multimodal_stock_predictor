"""
Fetch all raw data from Financial Modeling Prep API.

Run:
    FMP_API_KEY=<your_key> python src/ingest.py

Writes to data/raw/:
    daily_prices/<SYMBOL>.parquet
    quarterly_financials/{income_statement,balance_sheet,cash_flow}/<SYMBOL>.parquet
    benchmarks/sp500_index.parquet
    benchmarks/msci_acwi.parquet
    news/<SYMBOL>.parquet

Set FMP_API_KEY in your environment (do not hard-code it here).
Resume-safe: already-fetched symbols are skipped.
"""

import json
import os
import time
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"

API_KEY = os.environ.get("FMP_API_KEY", "")
if not API_KEY:
    raise EnvironmentError("Set FMP_API_KEY environment variable before running ingest.")

BASE_URL = "https://financialmodelingprep.com/stable"
PRICES_URL = "https://financialmodelingprep.com/api/v3"

LIMIT = 80
MIN_DATE = "2000-01-01"
SLEEP = 0.3
MAX_RETRIES = 5

_sentiment = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _get(url: str, params: dict[str, Any]) -> Any:
    params = {**params, "apikey": API_KEY}
    backoff = 1.0
    for _ in range(MAX_RETRIES):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            try:
                return r.json()
            except json.JSONDecodeError:
                return []
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff *= 2.0
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.url}")
    raise RuntimeError(f"Max retries exceeded: {url}")


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Quarterly fundamentals
# ---------------------------------------------------------------------------

ENDPOINTS = {
    "income_statement": "income-statement",
    "balance_sheet": "balance-sheet-statement",
    "cash_flow": "cash-flow-statement",
}


def fetch_fundamentals(symbol: str, endpoint_key: str) -> pd.DataFrame:
    data = _get(f"{BASE_URL}/{ENDPOINTS[endpoint_key]}", {"symbol": symbol, "period": "quarter", "limit": LIMIT})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["symbol"] = symbol
    for col in ("fiscalDateEnding", "date"):
        if col in df.columns:
            df["date"] = pd.to_datetime(df[col], errors="coerce")
            break
    if "date" in df.columns:
        df = df[df["date"] >= pd.to_datetime(MIN_DATE)]
        df["year"] = df["date"].dt.year
        df["quarter"] = df["date"].dt.quarter
    return df


def ingest_fundamentals(symbols: list[str]):
    for category, _ in ENDPOINTS.items():
        out_dir = RAW / "quarterly_financials" / category
        out_dir.mkdir(parents=True, exist_ok=True)
        done = {p.stem for p in out_dir.glob("*.parquet")}
        todo = [s for s in symbols if s not in done]
        print(f"\n{category}: {len(todo)}/{len(symbols)} remaining")
        for sym in tqdm(todo, desc=category):
            try:
                df = fetch_fundamentals(sym, category)
                (df if not df.empty else pd.DataFrame({"symbol": [sym]})).to_parquet(
                    out_dir / f"{sym}.parquet", index=False
                )
            except Exception as e:
                print(f"  [WARN] {sym}: {e}")
            time.sleep(SLEEP)


# ---------------------------------------------------------------------------
# Daily prices
# ---------------------------------------------------------------------------

def fetch_prices(symbol: str) -> pd.DataFrame:
    data = _get(f"{PRICES_URL}/historical-price-full/{symbol}", {"from": MIN_DATE, "to": _today()})
    if not data or "historical" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["historical"])
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] >= pd.to_datetime(MIN_DATE)].sort_values("date")


def ingest_prices(symbols: list[str]):
    out_dir = RAW / "daily_prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in out_dir.glob("*.parquet")}
    todo = [s for s in symbols if s not in done]
    print(f"\ndaily_prices: {len(todo)}/{len(symbols)} remaining")
    for sym in tqdm(todo, desc="prices"):
        try:
            df = fetch_prices(sym)
            (df if not df.empty else pd.DataFrame({"symbol": [sym]})).to_parquet(
                out_dir / f"{sym}.parquet", index=False
            )
        except Exception as e:
            print(f"  [WARN] {sym}: {e}")
        time.sleep(SLEEP)


# ---------------------------------------------------------------------------
# Benchmark indices
# ---------------------------------------------------------------------------

def fetch_index(symbol: str) -> pd.DataFrame:
    data = _get(f"{PRICES_URL}/historical-price-full/{symbol}", {"from": MIN_DATE, "to": _today()})
    if not data or "historical" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["historical"])
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] >= pd.to_datetime(MIN_DATE)].sort_values("date")


def ingest_benchmarks():
    out_dir = RAW / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol, name in [("^GSPC", "sp500_index"), ("ACWI", "msci_acwi")]:
        out = out_dir / f"{name}.parquet"
        if out.exists():
            print(f"  {name}: already exists, skipping")
            continue
        try:
            df = fetch_index(symbol)
            if not df.empty:
                df.to_parquet(out, index=False)
                print(f"  {name}: {len(df)} rows saved")
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
        time.sleep(SLEEP)


# ---------------------------------------------------------------------------
# News + sentiment
# ---------------------------------------------------------------------------

def _headline_sentiment(title: str) -> float:
    if not isinstance(title, str) or not title.strip():
        return 0.0
    return _sentiment.polarity_scores(title)["compound"]


def fetch_news(symbol: str) -> pd.DataFrame:
    start = pd.to_datetime(MIN_DATE)
    end = pd.to_datetime(_today())
    rows: list[dict] = []

    cur = start
    while cur <= end:
        to = min(cur + timedelta(days=89), end)
        for page in range(20):
            data = _get(
                f"{PRICES_URL}/stock_news",
                {"tickers": symbol, "from": cur.strftime("%Y-%m-%d"), "to": to.strftime("%Y-%m-%d"), "page": page},
            )
            if not data:
                break
            rows.extend(data if isinstance(data, list) else [data])
            time.sleep(SLEEP)
        cur = to + timedelta(days=1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["symbol"] = symbol
    if "publishedDate" in df.columns:
        df["date"] = pd.to_datetime(df["publishedDate"], errors="coerce")
        df = df.sort_values("date")
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"])
    if "title" in df.columns:
        df["sentiment_score"] = df["title"].astype(str).map(_headline_sentiment)
    return df


def ingest_news(symbols: list[str]):
    out_dir = RAW / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in out_dir.glob("*.parquet")}
    todo = [s for s in symbols if s not in done]
    print(f"\nnews: {len(todo)}/{len(symbols)} remaining")
    for sym in tqdm(todo, desc="news"):
        try:
            df = fetch_news(sym)
            (df if not df.empty else pd.DataFrame({"symbol": [sym]})).to_parquet(
                out_dir / f"{sym}.parquet", index=False
            )
        except Exception as e:
            print(f"  [WARN] {sym}: {e}")
        time.sleep(SLEEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    transcripts = pd.read_parquet(RAW / "earnings_calls" / "0000.parquet")
    symbols = sorted(transcripts["symbol"].dropna().astype(str).unique().tolist())
    print(f"Total symbols: {len(symbols)}")

    ingest_fundamentals(symbols)
    ingest_prices(symbols)
    ingest_benchmarks()
    ingest_news(symbols)
    print("\nAll done.")


if __name__ == "__main__":
    main()
