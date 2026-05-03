"""
Feature engineering for earnings-event stock prediction.

Leakage fixes:
  1. Target is `abnormal_ret = ret - mkt_ret` only. `pre_event_ret` is a feature,
     never subtracted from the target.
  2. Cross-sectional peer features use a running (expanding) peer median sorted
     by event date within each group. Only prior reporters contribute.
  3. Text normalization is done per walk-forward fold (in train.py).

Improvements:
  4. Analyst consensus EPS/revenue surprise replaces EWMA when available.
     (EWMA is a weak proxy for market expectations; analyst estimates are
     what stocks are actually priced against.)
  5. Cross-sectional grouping uses (sector, year, quarter) when sector data
     is available, giving a tighter peer comparison than all firms in a quarter.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    return s.clip(s.quantile(lower), s.quantile(upper))


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = a.astype(float), b.astype(float)
    return pd.Series(np.where(np.abs(b) > 1e-12, a / b, np.nan), index=a.index)


# ---------------------------------------------------------------------------
# Time-series helpers (per firm, no leakage — shift(1) before rolling)
# ---------------------------------------------------------------------------

def _ewma_expectation(s: pd.Series, span: int = 8, min_periods: int = 3) -> pd.Series:
    return s.shift(1).ewm(span=span, adjust=False, min_periods=min_periods).mean()


def _rolling_std(s: pd.Series, window: int = 8, min_periods: int = 3) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=min_periods).std()


# ---------------------------------------------------------------------------
# Running peer median (no within-quarter lookahead)
# ---------------------------------------------------------------------------

def _running_peer_median(
    df: pd.DataFrame,
    col: str,
    group_cols: list[str] | None = None,
    date_col: str = "date",
    min_peers: int = 3,
) -> pd.Series:
    """
    For each event, compute the median of `col` among all events in the same
    group that were dated STRICTLY BEFORE the current event.

    group_cols defaults to ["year", "quarter"]. Pass ["sector", "year", "quarter"]
    when sector data is available for a tighter peer comparison.
    """
    if group_cols is None:
        group_cols = ["year", "quarter"]

    valid_groups = [g for g in group_cols if g in df.columns]
    if not valid_groups:
        return pd.Series(np.nan, index=df.index)

    df_s = df.sort_values(valid_groups + [date_col])
    result = (
        df_s
        .groupby(valid_groups)[col]
        .transform(lambda s: s.expanding(min_periods=min_peers).median().shift(1))
    )
    return result.reindex(df.index)


# ---------------------------------------------------------------------------
# Numeric feature engineering
# ---------------------------------------------------------------------------

FUND_COLS = [
    "eps", "revenue", "ebitda", "operatingIncome",
    "freeCashFlow", "netIncome", "operatingCashFlow",
]

NUMERIC_FEATURE_NAMES = [
    # Analyst consensus surprises (improvement 4) — used when available
    "eps_surprise_analyst", "rev_surprise_analyst",
    # EWMA-based surprises (fallback / complementary signal)
    "surp_eps_z", "surp_revenue_z", "surp_freeCashFlow_z",
    "surp_ebitda_z", "surp_operatingIncome_z",
    # Sector-aware running peer cross-sectional context (improvement 5)
    "surp_eps_z_cs", "surp_revenue_z_cs", "surp_freeCashFlow_z_cs",
    # Margins + QoQ inflection
    "gross_margin", "op_margin", "gross_margin_qoq", "op_margin_qoq",
    # Earnings quality
    "accruals_ratio",
    # Leverage / liquidity
    "net_debt_to_assets", "cash_to_assets",
    # Market context
    "pre_event_ret", "news_count_7d", "news_sentiment_7d",
]


def add_numeric_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    Add all numeric features. Safe to run on the full dataset before the
    walk-forward split — every operation is either per-firm time-series with
    shift(1), a running peer median, or a static ratio/diff.
    """
    events = events.copy().sort_values(["symbol", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Improvement 4: analyst consensus EPS/revenue surprise
    # These columns are populated by build_dataset.py when analyst data
    # has been fetched. They're a much stronger signal than EWMA because
    # they reflect what the market actually priced in before the release.
    # ------------------------------------------------------------------
    if "eps_estimate" in events.columns and "eps" in events.columns:
        raw = (events["eps"] - events["eps_estimate"]) / (events["eps_estimate"].abs() + 1e-8)
        events["eps_surprise_analyst"] = winsorize(raw)

    if "rev_estimate" in events.columns and "revenue" in events.columns:
        raw = (events["revenue"] - events["rev_estimate"]) / (events["rev_estimate"].abs() + 1e-8)
        events["rev_surprise_analyst"] = winsorize(raw)

    # ------------------------------------------------------------------
    # EWMA expectations + z-scored surprises (fallback / complementary)
    # ------------------------------------------------------------------
    for c in FUND_COLS:
        if c not in events.columns:
            continue
        exp = events.groupby("symbol")[c].transform(lambda s: _ewma_expectation(s))
        std = events.groupby("symbol")[c].transform(lambda s: _rolling_std(s))
        z = (events[c] - exp) / (std + 1e-8)
        events[f"surp_{c}_z"] = winsorize(z)

    # ------------------------------------------------------------------
    # Improvement 5: sector-aware running peer CS context
    # Uses (sector, year, quarter) when sector is available, otherwise
    # falls back to (year, quarter).
    # ------------------------------------------------------------------
    cs_group = (
        ["sector", "year", "quarter"]
        if "sector" in events.columns
        else ["year", "quarter"]
    )
    cs_source_cols = [
        "surp_eps_z", "surp_revenue_z", "surp_freeCashFlow_z",
        "surp_ebitda_z", "surp_operatingIncome_z",
    ]
    for c in cs_source_cols:
        if c not in events.columns:
            continue
        peer_med = _running_peer_median(events, c, group_cols=cs_group)
        events[f"{c}_cs"] = winsorize(events[c] - peer_med)

    # ------------------------------------------------------------------
    # Margins + QoQ changes
    # ------------------------------------------------------------------
    if "grossProfit" in events.columns and "revenue" in events.columns:
        events["gross_margin"] = safe_div(events["grossProfit"], events["revenue"])
        events["gross_margin_qoq"] = events.groupby("symbol")["gross_margin"].diff()

    if "operatingIncome" in events.columns and "revenue" in events.columns:
        events["op_margin"] = safe_div(events["operatingIncome"], events["revenue"])
        events["op_margin_qoq"] = events.groupby("symbol")["op_margin"].diff()

    # Earnings quality (accruals)
    if all(c in events.columns for c in ["netIncome", "operatingCashFlow", "totalAssets"]):
        events["accruals_ratio"] = safe_div(
            events["netIncome"] - events["operatingCashFlow"], events["totalAssets"]
        )
        events["accruals_ratio"] = winsorize(events["accruals_ratio"])

    # Leverage / liquidity
    if "netDebt" in events.columns and "totalAssets" in events.columns:
        events["net_debt_to_assets"] = safe_div(events["netDebt"], events["totalAssets"])
    if "cashAndCashEquivalents" in events.columns and "totalAssets" in events.columns:
        events["cash_to_assets"] = safe_div(
            events["cashAndCashEquivalents"], events["totalAssets"]
        )

    # pre_event_ret is a feature only — never part of the target
    if "ret_1d" in events.columns:
        events["pre_event_ret"] = events.groupby("symbol")["ret_1d"].shift(1)
        events["pre_event_ret"] = winsorize(events["pre_event_ret"])

    return events


# ---------------------------------------------------------------------------
# Text feature engineering
# ---------------------------------------------------------------------------

def build_text_surprise(events: pd.DataFrame, E: np.ndarray) -> pd.DataFrame:
    """
    Compute per-firm text surprise: current embedding minus EWMA expectation.

    Uses shift(1) per firm — no temporal leakage. Normalization and PCA
    compression are done per fold in train.py (improvement 3).

    Args:
        events: DataFrame aligned row-for-row with E.
        E:      (N, D) float32 array of FinBERT embeddings.

    Returns:
        DataFrame of raw text surprises, shape (N, D), columns txt_surp_0..D-1.
    """
    assert len(events) == len(E), (
        f"Embedding count ({len(E)}) != event count ({len(events)}). "
        "Ensure events and E share the same sort order."
    )

    txt = pd.DataFrame(
        E.astype(np.float32),
        index=events.index,
        columns=[f"txt_{i}" for i in range(E.shape[1])],
    )

    exp = txt.groupby(events["symbol"].values).transform(
        lambda s: s.shift(1).ewm(span=8, adjust=False, min_periods=3).mean()
    )

    surp = txt - exp
    surp.columns = [f"txt_surp_{i}" for i in range(E.shape[1])]
    return surp
