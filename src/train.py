"""
Walk-forward multimodal earnings-event model.

Run:
    python src/train.py                  # full multimodal (numeric + text)
    python src/train.py --numeric-only   # numeric features only (faster)

Target: abnormal_ret_3d = ret_3d - mkt_ret_3d  (forward-looking, market-adjusted)
  - mkt_ret_3d is the forward 3-day S&P 500 return computed during dataset build.
  - pre_event_ret is a FEATURE only (see features.py for why).

Walk-forward: train on years 1..t-1, test on year t, starting from year 6.
OOS predictions from all test years are pooled for the final summary.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from features import (
    add_numeric_features,
    build_text_surprise,
    normalize_text_features,
    NUMERIC_FEATURE_NAMES,
)
from models import fit_ridge, fit_histgb, fit_hybrid
from evaluate import walk_forward_splits, eval_regression, selective_table, shuffle_test, hit_rate

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"

TARGET = "abnormal_ret_3d"
MIN_TRAIN_YEARS = 5


def _add_forward_mkt_return(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the FORWARD 3-day S&P 500 return for each event date and attach it.

    forward_mkt_ret_3d(t) = (sp500[t+3] - sp500[t]) / sp500[t]

    This is done by shifting the sp500 series 3 days back before pct_change,
    then merging to events via merge_asof.
    """
    sp = pd.read_parquet(DATA_RAW / "benchmarks" / "sp500_index.parquet")
    sp.columns = [c.lower() for c in sp.columns]
    sp["date"] = pd.to_datetime(sp["date"]).dt.normalize()
    if "adjclose" not in sp.columns and "close" in sp.columns:
        sp["adjclose"] = sp["close"]
    sp = sp.sort_values("date").reset_index(drop=True)

    # Forward return: shift prices 3 days into the past, then pct_change gives future return
    sp["mkt_ret_3d"] = sp["adjclose"].shift(-3).sub(sp["adjclose"]).div(sp["adjclose"])

    events = events.sort_values("date").reset_index(drop=True)
    events = pd.merge_asof(
        events,
        sp[["date", "mkt_ret_3d"]].rename(columns={"mkt_ret_3d": "_mkt_ret_3d_fwd"}),
        on="date",
        direction="backward",
    )
    events["mkt_ret_3d"] = events.pop("_mkt_ret_3d_fwd")
    return events.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_and_prepare(use_text: bool = True):
    events = pd.read_parquet(DATA_PROCESSED / "earnings_events.parquet")
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()

    assert "ret_3d" in events.columns, "Missing ret_3d — re-run build_dataset.py"

    # Compute forward market return if not already present
    if "mkt_ret_3d" not in events.columns:
        events = _add_forward_mkt_return(events)

    # FIX 1: target is purely market-adjusted forward return.
    # pre_event_ret is NOT subtracted here — it is a feature only (see features.py).
    events[TARGET] = events["ret_3d"] - events["mkt_ret_3d"]
    events[TARGET] = events[TARGET].clip(
        events[TARGET].quantile(0.01), events[TARGET].quantile(0.99)
    )

    events = add_numeric_features(events)

    if not use_text:
        model_df = events.dropna(subset=[TARGET, "year"]).copy()
        return model_df, None, None

    # Align embeddings: meta and .npy are both sorted by (date, symbol).
    # Sort events the same way, then assign embeddings positionally.
    E = np.load(DATA_PROCESSED / "finbert_embeddings.npy")
    meta = pd.read_parquet(DATA_PROCESSED / "finbert_embeddings_meta.parquet")
    meta["date"] = pd.to_datetime(meta["date"]).dt.normalize()
    assert len(meta) == len(E), "Meta row count must equal embedding row count"

    # Sort events to match embedding order, then align by position
    text_events = events.sort_values(["date", "symbol"]).reset_index(drop=True)
    assert len(text_events) == len(E), (
        f"Event count ({len(text_events)}) != embedding count ({len(E)}). "
        "Re-run build_dataset.py or check finbert_embeddings_meta.parquet."
    )
    # Validate that (date, symbol) pairs match between sorted events and meta
    sym_match = (text_events["symbol"].values == meta["symbol"].values).mean()
    if sym_match < 0.99:
        raise ValueError(
            f"Embedding alignment only {sym_match:.1%} — sort order mismatch. "
            "Check that earnings_events.parquet matches finbert_embeddings_meta.parquet."
        )

    # Build raw text surprises globally (per-firm EWMA, shift(1) — no leakage).
    # Normalization happens inside each fold (see normalize_text_features).
    txt_surp = build_text_surprise(text_events, E)
    text_events = pd.concat([text_events, txt_surp], axis=1)

    model_df = text_events.dropna(subset=[TARGET, "year"]).copy()
    txt_surp_cols = list(txt_surp.columns)
    return model_df, txt_surp_cols, E


def run_backtest(model_df, txt_surp_cols, use_text):
    feat_cols = [f for f in NUMERIC_FEATURE_NAMES if f in model_df.columns]
    rows, pred_store = [], []

    for train_years, test_year, tr_idx, te_idx in walk_forward_splits(
        model_df, min_train_years=MIN_TRAIN_YEARS
    ):
        train = model_df.loc[tr_idx]
        test = model_df.loc[te_idx]
        y_train = train[TARGET].values
        y_test = test[TARGET].values

        X_num_tr = train[feat_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_num_te = test[feat_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if use_text and txt_surp_cols:
            # FIX 3: normalize using training-set stats only
            z_tr, z_te = normalize_text_features(
                train[txt_surp_cols], test[txt_surp_cols]
            )
            X_tr = pd.concat([X_num_tr.reset_index(drop=True), z_tr.reset_index(drop=True)], axis=1)
            X_te = pd.concat([X_num_te.reset_index(drop=True), z_te.reset_index(drop=True)], axis=1)
        else:
            X_tr, X_te = X_num_tr, X_num_te

        pred_ridge = fit_ridge(X_tr, y_train, X_te)
        pred_hgb = fit_histgb(X_tr, y_train, X_te)
        pred_hyb, p_up = fit_hybrid(X_tr, y_train, X_te)

        hr = hit_rate(pred_hgb, y_test)
        print(f"  {int(test_year)} — HistGB hit={hr:.3f}  n={len(y_test)}")

        for name, pred in [("ridge", pred_ridge), ("histgb", pred_hgb), ("hybrid", pred_hyb)]:
            m = eval_regression(y_test, pred)
            m.update({"model": name, "test_year": int(test_year)})
            rows.append(m)

        pred_store.append(pd.DataFrame({
            "test_year": int(test_year),
            "y": y_test,
            "ridge": pred_ridge,
            "histgb": pred_hgb,
            "hybrid": pred_hyb,
            "p_up": p_up,
        }))

    return pd.DataFrame(rows), pd.concat(pred_store, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-only", action="store_true", help="Skip text features")
    args = parser.parse_args()
    use_text = not args.numeric_only

    mode = "numeric-only" if not use_text else "multimodal"
    print(f"\n=== Earnings Event Model — {mode} ===")
    print(f"Target: {TARGET}  (market-adjusted forward 3-day return)")
    print(f"Walk-forward: min {MIN_TRAIN_YEARS} training years\n")

    model_df, txt_surp_cols, _ = load_and_prepare(use_text=use_text)
    print(f"Modeling frame: {model_df.shape}  ({model_df['year'].min():.0f}–{model_df['year'].max():.0f})")

    results, pred_df = run_backtest(model_df, txt_surp_cols, use_text)

    print("\n=== OOS Summary (pooled across all test years) ===")
    summary = results.groupby("model")[["mae", "rmse", "r2", "hit_rate"]].mean()
    print(summary.sort_values("rmse").to_string())

    print("\n=== Selective Prediction — Hybrid (confidence = |prediction|) ===")
    print(selective_table(pred_df, "hybrid").to_string(index=False))

    print("\n=== Shuffle Sanity Check (Ridge) ===")
    check = shuffle_test(pred_df, "ridge")
    for k, v in check.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
