"""
Walk-forward multimodal earnings-event model.

Run:
    python src/train.py                        # multimodal, 1-day target (recommended)
    python src/train.py --target 3d            # 3-day target
    python src/train.py --numeric-only         # skip text features

Target: abnormal_ret = ret_Nd - mkt_ret_Nd  (forward market-adjusted return)
  - Default horizon is 1d: post-earnings drift is strongest on day 1.
  - pre_event_ret is a FEATURE only — never subtracted from the target.

Text fusion: per-fold PCA(50) on raw text surprises, fit on training data only.
  - Compresses 768 embedding dimensions to 50 principal components.
  - Fit on training fold, applied to test fold — no leakage.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import add_numeric_features, build_text_surprise, NUMERIC_FEATURE_NAMES
from models import fit_ridge, fit_histgb, fit_hybrid
from evaluate import walk_forward_splits, eval_regression, selective_table, shuffle_test, hit_rate

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"

MIN_TRAIN_YEARS = 5
TEXT_PCA_COMPONENTS = 50


# ---------------------------------------------------------------------------
# Market return (forward-looking)
# ---------------------------------------------------------------------------

def _add_forward_mkt_returns(events: pd.DataFrame, horizons: tuple = (1, 3)) -> pd.DataFrame:
    """
    Attach forward S&P 500 returns for each horizon to events.

    forward_mkt_ret_Nd(t) = (sp500[t+N] - sp500[t]) / sp500[t]
    Computed by shifting prices N days forward before pct_change.
    """
    sp = pd.read_parquet(DATA_RAW / "benchmarks" / "sp500_index.parquet")
    sp.columns = [c.lower() for c in sp.columns]
    sp["date"] = pd.to_datetime(sp["date"]).dt.normalize()
    if "adjclose" not in sp.columns and "close" in sp.columns:
        sp["adjclose"] = sp["close"]
    sp = sp.sort_values("date").reset_index(drop=True)

    for h in horizons:
        sp[f"mkt_ret_{h}d"] = sp["adjclose"].shift(-h).sub(sp["adjclose"]).div(sp["adjclose"])

    mkt_cols = [f"mkt_ret_{h}d" for h in horizons]
    events = events.sort_values("date").reset_index(drop=True)
    events = pd.merge_asof(events, sp[["date"] + mkt_cols], on="date", direction="backward")
    return events.sort_values(["symbol", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# In-fold PCA for text features
# ---------------------------------------------------------------------------

def _pca_text(surp_tr: pd.DataFrame, surp_te: pd.DataFrame,
              n_components: int = TEXT_PCA_COMPONENTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit PCA on training text surprises only, then transform both sets.

    Pipeline: constant-impute NaNs → StandardScale → PCA(n_components).
    Fitting on training data only prevents test-set distribution leakage.
    This also compresses 768 raw dimensions to 50, preventing the text
    features from drowning out the 18 numeric features.
    """
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])
    Z_tr = pipe.fit_transform(surp_tr.values)
    Z_te = pipe.transform(surp_te.values)
    cols = [f"txt_pc_{i}" for i in range(n_components)]
    return pd.DataFrame(Z_tr, columns=cols), pd.DataFrame(Z_te, columns=cols)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_prepare(target_horizon: str = "1d", use_text: bool = True):
    events = pd.read_parquet(DATA_PROCESSED / "earnings_events.parquet")
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()

    h = int(target_horizon[0])
    ret_col = f"ret_{h}d"
    mkt_col = f"mkt_ret_{h}d"
    target = f"abnormal_ret_{h}d"

    assert ret_col in events.columns, f"Missing {ret_col} — re-run build_dataset.py"

    if mkt_col not in events.columns:
        events = _add_forward_mkt_returns(events, horizons=(h,))

    events[target] = events[ret_col] - events[mkt_col]
    events[target] = events[target].clip(
        events[target].quantile(0.01), events[target].quantile(0.99)
    )

    events = add_numeric_features(events)

    if not use_text:
        return events.dropna(subset=[target, "year"]).copy(), None, None, target

    # Align embeddings: meta and .npy share (date, symbol) sort order
    E = np.load(DATA_PROCESSED / "finbert_embeddings.npy")
    meta = pd.read_parquet(DATA_PROCESSED / "finbert_embeddings_meta.parquet")
    meta["date"] = pd.to_datetime(meta["date"]).dt.normalize()
    assert len(meta) == len(E)

    text_events = events.sort_values(["date", "symbol"]).reset_index(drop=True)
    assert len(text_events) == len(E), (
        f"Event count ({len(text_events)}) != embedding count ({len(E)}). "
        "Re-run build_dataset.py."
    )
    sym_match = (text_events["symbol"].values == meta["symbol"].values).mean()
    if sym_match < 0.99:
        raise ValueError(f"Embedding alignment only {sym_match:.1%} — sort order mismatch.")

    txt_surp = build_text_surprise(text_events, E)
    text_events = pd.concat([text_events, txt_surp], axis=1)

    model_df = text_events.dropna(subset=[target, "year"]).copy()
    return model_df, list(txt_surp.columns), E, target


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def run_backtest(model_df: pd.DataFrame, txt_surp_cols, use_text: bool, target: str):
    feat_cols = [f for f in NUMERIC_FEATURE_NAMES if f in model_df.columns]
    rows, pred_store = [], []

    for train_years, test_year, tr_idx, te_idx in walk_forward_splits(
        model_df, min_train_years=MIN_TRAIN_YEARS
    ):
        train = model_df.loc[tr_idx]
        test = model_df.loc[te_idx]
        y_train = train[target].values
        y_test = test[target].values

        X_num_tr = train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_num_te = test[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if use_text and txt_surp_cols:
            pc_tr, pc_te = _pca_text(train[txt_surp_cols], test[txt_surp_cols])
            X_tr = pd.concat([X_num_tr.reset_index(drop=True), pc_tr], axis=1)
            X_te = pd.concat([X_num_te.reset_index(drop=True), pc_te], axis=1)
        else:
            X_tr, X_te = X_num_tr, X_num_te

        pred_ridge = fit_ridge(X_tr, y_train, X_te)
        pred_hgb = fit_histgb(X_tr, y_train, X_te)
        pred_hyb, p_up = fit_hybrid(X_tr, y_train, X_te)

        print(f"  {int(test_year)} — HistGB hit={hit_rate(pred_hgb, y_test):.3f}  n={len(y_test)}")

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument("--target", choices=["1d", "3d"], default="1d",
                        help="Return horizon for the target (default: 1d)")
    args = parser.parse_args()
    use_text = not args.numeric_only

    mode = "numeric-only" if not use_text else f"multimodal (PCA={TEXT_PCA_COMPONENTS})"
    print(f"\n=== Earnings Event Model — {mode} ===")
    print(f"Target horizon: {args.target}  |  min training years: {MIN_TRAIN_YEARS}\n")

    model_df, txt_surp_cols, _, target = load_and_prepare(
        target_horizon=args.target, use_text=use_text
    )
    print(f"Modeling frame: {model_df.shape}  "
          f"({int(model_df['year'].min())}–{int(model_df['year'].max())})\n")

    results, pred_df = run_backtest(model_df, txt_surp_cols, use_text, target)

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
