import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def hit_rate(pred, true):
    return float((np.sign(np.asarray(pred)) == np.sign(np.asarray(true))).mean())


def eval_regression(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "hit_rate": hit_rate(y_pred, y_true),
    }


def walk_forward_splits(df, year_col="year", min_train_years=5):
    years = sorted(df[year_col].dropna().unique())
    for i in range(min_train_years, len(years)):
        train_years = years[:i]
        test_year = years[i]
        yield (
            train_years,
            test_year,
            df[df[year_col].isin(train_years)].index,
            df[df[year_col] == test_year].index,
        )


def selective_table(pred_df, model_col, quantiles=(0.6, 0.7, 0.8, 0.9, 0.95)):
    conf = np.abs(pred_df[model_col].values)
    rows = [{
        "confidence": "All",
        "coverage": 1.0,
        "n": len(pred_df),
        "hit_rate": hit_rate(pred_df[model_col], pred_df["y"]),
    }]
    for q in quantiles:
        thr = np.quantile(conf, q)
        mask = conf >= thr
        rows.append({
            "confidence": f"Top {int((1 - q) * 100)}%",
            "coverage": float(mask.mean()),
            "n": int(mask.sum()),
            "hit_rate": hit_rate(pred_df.loc[mask, model_col], pred_df.loc[mask, "y"]),
        })
    return pd.DataFrame(rows)


def shuffle_test(pred_df, model_col, seed=42):
    """Sanity check: hit rate after shuffling targets should collapse to ~0.50."""
    y_shuf = pred_df["y"].sample(frac=1.0, random_state=seed).values
    return {
        "always_long_hit": float((pred_df["y"] > 0).mean()),
        "real_hit": hit_rate(pred_df[model_col], pred_df["y"]),
        "shuffled_hit": hit_rate(pred_df[model_col].values, y_shuf),
    }
