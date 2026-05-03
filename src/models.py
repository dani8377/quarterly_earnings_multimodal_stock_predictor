import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor


def fit_ridge(X_train, y_train, X_test):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_histgb(X_train, y_train, X_test):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("histgb", HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=600,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_hybrid(X_train, y_train, X_test):
    """Sign classifier (Logistic) × magnitude regressor (HistGB).

    Returns (predictions, p_up) where p_up is the probability of a positive return.
    Use |predictions| as a confidence score for selective evaluation.
    """
    y_sign = (y_train > 0).astype(int)
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(max_iter=3000)),
    ])
    clf.fit(X_train, y_sign)
    p_up = clf.predict_proba(X_test)[:, 1]
    sign = np.where(p_up >= 0.5, 1.0, -1.0)

    mag_pred = fit_histgb(X_train, np.abs(y_train), X_test)
    return sign * mag_pred, p_up
