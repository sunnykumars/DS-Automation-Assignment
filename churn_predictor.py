import json, numpy as np, pandas as pd
from typing import List, Dict, Optional
from sklearn.preprocessing import MinMaxScaler
from pycaret.classification import load_model, predict_model

_MODEL = None
_SCHEMA: Dict = None

def load_churn_model(path: str = "churn_best_model"):
    """Load model + training schema (expects training_schema.json in CWD)."""
    global _MODEL, _SCHEMA
    _MODEL = load_model(path)
    with open("training_schema.json","r") as f:
        _SCHEMA = json.load(f)
    return _MODEL

def _ensure_loaded():
    if _MODEL is None or _SCHEMA is None:
        raise RuntimeError("Model/schema not loaded. Call load_churn_model().")

def _to_numeric_safe(s): return pd.to_numeric(s, errors="coerce")

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop IDs, reconcile names, build ChargePerMonth if needed, one-hot (train was one-hot), align cols."""
    _ensure_loaded()
    X_cols: List[str] = list(_SCHEMA["x_columns"])
    is_onehot = bool(_SCHEMA.get("train_is_onehot", True))  # your train is one-hot
    rename_map = _SCHEMA.get("rename_map", {})

    X = df.copy()

    # drop IDs / target if present
    for col in ["customerID","CustomerID","Churn","churn","label","target","Churn_Yes"]:
        if col in X.columns: X = X.drop(columns=[col])

    # rename to match training (e.g., charge_per_tenure -> ChargePerMonth)
    X = X.rename(columns=rename_map)

    # ensure numeric columns are numeric
    for c in ["tenure","MonthlyCharges","TotalCharges","ChargePerMonth"]:
        if c in X.columns: X[c] = _to_numeric_safe(X[c])

    # build ChargePerMonth if missing and data available
    if "ChargePerMonth" not in X.columns:
        if {"TotalCharges","tenure"}.issubset(X.columns):
            denom = X["tenure"].replace(0, np.nan)
            X["ChargePerMonth"] = (X["TotalCharges"] / denom).fillna(0)
        else:
            X["ChargePerMonth"] = 0

    # training used pre-dummied columns; dummy + align
    if is_onehot:
        X = pd.get_dummies(X)
        X = X.reindex(columns=X_cols, fill_value=0)
    else:
        X = X[[c for c in X.columns if c in X_cols]]

    return X

def _proba_from_model(X: pd.DataFrame, positive_label: int = 1) -> Optional[np.ndarray]:
    """Try sklearn proba directly."""
    try:
        proba = _MODEL.predict_proba(X)
        # If Pipeline, classes_ may be on the final step:
        classes = getattr(_MODEL, "classes_", None)
        if classes is None and hasattr(_MODEL, "named_steps"):
            est = _MODEL.named_steps.get("trained_model", None)
            if est is None and hasattr(_MODEL, "steps") and len(_MODEL.steps):
                est = _MODEL.steps[-1][1]
            classes = getattr(est, "classes_", None)
        if classes is not None:
            # find index of positive label (1)
            if positive_label in list(classes):
                idx = list(classes).index(positive_label)
            else:
                # assume binary second column is positive
                idx = 1 if proba.shape[1] > 1 else 0
            return proba[:, idx]
        # no classes_ — assume binary, second column is positive
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
    except Exception:
        return None

def _proba_from_decision_function(X: pd.DataFrame) -> Optional[np.ndarray]:
    """Last resort: scale decision_function to [0,1]."""
    try:
        df = _MODEL.decision_function(X)
        df = np.asarray(df).reshape(-1, 1)
        return MinMaxScaler().fit_transform(df).ravel()
    except Exception:
        return None

def predict_churn_proba(df: pd.DataFrame, positive_label: int = 1) -> pd.Series:
    """Return per-row probability of churn."""
    X = _prepare_features(df)

    # 1) pycaret.predict_model with various column names
    try:
        preds = predict_model(_MODEL, data=X, raw_score=True, verbose=False)
        # Common possibilities:
        for col in (f"Score_{positive_label}", "prediction_score", "Score", "score"):
            if col in preds.columns: return preds[col]
        # Any score-like column
        score_like = [c for c in preds.columns if c.lower().startswith("score")]
        if score_like: return preds[score_like[-1]]
    except Exception:
        pass

    # 2) Fallback to sklearn predict_proba
    proba = _proba_from_model(X, positive_label=positive_label)
    if proba is not None: return pd.Series(proba, index=X.index, name="prob")

    # 3) Final fallback: decision_function scaled to [0,1]
    df_scaled = _proba_from_decision_function(X)
    if df_scaled is not None: return pd.Series(df_scaled, index=X.index, name="prob")

    raise ValueError("No probability column found; model provides neither prob nor decision_function.")

def predict_churn_label(df: pd.DataFrame, threshold: float = 0.5, positive_label: int = 1) -> pd.Series:
    proba = predict_churn_proba(df, positive_label=positive_label)
    return (proba >= threshold).astype(int)
