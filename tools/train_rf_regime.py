"""
Train RandomForestClassifier to predict TODAY's curve regime from 5-day-LAGGED
features. The 5-day lag is essential: without it, "current slope" trivially
predicts "current regime" (regime is defined by the slope).

Regime cutoffs (from backend/regime_classifier.py):
    slope < -4 -> "Steep Backwardation"
    -4 <= slope < -1 -> "Backwardation"
    -1 <= slope <= 1 -> "Flat"
    1 < slope <= 4 -> "Contango"
    slope > 4 -> "Steep Contango"

================================================================
RESULT: SKIP_RF  (beats_baseline=false, delta=-8.70pp)
================================================================
RF test accuracy 61.4% LOSES to persistence baseline 70.1%.
Cause: extreme class imbalance — 2021-2026 oil was 96% Backwardation,
only 3-4 Contango training samples. RF collapses to majority class.

DO NOT wire backend/data/rf_regime_model.joblib into the dashboard.
Hard-rule slope classifier in backend/regime_classifier.py remains
the production regime tagger.

To retry: collapse to binary Backwardation-vs-Other, extend training
window to include 2014-2016 / 2020 Apr-Jun Contango periods, OR apply
class_weight='balanced' + SMOTE. Re-run only if persistence is cleared
by ≥2pp AND every retained class has recall ≥ 0.3.
================================================================
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score


# --- Paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CL_XLSX = ROOT / "CL_data_trimmed_daily_close.xlsx"
RESULTS_DIR = ROOT / "tools" / "results"
RESULTS_PATH = RESULTS_DIR / "rf_regime_results.json"
MODEL_DIR = ROOT / "backend" / "data"
MODEL_PATH = MODEL_DIR / "rf_regime_model.joblib"

LAG = 5
REGIME_LABELS = [
    "Steep Backwardation",
    "Backwardation",
    "Flat",
    "Contango",
    "Steep Contango",
]


# --- Helpers ---------------------------------------------------------------

def label_regime(slope: float) -> str:
    if slope < -4:
        return "Steep Backwardation"
    if slope < -1:
        return "Backwardation"
    if slope <= 1:
        return "Flat"
    if slope <= 4:
        return "Contango"
    return "Steep Contango"


def load_cl_curves() -> pd.DataFrame:
    """Returns DataFrame indexed by date with columns m1, m3, m6, m9, m12."""
    df = pd.read_excel(CL_XLSX)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({
        "date": df["date"],
        "m1": df["c1||weighted_mid"],
        "m3": df["c3||weighted_mid"],
        "m6": df["c6||weighted_mid"],
        "m9": df["c9||weighted_mid"],
        "m12": df["c12||weighted_mid"],
    }).dropna()
    out = out.set_index("date").sort_index()
    return out


def load_dxy(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """DXY close series from yfinance, aligned to business days."""
    import yfinance as yf
    # DX-Y.NYB is the ICE US Dollar Index. Fall back to DXY if needed.
    pad_start = (start - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    pad_end = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_err = None
    for sym in ("DX-Y.NYB", "DX=F", "^DXY"):
        try:
            data = yf.download(
                sym,
                start=pad_start,
                end=pad_end,
                progress=False,
                auto_adjust=False,
            )
            if data is None or data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten when multi-symbol
                close = data["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
            else:
                close = data["Close"]
            close = close.dropna()
            if len(close) > 100:
                close.name = "dxy"
                return close
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not load DXY series; last error={last_err}")


def build_features(cl: pd.DataFrame, dxy: pd.Series) -> pd.DataFrame:
    """Build the feature matrix + label column. All features are 5-day-lagged
    so they cannot peek at today's regime."""
    df = cl.copy()

    # Curve geometry today (used for the target only)
    df["slope_today"] = df["m12"] - df["m1"]

    # Slope and curvature time series (we will lag these by 5d for features)
    slope = df["m12"] - df["m1"]
    curvature = df["m3"] - 2.0 * df["m6"] + df["m9"]

    df["lagged_slope"] = slope.shift(LAG)
    df["slope_5d_change"] = (slope.shift(LAG) - slope.shift(LAG + 5))
    df["slope_20d_change"] = (slope.shift(LAG) - slope.shift(LAG + 20))

    df["curvature_lag"] = curvature.shift(LAG)
    df["curvature_5d_change"] = (curvature.shift(LAG) - curvature.shift(LAG + 5))

    # 20-day realized vol of WTI front-month log returns, lagged 5 days
    log_ret = np.log(df["m1"] / df["m1"].shift(1))
    vol_20d = log_ret.rolling(window=20).std() * np.sqrt(252)
    df["vol_20d"] = vol_20d.shift(LAG)

    # DXY 20-day percent change, lagged 5 days
    dxy_aligned = dxy.reindex(df.index, method="ffill")
    dxy_20d_pct = (dxy_aligned / dxy_aligned.shift(20) - 1.0) * 100.0
    df["dxy_20d_change_pct"] = dxy_20d_pct.shift(LAG)

    # Seasonality (week of year; this is "known" today so no lag)
    iso_week = df.index.isocalendar().week.astype(float).to_numpy()
    angle = 2.0 * np.pi * iso_week / 52.0
    df["woy_sin"] = np.sin(angle)
    df["woy_cos"] = np.cos(angle)

    # Target: regime TODAY based on today's slope
    df["regime"] = df["slope_today"].apply(label_regime)

    feature_cols = [
        "lagged_slope",
        "slope_5d_change",
        "slope_20d_change",
        "curvature_lag",
        "curvature_5d_change",
        "vol_20d",
        "dxy_20d_change_pct",
        "woy_sin",
        "woy_cos",
    ]
    keep = feature_cols + ["regime", "slope_today"]
    out = df[keep].dropna().copy()
    return out, feature_cols


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] CL curves from {CL_XLSX}")
    cl = load_cl_curves()
    print(f"  -> {len(cl)} rows, {cl.index.min().date()} to {cl.index.max().date()}")

    print("[load] DXY from yfinance")
    dxy = load_dxy(cl.index.min(), cl.index.max())
    print(f"  -> DXY {len(dxy)} rows, {dxy.index.min().date()} to {dxy.index.max().date()}")

    print("[build] feature matrix")
    df, feature_cols = build_features(cl, dxy)
    print(f"  -> {len(df)} usable rows, {len(feature_cols)} features")
    print(f"  -> regime distribution:\n{df['regime'].value_counts()}")

    # Chronological 60/20/20 split
    n = len(df)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)
    n_test = n - n_train - n_val

    train = df.iloc[:n_train]
    # val is iloc[n_train : n_train + n_val] (held out, not used here)
    test = df.iloc[n_train + n_val:]

    X_train = train[feature_cols].to_numpy()
    y_train = train["regime"].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_test = test["regime"].to_numpy()

    print(f"[split] train={len(X_train)} val={n_val} test={len(X_test)}")

    # Persistence baseline: predict regime today == regime LAG days ago.
    # Use the slope from LAG days ago (which is `lagged_slope` feature, since
    # that is exactly slope.shift(LAG) on the same index).
    baseline_preds = test["lagged_slope"].apply(label_regime).to_numpy()
    baseline_acc = float(accuracy_score(y_test, baseline_preds))

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = float(accuracy_score(y_test, rf_preds))

    # Per-regime recall (compute against the set of labels present in y_test
    # union y_train so we cover the canonical 5)
    per_regime_recall = {}
    labels_present = sorted(set(REGIME_LABELS) & set(np.unique(np.concatenate([y_train, y_test]))))
    rec_arr = recall_score(
        y_test, rf_preds, labels=labels_present, average=None, zero_division=0
    )
    for lab, r in zip(labels_present, rec_arr):
        per_regime_recall[lab] = float(round(r, 4))

    # Feature importances
    importances = [
        {"feature": f, "importance": float(round(imp, 6))}
        for f, imp in sorted(
            zip(feature_cols, rf.feature_importances_),
            key=lambda x: -x[1],
        )
    ]

    delta_pp = float(round((rf_acc - baseline_acc) * 100.0, 4))

    results = {
        "script_path": str((Path(__file__)).resolve()).replace("\\", "/"),
        "results_path": str(RESULTS_PATH.resolve()).replace("\\", "/"),
        "model_path": str(MODEL_PATH.resolve()).replace("\\", "/"),
        "horizon_days": LAG,
        "n_train": int(len(X_train)),
        "n_val": int(n_val),
        "n_test": int(len(X_test)),
        "test_accuracy": float(round(rf_acc, 6)),
        "baseline_persistence_accuracy": float(round(baseline_acc, 6)),
        "beats_baseline": bool(rf_acc > baseline_acc),
        "delta_vs_baseline_pct": delta_pp,
        "per_regime_recall": per_regime_recall,
        "feature_importances": importances,
        "feature_columns": feature_cols,
        "regime_labels": REGIME_LABELS,
        "train_regime_counts": {
            str(k): int(v) for k, v in train["regime"].value_counts().to_dict().items()
        },
        "test_regime_counts": {
            str(k): int(v) for k, v in test["regime"].value_counts().to_dict().items()
        },
        "notes": (
            "Predicts TODAY's regime from features lagged 5 trading days. "
            "Persistence baseline = label(slope from 5d ago). RF must beat this "
            "to add information beyond 'regime is sticky'."
        ),
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    joblib.dump(
        {"model": rf, "feature_columns": feature_cols, "regime_labels": REGIME_LABELS},
        MODEL_PATH,
    )

    print(f"[done] RF acc={rf_acc:.4f}  persistence={baseline_acc:.4f}  "
          f"delta={delta_pp:+.2f}pp  beats={rf_acc > baseline_acc}")
    print(f"  results -> {RESULTS_PATH}")
    print(f"  model   -> {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
