"""
=============================================================
RESULT: SKIP  (beats_baseline=false, delta=-35.31pp)
=============================================================
Percentile RF (45.3% test) LOST to persistence baseline (80.6%) by 35pp.
Cause: train-time percentile cutoffs (fit on 2021-2024 deep backwardation)
became stale by test time (late 2025-2026 curve flattened). 60% of test
landed in Quintile_5 by definition; Q2/Q3 (middle quintiles in train)
shrank to 8 and 9 samples in test. Train accuracy 88.6% vs test 45.3% =
43pp overfit gap.

DO NOT wire backend/data/rf_regime_percentile_model.joblib into production.
The hard-rule slope classifier in backend/regime_classifier.py remains the
production regime tagger, with 5-day persistence as the forecasting baseline.

If you must retry: use ROLLING percentile cutoffs (refit every 60d), or
switch to SLOPE-CHANGE percentiles (momentum, not level). Both lose
predictive power to the persistence baseline at slow horizons.
=============================================================

Train RandomForestClassifier to predict TODAY's percentile-based curve regime
from 5-day-lagged features.

Difference vs train_rf_regime.py:
  - Regime labels are derived from the 20/40/60/80 PERCENTILES of the M12-M1
    slope distribution on the TRAINING slice ONLY. That guarantees ~20% of
    train rows per class instead of the hard cutoffs (-4, -1, 1, 4) which
    produced 96% Backwardation in 2021-2026 and starved RF of signal.

  - Labels are Quintile_1 (deepest backwardation) through Quintile_5 (deepest
    contango). The SAME cutoffs fitted on TRAIN are then applied to TEST.

  - Same 9 features, same chronological 60/20/20 split, same persistence
    baseline (regime today == regime label for slope 5 days ago using the
    train-fitted cutoffs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score


# --- Paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CL_XLSX = ROOT / "CL_data_trimmed_daily_close.xlsx"
RESULTS_DIR = ROOT / "tools" / "results"
RESULTS_PATH = RESULTS_DIR / "rf_regime_percentile_results.json"
MODEL_DIR = ROOT / "backend" / "data"
MODEL_PATH = MODEL_DIR / "rf_regime_percentile_model.joblib"

LAG = 5
REGIME_LABELS = [
    "Quintile_1",  # bottom 20% slope -> deepest backwardation
    "Quintile_2",
    "Quintile_3",  # middle / flat
    "Quintile_4",
    "Quintile_5",  # top 20% slope -> deepest contango
]


# --- Helpers ---------------------------------------------------------------

def label_from_cutoffs(slope: float, c20: float, c40: float, c60: float, c80: float) -> str:
    if slope <= c20:
        return "Quintile_1"
    if slope <= c40:
        return "Quintile_2"
    if slope <= c60:
        return "Quintile_3"
    if slope <= c80:
        return "Quintile_4"
    return "Quintile_5"


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
    # Keep last 5 years
    cutoff = out.index.max() - pd.DateOffset(years=5)
    out = out.loc[out.index >= cutoff]
    return out


def load_dxy(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """DXY close series from yfinance, aligned to business days."""
    import yfinance as yf
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


def build_features(cl: pd.DataFrame, dxy: pd.Series):
    """Build feature matrix; label column is added later once train-percentile
    cutoffs are known."""
    df = cl.copy()

    # Slope today (for labeling)
    df["slope_today"] = df["m12"] - df["m1"]

    slope = df["m12"] - df["m1"]
    curvature = df["m3"] - 2.0 * df["m6"] + df["m9"]

    df["lagged_slope"] = slope.shift(LAG)
    df["slope_5d_change"] = slope.shift(LAG) - slope.shift(LAG + 5)
    df["slope_20d_change"] = slope.shift(LAG) - slope.shift(LAG + 20)

    df["curvature_lag"] = curvature.shift(LAG)
    df["curvature_5d_change"] = curvature.shift(LAG) - curvature.shift(LAG + 5)

    log_ret = np.log(df["m1"] / df["m1"].shift(1))
    vol_20d = log_ret.rolling(window=20).std() * np.sqrt(252)
    df["vol_20d"] = vol_20d.shift(LAG)

    dxy_aligned = dxy.reindex(df.index, method="ffill")
    dxy_20d_pct = (dxy_aligned / dxy_aligned.shift(20) - 1.0) * 100.0
    df["dxy_20d_change_pct"] = dxy_20d_pct.shift(LAG)

    iso_week = df.index.isocalendar().week.astype(float).to_numpy()
    angle = 2.0 * np.pi * iso_week / 52.0
    df["woy_sin"] = np.sin(angle)
    df["woy_cos"] = np.cos(angle)

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
    keep = feature_cols + ["slope_today"]
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
    print(f"  -> {len(df)} usable rows")

    # Chronological 60/20/20 split
    n = len(df)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)
    n_test = n - n_train - n_val

    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train + n_val:].copy()

    # Fit percentile cutoffs on TRAIN slope only
    train_slope = train["slope_today"].to_numpy()
    c20 = float(np.percentile(train_slope, 20))
    c40 = float(np.percentile(train_slope, 40))
    c60 = float(np.percentile(train_slope, 60))
    c80 = float(np.percentile(train_slope, 80))
    percentile_cutoffs = {
        "20pct": round(c20, 6),
        "40pct": round(c40, 6),
        "60pct": round(c60, 6),
        "80pct": round(c80, 6),
    }
    print(f"[cutoffs] {percentile_cutoffs}")

    # Apply cutoffs to label train + test
    train["regime"] = train["slope_today"].apply(
        lambda s: label_from_cutoffs(s, c20, c40, c60, c80)
    )
    test["regime"] = test["slope_today"].apply(
        lambda s: label_from_cutoffs(s, c20, c40, c60, c80)
    )

    train_dist = train["regime"].value_counts().to_dict()
    test_dist = test["regime"].value_counts().to_dict()
    print(f"[dist] train: {train_dist}")
    print(f"[dist] test:  {test_dist}")

    X_train = train[feature_cols].to_numpy()
    y_train = train["regime"].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_test = test["regime"].to_numpy()

    print(f"[split] train={len(X_train)} val={n_val} test={len(X_test)}")

    # Persistence baseline: label(lagged_slope) using the train-fitted cutoffs
    baseline_preds = test["lagged_slope"].apply(
        lambda s: label_from_cutoffs(s, c20, c40, c60, c80)
    ).to_numpy()
    baseline_acc = float(accuracy_score(y_test, baseline_preds))

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_train_preds = rf.predict(X_train)
    rf_test_preds = rf.predict(X_test)
    train_acc = float(accuracy_score(y_train, rf_train_preds))
    test_acc = float(accuracy_score(y_test, rf_test_preds))

    # Per-regime recall over ALL 5 labels
    rec_arr = recall_score(
        y_test, rf_test_preds, labels=REGIME_LABELS, average=None, zero_division=0
    )
    per_regime_recall = {
        lab: float(round(r, 4)) for lab, r in zip(REGIME_LABELS, rec_arr)
    }

    importances = [
        {"feature": f, "importance": float(round(imp, 6))}
        for f, imp in sorted(
            zip(feature_cols, rf.feature_importances_),
            key=lambda x: -x[1],
        )
    ]

    delta_pp = float(round((test_acc - baseline_acc) * 100.0, 4))

    # Test skew check
    test_total = sum(test_dist.values())
    test_max_share = max(test_dist.values()) / test_total if test_total else 0.0
    skew_note = ""
    if test_max_share > 0.60:
        dominant = max(test_dist, key=test_dist.get)
        skew_note = (
            f" WARNING: test set is skewed -- {dominant} = "
            f"{test_max_share*100:.1f}% of test rows, indicating a regime shift "
            f"between train and test windows; recall for under-represented "
            f"classes may be 0."
        )

    train_class_distribution = {
        lab: int(train_dist.get(lab, 0)) for lab in REGIME_LABELS
    }
    test_class_distribution = {
        lab: int(test_dist.get(lab, 0)) for lab in REGIME_LABELS
    }

    results = {
        "script_path": str(Path(__file__).resolve()).replace("\\", "/"),
        "results_path": str(RESULTS_PATH.resolve()).replace("\\", "/"),
        "model_path": str(MODEL_PATH.resolve()).replace("\\", "/"),
        "horizon_days": LAG,
        "n_train": int(len(X_train)),
        "n_val": int(n_val),
        "n_test": int(len(X_test)),
        "train_accuracy": float(round(train_acc, 6)),
        "test_accuracy": float(round(test_acc, 6)),
        "baseline_persistence_accuracy": float(round(baseline_acc, 6)),
        "beats_baseline": bool(test_acc > baseline_acc),
        "delta_vs_baseline_pct": delta_pp,
        "percentile_cutoffs": percentile_cutoffs,
        "per_regime_recall": per_regime_recall,
        "feature_importances": importances,
        "feature_columns": feature_cols,
        "regime_labels": REGIME_LABELS,
        "train_class_distribution": train_class_distribution,
        "test_class_distribution": test_class_distribution,
        "notes": (
            "Percentile-based regime classifier. Slope cutoffs (20/40/60/80) "
            "fitted on TRAIN slope distribution only, then applied to TEST. "
            "Predicts today's quintile from 5-day-lagged features. Persistence "
            "baseline = label(slope from 5d ago) using the same train-fitted "
            f"cutoffs.{skew_note}"
        ),
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    joblib.dump(
        {
            "model": rf,
            "feature_columns": feature_cols,
            "regime_labels": REGIME_LABELS,
            "percentile_cutoffs": percentile_cutoffs,
        },
        MODEL_PATH,
    )

    print(f"[done] RF train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
          f"persistence={baseline_acc:.4f}  delta={delta_pp:+.2f}pp  "
          f"beats={test_acc > baseline_acc}")
    print(f"  results -> {RESULTS_PATH}")
    print(f"  model   -> {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
