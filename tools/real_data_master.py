"""Master analysis on the user's REAL curve data (5 xlsx files).

Inputs (project root):
  CL_data_trimmed_daily_close.xlsx              WTI M1..M14
  LCO_data_trimmed_daily_close.xlsx             Brent M1..M17
  LGO_data_trimmed_daily_close.xlsx             Gasoil M1..M14
  HO_data_trimmed_daily_close.xlsx              Heating Oil M1..M14
  wtcl_lco_outrights_1min_trimmed_daily_close.xlsx   WTCL-LCO spread M1..M12

For each product:
  1. Build M1..M12 daily price panel.
  2. Compute spreads SPR_i_j = M_i - M_j  (j = i+1)
  3. Compute flies   FLY_i_j_k = M_i - 2*M_j + M_k  (consecutive triplets)
  4. For each target series compute features:
       - lagged returns (1d, 5d, 20d) of the series itself
       - DXY 5d return (real macro, from yfinance)
       - 20d realized vol
       - Curve slope (M12-M1) of the SAME product (real, from this file)
       - Curve curvature (M3 - 2*M6 + M9)
  5. 60/20/20 chronological split (train/val/test).
  6. Fit Linear/Ridge/Lasso/Huber/LGBM/XGBoost. Pick winner by val R^2,
     report test R^2.
  7. From winner: today's prediction -> LONG/SHORT/FLAT signal.

NO synthetic macro data. Only DXY (yfinance, real) and curve-derived
features (real, from these files). If a feature is missing it's left
out of the model.

Outputs:
  - tools/results/real_data_verify.json    file vs dashboard M1..M12 check
  - tools/results/real_data_signals.json   per-product/spread/fly signals
  - tools/results/real_data_models.json    per-target model winners + R^2
"""
from __future__ import annotations
import json
import math
import warnings
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                   HuberRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.svm import SVR, LinearSVR


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "tools" / "results"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 5  # predict 5d forward change
NUM_TENORS = 12

PRODUCT_FILES = {
    "CL":   ROOT / "CL_data_trimmed_daily_close.xlsx",
    "LCO":  ROOT / "LCO_data_trimmed_daily_close.xlsx",
    "LGO":  ROOT / "LGO_data_trimmed_daily_close.xlsx",
    "HO":   ROOT / "HO_data_trimmed_daily_close.xlsx",
    "WTCL": ROOT / "wtcl_lco_outrights_1min_trimmed_daily_close.xlsx",
}


# ----------------------------- data loading ------------------------------ #
def load_curve(xlsx_path: Path) -> pd.DataFrame:
    """Return a DataFrame indexed by date with columns m1..m12 (float)."""
    df = pd.read_excel(xlsx_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    out = pd.DataFrame(index=df.index)
    for i in range(1, NUM_TENORS + 1):
        col = f"c{i}||weighted_mid"
        if col in df.columns:
            out[f"m{i}"] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[f"m{i}"] = np.nan
    return out


def load_dxy_5y() -> pd.Series:
    raw = yf.download("DX-Y.NYB", period="5y", progress=False,
                       auto_adjust=True, threads=True)["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return raw


# ------------------ macro / feature engineering -------------------------- #
def make_features(series: pd.Series, dxy: pd.Series,
                   curve: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Build feature matrix for ONE target series.

    For outrights (kind="ret") we predict next-5d % return.
    For spreads/flies (kind="diff") we predict next-5d delta of the spread.
    """
    df = pd.DataFrame(index=series.index)
    if kind == "ret":
        df["ret_1d"]  = series.pct_change(1)  * 100
        df["ret_5d"]  = series.pct_change(5)  * 100
        df["ret_20d"] = series.pct_change(20) * 100
        df["target"]  = (series.shift(-HORIZON) / series - 1) * 100
    else:
        df["ret_1d"]  = series.diff(1)
        df["ret_5d"]  = series.diff(5)
        df["ret_20d"] = series.diff(20)
        df["target"]  = series.shift(-HORIZON) - series

    # DXY 5d return (real macro)
    dxy_a = dxy.reindex(series.index, method="ffill")
    df["dxy_5d_ret"] = dxy_a.pct_change(5) * 100

    # Realized vol of the series (real, derived)
    logret = np.log(series.replace(0, np.nan) /
                     series.replace(0, np.nan).shift(1))
    df["vol_20d"] = logret.rolling(20).std() * math.sqrt(252) * 100

    # Curve slope (M12-M1) and curvature (M3-2M6+M9) — both real, in-file
    m1 = curve.get("m1"); m12 = curve.get("m12")
    if m1 is not None and m12 is not None:
        df["slope"] = (m12 - m1).reindex(series.index, method="ffill")
    else:
        df["slope"] = 0.0
    m3 = curve.get("m3"); m6 = curve.get("m6"); m9 = curve.get("m9")
    if all(x is not None for x in (m3, m6, m9)):
        df["curvature"] = (m3 - 2*m6 + m9).reindex(series.index, method="ffill")
    else:
        df["curvature"] = 0.0

    df = df.dropna()
    feat_cols = ["ret_1d", "ret_5d", "ret_20d", "dxy_5d_ret",
                 "vol_20d", "slope", "curvature"]
    return df[feat_cols + ["target"]]


# --------------- 60/20/20 chronological split + model stack -------------- #
def build_models():
    return {
        "Linear":     LinearRegression(),
        "Ridge":      Ridge(alpha=1.0),
        "Lasso":      Lasso(alpha=0.05, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000),
        "Huber":      HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=400),
        "LGBM":       LGBMRegressor(
            n_estimators=60, max_depth=3, num_leaves=6,
            learning_rate=0.03, min_child_samples=8,
            min_data_in_bin=2, min_split_gain=0.0,
            reg_alpha=0.5, reg_lambda=0.5,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            random_state=42, verbosity=-1, force_col_wise=True),
        "XGBoost":    xgb.XGBRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.03,
            min_child_weight=4, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=0.5,
            random_state=42, verbosity=0, n_jobs=1),
        # Support Vector Regression — 3 variants because the kernel choice
        # is the strategic decision, not a hyperparameter to ignore:
        #   SVR_RBF:    non-linear gaussian kernel — captures curvature
        #               + interactions like LGBM but in continuous feature space
        #   SVR_Linear: linear kernel via LinearSVR — ε-insensitive loss
        #               (robust like Huber but with explicit sparsity)
        #   SVR_Poly:   degree-2 polynomial — explicit pairwise interactions
        # All use ε=0.1 tube and C=1.0 (StandardScaler is already applied).
        "SVR_RBF":     SVR(kernel="rbf",   C=1.0, gamma="scale", epsilon=0.1),
        "SVR_Linear":  LinearSVR(C=1.0, epsilon=0.1, max_iter=5000,
                                  random_state=42, dual="auto"),
        "SVR_Poly":    SVR(kernel="poly",  C=1.0, degree=2, gamma="scale",
                            epsilon=0.1, coef0=1.0),
    }


def split_60_20_20(X, y):
    n = len(X)
    a = int(n * 0.60)
    b = int(n * 0.80)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def working_effect_lag1(series: pd.Series) -> float:
    """Lag-1 autocorrelation of daily DIFFS — diagnostic for the Working effect
    on weighted_mid-derived series. Negative readings (<-0.20) indicate that
    a chunk of the series' mean reversion is mechanical (price-averaging
    artifact, not tradeable). Used to flag CL/LCO fly signals as having
    inflated R² in the dashboard panels."""
    d = series.diff().dropna()
    if len(d) < 30:
        return 0.0
    try:
        return float(d.autocorr(1))
    except Exception:
        return 0.0


def we_haircut_pct(lag1: float) -> int:
    """Suggested PnL/R² haircut percentage given lag-1 autocorr of diffs."""
    if lag1 > -0.05:
        return 0
    if lag1 > -0.20:
        return 10
    if lag1 > -0.30:
        return 20
    return 35


def run_one_target(df_feat: pd.DataFrame) -> Dict:
    if len(df_feat) < 200:
        return {"insufficient": True, "n": int(len(df_feat))}
    feat_cols = [c for c in df_feat.columns if c != "target"]
    X = df_feat[feat_cols].values.astype(float)
    y = df_feat["target"].values.astype(float)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_60_20_20(X, y)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xva_s = sc.transform(Xva)
    Xte_s = sc.transform(Xte)

    results = {}
    last_feat = sc.transform(X[-1:].reshape(1, -1)) if len(X) else None
    for name, m in build_models().items():
        try:
            m.fit(Xtr_s, ytr)
            r2_va = float(r2_score(yva, m.predict(Xva_s)))
            r2_te = float(r2_score(yte, m.predict(Xte_s)))
            mae_te = float(mean_absolute_error(yte, m.predict(Xte_s)))
            pred_now = float(m.predict(last_feat)[0]) if last_feat is not None else float("nan")
            results[name] = {
                "val_r2":  round(r2_va, 4),
                "test_r2": round(r2_te, 4),
                "test_mae": round(mae_te, 4),
                "pred_now": round(pred_now, 4),
            }
        except Exception as e:
            results[name] = {"error": str(e)[:80]}

    valid = {k: v for k, v in results.items() if "val_r2" in v}
    winner = (max(valid.items(), key=lambda kv: kv[1]["val_r2"])[0]
              if valid else None)
    return {
        "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "models": results,
        "winner": winner,
        "winner_test_r2": valid[winner]["test_r2"] if winner else None,
        "winner_pred_now": valid[winner]["pred_now"] if winner else None,
    }


# -------------------- verification vs dashboard -------------------------- #
def verify_against_dashboard(curves: Dict[str, pd.DataFrame]) -> Dict:
    """The dashboard's term_structure_summary.py only stores RESULT stats
    (R2, RMSE, last act, last pred) — not raw M1..M12. So we can only
    check: was the regression model's `act` close to the file's actual
    M-on-M value for the matching structure on the matching date?
    """
    try:
        sys.path.insert(0, str(ROOT / "backend"))
        from data import term_structure_summary as tss
    except Exception as e:
        return {"available": False, "reason": str(e)}

    checks = []
    summary = tss.SUMMARY
    for (prod, struct, kind), (model, r2, rmse, act, pred, n_test) in summary.items():
        # Translate dashboard product code to file product code
        file_prod = {"CL": "CL", "WTCL": "WTCL", "LCO3": "LCO",
                      "LGO": "LGO", "HO": "HO"}.get(prod)
        if not file_prod or file_prod not in curves:
            continue
        curve = curves[file_prod]
        last_dt = tss.LAST_DATE.get(prod)
        if not last_dt:
            continue
        target_dt = pd.to_datetime(last_dt)
        # nearest available date <= target
        avail = curve.index[curve.index <= target_dt]
        if len(avail) == 0:
            continue
        dt = avail[-1]
        if struct.startswith("SPR"):
            _, i, j = struct.split("_")
            mi, mj = f"m{i}", f"m{j}"
            if mi not in curve.columns or mj not in curve.columns:
                continue
            file_val = float(curve.loc[dt, mi]) - float(curve.loc[dt, mj])
        else:
            _, i, j, k = struct.split("_")
            mi, mj, mk = f"m{i}", f"m{j}", f"m{k}"
            if not all(c in curve.columns for c in (mi, mj, mk)):
                continue
            file_val = (float(curve.loc[dt, mi])
                        - 2 * float(curve.loc[dt, mj])
                        + float(curve.loc[dt, mk]))
        # For "level" kind, dashboard `act` SHOULD equal file level.
        # For "change" kind, dashboard `act` is the last 1d change of level.
        # We only verify level here.
        if kind != "level":
            continue
        diff = file_val - float(act)
        pct = abs(diff) / abs(float(act)) * 100 if act != 0 else float("nan")
        checks.append({
            "product": prod, "struct": struct,
            "date": dt.strftime("%Y-%m-%d"),
            "file_value": round(file_val, 4),
            "dashboard_act": round(float(act), 4),
            "abs_diff":   round(diff, 4),
            "pct_diff":   round(pct, 2),
            "match":      bool(pct < 2.0),  # within 2% counts as a match
        })
    n = len(checks)
    matched = sum(1 for c in checks if c["match"])
    return {
        "available": True,
        "checked":   n,
        "matched":   matched,
        "match_pct": round(matched / n * 100, 1) if n else 0,
        "samples":   checks[:10],
        "mismatches": [c for c in checks if not c["match"]][:10],
    }


# ---------------------- main orchestration ------------------------------- #
def main():
    print("Loading 5 xlsx files...")
    curves: Dict[str, pd.DataFrame] = {}
    for prod, path in PRODUCT_FILES.items():
        curves[prod] = load_curve(path)
        print(f"  {prod:<6}  rows={len(curves[prod])}  range={curves[prod].index[0].date()} -> {curves[prod].index[-1].date()}")

    print()
    print("Verifying file data against dashboard term_structure_summary...")
    verify = verify_against_dashboard(curves)
    if verify.get("available"):
        print(f"  checked: {verify['checked']}  matched within 2%: {verify['matched']} ({verify['match_pct']}%)")
        print(f"  samples (first 5):")
        for s in verify["samples"][:5]:
            mark = "OK" if s["match"] else "MISMATCH"
            print(f"    {s['product']:<5} {s['struct']:<14} {s['date']}  file={s['file_value']:>+9.3f}  dash={s['dashboard_act']:>+9.3f}  diff={s['pct_diff']:.1f}%  [{mark}]")
        if verify["mismatches"]:
            print(f"  first 5 mismatches:")
            for s in verify["mismatches"][:5]:
                print(f"    {s['product']:<5} {s['struct']:<14} {s['date']}  file={s['file_value']:>+9.3f}  dash={s['dashboard_act']:>+9.3f}  diff={s['pct_diff']:.1f}%")
    (OUT / "real_data_verify.json").write_text(json.dumps(verify, indent=2))

    print()
    print("Pulling DXY (real macro factor)...")
    dxy = load_dxy_5y()
    print(f"  DXY: {len(dxy)} rows  {dxy.index[0].date()} -> {dxy.index[-1].date()}")

    print()
    print("Running 60/20/20 regression backtest across all targets...")
    print("=" * 100)
    model_results: Dict[str, Dict] = {}
    signals: List[Dict] = []

    for prod, curve in curves.items():
        # ---- OUTRIGHTS: each tenor m1..m12 ----
        for i in range(1, NUM_TENORS + 1):
            col = f"m{i}"
            if col not in curve.columns:
                continue
            ser = curve[col].dropna()
            if len(ser) < 250:
                continue
            df_feat = make_features(ser, dxy, curve, kind="ret")
            res = run_one_target(df_feat)
            if "winner" not in res:
                continue
            key = f"{prod}|OUT|{col}"
            model_results[key] = res
            # Signal from winner
            if res["winner"] is not None:
                pred = res["winner_pred_now"]
                # threshold = 0.5% predicted 5d return
                sig = "LONG" if pred > 0.5 else ("SHORT" if pred < -0.5 else "FLAT")
                we = working_effect_lag1(ser)
                signals.append({
                    "kind": "OUTRIGHT", "product": prod, "tenor": col,
                    "label": f"{prod} {col.upper()}",
                    "current_level": round(float(ser.iloc[-1]), 4),
                    "winner_model": res["winner"],
                    "test_r2": res["winner_test_r2"],
                    "predicted_5d": pred,
                    "signal": sig,
                    "we_lag1": round(we, 3),
                    "we_haircut_pct": we_haircut_pct(we),
                })

        # ---- SPREADS: M_i - M_(i+1) for i=1..11 ----
        for i in range(1, NUM_TENORS):
            j = i + 1
            mi, mj = f"m{i}", f"m{j}"
            if mi not in curve.columns or mj not in curve.columns:
                continue
            ser = (curve[mi] - curve[mj]).dropna()
            if len(ser) < 250:
                continue
            df_feat = make_features(ser, dxy, curve, kind="diff")
            res = run_one_target(df_feat)
            if "winner" not in res:
                continue
            key = f"{prod}|SPR|m{i}-m{j}"
            model_results[key] = res
            if res["winner"] is not None:
                pred = res["winner_pred_now"]
                vol_threshold = float(np.median(np.abs(np.diff(ser.values)))) * 0.5
                sig = ("LONG" if pred > vol_threshold else
                       "SHORT" if pred < -vol_threshold else "FLAT")
                we = working_effect_lag1(ser)
                signals.append({
                    "kind": "SPREAD", "product": prod,
                    "label": f"{prod} M{i}-M{j}",
                    "current_level": round(float(ser.iloc[-1]), 4),
                    "winner_model": res["winner"],
                    "test_r2": res["winner_test_r2"],
                    "predicted_5d": pred,
                    "signal": sig,
                    "we_lag1": round(we, 3),
                    "we_haircut_pct": we_haircut_pct(we),
                })

        # ---- FLIES: M_i - 2*M_(i+1) + M_(i+2) for i=1..10 ----
        for i in range(1, NUM_TENORS - 1):
            j, k = i + 1, i + 2
            mi, mj, mk = f"m{i}", f"m{j}", f"m{k}"
            if not all(c in curve.columns for c in (mi, mj, mk)):
                continue
            ser = (curve[mi] - 2*curve[mj] + curve[mk]).dropna()
            if len(ser) < 250:
                continue
            df_feat = make_features(ser, dxy, curve, kind="diff")
            res = run_one_target(df_feat)
            if "winner" not in res:
                continue
            key = f"{prod}|FLY|m{i}-2m{j}+m{k}"
            model_results[key] = res
            if res["winner"] is not None:
                pred = res["winner_pred_now"]
                vol_threshold = float(np.median(np.abs(np.diff(ser.values)))) * 0.5
                sig = ("LONG" if pred > vol_threshold else
                       "SHORT" if pred < -vol_threshold else "FLAT")
                we = working_effect_lag1(ser)
                signals.append({
                    "kind": "FLY", "product": prod,
                    "label": f"{prod} M{i}-2M{j}+M{k}",
                    "current_level": round(float(ser.iloc[-1]), 4),
                    "winner_model": res["winner"],
                    "test_r2": res["winner_test_r2"],
                    "predicted_5d": pred,
                    "signal": sig,
                    "we_lag1": round(we, 3),
                    "we_haircut_pct": we_haircut_pct(we),
                })

    # ---- Save results ----
    (OUT / "real_data_models.json").write_text(json.dumps(model_results, indent=2))
    (OUT / "real_data_signals.json").write_text(json.dumps(signals, indent=2))

    # ---- Summary report ----
    print(f"Trained {len(model_results)} targets total")
    print()
    print("=" * 100)
    print("MODEL WINNER COUNTS (selected on validation R^2)")
    print("=" * 100)
    from collections import Counter
    wcount = Counter(r["winner"] for r in model_results.values() if r["winner"])
    for name, n in wcount.most_common():
        pct = n / len(model_results) * 100
        print(f"  {name:<12}  {n:>4} wins  ({pct:5.1f}%)")

    print()
    print("=" * 100)
    print("TOP 15 TARGETS by test R^2 (out-of-sample)")
    print("=" * 100)
    ranked = sorted(model_results.items(),
                     key=lambda kv: -(kv[1].get("winner_test_r2") or -1))[:15]
    for key, r in ranked:
        print(f"  {key:<28}  winner={r['winner']:<10}  test_R²={r['winner_test_r2']:+.4f}  n_test={r['n_test']}")

    print()
    print("=" * 100)
    print("ACTIONABLE SIGNALS (test_r2 > 0.05, signal != FLAT)")
    print("=" * 100)
    actionable = [s for s in signals
                   if s["test_r2"] is not None and s["test_r2"] > 0.05
                   and s["signal"] != "FLAT"]
    actionable.sort(key=lambda s: -s["test_r2"])
    print(f"  {len(actionable)} actionable signals")
    print(f"  {'Label':<22}{'Kind':<10}{'Current':>10}{'Pred 5d':>10}{'Signal':>8}{'Model':>10}{'Test R²':>10}")
    for s in actionable[:30]:
        print(f"  {s['label']:<22}{s['kind']:<10}{s['current_level']:>+10.3f}"
              f"{s['predicted_5d']:>+10.3f}{s['signal']:>8}{s['winner_model']:>10}"
              f"{s['test_r2']:>+10.4f}")

if __name__ == "__main__":
    main()
