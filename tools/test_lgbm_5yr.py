"""LGBM vs linear families on 5 years of real daily data — ALL 5 products.

Extends the earlier single-product (WTI) test to ALL 5 dashboard products
to check whether the pattern holds: linear-family parity, LGBM heavy-tree
overfit on small data, conservative LGBM as the least-bad option.

Features (8, derivable from product Close + DXY):
    1. ret_1d, 2. ret_5d, 3. ret_20d
    4. dxy_5d_ret, 5. dxy_20d_ret
    6. bb_pos (Bollinger pos, 0-100)
    7. vol_z (z-score of 20d realized vol vs 60d window)
    8. rsi_14n (RSI(14) - 50, normalized to -1..+1)

Target: forward 5-day product return.

Split: 80/20 chronological, no shuffle.

Per product, prints train_r2 / test_r2 / test_mae for every family AND
LGBM feature importances. Then a cross-product summary table.
"""
from __future__ import annotations
import math
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                   ElasticNet, HuberRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

PERIOD = "5y"
HORIZON = 5

PRODUCTS = {
    "WTI":    "CL=F",
    "Brent":  "BZ=F",
    "RBOB":   "RB=F",
    "HO":     "HO=F",
    "NatGas": "NG=F",
}

FEAT_NAMES = ["ret_1d", "ret_5d", "ret_20d",
              "dxy_5d_ret", "dxy_20d_ret",
              "bb_pos", "vol_z", "rsi_14n"]


def _bb_pos(close: pd.Series, window: int = 20) -> pd.Series:
    m = close.rolling(window).mean()
    s = close.rolling(window).std()
    lo = m - 2 * s
    return ((close - lo) / (4 * s) * 100).clip(0, 100)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.where(delta > 0, 0.0).rolling(period).mean()
    dn = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _vol20_z60(close: pd.Series) -> pd.Series:
    logret = np.log(close / close.shift(1))
    vol20 = logret.rolling(20).std() * math.sqrt(252) * 100
    mu = vol20.rolling(60).mean()
    sd = vol20.rolling(60).std()
    return (vol20 - mu) / sd


def build_dataset_one(product_close: pd.Series, dxy_close: pd.Series):
    df = pd.DataFrame(index=product_close.index)
    df["ret_1d"]      = product_close.pct_change(1) * 100
    df["ret_5d"]      = product_close.pct_change(5) * 100
    df["ret_20d"]     = product_close.pct_change(20) * 100
    df["dxy_5d_ret"]  = dxy_close.pct_change(5) * 100
    df["dxy_20d_ret"] = dxy_close.pct_change(20) * 100
    df["bb_pos"]      = _bb_pos(product_close, 20)
    df["vol_z"]       = _vol20_z60(product_close)
    df["rsi_14n"]     = (_rsi(product_close, 14) - 50) / 50
    y = (product_close.shift(-HORIZON) / product_close - 1) * 100
    df = df.assign(target=y).dropna()
    return df[FEAT_NAMES], df["target"]


def fit_eval(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    return {
        "name":     name,
        "train_r2": r2_score(y_tr, pred_tr),
        "test_r2":  r2_score(y_te, pred_te),
        "test_mae": mean_absolute_error(y_te, pred_te),
    }


def build_families():
    return {
        "Linear":     LinearRegression(),
        "Ridge":      Ridge(alpha=1.0),
        "Lasso":      Lasso(alpha=0.05, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000),
        "Huber":      HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=400),
        "LGBM_dash":  LGBMRegressor(
            n_estimators=30, max_depth=2, num_leaves=4,
            learning_rate=0.03, min_child_samples=8,
            min_data_in_bin=2, min_split_gain=0.0,
            reg_alpha=1.0, reg_lambda=1.0,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
            random_state=42, verbosity=-1, force_col_wise=True,
        ),
        "LGBM_5yr":   LGBMRegressor(
            n_estimators=200, max_depth=4, num_leaves=15,
            learning_rate=0.03, min_child_samples=20,
            min_data_in_bin=5, min_split_gain=0.0,
            reg_alpha=0.1, reg_lambda=0.1,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
            random_state=42, verbosity=-1, force_col_wise=True,
        ),
    }


def run_product(name, prod_close, dxy_close):
    X_df, y_ser = build_dataset_one(prod_close, dxy_close)
    if len(X_df) < 200:
        print(f"  {name}: SKIPPED — only {len(X_df)} bars")
        return None

    X = X_df.values.astype(float)
    y = y_ser.values.astype(float)
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    fams = build_families()
    rows = [fit_eval(fn, m, X_tr_s, X_te_s, y_tr, y_te) for fn, m in fams.items()]

    print(f"\n=== {name}  ({len(X_tr)} train / {len(X_te)} test)  target_std(test)={y_te.std():.2f}%  ===")
    print(f"  {'Family':<14} {'train_r2':>9}  {'test_r2':>9}  {'test_mae':>9}")
    print("  " + "-" * 48)
    for r in sorted(rows, key=lambda r: -r["test_r2"]):
        marker = " <- best" if r is max(rows, key=lambda r: r["test_r2"]) else ""
        print(f"  {r['name']:<14} {r['train_r2']:>9.4f}  {r['test_r2']:>9.4f}  {r['test_mae']:>9.4f}{marker}")

    lgbm = fams["LGBM_5yr"]
    imp = lgbm.feature_importances_.astype(float)
    s = imp.sum()
    if s > 0:
        imp = imp / s
    top = sorted(zip(FEAT_NAMES, imp), key=lambda x: -x[1])[:4]
    print(f"  LGBM_5yr top features: " + ", ".join(f"{fn}({w*100:.0f}%)" for fn, w in top))

    return {
        "product": name,
        "n_train": len(X_tr),
        "n_test":  len(X_te),
        "rows":    {r["name"]: r for r in rows},
        "winner":  max(rows, key=lambda r: r["test_r2"])["name"],
    }


def main():
    tickers = list(PRODUCTS.values()) + ["DX-Y.NYB"]
    print(f"Pulling {PERIOD} of daily Close for {len(tickers)} symbols...")
    raw = yf.download(tickers, period=PERIOD,
                      progress=False, auto_adjust=True, threads=True)
    close = raw["Close"]
    dxy_close = close["DX-Y.NYB"]
    print(f"  Bars per product (pre-feature-engineering):")
    for p, tkr in PRODUCTS.items():
        print(f"    {p:<8} ({tkr:<7}): {close[tkr].dropna().shape[0]:,}")

    results = []
    for p, tkr in PRODUCTS.items():
        prod_close = close[tkr].dropna()
        dxy_aligned = dxy_close.reindex(prod_close.index, method="ffill")
        r = run_product(p, prod_close, dxy_aligned)
        if r is not None:
            results.append(r)

    # Cross-product summary
    print()
    print("=" * 70)
    print("CROSS-PRODUCT SUMMARY — test_r2 by family")
    print("=" * 70)
    family_order = ["Linear", "Ridge", "Lasso", "ElasticNet", "Huber",
                    "LGBM_dash", "LGBM_5yr"]
    header = f"  {'Family':<14}  " + "  ".join(f"{r['product']:>8}" for r in results) + "    avg"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for fam in family_order:
        scores = [r["rows"][fam]["test_r2"] for r in results if fam in r["rows"]]
        avg = sum(scores) / len(scores) if scores else float("nan")
        line = f"  {fam:<14}  " + "  ".join(f"{s:>+8.4f}" for s in scores) + f"  {avg:>+7.4f}"
        print(line)

    print()
    print("Winners (best test_r2 per product):")
    for r in results:
        win = r["winner"]
        score = r["rows"][win]["test_r2"]
        print(f"  {r['product']:<8} -> {win:<11} (test_r2={score:+.4f})")


if __name__ == "__main__":
    main()
