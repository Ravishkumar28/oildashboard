"""Value at Risk (VaR) and Expected Shortfall (CVaR) for the multi-product
LGBM strategy stack.

For each tradeable strategy (5 outright products + 3 spreads), pulls 5y
daily data, runs the same LGBM_dash signal generator used in the live
dashboard panel, and computes:

  * Historical VaR at 95% and 99% — empirical percentile of per-trade PnL
  * Parametric VaR (Gaussian) at 95% and 99% — mean - Z*std
  * Expected Shortfall (CVaR) at 95% — avg loss when 95% VaR is breached
  * Annualized VaR — scaled to a 252-day year (so it's comparable across
    products with different trade frequencies)
  * Suggested position size — given a $10,000 max daily loss budget

Outputs everything in one ranked table sorted by risk-adjusted edge
(Sharpe / 95% VaR magnitude — lower VaR per unit Sharpe is better).
"""
from __future__ import annotations
import math
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                   ElasticNet, HuberRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

PERIOD            = "5y"
HORIZON           = 5
SIG_THRESHOLD     = 0.5
COST_BPS          = 5.0
DAILY_LOSS_BUDGET = 10000.0   # $ — for position-sizing math

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


def _bb_pos(c, w=20):
    m = c.rolling(w).mean(); s = c.rolling(w).std()
    return ((c - (m - 2*s)) / (4*s) * 100).clip(0, 100)
def _rsi(c, p=14):
    d = c.diff()
    up = d.where(d > 0, 0.0).rolling(p).mean()
    dn = (-d.where(d < 0, 0.0)).rolling(p).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)
def _vol_z(c):
    lr = np.log(c / c.shift(1))
    v20 = lr.rolling(20).std() * math.sqrt(252) * 100
    return (v20 - v20.rolling(60).mean()) / v20.rolling(60).std()


def features(target, dxy, kind="ret"):
    df = pd.DataFrame(index=target.index)
    if kind == "ret":
        df["ret_1d"]  = target.pct_change(1)  * 100
        df["ret_5d"]  = target.pct_change(5)  * 100
        df["ret_20d"] = target.pct_change(20) * 100
        y = (target.shift(-HORIZON) / target - 1) * 100
    else:
        df["ret_1d"]  = target.diff(1)
        df["ret_5d"]  = target.diff(5)
        df["ret_20d"] = target.diff(20)
        y = target.shift(-HORIZON) - target
    df["dxy_5d_ret"]  = dxy.pct_change(5) * 100
    df["dxy_20d_ret"] = dxy.pct_change(20) * 100
    df["bb_pos"]      = _bb_pos(target, 20)
    df["vol_z"]       = _vol_z(target)
    df["rsi_14n"]     = (_rsi(target, 14) - 50) / 50
    df = df.assign(target=y).dropna()
    return df[FEAT_NAMES], df["target"]


def get_pnl_series(label, target, dxy, kind="ret"):
    """Return the full backtest PnL series for one strategy."""
    Xd, y = features(target, dxy, kind=kind)
    if len(Xd) < 200:
        return None
    X = Xd.values.astype(float); yv = y.values.astype(float)
    split = int(len(X) * 0.8)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[:split])
    X_te = sc.transform(X[split:])
    model = LGBMRegressor(
        n_estimators=30, max_depth=2, num_leaves=4,
        learning_rate=0.03, min_child_samples=8,
        min_data_in_bin=2, min_split_gain=0.0,
        reg_alpha=1.0, reg_lambda=1.0,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
        random_state=42, verbosity=-1, force_col_wise=True,
    )
    model.fit(X_tr, yv[:split])
    pred = model.predict(X_te)
    y_te = yv[split:]
    threshold = (SIG_THRESHOLD if kind == "ret"
                 else float(np.median(np.abs(np.diff(target.dropna().values)))) * 0.5)
    sig = np.where(pred > threshold, 1, np.where(pred < -threshold, -1, 0))
    pnl = []
    for i in range(0, len(sig), HORIZON):
        if sig[i] == 0:
            continue
        pnl.append(float(sig[i] * y_te[i] - COST_BPS/100.0))
    return np.array(pnl) if pnl else None


def compute_var(pnl, confidence=0.95):
    """Historical + Parametric + CVaR at the given confidence."""
    if len(pnl) < 5:
        return None
    alpha = 1 - confidence
    losses = -pnl   # convert to losses (positive = bad)
    # Historical VaR
    hist_var = float(np.percentile(losses, confidence * 100))
    # Parametric (Gaussian) VaR
    mu, sd = pnl.mean(), pnl.std(ddof=1)
    z = norm.ppf(confidence)
    param_var = float(-mu + z * sd)
    # Expected Shortfall (CVaR) = avg of losses worse than VaR
    tail_losses = losses[losses >= hist_var]
    cvar = float(tail_losses.mean()) if len(tail_losses) > 0 else hist_var
    return {
        "hist_var":  hist_var,
        "param_var": param_var,
        "cvar":      cvar,
        "mean":      float(mu),
        "std":       float(sd),
    }


def main():
    print(f"Pulling {PERIOD} of daily Close...")
    tickers = list(PRODUCTS.values()) + ["DX-Y.NYB"]
    raw = yf.download(tickers, period=PERIOD,
                      progress=False, auto_adjust=True, threads=True)
    close = raw["Close"]
    dxy = close["DX-Y.NYB"]

    strategies = []

    # Outright products
    for p, tkr in PRODUCTS.items():
        ts = close[tkr].dropna()
        dxy_a = dxy.reindex(ts.index, method="ffill")
        pnl = get_pnl_series(p, ts, dxy_a, kind="ret")
        if pnl is not None:
            strategies.append((p, "% return", pnl, ts.iloc[-1]))

    # Spreads
    wti = close["CL=F"].dropna()
    brent = close["BZ=F"].reindex(wti.index, method="ffill")
    rb = close["RB=F"].reindex(wti.index, method="ffill") * 42
    ho = close["HO=F"].reindex(wti.index, method="ffill") * 42
    spreads = {
        "WTI-Brent":  (wti - brent).dropna(),
        "321Crack":   (2*rb + ho - 3*wti).dropna(),
        "RBOB-HO":    (rb - ho).dropna(),
    }
    for name, ser in spreads.items():
        dxy_a = dxy.reindex(ser.index, method="ffill")
        pnl = get_pnl_series(name, ser, dxy_a, kind="diff")
        if pnl is not None:
            strategies.append((name, "$/bbl", pnl, ser.iloc[-1]))

    # ----- compute VaR for each strategy ------------------------------------
    rows = []
    for label, unit, pnl, level in strategies:
        v95 = compute_var(pnl, 0.95)
        v99 = compute_var(pnl, 0.99)
        if v95 is None:
            continue
        sharpe = (pnl.mean() / pnl.std(ddof=1) * math.sqrt(252.0 / HORIZON)
                  if pnl.std() > 0 else 0.0)
        # Position sizing: how many CONTRACTS can we trade given the $ budget?
        # Each strategy's PnL is in % or $/bbl (per unit). One contract = 1000 bbl.
        # Daily VaR (5d VaR scaled by sqrt(1/5))
        daily_var_95 = v95["hist_var"] * math.sqrt(1.0 / HORIZON)
        if unit == "% return":
            # $ loss per contract = price * 1000 * (daily_var_95/100)
            usd_loss_per_contract = float(level * 1000 * daily_var_95 / 100)
        else:
            # spread is in $/bbl, so $ loss per contract = daily_var_95 * 1000
            usd_loss_per_contract = float(daily_var_95 * 1000)
        if usd_loss_per_contract <= 0:
            max_contracts = 0
        else:
            max_contracts = int(DAILY_LOSS_BUDGET / usd_loss_per_contract)
        rows.append({
            "label":    label,
            "unit":     unit,
            "n_trades": len(pnl),
            "mean":     v95["mean"],
            "std":      v95["std"],
            "sharpe":   sharpe,
            "hist_95":  v95["hist_var"],
            "param_95": v95["param_var"],
            "cvar_95":  v95["cvar"],
            "hist_99":  v99["hist_var"],
            "param_99": v99["param_var"],
            "cvar_99":  v99["cvar"],
            "max_contracts": max_contracts,
            "usd_loss_per_contract": usd_loss_per_contract,
        })

    # ----- print results -----------------------------------------------------
    print()
    print("=" * 98)
    print(f"VaR ANALYSIS — 5d holding period · {DAILY_LOSS_BUDGET:.0f} USD daily loss budget · 5 bps cost")
    print("=" * 98)
    print()
    print("  Per-trade VaR (in strategy native units: % for outright, $/bbl for spreads)")
    print("  " + "-" * 94)
    print(f"  {'Strategy':<12}{'n':>4}{'Sharpe':>9}{'StdDev':>9}{'95%VaR':>9}"
          f"{'99%VaR':>9}{'95%CVaR':>10}{'99%CVaR':>10}{'Param95':>10}")
    print("  " + "-" * 94)
    for r in sorted(rows, key=lambda x: -x["sharpe"]):
        print(f"  {r['label']:<12}{r['n_trades']:>4}{r['sharpe']:>+9.2f}"
              f"{r['std']:>9.3f}{r['hist_95']:>9.3f}{r['hist_99']:>9.3f}"
              f"{r['cvar_95']:>10.3f}{r['cvar_99']:>10.3f}{r['param_95']:>10.3f}")

    print()
    print("  Position sizing — max contracts to keep 1-day 95% VaR <= " + f"${DAILY_LOSS_BUDGET:.0f}")
    print("  " + "-" * 94)
    print(f"  {'Strategy':<12}{'Current Lvl':>14}{'$Loss/Contract':>20}{'Max Contracts':>16}")
    print("  " + "-" * 94)
    for r in sorted(rows, key=lambda x: -x["max_contracts"]):
        lvl = 0.0
        for label, unit, pnl, level in strategies:
            if label == r["label"]:
                lvl = level
                break
        print(f"  {r['label']:<12}{lvl:>14.2f}{r['usd_loss_per_contract']:>20.2f}"
              f"{r['max_contracts']:>16}")

    print()
    print("=" * 98)
    print("INTERPRETATION")
    print("=" * 98)
    # Highlight high-conviction tradable strategies (Sharpe > 0.5)
    tradable = [r for r in rows if r["sharpe"] > 0.5]
    if tradable:
        print(f"  HIGH-CONFIDENCE STRATEGIES (Sharpe > 0.5):")
        for r in sorted(tradable, key=lambda x: -x["sharpe"]):
            risk_adj = abs(r["sharpe"] / r["hist_95"]) if r["hist_95"] != 0 else 0
            print(f"    * {r['label']:<12} Sharpe={r['sharpe']:+.2f}  "
                  f"95%VaR={r['hist_95']:.3f}{r['unit'].split()[0]}  "
                  f"CVaR={r['cvar_95']:.3f}  "
                  f"Max size={r['max_contracts']} contracts")
        print()
        print("  HOW TO READ:")
        print(f"    - 95%VaR=X means: 5% of trades lose MORE than X in {HORIZON} days")
        print(f"    - CVaR=Y means: when VaR is breached, the AVERAGE loss is Y")
        print(f"    - Max size = position cap to keep your worst expected day under ${DAILY_LOSS_BUDGET:.0f}")
    else:
        print("  No strategies meet the Sharpe > 0.5 threshold.")


if __name__ == "__main__":
    main()
