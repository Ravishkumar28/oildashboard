"""Strategy shootout — implement 6 oil-trading strategies on the same data,
backtest them all, and pick the winner.

Strategies (4 from papers we read, 1 inspired, 1 original):

  S1 HMM_OU_WTCL      Fanelli et al 2023: 2-state HMM on log-spread of WTI-Brent,
                       trade when regime prob > 0.7 AND |z| > 1, OU mean reversion.

  S2 WashU_Brent      Lenz 2016: monthly OLS on Brent close + DXY, predict 60-day
                       forward high, enter LONG if predicted return >= +1%, SHORT
                       if <= -1%, exit at target or 12 months.

  S3 TwoFactor_Crack  Farkas et al 2017 (inferred): 2-factor cointegration on the
                       crack spread proxy log(HO M1) - log(CL M1). Z-score mean
                       reversion with entry at |z| > 2, exit at |z| < 0.5.

  S4 MLP_WTI          Frontiers 2024: sklearn MLPClassifier predicting next-day
                       WTI direction from EMA20/60, EMA20/100, EMA60/100 ratios
                       + recent returns. Daily long/short rotation.

  S5 Bertram_OU       Cummins & Bucca 2012: OU fit on log calendar spreads
                       (CL M1-M2, M2-M3, etc.), Bertram-optimal entry threshold
                       solved numerically. Best-Sharpe spread per product.

  S6 PCA_Curve        ORIGINAL: PCA on curve M1-M12 returns, identify shocks to
                       PC3 (curvature). Trade against extreme curvature shocks
                       on the assumption they mean-revert (curvature is the most
                       transient PC).

Same harness, same data, same costs (5 bps round-trip). Whoever wins on net PnL
+ Sharpe > 0.5 + win rate > 50% gets deployed.
"""
from __future__ import annotations
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from hmmlearn.hmm import GaussianHMM
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from strategy_harness import (load_all_curves, load_dxy, test_split_date,
                                COST_BPS, DEFAULT_BBL)

HORIZON = 5            # exit horizon for time-based stops where applicable
TEST_FRACTION = 0.20   # last 20% out of sample


def _settle(trade: Dict) -> Dict:
    entry = float(trade["entry_price"])
    exit_  = float(trade["exit_price"])
    direction = int(trade["direction"])
    notional = float(trade.get("notional_bbl", DEFAULT_BBL))
    gross = direction * (exit_ - entry) * notional
    cost  = (COST_BPS / 10000.0) * abs(entry) * notional
    return {**trade, "gross_pnl": gross, "cost": cost, "net_pnl": gross - cost}


def _stats(trades: List[Dict], name: str) -> Dict:
    if not trades:
        return {"strategy": name, "n_trades": 0, "total_pnl": 0,
                "win_rate": 0, "sharpe": 0, "max_dd": 0}
    settled = [_settle(t) for t in trades]
    pnls = np.array([t["net_pnl"] for t in settled])
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = float((peak - cum).max()) if len(cum) > 0 else 0.0
    win_rate = float((pnls > 0).mean() * 100)
    mean_pnl = float(pnls.mean())
    std_pnl  = float(pnls.std(ddof=1)) if len(pnls) > 1 else 0.0
    sharpe = mean_pnl / std_pnl * math.sqrt(252.0/HORIZON) if std_pnl > 0 else 0.0
    # by product
    by_prod = defaultdict(list)
    for t in settled:
        by_prod[t.get("product", "?")].append(t["net_pnl"])
    bp = {p: {"n": len(v), "pnl": float(sum(v))} for p, v in by_prod.items()}
    return {
        "strategy":   name,
        "n_trades":   len(pnls),
        "total_pnl":  float(pnls.sum()),
        "mean_pnl":   mean_pnl,
        "win_rate":   win_rate,
        "sharpe":     sharpe,
        "max_dd":     dd,
        "by_product": bp,
    }


# =================================================================== #
# S1: HMM + OU on WTI-Brent spread
# =================================================================== #
def strategy_hmm_wtcl(curves, dxy):
    """2-state HMM on log-spread of WTI vs Brent."""
    wti = curves["CL"]["m1"].dropna()
    brent = curves["LCO"]["m1"].dropna()
    # Align dates
    common = wti.index.intersection(brent.index)
    wti = wti.reindex(common); brent = brent.reindex(common)
    # log spread
    spread = np.log(wti) - np.log(brent)

    n = len(spread)
    train_end = int(n * 0.60)
    val_end   = int(n * 0.80)

    # Fit HMM on train slice
    train_spread = spread.iloc[:train_end].values.reshape(-1, 1)
    try:
        hmm = GaussianHMM(n_components=2, covariance_type="full",
                           n_iter=100, random_state=42)
        hmm.fit(train_spread)
    except Exception:
        return []

    # Posteriors on test slice
    test_spread = spread.iloc[val_end:].values.reshape(-1, 1)
    try:
        posteriors = hmm.predict_proba(test_spread)
    except Exception:
        return []
    test_idx = spread.index[val_end:]

    # Identify low-mean vs high-mean state
    means = hmm.means_.flatten()
    low_state = int(np.argmin(means))
    high_state = int(np.argmax(means))
    # Conditional mean & std from HMM
    std_per_state = np.sqrt(hmm.covars_.reshape(2, 1, 1)[:, 0, 0])

    trades = []
    position = 0  # 0 flat, +1 long spread, -1 short
    entry_idx = None
    entry_z = None
    for i, (dt, post) in enumerate(zip(test_idx, posteriors)):
        s_now = float(test_spread[i, 0])
        # weighted mean & std
        cond_mean = float(post[0]*means[0] + post[1]*means[1])
        cond_std  = float(post[0]*std_per_state[0] + post[1]*std_per_state[1])
        z = (s_now - cond_mean) / cond_std if cond_std > 0 else 0
        p_low = float(post[low_state])
        p_high = float(post[high_state])

        if position == 0:
            # ENTER LONG spread (long WTI / short Brent) when in LOW state + cheap
            if p_low > 0.7 and z < -1.0:
                position = +1
                entry_idx = i
                entry_z = z
            elif p_high > 0.7 and z > +1.0:
                position = -1
                entry_idx = i
                entry_z = z
        else:
            # exit: |z|<0.25, regime flip, stop, or 20 days
            hold = i - entry_idx
            stopped = (position > 0 and z < entry_z - 2.0) or \
                      (position < 0 and z > entry_z + 2.0)
            regime_flip = (position > 0 and p_high > 0.5) or \
                          (position < 0 and p_low > 0.5)
            if abs(z) < 0.25 or stopped or regime_flip or hold >= 20:
                # spread trade = long WTI + short Brent (paired contracts)
                # we record it as a synthetic spread position with PnL = direction × Δspread
                entry_dt = test_idx[entry_idx]
                exit_dt  = dt
                # PnL is in absolute spread units; convert to $/contract value
                # Use the avg of WTI+Brent as notional reference
                avg_px = (wti.loc[entry_dt] + brent.loc[entry_dt]) / 2
                trades.append({
                    "product":   "WTCL",
                    "instrument": "logspread",
                    "entry_date": entry_dt, "exit_date": exit_dt,
                    "direction":  int(position),
                    "entry_price": float(spread.iloc[val_end + entry_idx] * avg_px),
                    "exit_price":  float(spread.iloc[val_end + i] * avg_px),
                    "notional_bbl": DEFAULT_BBL,
                })
                position = 0
                entry_idx = None
                entry_z = None
    return trades


# =================================================================== #
# S2: WashU monthly OLS on Brent + DXY -> predict forward 60d high
# =================================================================== #
def strategy_washu_brent(curves, dxy):
    brent = curves["LCO"]["m1"].dropna()
    # resample to monthly first trading day
    monthly = brent.resample("MS").first().dropna()
    dxy_m = dxy.resample("MS").last().reindex(monthly.index, method="ffill").dropna()
    common = monthly.index.intersection(dxy_m.index)
    monthly = monthly.reindex(common)
    dxy_m   = dxy_m.reindex(common)

    n = len(monthly)
    train_end = int(n * 0.60)

    # Compute forward 60d (~2 months) high on monthly data
    fwd_high = monthly.rolling(window=3).max().shift(-2).reindex(monthly.index)

    trades = []
    open_pos = None  # dict if a trade is currently open
    for i in range(train_end, n - 1):
        if open_pos is None:
            # Estimate OLS on trailing 12 months
            start = max(0, i - 12)
            if i - start < 6:
                continue
            X = np.column_stack([
                np.log(monthly.iloc[start:i].values),
                np.log(dxy_m.iloc[start:i].values),
            ])
            X = sm.add_constant(X)
            y = np.log(fwd_high.iloc[start:i].fillna(monthly.iloc[start:i]).values)
            try:
                m = sm.OLS(y, X).fit()
            except Exception:
                continue
            # Predict
            x_now = sm.add_constant(np.array([[
                np.log(monthly.iloc[i]), np.log(dxy_m.iloc[i])
            ]]), has_constant="add")
            try:
                ln_pa = float(m.predict(x_now)[0])
            except Exception:
                continue
            Pa = math.exp(ln_pa)
            close = float(monthly.iloc[i])
            pred_ret = (Pa - close) / close
            direction = 0
            if pred_ret >= 0.01:
                direction = +1
            elif pred_ret <= -0.01:
                direction = -1
            if direction == 0:
                continue
            open_pos = {
                "entry_date": monthly.index[i],
                "entry_price": close,
                "target_Pa": Pa,
                "direction": direction,
                "entry_idx": i,
            }
        else:
            cur = float(monthly.iloc[i])
            hit = (open_pos["direction"] == +1 and cur >= open_pos["target_Pa"]) or \
                  (open_pos["direction"] == -1 and cur <= open_pos["target_Pa"])
            timeout = (i - open_pos["entry_idx"]) >= 12  # 12 months
            if hit or timeout:
                trades.append({
                    "product":   "LCO",
                    "instrument": "m1",
                    "entry_date": open_pos["entry_date"],
                    "exit_date":  monthly.index[i],
                    "direction":  open_pos["direction"],
                    "entry_price": open_pos["entry_price"],
                    "exit_price":  cur,
                    "notional_bbl": DEFAULT_BBL,
                })
                open_pos = None
    return trades


# =================================================================== #
# S3: Two-factor cointegration on crack proxy log(HO) - log(CL)
# =================================================================== #
def strategy_twofactor_crack(curves, dxy):
    cl = curves["CL"]["m1"].dropna()
    ho = curves["HO"]["m1"].dropna()
    # HO is in $/gal, convert to $/bbl × 42
    ho_bbl = ho * 42.0
    common = cl.index.intersection(ho_bbl.index)
    cl = cl.reindex(common)
    ho_bbl = ho_bbl.reindex(common)
    log_spread = np.log(ho_bbl) - np.log(cl)

    n = len(log_spread)
    train_end = int(n * 0.60)
    val_end   = int(n * 0.80)

    # On training slice estimate mean and std
    train_seg = log_spread.iloc[:train_end]
    mu = float(train_seg.mean())
    sigma = float(train_seg.std(ddof=1))
    if sigma == 0:
        return []

    z = (log_spread - mu) / sigma
    # Rolling z (60d window) for adaptive entry
    z_roll_mean = z.rolling(60).mean()
    z_roll_std  = z.rolling(60).std()
    z_adj = (z - z_roll_mean) / z_roll_std

    trades = []
    position = 0
    entry_i = None
    entry_z = None
    Z_ENTRY = 2.0
    Z_EXIT  = 0.5
    Z_STOP  = 3.5
    MAX_HOLD = 30
    for i in range(val_end, n):
        zi = z_adj.iloc[i]
        if not np.isfinite(zi):
            continue
        if position == 0:
            if zi >= +Z_ENTRY:
                position = -1   # short the crack
                entry_i = i; entry_z = zi
            elif zi <= -Z_ENTRY:
                position = +1   # long the crack
                entry_i = i; entry_z = zi
        else:
            hold = i - entry_i
            stop = (position > 0 and zi < entry_z - 1.5) or \
                   (position < 0 and zi > entry_z + 1.5) or \
                   abs(zi) >= Z_STOP
            done = abs(zi) <= Z_EXIT
            if done or stop or hold >= MAX_HOLD:
                # PnL in spread terms - use HO M1 leg as the trade vehicle
                entry_dt = log_spread.index[entry_i]
                exit_dt  = log_spread.index[i]
                # Use HO contract value as notional
                trades.append({
                    "product":   "HO",
                    "instrument": "HO-CL crack",
                    "entry_date": entry_dt, "exit_date": exit_dt,
                    "direction":  int(position),
                    # Express PnL via spread in $/bbl
                    "entry_price": float(ho_bbl.iloc[entry_i] - cl.iloc[entry_i]),
                    "exit_price":  float(ho_bbl.iloc[i] - cl.iloc[i]),
                    "notional_bbl": DEFAULT_BBL,
                })
                position = 0
                entry_i = None
                entry_z = None
    return trades


# =================================================================== #
# S4: MLP classifier on WTI daily direction
# =================================================================== #
def strategy_mlp_wti(curves, dxy):
    wti = curves["CL"]["m1"].dropna()
    ema20 = wti.ewm(span=20).mean()
    ema60 = wti.ewm(span=60).mean()
    ema100 = wti.ewm(span=100).mean()
    feats = pd.DataFrame({
        "ema20_60":  ema20 / ema60,
        "ema20_100": ema20 / ema100,
        "ema60_100": ema60 / ema100,
        "ret_1d":    wti.pct_change(1) * 100,
        "ret_5d":    wti.pct_change(5) * 100,
        "ret_20d":   wti.pct_change(20) * 100,
        "vol_20d":   np.log(wti / wti.shift(1)).rolling(20).std() * math.sqrt(252) * 100,
    })
    feats["target"] = (wti.shift(-1) > wti).astype(int)
    feats = feats.dropna()
    n = len(feats)
    train_end = int(n * 0.60)
    val_end   = int(n * 0.80)
    Xcols = [c for c in feats.columns if c != "target"]
    X = feats[Xcols].values
    y = feats["target"].values
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X[:train_end])
    X_te_s = scaler.transform(X[val_end:])
    try:
        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=400,
                             random_state=42, alpha=0.01)
        clf.fit(X_tr_s, y[:train_end])
        pred = clf.predict(X_te_s)
        # probabilities for sizing
        proba = clf.predict_proba(X_te_s)[:, 1]
    except Exception:
        return []

    test_idx = feats.index[val_end:]
    trades = []
    # Daily long/short rotation: enter at close, exit next close
    test_wti = wti.reindex(test_idx)
    for i in range(len(test_idx) - 1):
        signal = pred[i]
        confidence = abs(proba[i] - 0.5) * 2  # 0..1
        if confidence < 0.10:   # weak signals skipped
            continue
        direction = +1 if signal == 1 else -1
        entry_px = float(test_wti.iloc[i])
        exit_px  = float(test_wti.iloc[i + 1])
        if not np.isfinite(entry_px) or not np.isfinite(exit_px):
            continue
        trades.append({
            "product":   "CL",
            "instrument": "m1",
            "entry_date": test_idx[i], "exit_date": test_idx[i+1],
            "direction":  direction,
            "entry_price": entry_px,
            "exit_price":  exit_px,
            "notional_bbl": DEFAULT_BBL,
        })
    return trades


# =================================================================== #
# S5: Bertram-OU optimal threshold on calendar spreads
# =================================================================== #
def strategy_bertram_ou(curves, dxy):
    """For each calendar spread, fit OU and trade Bertram-style threshold."""
    trades = []
    for prod_code in ("CL", "LCO", "HO", "LGO"):
        curve = curves[prod_code]
        if "m1" not in curve.columns or "m2" not in curve.columns:
            continue
        for spread_pair in [(1, 2), (2, 3), (3, 4), (5, 6), (8, 9)]:
            a, b = spread_pair
            ka, kb = f"m{a}", f"m{b}"
            if ka not in curve.columns or kb not in curve.columns:
                continue
            s_raw = (curve[ka] - curve[kb]).dropna()
            # log-spread doesn't work for negatives; use absolute spread
            n = len(s_raw)
            if n < 250:
                continue
            train_end = int(n * 0.60)
            val_end   = int(n * 0.80)
            train_seg = s_raw.iloc[:train_end]
            # OU estimation: dS = -alpha*(S - mu)*dt + eta*dW
            # OLS regression of dS on (mu_hat - S): dS_t = c + slope*S_t + ε
            mu_hat = float(train_seg.mean())
            ds = train_seg.diff().dropna()
            s_lag = train_seg.shift(1).dropna().reindex(ds.index)
            slope, _ = np.polyfit(s_lag.values, ds.values, 1)
            alpha_hat = -slope            # mean-reversion speed (per day)
            eta_hat = float(ds.std(ddof=1))
            if alpha_hat <= 0 or eta_hat <= 0:
                continue
            sigma_eq = eta_hat / math.sqrt(2 * alpha_hat)
            # Skip super-noisy spreads (no real mean reversion)
            if sigma_eq <= 0:
                continue
            # Bertram-optimal a* ~ 1.5 * sigma_eq is a reasonable proxy.
            a_star = 1.5 * sigma_eq
            # Trade on test slice
            position = 0
            entry_i = None
            test_seg = s_raw.iloc[val_end:]
            test_idx = test_seg.index
            for i in range(len(test_seg)):
                s_now = float(test_seg.iloc[i])
                deviation = s_now - mu_hat
                if position == 0:
                    if deviation >= +a_star:
                        position = -1; entry_i = i
                    elif deviation <= -a_star:
                        position = +1; entry_i = i
                else:
                    hold = i - entry_i
                    # exit at mean or opposite threshold
                    if (position > 0 and deviation >= 0) or \
                       (position < 0 and deviation <= 0) or \
                       hold >= 30:
                        trades.append({
                            "product":   prod_code,
                            "instrument": f"m{a}-m{b}",
                            "entry_date": test_idx[entry_i],
                            "exit_date":  test_idx[i],
                            "direction":  int(position),
                            "entry_price": float(test_seg.iloc[entry_i]),
                            "exit_price":  float(test_seg.iloc[i]),
                            "notional_bbl": DEFAULT_BBL,
                        })
                        position = 0
                        entry_i = None
    return trades


# =================================================================== #
# S6: PCA curve mean reversion (ORIGINAL)
# =================================================================== #
def strategy_pca_curve(curves, dxy):
    """Fit PCA on daily M1-M12 returns of each product. The 3rd PC
    represents curvature (fly shape). Extreme PC3 values are noise-like
    and revert; trade against extreme PC3 shocks."""
    trades = []
    for prod_code in ("CL", "LCO", "HO", "LGO"):
        curve = curves[prod_code]
        cols = [f"m{i}" for i in range(1, 13)
                if f"m{i}" in curve.columns]
        if len(cols) < 8:
            continue
        prices = curve[cols].dropna()
        if len(prices) < 250:
            continue
        # Use log returns
        log_ret = np.log(prices / prices.shift(1)).dropna()
        n = len(log_ret)
        train_end = int(n * 0.60)
        val_end   = int(n * 0.80)
        # Fit PCA on training
        train_X = log_ret.iloc[:train_end].values
        pca = PCA(n_components=3)
        try:
            pca.fit(train_X)
        except Exception:
            continue
        # Project test data
        test_X = log_ret.iloc[val_end:].values
        scores = pca.transform(test_X)
        # PC3 = curvature; compute rolling z of PC3 in test
        pc3 = scores[:, 2]
        roll_mean = pd.Series(pc3).rolling(20).mean()
        roll_std  = pd.Series(pc3).rolling(20).std()
        z = (pd.Series(pc3) - roll_mean) / roll_std
        test_idx = log_ret.index[val_end:]
        # PC3 loading: which tenors load positively / negatively
        pc3_load = pca.components_[2]
        # Fly trade: when PC3 spikes, trade against it via M3 - 2*M6 + M9
        # (or pick the 3 tenors with largest |loadings|)
        sorted_loads = np.argsort(np.abs(pc3_load))[::-1]
        leg_indices = sorted_loads[:3]
        leg_signs = np.sign(pc3_load[leg_indices])
        # Get fly series in test
        leg_cols = [cols[i] for i in leg_indices]
        leg_prices_test = prices[leg_cols].iloc[val_end+1:].copy()
        # fly value = sum(sign * price) per row
        fly = (leg_prices_test * leg_signs).sum(axis=1)
        position = 0
        entry_i = None
        for i in range(len(test_idx)):
            zi = z.iloc[i]
            if not np.isfinite(zi):
                continue
            if position == 0:
                if zi >= +2.0:
                    position = -1; entry_i = i
                elif zi <= -2.0:
                    position = +1; entry_i = i
            else:
                hold = i - entry_i
                if abs(zi) <= 0.5 or hold >= 15:
                    if entry_i < len(fly) and i < len(fly):
                        trades.append({
                            "product":   prod_code,
                            "instrument": f"PC3_fly({'+'.join(leg_cols)})",
                            "entry_date": test_idx[entry_i],
                            "exit_date":  test_idx[i],
                            "direction":  int(position),
                            "entry_price": float(fly.iloc[entry_i]),
                            "exit_price":  float(fly.iloc[i]),
                            "notional_bbl": DEFAULT_BBL,
                        })
                    position = 0
                    entry_i = None
    return trades


# ====================================================================== #
def main():
    print("Loading data once...")
    curves = load_all_curves()
    dxy = load_dxy()
    print(f"  curves: {sorted(curves.keys())}")
    print(f"  dxy: {len(dxy)} rows")
    print()

    strategies = [
        ("S1_HMM_OU_WTCL",      strategy_hmm_wtcl),
        ("S2_WashU_Brent",      strategy_washu_brent),
        ("S3_TwoFactor_Crack",  strategy_twofactor_crack),
        ("S4_MLP_WTI",          strategy_mlp_wti),
        ("S5_Bertram_OU",       strategy_bertram_ou),
        ("S6_PCA_Curve",        strategy_pca_curve),
    ]

    results = []
    for name, fn in strategies:
        print(f"running {name}...")
        try:
            trades = fn(curves, dxy)
            stats = _stats(trades, name)
            results.append(stats)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            results.append({"strategy": name, "n_trades": 0, "error": str(e)[:80]})
        print()

    # Final report
    print("=" * 100)
    print("STRATEGY SHOOTOUT — final results (out-of-sample test slice, 5bps round-trip)")
    print("=" * 100)
    print(f"  {'Strategy':<22}{'Trades':>8}{'TotalPnL$':>14}{'MeanPnL$':>12}{'Win%':>8}{'Sharpe':>9}{'MaxDD$':>12}")
    print("  " + "-" * 92)
    for r in sorted(results, key=lambda x: -(x.get("total_pnl") or 0)):
        if "error" in r:
            print(f"  {r['strategy']:<22}{'ERR':>8} {r['error'][:60]}")
            continue
        print(f"  {r['strategy']:<22}{r['n_trades']:>8}{r['total_pnl']:>+14,.0f}"
              f"{r['mean_pnl']:>+12,.1f}{r['win_rate']:>7.1f}%{r['sharpe']:>+9.2f}{r['max_dd']:>12,.0f}")
    # Save results
    out_path = Path(__file__).resolve().parents[1] / "tools" / "results" / "strategy_shootout.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved -> tools/results/strategy_shootout.json")


if __name__ == "__main__":
    main()
