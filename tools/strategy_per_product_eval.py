"""Per-product strategy evaluation — for each of the 6 shootout strategies,
break down PnL/Sharpe/WinRate/R² by product.

R² is computed where the strategy makes a continuous prediction:
  - PCA Curve: R² of (predicted PC3 mean reversion) vs (actual PC3 change)
  - Bertram OU: R² of (predicted reversion of spread) vs (actual reversion)
  - TwoFactor Crack: R² of z-score prediction vs realized return
  - HMM_OU: R² of HMM-conditional-mean prediction vs realized spread
  - MLP_WTI: classification accuracy (binary, no R²)
  - WashU Brent: R² of log-target vs realized log-high

The table that comes out tells us for each PRODUCT:
  (a) Which strategy makes money?
  (b) Which strategy has the highest predictive R²?
  (c) Best strategy per product = max(test_r2 × pnl_positive_indicator)

This is the question the user actually wants answered:
"For each product, which strategy is best?"
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
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from strategy_harness import load_all_curves, load_dxy, COST_BPS, DEFAULT_BBL

TEST_FRAC = 0.20


def _settle_pnl(trades):
    """Settle trades with 5bps round-trip cost. Returns list of net PnLs."""
    out = []
    for t in trades:
        gross = int(t["direction"]) * (t["exit_price"] - t["entry_price"]) * t.get("notional_bbl", DEFAULT_BBL)
        cost = (COST_BPS / 10000.0) * abs(t["entry_price"]) * t.get("notional_bbl", DEFAULT_BBL)
        out.append({**t, "net_pnl": gross - cost})
    return out


def _summary(trades, name, product=None):
    if not trades:
        return {"strategy": name, "product": product or "ALL",
                "n_trades": 0, "total_pnl": 0, "win_rate": 0,
                "sharpe": 0, "max_dd": 0, "test_r2": None}
    s = _settle_pnl(trades)
    pnls = np.array([t["net_pnl"] for t in s])
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = float((peak - cum).max()) if len(cum) else 0.0
    mean_pnl = float(pnls.mean())
    std_pnl  = float(pnls.std(ddof=1)) if len(pnls) > 1 else 0.0
    sharpe = mean_pnl / std_pnl * math.sqrt(252.0/5.0) if std_pnl > 0 else 0.0
    return {
        "strategy":  name,
        "product":   product or "ALL",
        "n_trades":  int(len(pnls)),
        "total_pnl": float(pnls.sum()),
        "win_rate":  float((pnls > 0).mean() * 100),
        "sharpe":    sharpe,
        "max_dd":    dd,
    }


# =========================================================== #
# Strategies that ALSO return their (pred, actual) arrays for R²
# =========================================================== #
def run_pca_curve(curves, dxy) -> Dict:
    """PCA Curve - per product, compute R² of predicted PC3 z-shock vs realized fly change."""
    per_prod = {}
    for prod_code in ("CL", "LCO", "HO", "LGO"):
        if prod_code not in curves:
            continue
        curve = curves[prod_code]
        cols = [f"m{i}" for i in range(1, 13) if f"m{i}" in curve.columns]
        if len(cols) < 8:
            continue
        prices = curve[cols].dropna()
        if len(prices) < 250:
            continue
        log_ret = np.log(prices / prices.shift(1)).dropna()
        n = len(log_ret)
        train_end = int(n * 0.60)
        val_end   = int(n * 0.80)
        train_X = log_ret.iloc[:train_end].values
        pca = PCA(n_components=3)
        pca.fit(train_X)

        test_X = log_ret.iloc[val_end:].values
        scores = pca.transform(test_X)
        pc3 = scores[:, 2]
        roll_mean = pd.Series(pc3).rolling(20).mean()
        roll_std  = pd.Series(pc3).rolling(20).std()
        z = (pd.Series(pc3) - roll_mean) / roll_std
        test_idx = log_ret.index[val_end:]
        pc3_load = pca.components_[2]
        sorted_loads = np.argsort(np.abs(pc3_load))[::-1]
        leg_indices = sorted_loads[:3]
        leg_signs = np.sign(pc3_load[leg_indices])
        leg_cols = [cols[i] for i in leg_indices]
        leg_prices_test = prices[leg_cols].iloc[val_end+1:].copy()
        fly = (leg_prices_test * leg_signs).sum(axis=1)
        # Predicted fly change = -z (mean reversion prediction)
        # Actual fly change over 15 days forward
        actual_chg = fly.diff(15).shift(-15)
        z_aligned = z.reindex(fly.index).fillna(0)
        # only valid bars
        valid = (~actual_chg.isna()) & (~z_aligned.isna())
        if valid.sum() < 10:
            continue
        # Predicted direction = -sign(z) × |z|
        pred = (-z_aligned.values).astype(float)
        actual = actual_chg.values.astype(float)
        ok = (~np.isnan(pred)) & (~np.isnan(actual)) & np.isfinite(pred) & np.isfinite(actual)
        if ok.sum() < 10:
            continue
        r2 = float(r2_score(actual[ok], pred[ok])) if np.var(actual[ok]) > 0 else 0.0

        # Trade list
        trades = []
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
                            "direction":  int(position),
                            "entry_price": float(fly.iloc[entry_i]),
                            "exit_price":  float(fly.iloc[i]),
                            "notional_bbl": DEFAULT_BBL,
                        })
                    position = 0
                    entry_i = None
        summary = _summary(trades, "S6_PCA_Curve", product=prod_code)
        summary["test_r2"] = r2
        per_prod[prod_code] = summary
    return per_prod


def run_bertram_ou(curves, dxy) -> Dict:
    """Bertram OU - aggregate per product with R² of next-step prediction vs actual."""
    per_prod = {}
    for prod_code in ("CL", "LCO", "HO", "LGO"):
        if prod_code not in curves:
            continue
        curve = curves[prod_code]
        all_preds = []
        all_actuals = []
        all_trades = []
        for spread_pair in [(1, 2), (2, 3), (3, 4), (5, 6), (8, 9)]:
            a, b = spread_pair
            ka, kb = f"m{a}", f"m{b}"
            if ka not in curve.columns or kb not in curve.columns:
                continue
            s_raw = (curve[ka] - curve[kb]).dropna()
            n = len(s_raw)
            if n < 250:
                continue
            train_end = int(n * 0.60)
            val_end   = int(n * 0.80)
            train_seg = s_raw.iloc[:train_end]
            mu_hat = float(train_seg.mean())
            ds = train_seg.diff().dropna()
            s_lag = train_seg.shift(1).dropna().reindex(ds.index)
            slope, _ = np.polyfit(s_lag.values, ds.values, 1)
            alpha_hat = -slope
            eta_hat = float(ds.std(ddof=1))
            if alpha_hat <= 0 or eta_hat <= 0:
                continue
            sigma_eq = eta_hat / math.sqrt(2 * alpha_hat)
            a_star = 1.5 * sigma_eq

            # Predicted next-step delta = -alpha*(S - mu)
            test_seg = s_raw.iloc[val_end:]
            preds = -alpha_hat * (test_seg - mu_hat)
            actuals = test_seg.diff().shift(-1)
            valid = (~preds.isna()) & (~actuals.isna()) & np.isfinite(preds) & np.isfinite(actuals)
            if valid.sum() > 5:
                all_preds.extend(preds[valid].tolist())
                all_actuals.extend(actuals[valid].tolist())

            # trade list
            position = 0
            entry_i = None
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
                    if (position > 0 and deviation >= 0) or \
                       (position < 0 and deviation <= 0) or \
                       hold >= 30:
                        all_trades.append({
                            "product":   prod_code,
                            "direction":  int(position),
                            "entry_price": float(test_seg.iloc[entry_i]),
                            "exit_price":  float(test_seg.iloc[i]),
                            "notional_bbl": DEFAULT_BBL,
                        })
                        position = 0
                        entry_i = None
        if all_preds and np.var(all_actuals) > 0:
            r2 = float(r2_score(all_actuals, all_preds))
        else:
            r2 = 0.0
        summary = _summary(all_trades, "S5_Bertram_OU", product=prod_code)
        summary["test_r2"] = r2
        per_prod[prod_code] = summary
    return per_prod


def run_twofactor_crack(curves, dxy) -> Dict:
    """HO-CL crack only."""
    cl = curves["CL"]["m1"].dropna()
    ho_bbl = curves["HO"]["m1"].dropna() * 42.0
    common = cl.index.intersection(ho_bbl.index)
    cl = cl.reindex(common); ho_bbl = ho_bbl.reindex(common)
    log_spread = np.log(ho_bbl) - np.log(cl)
    n = len(log_spread)
    train_end = int(n * 0.60)
    val_end   = int(n * 0.80)
    train_seg = log_spread.iloc[:train_end]
    mu = float(train_seg.mean()); sigma = float(train_seg.std(ddof=1))
    if sigma == 0:
        return {"HO": _summary([], "S3_TwoFactor_Crack", "HO")}
    z = (log_spread - mu) / sigma
    z_roll_mean = z.rolling(60).mean()
    z_roll_std  = z.rolling(60).std()
    z_adj = (z - z_roll_mean) / z_roll_std

    # R² of forward 5d log-spread change vs predicted (= -z_adj)
    actual = log_spread.diff(5).shift(-5).reindex(z_adj.index)
    pred = -z_adj
    ok = (~pred.isna()) & (~actual.isna()) & np.isfinite(pred) & np.isfinite(actual)
    if ok.sum() < 10 or np.var(actual[ok]) <= 0:
        r2 = 0.0
    else:
        r2 = float(r2_score(actual[ok], pred[ok]))

    Z_ENTRY = 2.0; Z_EXIT = 0.5; Z_STOP = 3.5; MAX_HOLD = 30
    position = 0; entry_i = None; entry_z = None
    trades = []
    for i in range(val_end, n):
        zi = z_adj.iloc[i]
        if not np.isfinite(zi):
            continue
        if position == 0:
            if zi >= +Z_ENTRY:
                position = -1; entry_i = i; entry_z = zi
            elif zi <= -Z_ENTRY:
                position = +1; entry_i = i; entry_z = zi
        else:
            hold = i - entry_i
            stop = (position > 0 and zi < entry_z - 1.5) or \
                   (position < 0 and zi > entry_z + 1.5) or abs(zi) >= Z_STOP
            done = abs(zi) <= Z_EXIT
            if done or stop or hold >= MAX_HOLD:
                trades.append({
                    "product":   "HO",
                    "direction":  int(position),
                    "entry_price": float(ho_bbl.iloc[entry_i] - cl.iloc[entry_i]),
                    "exit_price":  float(ho_bbl.iloc[i] - cl.iloc[i]),
                    "notional_bbl": DEFAULT_BBL,
                })
                position = 0; entry_i = None; entry_z = None
    summary = _summary(trades, "S3_TwoFactor_Crack", "HO")
    summary["test_r2"] = r2
    return {"HO": summary}


def run_hmm_wtcl(curves, dxy) -> Dict:
    wti = curves["CL"]["m1"].dropna()
    brent = curves["LCO"]["m1"].dropna()
    common = wti.index.intersection(brent.index)
    wti = wti.reindex(common); brent = brent.reindex(common)
    spread = np.log(wti) - np.log(brent)
    n = len(spread)
    train_end = int(n * 0.60)
    val_end   = int(n * 0.80)
    train_spread = spread.iloc[:train_end].values.reshape(-1, 1)
    try:
        hmm = GaussianHMM(n_components=2, covariance_type="full",
                           n_iter=100, random_state=42)
        hmm.fit(train_spread)
    except Exception:
        return {"WTCL": _summary([], "S1_HMM_OU_WTCL", "WTCL")}

    test_spread = spread.iloc[val_end:].values.reshape(-1, 1)
    posteriors = hmm.predict_proba(test_spread)
    test_idx = spread.index[val_end:]
    means = hmm.means_.flatten()
    low_state = int(np.argmin(means)); high_state = int(np.argmax(means))
    std_per_state = np.sqrt(hmm.covars_.reshape(2, 1, 1)[:, 0, 0])

    # R² of forward 5d spread change vs predicted (= cond_mean - current)
    actual_chg = pd.Series(spread.iloc[val_end:].values).diff(5).shift(-5)
    cond_means = (posteriors[:, 0] * means[0] + posteriors[:, 1] * means[1])
    pred_chg = cond_means - spread.iloc[val_end:].values
    ok_idx = ~np.isnan(actual_chg) & np.isfinite(pred_chg)
    if ok_idx.sum() > 10 and np.var(actual_chg[ok_idx]) > 0:
        r2 = float(r2_score(actual_chg[ok_idx], pred_chg[ok_idx]))
    else:
        r2 = 0.0

    trades = []
    position = 0; entry_i = None; entry_z = None
    for i in range(len(test_idx)):
        s_now = float(test_spread[i, 0])
        cond_mean = float(posteriors[i, 0] * means[0] + posteriors[i, 1] * means[1])
        cond_std  = float(posteriors[i, 0] * std_per_state[0] + posteriors[i, 1] * std_per_state[1])
        z = (s_now - cond_mean) / cond_std if cond_std > 0 else 0
        p_low = float(posteriors[i, low_state])
        p_high = float(posteriors[i, high_state])
        if position == 0:
            if p_low > 0.7 and z < -1.0:
                position = +1; entry_i = i; entry_z = z
            elif p_high > 0.7 and z > +1.0:
                position = -1; entry_i = i; entry_z = z
        else:
            hold = i - entry_i
            stopped = (position > 0 and z < entry_z - 2.0) or \
                      (position < 0 and z > entry_z + 2.0)
            regime_flip = (position > 0 and p_high > 0.5) or \
                          (position < 0 and p_low > 0.5)
            if abs(z) < 0.25 or stopped or regime_flip or hold >= 20:
                entry_dt = test_idx[entry_i]
                avg_px = float((wti.loc[entry_dt] + brent.loc[entry_dt]) / 2)
                trades.append({
                    "product":   "WTCL",
                    "direction":  int(position),
                    "entry_price": float(spread.iloc[val_end + entry_i] * avg_px),
                    "exit_price":  float(spread.iloc[val_end + i] * avg_px),
                    "notional_bbl": DEFAULT_BBL,
                })
                position = 0; entry_i = None; entry_z = None
    summary = _summary(trades, "S1_HMM_OU_WTCL", "WTCL")
    summary["test_r2"] = r2
    return {"WTCL": summary}


def run_mlp_wti(curves, dxy) -> Dict:
    wti = curves["CL"]["m1"].dropna()
    ema20 = wti.ewm(span=20).mean()
    ema60 = wti.ewm(span=60).mean()
    ema100 = wti.ewm(span=100).mean()
    feats = pd.DataFrame({
        "ema20_60":  ema20 / ema60, "ema20_100": ema20 / ema100,
        "ema60_100": ema60 / ema100,
        "ret_1d":  wti.pct_change(1) * 100,
        "ret_5d":  wti.pct_change(5) * 100,
        "ret_20d": wti.pct_change(20) * 100,
        "vol_20d": np.log(wti / wti.shift(1)).rolling(20).std() * math.sqrt(252) * 100,
    })
    feats["target"] = (wti.shift(-1) > wti).astype(int)
    feats = feats.dropna()
    n = len(feats)
    train_end = int(n * 0.60); val_end = int(n * 0.80)
    Xcols = [c for c in feats.columns if c != "target"]
    X = feats[Xcols].values; y = feats["target"].values
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X[:train_end])
    X_te_s = scaler.transform(X[val_end:])
    clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=400,
                         random_state=42, alpha=0.01)
    clf.fit(X_tr_s, y[:train_end])
    pred = clf.predict(X_te_s)
    proba = clf.predict_proba(X_te_s)[:, 1]
    # R²: treat proba - 0.5 as a continuous predictor of next-day return
    test_idx = feats.index[val_end:]
    test_wti = wti.reindex(test_idx)
    next_ret = test_wti.pct_change(1).shift(-1).values
    proba_centered = proba - 0.5
    ok = ~np.isnan(next_ret) & np.isfinite(proba_centered)
    if ok.sum() > 10 and np.var(next_ret[ok]) > 0:
        # scale: we predict direction, not magnitude — R² will be low
        # use sign agreement (not R² of raw values)
        agree = (np.sign(proba_centered[ok]) == np.sign(next_ret[ok])).mean()
        # Convert to pseudo-R² as 2*agree - 1 (correlation analog)
        r2 = float(2 * agree - 1)
    else:
        r2 = 0.0

    trades = []
    for i in range(len(test_idx) - 1):
        signal = pred[i]
        confidence = abs(proba[i] - 0.5) * 2
        if confidence < 0.10:
            continue
        direction = +1 if signal == 1 else -1
        entry_px = float(test_wti.iloc[i])
        exit_px  = float(test_wti.iloc[i + 1])
        if not np.isfinite(entry_px) or not np.isfinite(exit_px):
            continue
        trades.append({
            "product":   "CL",
            "direction":  direction,
            "entry_price": entry_px, "exit_price": exit_px,
            "notional_bbl": DEFAULT_BBL,
        })
    summary = _summary(trades, "S4_MLP_WTI", "CL")
    summary["test_r2"] = r2
    return {"CL": summary}


def run_washu_brent(curves, dxy) -> Dict:
    brent = curves["LCO"]["m1"].dropna()
    monthly = brent.resample("MS").first().dropna()
    dxy_m = dxy.resample("MS").last().reindex(monthly.index, method="ffill").dropna()
    common = monthly.index.intersection(dxy_m.index)
    monthly = monthly.reindex(common); dxy_m = dxy_m.reindex(common)
    n = len(monthly)
    train_end = int(n * 0.60)
    fwd_high = monthly.rolling(window=3).max().shift(-2).reindex(monthly.index)

    preds_history = []
    actuals_history = []
    trades = []
    open_pos = None
    for i in range(train_end, n - 1):
        if open_pos is None:
            start = max(0, i - 12)
            if i - start < 6:
                continue
            X = np.column_stack([
                np.log(monthly.iloc[start:i].values),
                np.log(dxy_m.iloc[start:i].values),
            ])
            X = sm.add_constant(X)
            y = np.log(fwd_high.iloc[start:i].fillna(monthly.iloc[start:i]).values)
            m = sm.OLS(y, X).fit()
            x_now = sm.add_constant(np.array([[
                np.log(monthly.iloc[i]), np.log(dxy_m.iloc[i])
            ]]), has_constant="add")
            ln_pa = float(m.predict(x_now)[0])
            Pa = math.exp(ln_pa)
            close = float(monthly.iloc[i])
            # record (predicted log change, actual log change of forward high)
            actual_log = float(np.log(fwd_high.iloc[i])) if not np.isnan(fwd_high.iloc[i]) else None
            if actual_log is not None:
                preds_history.append(ln_pa)
                actuals_history.append(actual_log)
            pred_ret = (Pa - close) / close
            direction = +1 if pred_ret >= 0.01 else (-1 if pred_ret <= -0.01 else 0)
            if direction == 0:
                continue
            open_pos = {"entry_date": monthly.index[i], "entry_price": close,
                         "target_Pa": Pa, "direction": direction, "entry_idx": i}
        else:
            cur = float(monthly.iloc[i])
            hit = (open_pos["direction"] == +1 and cur >= open_pos["target_Pa"]) or \
                  (open_pos["direction"] == -1 and cur <= open_pos["target_Pa"])
            timeout = (i - open_pos["entry_idx"]) >= 12
            if hit or timeout:
                trades.append({
                    "product":   "LCO",
                    "direction":  open_pos["direction"],
                    "entry_price": open_pos["entry_price"],
                    "exit_price":  cur,
                    "notional_bbl": DEFAULT_BBL,
                })
                open_pos = None
    if len(preds_history) > 5 and np.var(actuals_history) > 0:
        r2 = float(r2_score(actuals_history, preds_history))
    else:
        r2 = 0.0
    summary = _summary(trades, "S2_WashU_Brent", "LCO")
    summary["test_r2"] = r2
    return {"LCO": summary}


# ===================================================================== #
def main():
    print("Loading data...")
    curves = load_all_curves()
    dxy = load_dxy()
    print(f"  curves: {sorted(curves.keys())}\n")

    all_results = {}  # product -> [summaries]
    print("Running strategies per product...")
    for name, fn in [
        ("S6_PCA_Curve",        run_pca_curve),
        ("S5_Bertram_OU",       run_bertram_ou),
        ("S3_TwoFactor_Crack",  run_twofactor_crack),
        ("S1_HMM_OU_WTCL",      run_hmm_wtcl),
        ("S4_MLP_WTI",          run_mlp_wti),
        ("S2_WashU_Brent",      run_washu_brent),
    ]:
        try:
            per = fn(curves, dxy)
            for prod, s in per.items():
                all_results.setdefault(prod, []).append(s)
        except Exception as e:
            print(f"  {name}: ERROR {type(e).__name__}: {e}")

    # PRINT: per-product table
    print()
    print("=" * 110)
    print("PER-PRODUCT STRATEGY EVALUATION — all 6 strategies × 5 products, out-of-sample test")
    print("=" * 110)
    products_to_show = sorted(set(p for p in all_results.keys()))
    for prod in products_to_show:
        rows = all_results[prod]
        print(f"\n[{prod}]")
        print(f"  {'Strategy':<22}{'Trades':>8}{'TotalPnL$':>13}{'WinRate':>10}{'Sharpe':>9}{'MaxDD$':>11}{'TestR²':>9}")
        print("  " + "-" * 95)
        for r in sorted(rows, key=lambda x: -(x.get("total_pnl") or 0)):
            r2_str = f"{r['test_r2']:+.3f}" if r.get('test_r2') is not None else "n/a"
            print(f"  {r['strategy']:<22}{r['n_trades']:>8}{r['total_pnl']:>+13,.0f}"
                  f"{r['win_rate']:>9.0f}%{r['sharpe']:>+9.2f}{r['max_dd']:>11,.0f}{r2_str:>9}")
        # winner per product
        winners = [r for r in rows if r["total_pnl"] > 0 and r["sharpe"] > 0]
        if winners:
            best = max(winners, key=lambda r: r["sharpe"])
            print(f"  >>> BEST for {prod}: {best['strategy']} (Sharpe {best['sharpe']:+.2f}, PnL ${best['total_pnl']:+,.0f})")
        else:
            print(f"  >>> No profitable strategy for {prod}")

    # Save
    out_path = Path(__file__).resolve().parents[1] / "tools" / "results" / "per_product_eval.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved -> tools/results/per_product_eval.json")


if __name__ == "__main__":
    main()
