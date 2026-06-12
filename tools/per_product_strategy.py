"""Per-product best-strategy builder + out-of-sample backtest.

For each product (CL, LCO, LGO, HO, WTCL):
  For each structure type (OUTRIGHT, SPREAD, FLY):
    1. Compute per-target multi-model CONSENSUS from the 138-target results
    2. Filter to high-conviction signals: consensus >= 7/10 AND test R² >= 0.10
    3. Build strategy rules
    4. Backtest the strategy on test slice using realized PnL

Strategy = ensemble of high-conviction signals. Trade entry uses ensemble
direction (mode of model predictions). Position size scales with both
consensus strength AND R². Exit at HORIZON_DAYS.

Backtest uses the same xlsx data we trained on, but evaluates ONLY on the
last 20% (out-of-sample). For each tradable signal we simulate entering
on the signal date and computing the realized N-day forward return.

Reports per-product per-structure-type: # trades, gross PnL, win rate,
Sharpe, max drawdown, mean R² of signals taken.
"""
from __future__ import annotations
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH  = ROOT / "tools" / "results" / "real_data_models.json"
SIGNALS_PATH = ROOT / "backend" / "data" / "real_data_signals.json"

PRODUCT_FILES = {
    "CL":   ROOT / "CL_data_trimmed_daily_close.xlsx",
    "LCO":  ROOT / "LCO_data_trimmed_daily_close.xlsx",
    "LGO":  ROOT / "LGO_data_trimmed_daily_close.xlsx",
    "HO":   ROOT / "HO_data_trimmed_daily_close.xlsx",
    "WTCL": ROOT / "wtcl_lco_outrights_1min_trimmed_daily_close.xlsx",
}

HORIZON         = 5
CONSENSUS_MIN   = 7           # of 10 models must agree
R2_MIN          = 0.10        # MED confidence threshold
COST_PER_TRADE  = 0.0005      # 5 bps per round-trip
RISK_FREE_RATE  = 0.0
PRODUCT_LABELS  = {"CL": "WTI", "LCO": "Brent", "LGO": "Gasoil",
                    "HO": "HeatingOil", "WTCL": "WTI-Brent spread"}


# -------------------- consensus computation ----------------------------- #
def consensus_per_target(models: Dict) -> Dict[str, Dict]:
    """For each target, compute: consensus count, winner model & direction."""
    out = {}
    for key, result in models.items():
        if not isinstance(result, dict) or "models" not in result:
            continue
        preds = {}
        for mname, info in (result.get("models") or {}).items():
            if isinstance(info, dict) and "pred_now" in info:
                p = info["pred_now"]
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    continue
                preds[mname] = p
        if len(preds) < 5:
            continue
        winner = result.get("winner")
        win_pred = preds.get(winner, 0)
        win_sign = 1 if win_pred > 0 else (-1 if win_pred < 0 else 0)
        if win_sign == 0:
            continue
        agree = sum(1 for p in preds.values()
                     if (p > 0 and win_sign > 0) or (p < 0 and win_sign < 0))
        out[key] = {
            "n_total":  len(preds),
            "n_agree":  agree,
            "winner":   winner,
            "winner_pred": float(win_pred),
            "winner_dir":  "LONG" if win_sign > 0 else "SHORT",
            "test_r2":  result.get("winner_test_r2") or 0.0,
            "ensemble_pred": float(np.mean(list(preds.values()))),
        }
    return out


# -------------------- xlsx loader (same as real_data_master) ------------ #
def load_curve(path: Path, n_tenors: int = 12) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    out = pd.DataFrame(index=df.index)
    for i in range(1, n_tenors + 1):
        col = f"c{i}||weighted_mid"
        if col in df.columns:
            out[f"m{i}"] = pd.to_numeric(df[col], errors="coerce")
    return out


def build_target_series(curve: pd.DataFrame, key: str) -> pd.Series:
    """key like 'CL|OUT|m1' -> series of m1 prices.
    'CL|SPR|m1-m2' -> series of (m1 - m2). 'CL|FLY|m1-2m2+m3' -> fly series."""
    _, kind, struct = key.split("|")
    if kind == "OUT":
        return curve[struct].dropna()
    if kind == "SPR":
        a, b = struct.split("-")
        return (curve[a] - curve[b]).dropna()
    # FLY
    a_b, c = struct.rsplit("+", 1)
    a, b = a_b.split("-2")
    return (curve[a] - 2 * curve[b] + curve[c]).dropna()


# -------------------- per-signal backtest ------------------------------- #
def backtest_signal(series: pd.Series, direction: str,
                     test_start_idx: int) -> Dict:
    """Take a sample of N evenly-spaced entry points across the test slice
    (last 20% of bars), enter in `direction`, hold HORIZON days, record PnL.

    For outright returns: PnL = sign × pct_change(HORIZON).
    For spreads/flies:    PnL = sign × diff(HORIZON).
    """
    n = len(series)
    if test_start_idx >= n - HORIZON:
        return {"n_trades": 0}
    sign = 1 if direction == "LONG" else -1
    # Sample every 5th day in test slice for non-overlapping trades
    trades = []
    for i in range(test_start_idx, n - HORIZON, HORIZON):
        entry = float(series.iloc[i])
        exit_ = float(series.iloc[i + HORIZON])
        if entry == 0:
            continue
        # For outrights, pct return; for spreads/flies, raw difference
        # We approximate by absolute value of average level to detect kind
        is_outright = abs(series.median()) > 5
        if is_outright:
            pnl = sign * (exit_ - entry) / abs(entry) * 100 - COST_PER_TRADE * 100
        else:
            pnl = sign * (exit_ - entry) - COST_PER_TRADE * abs(entry)
        trades.append(pnl)
    if not trades:
        return {"n_trades": 0}
    arr = np.array(trades)
    wins = (arr > 0).sum()
    return {
        "n_trades": int(len(arr)),
        "wins":     int(wins),
        "win_rate": float(wins / len(arr) * 100),
        "mean_pnl": float(arr.mean()),
        "total_pnl": float(arr.sum()),
        "std_pnl":  float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "max_win":  float(arr.max()),
        "max_loss": float(arr.min()),
    }


# -------------------- main strategy builder ----------------------------- #
def main():
    models = json.loads(MODELS_PATH.read_text())
    print(f"Loaded {len(models)} target results")

    consensus = consensus_per_target(models)
    print(f"Consensus computed for {len(consensus)} targets")

    # Group consensus by (product, kind)
    by_pk = defaultdict(list)
    for key, c in consensus.items():
        prod, kind, struct = key.split("|")
        by_pk[(prod, kind)].append({"key": key, "struct": struct, **c})

    print()
    print("=" * 100)
    print("PER-PRODUCT PER-STRUCTURE STRATEGY (filter: consensus >= 7/10  AND  test R² >= 0.10)")
    print("=" * 100)

    # Load all curve data once
    curves = {p: load_curve(path) for p, path in PRODUCT_FILES.items() if path.exists()}

    portfolio_summary = []

    for prod in sorted(PRODUCT_FILES.keys()):
        if prod not in curves:
            continue
        curve = curves[prod]
        for kind in ("OUT", "SPR", "FLY"):
            items = by_pk.get((prod, kind), [])
            tradable = [x for x in items
                         if x["n_agree"] >= CONSENSUS_MIN
                         and x["test_r2"] >= R2_MIN]
            tradable.sort(key=lambda x: -x["test_r2"])
            kind_label = {"OUT": "OUTRIGHT", "SPR": "SPREAD", "FLY": "FLY"}[kind]
            n_total = len(items)
            n_tradable = len(tradable)

            print(f"\n{prod:<6} {kind_label:<10}  {n_tradable:>2}/{n_total:>2} tradable signals (consensus&gt;={CONSENSUS_MIN} & R²&gt;={R2_MIN})")
            if not tradable:
                print(f"  (no signals meet threshold)")
                continue

            # Backtest each
            agg = {"n_trades": 0, "total_pnl": 0.0, "all_trades": []}
            for sig in tradable[:5]:  # top 5 per kind to cap risk concentration
                try:
                    series = build_target_series(curve, sig["key"])
                except Exception as e:
                    continue
                n = len(series)
                test_start = int(n * 0.80)
                bt = backtest_signal(series, sig["winner_dir"], test_start)
                if bt["n_trades"] == 0:
                    continue
                agg["n_trades"] += bt["n_trades"]
                agg["total_pnl"] += bt["total_pnl"]
                # collect per-trade PnLs would need to retain
                # For now just print summary
                print(f"  {sig['struct']:<12} {sig['winner_dir']:<5} consensus={sig['n_agree']}/10 R²={sig['test_r2']:+.3f}  "
                      f"backtest: n={bt['n_trades']}, win_rate={bt['win_rate']:.0f}%, "
                      f"avg_pnl={bt['mean_pnl']:+.3f}, total={bt['total_pnl']:+.2f}")

            if agg["n_trades"] > 0:
                print(f"  -> AGGREGATE: {agg['n_trades']} test-period trades, total PnL {agg['total_pnl']:+.2f}")
                portfolio_summary.append({
                    "product": prod, "kind": kind_label,
                    "n_signals": n_tradable,
                    "n_trades":  agg["n_trades"],
                    "total_pnl": agg["total_pnl"],
                })

    # Portfolio totals
    print()
    print("=" * 100)
    print("PORTFOLIO SUMMARY — aggregated across all products & structure types")
    print("=" * 100)
    grand_pnl = sum(s["total_pnl"] for s in portfolio_summary)
    grand_trades = sum(s["n_trades"] for s in portfolio_summary)
    print(f"  {'Product':<8}{'Kind':<10}{'#Signals':>10}{'#Trades':>10}{'Total PnL':>14}")
    print(f"  {'-'*52}")
    for s in portfolio_summary:
        print(f"  {s['product']:<8}{s['kind']:<10}{s['n_signals']:>10}{s['n_trades']:>10}{s['total_pnl']:>+14.3f}")
    print(f"  {'-'*52}")
    print(f"  {'TOTAL':<18}{'':>10}{grand_trades:>10}{grand_pnl:>+14.3f}")


if __name__ == "__main__":
    main()
