"""Reusable backtest harness for testing different oil-trading strategies
on the user's xlsx data + DXY.

Each strategy is a function:
    strategy(curves_dict, dxy_series, kind) -> List[Trade]

where Trade = dict with keys:
    product:   str (CL/LCO/LGO/HO/WTCL)
    instrument: str (e.g. "m1", "m9-m10", "m3-2m4+m5")
    entry_date: pd.Timestamp
    exit_date:  pd.Timestamp
    direction: int (+1 long, -1 short)
    entry_price: float
    exit_price:  float
    notional_bbl: int (default 1000)

The harness then computes:
    - PnL per trade (with 5 bps round-trip cost)
    - Aggregate: total PnL, n_trades, win_rate, Sharpe, max_dd, profit_factor
    - Per-product breakdown
"""
from __future__ import annotations
import math
import warnings
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_FILES = {
    "CL":   ROOT / "CL_data_trimmed_daily_close.xlsx",
    "LCO":  ROOT / "LCO_data_trimmed_daily_close.xlsx",
    "LGO":  ROOT / "LGO_data_trimmed_daily_close.xlsx",
    "HO":   ROOT / "HO_data_trimmed_daily_close.xlsx",
    "WTCL": ROOT / "wtcl_lco_outrights_1min_trimmed_daily_close.xlsx",
}

COST_BPS = 5.0          # round-trip transaction cost, 5 bps
DEFAULT_BBL = 1000      # per contract
TEST_FRACTION = 0.20    # last 20% out-of-sample


def load_all_curves() -> Dict[str, pd.DataFrame]:
    """Load M1-M14 for each product into a dict of DataFrames."""
    out = {}
    for prod, path in PRODUCT_FILES.items():
        df = pd.read_excel(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        curve = pd.DataFrame(index=df.index)
        for i in range(1, 15):
            col = f"c{i}||weighted_mid"
            if col in df.columns:
                curve[f"m{i}"] = pd.to_numeric(df[col], errors="coerce")
        out[prod] = curve
    return out


def load_dxy() -> pd.Series:
    raw = yf.download("DX-Y.NYB", period="5y", progress=False,
                       auto_adjust=True, threads=True)["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return raw


def test_split_date(series: pd.Series) -> pd.Timestamp:
    """Date at which test slice begins (last TEST_FRACTION of bars)."""
    idx = int(len(series) * (1 - TEST_FRACTION))
    return series.index[idx]


def settle_trade(trade: Dict) -> Dict:
    """Apply 5 bps cost to gross PnL and compute net."""
    entry = float(trade["entry_price"])
    exit_  = float(trade["exit_price"])
    direction = int(trade["direction"])
    notional = float(trade["notional_bbl"])
    gross = direction * (exit_ - entry) * notional
    # 5 bps cost on entry NOTIONAL value, applied round-trip
    cost = (COST_BPS / 10000.0) * abs(entry) * notional
    return {
        **trade,
        "gross_pnl":  gross,
        "cost":       cost,
        "net_pnl":    gross - cost,
    }


def backtest(strategy_fn: Callable, name: str) -> Dict:
    """Run a strategy function on the loaded data, settle trades, report."""
    print(f"\n[{name}] loading data...", flush=True)
    curves = load_all_curves()
    dxy = load_dxy()

    print(f"[{name}] generating trades...", flush=True)
    raw_trades = strategy_fn(curves, dxy)
    if not raw_trades:
        return {"strategy": name, "n_trades": 0,
                "summary": "no trades generated"}

    trades = [settle_trade(t) for t in raw_trades]

    # Aggregate metrics
    pnls = np.array([t["net_pnl"] for t in trades])
    wins = (pnls > 0).sum()
    total = pnls.sum()
    win_rate = wins / len(pnls) * 100
    mean_pnl = pnls.mean()
    std_pnl = pnls.std(ddof=1) if len(pnls) > 1 else 0
    sharpe = mean_pnl / std_pnl * math.sqrt(252.0 / 5.0) if std_pnl > 0 else 0
    # max drawdown of cumulative PnL
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum).max() if len(cum) > 0 else 0
    pos_pnls = pnls[pnls > 0]
    neg_pnls = pnls[pnls < 0]
    profit_factor = abs(pos_pnls.sum() / neg_pnls.sum()) if len(neg_pnls) and neg_pnls.sum() else float("inf")

    # By-product breakdown
    by_prod = {}
    for prod in set(t["product"] for t in trades):
        prod_pnls = np.array([t["net_pnl"] for t in trades if t["product"] == prod])
        by_prod[prod] = {
            "n_trades": int(len(prod_pnls)),
            "total_pnl": float(prod_pnls.sum()),
            "win_rate": float((prod_pnls > 0).sum() / len(prod_pnls) * 100),
        }

    print(f"[{name}] n_trades={len(trades)}  PnL=${total:+,.0f}  Win%={win_rate:.0f}  Sharpe={sharpe:+.2f}", flush=True)

    return {
        "strategy":    name,
        "n_trades":    int(len(trades)),
        "total_pnl":   float(total),
        "mean_pnl":    float(mean_pnl),
        "win_rate":    float(win_rate),
        "sharpe":      float(sharpe),
        "max_dd":      float(dd),
        "profit_factor": float(profit_factor) if profit_factor != float("inf") else 999.0,
        "by_product":  by_prod,
    }


if __name__ == "__main__":
    # Quick sanity test: a baseline "buy and hold M1 of each product" strategy
    def buy_and_hold_m1(curves, dxy):
        trades = []
        for prod, curve in curves.items():
            if "m1" not in curve.columns:
                continue
            s = curve["m1"].dropna()
            split = test_split_date(s)
            test_s = s[s.index >= split]
            if len(test_s) < 5:
                continue
            trades.append({
                "product": prod,
                "instrument": "m1",
                "entry_date": test_s.index[0],
                "exit_date":  test_s.index[-1],
                "direction":  +1,
                "entry_price": float(test_s.iloc[0]),
                "exit_price":  float(test_s.iloc[-1]),
                "notional_bbl": DEFAULT_BBL,
            })
        return trades

    print("Harness test: buy-and-hold M1 of each product over test slice")
    print(backtest(buy_and_hold_m1, "BuyHold_M1"))
