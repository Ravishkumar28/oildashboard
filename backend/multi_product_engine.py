"""Live trading signals for all 5 products + 3 spreads.

Output is intentionally minimal — one row per tradeable instrument with:
  * current price/level
  * signal: LONG / SHORT / FLAT
  * predicted 5d move
  * confidence tier (HIGH / MED / LOW based on historical out-of-sample Sharpe)

The engine trains LGBM_dash on 5y of daily data, then makes ONE prediction
on TODAY's feature vector. The historical 80/20 backtest is run only to
attach a confidence tag (HIGH if Sharpe > 1.5 & WinRate > 60% & ntrades >=
10, MED if Sharpe > 0.5, else LOW). The backtest numbers themselves are
not exposed in the panel — only the actionable signal is.

Refits at most once per 6h (yfinance is daily so faster is wasted compute).
"""
from __future__ import annotations
import math
import time
import warnings
import threading
from typing import Dict, List, Optional

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                       ElasticNet, HuberRegressor)
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    _SK = True
except Exception:
    _SK = False

try:
    from lightgbm import LGBMRegressor
    _LGBM = True
except Exception:
    LGBMRegressor = None
    _LGBM = False

warnings.filterwarnings("ignore")

PERIOD          = "5y"
HORIZON         = 5
LONG_THRESH     = 0.5    # % (or $/bbl for spreads, scaled below)
COST_BPS        = 5.0
REFIT_EVERY_SEC = 6 * 60 * 60

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


def _bb_pos(close, w=20):
    m = close.rolling(w).mean()
    s = close.rolling(w).std()
    return ((close - (m - 2*s)) / (4*s) * 100).clip(0, 100)


def _rsi(close, p=14):
    d = close.diff()
    up = d.where(d > 0, 0.0).rolling(p).mean()
    dn = (-d.where(d < 0, 0.0)).rolling(p).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _vol_z(close):
    lr = np.log(close / close.shift(1))
    v20 = lr.rolling(20).std() * math.sqrt(252) * 100
    return (v20 - v20.rolling(60).mean()) / v20.rolling(60).std()


def _features(target_close, dxy_close, kind="ret"):
    df = pd.DataFrame(index=target_close.index)
    if kind == "ret":
        df["ret_1d"]  = target_close.pct_change(1)  * 100
        df["ret_5d"]  = target_close.pct_change(5)  * 100
        df["ret_20d"] = target_close.pct_change(20) * 100
        y = (target_close.shift(-HORIZON) / target_close - 1) * 100
    else:
        df["ret_1d"]  = target_close.diff(1)
        df["ret_5d"]  = target_close.diff(5)
        df["ret_20d"] = target_close.diff(20)
        y = target_close.shift(-HORIZON) - target_close
    df["dxy_5d_ret"]  = dxy_close.pct_change(5) * 100
    df["dxy_20d_ret"] = dxy_close.pct_change(20) * 100
    df["bb_pos"]      = _bb_pos(target_close, 20)
    df["vol_z"]       = _vol_z(target_close)
    df["rsi_14n"]     = (_rsi(target_close, 14) - 50) / 50
    df_ft = df[FEAT_NAMES].copy()
    df_full = df.assign(target=y).dropna()
    # The most recent row of df_ft (where features are computed but target
    # is unknown — that's our "today" feature vector).
    df_ft_now = df_ft.dropna()
    return df_full[FEAT_NAMES], df_full["target"], df_ft_now.iloc[-1].values


def _build_model():
    if _LGBM:
        return LGBMRegressor(
            n_estimators=30, max_depth=2, num_leaves=4,
            learning_rate=0.03, min_child_samples=8,
            min_data_in_bin=2, min_split_gain=0.0,
            reg_alpha=1.0, reg_lambda=1.0,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
            random_state=42, verbosity=-1, force_col_wise=True,
        )
    # Fallback if lightgbm somehow not present at runtime.
    return Lasso(alpha=0.05, max_iter=5000)


def _backtest_sharpe(pred_te, y_te, threshold):
    sig = np.where(pred_te > threshold, 1,
            np.where(pred_te < -threshold, -1, 0))
    pnl = []
    for i in range(0, len(sig), HORIZON):
        if sig[i] == 0:
            continue
        pnl.append(float(sig[i] * y_te[i] - COST_BPS/100.0))
    if not pnl:
        return 0.0, 0.0, 0
    arr = np.array(pnl)
    sh = (arr.mean() / arr.std() * math.sqrt(252.0 / HORIZON)
          if arr.std() > 0 else 0.0)
    wr = float((arr > 0).mean() * 100)
    return float(sh), wr, int(len(arr))


def _confidence_tier(sharpe: float, win_rate: float, n_trades: int) -> str:
    if n_trades < 10:
        return "LOW"
    if sharpe >= 1.5 and win_rate >= 60:
        return "HIGH"
    if sharpe >= 0.5 and win_rate >= 50:
        return "MED"
    return "LOW"


def _analyze_one(label, target_close, dxy_close, kind="ret"):
    X_df, y_ser, x_now = _features(target_close, dxy_close, kind=kind)
    if len(X_df) < 200 or x_now is None:
        return None

    X = X_df.values.astype(float)
    y = y_ser.values.astype(float)
    split = int(len(X) * 0.8)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[:split])
    X_te = sc.transform(X[split:])

    model = _build_model()
    try:
        model.fit(X_tr, y[:split])
    except Exception:
        return None
    pred_te = model.predict(X_te)

    # Threshold: percent for returns, half a typical 1-day move for spreads.
    threshold = (LONG_THRESH if kind == "ret"
                 else float(np.median(np.abs(np.diff(target_close.dropna().values)))) * 0.5)
    sh, wr, n_tr = _backtest_sharpe(pred_te, y[split:], threshold)
    tier = _confidence_tier(sh, wr, n_tr)

    # Now — refit on ALL data and predict on today's features.
    sc_full = StandardScaler()
    X_all_s = sc_full.fit_transform(X)
    model_full = _build_model()
    try:
        model_full.fit(X_all_s, y)
    except Exception:
        model_full = model
        X_all_s = X_tr
    x_now_s = sc_full.transform(x_now.reshape(1, -1))
    pred_now = float(model_full.predict(x_now_s)[0])

    signal = "LONG" if pred_now > threshold else ("SHORT" if pred_now < -threshold else "FLAT")
    current_level = float(target_close.iloc[-1])

    return {
        "label":           label,
        "kind":            kind,
        "current_level":   round(current_level, 4),
        "predicted_move":  round(pred_now, 3),
        "signal":          signal,
        "confidence":      tier,
        "hist_sharpe":     round(sh, 2),
        "hist_win_rate":   round(wr, 1),
        "hist_n_trades":   n_tr,
    }


class _Cache:
    def __init__(self):
        self.payload: Optional[Dict] = None
        self.last: float = 0.0
        self.lock = threading.Lock()

    def get(self) -> Dict:
        with self.lock:
            now = time.time()
            if self.payload and (now - self.last) < REFIT_EVERY_SEC:
                return self.payload
            self.payload = self._compute() or {"available": False,
                                               "reason": "engine returned no data"}
            self.last = now
            return self.payload

    def _compute(self) -> Optional[Dict]:
        if not _SK:
            return {"available": False, "reason": "scikit-learn missing"}
        try:
            raw = yf.download(list(PRODUCTS.values()) + ["DX-Y.NYB"],
                              period=PERIOD, progress=False,
                              auto_adjust=True, threads=True)
            close = raw["Close"]
        except Exception as e:
            return {"available": False, "reason": f"yfinance fetch failed: {e}"}

        dxy = close["DX-Y.NYB"]
        signals: List[Dict] = []

        # Per-product outright signals.
        for p, tkr in PRODUCTS.items():
            ts = close[tkr].dropna()
            dxy_a = dxy.reindex(ts.index, method="ffill")
            r = _analyze_one(p, ts, dxy_a, kind="ret")
            if r is not None:
                signals.append(r)

        # Spread signals — known oil-desk trades.
        wti   = close["CL=F"].dropna()
        brent = close["BZ=F"].reindex(wti.index, method="ffill")
        rb    = close["RB=F"].reindex(wti.index, method="ffill") * 42  # $/gal -> $/bbl
        ho    = close["HO=F"].reindex(wti.index, method="ffill") * 42

        spreads = {
            "WTI-Brent": (wti - brent).dropna(),
            "321Crack":  (2*rb + ho - 3*wti).dropna(),
            "RBOB-HO":   (rb - ho).dropna(),
        }
        for name, ser in spreads.items():
            dxy_a = dxy.reindex(ser.index, method="ffill")
            r = _analyze_one(name, ser, dxy_a, kind="diff")
            if r is not None:
                signals.append(r)

        # Order: HIGH confidence LONG/SHORT first, then MED, then LOW; FLAT last.
        rank = {"HIGH": 0, "MED": 1, "LOW": 2}
        signals.sort(key=lambda s: (
            0 if s["signal"] != "FLAT" else 1,
            rank.get(s["confidence"], 9),
            -abs(s["predicted_move"]),
        ))

        return {
            "available":    True,
            "horizon_days": HORIZON,
            "signals":      signals,
            "computed_at":  time.time(),
        }


_CACHE = _Cache()


def build_panel() -> Dict:
    return _CACHE.get()
