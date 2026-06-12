"""Multi-model regression comparison per regime.

For the current curve regime (Steep Backwardation / Backwardation / Flat /
Contango / Steep Contango), fit 4 different regression families on
historical (features → next-tick fly):

  * LinearRegression — baseline, no regularization
  * Ridge(α=1.0)     — L2 regularization, dense coefficients, robust to collinearity
  * Lasso(α=0.05)    — L1 regularization, SPARSE coefficients (drops features)
  * ElasticNet(α=0.05, l1_ratio=0.5) — blend of L1 + L2

Report for each:
  * Train R² (in-sample fit quality)
  * Test R²  (out-of-sample on the most recent 20% of regime samples)
  * MAE      (mean absolute error on test split)
  * Coefs    (for sparsity comparison)

Plus "winner per regime" = highest test R². The point isn't to find one
universal model; it's to surface which family wins WHERE — Lasso often
beats Linear in low-sample regimes, Ridge often wins when features collinear.

Refits at most every REFIT_EVERY_SEC. Results cached otherwise.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

try:
    import numpy as np
    from sklearn.linear_model import (LinearRegression, Lasso, Ridge,
                                       ElasticNet, HuberRegressor)
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error
    _SKLEARN = True
except Exception:
    _SKLEARN = False

# LightGBM is optional. If the package isn't installed, the panel just runs
# without the LGBM column rather than crashing.
_LGBM_ERR = ""
try:
    from lightgbm import LGBMRegressor
    _LGBM = True
except Exception as _e:
    LGBMRegressor = None       # type: ignore
    _LGBM = False
    _LGBM_ERR = f"{type(_e).__name__}: {_e}"


REFIT_EVERY_SEC = 300
MIN_PER_REGIME = 5     # lowered from 12 so rare regimes (Contango / Steep
                       # Contango on WTI in 2025-26) surface a model when
                       # they occur — even briefly. Fits with n<8 are noisy
                       # and the panel shows them with a low-conf flag.

FACTOR_NAMES = [
    "fly_now", "wti_5d_ret", "wti_20d_ret",
    "dxy_5d_ret", "bb_pos", "inv_z",
    "vol_z", "slope",
]

REGIME_NAMES = ["Steep Backwardation", "Backwardation", "Flat",
                "Contango", "Steep Contango"]


def _classify_slope(slope: float) -> int:
    if slope < -4: return 0
    if slope < -1: return 1
    if slope <= 1: return 2
    if slope <= 4: return 3
    return 4


def _pct_return(series: List[float], lag: int) -> float:
    if not series or len(series) <= lag:
        return 0.0
    prev = series[-(lag + 1)]
    if prev == 0:
        return 0.0
    return (series[-1] - prev) / prev * 100.0


def _bb_pos(series: List[float]) -> float:
    if len(series) < 20:
        return 50.0
    tail = series[-20:]
    mu = sum(tail) / 20
    sd = (sum((x - mu) ** 2 for x in tail) / 20) ** 0.5
    if sd == 0:
        return 50.0
    return max(0.0, min(100.0, (series[-1] - (mu - 2 * sd)) / (4 * sd) * 100))


def _z(series: List[float], window: int = 60) -> float:
    if not series or len(series) < 5:
        return 0.0
    tail = series[-window:]
    mu = sum(tail) / len(tail)
    var = sum((x - mu) ** 2 for x in tail) / len(tail)
    sd = var ** 0.5
    if sd == 0:
        return 0.0
    return (series[-1] - mu) / sd


def _realized_vol_z(price_hist: List[float]) -> float:
    """Z-score of current 20d realized vol vs trailing 60d of vols."""
    import math
    if len(price_hist) < 80:
        return 0.0
    vols: List[float] = []
    for t in range(20, len(price_hist) + 1):
        rets = []
        for i in range(t - 20, t):
            if i == 0 or price_hist[i - 1] <= 0:
                continue
            rets.append(math.log(price_hist[i] / price_hist[i - 1]))
        if len(rets) < 2:
            vols.append(0.0)
            continue
        m = sum(rets) / len(rets)
        v = (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5
        vols.append(v * math.sqrt(252) * 100)
    return _z(vols, window=60)


def _build_dataset(market) -> Tuple[List[List[float]], List[float], List[int]]:
    """Build (X, y_next_fly, regime_idx_per_row) from curve history + WTI hist."""
    curves = list(market.curve_hist)
    wti = list(market.hist.get("wti", []))
    dxy = list(market.hist.get("dxy", []))
    inv = list(market.hist.get("crude_inventory", []))
    n_curves = len(curves)
    n_wti = len(wti)
    if n_curves < MIN_PER_REGIME + 5:
        return [], [], []

    X: List[List[float]] = []
    y: List[float] = []
    rg: List[int] = []

    for i in range(20, n_curves - 1):
        curve = curves[i]
        if len(curve) < 9:
            continue
        next_curve = curves[i + 1]
        if len(next_curve) < 9:
            continue
        fly_i    = curve[2] - 2 * curve[5] + curve[8]
        fly_next = next_curve[2] - 2 * next_curve[5] + next_curve[8]
        slope = curve[-1] - curve[0]
        wti_idx = n_wti - (n_curves - i) - 1
        if wti_idx < 21 or wti_idx >= n_wti:
            continue
        w_slice = wti[:wti_idx + 1]
        d_slice = dxy[:wti_idx + 1] if dxy else []
        i_slice = inv[:wti_idx + 1] if inv else []
        feats = [
            fly_i,
            _pct_return(w_slice, 5),
            _pct_return(w_slice, 20),
            _pct_return(d_slice, 5) if d_slice else 0.0,
            _bb_pos(w_slice),
            _z(i_slice, 60) if i_slice else 0.0,
            _realized_vol_z(w_slice),
            slope,
        ]
        X.append(feats)
        y.append(fly_next)
        rg.append(_classify_slope(slope))
    return X, y, rg


class _MultiModelCache:
    def __init__(self) -> None:
        self.last_fit: float = 0.0
        # regime_idx -> dict {"models": {model_name: result_dict}, "winner": str}
        self.results: Dict[int, Dict] = {}
        self.n_total: int = 0

    def maybe_refit(self, market) -> None:
        if not _SKLEARN:
            return
        now = time.time()
        if self.results and now - self.last_fit < REFIT_EVERY_SEC:
            return
        X, y, rg = _build_dataset(market)
        self.n_total = len(X)
        if not X:
            return

        Xn = np.asarray(X, dtype=float)
        yn = np.asarray(y, dtype=float)
        rg_arr = np.asarray(rg, dtype=int)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xn)

        new_results: Dict[int, Dict] = {}
        for r in range(len(REGIME_NAMES)):
            mask = (rg_arr == r)
            n_r = int(mask.sum())
            if n_r < MIN_PER_REGIME:
                continue
            X_r = Xs[mask]
            y_r = yn[mask]
            # Out-of-sample: train on first 80%, test on last 20%
            split = max(1, int(n_r * 0.8))
            X_tr, X_te = X_r[:split], X_r[split:]
            y_tr, y_te = y_r[:split], y_r[split:]

            family_models = {
                "Linear":     LinearRegression(),
                "Ridge":      Ridge(alpha=1.0),
                "Lasso":      Lasso(alpha=0.05, max_iter=5000),
                "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.5,
                                          max_iter=5000),
                # Huber — robust to fat-tailed residuals. The batch run
                # the user supplied (see backend/data/term_structure_summary.py)
                # picked Huber as the winning family on 50/188 targets
                # across the full term structure of all 5 products, more
                # than any other family. Adding it here lets the Phase 2
                # multi-model panel pick up that lift on regime-specific
                # WTI fits as well.
                "Huber":      HuberRegressor(epsilon=1.35, alpha=0.0,
                                              max_iter=200),
            }
            # LightGBM (tree-based gradient booster). Conservative settings
            # because per-regime sample counts can be as low as ~5: small
            # leaves, shallow trees, low LR. Captures non-linearities the
            # linear families cannot, so the comparison is meaningful.
            if _LGBM and len(X_tr) >= MIN_PER_REGIME:
                # LGBM hyperparameters SCALED to per-regime sample count.
                # The dataset is tiny per regime (n_tr in 19..67) with 8
                # features, so the dominant risk is overfit, not underfit.
                # Tier 1 (n<25): tiny model, no split-gain floor (it killed
                #   the Backwardation fit at 0.01 → train_r2=0).
                # Tier 2 (25-60): medium model.
                # Tier 3 (n>=60): heavy bagging + few trees + strong reg —
                #   80-tree config in r7 still overfit Flat (test_r2=-3.88).
                # bagging_fraction + bagging_freq + feature_fraction are the
                # textbook small-data regularizers for boosting.
                n_tr = len(X_tr)
                if n_tr < 25:
                    lgbm_kwargs = dict(
                        n_estimators=20, max_depth=2, num_leaves=3,
                        learning_rate=0.05, min_child_samples=2,
                        min_data_in_bin=1, min_split_gain=0.0,
                        reg_alpha=0.3, reg_lambda=0.3,
                        subsample=0.85, subsample_freq=1,
                        colsample_bytree=0.7,
                    )
                elif n_tr < 60:
                    lgbm_kwargs = dict(
                        n_estimators=40, max_depth=3, num_leaves=5,
                        learning_rate=0.05, min_child_samples=3,
                        min_data_in_bin=1, min_split_gain=0.0,
                        reg_alpha=0.5, reg_lambda=0.5,
                        subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.7,
                    )
                else:
                    lgbm_kwargs = dict(
                        n_estimators=30, max_depth=2, num_leaves=4,
                        learning_rate=0.03, min_child_samples=8,
                        min_data_in_bin=2, min_split_gain=0.0,
                        reg_alpha=1.0, reg_lambda=1.0,
                        subsample=0.7, subsample_freq=1,
                        colsample_bytree=0.6,
                    )
                family_models["LGBM"] = LGBMRegressor(
                    random_state=42,
                    verbosity=-1,
                    force_col_wise=True,
                    **lgbm_kwargs,
                )
            model_results: Dict[str, Dict] = {}
            for fam_name, model in family_models.items():
                try:
                    model.fit(X_tr, y_tr)
                    train_r2 = float(r2_score(y_tr, model.predict(X_tr)))
                    if len(X_te) >= 2:
                        y_pr = model.predict(X_te)
                        test_r2 = float(r2_score(y_te, y_pr))
                        test_mae = float(mean_absolute_error(y_te, y_pr))
                    else:
                        test_r2 = float("nan")
                        test_mae = float("nan")
                    if hasattr(model, "coef_"):
                        coef = model.coef_.tolist()
                    elif hasattr(model, "feature_importances_"):
                        # Tree models (LGBM): expose normalized split-gain
                        # importances in the coef slot so the UI's per-factor
                        # bars still render. Scaled to sum to 1.0.
                        imp = model.feature_importances_.astype(float)
                        s = float(imp.sum())
                        coef = (imp / s).tolist() if s > 0 else [0.0] * len(FACTOR_NAMES)
                    else:
                        coef = [0.0] * len(FACTOR_NAMES)
                    n_nonzero = sum(1 for c in coef if abs(c) > 1e-6)
                    # LGBM and other non-linear models don't have an intercept_
                    # attribute. Fall back to 0.0 — the panel uses intercept
                    # only as a display field, not in any math.
                    raw_int = getattr(model, "intercept_", 0.0)
                    try:
                        intercept_val = float(raw_int)
                    except (TypeError, ValueError):
                        intercept_val = 0.0
                    model_results[fam_name] = {
                        "train_r2":   round(train_r2, 3),
                        "test_r2":    None if test_r2 != test_r2
                                       else round(test_r2, 3),
                        "test_mae":   None if test_mae != test_mae
                                       else round(test_mae, 4),
                        "intercept":  round(intercept_val, 4),
                        "coefs":      [round(c, 4) for c in coef],
                        "n_nonzero":  n_nonzero,
                    }
                except Exception as _fit_err:
                    # Surface the failure so we can see WHY a family was
                    # dropped — silent skips have bitten us with LGBM.
                    model_results[fam_name] = {
                        "fit_err": f"{type(_fit_err).__name__}: {str(_fit_err)[:200]}"
                    }

            # Pick winner: highest test_r2 (or train_r2 if test unavailable).
            # Skip entries that are error stubs (only contain "fit_err").
            winner = None
            best_score = -float("inf")
            for fam_name, info in model_results.items():
                if "fit_err" in info and "train_r2" not in info:
                    continue
                score = info.get("test_r2")
                if score is None:
                    score = info.get("train_r2", -float("inf"))
                if score is not None and score > best_score:
                    best_score = score
                    winner = fam_name

            new_results[r] = {
                "n_samples":     n_r,
                "n_train":       split,
                "n_test":        n_r - split,
                "models":        model_results,
                "winner":        winner,
                "winner_score":  round(best_score, 3) if winner else None,
            }

        self.results = new_results
        self.last_fit = now


_CACHE = _MultiModelCache()


def build_panel(market, current_slope: float) -> Dict:
    """Returns the model comparison payload for the snapshot."""
    if not _SKLEARN:
        return {"available": False,
                "reason": "scikit-learn not installed."}
    _CACHE.maybe_refit(market)
    cur_regime_idx = _classify_slope(current_slope)
    return {
        "available":         True,
        "n_total_samples":   _CACHE.n_total,
        "current_regime_idx": cur_regime_idx,
        "current_regime":     REGIME_NAMES[cur_regime_idx],
        "lgbm_ok":            _LGBM,
        "lgbm_err":           _LGBM_ERR,
        "regimes":            [
            {
                "regime_idx": r,
                "regime":     REGIME_NAMES[r],
                **(_CACHE.results.get(r) or {"n_samples": 0,
                                              "models": {},
                                              "winner": None}),
            } for r in range(len(REGIME_NAMES))
        ],
        "factor_names":       list(FACTOR_NAMES),
    }
