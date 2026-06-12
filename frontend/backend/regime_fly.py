"""Regime-aware butterfly (M3-M6-M9) signal.

Architecture (follows the note):

  1. Classify the curve into one of 5 regimes by the front-vs-12M slope:
        Steep Backwardation  (slope < -4 $/bbl)
        Backwardation        (-4 ≤ slope < -1)
        Flat                 (-1 ≤ slope ≤ +1)
        Contango             (+1 < slope ≤ +4)
        Steep Contango       (slope > +4)

  2. Use LogisticRegression on (slope, curvature, vol-of-fly) to produce
     SMOOTH regime probabilities (not just a hard rule). The hard rule
     supplies the training labels; the LR provides graded confidence so a
     borderline case ("almost steep contango") is visible as such.

  3. For each regime with ≥20 historical observations, fit a Lasso
     regression that predicts the *next-period* fly z-score from a set of
     factors (BB position, 3-2-1 crack, DXY 5-day return, WTI 5-day return,
     inventory deviation, current fly value). Lasso's L1 penalty zeros out
     factors that don't matter for that regime — the surviving non-zero
     weights "tell us where to plug in backwardation vs contango" exactly
     as the note describes.

  4. At prediction time, classify the current regime, look up its Lasso
     model, compute the regression in real time, and combine with the
     current fly z-score to produce a recommendation:
        - high current fly + Lasso predicts decrease → SHORT fly
        - low current fly + Lasso predicts increase → LONG fly
        - else → WATCH

  Refits at most once per `REFIT_EVERY_SECONDS`; results cached otherwise."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    from sklearn.linear_model import Lasso, LogisticRegression, HuberRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    _SKLEARN = True
except Exception:
    _SKLEARN = False


REGIMES: List[str] = [
    "Steep Backwardation",
    "Backwardation",
    "Flat",
    "Contango",
    "Steep Contango",
]

FACTOR_NAMES: List[str] = [
    "fly_value",      # current butterfly $ value
    "wti_5d_return",  # 5-day WTI front-month return %
    "wti_20d_return", # 20-day WTI return %
    "dxy_5d_return",  # 5-day dollar-index move %
    "bb_position",    # WTI Bollinger position 0-100
    "crack_321",      # 3-2-1 refining margin $/bbl
    "inv_zscore",     # crude inventory z-score vs trailing 60d
]

REFIT_EVERY_SECONDS = 300   # refit Lasso models at most every 5 min
MIN_PER_REGIME = 20          # minimum samples needed to fit a regime model
LASSO_ALPHA = 0.05           # L1 penalty — moderate sparsity


def classify_slope(slope: float) -> Tuple[int, str]:
    """Hard-rule regime from M12-M1 slope. Returns (index 0-4, label)."""
    if slope < -4.0:  return 0, REGIMES[0]
    if slope < -1.0:  return 1, REGIMES[1]
    if slope <= 1.0:  return 2, REGIMES[2]
    if slope <= 4.0:  return 3, REGIMES[3]
    return 4, REGIMES[4]


def _safe_return(series: List[float], lag: int) -> float:
    """Percentage return over `lag` periods. 0.0 if series too short."""
    if not series or len(series) <= lag:
        return 0.0
    prev = series[-(lag + 1)]
    cur  = series[-1]
    if prev == 0:
        return 0.0
    return (cur - prev) / prev * 100.0


def _zscore_vs_trailing(series: List[float], window: int = 60) -> float:
    if not series or len(series) < window:
        return 0.0
    arr = series[-window:]
    mean = sum(arr) / len(arr)
    var = sum((x - mean) ** 2 for x in arr) / len(arr)
    sd = var ** 0.5
    if sd == 0:
        return 0.0
    return (series[-1] - mean) / sd


def _fly_from_curve(curve: List[float]) -> Optional[float]:
    """M3-2*M6+M9 from a 12-month curve (curve[0]=M1)."""
    if not curve or len(curve) < 9:
        return None
    return curve[2] - 2 * curve[5] + curve[8]


def _slope_from_curve(curve: List[float]) -> Optional[float]:
    if not curve or len(curve) < 12:
        return None
    return curve[11] - curve[0]


def _build_historical_dataset(market) -> Tuple[List[List[float]], List[int], List[float]]:
    """Walk curve_hist + price/dxy/etc histories to build per-step
    (features, regime_label, forward_fly) tuples.

    Returns (X, y_regime, y_fly_next) — X is a list of feature vectors
    aligned with regime labels and next-step fly values."""
    if not _SKLEARN:
        return [], [], []
    curves = list(market.curve_hist)
    wti = list(market.hist.get("wti", []))
    dxy = list(market.hist.get("dxy", []))
    inv = list(market.hist.get("crude_inventory", []))
    n_curves = len(curves)
    if n_curves < 25:
        return [], [], []

    # Align curves to the END of the price series — curve_hist grows in
    # lock-step with the WTI series, so the last curve corresponds to the
    # last WTI tick. Step backward i positions for the i-th-from-end curve.
    n_wti = len(wti)
    X: List[List[float]] = []
    y_regime: List[int] = []
    y_fly_next: List[float] = []

    # Iterate over curves[5 ... n-2] so we have lookback for returns AND a
    # next-step curve for the forward target.
    for i in range(5, n_curves - 1):
        curve = curves[i]
        fly = _fly_from_curve(curve)
        if fly is None:
            continue
        next_fly = _fly_from_curve(curves[i + 1])
        if next_fly is None:
            continue

        # Map curve index i back to a WTI index. Curves are appended every
        # tick; if histories are the same length, align last-to-last.
        wti_idx = n_wti - (n_curves - i)
        if wti_idx < 21 or wti_idx >= n_wti:
            continue

        wti_slice = wti[:wti_idx + 1]
        dxy_slice = dxy[:wti_idx + 1] if dxy else []
        inv_slice = inv[:wti_idx + 1] if inv else []

        # Build feature vector identical in shape to live build_features()
        wti_5d  = _safe_return(wti_slice, 5)
        wti_20d = _safe_return(wti_slice, 20)
        dxy_5d  = _safe_return(dxy_slice, 5) if dxy_slice else 0.0
        # cheap BB position proxy from 20d rolling
        if len(wti_slice) >= 20:
            mu = sum(wti_slice[-20:]) / 20
            sd = (sum((x - mu) ** 2 for x in wti_slice[-20:]) / 20) ** 0.5
            bb = ((wti_slice[-1] - (mu - 2 * sd)) / (4 * sd) * 100) \
                if sd > 0 else 50.0
        else:
            bb = 50.0
        crack_321 = 0.0   # crack history not stored; live value only
        inv_z = _zscore_vs_trailing(inv_slice, 60) if inv_slice else 0.0

        features = [fly, wti_5d, wti_20d, dxy_5d, bb, crack_321, inv_z]
        slope = _slope_from_curve(curve)
        if slope is None:
            continue
        regime_idx, _ = classify_slope(slope)
        X.append(features)
        y_regime.append(regime_idx)
        y_fly_next.append(next_fly)

    return X, y_regime, y_fly_next


def build_live_features(market, fly_z: float = 0.0,
                        crack_321: float = 0.0) -> List[float]:
    """Build the feature vector at the CURRENT tick. Order must match
    FACTOR_NAMES exactly."""
    wti = list(market.hist.get("wti", []))
    dxy = list(market.hist.get("dxy", []))
    inv = list(market.hist.get("crude_inventory", []))
    fly = market.butterfly_value() or 0.0

    wti_5d  = _safe_return(wti, 5)
    wti_20d = _safe_return(wti, 20)
    dxy_5d  = _safe_return(dxy, 5) if dxy else 0.0
    if len(wti) >= 20:
        mu = sum(wti[-20:]) / 20
        sd = (sum((x - mu) ** 2 for x in wti[-20:]) / 20) ** 0.5
        bb = ((wti[-1] - (mu - 2 * sd)) / (4 * sd) * 100) if sd > 0 else 50.0
    else:
        bb = 50.0
    inv_z = _zscore_vs_trailing(inv, 60) if inv else 0.0
    return [fly, wti_5d, wti_20d, dxy_5d, bb, crack_321, inv_z]


class RegimeFlyModel:
    """Holds the fitted per-regime Lasso coefficients + logistic-regression
    classifier. Refits at most once per REFIT_EVERY_SECONDS."""

    def __init__(self) -> None:
        self.last_fit_ts: float = 0.0
        # regime_idx -> (coef_list, intercept, n_samples)
        # regime_idx -> (coef_list, intercept, n_samples, model_name).
        # model_name is whichever of {Lasso, Huber} won that regime's
        # in-sample R² contest at fit time.
        self.lasso_per_regime: Dict[int, Tuple[List[float], float, int, str]] = {}
        # LR for soft regime probability — features: [slope, fly, fly_vol]
        self.lr_coef: Optional[List[List[float]]] = None
        self.lr_intercept: Optional[List[float]] = None
        self.scaler_mean: Optional[List[float]] = None
        self.scaler_scale: Optional[List[float]] = None
        self.regime_sample_counts: Dict[int, int] = {}
        self.n_total: int = 0

    # ---------- fitting --------------------------------------------------
    def maybe_refit(self, market) -> None:
        if not _SKLEARN:
            return
        now = time.time()
        # Only skip if BOTH Lasso AND LR are already fitted — otherwise retry
        # so LR can catch up after a transient failure.
        if now - self.last_fit_ts < REFIT_EVERY_SECONDS \
                and self.lasso_per_regime and self.lr_coef:
            return
        X, y_regime, y_fly_next = _build_historical_dataset(market)
        self.n_total = len(X)
        if not X:
            return

        Xn = np.asarray(X, dtype=float)
        y_reg = np.asarray(y_regime, dtype=int)
        y_fly = np.asarray(y_fly_next, dtype=float)

        # Per-regime sample counts
        counts: Dict[int, int] = {}
        for r in y_reg:
            counts[int(r)] = counts.get(int(r), 0) + 1
        self.regime_sample_counts = counts

        # Scale features once for the whole panel — Lasso needs scaled inputs
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xn)
        self.scaler_mean  = scaler.mean_.tolist()
        self.scaler_scale = scaler.scale_.tolist()

        # Per regime: fit BOTH Lasso (L1, sparse) AND Huber (robust to
        # fat-tailed residuals) and keep whichever wins on in-sample R².
        # Spread/fly residuals are well-known to be fat-tailed (rolls,
        # refinery shocks, weather), so Huber beats Lasso on most
        # regimes — but Lasso wins when the residuals are clean and the
        # task is really feature selection. We let each regime pick.
        new_models: Dict[int, Tuple[List[float], float, int, str]] = {}
        for r in range(len(REGIMES)):
            mask = (y_reg == r)
            n_r = int(mask.sum())
            if n_r < MIN_PER_REGIME:
                continue
            X_r, y_r = Xs[mask], y_fly[mask]
            best: Optional[Tuple[List[float], float, str, float]] = None
            # Try Lasso
            try:
                lasso = Lasso(alpha=LASSO_ALPHA, max_iter=5000)
                lasso.fit(X_r, y_r)
                r2 = float(r2_score(y_r, lasso.predict(X_r)))
                best = (lasso.coef_.tolist(),
                        float(lasso.intercept_), "Lasso", r2)
            except Exception:
                pass
            # Try Huber - epsilon=1.35 is the textbook default, alpha=0.0
            # avoids fighting Lasso for sparsity duty.
            try:
                huber = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=200)
                huber.fit(X_r, y_r)
                r2 = float(r2_score(y_r, huber.predict(X_r)))
                if best is None or r2 > best[3]:
                    best = (huber.coef_.tolist(),
                            float(huber.intercept_), "Huber", r2)
            except Exception:
                pass
            if best is not None:
                new_models[r] = (best[0], best[1], n_r, best[2])
        self.lasso_per_regime = new_models

        # Logistic regression: classify regime from compact summary stats
        # (slope inferred from feature pattern). Use fly + return features.
        # NOTE: previous version passed `multi_class="auto"` which is removed
        # in scikit-learn ≥1.7. Default behavior IS multinomial when n_classes
        # > 2 so the parameter is unnecessary.
        self.lr_error = None
        try:
            lr = LogisticRegression(max_iter=1000)
            # Need at least 2 distinct classes
            if len(set(y_reg.tolist())) >= 2:
                lr.fit(Xs, y_reg)
                self.lr_coef = lr.coef_.tolist()
                self.lr_intercept = lr.intercept_.tolist()
            else:
                self.lr_error = "only one regime class in history"
        except Exception as ex:
            self.lr_coef = None
            self.lr_intercept = None
            self.lr_error = f"{type(ex).__name__}: {ex}"

        self.last_fit_ts = now

    # ---------- prediction -----------------------------------------------
    def _scale(self, feats: List[float]) -> List[float]:
        if not self.scaler_mean or not self.scaler_scale:
            return feats
        out = []
        for i, f in enumerate(feats):
            mu = self.scaler_mean[i]
            sc = self.scaler_scale[i] or 1.0
            out.append((f - mu) / sc)
        return out

    def predict(self, market, regime_idx: int, fly_z: float,
                crack_321: float) -> Dict:
        """Return the regime-aware prediction dict for the snapshot."""
        feats = build_live_features(market, fly_z=fly_z, crack_321=crack_321)
        feats_named = list(zip(FACTOR_NAMES, feats))

        model_entry = self.lasso_per_regime.get(regime_idx)
        predicted_fly = None
        weights_sparse: List[Dict] = []
        winning_model: Optional[str] = None
        if model_entry is not None:
            coef, intercept, _n_r, winning_model = model_entry
            xs = self._scale(feats)
            predicted_fly = intercept + sum(c * x for c, x in zip(coef, xs))
            # Lasso produces sparse coefficients; Huber produces dense.
            # In both cases sort by absolute weight; for Huber we keep all
            # coefficients (no threshold), for Lasso we drop sub-µ noise.
            threshold = 1e-6 if winning_model == "Lasso" else 0.0
            named_coef = sorted(
                [(n, c) for n, c in zip(FACTOR_NAMES, coef) if abs(c) > threshold],
                key=lambda nc: abs(nc[1]), reverse=True,
            )
            weights_sparse = [{"factor": n, "weight": round(c, 4)}
                              for n, c in named_coef]

        # Soft regime probabilities from LR (if fitted)
        regime_probs: Optional[List[float]] = None
        if self.lr_coef and self.lr_intercept:
            xs = self._scale(feats)
            logits = [self.lr_intercept[r] +
                      sum(c * x for c, x in zip(self.lr_coef[r], xs))
                      for r in range(len(self.lr_coef))]
            # Softmax
            m = max(logits)
            exps = [pow(2.71828, lg - m) for lg in logits]
            tot = sum(exps) or 1.0
            regime_probs = [round(e / tot, 3) for e in exps]
            # Pad / truncate to REGIMES length if LR saw fewer classes
            if len(regime_probs) < len(REGIMES):
                regime_probs = (regime_probs +
                                [0.0] * (len(REGIMES) - len(regime_probs)))

        # Trade direction logic
        current_fly = market.butterfly_value() or 0.0
        action = "WATCH"
        rationale = ""
        if model_entry is not None and predicted_fly is not None:
            delta_predicted = predicted_fly - current_fly
            # If we expect a meaningful decrease in fly AND we're already
            # at an elevated z, mean-reversion → SHORT fly. Symmetric for long.
            mdl = winning_model or "Lasso"
            if fly_z >= 1.0 and delta_predicted < -0.1:
                action = "SHORT FLY"
                rationale = (f"Regime '{REGIMES[regime_idx]}' {mdl} predicts "
                             f"fly to decline {-delta_predicted:.2f} $/bbl; "
                             f"current z={fly_z:+.1f}σ supports mean-reversion sell.")
            elif fly_z <= -1.0 and delta_predicted > 0.1:
                action = "LONG FLY"
                rationale = (f"Regime '{REGIMES[regime_idx]}' {mdl} predicts "
                             f"fly to rise {delta_predicted:.2f} $/bbl; "
                             f"current z={fly_z:+.1f}σ supports mean-reversion buy.")
            else:
                rationale = (f"Regime '{REGIMES[regime_idx]}', "
                             f"fly z={fly_z:+.1f}σ, predicted change "
                             f"{delta_predicted:+.2f} — no edge.")
        elif not _SKLEARN:
            rationale = "scikit-learn not installed — using plain z-score signal."
        else:
            rationale = (f"Regime '{REGIMES[regime_idx]}' has only "
                         f"{self.regime_sample_counts.get(regime_idx, 0)} "
                         f"historical samples — need ≥{MIN_PER_REGIME} for Lasso.")

        return {
            "available":      _SKLEARN,
            "regime_idx":     regime_idx,
            "regime":         REGIMES[regime_idx],
            "regime_probs":   regime_probs,
            "regime_names":   list(REGIMES),
            "lr_error":       getattr(self, "lr_error", None),
            "current_fly":    round(current_fly, 3),
            "current_fly_z":  round(fly_z, 2),
            "predicted_fly":  round(predicted_fly, 3) if predicted_fly is not None else None,
            "weights":        weights_sparse,
            "features":       [{"name": n, "value": round(v, 3)}
                               for n, v in feats_named],
            "factor_names":   FACTOR_NAMES,
            "action":         action,
            "rationale":      rationale,
            "n_total_samples": self.n_total,
            "regime_sample_counts": {REGIMES[k]: v
                                     for k, v in self.regime_sample_counts.items()},
            "model_fitted":   model_entry is not None,
            "winning_model":  winning_model,
            "per_regime_winner": {REGIMES[k]: v[3]
                                   for k, v in self.lasso_per_regime.items()},
        }


_GLOBAL_MODEL = RegimeFlyModel()


def build_panel(market, fly_z: float, crack_321: float) -> Dict:
    """Public entrypoint. Returns the snapshot payload for the UI panel."""
    curve = []
    if market.curve_hist:
        curve = list(market.curve_hist[-1])
    elif market.real_curve:
        curve = [row.get("price", 0.0) for row in market.real_curve]
    slope = _slope_from_curve(curve) if curve else 0.0
    if slope is None:
        slope = 0.0
    regime_idx, regime_name = classify_slope(slope)

    _GLOBAL_MODEL.maybe_refit(market)
    panel = _GLOBAL_MODEL.predict(market, regime_idx, fly_z, crack_321)
    panel["curve_slope_m12_m1"] = round(slope, 2)
    return panel
