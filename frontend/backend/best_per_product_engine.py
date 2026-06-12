"""Best-per-product trade engine.

For each product, we identified the model family that wins the most targets
in the 138-target 60/20/20 backtest. This engine filters real_data_signals
to ONLY surface signals where the per-product best model was the winner —
so the trade decisions on each product come from the model that's actually
been validated as the best fit for that product's signal structure.

Per-product winners (from backend/data/per_product_winners.json):
    CL   (WTI):       XGBoost     — 7/25 wins
    HO   (Heating):   Huber       — 7/26 wins
    LCO  (Brent):     LGBM        — 13/31 wins (dominant)
    LGO  (Gasoil):    SVR_Poly    — 13/33 wins (dominant)
    WTCL (spread):    SVR_Linear  — 6/23 wins

Filters applied:
  * winner_model == per-product designated winner
  * test_r2 >= 0.10 (MED or HIGH confidence)
  * signal != FLAT (must be actionable LONG or SHORT)

Output is the per-product top-3 (by test R²) actionable trade ideas.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

_SIGNALS_PATH = Path(__file__).parent / "data" / "real_data_signals.json"
_WINNERS_PATH = Path(__file__).parent / "data" / "per_product_winners.json"

# These confidence-tier thresholds match the dashboard's REAL panel:
HIGH_R2 = 0.30
MED_R2  = 0.10


def _load_signals() -> List[Dict]:
    if not _SIGNALS_PATH.exists():
        return []
    try:
        return json.loads(_SIGNALS_PATH.read_text())
    except Exception:
        return []


def _load_winners() -> Dict[str, Dict]:
    if not _WINNERS_PATH.exists():
        return {}
    try:
        return json.loads(_WINNERS_PATH.read_text())
    except Exception:
        return {}


def _confidence(r2: float) -> str:
    if r2 >= HIGH_R2:
        return "HIGH"
    if r2 >= MED_R2:
        return "MED"
    return "LOW"


def build_panel() -> Dict:
    signals = _load_signals()
    winners = _load_winners()
    if not signals or not winners:
        return {"available": False,
                "reason": "missing signals or per-product winners file"}

    by_product: Dict[str, Dict] = {}
    actionable_total = 0

    for prod, win_info in winners.items():
        best_model = win_info.get("model")
        if not best_model:
            continue
        # All signals for this product where the per-product best model won
        prod_signals = [
            s for s in signals
            if s.get("product") == prod
            and s.get("winner_model") == best_model
            and (s.get("test_r2") or 0) >= MED_R2
            and s.get("signal") in ("LONG", "SHORT")
        ]
        prod_signals.sort(key=lambda s: -(s.get("test_r2") or 0))
        # Top 3 per product (caps risk concentration)
        top = prod_signals[:3]
        for s in top:
            r2 = s.get("test_r2") or 0
            s["confidence"] = _confidence(r2)
            haircut_pct = s.get("we_haircut_pct") or 0
            s["haircut_r2"] = round(r2 * (1 - haircut_pct / 100), 4)
        actionable_total += len(top)
        by_product[prod] = {
            "best_model":      best_model,
            "wins_by_model":   win_info.get("n_wins_by_this_model"),
            "total_targets":   win_info.get("n_total_targets"),
            "mean_r2_overall": win_info.get("mean_test_r2"),
            "n_actionable":    len(top),
            "trades":          top,
        }

    return {
        "available":  True,
        "n_total":    actionable_total,
        "per_product": by_product,
        "explainer":  ("For each product, the model with the most wins in "
                       "the 138-target 60/20/20 backtest is the ONLY model "
                       "we trade on that product. Top-3 highest-R² actionable "
                       "signals per product."),
    }
