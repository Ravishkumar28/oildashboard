"""Real-data signal engine — surfaces pre-computed signals from the user's
5 xlsx files (CL/LCO/LGO/HO/WTCL M1..M14 curves).

Static data: backend/data/real_data_signals.json
Generated offline by tools/real_data_master.py from:
  - CL_data_trimmed_daily_close.xlsx   (WTI M1..M14)
  - LCO_data_trimmed_daily_close.xlsx  (Brent M1..M17)
  - LGO_data_trimmed_daily_close.xlsx  (Gasoil M1..M14)
  - HO_data_trimmed_daily_close.xlsx   (Heating Oil M1..M14)
  - wtcl_lco_outrights_1min_trimmed_daily_close.xlsx  (WTI-Brent M1..M12)

Each signal is the prediction of the BEST out-of-sample model (chosen on
the 20% validation slice of a 60/20/20 chronological split) applied to
the latest available feature vector.  Features used (real, no synthetic):
  - lagged 1d/5d/20d returns/diffs of the target itself
  - DXY 5d % return (yfinance, real)
  - 20d realized vol of the target
  - Curve slope (M12 - M1) of the product
  - Curve curvature (M3 - 2*M6 + M9) of the product
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

_DATA_PATH    = Path(__file__).parent / "data" / "real_data_signals.json"
_VERIFY_PATH  = Path(__file__).parent / "data" / "real_data_verify.json"


def _confidence_tier(test_r2: float) -> str:
    if test_r2 >= 0.30:
        return "HIGH"
    if test_r2 >= 0.10:
        return "MED"
    if test_r2 >= 0.05:
        return "LOW"
    return "NOISE"


def build_panel() -> Dict:
    if not _DATA_PATH.exists():
        return {"available": False,
                "reason": "real_data_signals.json not found — run tools/real_data_master.py"}
    try:
        raw = json.loads(_DATA_PATH.read_text())
    except Exception as e:
        return {"available": False, "reason": f"parse error: {e}"}

    # Attach confidence tier + Working-effect haircut R²
    enriched: List[Dict] = []
    for s in raw:
        r2 = s.get("test_r2") or 0.0
        haircut_pct = s.get("we_haircut_pct") or 0
        haircut_r2 = round(r2 * (1 - haircut_pct / 100), 4)
        s2 = {**s,
              "confidence":     _confidence_tier(r2),
              "haircut_r2":     haircut_r2,
              "we_lag1":        s.get("we_lag1"),
              "we_haircut_pct": haircut_pct}
        enriched.append(s2)

    # Sort: actionable non-FLAT first, then by test R^2 desc; FLAT after.
    rank = {"HIGH": 0, "MED": 1, "LOW": 2, "NOISE": 3}
    def _key(s):
        return (
            0 if s["signal"] != "FLAT" else 1,
            rank.get(s["confidence"], 9),
            -(s["test_r2"] or 0),
        )
    enriched.sort(key=_key)

    # Verification check
    verify = {}
    if _VERIFY_PATH.exists():
        try:
            verify = json.loads(_VERIFY_PATH.read_text())
        except Exception:
            verify = {"available": False}

    counts = {"HIGH": 0, "MED": 0, "LOW": 0, "NOISE": 0}
    actionable = 0
    for s in enriched:
        counts[s["confidence"]] += 1
        if s["confidence"] in ("HIGH", "MED") and s["signal"] != "FLAT":
            actionable += 1

    return {
        "available":  True,
        "total":      len(enriched),
        "actionable": actionable,
        "by_conf":    counts,
        "signals":    enriched,
        "verify": {
            "checked":   verify.get("checked"),
            "matched":   verify.get("matched"),
            "match_pct": verify.get("match_pct"),
        },
    }
