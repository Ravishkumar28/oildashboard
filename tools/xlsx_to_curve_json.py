"""Convert the 5 xlsx curve files to a compact JSON snapshot.

For each product, exports:
  - last_date: ISO date of the most recent curve
  - curve: list of {month, price, contract} for M1..M14
  - history: list of {date, prices[m1..m12]} for the LAST 250 days
              (used to compute slope / curvature trends without pandas at runtime)

Output: backend/data/real_curves.json   (loaded by backend/real_curves.py)

Run this whenever new xlsx data lands.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = {
    "CL":   ROOT / "CL_data_trimmed_daily_close.xlsx",
    "LCO":  ROOT / "LCO_data_trimmed_daily_close.xlsx",
    "LGO":  ROOT / "LGO_data_trimmed_daily_close.xlsx",
    "HO":   ROOT / "HO_data_trimmed_daily_close.xlsx",
    "WTCL": ROOT / "wtcl_lco_outrights_1min_trimmed_daily_close.xlsx",
}
NUM_TENORS = 12
HIST_DAYS  = 250

OUT_PATH = ROOT / "backend" / "data" / "real_curves.json"


def export_one(path: Path) -> dict:
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    last_idx = len(df) - 1
    last_row = df.iloc[last_idx]

    curve = []
    for i in range(1, NUM_TENORS + 1):
        price_col = f"c{i}||weighted_mid"
        ct_col    = f"c{i}||contract"
        if price_col not in df.columns:
            continue
        price = last_row[price_col]
        contract = last_row.get(ct_col, "")
        if pd.isna(price):
            continue
        curve.append({
            "month":    i,
            "price":    float(round(price, 4)),
            "contract": str(contract) if contract and not pd.isna(contract) else "",
        })

    history = []
    start = max(0, last_idx - HIST_DAYS + 1)
    cols_avail = [f"c{i}||weighted_mid" for i in range(1, NUM_TENORS + 1)
                  if f"c{i}||weighted_mid" in df.columns]
    for idx in range(start, last_idx + 1):
        row = df.iloc[idx]
        prices = []
        for col in cols_avail:
            v = row[col]
            prices.append(float(round(v, 4)) if not pd.isna(v) else None)
        history.append({
            "date":   row["date"].strftime("%Y-%m-%d"),
            "prices": prices,
        })

    return {
        "last_date": last_row["date"].strftime("%Y-%m-%d"),
        "curve":     curve,
        "history":   history,
        "n_history": len(history),
    }


def main():
    out = {}
    for prod, path in PRODUCTS.items():
        if not path.exists():
            print(f"  {prod}: MISSING ({path.name}) — skipping")
            continue
        out[prod] = export_one(path)
        print(f"  {prod}: last={out[prod]['last_date']}  "
              f"tenors={len(out[prod]['curve'])}  history_days={out[prod]['n_history']}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    sz = OUT_PATH.stat().st_size
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)}  ({sz:,} bytes)")


if __name__ == "__main__":
    main()
