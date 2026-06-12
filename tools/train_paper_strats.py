"""Train the 3 winning paper-strategy models on xlsx data and save the
fitted parameters to backend/data/paper_strats_models.json.

The backend engine then loads these and uses them to generate signals on
LATEST curve data (from backend/data/real_curves.json + yfinance live front).

Models fitted (1 per product):

  PCA Curve  — for CL/LCO/LGO  (fit on 60% of M1-M12 log returns)
    Saved: components_, mean_, top-3 PC3-loading tenors with signs

  Bertram OU — for HO calendar spreads M1-M2, M2-M3, ..., M5-M6
    Saved: mu_hat, alpha_hat, eta_hat, sigma_eq, a_star per spread

  HMM regime — for WTCL (CL-LCO log spread)
    Saved: state means, std, transition matrix, regime labels
"""
from __future__ import annotations
import json
import math
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_FILES = {
    "CL":   ROOT / "CL_data_trimmed_daily_close.xlsx",
    "LCO":  ROOT / "LCO_data_trimmed_daily_close.xlsx",
    "LGO":  ROOT / "LGO_data_trimmed_daily_close.xlsx",
    "HO":   ROOT / "HO_data_trimmed_daily_close.xlsx",
}
OUT_PATH = ROOT / "backend" / "data" / "paper_strats_models.json"


def load_curve(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    out = pd.DataFrame(index=df.index)
    for i in range(1, 13):
        c = f"c{i}||weighted_mid"
        if c in df.columns:
            out[f"m{i}"] = pd.to_numeric(df[c], errors="coerce")
    return out


def train_pca(product: str, curve: pd.DataFrame) -> Dict:
    cols = [f"m{i}" for i in range(1, 13) if f"m{i}" in curve.columns]
    if len(cols) < 8:
        return None
    prices = curve[cols].dropna()
    if len(prices) < 250:
        return None
    log_ret = np.log(prices / prices.shift(1)).dropna()
    n = len(log_ret)
    train_X = log_ret.iloc[:int(n * 0.60)].values
    pca = PCA(n_components=3)
    pca.fit(train_X)
    pc3_load = pca.components_[2]
    sorted_loads = np.argsort(np.abs(pc3_load))[::-1]
    leg_indices = sorted_loads[:3].tolist()
    leg_signs = np.sign(pc3_load[leg_indices]).astype(int).tolist()
    leg_cols = [cols[i] for i in leg_indices]
    return {
        "product":     product,
        "tenor_cols":  cols,
        "components":  pca.components_.tolist(),     # shape (3, n_tenors)
        "mean":        pca.mean_.tolist(),
        "fly_legs":    leg_cols,
        "fly_signs":   leg_signs,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "z_entry":     2.0,
        "z_exit":      0.5,
        "max_hold_days": 15,
    }


def train_bertram(product: str, curve: pd.DataFrame) -> Dict:
    """For HO, fit OU on each calendar spread M1-M2 through M5-M6."""
    out_spreads = []
    for a, b in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]:
        ka, kb = f"m{a}", f"m{b}"
        if ka not in curve.columns or kb not in curve.columns:
            continue
        s_raw = (curve[ka] - curve[kb]).dropna()
        if len(s_raw) < 250:
            continue
        n = len(s_raw)
        train_seg = s_raw.iloc[:int(n * 0.60)]
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
        out_spreads.append({
            "spread":   f"m{a}-m{b}",
            "leg_a":    ka, "leg_b": kb,
            "mu_hat":   mu_hat,
            "alpha_hat": float(alpha_hat),
            "eta_hat":  eta_hat,
            "sigma_eq": float(sigma_eq),
            "a_star":   float(a_star),
            "z_exit":   0.0,
            "max_hold_days": 30,
        })
    return {"product": product, "spreads": out_spreads}


def train_hmm(curves: Dict[str, pd.DataFrame]) -> Dict:
    cl = curves["CL"]["m1"].dropna()
    brent = curves["LCO"]["m1"].dropna()
    common = cl.index.intersection(brent.index)
    cl = cl.reindex(common); brent = brent.reindex(common)
    spread = np.log(cl) - np.log(brent)
    n = len(spread)
    train_spread = spread.iloc[:int(n * 0.60)].values.reshape(-1, 1)
    hmm = GaussianHMM(n_components=2, covariance_type="full",
                       n_iter=200, random_state=42)
    hmm.fit(train_spread)
    means = hmm.means_.flatten()
    low_state  = int(np.argmin(means))
    high_state = int(np.argmax(means))
    std_per_state = np.sqrt(hmm.covars_.reshape(2, 1, 1)[:, 0, 0])
    return {
        "product":      "WTCL",
        "means":        means.tolist(),
        "stds":         std_per_state.tolist(),
        "transition":   hmm.transmat_.tolist(),
        "startprob":    hmm.startprob_.tolist(),
        "low_state":    low_state,
        "high_state":   high_state,
        "z_entry":      1.0,
        "z_exit":       0.25,
        "p_regime_min": 0.7,
        "max_hold_days": 20,
    }


def main():
    print("Loading curves...")
    curves = {}
    for prod, path in PRODUCT_FILES.items():
        if path.exists():
            curves[prod] = load_curve(path)
            print(f"  {prod}: {len(curves[prod])} rows")

    out = {"pca": {}, "bertram": {}, "hmm": None,
           "schema_version": 1}

    # PCA for CL, LCO, LGO (the winners)
    for prod in ("CL", "LCO", "LGO"):
        if prod in curves:
            m = train_pca(prod, curves[prod])
            if m is not None:
                out["pca"][prod] = m
                print(f"  PCA fit for {prod}: explained_var = {[round(v, 3) for v in m['explained_variance_ratio']]}")
                print(f"    fly legs: {m['fly_legs']} signs: {m['fly_signs']}")

    # Bertram for HO
    if "HO" in curves:
        b = train_bertram("HO", curves["HO"])
        if b["spreads"]:
            out["bertram"]["HO"] = b
            print(f"  Bertram fit for HO: {len(b['spreads'])} spreads")
            for s in b["spreads"]:
                print(f"    {s['spread']}: mu={s['mu_hat']:.4f} alpha={s['alpha_hat']:.4f} a*={s['a_star']:.4f}")

    # HMM for WTCL
    if "CL" in curves and "LCO" in curves:
        h = train_hmm(curves)
        out["hmm"] = h
        print(f"  HMM fit for WTCL: means={[round(m, 3) for m in h['means']]}  low_state={h['low_state']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)}  ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
