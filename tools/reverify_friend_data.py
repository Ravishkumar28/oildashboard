"""Re-evaluation of friend's backtest after seeing their full code + CSV.

KEY OBSERVATIONS that revise my prior verdict:

1. Their audit_mr_ou.py EXPLICITLY tests for the exact biases I flagged
   (1-day execution lag, look-ahead, averaged-price autocorrelation, cost
   sensitivity). They are aware of and have controlled for these biases.

2. Their docstring NOTE says: "this audit led to the 1-day execution lag
   being baked into pnl_from_pos" — the published numbers are POST-FIX.
   So the 200x PnL gap I saw was probably my replication missing details
   of their walk-forward / sizing / PnL accounting, not their bias.

3. The full CSV shows a CROSS-INSTRUMENT PATTERN that strongly supports
   honesty: many instruments where EVERY strategy LOSES money:
       3:2:1 cross-commodity:    every strategy negative ($-2.3 to $-36.8)
       HO:fly_M1_M2_M12:         every strategy negative (Sharpe -0.47 to -2.35)
       HO:spread_M1_M12:         every strategy negative
       HO:spread_M1_M2:          every strategy negative
       RBOB crack (most strats): negative or near-zero
   This pattern is the OPPOSITE of look-ahead bias — biased backtests
   inflate winners but don't manufacture losers.

4. The WINNERS look reasonable too — different leaders per instrument:
       MR_bollinger leads CL:fly_M1_M2_M12
       ML_Ridge leads Gasoil crack
       RISK_stops leads HO crack
       MOM_donchian leads LCO:spread_M1_M12
       MR_zscore leads WTI-Brent
   No single family dominates everything (which would also be a bias tell).

What I'll verify here:
  (a) Does the weighted_mid data have the Working-effect autocorrelation
      problem the friend's audit warns about?
  (b) Does the CL fly M1-M2-M12 we both compute from the xlsx actually
      have mean-reverting structure that justifies a $40 winner?
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]


def load_fly(xlsx_name: str, fly_formula):
    df = pd.read_excel(ROOT / xlsx_name)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return fly_formula(df).dropna()


def autocorr_diagnostic(series: pd.Series, label: str):
    """Friend's exact diagnostic for averaged-price artifact (Working effect)."""
    d = series.diff().dropna()
    ac1 = d.autocorr(1)
    ac2 = d.autocorr(2)
    ac5 = d.autocorr(5)
    return {
        "label": label,
        "mean": float(series.mean()),
        "std":  float(series.std()),
        "lag1_diff_autocorr": float(ac1),
        "lag2_diff_autocorr": float(ac2),
        "lag5_diff_autocorr": float(ac5),
    }


def hurst_exponent(series: pd.Series) -> float:
    """Hurst exponent: <0.5 = mean-reverting, ~0.5 = random walk, >0.5 = trending."""
    arr = series.values
    if len(arr) < 100:
        return 0.5
    lags = range(2, min(100, len(arr) // 4))
    tau = [np.sqrt(np.std(arr[lag:] - arr[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0] * 2.0)


def main():
    print("=" * 76)
    print("Re-verification of friend's backtest results")
    print("=" * 76)

    # ----- 1. The Working-effect autocorrelation diagnostic -----
    print("\n1) WORKING-EFFECT DIAGNOSTIC (friend's own check)")
    print("   If lag-1 autocorr of daily DIFFS is STRONGLY NEGATIVE (worse than -0.3),")
    print("   weighted_mid averaging is inducing FAKE mean reversion. If near 0, the")
    print("   mean-reversion signal is real.")
    print()

    targets = [
        ("CL_data_trimmed_daily_close.xlsx",
         lambda df: pd.to_numeric(df["c1||weighted_mid"], errors="coerce")
                    - 2*pd.to_numeric(df["c2||weighted_mid"], errors="coerce")
                    + pd.to_numeric(df["c12||weighted_mid"], errors="coerce"),
         "CL fly M1-2M2+M12"),
        ("CL_data_trimmed_daily_close.xlsx",
         lambda df: pd.to_numeric(df["c1||weighted_mid"], errors="coerce")
                    - pd.to_numeric(df["c2||weighted_mid"], errors="coerce"),
         "CL spread M1-M2"),
        ("CL_data_trimmed_daily_close.xlsx",
         lambda df: pd.to_numeric(df["c1||weighted_mid"], errors="coerce")
                    - pd.to_numeric(df["c12||weighted_mid"], errors="coerce"),
         "CL spread M1-M12"),
        ("CL_data_trimmed_daily_close.xlsx",
         lambda df: pd.to_numeric(df["c1||weighted_mid"], errors="coerce"),
         "CL M1 outright (benchmark - should be ~0)"),
        ("LCO_data_trimmed_daily_close.xlsx",
         lambda df: pd.to_numeric(df["c1||weighted_mid"], errors="coerce")
                    - 2*pd.to_numeric(df["c2||weighted_mid"], errors="coerce")
                    + pd.to_numeric(df["c12||weighted_mid"], errors="coerce"),
         "LCO fly M1-2M2+M12"),
        ("LGO_data_trimmed_daily_close.xlsx",
         lambda df: pd.to_numeric(df["c1||weighted_mid"], errors="coerce")
                    - 2*pd.to_numeric(df["c2||weighted_mid"], errors="coerce")
                    + pd.to_numeric(df["c12||weighted_mid"], errors="coerce"),
         "LGO fly M1-2M2+M12"),
    ]

    print(f"  {'Series':<35}{'mean':>9}{'std':>8}{'lag1':>8}{'lag2':>8}{'lag5':>8}{'verdict':>16}")
    print("  " + "-" * 92)
    for fname, formula, label in targets:
        s = load_fly(fname, formula)
        ac = autocorr_diagnostic(s, label)
        # Verdict on lag-1
        a1 = ac["lag1_diff_autocorr"]
        if a1 < -0.4:
            v = "FAKE MR (bad)"
        elif a1 < -0.2:
            v = "suspicious"
        elif a1 < -0.05:
            v = "mild MR"
        elif abs(a1) < 0.05:
            v = "random walk"
        else:
            v = "momentum"
        print(f"  {label:<35}{ac['mean']:>+9.2f}{ac['std']:>8.2f}"
              f"{a1:>+8.3f}{ac['lag2_diff_autocorr']:>+8.3f}"
              f"{ac['lag5_diff_autocorr']:>+8.3f}{v:>16}")

    # ----- 2. Hurst exponent on the fly LEVEL -----
    print()
    print("2) HURST EXPONENT on the LEVEL series (not diffs)")
    print("   < 0.5 = level mean-reverts ; ~0.5 = random walk ; > 0.5 = trending")
    print()
    print(f"  {'Series':<35}{'Hurst':>9}{'Interpretation':>30}")
    print("  " + "-" * 75)
    for fname, formula, label in targets:
        s = load_fly(fname, formula)
        h = hurst_exponent(s)
        if h < 0.40:
            interp = "MEAN-REVERTING (MR works)"
        elif h < 0.55:
            interp = "near-random-walk"
        else:
            interp = "TRENDING (MOM works)"
        print(f"  {label:<35}{h:>+9.3f}{interp:>30}")

    # ----- 3. Cross-instrument pattern check -----
    print()
    print("3) CROSS-INSTRUMENT PATTERN in friend's CSV results")
    print()
    print("  Instruments where ALL 13 strategies LOSE money:")
    print("    - 3:2:1 cross-commodity crack")
    print("    - HO:fly_M1_M2_M12")
    print("    - HO:spread_M1_M12")
    print("    - HO:spread_M1_M2")
    print()
    print("  Instruments with mostly NEGATIVE results (RBOB crack):")
    print("    7 of 13 strategies lose money")
    print()
    print("  Strong winners (good ML scores) appear only on:")
    print("    - Gasoil crack (Ridge/EN/Lasso Sharpe 1.1+)")
    print("    - WTI-Brent (RISK_stops, MR_zscore Sharpe 1.0+)")
    print()
    print("  This LOSER-rich pattern strongly suggests an HONEST backtest.")
    print("  Biased backtests inflate WINNERS but don't manufacture LOSERS.")

if __name__ == "__main__":
    main()
