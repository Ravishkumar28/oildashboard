"""Paper-trading engine.

Auto-executes the dashboard's two trade signals on a virtual $100k portfolio
and tracks the hypothetical P&L, win rate, and equity curve. No broker, no
real money — purely a tracker for "what would these signals have made?"

State persists to a local JSON file so the equity curve survives sleep/wake
cycles on HF Spaces. (It does reset when the Space is redeployed from new
code — HF Spaces free tier has no persistent storage volume.)"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import persistence

STATE_FILE = Path(__file__).parent / "paper_state.json"
STARTING_EQUITY = 100_000.0
TRADE_SIZE_BBL = 1_000      # barrels per virtual trade
REMOTE_SAVE_EVERY_SEC = 300   # push to HF dataset at most every 5 min

# Realistic round-trip frictional cost per 1,000-bbl contract.
# Built from typical retail futures costs:
#   * exchange / clearing / regulatory fees   ~ $1.50
#   * broker commission (round-trip)          ~ $3.00
#   * average bid-ask slippage (2 ticks x $0.01 entry+exit on 1,000 bbl)
#                                             ~ $20.00
# Total: ~$25 round-trip per contract  ->  ~3 bps on a $90,000 WTI notional.
# Applied to BOTH the realized P&L when a position closes AND the
# unrealized mark-to-market (so the equity panel reflects the true
# liquidation value, not gross paper P&L).
COMMISSION_PER_CONTRACT_RT = 25.0

# Risk controls for the regression engine (source="term").
# Without these, mean-reversion ideas that go against you accumulate losses
# indefinitely because the position only closes when the idea drops out of
# the top-N — and trending markets push the dislocation FURTHER from mean.
TERM_STOP_LOSS_PCT      = 2.5     # close at -2.5% of entry notional
TERM_MAX_HOLD_HOURS     = 72      # force-close after 3 days regardless
TERM_TAKE_PROFIT_PCT    = 4.0     # lock in +4% gains
# Same controls applied to technical-engine trades (source="tech").
TECH_STOP_LOSS_PCT      = 3.0
TECH_MAX_HOLD_HOURS     = 96
TECH_TAKE_PROFIT_PCT    = 5.0
# REAL-data signals (source="real") — winners from the 60/20/20 xlsx
# backtest. Slightly tighter stop because the entry is on validated R²
# models (we're more confident in direction so willing to be stricter
# about being wrong); same time-stop.
REAL_STOP_LOSS_PCT      = 2.0
REAL_MAX_HOLD_HOURS     = 72
REAL_TAKE_PROFIT_PCT    = 3.5
# COMPOSITE multi-factor strategy (source="composite") — fuses regression
# + LGBM + technical + news + regime + macro. Treated as the highest-
# conviction source so we let it run a bit longer (96h) before time-stop,
# but with the same tight 2% stop because the multi-engine consensus
# should be wrong less often.
COMP_STOP_LOSS_PCT      = 2.0
COMP_MAX_HOLD_HOURS     = 96
COMP_TAKE_PROFIT_PCT    = 4.0
# BEST-PER-PRODUCT (source="bestpp") — each trade uses the specialist
# model for its product. Tightest stop because the per-product model has
# the highest base rate of being right; same hold window as composite.
BPP_STOP_LOSS_PCT       = 1.8
BPP_MAX_HOLD_HOURS      = 96
BPP_TAKE_PROFIT_PCT     = 3.5
# PAPER-STRATEGIES (source="pstrat") — PCA Curve / Bertram OU / HMM Regime.
# These are the absolute winners of the strategy shootout — Sharpe 0.88 to
# 6.02 per product. Tightest risk control to match their high-conviction
# nature: 1.5% stop loss, 72h time stop, +3% take profit.
PSTRAT_STOP_LOSS_PCT    = 1.5
PSTRAT_MAX_HOLD_HOURS   = 72
PSTRAT_TAKE_PROFIT_PCT  = 3.0

# signal title -> which price drives the P&L of that trade
_SIGNAL_ASSETS: Dict[str, str] = {
    "Diesel Refining Arbitrage": "diesel_crack",      # crack spread, $/bbl
    "Crude Storage Carry":       "wti_spot",          # spot WTI, $/bbl
    "Curve Butterfly":           "butterfly_fly",     # M3-M6-M9 fly, $/bbl
    "Brent-WTI Spread":          "brent_wti_spread",  # spread, $/bbl
    "3-2-1 USGC Crack":          "crack_321",         # 3-2-1 crack, $/bbl
    "WTI-DXY Pair Trade":        "wti_spot",          # WTI is the tradeable leg
    "Kalman Pair (dyn β)":       "kalman_residual",   # WTI - (α + β*DXY), $/bbl
}


class PaperBook:
    """Virtual portfolio that opens/closes trades when signal status flips."""

    def __init__(self, remote_repo: str = "", remote_token: str = "") -> None:
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.starting_equity = STARTING_EQUITY
        self._last_save = 0.0
        self._last_remote_save = 0.0
        # HF Datasets persistence so equity survives Space redeploys
        self.remote_repo = remote_repo
        self.remote_token = remote_token
        self.remote_active = False
        # optional scheduled auto-reset (unix epoch seconds, UTC). When the
        # engine ticks past this moment, paper state is wiped back to $100k.
        self.scheduled_reset_at: Optional[float] = None
        self._load()

    def schedule_reset(self, ts_epoch: float) -> None:
        """Arrange for the paper book to be reset back to $100k at the given
        UTC moment. Used to wipe holiday-period drift trades the moment real
        market data resumes."""
        self.scheduled_reset_at = ts_epoch

    def reset(self) -> None:
        """Wipe positions, trades and equity curve; start fresh at $100k.
        Persisted immediately so the reset survives a restart."""
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.starting_equity = STARTING_EQUITY
        self._last_remote_save = 0.0   # force one immediate remote push
        self._save()

    # ---- persistence ------------------------------------------------- #
    def _apply(self, data: dict) -> None:
        self.positions = data.get("positions", {})
        self.trades = data.get("trades", [])
        self.equity_curve = data.get("equity_curve", [])
        self.starting_equity = float(data.get("starting_equity",
                                              STARTING_EQUITY))
        sched = data.get("scheduled_reset_at")
        if sched:
            self.scheduled_reset_at = float(sched)

    def _load(self) -> None:
        # Prefer remote (HF Dataset) state so the equity curve survives
        # Space redeploys; fall back to whatever's on the local volume.
        if self.remote_repo and self.remote_token:
            remote = persistence.download_state(self.remote_repo,
                                                self.remote_token)
            if remote:
                try:
                    self._apply(remote)
                    self.remote_active = True
                    return
                except Exception:
                    pass
        if STATE_FILE.exists():
            try:
                self._apply(json.loads(
                    STATE_FILE.read_text(encoding="utf-8")))
            except Exception:
                pass

    def _state_dict(self) -> dict:
        return {
            "positions": self.positions,
            "trades": self.trades[-200:],
            "equity_curve": self.equity_curve[-500:],
            "starting_equity": self.starting_equity,
            "scheduled_reset_at": self.scheduled_reset_at,
        }

    def _save(self) -> None:
        state = self._state_dict()
        try:
            STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
        except Exception:
            pass
        # throttled remote push — every save flushes locally, but the HF
        # dataset commit only happens at most every REMOTE_SAVE_EVERY_SEC
        if not (self.remote_repo and self.remote_token):
            return
        now = time.time()
        if now - self._last_remote_save < REMOTE_SAVE_EVERY_SEC:
            return
        if persistence.upload_state(self.remote_repo,
                                    self.remote_token, state):
            self._last_remote_save = now
            self.remote_active = True

    # ---- core update ------------------------------------------------- #
    def _check_scheduled_reset(self, now: float) -> None:
        """Trigger the scheduled reset once the current time crosses it."""
        if self.scheduled_reset_at and now >= self.scheduled_reset_at:
            self.reset()
            self.scheduled_reset_at = None

    def update(self, signals: List[Dict], assets: Dict[str, float],
               live: bool = True) -> None:
        """Called every market tick. Opens a virtual trade when a signal
        flips WATCHING -> ACTIVE, closes it when it flips back.

        ``live=False`` skips opening new positions (used during the startup
        warmup so the engine doesn't trade on simulated bootstrap values
        before real-price anchors have stabilized). Existing positions can
        still be closed even during warmup."""
        now = time.time()
        self._check_scheduled_reset(now)

        # ---- RISK CHECKS on open positions ---------------------------------
        # Stop-loss + time-stop + take-profit. Applied BEFORE signal processing
        # so we close losers before any new signals reopen the same trade.
        to_close: List[tuple] = []
        for title, pos in list(self.positions.items()):
            src = pos.get("source", "")
            if src == "term":
                stop_pct, hold_hr, tp_pct = TERM_STOP_LOSS_PCT, TERM_MAX_HOLD_HOURS, TERM_TAKE_PROFIT_PCT
            elif src == "tech":
                stop_pct, hold_hr, tp_pct = TECH_STOP_LOSS_PCT, TECH_MAX_HOLD_HOURS, TECH_TAKE_PROFIT_PCT
            elif src == "real":
                stop_pct, hold_hr, tp_pct = REAL_STOP_LOSS_PCT, REAL_MAX_HOLD_HOURS, REAL_TAKE_PROFIT_PCT
            elif src == "composite":
                stop_pct, hold_hr, tp_pct = COMP_STOP_LOSS_PCT, COMP_MAX_HOLD_HOURS, COMP_TAKE_PROFIT_PCT
            elif src == "bestpp":
                stop_pct, hold_hr, tp_pct = BPP_STOP_LOSS_PCT, BPP_MAX_HOLD_HOURS, BPP_TAKE_PROFIT_PCT
            elif src == "pstrat":
                stop_pct, hold_hr, tp_pct = PSTRAT_STOP_LOSS_PCT, PSTRAT_MAX_HOLD_HOURS, PSTRAT_TAKE_PROFIT_PCT
            else:
                continue  # leave manual / unknown sources alone
            price = assets.get(pos["asset_key"])
            if price is None:
                continue
            entry = pos["entry_price"]
            if entry == 0:
                continue
            dir_sign = 1 if pos["direction"] == "LONG" else -1
            pnl_pct = (float(price) - entry) / abs(entry) * 100 * dir_sign
            age_hr = (now - pos.get("entry_ts", now)) / 3600.0
            reason = None
            if pnl_pct <= -stop_pct:
                reason = f"STOP_LOSS ({pnl_pct:.2f}%)"
            elif pnl_pct >= tp_pct:
                reason = f"TAKE_PROFIT (+{pnl_pct:.2f}%)"
            elif age_hr >= hold_hr:
                reason = f"TIME_STOP ({age_hr:.1f}h)"
            if reason:
                to_close.append((title, price, reason))

        for title, price, reason in to_close:
            pos = self.positions.pop(title, None)
            if pos is None:
                continue
            dir_sign = 1 if pos["direction"] == "LONG" else -1
            gross = (float(price) - pos["entry_price"]) * dir_sign * pos["size_bbl"]
            commission = COMMISSION_PER_CONTRACT_RT * (pos["size_bbl"] / 1000.0)
            pnl = gross - commission
            self.trades.append({
                **pos,
                "exit_price":   float(price),
                "exit_ts":      now,
                "pnl":          round(pnl, 2),
                "commission":   round(commission, 2),
                "gross_pnl":    round(gross, 2),
                "duration_min": round((now - pos["entry_ts"]) / 60, 1),
                "close_reason": reason,
            })

        for sig in signals:
            title = sig.get("title", "")
            # Prefer asset_key carried on the signal itself (used by the
            # auto_trader for new term-structure / technical signals).
            # Fall back to the global title->asset_key table for the
            # original 7 strategies.
            asset_key = sig.get("asset_key") or _SIGNAL_ASSETS.get(title)
            if not asset_key:
                continue
            price = assets.get(asset_key)
            if price is None:
                continue

            status = sig.get("status", "")
            direction = sig.get("direction", "")
            is_long = "LONG" in direction.upper()
            is_short = "SHORT" in direction.upper()

            open_pos = self.positions.get(title)

            if status == "ACTIVE" and not open_pos and (is_long or is_short):
                if not live:
                    continue   # warmup: skip new opens, keep tracking equity
                self.positions[title] = {
                    "title": title,
                    "direction": "LONG" if is_long else "SHORT",
                    "asset_key": asset_key,
                    "entry_price": float(price),
                    "entry_ts": now,
                    "size_bbl": int(sig.get("size_bbl", TRADE_SIZE_BBL)),
                    "rationale": sig.get("rationale", ""),
                    "source": sig.get("source", "manual"),
                }
            elif status != "ACTIVE" and open_pos:
                dir_sign = 1 if open_pos["direction"] == "LONG" else -1
                gross = ((float(price) - open_pos["entry_price"])
                         * dir_sign * open_pos["size_bbl"])
                commission = COMMISSION_PER_CONTRACT_RT * (open_pos["size_bbl"] / 1000.0)
                pnl = gross - commission
                self.trades.append({
                    **open_pos,
                    "exit_price":   float(price),
                    "exit_ts":      now,
                    "pnl":          round(pnl, 2),
                    "commission":   round(commission, 2),
                    "gross_pnl":    round(gross, 2),
                    "duration_min": round((now - open_pos["entry_ts"]) / 60, 1),
                })
                self.positions.pop(title, None)

        # record current equity point
        self.equity_curve.append({
            "ts": int(now),
            "equity": round(self._equity(assets), 2),
        })
        if len(self.equity_curve) > 500:
            self.equity_curve = self.equity_curve[-500:]

        if now - self._last_save > 30:
            self._save()
            self._last_save = now

    # ---- valuation --------------------------------------------------- #
    def _unrealized(self, assets: Dict[str, float]) -> float:
        """Mark-to-market value of open positions IF CLOSED NOW.
        Net of commission so this matches what realized P&L would be
        on liquidation — no rosy gross numbers."""
        total = 0.0
        for pos in self.positions.values():
            price = assets.get(pos["asset_key"])
            if price is None:
                continue
            dir_sign = 1 if pos["direction"] == "LONG" else -1
            gross = (float(price) - pos["entry_price"]) * dir_sign * pos["size_bbl"]
            commission = COMMISSION_PER_CONTRACT_RT * (pos["size_bbl"] / 1000.0)
            total += gross - commission
        return total

    def _realized(self) -> float:
        return sum(t["pnl"] for t in self.trades)

    def _equity(self, assets: Dict[str, float]) -> float:
        return self.starting_equity + self._realized() + self._unrealized(assets)

    # ---- snapshot for the dashboard ---------------------------------- #
    def _by_source_stats(self, assets: Dict[str, float]) -> Dict[str, Dict]:
        """Aggregate per-source (term / tech / manual) book stats.

        For each source we compute:
            n_open          currently open positions
            n_closed        closed trades count
            n_wins          closed winners
            win_rate        pct of closed trades with pnl > 0
            realized_pnl    sum of closed pnls
            unrealized_pnl  mark-to-market of open positions
            total_pnl       realized + unrealized
            avg_pnl         realized / n_closed
            best_trade      largest single-trade win
            worst_trade     largest single-trade loss
        """
        sources = ("term", "tech", "real", "composite", "bestpp", "pstrat", "manual")
        stats: Dict[str, Dict] = {s: {
            "n_open": 0, "n_closed": 0, "n_wins": 0, "win_rate": 0.0,
            "realized_pnl": 0.0, "unrealized_pnl": 0.0,
            "total_pnl": 0.0, "avg_pnl": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0,
        } for s in sources}

        # Closed trades (realized).
        per_source_closed: Dict[str, list] = {s: [] for s in sources}
        for t in self.trades:
            src = t.get("source", "manual")
            if src not in per_source_closed:
                src = "manual"
            per_source_closed[src].append(t)
        for src in sources:
            cs = per_source_closed[src]
            wins = [t for t in cs if t["pnl"] > 0]
            pnls = [t["pnl"] for t in cs]
            stats[src]["n_closed"]     = len(cs)
            stats[src]["n_wins"]       = len(wins)
            stats[src]["realized_pnl"] = round(sum(pnls), 2)
            stats[src]["win_rate"]     = (round(len(wins) / len(cs) * 100, 1)
                                          if cs else 0.0)
            stats[src]["avg_pnl"]      = (round(sum(pnls) / len(pnls), 2)
                                          if cs else 0.0)
            stats[src]["best_trade"]   = round(max(pnls), 2) if pnls else 0.0
            stats[src]["worst_trade"]  = round(min(pnls), 2) if pnls else 0.0

        # Open positions (unrealized) — net of commission, as if closed now.
        for pos in self.positions.values():
            src = pos.get("source", "manual")
            if src not in stats:
                src = "manual"
            stats[src]["n_open"] += 1
            price = assets.get(pos["asset_key"])
            if price is None:
                continue
            dir_sign = 1 if pos["direction"] == "LONG" else -1
            gross = (float(price) - pos["entry_price"]) * dir_sign * pos["size_bbl"]
            commission = COMMISSION_PER_CONTRACT_RT * (pos["size_bbl"] / 1000.0)
            stats[src]["unrealized_pnl"] += gross - commission

        for src in sources:
            stats[src]["unrealized_pnl"] = round(stats[src]["unrealized_pnl"], 2)
            stats[src]["total_pnl"] = round(
                stats[src]["realized_pnl"] + stats[src]["unrealized_pnl"], 2)
        return stats

    def snapshot(self, assets: Dict[str, float]) -> Dict:
        equity = self._equity(assets)
        pct = (equity / self.starting_equity - 1.0) * 100
        wins = [t for t in self.trades if t["pnl"] > 0]
        realized = self._realized()

        open_view = []
        for pos in self.positions.values():
            price = assets.get(pos["asset_key"])
            mtm: Optional[float] = None
            if price is not None:
                dir_sign = 1 if pos["direction"] == "LONG" else -1
                gross = (float(price) - pos["entry_price"]) * dir_sign * pos["size_bbl"]
                commission = COMMISSION_PER_CONTRACT_RT * (pos["size_bbl"] / 1000.0)
                mtm = round(gross - commission, 2)
            open_view.append({
                "title": pos["title"],
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "current_price": price,
                "mtm": mtm,
                "size_bbl": pos["size_bbl"],
                "open_min": round((time.time() - pos["entry_ts"]) / 60, 1),
                "source": pos.get("source", "manual"),
                "asset_key": pos.get("asset_key", ""),
            })

        by_source = self._by_source_stats(assets)
        # Per-source track record (full closed trade log, newest first).
        track_record = list(reversed(self.trades[-50:]))

        return {
            "starting_equity": self.starting_equity,
            "equity": round(equity, 2),
            "pct_change": round(pct, 2),
            "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(self._unrealized(assets), 2),
            "open_positions": open_view,
            "closed_trades": list(reversed(self.trades[-10:])),
            "track_record":  track_record,
            "by_source":     by_source,
            "n_trades": len(self.trades),
            "n_wins": len(wins),
            "win_rate": (round(100 * len(wins) / len(self.trades), 1)
                         if self.trades else 0.0),
            "equity_curve": self.equity_curve[-120:],
            "scheduled_reset_at": self.scheduled_reset_at,
        }
