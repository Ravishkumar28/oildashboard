"""Oil-market fundamentals: rig count, crude storage, OPEC output, Cushing
stocks and refinery utilization.

These move on a weekly cadence in reality, so the simulation drifts them
slowly. If an EIA_API_KEY environment variable is present, datafeeds.py can
override the simulated values with real EIA series."""
from __future__ import annotations

import random
from typing import Dict, List


class Fundamentals:
    """Slow-drifting weekly fundamentals with week-over-week deltas."""

    def __init__(self) -> None:
        self.rig_count = 482          # Baker Hughes US oil rigs
        self.opec_output = 26.9       # million bbl/day (OPEC crude)
        self.opec_quota = 26.6        # agreed production ceiling
        self.cushing = 32.4           # million bbl at Cushing, OK hub
        self.utilization = 90.6       # refinery utilization %
        self._prev = self._levels()
        self.source = "simulated"     # set to "EIA" when real data is wired

    def _levels(self) -> Dict[str, float]:
        return {
            "rig_count": self.rig_count,
            "opec_output": self.opec_output,
            "cushing": self.cushing,
            "utilization": self.utilization,
        }

    def apply_eia(self, data: Dict[str, float]) -> None:
        """Replace simulated values with real EIA observations. Cushing,
        refinery utilization, OPEC supply, and rig count are real from EIA;
        only OPEC quota stays simulated (not a published EIA series)."""
        self._prev = self._levels()
        if "cushing_mbbl" in data:
            self.cushing = float(data["cushing_mbbl"])
            self._prev["cushing"] = float(
                data.get("cushing_mbbl_prev", self.cushing))
        if "refinery_utilization" in data:
            self.utilization = float(data["refinery_utilization"])
            self._prev["utilization"] = float(
                data.get("refinery_utilization_prev", self.utilization))
        if "opec_production_mbpd" in data:
            self.opec_output = float(data["opec_production_mbpd"])
        if "rig_count" in data:
            self.rig_count = int(data["rig_count"])
            self._prev["rig_count"] = int(
                data.get("rig_count_prev", self.rig_count))
        # OPEC quota field removed — was synthetic. Source is now EIA-only.
        self.source = "EIA (real, weekly)"

    def weekly_update(self) -> None:
        """Advance fundamentals by one simulated week. When real EIA data is
        active this becomes a no-op — every numeric field is refreshed by
        ``apply_eia`` from real series (including rigs)."""
        self._prev = self._levels()
        if "EIA" in self.source:
            return
        self.rig_count = max(380, min(620,
                         self.rig_count + random.randint(-7, 7)))
        self.opec_output = max(24.0, min(30.0,
                          self.opec_output + random.uniform(-0.18, 0.18)))
        self.opec_quota = max(24.0, min(30.0,
                          self.opec_quota + random.uniform(-0.05, 0.05)))
        self.cushing = max(18.0, min(60.0,
                       self.cushing + random.uniform(-1.4, 1.4)))
        self.utilization = max(78.0, min(97.0,
                           self.utilization + random.uniform(-1.6, 1.6)))

    def cards(self, crude_inventory: float) -> List[Dict[str, object]]:
        """Five fundamentals cards for the dashboard strip."""
        def card(label, value, unit, prev, fmt="{:.1f}", good_up=True,
                 note=""):
            delta = value - prev
            trend = "flat"
            if abs(delta) > 1e-6:
                up = delta > 0
                trend = "up" if up else "down"
            return {
                "label": label,
                "value": fmt.format(value),
                "unit": unit,
                "delta": fmt.format(delta) if delta >= 0
                         else "-" + fmt.format(abs(delta)),
                "trend": trend,
                "bullish": (trend == "up") == good_up if trend != "flat"
                           else None,
                "note": note,
            }

        opec_gap = self.opec_output - self.opec_quota
        return [
            card("Active US Rotary Rigs", self.rig_count, "rigs",
                 self._prev["rig_count"], "{:.0f}", good_up=True,
                 note="EIA monthly · oil + gas combined · ~3mo lag"),
            card("US Crude Storage", crude_inventory, "M bbl",
                 crude_inventory, "{:.1f}", good_up=False,
                 note="EIA commercial crude stocks"),
            {
                "label": "OPEC Production",
                "value": "{:.2f}".format(self.opec_output),
                "unit": "M bpd",
                "delta": ("+" if opec_gap >= 0 else "") +
                         "{:.2f}".format(opec_gap),
                "trend": "up" if opec_gap > 0.05 else
                         ("down" if opec_gap < -0.05 else "flat"),
                "bullish": opec_gap < 0,
                "note": ("vs {:.2f} M bpd quota "
                         "<span class='sim-inline'>SIMULATED</span>"
                         ).format(self.opec_quota),
            },
            card("Cushing Inventory", self.cushing, "M bbl",
                 self._prev["cushing"], "{:.1f}", good_up=False,
                 note="WTI delivery hub stocks"),
            card("Refinery Utilization", self.utilization, "%",
                 self._prev["utilization"], "{:.1f}", good_up=True,
                 note="% of capacity running"),
        ]
