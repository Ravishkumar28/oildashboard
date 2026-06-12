---
title: Oil Trading Desk
sdk: docker
app_port: 7860
pinned: false
---

Live oil trading dashboard: WTI/Brent (yfinance), Dollar Index (Twelve Data
forex), crack spreads (real RBOB/HO), 12-month futures curve, EIA
fundamentals (US crude stocks, Cushing, refinery utilization, OPEC), and
live OilPrice RSS news. WebSocket push from a FastAPI backend.

## Required secrets (set in Space Settings → Variables and secrets)

| Name | Purpose | Fallback if unset |
|---|---|---|
| `TWELVE_DATA_API_KEY` | Real Dollar Index from forex pairs | Simulated DXY |
| `EIA_API_KEY` | Real US crude stocks, Cushing, refinery use, OPEC | Simulated fundamentals |
