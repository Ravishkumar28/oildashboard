# Oil Trading Desk — Technical Documentation

A real-time oil trading dashboard streaming live WTI/Brent prices, EIA fundamentals, CFTC positioning, AIS tanker tracking, NOAA hurricane intelligence, and dual-engine (VADER + FinBERT) news sentiment, deployed on Hugging Face Spaces.

**Live URL:** https://Ravish28-oil-trading-desk.hf.space
**Repo (HF Space):** https://huggingface.co/spaces/Ravish28/oil-trading-desk
**Total codebase:** ~6,984 lines across Python + JS + CSS + HTML.

---

## 1. System architecture

```
                         ┌──────────────────────────────────┐
                         │  External free data sources      │
                         │                                  │
                         │  Yahoo Finance  (WTI, Brent,     │
                         │     RBOB, HO, DXY, curve)        │
                         │  EIA v2 API     (stocks, refin.  │
                         │     util, OPEC, rigs, STEO)      │
                         │  CFTC Socrata   (COT positions)  │
                         │  Twelve Data    (DXY forex pairs)│
                         │  RSS x4         (news headlines) │
                         │  Google News    (analyst proxy)  │
                         │  NOAA NHC       (Atlantic storms)│
                         │  aisstream.io   (AIS tankers)    │
                         │  TradingView    (embedded chart) │
                         └──────────────┬───────────────────┘
                                        │ poll on cadence
                                        ▼
        ┌───────────────────────────────────────────────────────┐
        │  FastAPI backend (Python 3.12)                        │
        │                                                       │
        │  ┌──────────────┐  ┌────────────┐  ┌───────────────┐  │
        │  │ MarketEngine │  │ Periodic   │  │ AIS WS client │  │
        │  │ (real-prices │  │ refresh    │  │ (long-lived)  │  │
        │  │  mirror)     │  │ tasks      │  └───────────────┘  │
        │  └──────────────┘  └────────────┘                     │
        │  ┌────────────────────────────────────────────────┐   │
        │  │  build_snapshot() — assemble JSON every 2s    │   │
        │  └────────────────────────────────────────────────┘   │
        └───────────────────────────────┬───────────────────────┘
                                        │ WebSocket push every 2s
                                        ▼
        ┌───────────────────────────────────────────────────────┐
        │  Frontend (vanilla JS + Chart.js)                     │
        │  24 panels split across 2 tabs                        │
        │  — Main Dashboard (14 panels: prices, BB, spreads,    │
        │    curve, cracks, freight, covmatrix, fundamentals,   │
        │    COT, STEO balance, signals, 5y range, paper, news) │
        │  — Pro Tools     (11 panels: 6 TradingView widgets,   │
        │    spread matrices, analyst news, seasonality,        │
        │    storms, AIS tankers)                               │
        └───────────────────────────────────────────────────────┘
```

**Key architectural decision: real-prices-only mode.** The backend used to run a per-tick random-walk simulation pulled toward real anchors. This was removed in favor of a "mirror anchors directly onto live values" approach. Every value the dashboard displays now reflects the most-recent real fetch from its underlying API. Trade-off: chart looks frozen between refreshes; benefit: 100% data honesty, no smoothing/snapping artifacts.

---

## 2. Repository layout

```
.
├── README.md             # HF Space frontmatter (title, sdk, app_port)
├── TECHNICAL.md          # this file
├── Dockerfile            # python:3.12-slim + uvicorn + cwd=/app/backend
├── requirements.txt      # 8 runtime dependencies
├── .env                  # API keys (gitignored, NEVER committed)
├── backend/              # 19 Python modules, ~5,000 lines
│   ├── main.py           # FastAPI app, lifespan, simulation loop
│   ├── market.py         # MarketEngine — anchor mirror + history
│   ├── datafeed.py       # yfinance: WTI/Brent/RBOB/HO/DXY/curve
│   ├── eia.py            # EIA v2: stocks/Cushing/refin/OPEC/rigs
│   ├── steo.py           # EIA STEO: world supply/demand balance
│   ├── cot.py            # CFTC Socrata: NYMEX WTI positioning
│   ├── twelvedata.py     # Twelve Data forex pairs → DXY
│   ├── news.py           # RSS aggregator + analyst feeds
│   ├── sentiment.py      # VADER + FinBERT classifiers
│   ├── seasonality.py    # 5y refinery utilization patterns
│   ├── hurricane.py      # NOAA NHC + Gulf oil-asset overlay
│   ├── ais.py            # aisstream.io WebSocket tanker tracker
│   ├── fundamentals.py   # weekly fundamentals state (EIA-anchored)
│   ├── kalman.py         # 2-state Kalman pair filter (WTI/DXY)
│   ├── indicators.py     # BB, EMA, MA, VWAP, returns, correlation
│   ├── paper.py          # virtual $100k auto-trading book
│   ├── persistence.py    # HF Datasets sync for paper_state.json
│   ├── config.py         # .env loader → os.environ
│   └── backtest.py       # historical strategy backtests
├── frontend/             # 3 files, ~2,000 lines
│   ├── index.html        # 24 panels in 2 tabs
│   ├── app.js            # render functions + WebSocket client
│   └── style.css         # dark theme, grid layout
└── memory/               # local Claude memory (not deployed)
```

---

## 3. Data sources — real vs simulated honest map

| Field | Real source | Simulated fallback | Cadence |
|---|---|---|---|
| WTI / Brent spot | yfinance `CL=F`, `BZ=F` (15-min delayed) | none — synthetic random walk only if yfinance fails 5 retries | 60s |
| Futures curve (12 months) | yfinance `CL[F-Z][year].NYM` settlements | parameterized contango/backwardation slope | 60s |
| RBOB / Heating Oil | yfinance `RB=F`, `HO=F` | scaled from WTI | 60s (with prices) |
| Dollar Index (DXY) | Twelve Data 6-pair forex composite + ICE `DX-Y.NYB` history | synthetic random walk if both fail | 5 min |
| US crude stocks | EIA `PET.WCESTUS1.W` | synthetic drift | 30 min |
| Cushing inventory | EIA `PET.W_EPC0_SAX_YCUOK_MBBL.W` | synthetic | 30 min |
| Refinery utilization | EIA `PET.WPULEUS3.W` | synthetic | 30 min |
| US crude production | EIA `PET.WCRFPUS2.W` | synthetic | 30 min |
| OPEC production | EIA STEO `STEO.PAPR_OPEC.M` | synthetic | 30 min |
| Rig count | EIA `PET.E_ERTRR0_XR0_NUS_C.M` (monthly, 3-mo lag) | synthetic | 30 min |
| OPEC quota ceiling | hardcoded (no published API) | always hardcoded | n/a |
| Global S/D balance (STEO) | EIA 6 series: PAPR_WORLD, PATC_WORLD, OPEC, NONOPEC, OECD, NON_OECD | none — panel hides if STEO fails | 60 min |
| CFTC COT positioning | Socrata `gpe5-46if` Disaggregated weekly | none | 30 min |
| RSS news (4 feeds) | OilPrice + 2x Google News + Hellenic Shipping | none | 16s |
| Analyst news | 3x Google News searches (Bakr/Blas/Trump) | none | 15 min |
| BDTI freight index | none — Baltic Exchange paid-only | static at bootstrap value | n/a |
| 5-yr same-week range | yfinance daily closes over 5 years | none | 6h |
| Refinery seasonality | 5y EIA weekly utilization | none | 6h with 5y refresh |
| Hurricane tracking | NOAA NHC `CurrentStorms.json` + GoM asset overlay | none | 10 min |
| AIS tanker positions | aisstream.io WebSocket (terrestrial AIS) | none | continuous push |
| TradingView widgets | TradingView CDN (real-time CFD) | n/a (iframe) | live |
| News sentiment (per item) | VADER (always) + ProsusAI/finbert (transformer) | none | per-item async |

**Things that are genuinely paid-only and we DON'T attempt:**
- Baltic Exchange BDTI freight index
- Kpler/Vortexa physical cargo flows
- IIR/Genscape real-time refinery outages
- Crude grade differentials beyond what Yahoo provides
- Pipeline flows (Permian, Cushing)
- Options chain / implied volatility surface
- Satellite AIS for open-ocean coverage (free tier is terrestrial-only, ~40 mi from shore)

---

## 4. Backend module reference

### 4.1 `main.py` — FastAPI app + simulation loop

The top-level orchestrator. Defines:

- **Refresh cadences** (constants in seconds → ticks at 2s/tick):
  ```python
  PRICE_EVERY_TICKS = 30           # 60s    WTI/Brent
  CURVE_EVERY_TICKS = 30           # 60s    12-month futures curve
  NEWS_EVERY_TICKS = 8             # 16s    RSS aggregator
  DXY_EVERY_TICKS = 150            # 5 min  Twelve Data forex
  EIA_EVERY_TICKS = 900            # 30 min EIA fundamentals
  COT_EVERY_TICKS = 900            # 30 min CFTC positioning
  STEO_EVERY_TICKS = 1800          # 60 min EIA STEO balance
  ANALYST_EVERY_TICKS = 450        # 15 min Google News analyst feeds
  HURRICANE_EVERY_TICKS = 300      # 10 min NOAA NHC
  FIVE_YEAR_EVERY_TICKS = 10800    # 6h     5y same-week closes
  HISTORY_REFRESH_EVERY_TICKS = 10800   # 6h yfinance daily history
  ```

- **`Hub`** — owns the `MarketEngine`, `Fundamentals`, `NewsFeed`, `PaperBook`, `TankerTracker`, `StormTracker`, plus the WebSocket client set. Single instance shared across the FastAPI app.

- **`build_snapshot()`** — assembles the full JSON payload pushed every 2s to all WebSocket clients. Aggregates: header strip, price chart, BB, spread, futures curve, DXY correlation, crack spreads, freight, EWMA covariance matrix, fundamentals, signals, 5y range, news, news regions, news sentiment, analyst news, curve matrix, spread covariance, seasonality, FinBERT status, paper book, COT, storms, tankers, STEO.

- **`build_signals()`** — 7 trading-signal generators (see §6).

- **`lifespan`** — startup: yfinance history fetch with 5 retries, set_anchor with 3 retries, refresh_curve, refresh_five_year, refresh_seasonality, refresh_cot, refresh_analyst, refresh_dxy, refresh_eia, refresh_steo, refresh_hurricane, news warm-up, FinBERT warm-up, AIS WebSocket task. Shutdown: cancel simulation loop, stop AIS tracker.

- **`simulation_loop`** — ticks every 2s, calls `market.tick()`, runs cadence-gated refreshes, pushes snapshot to every WebSocket client.

### 4.2 `market.py` — MarketEngine (real-prices-only)

State container with **NO random-walk simulation** since the refactor. Holds:

- `self.wti`, `self.brent`, `self.dxy`, `self.rbob`, `self.heat`, `self.crude_inventory` — mirrored from anchors
- `self.anchor_*` — the most-recent real value from upstream APIs
- `self.hist["wti" | "brent" | "dxy" | "spread" | "bdti" | "crude_inventory"]` — rolling daily history (260 days)
- `self.crack_hist` — historical crack-spread time series
- `self.curve_hist` — rolling 200-snapshot deque of full 12-month curves (for spread covariance)
- `self.fly_history`, `self.kalman_resid_history` — strategy state

Key methods:
- `tick()` — mirrors anchors onto live values, appends curve snapshot to history. No simulation.
- `set_anchor(wti, brent)` — snaps `self.wti`/`self.brent` to new real values
- `set_real_curve(curve)` — accepts yfinance 12-month settlement curve
- `futures_curve()` — returns current curve from the cached real settlement
- `butterfly_value()` — M3-2*M6+M9 fly on the current curve
- `kalman_live_residual()` — current residual from Kalman pair model
- `_bootstrap_history(real_history)` — populates 1y of real WTI/Brent/DXY/RBOB/HO closes at startup

### 4.3 `datafeed.py` — yfinance integration

- `fetch_history(period="1y")` — joined daily closes for WTI + Brent + DXY + RBOB + Heat + WTI volume, aligned on common dates (drops mismatches via pandas inner-join). Used by lifespan bootstrap and the 6-h `refresh_history`.
- `fetch_latest()` — current (~15-min delayed) WTI + Brent.
- `fetch_products()` — current RBOB + HO.
- `fetch_curve()` — 12-month WTI futures settlements via NYMEX symbol composition (`CL[F-Z][YY].NYM`).
- `fetch_5y_same_week()` — 5 years of WTI same-week closes.

### 4.4 `eia.py` — EIA v2 API

Three callable shapes:
- `fetch_fundamentals(api_key)` — 7 series in one async batch: US crude stocks, Cushing, refinery utilization, US production, OPEC supply, rig count. Returns `{key: latest, key_prev: previous, key_period: date}` dict. Per-request timeout: 60s (EIA is slow on weekends/holidays).
- `fetch_refinery_history(api_key, years=5)` — 5y of weekly refinery utilization, used by `seasonality.py` for week-of-year averaging.
- `_fetch_series` — low-level helper.

### 4.5 `steo.py` — STEO global oil balance

Pulls 6 EIA STEO series (`PAPR_WORLD`, `PATC_WORLD`, `PAPR_OPEC`, `PAPR_NONOPEC`, `PATC_OECD`, `PATC_NON_OECD`). Returns 30 monthly observations (~10 historical + ~20 forecast). Computes:
- Per-month balance = supply − demand
- Forward 6M and 12M average balance
- Each row tagged historical vs forecast

### 4.6 `cot.py` — CFTC Commitment of Traders

- Socrata public API `publicreporting.cftc.gov/resource/gpe5-46if` (no auth needed)
- Filters to NYMEX WTI contract code `067411`
- Returns 4 trader categories: Managed Money, Producers/Commercials, Swap Dealers, Other Reportables — each with long, short, net, weekly change.
- Plus total open interest + WoW change.

### 4.7 `twelvedata.py` — DXY reconstruction

Forex-only on Twelve Data free tier. Rebuilds the ICE DXY from 6 component pairs using the published formula:
```
DXY = 50.14348112 × EUR/USD^-0.576 × USD/JPY^+0.136 × GBP/USD^-0.119
                  × USD/CAD^+0.091 × USD/SEK^+0.042 × USD/CHF^+0.036
```
One batched request returns all 6 pairs.

### 4.8 `news.py` — RSS aggregator + analyst tracker

- **Main feed** (`NewsFeed` class) — 4 RSS sources:
  1. OilPrice.com main feed
  2. Google News query `crude+oil+price`
  3. Google News query `WTI+OPEC+brent`
  4. Hellenic Shipping News
- **Three-layer stale-news defense:**
  1. `MAX_AGE_SECONDS = 6h` cap on `<pubDate>`
  2. URL date regex — drops items whose URL path encodes a date older than 14 days (catches Google News re-indexing of year-old articles)
  3. Headline year regex — drops headlines mentioning only old years
- Rolling deque (capacity 60). Per-item sentiment: VADER classification at fetch time, FinBERT backfill in async refresh loop.
- **Analyst feed** (`fetch_analyst_news`) — 3 Google News searches for Amena Bakr, Javier Blas, Trump+oil. Twitter API is paid-only since 2023; Google News mentions are the free proxy.

### 4.9 `sentiment.py` — VADER + FinBERT

- **VADER** — rule-based, fast, always available. Wrapped with an oil-finance lexicon overlay (`_OIL_LEXICON`) that handles terms VADER misses: `draw` (bullish), `build` (bearish), `cut` (bullish for crude), `outage` (bullish for products), `opec`, `embargo`, etc.
- **FinBERT** (`ProsusAI/finbert`) — 110M-parameter BERT fine-tuned on Financial PhraseBank (4,840 finance sentences, 16-annotator labels). Lazy-loaded in a background thread (`warm_finbert`) so the dashboard starts immediately. CPU inference takes ~30-50ms/headline. Compound score = `p_positive − p_negative`. Threshold ±0.15 for bullish/bearish labels.
- Combined per-item output: both V and F scores shown side-by-side in the news panel.

### 4.10 `seasonality.py` — Refinery utilization patterns

Computes week-of-year averages across 5 years of EIA weekly refinery utilization data:
- `SEASONAL_CALENDAR` — 6 hand-coded phases (Heating peak, Spring turnaround, Pre-driving build, Summer driving, Fall turnaround, Winter prep)
- `HURRICANE_START_WEEK = 22`, `HURRICANE_END_WEEK = 48` — Atlantic season weeks
- `compute_seasonal_pattern(history)` — week-of-year mean ± stddev band
- `current_phase()`, `next_phase()`, `build_summary()` — UI-ready output

### 4.11 `hurricane.py` — NOAA NHC + Gulf asset overlay

- Pulls `https://www.nhc.noaa.gov/CurrentStorms.json` (Bloomberg uses the same source)
- Filters to Atlantic basin (`id` starts with `AL`)
- **GULF_REFINERIES** — 13 hardcoded refineries with lat/lon/capacity (~5,393 kbpd combined = ~90% of PADD 3)
- **GOM_PRODUCTION_BBOX** — federal OCS production zone (~1,700 kbpd baseline)
- For each active storm: haversine distance to each refinery, capacity within 150 nm flagged as at-risk, GoM offshore production at risk if storm sits in OCS bbox
- Emits per-storm direction tag: `CRUDE BULLISH` (offshore shut) / `PRODUCTS BULLISH` (refinery shut) / `WATCH` / `MONITOR`

### 4.12 `ais.py` — aisstream.io WebSocket

- Long-lived WebSocket to `wss://stream.aisstream.io/v0/stream`
- Subscribes with 7 tight bounding boxes around real petroleum anchorages (Houston/Galveston, LOOP, Rotterdam/ARA, Singapore/Malacca, Fujairah, Caribbean Vz/Cu, Saldanha Bay)
- Filters AIS messages: `PositionReport` (every 2-10s per vessel) + `ShipStaticData` (every ~6 min per vessel)
- Identifies tankers via ship-type code 80-89 from ShipStaticData
- 90-min stale-purge on every snapshot
- Reconnects with exponential backoff on disconnect

### 4.13 `paper.py` — Virtual trading book

- Virtual $100k account that auto-trades the dashboard's signals
- Position MTM via current live anchor (every 60s for WTI/Brent)
- Optional persistent state via HF Datasets sync (when `PAPER_STATE_REPO` + `HF_TOKEN` Space secrets are set)
- Optional scheduled auto-reset via `PAPER_RESET_AT_UTC` (ISO 8601) — used to wipe holiday-period drift trades when real markets reopen

### 4.14 `kalman.py` — WTI/DXY pair filter

2-state Kalman filter tracking the linear relationship `WTI_t = α_t + β_t × DXY_t`:
- Hidden state: `[α, β]`
- Observation: WTI given DXY
- Process noise: small (slow regime drift)
- Used by the "Kalman Pair" strategy signal — when the residual z-score exceeds ±2σ, signals a decoupling trade

### 4.15 `indicators.py` — Technical indicators

Pure functions, no dependencies on engine state:
- `moving_average`, `ema`, `vwap`, `bollinger_bands`, `zscore`
- `correlation`, `covariance` (Pearson, equal-weighted)
- `ewma_correlation_matrix` (RiskMetrics λ-decay, equity covariance panel 08)
- `covariance_matrix` (Pearson correlation matrix of returns)
- `returns` (% change series)

### 4.16 `fundamentals.py` — Weekly fundamentals state

`Fundamentals` class — holds the 5 cards rendered in panel 09. When EIA data arrives via `apply_eia`, real values replace the simulated defaults. When EIA is unavailable, slow drift simulates weekly updates. `weekly_update()` is a no-op when `source` contains "EIA".

---

## 5. Frontend structure

### 5.1 Main Dashboard tab (14 panels)

| # | Panel | Source |
|---|---|---|
| Header | 9 key numbers + trend arrows | header strip from snapshot |
| 01 | Price chart + EMA/MA/VWAP | `m.series("wti")`, `m.series("brent")` |
| 02 | Bollinger Bands (20, 2σ) | `bollinger_bands(wti_hist)` |
| 03 | WTI-Brent spread | `spread_hist`, 120d mean/z-score |
| 04 | Futures curve 12-month | `m.futures_curve()` from real settlements |
| 05 | WTI · Dollar correlation | Pearson correlation of returns |
| 06 | Crack spreads (6 types) | `m.crack_spreads()` from real product prices |
| 07 | BDTI freight (SIM badge) | `m.bdti` (synthetic) |
| 08 | EWMA covariance matrix | `ewma_correlation_matrix(λ=0.94)` over 6 series |
| 09 | Fundamentals strip | `m.fundamentals.cards()` (EIA real) |
| 10 | CFTC COT positioning | `hub.cot` (Socrata) |
| STEO | Global oil balance | `hub.steo` (EIA STEO) |
| 11 | 7 trade signals | `build_signals(m, cracks, m12, sp_z)` |
| 12 | 5-year week range | `m.five_year_week()` |
| 13 | Paper trading | `hub.paper.snapshot(assets)` |
| 14 | Live news feed | `hub.news.snapshot()` + sentiment per item |

### 5.2 Pro Tools tab (11 panels)

| # | Panel | Source |
|---|---|---|
| T1 | WTI Crude (Live) | TradingView widget `TVC:USOIL` |
| T2 | Brent Crude (Live) | TradingView widget `TVC:UKOIL` |
| T3 | WTI-Brent spread | TradingView widget `TVC:USOIL-TVC:UKOIL` |
| T4 | Gasoline ETF (UGA) | TradingView widget `AMEX:UGA` |
| T5 | Heating Oil ETF (UHN) | TradingView widget `AMEX:UHN` |
| T6 | Natural Gas | TradingView widget `TVC:NGAS` |
| T7 | Calendar spread matrix | `build_curve_matrix(curve, spot)` |
| T7B | Spread covariance matrix | sample covariance over `curve_hist` |
| T8 | Analyst watch | `hub.analyst_news` (Bakr/Blas/Trump) |
| T9 | Refinery utilization seasonality | `hub.seasonality` |
| T10 | Tanker watch (Live AIS) | `hub.tankers.snapshot()` |
| T11 | Storm watch (Atlantic / US Gulf) | `hub.storms.snapshot()` |

### 5.3 Render flow (`app.js`)

```javascript
WebSocket /ws → JSON snapshot every 2s
  ↓
render(snapshot)
  ↓
renderHeader / renderPrice / renderBB / renderDXY / renderSpread /
renderFutures / renderFreight / renderCracks / renderCovMatrix /
renderFundamentals / renderSignals / renderFiveYear / renderCOT /
renderPaper / renderNewsMood / renderFinbertStatus / renderRegions /
renderNews / renderCurveMatrix / renderAnalystNews / renderSeasonality /
renderTankers / renderStorms / renderSpreadCovMatrix / renderSteo
```

Each renderer mutates only its own DOM subtree. Chart.js instances are cached in `charts[id]` and updated in-place (no recreation per tick).

---

## 6. Trade signals (panel 11)

7 statistical strategies emit `ACTIVE` / `WATCHING` with `LONG` / `SHORT` direction:

| # | Signal | Trigger | Logic |
|---|---|---|---|
| 1 | Diesel Refining Arbitrage | diesel crack z ≥ ±1.0σ | mean-reversion on Heating Oil − WTI |
| 2 | Crude Storage Carry | 12M contango > $6.60 (12 × $0.55 storage) | cash-and-carry economics |
| 3 | Curve Butterfly (M3-M6-M9) | fly z ≥ ±1.5σ | M3 + M9 − 2×M6 mean-reverts |
| 4 | Brent-WTI Spread | spread z ≥ ±2σ | convergence on extreme dislocation |
| 5 | 3-2-1 USGC Crack | crack z ≥ ±1.25σ | refining margin mean-reversion |
| 6 | WTI-DXY Pair | (WTI + 0.5·DXY) z ≥ ±2σ | inverse-relationship decoupling |
| 7 | Kalman Pair (dynamic β) | Kalman residual z ≥ ±2σ | 2-state filter for time-varying β |

Each signal is fed into `paper.py` for auto-trading the virtual $100k book.

---

## 7. Deployment

### 7.1 Hugging Face Spaces (Docker SDK)

- **Space:** `Ravish28/oil-trading-desk`
- **SDK:** Docker
- **Hardware:** CPU basic (free tier)
- **Port:** 7860 (set in `README.md` frontmatter + Dockerfile EXPOSE)
- **App command:** `uvicorn main:app --host 0.0.0.0 --port 7860` (from `/app/backend`)

### 7.2 Space Secrets (Settings → Variables and secrets)

Set as encrypted environment variables, never committed:

| Secret | Purpose | Fallback if unset |
|---|---|---|
| `TWELVE_DATA_API_KEY` | Real DXY from forex pairs | Synthetic DXY drift |
| `EIA_API_KEY` | Real US stocks / Cushing / refining / OPEC / rigs / STEO | Synthetic fundamentals |
| `AIS_API_KEY` | aisstream.io tanker tracking | Tanker panel disabled |
| `HF_TOKEN` | Persistent paper-book state via HF Datasets | Local-only paper state |
| `PAPER_STATE_REPO` | HF Dataset repo for paper book | Local-only paper state |
| `PAPER_RESET_AT_UTC` | ISO 8601 timestamp to auto-reset paper to $100k | No scheduled reset |

### 7.3 Deploy script pattern

```python
from huggingface_hub import HfApi
api = HfApi(token="hf_...")
api.upload_folder(
    folder_path=".",
    repo_id="Ravish28/oil-trading-desk",
    repo_type="space",
    commit_message="...",
    ignore_patterns=[".env", "__pycache__/*", "*.pyc",
                     ".venv/*", "venv/*",
                     "backend/paper_state.json",
                     ".git/*", "backtest_results.json",
                     "backend/__pycache__/*"],
)
api.restart_space(repo_id="Ravish28/oil-trading-desk")
```

`.env` is excluded from uploads. Secrets ride on the Space's encrypted secret store instead.

### 7.4 Dockerfile

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY frontend/ ./frontend/
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
EXPOSE 7860
WORKDIR /app/backend
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 7.5 Runtime dependencies (`requirements.txt`)

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
httpx==0.28.1
yfinance>=0.2.50
huggingface_hub>=0.26.0
vaderSentiment>=3.3.2
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.0.0
transformers>=4.40.0
websockets>=12.0
```

CPU-only torch (~150 MB) is intentional — FinBERT runs comfortably on CPU at ~30-50 ms/headline.

---

## 8. WebSocket protocol

- **Endpoint:** `wss://{host}/ws`
- **Direction:** server-to-client push only (clients optionally send `"ping"` keep-alive every 15s)
- **Payload:** one JSON snapshot per push (~50 KB typical)
- **Push cadence:** every 2 seconds (`TICK_SECONDS = 2.0`)
- **Initial paint:** on connect, server sends one snapshot immediately (no waiting for next tick)
- **Disconnect handling:** client auto-reconnects after 2s with exponential backoff (capped at 16s)

Snapshot top-level keys: `ts`, `tick`, `sources`, `header`, `price`, `bb`, `dxy`, `spread`, `futures`, `freight`, `cracks`, `covmatrix`, `fundamentals`, `signals`, `fiveyear`, `news`, `news_regions`, `news_sentiment`, `analyst_news`, `curve_matrix`, `spread_covmatrix`, `seasonality`, `finbert`, `paper`, `cot`, `tankers`, `storms`, `steo`.

---

## 9. Local development

```bash
git clone <hf-space-or-mirror>
cd dsa
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env               # then edit with your own keys
cd backend
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
# open http://localhost:7860
```

For the dashboard to run with full real data, populate `.env` with keys for Twelve Data, EIA, and aisstream.io. Without those, the panels gracefully fall back (DXY/fundamentals to simulation; tankers panel disabled).

---

## 10. Extending the dashboard

### Adding a new data source

1. Create `backend/<source>.py` with an async `fetch_xxx(api_key)` function returning a JSON-serialisable dict (or `None` on failure).
2. Add cadence constant in `main.py` (`<SOURCE>_EVERY_TICKS = N`).
3. Add `async def refresh_<source>` calling the fetch, storing result on `hub`.
4. Wire into `simulation_loop`: `if hub.tick % <SOURCE>_EVERY_TICKS == 0: await refresh_<source>()`.
5. Wire into `lifespan` for warm-up (with retries for slow APIs).
6. Add key to `build_snapshot()` return dict.
7. Add API key (if needed) to `config.py` and Space Secrets.

### Adding a new dashboard panel

1. Add `<article class="panel">` block to `frontend/index.html` with the unique panel number.
2. Add a `render<Name>(data)` function to `app.js`.
3. Call it in the master `render(d)` function.
4. Add CSS styles to `frontend/style.css`.

### Adding a new trading signal

1. Add the signal logic to `build_signals()` in `main.py`.
2. Append the result dict to the `return [...]` list.
3. Map the trigger asset to the `assets` dict consumed by `paper.update()` if you want auto-trading.

---

## 11. Architectural decisions log

### Real-prices-only mode (May 2026)

**Decision:** Remove per-tick random-walk simulation; mirror real anchors directly onto live values.

**Why:** The hybrid sim-plus-real-anchor architecture produced visible chart artifacts whenever the simulated drift diverged from reality (e.g. WTI drifted to $70 while real anchor jumped to $89 → 1-tick vertical jump). Smoothing the jump only hid the symptom.

**Trade-off:** Chart no longer ticks every 2s. Values update once per real-data refresh (every 60s for prices, 30 min for EIA, etc.). Lost the "live trading screen" feel; gained 100% data honesty.

**Backward-incompatible removals:** `tick()` no longer does random walk. `_commit_day()` no longer runs (dead code retained for reference). `_bounded()` removed entirely. `set_anchor()` no longer smooths/snaps incrementally — direct assignment.

### EWMA covariance matrix uses real RBOB/HO history (May 2026)

**Decision:** Extend `fetch_history` to include RBOB and Heating Oil daily closes; recompute crack-spread history from real product prices at startup.

**Why:** Previously the crack-spread history was "real WTI minus synthetic RBOB" — the resulting covariances were structurally wrong (overstating gasoline-diesel correlation at 0.65 vs real 0.37, missing the −0.21 DXY-crack relationship entirely).

**Trade-off:** None practically. Adds 2 more yfinance series to the bootstrap (cheap on a 1y fetch).

### Stale-news three-layer defense (May 2026)

**Decision:** Drop pubDate cap from 24h to 6h; add URL date regex (drop items with `/YYYY/MM/` paths >14 days old); add headline year regex.

**Why:** Google News RSS reports its *indexing* time as `<pubDate>`, not the article's actual publication date. Year-old articles re-indexed get fresh-looking timestamps.

**Trade-off:** ~50% fewer news items in the panel. The remaining items are all genuinely fresh.

### AIS bounding boxes tightened to actual petroleum anchorages (May 2026)

**Decision:** Shrink each AIS subscription bounding box from rough metropolitan areas to tight petroleum-terminal coordinates.

**Why:** The original Rotterdam box covered Antwerp + half the southern North Sea, returning 3000+ vessels (mostly container/ferry/fishing). Real Rotterdam tanker traffic is ~50-150 vessels.

**Trade-off:** Coverage gaps for vessels in transit far from terminals. Acceptable since the panel headline is "confirmed tankers" (ship type 80-89), not "all vessels".

---

## 12. Honest limitations

What the dashboard does NOT have, with the cost to add:

| Gap | Cost to close |
|---|---|
| Sub-15-min real-time WTI/Brent prints | Twelve Data crude tier ~$30/mo, or use embedded TradingView widgets (already done for T1-T6) |
| Options chain + implied volatility surface | CBOE/ICE options feed ~$500-3000/mo |
| Physical cargo flows (floating storage, STS) | Kpler / Vortexa ~$30-100k/yr |
| Real-time refinery outages | IIR / Genscape ~$15-40k/yr |
| Pipeline utilization | Genscape ~$10k/yr |
| Crude grade differentials (Urals, Mars, WCS) | Argus / Platts ~$10-20k/yr |
| Open-ocean AIS (satellite) | Spire / exactEarth ~$5-20k/yr |
| Bloomberg / Reuters terminal-class news | $25-60k/seat/yr |

What's free and could still be added (~1 day each):

| Gap | Effort |
|---|---|
| Risk metrics on paper book (Sharpe, max drawdown, VaR, profit factor) | ~4 hours |
| Backtest results panel | ~3 hours |
| Multi-asset macro correlations (HY credit, copper, gold, 10Y) | ~4 hours |
| Event markers on price chart (EIA Wed releases, OPEC dates, storm landfalls) | ~1 day |
| OPEC+ meeting calendar with countdown | ~2 hours |
| SPR release tracker | ~2 hours (EIA series, already have key) |
| Geopolitical Risk Index (Caldara-Iacoviello via FRED) | ~3 hours |

---

## 13. File-by-file diff history (last significant refactors)

- **2026-05-28** — Spread covariance switched from EWMA correlation to plain sample covariance ($²/bbl²)
- **2026-05-28** — Real RBOB/HO/DXY history added; auto-refresh every 6h
- **2026-05-28** — Real-prices-only refactor (random walk removed from `tick()`)
- **2026-05-28** — STEO global oil balance panel added
- **2026-05-28** — NOAA NHC hurricane tracking + Gulf asset overlay (T11)
- **2026-05-28** — aisstream.io tanker tracking (T10), boxes tightened
- **2026-05-28** — Three-layer stale-news defense (pubDate cap + URL regex + headline year)
- **2026-05-28** — Refresh cadences dropped to absolute minimum per source

---

## 14. Contact / contributing

- **Build:** Ravish (b23me1053@iitj.ac.in), IIT Jodhpur
- **Live dashboard:** https://Ravish28-oil-trading-desk.hf.space
- **Stack:** Python 3.12, FastAPI, vanilla JS, Chart.js, Docker, HF Spaces
- **License:** unspecified — treat as personal portfolio project

This dashboard is a working demonstration that real-time oil-market analytics can be built end-to-end on entirely free public data sources (yfinance, EIA, CFTC, NOAA, aisstream, Twelve Data, RSS), without paid Bloomberg/Reuters-class feeds, deployed to a free hosting tier. The honest trade-off is real-time fidelity (15-min delayed prices, 30-min-to-6-hour fundamentals lag) in exchange for zero ongoing cost.
