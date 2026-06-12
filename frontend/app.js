/* Oil Trading Desk — live dashboard client.
   Connects to the FastAPI WebSocket, renders 13 panels, auto-reconnects. */

"use strict";

// Color constants used by every chart and inline-styled cell. Values are
// hydrated from CSS variables at boot so the SAME object reflects whichever
// theme is active. refreshThemeColors() re-reads them whenever the user
// toggles, then triggers chart re-renders on the next snapshot tick.
const C = {
  up: "#29c46f", down: "#f0556a", accent: "#f3a712",
  blue: "#4aa3df", dim: "#7c8aa0", line: "#1f2937", text: "#d6dde8",
};

// Declared up here so refreshThemeColors() (called from the initTheme IIFE
// below) does not hit a TDZ ReferenceError. Populated lazily by lineChart().
const charts = {};

function refreshThemeColors() {
  const css = getComputedStyle(document.documentElement);
  const v = (name, fallback) => {
    const x = css.getPropertyValue(name).trim();
    return x || fallback;
  };
  C.up     = v("--up",     C.up);
  C.down   = v("--down",   C.down);
  C.accent = v("--accent", C.accent);
  C.blue   = v("--blue",   C.blue);
  C.dim    = v("--dim",    C.dim);
  C.line   = v("--line",   C.line);
  C.text   = v("--text",   C.text);
  Chart.defaults.color = C.dim;
  Chart.defaults.borderColor = C.line;
  // Force every live chart to repaint with the new palette
  Object.values(charts || {}).forEach((ch) => {
    try { ch.update("none"); } catch (e) {}
  });
}

function setTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem("theme", t); } catch (e) {}
  // Update button label so the user sees what state they're in
  const btn = document.getElementById("themeToggle");
  if (btn) btn.textContent = (t === "light" ? "☾ Dark" : "☼ Light");
  refreshThemeColors();
}

// Initialize theme BEFORE Chart.defaults are read so charts start
// in the correct palette and don't have to re-render once.
(function initTheme() {
  let t = "dark";
  try { t = localStorage.getItem("theme") || "dark"; } catch (e) {}
  document.documentElement.setAttribute("data-theme", t);
  refreshThemeColors();
  const btn = document.getElementById("themeToggle");
  if (btn) btn.textContent = (t === "light" ? "☾ Dark" : "☼ Light");
})();

Chart.defaults.color = C.dim;
Chart.defaults.borderColor = C.line;
Chart.defaults.font.family = "Consolas, monospace";
Chart.defaults.font.size = 10;
Chart.defaults.animation.duration = 300;

const seenNews = new Set();
let regionFilter = "All";
let lastNews = [];

/* ---------- helpers ---------- */
const $ = (id) => document.getElementById(id);

function fmt(n, d = 2) {
  if (n === null || n === undefined) return "—";
  return Number(n).toLocaleString("en-US",
    { minimumFractionDigits: d, maximumFractionDigits: d });
}
function arrowFor(trend) {
  return trend === "up" ? "▲" : trend === "down" ? "▼" : "▬";
}
function timeAgo(ts) {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (s < 60) return s + "s ago";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  return Math.floor(s / 3600) + "h ago";
}

function lineChart(id, cfg) {
  if (charts[id]) return charts[id];
  charts[id] = new Chart($(id), {
    type: cfg.type || "line",
    data: { labels: cfg.labels || [], datasets: cfg.datasets },
    options: Object.assign({
      responsive: true, maintainAspectRatio: false,
      interaction: { intersect: false, mode: "index" },
      elements: { point: { radius: 0 }, line: { borderWidth: 1.6, tension: 0.25 } },
      plugins: { legend: { display: cfg.legend !== false,
        labels: { boxWidth: 10, boxHeight: 2, padding: 8 } } },
      scales: cfg.scales || { x: { display: false }, y: {} },
    }, cfg.options || {}),
  });
  return charts[id];
}

/* ---------- 01 key numbers ---------- */
function renderHeader(items) {
  $("keynums").innerHTML = items.map((k) => {
    const isNum = typeof k.value === "number";
    const val = isNum ? fmt(k.value, Number.isInteger(k.value) ? 0 : 2)
                      : k.value;
    return `<div class="kn">
      <span class="kl">${k.label}</span>
      <span class="kv">${val}<span class="ku">${k.unit || ""}</span>
        <span class="arrow ${k.trend}">${arrowFor(k.trend)}</span></span>
    </div>`;
  }).join("");
}

/* ---------- 02 price chart ---------- */
function renderPrice(p) {
  const ch = lineChart("priceChart", {
    labels: p.labels,
    datasets: [
      // WTI + WTI overlays
      { label: "WTI", data: p.wti, borderColor: C.accent,
        backgroundColor: "rgba(243,167,18,.07)", fill: true },
      { label: "WTI EMA20", data: p.ema20, borderColor: C.up,
        borderDash: [4, 3], borderWidth: 1.2 },
      { label: "WTI MA50",  data: p.ma50,  borderColor: C.down,
        borderDash: [6, 4], borderWidth: 1.2 },
      { label: "WTI VWAP20", data: p.vwap20, borderColor: C.dim,
        borderDash: [2, 4], borderWidth: 1 },
      // Brent + Brent overlays — distinct colors so the two pairs are
      // visually separable: solid blue Brent, cyan EMA, light-pink MA.
      { label: "Brent", data: p.brent, borderColor: C.blue, borderWidth: 1.6 },
      { label: "Brent EMA20", data: p.brent_ema20, borderColor: "#7ad7ff",
        borderDash: [4, 3], borderWidth: 1.2 },
      { label: "Brent MA50",  data: p.brent_ma50,  borderColor: "#f0a0b0",
        borderDash: [6, 4], borderWidth: 1.2 },
    ],
  });
  ch.data.labels = p.labels;
  ch.data.datasets[0].data = p.wti;
  ch.data.datasets[1].data = p.ema20;
  ch.data.datasets[2].data = p.ma50;
  ch.data.datasets[3].data = p.vwap20;
  ch.data.datasets[4].data = p.brent;
  ch.data.datasets[5].data = p.brent_ema20;
  ch.data.datasets[6].data = p.brent_ma50;
  ch.update("none");

  // Per-product trend tag: "WTI ↗ / Brent ↘" or similar, color-coded by mix.
  function trend(price, ema) {
    if (ema == null || price == null) return null;
    return price > ema ? "up" : "down";
  }
  const w = trend(p.wti[p.wti.length - 1],   p.ema20[p.ema20.length - 1]);
  const b = trend(p.brent[p.brent.length-1], p.brent_ema20[p.brent_ema20.length-1]);
  const t = $("priceTrend");
  if (!w && !b) { t.textContent = "—"; t.className = "tag"; }
  else {
    const arrow = (s) => s === "up" ? "↗" : s === "down" ? "↘" : "·";
    t.textContent = `WTI ${arrow(w)} / Brent ${arrow(b)}`;
    if (w === "up" && b === "up")       t.className = "tag good";
    else if (w === "down" && b === "down") t.className = "tag bad";
    else                                    t.className = "tag warn";
  }
}

/* ---------- 03 Bollinger Bands ---------- */
function renderBB(b, opts) {
  // opts: { prefix, chartId, label, color, decimals }
  // Defaults preserve the legacy WTI panel that calls renderBB(d.bb).
  opts = opts || {};
  const prefix   = opts.prefix   || "bb";
  const chartId  = opts.chartId  || "bbChart";
  const label    = opts.label    || "WTI";
  const color    = opts.color    || C.accent;
  const dec      = opts.decimals != null ? opts.decimals : 2;
  if (!b) {
    const el = $(prefix + "Pos"); if (el) el.textContent = "—";
    return;
  }
  $(prefix + "Pos").textContent = fmt(b.position, 1) + "%";
  const st = $(prefix + "State");
  st.textContent = b.state;
  const bearish = b.position >= 80;
  const bullish = b.position <= 20;
  st.className = "tag " + (bearish ? "bad" : bullish ? "good" : "warn");
  $(prefix + "Pos").style.color = bearish ? C.down : bullish ? C.up : C.text;

  $(prefix + "Upper").textContent = "$" + fmt(b.upper, dec);
  $(prefix + "Mid").textContent   = "$" + fmt(b.middle, dec);
  $(prefix + "Lower").textContent = "$" + fmt(b.lower, dec);

  const ch = lineChart(chartId, {
    labels: b.history.labels,
    datasets: [
      { label: label,    data: b.history.price,  borderColor: color,
        backgroundColor: "rgba(243,167,18,.06)", fill: false },
      { label: "Upper",  data: b.history.upper,  borderColor: C.down,
        borderDash: [3, 3], borderWidth: 1 },
      { label: "Middle", data: b.history.middle, borderColor: C.dim,
        borderWidth: 1 },
      { label: "Lower",  data: b.history.lower,  borderColor: C.up,
        borderDash: [3, 3], borderWidth: 1 },
    ],
    legend: false,
    scales: { x: { display: false }, y: {} },
  });
  ch.data.labels = b.history.labels;
  ch.data.datasets[0].data = b.history.price;
  ch.data.datasets[0].borderColor = color;
  ch.data.datasets[0].label = label;
  ch.data.datasets[1].data = b.history.upper;
  ch.data.datasets[2].data = b.history.middle;
  ch.data.datasets[3].data = b.history.lower;
  ch.update("none");
}

/* ---------- 04B Regime-Aware Butterfly (Lasso ⨯ LogReg) ---------- */
function renderRegimeButterfly(r) {
  if (!r) {
    $("regimeFitStatus").textContent = "no data";
    return;
  }
  // Fit-status badge
  const fitEl = $("regimeFitStatus");
  if (!r.available) {
    fitEl.textContent = "sklearn missing";
    fitEl.className = "tag mini bad";
  } else if (r.model_fitted) {
    const nReg = r.regime_sample_counts ? r.regime_sample_counts[r.regime] || 0 : 0;
    fitEl.textContent = `fit · n=${nReg} samples · total=${r.n_total_samples || 0}`;
    fitEl.className = "tag mini good";
  } else {
    fitEl.textContent = `warming up · ${r.n_total_samples || 0} samples`;
    fitEl.className = "tag mini warn";
  }

  // Winning-model badge (Lasso vs Huber for the CURRENT regime).
  const winEl = $("regimeWinner");
  if (winEl) {
    const w = r.winning_model;
    if (!w) {
      winEl.textContent = "model: —";
      winEl.className = "tag mini";
    } else {
      winEl.textContent = `winner: ${w}`;
      winEl.className = w === "Huber" ? "tag mini good" : "tag mini";
    }
  }

  // Right-column heading dynamically reflects which model's weights we're showing.
  const headEl = $("regimeWeightsHead");
  if (headEl) {
    const mdl = r.winning_model || "model";
    headEl.textContent = `${mdl} factor weights for this regime`;
  }

  // Action pill
  const actEl = $("regimeAction");
  actEl.textContent = r.action || "WATCH";
  if (r.action === "LONG FLY") {
    actEl.className = "tag good";
  } else if (r.action === "SHORT FLY") {
    actEl.className = "tag bad";
  } else {
    actEl.className = "tag warn";
  }

  // Stats
  $("regimeName").textContent = r.regime || "—";
  const slope = r.curve_slope_m12_m1;
  $("regimeName").nextElementSibling.textContent =
    `current regime · M12−M1 = ${slope != null ? slope.toFixed(2) + " $/bbl" : "—"}`;

  const flyVal = r.current_fly != null ? "$" + r.current_fly.toFixed(2) : "—";
  const flyZ   = r.current_fly_z != null ? r.current_fly_z.toFixed(1) + "σ" : "—";
  $("regimeFlyNow").textContent = `${flyVal} · ${flyZ}`;

  $("regimeFlyPred").textContent =
    r.predicted_fly != null ? "$" + r.predicted_fly.toFixed(2) : "—";

  $("regimeRationale").textContent = r.rationale || "—";

  // Regime probability bars
  const probs = r.regime_probs;
  const names = r.regime_names || [];
  const probBars = $("regimeProbBars");
  if (probs && names.length) {
    probBars.innerHTML = names.map((nm, i) => {
      const p = probs[i] || 0;
      const isCurrent = (i === r.regime_idx);
      const bg = isCurrent ? C.accent : C.dim;
      return `<div style="display:flex; align-items:center; gap:8px; margin-bottom:3px">
        <span style="width:120px; font-size:11px; color:${isCurrent ? C.text : C.dim}">${nm}</span>
        <div style="flex:1; height:10px; background:rgba(255,255,255,.05); border-radius:2px; overflow:hidden">
          <div style="width:${Math.round(p*100)}%; height:100%; background:${bg}"></div>
        </div>
        <span style="width:42px; font-size:11px; color:${C.dim}">${(p*100).toFixed(0)}%</span>
      </div>`;
    }).join("");
  } else {
    const err = r.lr_error ? `LR failed: ${r.lr_error}` : "LR not yet fitted.";
    probBars.innerHTML = `<div style="font-size:11px; color:var(--dim)">${err}</div>`;
  }

  // Lasso weights bars
  const wEl = $("regimeLassoWeights");
  const ws = r.weights || [];
  if (ws.length) {
    const maxAbs = Math.max(...ws.map(w => Math.abs(w.weight)), 0.0001);
    wEl.innerHTML = ws.map(w => {
      const pct = Math.abs(w.weight) / maxAbs * 100;
      const colour = w.weight >= 0 ? C.up : C.down;
      return `<div style="display:flex; align-items:center; gap:8px; margin-bottom:3px">
        <span style="width:110px; font-size:11px; color:var(--text)">${w.factor}</span>
        <div style="flex:1; height:10px; background:rgba(255,255,255,.05); border-radius:2px; overflow:hidden; position:relative">
          <div style="width:${pct}%; height:100%; background:${colour}"></div>
        </div>
        <span style="width:60px; font-size:11px; text-align:right; color:${colour}">${w.weight >= 0 ? "+" : ""}${w.weight.toFixed(3)}</span>
      </div>`;
    }).join("");
  } else {
    wEl.innerHTML = '<div style="font-size:11px; color:var(--dim)">Lasso zeroed out all factors (or not yet fitted).</div>';
  }

  // Per-regime winner table (Lasso vs Huber for each fitted regime).
  const pwEl = $("regimePerWinner");
  if (pwEl) {
    const winners = r.per_regime_winner || {};
    const counts  = r.regime_sample_counts || {};
    const regimes = r.regime_names || [];
    let html = "";
    regimes.forEach((name) => {
      const winner = winners[name];
      const n = counts[name] || 0;
      const isCurrent = (name === r.regime);
      let badge;
      if (!winner) {
        badge = `<span class="rpw-badge rpw-none">not fit · n=${n}</span>`;
      } else {
        const cls = winner === "Huber" ? "rpw-huber" : "rpw-lasso";
        badge = `<span class="rpw-badge ${cls}">${winner} · n=${n}</span>`;
      }
      html += `<div class="rpw-row${isCurrent ? ' rpw-current' : ''}">
        <span class="rpw-name">${name}${isCurrent ? ' ◀' : ''}</span>
        ${badge}
      </div>`;
    });
    pwEl.innerHTML = html;
  }
}

/* ---------- T7E Multi-Product Strategy Matrix ---------- */
function renderMultiProductMatrix(m) {
  if (!m) {
    $("matrixFitStatus").textContent = "no data";
    $("multiProductMatrix").innerHTML = "";
    return;
  }
  const fitEl = $("matrixFitStatus");
  fitEl.textContent = m.available
    ? `live · 5 products`
    : "sklearn missing";
  fitEl.className = "tag mini " + (m.available ? "good" : "bad");

  const verdictColor = (v) =>
    v === "LONG FLY" || v === "LONG"  ? C.up :
    v === "SHORT FLY" || v === "SHORT" ? C.down : C.dim;

  const regimeColor = (r) => {
    if (!r) return C.dim;
    if (r.startsWith("Steep Backwardation")) return "#ff5e7e";
    if (r === "Backwardation") return "#f0a0b0";
    if (r === "Flat") return C.dim;
    if (r === "Contango") return "#7ad7ff";
    if (r === "Steep Contango") return "#4aa3df";
    return C.dim;
  };

  const rows = (m.rows || []).map(p => {
    if (!p.available) {
      return `<div class="mp-row mp-empty">
        <span class="mp-name"><b>${p.name}</b><span class="mp-unit">${p.unit||""}</span></span>
        <span style="grid-column: 2 / -1; color: var(--dim)">${p.reason||"unavailable"}</span>
      </div>`;
    }
    const verdict = p.combined || {};
    const fund = p.fundamental || {};
    const tech = p.technical || {};
    const spreads = p.spreads || {};
    const lasso = p.lasso;

    const sprStr = ["M1-M2","M1-M3","M3-M6","M6-M9"]
      .map(k => `${k}:&nbsp;${(spreads[k] != null ? spreads[k].toFixed(3) : "—")}`)
      .join(" · ");

    let weightsBars = "";
    if (lasso && lasso.weights && lasso.weights.length) {
      const maxAbs = Math.max(...lasso.weights.map(w => Math.abs(w.weight)), 0.0001);
      weightsBars = lasso.weights.slice(0, 5).map(w => {
        const pct = Math.abs(w.weight) / maxAbs * 100;
        const col = w.weight >= 0 ? C.up : C.down;
        return `<div class="mp-wt-row">
          <span class="mp-wt-name">${w.factor}</span>
          <div class="mp-wt-bar"><div style="width:${pct}%; background:${col}"></div></div>
          <span class="mp-wt-val" style="color:${col}">${w.weight>=0?"+":""}${w.weight.toFixed(3)}</span>
        </div>`;
      }).join("");
    } else {
      weightsBars = `<div class="mp-wt-empty">No fit yet (n=${p.n_fly_history||0} fly obs).</div>`;
    }

    const macros = (p.macros || []).map(x =>
      `<span class="mp-macro">${x}</span>`).join("");

    return `
    <div class="mp-row">
      <div class="mp-col mp-product">
        <div class="mp-name"><b>${p.name}</b> <span class="mp-unit">${p.unit||""}</span></div>
        <div class="mp-regime" style="color:${regimeColor(p.regime)}">
          ${p.regime} · slope ${p.slope!=null ? (p.slope>=0?"+":"") + p.slope.toFixed(2) : "—"}
        </div>
        <div class="mp-fly">Fly: <b>${p.fly!=null ? p.fly.toFixed(3) : "—"}</b>
          · z = <b style="color:${Math.abs(p.fly_z||0)>=1.5 ? C.accent : C.text}">${p.fly_z!=null ? (p.fly_z>=0?"+":"") + p.fly_z.toFixed(1) + "σ" : "—"}</b></div>
        <div class="mp-spreads">${sprStr}</div>
        <div class="mp-macros">${macros}</div>
      </div>
      <div class="mp-col mp-fund">
        <div class="mp-h">Fundamental (Lasso · regime)</div>
        <div class="mp-action" style="color:${verdictColor(fund.action)}">${fund.action||"WATCH"}</div>
        <div class="mp-reason">${fund.reason||""}</div>
        <div class="mp-weights">${weightsBars}</div>
      </div>
      <div class="mp-col mp-tech">
        <div class="mp-h">Technical (BB · MA · momentum)</div>
        <div class="mp-action" style="color:${verdictColor(tech.action)}">${tech.action||"WATCH"}</div>
        <div class="mp-reason">${tech.reason||""}</div>
      </div>
      <div class="mp-col mp-verdict">
        <div class="mp-h">Combined Verdict</div>
        <div class="mp-action mp-verdict-action"
             style="color:${verdictColor(verdict.verdict)}; font-size:18px">${verdict.verdict||"WATCH"}</div>
        <div class="mp-trade-size">${verdict.contracts||0} contracts
          ${verdict.notional_usd ? `· ≈ $${(verdict.notional_usd).toLocaleString()}` : ""}</div>
        <div class="mp-reason">${verdict.rationale||""}</div>
      </div>
    </div>`;
  }).join("");

  $("multiProductMatrix").innerHTML = rows;
}

/* ---------- P2A Regime Fingerprint ---------- */
function renderP2Fingerprint(r) {
  if (!r) { $("p2RegimeLabel").textContent = "no data"; return; }
  $("p2RegimeLabel").textContent = r.fingerprint_label || "—";
  const dims = r.dimensions || [];
  const dimColor = (label) => {
    if (!label) return C.dim;
    if (label.includes("Very Low") || label.includes("Low ")) return C.up;
    if (label.includes("Very High") || label.includes("High ")) return C.down;
    if (label.includes("Steep Backwardation")) return "#ff5e7e";
    if (label.includes("Steep Contango")) return "#4aa3df";
    if (label.includes("Backwardation")) return "#f0a0b0";
    if (label.includes("Contango")) return "#7ad7ff";
    if (label.includes("Summer Driving")) return C.accent;
    if (label.includes("Winter Heating")) return C.blue;
    if (label.includes("Refinery Turnaround")) return C.up;
    if (label.includes("Strong Dollar") || label.includes("Spiking Dollar")) return C.down;
    if (label.includes("Weak Dollar") || label.includes("Crashing Dollar")) return C.up;
    return C.text;
  };
  $("p2RegimeFingerprint").innerHTML = dims.map(d => `
    <div class="p2-dim">
      <div class="p2-dim-name">${d.name}</div>
      <div class="p2-dim-label" style="color:${dimColor(d.label)}">${d.label}</div>
      <div class="p2-dim-metric">${d.metric}: <b>${d.value}</b></div>
      <div class="p2-dim-bucket">bucket ${d.bucket}</div>
    </div>`).join("");
}

/* ---------- PERCENTILE REGIME — per-product quintile labels ---------- */
function renderPctRegime(p) {
  const el = $("pctRegimeTable");
  const status = $("pctRegimeStatus");
  if (!p || !p.available) {
    status.textContent = (p && p.reason) ? p.reason : "no data";
    el.innerHTML = "";
    return;
  }
  status.textContent =
    `${p.n_total} products · ${p.n_agreement}/${p.n_total} schemes agree ordinally`;
  status.className = "tag mini good";

  // Color the percentile quintile (Q1 = backwardation = red, Q5 = contango = green)
  const qColor = (idx) => {
    const colors = [C.down, "#ff8a4d", C.dim, "#7cc26a", C.up];
    return colors[idx] || C.dim;
  };

  let html = `<div class="pr-header">
    <div>Product</div><div>Slope (M12−M1)</div><div>Hard Label</div>
    <div>Percentile Label</div><div>Agree?</div><div>Cutoffs (p20/p40/p60/p80)</div>
  </div>`;
  (p.products || []).forEach(x => {
    const rowClass = !x.agreement_ordinal ? "pr-row pr-row-disagree" : "pr-row";
    const slopeStr = (x.current_slope >= 0 ? "+" : "") + Number(x.current_slope).toFixed(2);
    const agreeMark = x.agreement_ordinal
      ? '<span style="color:' + C.up + '">YES</span>'
      : '<span style="color:' + C.accent + ';font-weight:700">NO</span>';
    const cutoffs = `${x.p20.toFixed(2)} / ${x.p40.toFixed(2)} / ${x.p60.toFixed(2)} / ${x.p80.toFixed(2)}`;
    html += `<div class="${rowClass}">
      <div><b>${x.product}</b></div>
      <div style="font-family:var(--mono)">${slopeStr}</div>
      <div>${x.hard_label}</div>
      <div style="color:${qColor(x.percentile_idx)};font-weight:700">${x.percentile_label}</div>
      <div>${agreeMark}</div>
      <div class="pr-sub">${cutoffs}</div>
    </div>`;
  });
  el.innerHTML = html;
}

/* ---------- PAPER STRATEGIES: PCA Curve / Bertram OU / HMM Regime ---------- */
function renderPStrat(p) {
  const el = $("pstratTable");
  const status = $("pstratStatus");
  if (!p || !p.available) {
    status.textContent = (p && p.reason) ? p.reason : "no data";
    el.innerHTML = "";
    return;
  }
  status.textContent =
    `${p.n_total} engines · ${p.n_long} LONG · ${p.n_short} SHORT · ${p.n_monitor} monitoring`;
  status.className = "tag mini good";

  const sigColor = (v) =>
    v === "LONG" ? C.up : v === "SHORT" ? C.down : C.dim;
  const confTag = (c) =>
    c === "HIGH" ? '<span class="ls-conf ls-conf-hi">HIGH</span>' :
    c === "MED"  ? '<span class="ls-conf ls-conf-md">MED</span>' :
    c === "MONITOR" ? '<span class="ls-conf ls-conf-lo">MON</span>' :
                      '<span class="ls-conf ls-conf-lo">LOW</span>';

  let html = `<div class="ps-header">
    <div>Engine</div><div>Product</div><div>Instrument</div>
    <div>Signal</div><div>Reading</div><div>Conf</div>
  </div>`;
  (p.signals || []).forEach(s => {
    const rowClass = (s.direction !== "FLAT" && s.confidence === "HIGH")
                      ? "ps-row ps-row-hi" : "ps-row";
    let reading = "";
    if (s.pc3_z != null) reading = `PC3 z=${Number(s.pc3_z).toFixed(2)}`;
    else if (s.z_score != null) {
      reading = `z=${Number(s.z_score).toFixed(2)}`;
      if (s.p_high_regime != null) reading += ` · P(high)=${(s.p_high_regime*100).toFixed(0)}%`;
    } else if (s.deviation != null) {
      reading = `dev=${Number(s.deviation).toFixed(3)}`;
      if (s.pct_to_entry != null) reading += ` · ${s.pct_to_entry.toFixed(0)}% to a*`;
    }
    html += `<div class="${rowClass}">
      <div><b>${s.engine}</b></div>
      <div>${s.product}</div>
      <div class="ps-sub">${s.instrument}</div>
      <div style="color:${sigColor(s.direction)};font-weight:700">${s.direction}</div>
      <div class="ps-sub">${reading}</div>
      <div>${confTag(s.confidence)}</div>
    </div>`;
  });
  el.innerHTML = html;
}

/* ---------- BEST-PER-PRODUCT: specialist model per product ---------- */
function renderBPP(p) {
  const el = $("bppTable");
  const status = $("bppStatus");
  if (!p || !p.available) {
    status.textContent = (p && p.reason) ? p.reason : "no data";
    el.innerHTML = "";
    return;
  }
  const products = p.per_product || {};
  const keys = Object.keys(products);
  status.textContent = `${keys.length} products · ${p.n_total} actionable trades`;
  status.className = "tag mini good";

  const sigColor = (v) =>
    v === "LONG" ? C.up : v === "SHORT" ? C.down : C.dim;
  const confTag = (c) =>
    c === "HIGH" ? '<span class="ls-conf ls-conf-hi">HIGH</span>' :
    c === "MED"  ? '<span class="ls-conf ls-conf-md">MED</span>' :
                   '<span class="ls-conf ls-conf-lo">LOW</span>';

  let html = `<div class="bpp-header">
    <div>Product</div><div>Specialist Model</div><div>Win Rate</div>
    <div>Active Trades</div>
  </div>`;
  keys.forEach(prod => {
    const d = products[prod];
    const wr = d.total_targets > 0
      ? Math.round(d.wins_by_model / d.total_targets * 100)
      : 0;
    const tradeRows = (d.trades || []).map(t => {
      const moveStr = (t.predicted_5d >= 0 ? "+" : "") + Number(t.predicted_5d).toFixed(3);
      const haircut = t.we_haircut_pct ?? 0;
      const r2Display = haircut > 0
        ? `R²=${(t.test_r2 || 0).toFixed(2)}→${(t.haircut_r2 || 0).toFixed(2)} <span class="we-${haircut <= 10 ? "mild" : haircut <= 20 ? "med" : "bad"}" title="Working effect lag-1 ${(t.we_lag1 || 0).toFixed(3)}, ~${haircut}% R² mechanical">⚠${haircut}%</span>`
        : `R²=${(t.test_r2 || 0).toFixed(2)} <span class="we-clean" title="No Working effect">✓</span>`;
      return `<div class="bpp-trade">
        <span class="bpp-trade-label">${t.label}</span>
        <span style="color:${sigColor(t.signal)};font-weight:700">${t.signal}</span>
        <span class="bpp-sub">${moveStr} · ${r2Display}</span>
        ${confTag(t.confidence)}
      </div>`;
    }).join("") || `<div class="bpp-sub">no trades pass R²≥0.10</div>`;
    html += `<div class="bpp-row">
      <div><b>${prod}</b></div>
      <div>${d.best_model}<br><span class="bpp-sub">${d.wins_by_model}/${d.total_targets} wins</span></div>
      <div><b>${wr}%</b><br><span class="bpp-sub">of targets won</span></div>
      <div class="bpp-trades">${tradeRows}</div>
    </div>`;
  });
  el.innerHTML = html;
}

/* ---------- COMPOSITE multi-factor strategy ---------- */
function renderComposite(c) {
  const el = $("compositeTable");
  const status = $("compositeStatus");
  if (!c || !c.available) {
    status.textContent = (c && c.reason) ? c.reason : "no data";
    el.innerHTML = "";
    return;
  }
  status.textContent =
    `${c.products.length} products · ${c.n_long} LONG · ${c.n_short} SHORT · ${c.n_flat} FLAT`;
  status.className = "tag mini good";

  const sigColor = (v) =>
    v === "LONG" ? C.up : v === "SHORT" ? C.down : C.dim;
  const confTag = (cv) =>
    cv === "HIGH" ? '<span class="ls-conf ls-conf-hi">HIGH</span>' :
    cv === "MED"  ? '<span class="ls-conf ls-conf-md">MED</span>' :
                    '<span class="ls-conf ls-conf-lo">LOW</span>';

  let html = `<div class="cm-header">
    <div>Product</div><div>Score</div><div>Verdict</div><div>Conv</div>
    <div>Factor votes (engine × weight)</div>
  </div>`;
  c.products.forEach(p => {
    const rowClass = (p.verdict !== "FLAT" && p.conviction === "HIGH") ? "cm-row cm-row-hi" : "cm-row";
    // Bar visualization of the composite score (-1 to +1, center at 0)
    const score = Number(p.composite_score);
    const pct = Math.min(100, Math.max(0, (score + 1) * 50));
    const barColor = score > 0 ? C.up : score < 0 ? C.down : C.dim;
    const facStr = (p.factors || []).map(f => {
      const v = Number(f.vote);
      const color = v > 0 ? C.up : v < 0 ? C.down : C.dim;
      const sign = v > 0 ? "+" : v < 0 ? "" : "";
      return `<span style="color:${color}">${f.engine}=${sign}${v.toFixed(1)}</span>`;
    }).join(" · ");
    html += `<div class="${rowClass}">
      <div><b>${p.name}</b></div>
      <div>
        <div class="cm-score-track">
          <div class="cm-score-fill" style="left:${pct}%;background:${barColor}"></div>
        </div>
        <div class="cm-sub" style="color:${barColor}">${score >= 0 ? "+" : ""}${score.toFixed(2)}</div>
      </div>
      <div style="color:${sigColor(p.verdict)};font-weight:700">${p.verdict}</div>
      <div>${confTag(p.conviction)}</div>
      <div class="cm-sub">${facStr}</div>
    </div>`;
  });
  el.innerHTML = html;
}

/* ---------- REAL-Curve Signals — from user's xlsx files ---------- */
function renderRealData(r) {
  const el = $("realDataTable");
  const status = $("realDataStatus");
  if (!r || !r.available) {
    status.textContent = (r && r.reason) ? r.reason : "no data";
    el.innerHTML = "";
    return;
  }
  const vpct = r.verify && r.verify.match_pct != null ? r.verify.match_pct : null;
  status.textContent =
    `${r.total} targets · ${r.actionable} actionable · HIGH:${r.by_conf.HIGH} ` +
    `MED:${r.by_conf.MED} LOW:${r.by_conf.LOW}` +
    (vpct != null ? ` · file⇆dash verify: ${vpct}%` : "");
  status.className = "tag mini good";

  // Filter: only show HIGH and MED confidence (the actionable set)
  const shown = (r.signals || []).filter(s =>
    (s.confidence === "HIGH" || s.confidence === "MED") &&
    s.signal !== "FLAT").slice(0, 25);

  const sigColor = (sig) =>
    sig === "LONG" ? C.up : sig === "SHORT" ? C.down : C.dim;
  const confTag = (c) =>
    c === "HIGH" ? '<span class="ls-conf ls-conf-hi">HIGH</span>' :
    c === "MED"  ? '<span class="ls-conf ls-conf-md">MED</span>' :
                   '<span class="ls-conf ls-conf-lo">' + c + '</span>';
  // Working-effect tag: shows lag-1 diff autocorr + haircut % when material.
  // Clean signals (haircut=0%) get a plain check; flagged ones get a warning.
  const weTag = (lag1, haircut) => {
    if (lag1 == null) return "";
    if (haircut === 0) return '<span class="we-clean" title="No Working effect detected (lag-1 autocorr ' + lag1.toFixed(3) + ')">&#10003; clean</span>';
    if (haircut <= 10) return '<span class="we-mild" title="Mild Working effect: lag-1 ' + lag1.toFixed(3) + ', ~' + haircut + '% R² inflated">&#9888; WE ' + lag1.toFixed(2) + '</span>';
    if (haircut <= 20) return '<span class="we-med" title="Material Working effect: ~' + haircut + '% R² mechanical">&#9888; WE ' + lag1.toFixed(2) + ' (-' + haircut + '%)</span>';
    return '<span class="we-bad" title="Severe Working effect: ~' + haircut + '% R² mechanical, live result will be much smaller">&#9888; WE ' + lag1.toFixed(2) + ' (-' + haircut + '%)</span>';
  };

  let html = `<div class="rd-header">
    <div>Product</div><div>Target</div><div>Kind</div><div>Current</div>
    <div>Pred 5d</div><div>Signal</div><div>Model</div><div>Confidence + WE</div>
  </div>`;
  shown.forEach(s => {
    const rowClass = s.confidence === "HIGH" ? "rd-row rd-row-hi" : "rd-row";
    const moveStr = (s.predicted_5d >= 0 ? "+" : "") +
                    Number(s.predicted_5d).toFixed(3);
    const rawR2 = (s.test_r2 ?? 0).toFixed(2);
    const haircut = s.we_haircut_pct ?? 0;
    const r2Display = haircut > 0
      ? `R²=${rawR2}<span class="rd-sub" style="margin-left:3px">→${(s.haircut_r2 ?? 0).toFixed(2)}</span>`
      : `R²=${rawR2}`;
    html += `<div class="${rowClass}">
      <div><b>${s.product}</b></div>
      <div>${s.label.replace(s.product + ' ', '')}</div>
      <div class="rd-sub">${s.kind}</div>
      <div>${Number(s.current_level).toFixed(3)}</div>
      <div>${moveStr}</div>
      <div style="color:${sigColor(s.signal)};font-weight:700">${s.signal}</div>
      <div class="rd-sub">${s.winner_model}</div>
      <div>${confTag(s.confidence)}<span class="rd-sub"> ${r2Display}</span> ${weTag(s.we_lag1, haircut)}</div>
    </div>`;
  });
  if (shown.length === 0) {
    html += `<div class="rd-row"><div style="grid-column:span 8;color:var(--dim)">No HIGH/MED-confidence actionable signals right now.</div></div>`;
  }
  el.innerHTML = html;
}

/* ---------- LIVE Signals — 5 products + 3 spreads ---------- */
function renderLiveSignals(s) {
  const el = $("liveSignalsTable");
  const status = $("liveSignalsStatus");
  if (!s || !s.available) {
    status.textContent = (s && s.reason) ? s.reason : "no data";
    el.innerHTML = "";
    return;
  }
  const rows = s.signals || [];
  const hi  = rows.filter(r => r.signal !== "FLAT" && r.confidence === "HIGH").length;
  const med = rows.filter(r => r.signal !== "FLAT" && r.confidence === "MED").length;
  status.textContent = `${rows.length} instruments · ${hi} HIGH-conf · ${med} MED-conf · ${s.horizon_days}d horizon`;
  status.className = "tag mini good";

  const sigColor = (sig) =>
    sig === "LONG" ? C.up :
    sig === "SHORT" ? C.down : C.dim;
  const confTag = (c) =>
    c === "HIGH" ? '<span class="ls-conf ls-conf-hi">HIGH</span>' :
    c === "MED"  ? '<span class="ls-conf ls-conf-md">MED</span>' :
                   '<span class="ls-conf ls-conf-lo">LOW</span>';

  let html = `<div class="ls-header">
    <div>Instrument</div><div>Current</div><div>Signal</div>
    <div>Pred 5d</div><div>Confidence</div><div>Hist Edge</div>
  </div>`;
  rows.forEach(r => {
    const rowClass = (r.signal !== "FLAT" && r.confidence === "HIGH") ? "ls-row ls-row-hi" : "ls-row";
    const isSpread = r.kind === "diff";
    const curStr = isSpread
      ? r.current_level.toFixed(2)
      : "$" + r.current_level.toFixed(2);
    const moveStr = isSpread
      ? (r.predicted_move >= 0 ? "+" : "") + r.predicted_move.toFixed(2)
      : (r.predicted_move >= 0 ? "+" : "") + r.predicted_move.toFixed(2) + "%";
    const edgeStr = r.hist_n_trades > 0
      ? `Sharpe ${r.hist_sharpe >= 0 ? "+" : ""}${r.hist_sharpe.toFixed(2)} · ${r.hist_win_rate.toFixed(0)}% wins · ${r.hist_n_trades} tr`
      : "no backtest";
    html += `<div class="${rowClass}">
      <div><b>${r.label}</b>${isSpread ? ' <span class="ls-spread-tag">SPREAD</span>' : ''}</div>
      <div>${curStr}</div>
      <div style="color:${sigColor(r.signal)};font-weight:700">${r.signal}</div>
      <div>${moveStr}</div>
      <div>${confTag(r.confidence)}</div>
      <div class="ls-sub">${edgeStr}</div>
    </div>`;
  });
  el.innerHTML = html;
}

/* ---------- P2B Model Comparison ---------- */
function renderP2Models(m) {
  if (!m) { $("p2ModelStatus").textContent = "no data"; return; }
  if (!m.available) {
    $("p2ModelStatus").textContent = m.reason || "unavailable";
    $("p2ModelTable").innerHTML = "";
    return;
  }
  $("p2ModelStatus").textContent =
    `n=${m.n_total_samples} samples · current regime: ${m.current_regime}`;
  $("p2ModelStatus").className = "tag mini good";

  const regimes = m.regimes || [];
  const families = ["Linear", "Ridge", "Lasso", "ElasticNet", "Huber", "LGBM"];

  let html = `<div class="p2-mt-header">
    <div>Regime</div><div>n</div>`;
  families.forEach(f => { html += `<div>${f}<br><span class="p2-mt-sub">train R² / test R² / MAE / nz</span></div>`; });
  html += `<div>Winner</div></div>`;

  regimes.forEach(r => {
    const isCurrent = r.regime_idx === m.current_regime_idx;
    const rowClass = isCurrent ? "p2-mt-row p2-mt-current" : "p2-mt-row";
    html += `<div class="${rowClass}">
      <div><b>${r.regime}</b>${isCurrent ? ' <span class="p2-cur-tag">current</span>' : ''}</div>
      <div>${r.n_samples || 0}</div>`;
    families.forEach(f => {
      const mi = (r.models || {})[f];
      if (!mi) {
        html += `<div class="p2-mt-empty">—</div>`;
      } else {
        const isWinner = r.winner === f;
        const winClass = isWinner ? "p2-mt-cell p2-mt-winner" : "p2-mt-cell";
        html += `<div class="${winClass}">
          <div>${mi.train_r2 != null ? mi.train_r2.toFixed(2) : "—"}
            / <b>${mi.test_r2 != null ? mi.test_r2.toFixed(2) : "—"}</b></div>
          <div class="p2-mt-sub">MAE ${mi.test_mae != null ? mi.test_mae.toFixed(3) : "—"}
            · nz=${mi.n_nonzero}/${(mi.coefs||[]).length}</div>
        </div>`;
      }
    });
    html += `<div class="p2-mt-winner-cell">${r.winner || "—"}
      ${r.winner_score != null ? `<span class="p2-mt-sub">(${r.winner_score.toFixed(2)})</span>` : ''}</div>
    </div>`;
  });

  $("p2ModelTable").innerHTML = html;
}

/* ---------- P2C Opportunities ---------- */
function renderP2Opportunities(o) {
  if (!o) { $("p2OppCount").textContent = "no data"; return; }
  if (!o.available) {
    $("p2OppCount").textContent = o.reason || "unavailable";
    $("p2OpportunityTable").innerHTML = "";
    return;
  }
  const top = o.ranked_top || [];
  $("p2OppCount").textContent =
    `${(o.opportunities || []).length} pairs scanned · top ${top.length} shown`;
  $("p2OppCount").className = "tag mini good";

  const directionColor = (dir) =>
    dir === "LONG" ? C.up : dir === "SHORT" ? C.down : C.dim;

  let html = `<div class="p2-opp-header">
    <div>Rank</div><div>Product</div><div>Structure</div>
    <div>Actual</div><div>Regime mean</div><div>z avg</div>
    <div>Conf</div><div>Robust</div><div>Score</div><div>Action</div>
  </div>`;

  top.forEach((op, idx) => {
    html += `<div class="p2-opp-row">
      <div class="p2-opp-rank">${idx + 1}</div>
      <div>${op.product}</div>
      <div>${op.structure_label}</div>
      <div>${op.actual}</div>
      <div>${op.expected != null ? op.expected : '—'}</div>
      <div style="color:${Math.abs(op.z_avg || 0) >= 1.5 ? C.accent : C.text}">
        ${op.z_avg != null ? (op.z_avg >= 0 ? '+' : '') + op.z_avg.toFixed(2) : '—'}</div>
      <div>${(op.confidence * 100).toFixed(0)}%</div>
      <div>${(op.robustness * 100).toFixed(0)}%</div>
      <div><b>${op.score.toFixed(2)}</b></div>
      <div style="color:${directionColor(op.direction)}; font-weight:700">${op.direction}</div>
    </div>
    <div class="p2-opp-rationale">${op.rationale}</div>`;
  });

  $("p2OpportunityTable").innerHTML = html;
}

/* ---------- P2D Historical Regime DB ---------- */
function renderP2History(h) {
  if (!h) { $("p2DbStatus").textContent = "no data"; return; }
  if (!h.available) {
    $("p2DbStatus").textContent = h.reason || "unavailable";
    $("p2HistoryTable").innerHTML = "";
    return;
  }
  $("p2DbStatus").textContent = `n=${h.n_total} historical observations`;
  $("p2DbStatus").className = "tag mini good";

  const dims = h.per_dim || [];
  let html = "";
  dims.forEach(dim => {
    html += `<div class="p2-hist-dim">
      <div class="p2-hist-dim-name">${dim.dimension}</div>
      <div class="p2-hist-buckets">`;
    (dim.buckets || []).forEach(b => {
      const fly = b.fly || {};
      html += `<div class="p2-hist-bucket">
        <div class="p2-hist-bucket-label">${b.label}</div>
        <div class="p2-hist-bucket-n">n=${fly.n || 0}</div>
        <div class="p2-hist-bucket-stats">
          fly: μ=${fly.mean != null ? fly.mean : '—'} σ=${fly.std != null ? fly.std : '—'}<br>
          p10/p50/p90: ${fly.p10 != null ? fly.p10 : '—'}/${fly.p50 != null ? fly.p50 : '—'}/${fly.p90 != null ? fly.p90 : '—'}
        </div>
      </div>`;
    });
    html += `</div></div>`;
  });

  $("p2HistoryTable").innerHTML = html;
}

/* ---------- P3 Term-Structure dashboard ---------- */
function _tsConfColor(conf) {
  return conf === "high" ? "#4caf7e"
       : conf === "med"  ? "#e0b34a"
       : conf === "low"  ? "#d88a3f"
       :                   "#e2616b";
}
function _tsDirColor(dir) {
  return dir === "LONG" ? C.up : dir === "SHORT" ? C.down : C.dim;
}
function _tsRegimeColor(r) {
  if (r && r.includes("backwardation")) return C.up;
  if (r && r.includes("contango"))      return C.down;
  return C.dim;
}

/* P3A: per-product R² quality grid (top 10 level models). */
function renderTSQuality(ts) {
  const el = $("tsQualityGrid"); if (!el) return;
  if (!ts || !ts.available) {
    $("tsQuality").textContent = "no data";
    el.innerHTML = `<div class="ts-empty">Term-structure regression panel not available.</div>`;
    return;
  }
  const q = ts.quality_counts || {};
  $("tsQuality").textContent =
    `${ts.n_total_signals} signals · ${q.high||0} high · ${q.med||0} med · ${q.low||0} low conf`;
  $("tsQuality").className = "tag mini good";

  let html = "";
  Object.entries(ts.products || {}).forEach(([code, info]) => {
    const lvls = (info.spreads || []).concat(info.flies || [])
      .filter(s => s.kind === "level")
      .sort((a,b) => b.r2 - a.r2)
      .slice(0, 10);
    html += `<div class="ts-prod-card">
      <div class="ts-prod-head">
        <div class="ts-prod-name">${info.name}</div>
        <div class="ts-prod-regime" style="color:${_tsRegimeColor(info.regime)}">
          ${(info.regime||"unknown").replace(/_/g," ")}
          ${info.slope != null ? `<span class="ts-slope">slope ${info.slope >= 0 ? '+' : ''}${info.slope}</span>` : ""}
        </div>
      </div>
      <div class="ts-r2-list">`;
    lvls.forEach(s => {
      html += `<div class="ts-r2-row">
        <div class="ts-r2-label">${s.label}</div>
        <div class="ts-r2-model">${s.model}</div>
        <div class="ts-r2-val" style="color:${_tsConfColor(s.conf)}">${s.r2.toFixed(3)}</div>
      </div>`;
    });
    html += `</div></div>`;
  });
  el.innerHTML = html;
}

/* P3B: cross-product top trade ideas. */
function renderTSCrossTop(ts) {
  const el = $("tsCrossTopTable"); if (!el) return;
  if (!ts || !ts.available) {
    $("tsCrossTopCount").textContent = "no data";
    el.innerHTML = "";
    return;
  }
  const top = ts.cross_product_top || [];
  $("tsCrossTopCount").textContent = `${top.length} ideas · ${ts.n_actionable} total actionable`;
  $("tsCrossTopCount").className = top.length ? "tag mini good" : "tag mini";

  let html = `<div class="ts-top-header">
    <div>#</div><div>Product</div><div>Structure</div><div>Kind</div>
    <div>Model</div><div>R²</div><div>Actual</div><div>Pred</div>
    <div>z</div><div>Direction</div><div>Lots</div>
  </div>`;
  top.forEach((s, ix) => {
    html += `<div class="ts-top-row">
      <div>${ix + 1}</div>
      <div>${s.product}</div>
      <div><b>${s.label}</b></div>
      <div class="ts-kind">${s.kind}</div>
      <div>${s.model}</div>
      <div style="color:${_tsConfColor(s.conf)}">${s.r2.toFixed(3)}</div>
      <div>${s.actual}</div>
      <div>${s.predicted}</div>
      <div style="color:${Math.abs(s.z) >= 1.5 ? C.accent : C.text}">${s.z >= 0 ? '+' : ''}${s.z}</div>
      <div style="color:${_tsDirColor(s.direction)}; font-weight:700">${s.direction}</div>
      <div><b>${s.lots}</b></div>
    </div>`;
  });
  if (top.length === 0) {
    html += `<div class="ts-empty">No actionable signals (all spreads/flies within ±0.75σ of model expectation, or R² &lt; 0.30). Hold positions.</div>`;
  }
  el.innerHTML = html;
}

/* P3C: per-product card with regime + top 5 ideas. */
function renderTSByProduct(ts) {
  const el = $("tsByProduct"); if (!el) return;
  if (!ts || !ts.available) {
    $("tsRegimeCount").textContent = "no data";
    el.innerHTML = "";
    return;
  }
  const products = ts.products || {};
  $("tsRegimeCount").textContent = `${Object.keys(products).length} products`;
  $("tsRegimeCount").className = "tag mini good";

  let html = "";
  Object.entries(products).forEach(([code, info]) => {
    const ideas = info.best_ideas || [];
    html += `<div class="ts-regime-card">
      <div class="ts-regime-head">
        <div class="ts-regime-prod">${info.name}</div>
        <div class="ts-regime-tag" style="color:${_tsRegimeColor(info.regime)}; border-color:${_tsRegimeColor(info.regime)}">
          ${(info.regime||"unknown").replace(/_/g," ")}
        </div>
        <div class="ts-regime-meta">
          ${info.n_actionable}/${info.n_spreads + info.n_flies} actionable
          ${info.slope != null ? `· slope ${info.slope >= 0 ? '+' : ''}${info.slope}` : ""}
        </div>
      </div>`;
    if (ideas.length === 0) {
      html += `<div class="ts-regime-empty">No actionable spread / fly trades — all structures within model tolerance.</div>`;
    } else {
      html += `<div class="ts-regime-ideas">`;
      ideas.forEach(idea => {
        const dollarMove = (Math.abs(idea.residual) * (info.contract_per_dollar || 1000)).toFixed(0);
        html += `<div class="ts-idea-row">
          <div class="ts-idea-struct"><b>${idea.label}</b> <span class="ts-idea-kind">(${idea.kind})</span></div>
          <div class="ts-idea-action" style="color:${_tsDirColor(idea.direction)}; font-weight:700">
            ${idea.direction} ${idea.lots} lot${idea.lots !== 1 ? "s" : ""}
          </div>
          <div class="ts-idea-stats">
            ${idea.model} · R²=${idea.r2.toFixed(2)} · z=${idea.z >= 0 ? '+' : ''}${idea.z}
            <br><span class="dim">actual ${idea.actual} vs model ${idea.predicted} (residual ${idea.residual >= 0 ? '+' : ''}${idea.residual})</span>
          </div>
          <div class="ts-idea-pnl">≈ \$${dollarMove}<br><span class="dim">per lot to revert</span></div>
        </div>`;
      });
      html += `</div>`;
    }
    html += `</div>`;
  });
  el.innerHTML = html;
}

/* P3D: full strip table. */
function renderTSFull(ts) {
  const el = $("tsFullTable"); if (!el) return;
  if (!ts || !ts.available) {
    $("tsFullCount").textContent = "no data";
    el.innerHTML = "";
    return;
  }
  // Flatten all signals.
  const rows = [];
  Object.entries(ts.products || {}).forEach(([code, info]) => {
    (info.spreads || []).forEach(s => rows.push({...s, product_name: info.name}));
    (info.flies || []).forEach(s => rows.push({...s, product_name: info.name}));
  });
  rows.sort((a,b) => Math.abs(b.z) * b.r2 - Math.abs(a.z) * a.r2);

  $("tsFullCount").textContent = `${rows.length} rows`;
  $("tsFullCount").className = "tag mini";

  let html = `<div class="ts-full-header">
    <div>Product</div><div>Structure</div><div>Kind</div>
    <div>Model</div><div>R²</div><div>RMSE</div>
    <div>Actual</div><div>Pred</div><div>z</div><div>Dir</div><div>Lots</div>
  </div>`;
  rows.slice(0, 100).forEach(s => {
    html += `<div class="ts-full-row">
      <div>${s.product}</div>
      <div>${s.label}</div>
      <div class="ts-kind">${s.kind}</div>
      <div>${s.model}</div>
      <div style="color:${_tsConfColor(s.conf)}">${s.r2.toFixed(2)}</div>
      <div>${s.rmse}</div>
      <div>${s.actual}</div>
      <div>${s.predicted}</div>
      <div>${s.z >= 0 ? '+' : ''}${s.z}</div>
      <div style="color:${_tsDirColor(s.direction)}; font-weight:600">${s.direction}</div>
      <div>${s.lots || ""}</div>
    </div>`;
  });
  if (rows.length > 100) {
    html += `<div class="ts-full-footer">… ${rows.length - 100} more rows (sorted by |z| × R²)</div>`;
  }
  el.innerHTML = html;
}

/* ---------- P6 Strategy backtest (split-period) ---------- */
fetch("data/strategy_backtest.json")
  .then(r => r.json())
  .then(d => renderBacktest(d))
  .catch(e => console.warn("backtest data fetch failed:", e));

function renderBacktest(bt) {
  const el = $("p6Table"); if (!el) return;
  if (!bt || !bt.results) {
    $("p6Meta").textContent = "no data";
    el.innerHTML = "";
    return;
  }
  const meta = $("p6Meta");
  if (meta) {
    meta.textContent =
      `split ${bt.split_date} · ${bt.fetch_period} history · ${bt.cost_bps}bps cost`;
    meta.className = "tag mini good";
  }

  const PROD_NAMES = {
    wti: "WTI", brent: "Brent", rbob: "RBOB", heat: "Heating Oil", natgas: "NatGas",
  };
  const STRATS = [
    {key: "ta",          label: "TA (base)"},
    {key: "ta_trend",    label: "TA + 200-SMA trend filter"},
    {key: "ta_robust",   label: "TA + trend + vol gate + lot cap"},
    {key: "ta_smart",    label: "TA regime-adaptive (trend vs range)"},
    {key: "kalman_pair", label: "Kalman pair-trade (dynamic β residual)"},
    {key: "combined",    label: "TA + Seasonality"},
    {key: "bh",          label: "Buy&Hold"},
  ];
  const colorReturn = (r) =>
    r >= 5  ? "#4caf7e"
    : r > 0  ? "#a8d8a8"
    : r <= -10 ? "#e2616b"
    : r < 0  ? "#e09090" : C.dim;
  const colorSharpe = (s) =>
    s >= 0.5 ? "#4caf7e" : s > 0 ? C.accent : s < 0 ? "#e2616b" : C.dim;
  const colorDD = (d) =>
    d <= 5 ? "#4caf7e" : d <= 25 ? C.accent : "#e2616b";

  let html = `<div class="p6-head">
    <div>Product</div>
    <div>Strategy</div>
    <div>n trades</div>
    <div>Win rate</div>
    <div>In-sample return</div>
    <div>OOS return</div>
    <div>In-sample DD</div>
    <div>OOS DD</div>
    <div>OOS Sharpe</div>
    <div>OOS P&amp;L</div>
  </div>`;

  Object.entries(bt.results).forEach(([prodKey, byStrat]) => {
    STRATS.forEach((s) => {
      const ins  = byStrat[s.key]?.in_sample || {};
      const oos  = byStrat[s.key]?.out_of_sample || {};
      const isReturn = ins.return_pct ?? 0;
      const oosReturn = oos.return_pct ?? 0;
      const oosSharpe = oos.sharpe ?? 0;
      html += `<div class="p6-row">
        <div class="p6-prod">${PROD_NAMES[prodKey] || prodKey}</div>
        <div class="p6-strat">${s.label}</div>
        <div class="p6-px">${ins.n_trades ?? 0} / ${oos.n_trades ?? 0}</div>
        <div class="p6-px">${(ins.win_rate_pct ?? 0).toFixed(0)}% / ${(oos.win_rate_pct ?? 0).toFixed(0)}%</div>
        <div class="p6-px" style="color:${colorReturn(isReturn)}; font-weight:600">${isReturn >= 0 ? "+" : ""}${isReturn.toFixed(1)}%</div>
        <div class="p6-px" style="color:${colorReturn(oosReturn)}; font-weight:700">${oosReturn >= 0 ? "+" : ""}${oosReturn.toFixed(1)}%</div>
        <div class="p6-px" style="color:${colorDD(ins.max_dd_pct ?? 0)}">${(ins.max_dd_pct ?? 0).toFixed(1)}%</div>
        <div class="p6-px" style="color:${colorDD(oos.max_dd_pct ?? 0)}">${(oos.max_dd_pct ?? 0).toFixed(1)}%</div>
        <div class="p6-px" style="color:${colorSharpe(oosSharpe)}; font-weight:600">${oosSharpe.toFixed(2)}</div>
        <div class="p6-px" style="color:${colorReturn(oosReturn)}">$${Math.round(oos.pnl ?? 0).toLocaleString()}</div>
      </div>`;
    });
  });

  el.innerHTML = html;
}

/* ---------- P4 Regime detection (HMM/GMM/KMeans + STL) ---------- */
let _REGIME_DETECTION = null;
fetch("data/regime_detection.json")
  .then(r => r.json())
  .then(d => { _REGIME_DETECTION = d; renderRegimeDetection(d); })
  .catch(e => console.warn("regime detection data fetch failed:", e));

function renderRegimeDetection(rd) {
  if (!rd || !rd.products) return;

  /* P4A method fit-off */
  const wEl = $("p4Winner");
  if (wEl) {
    wEl.textContent = "winner: " + (rd.verdict?.winner || "—");
    wEl.className = "tag mini good";
  }
  const vEl = $("p4Verdict");
  if (vEl) vEl.innerHTML = `<b>${rd.verdict?.winner || "—"}</b>: ${rd.verdict?.summary || ""}`;

  const mgEl = $("p4MethodGrid");
  if (mgEl) {
    let html = "";
    Object.values(rd.products).forEach((p) => {
      html += `<div class="p4-mp-card">
        <div class="p4-mp-prod">${p.name}
          <span class="dim">n=${p.n} · ${p.date_range || ""}</span></div>
        <div class="p4-mp-row">`;
      ["HMM", "GMM", "KMeans"].forEach((m) => {
        const stat = p.metrics[m] || {};
        const isBest = (m === "HMM");
        html += `<div class="p4-mp-box ${isBest ? "p4-mp-best" : ""}">
          <div class="p4-mp-name">${m}${isBest ? ` <span style="color:${C.up}">★</span>` : ""}</div>
          <div class="p4-mp-stat"><span>silhouette</span><b>${(stat.silhouette ?? 0).toFixed(3)}</b></div>
          <div class="p4-mp-stat"><span>switches</span><b style="color:${stat.switches > 100 ? C.down : C.up}">${stat.switches ?? "—"}</b></div>
          ${stat.persistence != null ? `<div class="p4-mp-stat"><span>persistence</span><b style="color:${C.up}">${stat.persistence.toFixed(3)}</b></div>` : ""}
          ${stat.bic != null ? `<div class="p4-mp-stat"><span>BIC</span><b>${stat.bic.toFixed(0)}</b></div>` : ""}
          ${stat.loglik != null ? `<div class="p4-mp-stat"><span>log-lik</span><b>${stat.loglik.toFixed(0)}</b></div>` : ""}
        </div>`;
      });
      html += `</div></div>`;
    });
    mgEl.innerHTML = html;
  }

  /* P4B HMM state characteristics */
  const sgEl = $("p4StateGrid");
  if (sgEl) {
    let totalStates = 0;
    let html = "";
    Object.values(rd.products).forEach((p) => {
      const states = p.hmm_states || {};
      const curIdx = p.current_state_idx;
      html += `<div class="p4-sg-prod">
        <div class="p4-sg-head">${p.name}
          <span class="dim">3 persistent states · HMM persistence ${p.metrics?.HMM?.persistence?.toFixed(3) || "—"}</span></div>
        <div class="p4-sg-row">`;
      Object.entries(states).forEach(([sIdx, s]) => {
        totalStates++;
        const isCurrent = (parseInt(sIdx) === curIdx);
        html += `<div class="p4-sg-state ${isCurrent ? "p4-sg-current" : ""}" style="border-left-color:${s.color}">
          <div class="p4-sg-header">
            <span class="p4-sg-dot" style="background:${s.color}"></span>
            <span class="p4-sg-label">${s.label}</span>
            ${isCurrent ? `<span class="p4-sg-now">NOW</span>` : ""}
          </div>
          <div class="p4-sg-stats">
            <div><span>share</span><b>${(s.share*100).toFixed(0)}%</b></div>
            <div><span>mean ret</span><b style="color:${s.ret>=0?C.up:C.down}">${(s.ret*100).toFixed(2)}%</b></div>
            <div><span>20d vol</span><b>${(s.vol20*100).toFixed(1)}%</b></div>
            <div><span>mean spread</span><b>${s.spread.toFixed(2)}</b></div>
          </div>
        </div>`;
      });
      html += `</div></div>`;
    });
    sgEl.innerHTML = html;
    if ($("p4StateCount")) {
      $("p4StateCount").textContent = `${totalStates} persistent states across ${Object.keys(rd.products).length} products`;
      $("p4StateCount").className = "tag mini good";
    }
  }

  /* P4D Trade signals (regime drift × seasonality) */
  renderP4Signals(rd);

  /* P4C STL seasonality bars */
  const ssEl = $("p4SeasonalityGrid");
  if (ssEl) {
    const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    let html = "";
    Object.values(rd.products).forEach((p) => {
      const monthly = p.seasonality?.monthly || [];
      const strength = p.seasonality?.strength ?? 0;
      const maxAbs = Math.max(...monthly.map(Math.abs), 0.001);
      html += `<div class="p4-seas-card">
        <div class="p4-seas-head">${p.name}
          <span class="dim">seasonality strength ${strength.toFixed(3)} / 1.0
            ${strength < 0.15 ? "(weak — global benchmark)" : "(meaningful)"}</span>
        </div>
        <div class="p4-seas-bars">`;
      monthly.forEach((v, i) => {
        const pct = Math.abs(v) / maxAbs * 100;
        const isPos = v >= 0;
        html += `<div class="p4-seas-month">
          <div class="p4-seas-mname">${months[i]}</div>
          <div class="p4-seas-bar-wrap">
            <div class="p4-seas-bar" style="width:${pct}%; background:${isPos?C.up:C.down}"></div>
          </div>
          <div class="p4-seas-val" style="color:${isPos?C.up:C.down}">${isPos?"+":""}${v.toFixed(1)}</div>
        </div>`;
      });
      html += `</div></div>`;
    });
    ssEl.innerHTML = html;
    if ($("p4SeasCount")) {
      $("p4SeasCount").textContent = `${Object.keys(rd.products).length} products · STL decomposed`;
      $("p4SeasCount").className = "tag mini";
    }
  }
}

/* --- P4D: regime + seasonality -> buy/sell + entry/TP/SL --- */
function renderP4Signals(rd) {
  const el = $("p4SignalTable"); if (!el) return;
  if (!rd || !rd.products) return;

  // Today's calendar month for the seasonality vote.
  const nowMonth = new Date().getMonth(); // 0-11
  const monthNames = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"];

  let html = `<div class="p4-sig-head">
    <div>Product</div><div>Regime drift</div><div>Seasonality (${monthNames[nowMonth]})</div>
    <div>Verdict</div><div>Entry</div><div>Take Profit</div><div>Stop Loss</div>
    <div>R:R</div><div>Lots</div>
  </div>`;

  let n_long = 0, n_short = 0, n_hold = 0;

  Object.values(rd.products).forEach((p) => {
    const cur   = p.hmm_states[String(p.current_state_idx)];
    const ret   = cur.ret;          // daily log-return mean
    const vol20 = cur.vol20;        // daily log-return std
    const pers  = p.metrics?.HMM?.persistence || 0;
    const seas  = (p.seasonality?.monthly || [])[nowMonth] || 0;

    // ----- Vote 1: regime drift (sign of mean return, weighted by persistence) -----
    // High persistence -> we can trust the state will continue, so the drift
    // edge compounds. Score = ret * persistence, normalized.
    const regimeScore = ret * pers * 1000;   // typical ~0.5-1.5
    const regimeVote  = regimeScore > 0.3 ? +1
                       : regimeScore < -0.3 ? -1 : 0;
    const regimeLabel = regimeScore > 0
      ? `LONG bias · drift +${(ret*100).toFixed(2)}%/day × pers ${pers.toFixed(2)}`
      : `SHORT bias · drift ${(ret*100).toFixed(2)}%/day × pers ${pers.toFixed(2)}`;

    // ----- Vote 2: seasonality (sign of current month's STL component) -----
    // Negative seasonal => seasonally cheap => LONG bias.
    // Positive seasonal => seasonally rich => SHORT bias.
    const seasMagnitudeThreshold = 0.5 * Math.max(
      ...(p.seasonality?.monthly || []).map(Math.abs), 0.01);
    const seasVote = seas <= -seasMagnitudeThreshold ? +1
                   : seas >=  seasMagnitudeThreshold ? -1 : 0;
    const seasLabel = seas === 0
      ? "—"
      : `${seas > 0 ? "rich" : "cheap"} ${seas > 0 ? "+" : ""}${seas.toFixed(2)} → ${seasVote === +1 ? "LONG" : seasVote === -1 ? "SHORT" : "NEUTRAL"}`;

    // ----- Combined verdict -----
    const combined = regimeVote + seasVote;
    let direction, conviction;
    if (combined >= 2)       { direction = "LONG";  conviction = "high"; }
    else if (combined === 1) { direction = "LONG";  conviction = "med";  }
    else if (combined === 0) { direction = "HOLD";  conviction = "low";  }
    else if (combined === -1){ direction = "SHORT"; conviction = "med";  }
    else                     { direction = "SHORT"; conviction = "high"; }

    // Lots: high conv = up to 5, med = 2, hold = 0
    let lots = conviction === "high" ? 5 : conviction === "med" ? 2 : 0;
    // High-vol regimes (vol20 > 5%) -> halve lots (max 3)
    if (vol20 > 0.05) lots = Math.max(0, Math.floor(lots / 2));

    // ----- Entry / TP / SL from HMM state vol -----
    // Daily 1σ dollar move ≈ |price| × vol20.
    // Take profit at +4.5σ × √5 (5-day, 2σ horizon) in trade direction.
    // Stop loss at 1.5σ daily against the trade.
    // 2:1 reward:risk by construction.
    const price = p.current_price;
    const sigmaDollar = Math.abs(price) * vol20;
    const horizon = Math.sqrt(5);
    const tpDist = 2.0 * sigmaDollar * horizon;   // ~4.47σ
    const slDist = 1.5 * sigmaDollar;
    const dir = direction === "LONG" ? +1 : direction === "SHORT" ? -1 : 0;
    const tpPrice = dir === 0 ? null : price + dir * tpDist;
    const slPrice = dir === 0 ? null : price - dir * slDist;
    const rr = (tpDist / slDist).toFixed(2);

    if      (direction === "LONG")  n_long++;
    else if (direction === "SHORT") n_short++;
    else                            n_hold++;

    const dirColor = direction === "LONG" ? C.up
                    : direction === "SHORT" ? C.down : C.dim;
    const convClr = conviction === "high" ? "#4caf7e"
                   : conviction === "med"  ? "#e0b34a" : "#8b97a8";

    html += `<div class="p4-sig-row">
      <div>
        <div class="p4-sig-prodname">${p.name}</div>
        <div class="p4-sig-asset">${p.asset_label || "M1"} @ ${price?.toFixed(2)} ${p.price_unit || ""}</div>
      </div>
      <div>
        <span class="p4-sig-vote" style="color:${regimeVote>0?C.up:regimeVote<0?C.down:C.dim}">
          ${regimeVote>0?"▲ LONG":regimeVote<0?"▼ SHORT":"● flat"}</span>
        <div class="p4-sig-detail">${regimeLabel}</div>
      </div>
      <div>
        <span class="p4-sig-vote" style="color:${seasVote>0?C.up:seasVote<0?C.down:C.dim}">
          ${seasVote>0?"▲ LONG":seasVote<0?"▼ SHORT":"● flat"}</span>
        <div class="p4-sig-detail">${seasLabel}</div>
      </div>
      <div>
        <div class="p4-sig-verdict" style="color:${dirColor}; border-color:${dirColor}">${direction}</div>
        <div class="p4-sig-detail" style="color:${convClr}">conv: ${conviction}${vol20>0.05?" · vol↓size":""}</div>
      </div>
      <div class="p4-sig-px">${price?.toFixed(2)}</div>
      <div class="p4-sig-px" style="color:${C.up}">${tpPrice!=null?tpPrice.toFixed(2):"—"}</div>
      <div class="p4-sig-px" style="color:${C.down}">${slPrice!=null?slPrice.toFixed(2):"—"}</div>
      <div class="p4-sig-px"><b>${dir===0?"—":rr+":1"}</b></div>
      <div class="p4-sig-lots"><b style="color:${dirColor}">${lots||"—"}</b></div>
    </div>`;
  });

  el.innerHTML = html;
  if ($("p4SignalCount")) {
    $("p4SignalCount").textContent =
      `${n_long} long · ${n_short} short · ${n_hold} hold · ${monthNames[nowMonth]}`;
    $("p4SignalCount").className = "tag mini good";
  }
}

/* ---------- P5 News-aware trade signals ---------- */
function renderNewsSignals(ns) {
  const el = $("p5Table"); if (!el) return;
  if (!ns || !ns.available) {
    $("p5Count").textContent = "no data";
    el.innerHTML = "";
    return;
  }
  $("p5Count").textContent =
    `${ns.n_long} long · ${ns.n_short} short · ${ns.n_hold} hold · 24h window`;
  $("p5Count").className = "tag mini good";

  const dirColor = (d) =>
    d === "LONG" ? C.up : d === "SHORT" ? C.down : C.dim;

  const voteBadge = (vote, label) => {
    const col = vote > 0 ? C.up : vote < 0 ? C.down : C.dim;
    const sym = vote > 0 ? "▲" : vote < 0 ? "▼" : "●";
    return `<div class="p5-vote" style="color:${col}">${sym} ${vote>0?"+1":vote<0?"−1":"0"}</div>
            <div class="p5-vote-lbl">${label || "—"}</div>`;
  };

  let html = `<div class="p5-head">
    <div>Product</div>
    <div>News vote</div>
    <div>Z-score vote</div>
    <div>Verdict</div>
    <div>Entry</div>
    <div>TP</div>
    <div>SL</div>
    <div>R:R</div>
    <div>Lots</div>
  </div>`;

  (ns.products || []).forEach((p) => {
    const pln = p.plan || {};
    const dc = dirColor(p.direction);
    const headlines = p.top_headlines || [];

    html += `<details class="p5-row-wrap">
      <summary class="p5-row">
        <div>
          <div class="p5-prodname">${p.name}</div>
          <div class="p5-pricelbl">${p.price != null ? p.price : "—"} ${p.price_unit || ""}</div>
        </div>
        <div>${voteBadge(p.news_vote, p.news_label)}
          <div class="p5-counts">
            <span style="color:${C.up}">▲ ${p.news_bull}</span>
            <span style="color:${C.dim}">● ${p.news_neutral}</span>
            <span style="color:${C.down}">▼ ${p.news_bear}</span>
          </div>
        </div>
        <div>${voteBadge(p.z_vote, p.z_label)}</div>
        <div>
          <div class="p5-verdict" style="color:${dc}; border-color:${dc}">${p.direction}</div>
          <div class="p5-conv" style="color:${p.conviction==='high'?C.up:p.conviction==='med'?C.accent:C.dim}">
            conv: ${p.conviction}
          </div>
        </div>
        <div class="p5-px">${pln.entry != null ? pln.entry : "—"}</div>
        <div class="p5-px" style="color:${C.up}">${pln.tp != null ? pln.tp : "—"}${pln.tp_pct != null ? `<br><span class="p5-pct">+${pln.tp_pct}%</span>` : ""}</div>
        <div class="p5-px" style="color:${C.down}">${pln.sl != null ? pln.sl : "—"}${pln.sl_pct != null ? `<br><span class="p5-pct">−${pln.sl_pct}%</span>` : ""}</div>
        <div class="p5-px"><b>${pln.rr != null ? pln.rr + ":1" : "—"}</b></div>
        <div class="p5-lots"><b style="color:${dc}">${p.lots || "—"}</b></div>
      </summary>`;

    if (headlines.length) {
      html += `<div class="p5-expand">
        <div class="p5-expand-title">Top ${headlines.length} news items driving this verdict</div>`;
      headlines.forEach((h) => {
        const sc = h.score >= 0.15 ? C.up : h.score <= -0.15 ? C.down : C.dim;
        const imp = h.impact > 0 ? `<span style="color:${C.up}">▲ supply shock</span>`
                  : h.impact < 0 ? `<span style="color:${C.down}">▼ weak demand</span>` : "";
        const age_h = ((Date.now()/1000 - h.ts) / 3600).toFixed(1);
        html += `<div class="p5-headline">
          <div class="p5-headline-text">${h.headline}</div>
          <div class="p5-headline-meta">
            <span style="color:${sc}">score ${h.score >= 0 ? "+" : ""}${h.score}</span>
            ${imp ? `· ${imp}` : ""}
            <span class="dim">· ${h.source} · ${age_h}h ago</span>
          </div>
        </div>`;
      });
      html += `</div>`;
    } else {
      html += `<div class="p5-expand p5-expand-empty">No news items routed to this product in the last 24h.</div>`;
    }
    html += `</details>`;
  });

  el.innerHTML = html;
}

/* ---------- P3F Per-product trade plan ---------- */
function renderTAPlan(t) {
  const el = $("taPlanTable"); if (!el) return;
  if (!t || !t.available) {
    $("taPlanCount").textContent = "no data";
    el.innerHTML = "";
    return;
  }
  const dirColor = (d) =>
    d === "LONG" ? C.up : d === "SHORT" ? C.down : C.dim;

  let html = `<div class="ta-plan-head">
    <div>Product</div><div>Verdict</div><div>Lots</div>
    <div>Entry</div><div>Take Profit</div><div>Stop Loss</div>
    <div>TP %</div><div>SL %</div><div>R:R</div><div>20d vol</div>
  </div>`;

  let n_long = 0, n_short = 0, n_hold = 0;
  (t.products || []).forEach((p) => {
    const pln = p.plan || {};
    if      (p.direction === "LONG")  n_long++;
    else if (p.direction === "SHORT") n_short++;
    else                              n_hold++;
    const dc = dirColor(p.direction);
    html += `<div class="ta-plan-row">
      <div>
        <div class="ta-plan-name">${p.name}</div>
        <div class="ta-plan-pricelbl">spot ${p.price != null ? p.price : "—"}</div>
      </div>
      <div>
        <div class="ta-plan-verdict" style="color:${dc}; border-color:${dc}">${p.direction}</div>
      </div>
      <div class="ta-plan-lots"><b style="color:${dc}">${p.lots || "—"}</b></div>
      <div class="ta-plan-px">${pln.entry != null ? pln.entry : "—"}</div>
      <div class="ta-plan-px" style="color:${C.up}">${pln.tp != null ? pln.tp : "—"}</div>
      <div class="ta-plan-px" style="color:${C.down}">${pln.sl != null ? pln.sl : "—"}</div>
      <div class="ta-plan-px" style="color:${C.up}">${pln.tp_pct != null ? "+" + pln.tp_pct + "%" : "—"}</div>
      <div class="ta-plan-px" style="color:${C.down}">${pln.sl_pct != null ? "−" + pln.sl_pct + "%" : "—"}</div>
      <div class="ta-plan-px"><b>${pln.rr != null ? pln.rr + ":1" : "—"}</b></div>
      <div class="ta-plan-px">${pln.daily_vol_pct != null ? pln.daily_vol_pct + "%" : "—"}</div>
    </div>`;
  });

  el.innerHTML = html;
  $("taPlanCount").textContent =
    `${n_long} long · ${n_short} short · ${n_hold} hold · 5-day horizon`;
  $("taPlanCount").className = "tag mini good";
}

/* ---------- P3E Technical analysis ---------- */
function renderTechnicalSignals(t) {
  const el = $("taPanel"); if (!el) return;
  if (!t || !t.available) {
    $("taCount").textContent = "no data";
    el.innerHTML = `<div class="ts-empty">Technical-analysis panel not available.</div>`;
    return;
  }
  $("taCount").textContent =
    `${t.n_long} long · ${t.n_short} short · ${t.n_hold} hold`;
  $("taCount").className = "tag mini good";

  const dirColor = (d) =>
    d === "LONG" ? C.up : d === "SHORT" ? C.down : C.dim;
  const signalDot = (s) =>
    s > 0 ? `<span style="color:${C.up}">▲</span>`
    : s < 0 ? `<span style="color:${C.down}">▼</span>`
    : `<span style="color:${C.dim}">●</span>`;

  let html = "";
  (t.products || []).forEach(p => {
    const conf = p.conf_pct || 0;
    html += `<div class="ta-card">
      <div class="ta-head">
        <div class="ta-name">${p.name}</div>
        <div class="ta-price">${p.price != null ? p.price : "—"}</div>
        <div class="ta-verdict" style="color:${dirColor(p.direction)};
                                       border-color:${dirColor(p.direction)}">
          ${p.direction} ${p.direction !== "HOLD" ? `${p.lots} lot${p.lots !== 1 ? "s" : ""}` : ""}
        </div>
      </div>
      <div class="ta-meta">
        <span>raw score <b>${p.raw_score >= 0 ? "+" : ""}${p.raw_score}</b> / ±4</span>
        <span>confidence <b>${conf}%</b></span>
        <span class="dim">n=${p.n_hist} bars</span>
      </div>
      <div class="ta-signals">
        <div class="ta-sig">${signalDot(p.rsi.signal)} <span class="ta-sig-lbl">RSI</span>
          <span class="ta-sig-val">${p.rsi.value != null ? p.rsi.value : "—"}</span>
          <span class="ta-sig-txt">${p.rsi.label}</span></div>
        <div class="ta-sig">${signalDot(p.bb.signal)} <span class="ta-sig-lbl">BB%</span>
          <span class="ta-sig-val">${p.bb.value != null ? p.bb.value + "%" : "—"}</span>
          <span class="ta-sig-txt">${p.bb.label}</span></div>
        <div class="ta-sig">${signalDot(p.ema_cross.signal)} <span class="ta-sig-lbl">EMA</span>
          <span class="ta-sig-val">${p.ema_cross.value != null ? (p.ema_cross.value >= 0 ? "+" : "") + p.ema_cross.value + "%" : "—"}</span>
          <span class="ta-sig-txt">${p.ema_cross.label}</span></div>
        <div class="ta-sig">${signalDot(p.momentum.signal)} <span class="ta-sig-lbl">MOM</span>
          <span class="ta-sig-val">${p.momentum.value != null ? (p.momentum.value >= 0 ? "+" : "") + p.momentum.value + "%" : "—"}</span>
          <span class="ta-sig-txt">${p.momentum.label}</span></div>
      </div>
    </div>`;
  });
  el.innerHTML = html;
}

/* ---------- 04 WTI-Dollar ---------- */
function renderDXY(d) {
  $("corrVal").textContent = fmt(d.correlation, 3);
  $("corrVal").style.color = d.correlation < 0 ? C.up : C.down;
  $("covVal").textContent = fmt(d.covariance, 2);
  $("explVal").textContent = fmt(d.explained, 1) + "%";

  const ch = lineChart("dxyChart", {
    labels: d.wti.map((_, i) => i),
    datasets: [
      { label: "WTI", data: d.wti, borderColor: C.accent, yAxisID: "y" },
      { label: "DXY", data: d.dxy, borderColor: C.blue, yAxisID: "y1" },
    ],
    scales: {
      x: { display: false },
      y: { position: "left" },
      y1: { position: "right", grid: { drawOnChartArea: false } },
    },
  });
  ch.data.labels = d.wti.map((_, i) => i);
  ch.data.datasets[0].data = d.wti;
  ch.data.datasets[1].data = d.dxy;
  ch.update("none");
}

/* ---------- 05 WTI-Brent spread ---------- */
function renderSpread(s) {
  $("spVal").textContent = "$" + fmt(s.value, 2);
  $("spMean").textContent = "$" + fmt(s.mean, 2);
  $("spZ").textContent = fmt(s.zscore, 2) + "σ";
  const tag = $("spreadTag");
  tag.textContent = s.state;
  tag.className = "tag " + (Math.abs(s.zscore) > 1.3 ? "warn" : "good");

  const ch = lineChart("spreadChart", {
    labels: s.series.map((_, i) => i),
    datasets: [
      { label: "WTI-Brent", data: s.series, borderColor: C.blue,
        backgroundColor: "rgba(74,163,223,.08)", fill: true },
      { label: "mean", data: s.series.map(() => s.mean),
        borderColor: C.dim, borderDash: [4, 4], borderWidth: 1 },
    ],
    legend: false,
  });
  ch.data.labels = s.series.map((_, i) => i);
  ch.data.datasets[0].data = s.series;
  ch.data.datasets[1].data = s.series.map(() => s.mean);
  ch.update("none");
}

/* ---------- 06 futures curve ---------- */
function renderFutures(f) {
  $("m12").textContent = "$" + fmt(f.m12_spread, 2);
  $("m12").style.color = f.m12_spread >= 0 ? C.up : C.down;
  $("carry").textContent = "$" + fmt(f.monthly_carry, 3);
  $("spotV").textContent = "$" + fmt(f.spot, 2);
  const tag = $("curveTag");
  tag.textContent = f.structure;
  tag.className = "tag " + (f.structure === "Contango" ? "good"
    : f.structure === "Backwardation" ? "warn" : "");

  const ch = lineChart("futChart", {
    labels: f.curve.map((c) => "M" + c.month),
    datasets: [
      { label: "Futures", data: f.curve.map((c) => c.price),
        borderColor: C.accent, backgroundColor: "rgba(243,167,18,.1)",
        fill: true, tension: 0.15 },
      { label: "Spot", data: f.curve.map(() => f.spot),
        borderColor: C.dim, borderDash: [4, 4], borderWidth: 1 },
    ],
    legend: false,
    scales: { x: {}, y: {} },
  });
  ch.data.labels = f.curve.map((c) => "M" + c.month);
  ch.data.datasets[0].data = f.curve.map((c) => c.price);
  ch.data.datasets[1].data = f.curve.map(() => f.spot);
  ch.update("none");
}

/* ---------- 07 freight ---------- */
function renderFreight(f) {
  $("bdtiVal").textContent = fmt(f.value, 0);
  $("bdtiAvg").textContent = fmt(f.avg90, 0);
  const rel = ((f.value - f.avg90) / f.avg90) * 100;
  $("bdtiRel").textContent = (rel >= 0 ? "+" : "") + fmt(rel, 1) + "%";
  $("bdtiRel").style.color = rel >= 0 ? C.down : C.up;

  const ch = lineChart("freightChart", {
    labels: f.series.map((_, i) => i),
    datasets: [
      { label: "BDTI", data: f.series, borderColor: C.blue,
        backgroundColor: "rgba(74,163,223,.08)", fill: true },
    ],
    legend: false,
  });
  ch.data.labels = f.series.map((_, i) => i);
  ch.data.datasets[0].data = f.series;
  ch.update("none");
}

/* ---------- 08 crack spreads ---------- */
function renderCracks(cracks) {
  $("cracks").innerHTML = cracks.map((c) => {
    const z = c.zscore;
    const pct = Math.min(50, Math.abs(z) / 3 * 50);
    const pos = z >= 0;
    const col = Math.abs(z) > 2 ? C.down : Math.abs(z) > 1 ? C.accent : C.up;
    const fill = pos
      ? `left:50%;width:${pct}%;background:${col};`
      : `right:50%;width:${pct}%;background:${col};`;
    return `<div class="crack">
      <div class="cn">${c.name}</div>
      <div class="cv">$${fmt(c.value, 2)}</div>
      <div class="zbar"><span class="mid"></span>
        <span class="fill" style="${fill}"></span></div>
      <div class="zlabel"><span>z-score</span>
        <span style="color:${col}">${z >= 0 ? "+" : ""}${fmt(z, 2)}σ</span></div>
    </div>`;
  }).join("");
}

/* ---------- 09 covariance matrix ---------- */
function heat(v) {
  // v in [-1,1]: green (low/diversifying) -> amber -> red (concentrated)
  if (v >= 0) {
    const t = v;
    const r = Math.round(41 + t * (240 - 41));
    const g = Math.round(196 + t * (85 - 196));
    const b = Math.round(111 + t * (106 - 111));
    return `rgb(${r},${g},${b})`;
  }
  const t = -v;
  return `rgb(${Math.round(41 + t * 30)},${Math.round(196 - t * 60)},${Math.round(111 + t * 100)})`;
}
function renderCovMatrix(cm) {
  const { labels, matrix } = cm;
  let html = "<table class='cov'><tr><th></th>";
  html += labels.map((l) => `<th>${l}</th>`).join("") + "</tr>";
  matrix.forEach((row, i) => {
    html += `<tr><th class="row">${labels[i]}</th>`;
    html += row.map((v) =>
      `<td style="background:${heat(v)}">${v.toFixed(2)}</td>`).join("");
    html += "</tr>";
  });
  html += "</table>";
  $("covmatrix").innerHTML = html;
}

/* ---------- 10 fundamentals ---------- */
function renderFundamentals(cards) {
  $("fundamentals").innerHTML = cards.map((c) => {
    const cls = c.bullish === true ? "bull" : c.bullish === false ? "bear" : "";
    const dcol = c.trend === "up" ? C.up : c.trend === "down" ? C.down : C.dim;
    return `<div class="fund ${cls}">
      <div class="fl">${c.label}</div>
      <div class="fv">${c.value}<small> ${c.unit}</small></div>
      <div class="fd" style="color:${dcol}">${arrowFor(c.trend)} ${c.delta}</div>
      <div class="fn">${c.note}</div>
    </div>`;
  }).join("");
}

/* ---------- 11 signals ---------- */
function renderSignals(signals) {
  $("signals").innerHTML = signals.map((s) => `
    <div class="signal">
      <div class="sh">
        <span class="st">${s.title}</span>
        <span class="badge ${s.status}">${s.status}</span>
      </div>
      <div class="sdir">${s.direction}</div>
      <div class="srat">${s.rationale}</div>
      <div class="smetric"><span>${s.metric}</span>
        <b>${s.metric_value}</b></div>
    </div>`).join("");
}

/* ---------- 12 five-year week ---------- */
function renderFiveYear(fy) {
  const rows = fy.years.slice().reverse();
  const labels = rows.map((y) => String(y.year)).concat(["2026 (now)"]);
  const data = rows.map((y) => y.price).concat([fy.current]);
  const colors = data.map((_, i) =>
    i === data.length - 1 ? (fy.buy_signal ? C.up : C.accent) : C.blue);

  const ch = lineChart("fyChart", {
    type: "bar",
    labels,
    datasets: [{ label: "Same-week close", data, backgroundColor: colors,
      borderRadius: 3 }],
    legend: false,
    scales: { x: {}, y: { beginAtZero: false } },
  });
  ch.data.labels = labels;
  ch.data.datasets[0].data = data;
  ch.data.datasets[0].backgroundColor = colors;
  ch.update("none");

  const tag = $("fyTag");
  tag.textContent = fy.buy_signal ? "BUY SIGNAL" : "neutral";
  tag.className = "tag " + (fy.buy_signal ? "good" : "");
  $("fyNote").textContent = fy.buy_signal
    ? `Current $${fmt(fy.current, 2)} is the lowest same-week price in 5 years `
      + `(range $${fmt(fy.low, 2)}–$${fmt(fy.high, 2)}).`
    : `Current $${fmt(fy.current, 2)} sits within the 5-year same-week range `
      + `$${fmt(fy.low, 2)}–$${fmt(fy.high, 2)}.`;
}

/* ---------- 13 news ---------- */
function renderFinbertStatus(f) {
  const el = $("finbertStatus");
  if (!el) return;
  if (!f) { el.textContent = ""; return; }
  if (f.ready) {
    el.textContent = "FinBERT: ready";
    el.style.color = C.up;
  } else if (f.error) {
    el.textContent = "FinBERT: " + String(f.error).slice(0, 30) + " (VADER fallback)";
    el.style.color = C.down;
  } else if (f.loading) {
    el.textContent = "FinBERT: loading model…";
    el.style.color = C.accent;
  } else {
    el.textContent = "FinBERT: not started";
    el.style.color = C.dim;
  }
}

function renderNewsMood(m) {
  const moodEl = $("newsMood");
  const breakEl = $("newsMoodBreakdown");
  if (!m || m.count === 0) {
    moodEl.textContent = "no data";
    moodEl.className = "tag";
    breakEl.textContent = "";
    return;
  }
  // arrow + label + compound score, color-coded
  const arrow = m.label === "bullish" ? "▲"
              : m.label === "bearish" ? "▼" : "▬";
  const sign = m.compound >= 0 ? "+" : "";
  moodEl.textContent = `${arrow} ${m.label.toUpperCase()} ${sign}${m.compound.toFixed(2)}`;
  moodEl.className = "tag " + (m.label === "bullish" ? "good"
                             : m.label === "bearish" ? "bad" : "warn");
  breakEl.textContent = `${m.bullish} bull · ${m.bearish} bear · ${m.neutral} neutral · ${m.count} total`;
  breakEl.style.color = C.dim;
}

function renderRegions(regions) {
  const total = Object.values(regions).reduce((a, b) => a + b, 0);
  const entries = [["All", total]].concat(
    Object.entries(regions).sort((a, b) => b[1] - a[1]));
  $("newsRegions").innerHTML = entries.map(([r, n]) =>
    `<div class="rchip ${r === regionFilter ? "active" : ""}" data-r="${r}">
      <span>${r}</span><span class="rc">${n}</span></div>`).join("");
  document.querySelectorAll(".rchip").forEach((el) => {
    el.onclick = () => { regionFilter = el.dataset.r; renderNews(lastNews);
      renderRegions(regions); };
  });
}
function renderNews(items) {
  lastNews = items;
  const shown = items.filter((n) =>
    regionFilter === "All" || n.region === regionFilter);
  $("news").innerHTML = shown.map((n) => {
    const fresh = !seenNews.has(n.id);
    const hl = n.url
      ? `<a href="${n.url}" target="_blank" rel="noopener">${n.headline}</a>`
      : n.headline;
    // VADER (always present) — rule-based + oil-finance lexicon
    const v = Number(n.sentiment_score || 0);
    const vColor = v >= 0.15 ? C.up : v <= -0.15 ? C.down : C.dim;
    const vText = "V" + (v >= 0 ? "+" : "") + v.toFixed(2);
    // FinBERT (transformer, may be null while still loading)
    const fScore = n.finbert_score;
    const fLabel = n.finbert_label;
    const fAvailable = fScore !== null && fScore !== undefined;
    const f = fAvailable ? Number(fScore) : 0;
    const fColor = !fAvailable ? C.dim
                 : f >= 0.15 ? C.up : f <= -0.15 ? C.down : C.dim;
    const fText = fAvailable ? "F" + (f >= 0 ? "+" : "") + f.toFixed(2) : "F…";
    // primary label = FinBERT if available, else VADER
    const primaryLabel = fLabel || n.sentiment;
    const primaryColor = fAvailable ? fColor : vColor;
    // bar uses average of the two scores (or just VADER if FinBERT missing)
    const combined = fAvailable ? (v + f) / 2 : v;
    const barPct = Math.min(100, Math.abs(combined) * 100);
    return `<div class="news-item ${primaryLabel} ${fresh ? "fresh" : ""}">
      <div class="meta">
        <span>${n.source}</span>
        <span>${timeAgo(n.ts)}</span>
        ${n.live ? '<span class="live-pill">LIVE</span>' : ""}
        <span class="senti-score" style="color:${vColor}">${vText}</span>
        <span class="senti-score" style="color:${fColor}">${fText}</span>
      </div>
      <div class="hl">${hl}
        <div class="rg">${n.region} · <span style="color:${primaryColor}">${primaryLabel}</span></div>
        <div class="senti-bar"><span style="width:${barPct}%;background:${primaryColor}"></span></div>
      </div>
    </div>`;
  }).join("");
  items.forEach((n) => seenNews.add(n.id));
}

/* ---------- 10 CFTC COT positioning ---------- */
function fmtK(n) {
  if (n === null || n === undefined) return "—";
  const abs = Math.abs(n);
  if (abs >= 1000) return (n / 1000).toFixed(1) + "k";
  return n.toFixed(0);
}
function renderCOT(c) {
  if (!c || !c.categories) {
    $("cot").innerHTML =
      '<div class="paperempty">Waiting for first CFTC fetch ...</div>';
    return;
  }
  $("cotDate").textContent = "report " + (c.report_date || "—");
  $("cotSrc").textContent = c.source || "CFTC";
  $("cotSrc").style.color = C.dim;

  const rows = c.categories.map((cat) => {
    const netColor = cat.net >= 0 ? C.up : C.down;
    const chColor = cat.net_change >= 0 ? C.up : C.down;
    const ch = cat.net_change >= 0 ? "+" + fmtK(cat.net_change)
                                   :     fmtK(cat.net_change);
    return `<div class="fund ${cat.net >= 0 ? 'bull' : 'bear'}">
      <div class="fl">${cat.label}</div>
      <div class="fv" style="color:${netColor}">${cat.net >= 0 ? "+" : ""}${fmtK(cat.net)}
        <small>net</small></div>
      <div class="fd" style="color:${chColor}">WoW ${ch}</div>
      <div class="fn">L ${fmtK(cat.long)} · S ${fmtK(cat.short)}</div>
    </div>`;
  });
  // Total OI as a 5th card
  const oiCh = c.open_interest_change || 0;
  rows.push(`<div class="fund">
    <div class="fl">Open Interest</div>
    <div class="fv">${fmtK(c.open_interest)}<small> contracts</small></div>
    <div class="fd" style="color:${oiCh >= 0 ? C.up : C.down}">
      WoW ${oiCh >= 0 ? "+" : ""}${fmtK(oiCh)}</div>
    <div class="fn">total positions</div>
  </div>`);

  $("cot").innerHTML = rows.join("");
}

/* ---------- 14 paper trading ---------- */
function fmtMoney(n) {
  if (n === null || n === undefined) return "—";
  const s = (n >= 0 ? "+" : "-") + "$" +
    Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: 0,
                                           maximumFractionDigits: 0 });
  return s;
}
function renderPaper(p) {
  const eqColor = p.pct_change >= 0 ? C.up : C.down;
  $("paperEquity").textContent = "$" + p.equity.toLocaleString("en-US",
    { minimumFractionDigits: 0, maximumFractionDigits: 0 });
  $("paperEquity").style.color = eqColor;

  const pctEl = $("paperPct");
  pctEl.textContent = (p.pct_change >= 0 ? "+" : "") + p.pct_change.toFixed(2) + "%";
  pctEl.className = "tag " + (p.pct_change >= 0 ? "good" : "bad");

  const resetEl = $("paperResetAt");
  if (p.scheduled_reset_at) {
    const d = new Date(p.scheduled_reset_at * 1000);
    const local = d.toLocaleString(undefined,
      { dateStyle: "medium", timeStyle: "short" });
    resetEl.textContent = "auto-reset @ " + local;
    resetEl.style.color = C.accent;
  } else {
    resetEl.textContent = "";
  }

  $("paperRealized").textContent = fmtMoney(p.realized_pnl);
  $("paperRealized").style.color = p.realized_pnl >= 0 ? C.up : C.down;
  $("paperUnreal").textContent = fmtMoney(p.unrealized_pnl);
  $("paperUnreal").style.color = p.unrealized_pnl >= 0 ? C.up : C.down;
  $("paperTrades").textContent = p.n_trades + " (" + p.n_wins + "W)";
  $("paperWinRate").textContent = p.win_rate.toFixed(0) + "%";

  // open positions
  if (p.open_positions.length === 0) {
    $("paperOpen").innerHTML =
      '<div class="paperempty">No open positions. Waiting for a signal to trigger.</div>';
  } else {
    $("paperOpen").innerHTML = p.open_positions.map((pos) => {
      const mtmColor = (pos.mtm ?? 0) >= 0 ? C.up : C.down;
      const src = pos.source || "manual";
      const srcTag = src === "term"
        ? `<span class="src-tag src-term" title="From regression engine (P3B)">TS</span>`
        : src === "tech"
        ? `<span class="src-tag src-tech" title="From technical-analysis engine (P3E)">TA</span>`
        : "";
      return `<div class="paperrow ${pos.direction}">
        <span>${srcTag}${pos.title} <b>${pos.direction}</b></span>
        <span class="pl" style="color:${mtmColor}">${fmtMoney(pos.mtm)}</span>
        <span class="meta">entry $${pos.entry_price.toFixed(4)}
          · now $${(pos.current_price ?? 0).toFixed(4)}
          · ${pos.size_bbl.toLocaleString()} bbl
          · open ${pos.open_min.toFixed(0)}m</span>
      </div>`;
    }).join("");
  }

  // closed trades
  if (p.closed_trades.length === 0) {
    $("paperClosed").innerHTML =
      '<div class="paperempty">No closed trades yet.</div>';
  } else {
    $("paperClosed").innerHTML = p.closed_trades.map((t) => {
      const pnlColor = t.pnl >= 0 ? C.up : C.down;
      return `<div class="paperrow ${t.direction}">
        <span>${t.title.split(" ")[0]} <b>${t.direction}</b></span>
        <span class="pl" style="color:${pnlColor}">${fmtMoney(t.pnl)}</span>
        <span class="meta">$${t.entry_price.toFixed(2)} → $${t.exit_price.toFixed(2)}
          · ${t.duration_min.toFixed(0)}m</span>
      </div>`;
    }).join("");
  }

  // equity curve chart
  const labels = p.equity_curve.map((e) => e.ts);
  const data = p.equity_curve.map((e) => e.equity);
  const ch = lineChart("equityChart", {
    labels,
    datasets: [
      { label: "Equity", data, borderColor: C.accent,
        backgroundColor: "rgba(243,167,18,.08)", fill: true },
      { label: "Start", data: data.map(() => p.starting_equity),
        borderColor: C.dim, borderDash: [4, 4], borderWidth: 1 },
    ],
    legend: false,
    scales: { x: { display: false }, y: { beginAtZero: false } },
  });
  ch.data.labels = labels;
  ch.data.datasets[0].data = data;
  ch.data.datasets[1].data = data.map(() => p.starting_equity);
  ch.update("none");

  // ------- per-source breakdown -------
  const bs = p.by_source || {};
  const SRC_META = {
    term:   { label: "Regression engine (P3B)", tag: "TS", cls: "src-term" },
    tech:   { label: "Technical analysis (P3E)", tag: "TA", cls: "src-tech" },
    manual: { label: "Original 4 strategies",    tag: "M",  cls: "src-manual" },
  };
  const bsEl = $("paperBySource");
  if (bsEl) {
    let html = "";
    ["term", "tech", "manual"].forEach((src) => {
      const s = bs[src] || {};
      const meta = SRC_META[src];
      const totalCol = (s.total_pnl ?? 0) >= 0 ? C.up : C.down;
      const realCol  = (s.realized_pnl ?? 0) >= 0 ? C.up : C.down;
      const unrlCol  = (s.unrealized_pnl ?? 0) >= 0 ? C.up : C.down;
      html += `<div class="paper-srccard">
        <div class="paper-srchead">
          <span class="src-tag ${meta.cls}">${meta.tag}</span>
          <span class="paper-srclabel">${meta.label}</span>
          <span class="paper-srctotal" style="color:${totalCol}">${fmtMoney(s.total_pnl)}</span>
        </div>
        <div class="paper-srcgrid">
          <div><span>open</span><b>${s.n_open ?? 0}</b></div>
          <div><span>closed</span><b>${s.n_closed ?? 0}</b></div>
          <div><span>win rate</span><b>${(s.win_rate ?? 0).toFixed(0)}%</b></div>
          <div><span>realized</span><b style="color:${realCol}">${fmtMoney(s.realized_pnl)}</b></div>
          <div><span>unrealized</span><b style="color:${unrlCol}">${fmtMoney(s.unrealized_pnl)}</b></div>
          <div><span>avg / trade</span><b>${fmtMoney(s.avg_pnl)}</b></div>
          <div><span>best</span><b style="color:${C.up}">${fmtMoney(s.best_trade)}</b></div>
          <div><span>worst</span><b style="color:${C.down}">${fmtMoney(s.worst_trade)}</b></div>
        </div>
      </div>`;
    });
    bsEl.innerHTML = html;
  }

  // ------- full track-record table (last 50 closed) -------
  const trEl = $("paperTrackTable");
  const tr = p.track_record || [];
  if (trEl) {
    if (tr.length === 0) {
      trEl.innerHTML = '<div class="paperempty">No closed trades yet — open positions will appear here once they exit.</div>';
    } else {
      let html = `<div class="paper-trackrow paper-trackhdr">
        <div>src</div><div>title</div><div>dir</div>
        <div>entry</div><div>exit</div><div>size</div>
        <div>hold</div><div>P&amp;L</div>
      </div>`;
      tr.forEach((t) => {
        const src = t.source || "manual";
        const meta = SRC_META[src] || SRC_META.manual;
        const pnlCol = (t.pnl ?? 0) >= 0 ? C.up : C.down;
        html += `<div class="paper-trackrow">
          <div><span class="src-tag ${meta.cls}">${meta.tag}</span></div>
          <div class="paper-tracktitle">${t.title}</div>
          <div style="color:${t.direction === 'LONG' ? C.up : C.down}; font-weight:600">${t.direction}</div>
          <div>${(t.entry_price ?? 0).toFixed(3)}</div>
          <div>${(t.exit_price ?? 0).toFixed(3)}</div>
          <div>${(t.size_bbl ?? 0).toLocaleString()}</div>
          <div>${(t.duration_min ?? 0).toFixed(0)}m</div>
          <div style="color:${pnlCol}; font-weight:700">${fmtMoney(t.pnl)}</div>
        </div>`;
      });
      trEl.innerHTML = html;
    }
  }
}

/* ---------- master render ---------- */
function render(d) {
  renderHeader(d.header);
  renderPrice(d.price);
  renderBB(d.bb);
  renderBB(d.bb_brent, { prefix: "bbBrent", chartId: "bbBrentChart",
                         label: "Brent", color: C.blue, decimals: 2 });
  renderBB(d.bb_rbob,  { prefix: "bbRbob",  chartId: "bbRbobChart",
                         label: "RBOB",  color: "#9be15d", decimals: 4 });
  renderBB(d.bb_heat,  { prefix: "bbHeat",  chartId: "bbHeatChart",
                         label: "HO",    color: "#f0a0b0", decimals: 4 });
  renderRegimeButterfly(d.regime_butterfly);
  renderMultiProductMatrix(d.multi_product_matrix);
  renderP2Fingerprint(d.phase2_regime);
  renderPctRegime(d.percentile_regime);
  renderPStrat(d.paper_strats);
  renderBPP(d.best_per_product);
  renderComposite(d.composite_panel);
  renderRealData(d.real_data_panel);
  renderLiveSignals(d.live_signals);
  renderP2Models(d.phase2_models);
  renderP2Opportunities(d.phase2_opportunities);
  renderP2History(d.phase2_history);
  renderTSQuality(d.term_structure);
  renderTSCrossTop(d.term_structure);
  renderTSByProduct(d.term_structure);
  renderTSFull(d.term_structure);
  renderTAPlan(d.technical_signals);
  renderTechnicalSignals(d.technical_signals);
  renderNewsSignals(d.news_signals);
  renderDXY(d.dxy);
  renderSpread(d.spread);
  renderFutures(d.futures);
  renderFreight(d.freight);
  renderCracks(d.cracks);
  renderCovMatrix(d.covmatrix);
  renderFundamentals(d.fundamentals);
  renderSignals(d.signals);
  renderFiveYear(d.fiveyear);
  renderCOT(d.cot);
  if (d.manufacturing) renderManufacturing(d.manufacturing);
  if (d.steo) renderSteo(d.steo);
  renderPaper(d.paper);
  renderNewsMood(d.news_sentiment);
  renderFinbertStatus(d.finbert);
  renderRegions(d.news_regions);
  renderNews(d.news);
  if (d.curve_matrix) renderCurveMatrix(d.curve_matrix);
  if (d.spread_covmatrix) renderSpreadCovMatrix(d.spread_covmatrix);
  if (d.product_curves) renderProductCurves(d.product_curves);
  if (d.commodity_correlation) renderCommodityCorrelation(d.commodity_correlation);
  if (d.analyst_news) renderAnalystNews(d.analyst_news);
  if (d.seasonality) renderSeasonality(d.seasonality);
  if (d.tankers) {
    renderTankers(d.tankers);
    renderChokePoints(d.tankers);
    renderFloatingStorage(d.tankers);
    renderStsCandidates(d.tankers);
  }
  if (d.storms) renderStorms(d.storms);
  if (d.products) renderProducts(d.products);
  if (d.composite_signal) renderCompositeSignal(d.composite_signal);
  $("priceSrc").textContent = d.sources.prices;
  $("dxySrc").textContent = d.sources.dollar;
  $("fySrc").textContent = d.sources.five_year;
  $("cracksSrc").textContent = d.sources.cracks;
  $("curveSrc").textContent = d.sources.curve;
  $("freightSrc").textContent = d.sources.freight;
  $("fundSrc").textContent = "source: " + d.sources.fundamentals;
  $("newsSrc").textContent = "source: " + d.sources.news;
  $("tickInfo").textContent = "tick " + d.tick;
}

/* ---------- websocket ---------- */
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    $("dot").classList.add("live");
    $("connText").textContent = "live · websocket";
  };
  ws.onmessage = (ev) => {
    try { render(JSON.parse(ev.data)); }
    catch (e) { console.error("render error", e); }
  };
  ws.onclose = () => {
    $("dot").classList.remove("live");
    $("connText").textContent = "reconnecting…";
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();

  // keep-alive ping
  setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send("ping");
  }, 15000);
}

/* ---------- T9 Seasonality (refinery utilization vs 5y norm) ---------- */
function renderSeasonality(s) {
  if (!s || !s.available) {
    $("seasonPhase").textContent = "warming up";
    $("seasonPhase").className = "tag";
    return;
  }
  // headline phase
  const phase = s.phase || {};
  const next = s.next_phase || {};
  $("seasonPhase").textContent =
    `Wk ${s.current_week}: ${phase.phase || "—"}`;
  $("seasonPhase").className = "tag " + (
    phase.impact && phase.impact.indexOf("bullish") >= 0 ? "good" : "warn");
  $("seasonNext").textContent =
    `next: ${next.phase || "—"} in ${next.starts_in_weeks || "?"}w`;
  $("seasonNext").style.color = C.dim;

  const huri = phase.in_peak_hurricane ? "🌀 PEAK hurricane risk"
             : phase.in_hurricane_season ? "Atlantic hurricane season"
             : "";
  $("seasonHurricane").textContent = huri;
  $("seasonHurricane").style.color = phase.in_peak_hurricane
                                       ? C.down : C.dim;

  // numeric readout
  $("seasonNow").textContent = s.current_value != null
    ? `${s.current_value.toFixed(1)}%` : "—";
  $("seasonNorm").textContent = s.seasonal_mean != null
    ? `${s.seasonal_mean.toFixed(1)}%` : "—";
  const z = s.seasonal_z;
  $("seasonDev").textContent = z != null
    ? `${z >= 0 ? "+" : ""}${z.toFixed(2)}σ` : "—";
  $("seasonDev").style.color = z == null ? C.text
                              : z <= -1 ? C.up
                              : z >= 1 ? C.down : C.text;

  // chart: 52 weeks normal band + this year's actuals
  const labels = s.chart.map((c) => "w" + c.week);
  const means = s.chart.map((c) => c.mean);
  const his = s.chart.map((c) => c.hi);
  const los = s.chart.map((c) => c.lo);
  const thisYear = s.chart.map((c) => c.this_year);

  const ch = lineChart("seasonChart", {
    labels,
    datasets: [
      { label: "5y high band", data: his, borderColor: C.dim,
        borderDash: [2, 3], borderWidth: 1, fill: false },
      { label: "5y mean", data: means, borderColor: C.dim,
        borderWidth: 1.5, fill: false },
      { label: "5y low band", data: los, borderColor: C.dim,
        borderDash: [2, 3], borderWidth: 1, fill: false },
      { label: "this year", data: thisYear, borderColor: C.accent,
        backgroundColor: "rgba(243,167,18,.1)", fill: false,
        borderWidth: 2.4, spanGaps: false },
    ],
    legend: true,
    scales: { x: { display: true, ticks: { maxTicksLimit: 13 } },
              y: { ticks: { callback: (v) => v + "%" } } },
  });
  ch.data.labels = labels;
  ch.data.datasets[0].data = his;
  ch.data.datasets[1].data = means;
  ch.data.datasets[2].data = los;
  ch.data.datasets[3].data = thisYear;
  ch.update("none");
}

/* ---------- tabs ---------- */
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll(".tab-btn").forEach((b) =>
      b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tab-content").forEach((c) =>
      c.classList.remove("active"));
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  };
});

/* ---------- T7 Calendar spread matrix ---------- */
function renderCurveMatrix(cm) {
  if (!cm || !cm.labels) { $("curveMatrix").innerHTML = ""; return; }
  const { labels, matrix, prices } = cm;
  // color cell by sign and magnitude of spread
  const maxAbs = Math.max(1,
    ...matrix.flat().map((v) => Math.abs(v)));
  function heatSpread(v) {
    if (v > 0) {
      const t = Math.min(1, v / maxAbs);
      return `rgb(${Math.round(41 + (1-t)*60)},${Math.round(196 - t*40)},${Math.round(111 - t*40)})`;
    }
    if (v < 0) {
      const t = Math.min(1, -v / maxAbs);
      return `rgb(${Math.round(240 - (1-t)*100)},${Math.round(85 + (1-t)*60)},${Math.round(106 - (1-t)*30)})`;
    }
    return "#1c2533";
  }
  let html = "<table class='cov'><tr><th></th>";
  labels.forEach((l, i) => {
    html += `<th title="${prices[i]}">${l}<br><span style="color:#7c8aa0;font-weight:400">$${prices[i]}</span></th>`;
  });
  html += "</tr>";
  matrix.forEach((row, i) => {
    html += `<tr><th class="row">${labels[i]} <small>$${prices[i]}</small></th>`;
    row.forEach((v) => {
      const c = v === 0 ? "#1c2533" : heatSpread(v);
      const txt = v === 0 ? "—" : (v > 0 ? "+" : "") + v.toFixed(2);
      html += `<td style="background:${c}">${txt}</td>`;
    });
    html += "</tr>";
  });
  html += "</table>";
  $("curveMatrix").innerHTML = html;
}

/* ---------- T7C Multi-product 12-month curves ---------- */
function renderProductCurves(p) {
  if (!p) return;
  // Normalize each curve to M1 = 100 so shapes are visually comparable
  function normalize(curve) {
    if (!curve || curve.length === 0) return [];
    const m1 = curve[0].price;
    if (!m1) return [];
    return curve.map((c) => 100 * c.price / m1);
  }
  const labels = ["M1","M2","M3","M4","M5","M6","M7","M8","M9","M10","M11","M12"];
  const sets = [
    { label: "WTI",    color: C.accent, data: normalize(p.wti) },
    { label: "Brent",  color: C.blue,   data: normalize(p.brent) },
    { label: "RBOB",   color: "#29c46f", data: normalize(p.rbob) },
    { label: "HO",     color: "#ff6e3a", data: normalize(p.heat) },
    { label: "NatGas", color: "#a37bff", data: normalize(p.natgas) },
  ];
  const ch = lineChart("productCurvesChart", {
    labels,
    datasets: sets.map((s) => ({
      label: s.label,
      data: s.data.length ? s.data : Array(12).fill(null),
      borderColor: s.color,
      backgroundColor: "transparent",
      tension: 0.2, pointRadius: 0, borderWidth: 1.8,
      spanGaps: false,
    })),
    legend: true,
    scales: { x: { display: true },
              y: { ticks: { callback: (v) => v + "" },
                   title: { display: true, text: "M1 = 100 (normalized)",
                            color: C.dim } } },
  });
  ch.data.labels = labels;
  ch.data.datasets.forEach((ds, i) => {
    ds.data = sets[i].data.length ? sets[i].data : Array(12).fill(null);
  });
  ch.update("none");
}

/* ---------- T7D Commodity correlation matrix ---------- */
function renderCommodityCorrelation(cm) {
  const nEl = $("commCorrN");
  const target = $("commCorrMatrix");
  if (!cm || !cm.labels || cm.labels.length === 0) {
    nEl.textContent = (cm && cm.note) || "warming up";
    nEl.style.color = C.dim;
    target.innerHTML = '<div class="paperempty">Building commodity correlation from daily history... ' +
                       'need ≥60 trading days per series.</div>';
    return;
  }
  nEl.textContent = `${cm.n_days} days · ${cm.labels.length}×${cm.labels.length}`;
  nEl.style.color = C.dim;

  const { labels, matrix } = cm;
  let html = "<table class='cov'><tr><th></th>";
  html += labels.map((l) => `<th>${l}</th>`).join("") + "</tr>";
  matrix.forEach((row, i) => {
    html += `<tr><th class="row">${labels[i]}</th>`;
    html += row.map((v) =>
      `<td style="background:${heat(v)}">${v.toFixed(3)}</td>`).join("");
    html += "</tr>";
  });
  html += "</table>";
  target.innerHTML = html;
}

/* ---------- T7B Calendar spread covariance ---------- */
function renderSpreadCovMatrix(cm) {
  const nEl = $("spreadCovN");
  const target = $("spreadCovMatrix");
  if (!cm) { target.innerHTML = ""; nEl.textContent = ""; return; }

  if (cm.warming_up || !cm.labels || cm.labels.length === 0) {
    nEl.textContent = `warming up ${cm.n_snapshots || 0}/30 snapshots`;
    nEl.style.color = C.accent;
    target.innerHTML =
      '<div class="paperempty">Building spread covariance from live curve history... ' +
      'need ≥30 snapshots (~1 min of ticks).</div>';
    return;
  }

  nEl.textContent = `${cm.n_snapshots} snapshots · ${cm.labels.length} spreads · units ${cm.units || "$²/bbl²"}`;
  nEl.style.color = C.dim;

  const { labels, matrix } = cm;
  // Color scale by sign + magnitude relative to matrix max-absolute value
  // (covariance values are tiny — fractions of a $² — so we normalize).
  const maxAbs = Math.max(1e-9, cm.max_abs || 0);
  function heatCov(v) {
    if (v > 0) {
      const t = Math.min(1, v / maxAbs);
      return `rgb(${Math.round(41 + (1 - t) * 60)},${Math.round(196 - t * 110)},${Math.round(111 - t * 5)})`;
    }
    if (v < 0) {
      const t = Math.min(1, -v / maxAbs);
      return `rgb(${Math.round(41 + t * 30)},${Math.round(196 - t * 60)},${Math.round(111 + t * 100)})`;
    }
    return "#1c2533";
  }

  let html = "<table class='cov'><tr><th></th>";
  html += labels.map((l) => `<th>${l}</th>`).join("") + "</tr>";
  matrix.forEach((row, i) => {
    html += `<tr><th class="row">${labels[i]}</th>`;
    html += row.map((v, j) => {
      const isDiag = i === j;
      // diagonal = variance (always positive) — show 4 decimals
      // off-diag = covariance — show 4 decimals with sign
      const txt = isDiag ? v.toFixed(4)
                : (v >= 0 ? "+" : "") + v.toFixed(4);
      const bg = isDiag ? "#1c2533" : heatCov(v);
      return `<td style="background:${bg}">${txt}</td>`;
    }).join("");
    html += "</tr>";
  });
  html += "</table>";
  target.innerHTML = html;
}

/* ---------- T8 Analyst watch ---------- */
function renderAnalystNews(an) {
  if (!an) { $("analystNews").innerHTML =
    '<div class="paperempty">Waiting for first analyst fetch...</div>'; return; }
  const html = Object.entries(an).map(([name, data]) => {
    const summary = (data && data.summary) || {};
    const items = (data && data.items) || [];
    const mood = summary.label || "neutral";
    const moodColor = mood === "bullish" ? C.up
                    : mood === "bearish" ? C.down : C.dim;
    const compound = (summary.compound || 0).toFixed(2);
    const moodTxt = `${mood.toUpperCase()} ${compound >= 0 ? "+" : ""}${compound}`;
    const itemsHtml = items.length === 0
      ? '<div class="paperempty">No recent oil-related mentions.</div>'
      : items.map((it) => {
          const s = Number(it.sentiment_score || 0);
          const scoreColor = s >= 0.15 ? C.up
                          : s <= -0.15 ? C.down : C.dim;
          const scoreText = (s >= 0 ? "+" : "") + s.toFixed(2);
          const linkOpen = it.url ? `<a href="${it.url}" target="_blank" rel="noopener">` : "";
          const linkClose = it.url ? "</a>" : "";
          return `<div class="analyst-item ${it.sentiment}">
            <div class="ai-meta">
              <span>${it.source || "?"}</span>
              <span>${timeAgo(it.ts)}</span>
              <span style="color:${scoreColor};font-weight:700;margin-left:auto">${scoreText}</span>
            </div>
            <div>${linkOpen}${it.headline}${linkClose}</div>
          </div>`;
        }).join("");
    return `<div class="analyst-col">
      <div class="ahead">
        <span class="aname">${name}</span>
        <span class="amood" style="background:${moodColor}25;color:${moodColor}">${moodTxt}</span>
      </div>
      <div class="acount">${items.length} mentions · last 3 days</div>
      <div class="alist">${itemsHtml}</div>
    </div>`;
  }).join("");
  $("analystNews").innerHTML = html;
}

/* ---------- US Manufacturing Health (FRED PMI proxy) ---------- */
function renderManufacturing(m) {
  if (!m) return;
  const srcEl = $("pmiSrc");
  if (srcEl) {
    srcEl.textContent = (m.source || "FRED").slice(0, 60);
    srcEl.style.color = C.dim;
  }
  const comp = m.composite_proxy || {};
  const cEl = $("pmiComposite");
  const dEl = $("pmiCompDelta");
  const sEl = $("pmiCompState");
  const nEl = $("pmiNSurveys");
  const tagEl = $("pmiCompositeTag");

  cEl.textContent = comp.value != null ? comp.value : "—";
  // colour: >50 expansion green, <50 contraction red
  cEl.style.color = comp.value == null ? C.text
                  : comp.value > 50 ? C.up
                  : comp.value < 50 ? C.down : C.text;
  dEl.textContent = comp.delta != null
    ? (comp.delta >= 0 ? "+" : "") + comp.delta : "—";
  dEl.style.color = comp.delta == null ? C.text
                  : comp.delta >= 0 ? C.up : C.down;
  sEl.textContent = comp.state || "—";
  sEl.style.color = comp.state === "expansion" ? C.up
                  : comp.state === "contraction" ? C.down : C.dim;
  nEl.textContent = comp.n_surveys != null ? comp.n_surveys : "—";

  if (comp.state === "expansion") {
    tagEl.textContent = "EXPANSION";
    tagEl.className = "tag good";
  } else if (comp.state === "contraction") {
    tagEl.textContent = "CONTRACTION";
    tagEl.className = "tag bad";
  } else {
    tagEl.textContent = comp.state ? comp.state.toUpperCase() : "—";
    tagEl.className = "tag warn";
  }

  // per-indicator cards
  $("pmiCards").innerHTML = (m.cards || []).map((c) => {
    let dcol = C.dim;
    if (c.trend === "up") dcol = C.up;
    else if (c.trend === "down") dcol = C.down;
    const cls = c.state === "expansion" ? "bull"
              : c.state === "contraction" ? "bear" : "";
    const stateBadge = c.state
      ? `<span style="color:${c.state === "expansion" ? C.up : C.down};font-weight:700">${c.state}</span>`
      : "";
    const deltaSign = c.delta >= 0 ? "+" : "";
    return `<div class="fund ${cls}">
      <div class="fl">${c.label}</div>
      <div class="fv">${c.value}<small> ${c.units}</small></div>
      <div class="fd" style="color:${dcol}">${arrowFor(c.trend)} ${deltaSign}${c.delta} ${stateBadge}</div>
      <div class="fn">${c.series_id} · ${c.period || ""}</div>
    </div>`;
  }).join("");
}

/* ---------- STEO global oil balance ---------- */
function renderSteo(s) {
  if (!s || !s.balance) return;
  const h = s.headline || {};
  $("steoPeriod").textContent = "headline " + (h.period || "—");
  $("steoSrc").textContent = s.source || "EIA STEO";
  $("steoSrc").style.color = C.dim;

  $("steoSupply").textContent = h.supply != null ? h.supply.toFixed(2) : "—";
  $("steoDemand").textContent = h.demand != null ? h.demand.toFixed(2) : "—";

  const bal = h.balance;
  const balEl = $("steoBalance");
  balEl.textContent = bal != null
    ? (bal >= 0 ? "+" : "") + bal.toFixed(2)
    : "—";
  // surplus (supply > demand) → bearish prices → red; deficit → bullish → green
  balEl.style.color = bal == null ? C.text
                    : bal >= 0.3 ? C.down
                    : bal <= -0.3 ? C.up : C.text;

  function fmtBal(v) {
    if (v == null) return "—";
    return (v >= 0 ? "+" : "") + v.toFixed(2);
  }
  function balColor(v) {
    if (v == null) return C.text;
    return v >= 0.3 ? C.down : v <= -0.3 ? C.up : C.text;
  }
  $("steoFwd6").textContent = fmtBal(s.fwd6_avg_balance);
  $("steoFwd6").style.color = balColor(s.fwd6_avg_balance);
  $("steoFwd12").textContent = fmtBal(s.fwd12_avg_balance);
  $("steoFwd12").style.color = balColor(s.fwd12_avg_balance);

  // headline fwd tag
  const fwdTag = $("steoFwdTag");
  const f12 = s.fwd12_avg_balance;
  if (f12 == null) {
    fwdTag.textContent = "—"; fwdTag.className = "tag";
  } else if (f12 <= -0.5) {
    fwdTag.textContent = "fwd 12M: TIGHTENING (deficit)";
    fwdTag.className = "tag good";
  } else if (f12 >= 0.5) {
    fwdTag.textContent = "fwd 12M: LOOSENING (surplus)";
    fwdTag.className = "tag bad";
  } else {
    fwdTag.textContent = "fwd 12M: BALANCED";
    fwdTag.className = "tag warn";
  }

  // breakdown: latest historical OPEC/NonOPEC/OECD/NonOECD
  function lastValue(series) {
    if (!series || series.length === 0) return null;
    return series[series.length - 1].value;
  }
  const opec = lastValue(s.opec_supply);
  const nonopec = lastValue(s.non_opec_supply);
  const oecd = lastValue(s.oecd_demand);
  const nonoecd = lastValue(s.non_oecd_demand);
  $("steoOpec").textContent = opec != null ? opec.toFixed(2) + " Mbpd" : "—";
  $("steoNonOpec").textContent = nonopec != null ? nonopec.toFixed(2) + " Mbpd" : "—";
  $("steoOecd").textContent = oecd != null ? oecd.toFixed(2) + " Mbpd" : "—";
  $("steoNonOecd").textContent = nonoecd != null ? nonoecd.toFixed(2) + " Mbpd" : "—";

  // chart: supply (line), demand (line), balance (bars) — with fcst styling
  const labels = s.balance.map((b) => b.period);
  const supply = s.balance.map((b) => b.supply);
  const demand = s.balance.map((b) => b.demand);
  const balance = s.balance.map((b) => b.balance);
  // colour the balance bars: green deficit, red surplus, dim near zero
  const balColors = s.balance.map((b) => {
    const v = b.balance;
    const base = v >= 0.3 ? "240,85,106" : v <= -0.3 ? "41,196,111" : "124,138,160";
    // forecast cells get half-alpha to distinguish from historical
    return `rgba(${base},${b.is_forecast ? 0.35 : 0.85})`;
  });

  if (!charts.steoChart) {
    charts.steoChart = new Chart($("steoChart"), {
      type: "bar",
      data: {
        labels,
        datasets: [
          { type: "bar", label: "Balance (supply − demand)", data: balance,
            backgroundColor: balColors, yAxisID: "y1", order: 2,
            borderRadius: 2 },
          { type: "line", label: "World supply", data: supply,
            borderColor: C.accent, backgroundColor: "rgba(243,167,18,.08)",
            yAxisID: "y", order: 1, fill: false,
            pointRadius: 0, borderWidth: 1.6, tension: 0.2 },
          { type: "line", label: "World demand", data: demand,
            borderColor: C.blue, yAxisID: "y", order: 0, fill: false,
            pointRadius: 0, borderWidth: 1.6, tension: 0.2 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { intersect: false, mode: "index" },
        plugins: { legend: { display: true,
          labels: { boxWidth: 10, boxHeight: 2, padding: 8 } } },
        scales: {
          x: { ticks: { maxTicksLimit: 14 } },
          y:  { position: "left",
                title: { display: true, text: "M bpd (supply/demand)",
                         color: C.dim } },
          y1: { position: "right",
                title: { display: true, text: "balance (M bpd)",
                         color: C.dim },
                grid: { drawOnChartArea: false } },
        },
      },
    });
  } else {
    const ch = charts.steoChart;
    ch.data.labels = labels;
    ch.data.datasets[0].data = balance;
    ch.data.datasets[0].backgroundColor = balColors;
    ch.data.datasets[1].data = supply;
    ch.data.datasets[2].data = demand;
    ch.update("none");
  }
}

/* ---------- Composite oil signal ---------- */
function renderCompositeSignal(c) {
  if (!c) return;
  const score = c.score;
  const label = c.label || "—";
  const color = c.color || "neutral";

  // headline
  const lbl = $("csLabel");
  lbl.textContent = label;
  lbl.className = "cs-verdict cs-" + color;
  $("csScore").textContent = (score > 0 ? "+" : "") + score.toFixed(0);

  // gauge pointer position: score -100..+100 → 0..100% along the bar
  const pct = Math.max(0, Math.min(100, (score + 100) / 2));
  $("csPointer").style.left = pct + "%";

  // bucket scores + drivers
  function fillBucket(scoreId, listId, bucket) {
    if (!bucket) return;
    const sEl = $(scoreId);
    const s = bucket.score;
    sEl.textContent = (s > 0 ? "+" : "") + s.toFixed(0);
    sEl.style.color = s > 5 ? C.up : s < -5 ? C.down : C.dim;
    const drivers = bucket.drivers || [];
    const html = drivers.length === 0
      ? '<li class="cb-empty">no contributions</li>'
      : drivers.map((d) => {
          const contrib = d.contribution;
          const col = contrib > 0 ? C.up : contrib < 0 ? C.down : C.dim;
          const sign = contrib >= 0 ? "+" : "";
          const val = d.value ? `<span class="dval">${d.value}</span>` : "";
          return `<li>
            <span class="dctrb" style="color:${col}">${sign}${contrib}</span>
            <span class="dlbl">${d.label}</span>
            ${val}
          </li>`;
        }).join("");
    $(listId).innerHTML = html;
  }
  fillBucket("csTechScore", "csTechDrivers", c.buckets && c.buckets.technical);
  fillBucket("csFundScore", "csFundDrivers", c.buckets && c.buckets.fundamental);
  fillBucket("csNewsScore", "csNewsDrivers", c.buckets && c.buckets.news);
}

/* ---------- T6A/B/C Product terminals (RBOB / HO / Gas Oil) ---------- */
function renderProducts(p) {
  if (!p) return;
  function fill(prefix, data) {
    const priceEl = $(`${prefix}Price`);
    const anchorEl = $(`${prefix}Anchor`);
    const srcEl = $(`${prefix}Src`);
    if (!priceEl) return;
    if (data && data.price != null) {
      priceEl.textContent = "$" + Number(data.price).toFixed(4);
      priceEl.style.color = C.accent;
    } else if (data && data.price == null && prefix === "gasoil") {
      // already set to "paid feed only" by HTML, leave label
      priceEl.style.color = C.dim;
    } else if (priceEl.textContent === "—") {
      priceEl.style.color = C.text;
    }
    if (anchorEl) {
      anchorEl.textContent = data && data.anchor != null
        ? "$" + Number(data.anchor).toFixed(4)
        : "—";
      anchorEl.style.color = C.blue;
    }
    if (srcEl) {
      srcEl.textContent = (data && data.source) || "";
      srcEl.style.color = C.dim;
    }
  }
  fill("rbob",   p.rbob);
  fill("heat",   p.heat);
  fill("gasoil", p.gasoil);
}

/* ---------- T11 Storm watch (NOAA NHC) ---------- */
function renderStorms(s) {
  const statusEl = $("stormStatus");
  const srcEl = $("stormSrc");
  if (!s) {
    statusEl.textContent = "no data";
    srcEl.textContent = "";
    $("stormList").innerHTML = "";
    return;
  }
  srcEl.textContent = s.source || "—";
  srcEl.style.color = C.dim;

  // overall status pill colour
  statusEl.textContent = (s.overall_status || "—") + ` · ${s.count || 0}`;
  if (s.overall_tag === "CLEAR") {
    statusEl.style.background = "rgba(41,196,111,.15)";
    statusEl.style.color = C.up;
  } else if (s.overall_tag === "DISTANT") {
    statusEl.style.background = "#1c2533";
    statusEl.style.color = C.dim;
  } else if (s.overall_tag === "WATCH") {
    statusEl.style.background = "rgba(243,167,18,.15)";
    statusEl.style.color = C.accent;
  } else {
    statusEl.style.background = "rgba(240,85,106,.15)";
    statusEl.style.color = C.down;
  }

  if (!s.storms || s.storms.length === 0) {
    $("stormList").innerHTML = `<div class="storm-empty">
      <div class="se-big">🌤  No active tropical cyclones globally</div>
      <div class="se-sub">NOAA NHC + JTWC both clean across all 6 basins
        (Atlantic, Eastern Pacific, Central Pacific, Western Pacific,
        North Indian Ocean, Southern Hemisphere).</div>
      <div class="se-meta">Tracking ${s.n_refineries_tracked || 0} refineries
        across ${s.n_regions_tracked || 0} regions
        (~${((s.total_capacity_kbpd || 0)/1000).toFixed(1)} M bpd total capacity).
        Alert radius: ${s.risk_radius_nm} nm. Refresh every 10 min.</div>
    </div>`;
    return;
  }

  $("stormList").innerHTML = s.storms.map((st) => {
    const imp = st.oil_impact || {};
    // category colour
    let catColor = C.dim;
    const cat = st.category || "";
    if (cat.startsWith("Cat 5") || cat.startsWith("Cat 4")) catColor = C.down;
    else if (cat.startsWith("Cat 3")) catColor = "#ff6e3a";
    else if (cat.startsWith("Cat 2") || cat.startsWith("Cat 1")) catColor = C.accent;
    else if (cat === "TS") catColor = "#e8c547";
    else catColor = C.blue;

    // tag colour
    let tagColor = C.dim, tagBg = "#1c2533";
    if (imp.tag === "PRODUCTS BULLISH" || imp.tag === "CRUDE BULLISH") {
      tagColor = C.down; tagBg = "rgba(240,85,106,.15)";
    } else if (imp.tag === "WATCH") {
      tagColor = C.accent; tagBg = "rgba(243,167,18,.15)";
    } else {
      tagColor = C.up; tagBg = "rgba(41,196,111,.10)";
    }

    const refList = (imp.refineries_at_risk || []).slice(0, 5).map((r) =>
      `<div class="ref-row">
        <span class="rn">${r.name}${r.region ? ` <i style="color:var(--dim)">[${r.region}]</i>` : ""}</span>
        <span class="rd">${r.distance_nm} nm</span>
        <span class="rc">${r.capacity_kbpd} kbpd</span>
      </div>`).join("") || '<div class="ref-row dim">no refineries within 150 nm</div>';
    // Per-region at-risk breakdown
    const regionList = Object.entries(imp.regions_at_risk || {}).map(
      ([region, info]) => `<span class="region-tag">${region}: <b>${info.capacity_kbpd} kbpd</b></span>`
    ).join(" ");

    const mvDir = st.movement_dir_deg != null ? `${st.movement_dir_deg}°` : "—";
    const mvSpd = st.movement_speed_kt != null ? `${st.movement_speed_kt} kt` : "—";
    const pressureTxt = st.pressure_mb ? `${st.pressure_mb} mb` : "—";
    const updatedTxt = st.last_update
      ? new Date(st.last_update).toLocaleString(undefined,
          { dateStyle: "medium", timeStyle: "short" })
      : "—";
    const link = st.public_advisory_url
      ? `<a href="${st.public_advisory_url}" target="_blank" rel="noopener" class="adv-link">NHC advisory ↗</a>` : "";

    const basinTag = st.basin_name
      ? `<span class="sc-basin">${st.basin_name} · ${st.source || ""}</span>` : "";
    const latHemi = st.lat >= 0 ? "N" : "S";
    const lonHemi = st.lon >= 0 ? "E" : "W";

    return `<div class="storm-card">
      <div class="sc-head">
        <span class="sc-name">${st.name}</span>
        <span class="sc-cat" style="color:${catColor};border-color:${catColor}">${st.category}
          · ${Math.round(st.intensity_kt)}kt</span>
        <span class="sc-tag" style="color:${tagColor};background:${tagBg}">${imp.tag || "—"}</span>
      </div>
      ${basinTag}
      <div class="sc-pos">
        <span>📍 ${Math.abs(st.lat)}°${latHemi}, ${Math.abs(st.lon)}°${lonHemi}</span>
        <span>↗ moving ${mvDir} @ ${mvSpd}</span>
        <span>⏲ ${pressureTxt}</span>
        <span class="dim">updated ${updatedTxt}</span>
        ${link}
      </div>
      <div class="sc-impact">${imp.hint || ""}</div>
      ${regionList ? `<div class="sc-regions">${regionList}</div>` : ""}
      <div class="sc-stats">
        <div class="sc-stat">
          <b>${(imp.refining_capacity_at_risk_kbpd || 0).toLocaleString()}</b>
          <span>kbpd refining at risk (${imp.refining_capacity_at_risk_pct || 0}% of US Gulf)</span>
        </div>
        <div class="sc-stat">
          <b>${(imp.production_at_risk_kbpd || 0).toLocaleString()}</b>
          <span>kbpd offshore production at risk</span>
        </div>
        <div class="sc-stat">
          <b>${imp.nearest_distance_nm != null ? imp.nearest_distance_nm : "—"} nm</b>
          <span>nearest refinery: ${imp.nearest_refinery || "—"}</span>
        </div>
      </div>
      <div class="sc-refs">${refList}</div>
    </div>`;
  }).join("");
}

/* ---------- T12 Choke Point Watch (live AIS) ---------- */
function renderChokePoints(t) {
  const statusEl = $("chokeStatus");
  const totalsEl = $("chokeTotals");
  const target = $("chokePoints");
  if (!t) {
    target.innerHTML =
      '<div class="paperempty">No AIS connection yet.</div>';
    return;
  }
  const chokes = t.chokes || {};
  const names = Object.keys(chokes);
  if (!names.length) {
    target.innerHTML = '<div class="paperempty">No choke points configured.</div>';
    return;
  }

  statusEl.textContent = (t.status || "—").slice(0, 40);
  if ((t.status || "").startsWith("live")) {
    statusEl.style.background = "rgba(41,196,111,.15)";
    statusEl.style.color = C.up;
  } else {
    statusEl.style.background = "#1c2533";
    statusEl.style.color = C.dim;
  }
  const totals = t.totals || {};
  totalsEl.textContent =
    `${totals.vessels_in_chokes || 0} vessels in 7 chokes · `
    + `${totals.messages_seen || 0} AIS msgs total`;
  totalsEl.style.color = C.dim;

  // sort by confirmed tanker count desc — busiest choke first
  names.sort((a, b) =>
    (chokes[b].confirmed_tankers || 0) - (chokes[a].confirmed_tankers || 0));

  target.innerHTML = names.map((name) => {
    const c = chokes[name];
    const tankers = c.confirmed_tankers || 0;
    const total = c.total || 0;
    const cap = c.est_capacity_mbbl || 0;
    const baseline = c.baseline_flow_mbpd || 0;
    const anch = c.anchored || 0;
    const under = c.underway || 0;
    const anchPct = total > 0 ? (anch / total) * 100 : 0;
    const anchColor = anchPct >= 60 ? C.down
                    : anchPct >= 30 ? C.accent : C.up;

    const samples = (c.samples || []).slice(0, 4).map((s) => {
      const tag = s.anchored ? "⚓ ANCH" : "→ UWAY";
      const tagColor = s.anchored ? C.accent : C.up;
      const sog = s.sog != null ? `${s.sog}kn` : "";
      const cog = s.cog != null ? `@${s.cog}°` : "";
      return `<div class="choke-sample">
        <span style="color:${tagColor}">${tag}</span>
        <span class="cn">${s.name}</span>
        <span class="cs">${sog} ${cog}</span>
      </div>`;
    }).join("") || '<div class="choke-sample dim">no recent reports</div>';

    // size mix breakdown (only nonzero classes)
    const mix = c.size_mix || {};
    const mixHtml = Object.entries(mix).filter(([,n]) => n>0)
      .map(([k,n]) => `<span class="sz-${k.toLowerCase().replace(/[\/ ]/g,'-')}">${k}: ${n}</span>`).join(" ");
    // 24h flow line
    const tr24 = c.transits_24h != null ? c.transits_24h : 0;
    const flow = c.throughput_mbpd_24h != null ? c.throughput_mbpd_24h : 0;
    const pct = c.throughput_pct_baseline;
    const pctColor = pct == null ? C.dim
                   : pct >= 90 ? C.up
                   : pct >= 50 ? C.accent : C.down;
    const pctTxt = pct != null ? `${pct}% of baseline` : "—";
    // laden vs ballast
    const laden = c.laden_count || 0;
    const ballast = c.ballast_count || 0;

    return `<div class="choke-card">
      <div class="cp-head">
        <span class="cp-name">${name}</span>
        <span class="cp-tankers" title="Confirmed tankers in choke right now">${tankers}</span>
      </div>
      <div class="cp-stats">
        <div><b>${cap}</b><span>M bbl in box</span></div>
        <div><b>${baseline}</b><span>Mbpd baseline</span></div>
        <div><b>${total}</b><span>total vessels</span></div>
      </div>
      <div class="cp-flow24" style="border-top:1px solid var(--line);padding-top:6px;margin-top:2px">
        <div><b>${tr24}</b><span>transits 24h</span></div>
        <div><b>${flow}</b><span>Mbpd 24h</span></div>
        <div style="color:${pctColor}"><b>${pctTxt}</b></div>
      </div>
      ${mixHtml ? `<div class="cp-mix">${mixHtml}</div>` : ""}
      <div class="cp-bar">
        <span class="cp-anch" style="width:${anchPct}%;background:${anchColor}"></span>
      </div>
      <div class="cp-flow">
        <span style="color:${anchColor}">⚓ ${anch} anchored</span>
        <span>→ ${under} underway</span>
      </div>
      ${(laden+ballast)>0 ? `<div class="cp-laden">⚖ ${laden} laden · ${ballast} ballast (from AIS draught)</div>` : ""}
      <div class="cp-ctx">${c.context || ""}</div>
      <div class="cp-samples">${samples}</div>
    </div>`;
  }).join("");
}

/* ---------- T13 Floating storage detection ---------- */
function renderFloatingStorage(t) {
  const list = (t && t.floating_storage) || [];
  const totals = (t && t.totals) || {};
  const msgs    = totals.messages_seen || 0;
  const vessels = totals.vessels_in_zones || 0;
  const tankers = totals.tanker_types_known || 0;
  const total   = list.reduce((sum, v) => sum + (v.capacity_bbl || 0), 0);

  $("floatingStatus").textContent =
    `${list.length} vessels · ${(total/1_000_000).toFixed(1)} M bbl · `
    + `AIS: ${msgs.toLocaleString()} msgs · ${vessels.toLocaleString()} ships in zones · `
    + `${tankers.toLocaleString()} tankers classified`;
  $("floatingStatus").style.color = list.length ? C.accent : C.dim;

  if (!list.length) {
    $("floatingStorage").innerHTML =
      `<div class="paperempty">No tankers anchored &gt;2 days at our 10 zones yet.
       <br><br>
       <b>AIS feed is live:</b> ${msgs.toLocaleString()} messages received,
       ${vessels.toLocaleString()} ships currently in tracked zones,
       ${tankers.toLocaleString()} confirmed as tankers via ShipStaticData.
       <br><br>
       Floating-storage candidates require <b>continuous observation ≥48h</b>
       at the same hub — they'll surface as the rolling window fills.
       A rising count would be a bullish supply signal.</div>`;
    return;
  }
  $("floatingStorage").innerHTML = list.slice(0, 30).map((v) => {
    const days = v.anchored_days || 0;
    const ageColor = days > 30 ? C.down
                   : days > 14 ? C.accent : C.up;
    return `<div class="fs-row">
      <span class="fs-name">${v.name}</span>
      <span class="fs-size">${v.size_class}</span>
      <span class="fs-zone">${v.zone}</span>
      <span class="fs-days" style="color:${ageColor}">${days.toFixed(1)} days</span>
      <span class="fs-cap">${(v.capacity_bbl/1_000_000).toFixed(2)}M bbl</span>
    </div>`;
  }).join("");
}

/* ---------- T14 STS rendezvous candidates ---------- */
function renderStsCandidates(t) {
  const list = (t && t.sts_candidates) || [];
  const totals = (t && t.totals) || {};
  const msgs    = totals.messages_seen || 0;
  const vessels = totals.vessels_in_zones || 0;
  const tankers = totals.tanker_types_known || 0;

  $("stsStatus").textContent =
    `${list.length} pair${list.length===1?"":"s"} flagged · `
    + `AIS: ${msgs.toLocaleString()} msgs · ${tankers.toLocaleString()} tankers tracked`;
  $("stsStatus").style.color = list.length ? C.accent : C.dim;

  if (!list.length) {
    $("stsCandidates").innerHTML =
      `<div class="paperempty">No STS rendezvous candidates right now across our 10 zones
       (incl. Lakonikos / Ceuta / Sohar dark-fleet hotspots).
       <br><br>
       <b>AIS feed is live:</b> ${msgs.toLocaleString()} messages received,
       ${vessels.toLocaleString()} ships currently in tracked zones,
       ${tankers.toLocaleString()} confirmed as tankers.
       <br><br>
       STS pairs surface in bursts — two tankers within 600m at &lt;1.5kn.
       Activity rises around sanctioned-cargo transfer windows.</div>`;
    return;
  }
  $("stsCandidates").innerHTML = list.slice(0, 25).map((p) => {
    return `<div class="sts-row">
      <span class="sts-loc">${p.location}</span>
      <span class="sts-ab">
        <b>${p.vessel_a.name}</b> <span class="sts-sz">${p.vessel_a.size_class}</span>
        <span class="sts-sep">↔</span>
        <b>${p.vessel_b.name}</b> <span class="sts-sz">${p.vessel_b.size_class}</span>
      </span>
      <span class="sts-dist">${p.distance_m} m apart</span>
    </div>`;
  }).join("");
}

/* ---------- T10 Tanker watch (live AIS) ---------- */
function renderTankers(t) {
  const statusEl = $("aisStatus");
  const totalsEl = $("aisTotals");
  if (!t) {
    statusEl.textContent = "no data";
    totalsEl.textContent = "";
    $("tankerZones").innerHTML =
      '<div class="paperempty">Waiting for first AIS message...</div>';
    return;
  }

  // status pill colour
  statusEl.textContent = t.status || "—";
  if ((t.status || "").startsWith("live")) {
    statusEl.style.background = "rgba(41,196,111,.15)";
    statusEl.style.color = C.up;
  } else if ((t.status || "").startsWith("reconnect")) {
    statusEl.style.background = "rgba(243,167,18,.15)";
    statusEl.style.color = C.accent;
  } else {
    statusEl.style.background = "#1c2533";
    statusEl.style.color = C.dim;
  }

  const totals = t.totals || {};
  totalsEl.textContent =
    `${totals.tanker_types_known || 0} confirmed tankers · `
    + `${totals.vessels_in_zones || 0} total in boxes · `
    + `${totals.messages_seen || 0} AIS msgs`;

  const zones = t.zones || {};
  const zoneNames = Object.keys(zones);
  if (zoneNames.length === 0) {
    $("tankerZones").innerHTML =
      '<div class="paperempty">No zones configured.</div>';
    return;
  }

  // sort by total descending so busy hubs show first
  zoneNames.sort((a, b) => (zones[b].total || 0) - (zones[a].total || 0));

  // Sort zones by confirmed tanker count descending — the real signal
  zoneNames.sort((a, b) =>
    (zones[b].confirmed_tankers || 0) - (zones[a].confirmed_tankers || 0));

  $("tankerZones").innerHTML = zoneNames.map((name) => {
    const z = zones[name];
    const confirmed = z.confirmed_tankers || 0;
    const unknown = z.unknown_type || 0;
    const total = z.total || 0;
    const anch = z.anchored || 0;
    const under = z.underway || 0;
    const anchPct = total > 0 ? (anch / total) * 100 : 0;
    // colour: heavy anchoring (>60%) flags congestion / storage
    const anchColor = anchPct >= 60 ? C.down
                    : anchPct >= 30 ? C.accent : C.up;

    const samples = (z.samples || []).map((s) => {
      const tag = s.anchored ? "⚓ ANCH" : "→ UWAY";
      const tagColor = s.anchored ? C.accent : C.up;
      const sog = s.sog != null ? `${s.sog}kn` : "";
      return `<div class="tanker-sample">
        <span style="color:${tagColor}">${tag}</span>
        <span class="tn">${s.name}</span>
        <span class="ts">${sog}</span>
      </div>`;
    }).join("") || '<div class="tanker-sample dim">no recent reports</div>';

    return `<div class="tanker-zone">
      <div class="tz-head">
        <span class="tz-name">${name}</span>
        <span class="tz-total" title="Confirmed tankers (AIS type 80-89)">${confirmed}</span>
      </div>
      <div class="tz-bar">
        <span class="tz-anch" style="width:${anchPct}%;background:${anchColor}"></span>
      </div>
      <div class="tz-stats">
        <span style="color:${anchColor}">⚓ ${anch} anchored</span>
        <span>→ ${under} underway</span>
      </div>
      <div class="tz-meta">
        ${confirmed} confirmed tankers · ${total} total vessels in box · ${unknown} unidentified
      </div>
      <div class="tz-samples">${samples}</div>
    </div>`;
  }).join("");
}

setInterval(() => {
  $("clock").textContent = new Date().toLocaleTimeString("en-GB");
}, 1000);

// Theme toggle wiring — has to happen AFTER initTheme has run, which it
// did at module top. Clicking swaps the data-theme attribute, persists
// the choice in localStorage, and re-paints all live charts.
(function wireThemeToggle() {
  const btn = document.getElementById("themeToggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme")
                    || "dark";
    setTheme(current === "light" ? "dark" : "light");
  });
})();

connect();
