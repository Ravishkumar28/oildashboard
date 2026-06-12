"""Pure-Python market indicators: moving averages, RSI, z-score, correlation,
covariance and covariance matrices. No numpy dependency to keep installs light."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    """Sample standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def moving_average(values: Sequence[float], window: int) -> List[Optional[float]]:
    """Simple moving average. Positions with insufficient history are None."""
    out: List[Optional[float]] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(None)
        else:
            out.append(mean(values[i + 1 - window : i + 1]))
    return out


def rsi(values: Sequence[float], period: int = 14) -> List[Optional[float]]:
    """Relative Strength Index using Wilder's smoothing."""
    out: List[Optional[float]] = [None] * len(values)
    if len(values) <= period:
        return out

    gains, losses = [], []
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = mean(gains[:period])
    avg_loss = mean(losses[:period])

    def to_rsi(g: float, l: float) -> float:
        if l == 0:
            return 100.0
        rs = g / l
        return 100.0 - (100.0 / (1.0 + rs))

    out[period] = to_rsi(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        g = gains[i - 1]
        l = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        out[i] = to_rsi(avg_gain, avg_loss)
    return out


def bollinger_bands(values: Sequence[float], window: int = 20,
                    k: float = 2.0) -> tuple:
    """Bollinger Bands. Returns three lists (middle, upper, lower) aligned
    with ``values`` — positions with insufficient history are None.

    Middle = SMA(window); Upper/Lower = Middle ± k × population stddev
    (population stddev is the textbook BB convention, not sample stddev)."""
    n = len(values)
    middle: List[Optional[float]] = [None] * n
    upper: List[Optional[float]] = [None] * n
    lower: List[Optional[float]] = [None] * n
    for i in range(n):
        if i + 1 < window:
            continue
        w = values[i + 1 - window : i + 1]
        m = sum(w) / window
        var = sum((v - m) ** 2 for v in w) / window
        sd = math.sqrt(var)
        middle[i] = m
        upper[i] = m + k * sd
        lower[i] = m - k * sd
    return middle, upper, lower


def vwap(prices: Sequence[float], volumes: Sequence[float],
         window: int) -> List[Optional[float]]:
    """Rolling N-day Volume-Weighted Average Price.

    ``VWAP_t = Σ(price_i * volume_i) / Σ(volume_i)`` over the last ``window``
    bars. Weights big-volume days more, so it tracks where the bulk of
    actual trading happened — a complement to time-weighted EMA/SMA which
    weights every bar equally regardless of activity."""
    n = min(len(prices), len(volumes))
    out: List[Optional[float]] = [None] * n
    if n < window:
        return out
    for i in range(window - 1, n):
        p = prices[i + 1 - window: i + 1]
        v = volumes[i + 1 - window: i + 1]
        tot_v = sum(v)
        if tot_v <= 0:
            continue
        out[i] = sum(p[j] * v[j] for j in range(window)) / tot_v
    return out


def ema(values: Sequence[float], period: int) -> List[Optional[float]]:
    """Exponential moving average. EMA_t = α·price_t + (1-α)·EMA_{t-1}
    with α = 2/(period+1). Seeded with the SMA of the first ``period``
    values; earlier positions are None."""
    n = len(values)
    out: List[Optional[float]] = [None] * n
    if n < period:
        return out
    alpha = 2.0 / (period + 1)
    sma_seed = sum(values[:period]) / period
    out[period - 1] = sma_seed
    for i in range(period, n):
        prev = out[i - 1] if out[i - 1] is not None else sma_seed
        out[i] = alpha * values[i] + (1.0 - alpha) * prev
    return out


def zscore(values: Sequence[float], window: Optional[int] = None) -> float:
    """Z-score of the most recent value vs a trailing window (or full series)."""
    series = list(values) if window is None else list(values)[-window:]
    if len(series) < 2:
        return 0.0
    sd = stdev(series)
    if sd == 0:
        return 0.0
    return (series[-1] - mean(series)) / sd


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation coefficient over the overlapping tail of x and y."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    xs, ys = list(x)[-n:], list(y)[-n:]
    mx, my = mean(xs), mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def covariance(x: Sequence[float], y: Sequence[float]) -> float:
    """Sample covariance over the overlapping tail of x and y."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    xs, ys = list(x)[-n:], list(y)[-n:]
    mx, my = mean(xs), mean(ys)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (n - 1)


def returns(values: Sequence[float]) -> List[float]:
    """Percentage returns series (used so covariance reflects co-movement,
    not raw price scale)."""
    out: List[float] = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        out.append((values[i] - prev) / prev if prev else 0.0)
    return out


def covariance_matrix(series: Dict[str, Sequence[float]]) -> Dict[str, object]:
    """Correlation matrix of the return series of each named input.

    Returns labels plus a row-major matrix of correlations in [-1, 1].
    Uses equal-weighted (Pearson) correlation — slow-moving by design."""
    labels = list(series.keys())
    rets = {k: returns(v) for k, v in series.items()}
    matrix: List[List[float]] = []
    for a in labels:
        row: List[float] = []
        for b in labels:
            row.append(round(correlation(rets[a], rets[b]), 3))
        matrix.append(row)
    return {"labels": labels, "matrix": matrix}


def ewma_correlation_matrix(series: Dict[str, Sequence[float]],
                            lam: float = 0.94) -> Dict[str, object]:
    """Exponentially-weighted correlation matrix (RiskMetrics convention).

    Each observation's weight decays by ``lam`` per step:
        σ²_t   = lam · σ²_{t-1}  + (1-lam) · r²_t
        σ_xy_t = lam · σ_xy_{t-1} + (1-lam) · r_x · r_y

    With lam=0.94 the effective lookback is ~32 days, but the most recent
    observation carries ~6 % weight instead of 0.8 % under equal-weighted
    120-day windows. Much more responsive to current regime — and the live
    tick value visibly shifts the correlations each render."""
    labels = list(series.keys())
    rets = {k: returns(v) for k, v in series.items()}
    n = min((len(r) for r in rets.values()), default=0)
    matrix: List[List[float]] = []
    if n < 2:
        for _ in labels:
            matrix.append([1.0 if i == j else 0.0
                           for j, i in enumerate(range(len(labels)))])
        return {"labels": labels, "matrix": matrix}

    for a in labels:
        row: List[float] = []
        ra = list(rets[a])[-n:]
        for b in labels:
            rb = list(rets[b])[-n:]
            var_a = ra[0] * ra[0]
            var_b = rb[0] * rb[0]
            cov_ab = ra[0] * rb[0]
            for k in range(1, n):
                var_a = lam * var_a + (1 - lam) * ra[k] * ra[k]
                var_b = lam * var_b + (1 - lam) * rb[k] * rb[k]
                cov_ab = lam * cov_ab + (1 - lam) * ra[k] * rb[k]
            denom = math.sqrt(var_a * var_b)
            row.append(round(cov_ab / denom, 3) if denom > 0 else 0.0)
        matrix.append(row)
    return {"labels": labels, "matrix": matrix}
