"""Real tweets from Bakr / Blas / Trump via Nitter RSS.

Twitter's free API was killed in 2023 ($100/mo minimum since). Nitter is
an open-source Twitter front-end that exposes public timelines as RSS
feeds — completely free and TOS-grey (third-party scrapers, not your IP).

Several public Nitter instances have died or been blocked over the years.
We try a list of instances in order and fall back when one returns 429
(rate-limited), 5xx, or fails to connect. nitter.net is the primary —
when it's up, it serves the cleanest RSS.

Returns each tweet as a dict matching the existing analyst-news shape so
the frontend renderer needs no changes. VADER + (optional) FinBERT
sentiment is applied per tweet."""
from __future__ import annotations

import asyncio
import email.utils
import html
import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import httpx

import sentiment

# Try these Nitter instances in order. nitter.net is primary; the others
# are public mirrors with varying uptime. Adding more is cheap — we only
# pay the cost when a higher-priority instance is rate-limited or down.
NITTER_INSTANCES = [
    "nitter.net",
    "lightbrd.com",
    "nitter.privacydev.net",
    "nitter.poast.org",
    "nitter.fdn.fr",
    "nitter.unixfox.eu",
]

# Twitter handles (without leading @) for the three analysts we track.
# Order matters only for diagnostic display.
ANALYST_HANDLES: Dict[str, str] = {
    "Amena Bakr":  "Amena__Bakr",
    "Javier Blas": "JavierBlas",
    "Trump":       "realDonaldTrump",
}

# Cap how many recent tweets we surface per analyst (Nitter returns up
# to ~20; we keep the freshest few because old tweets bloat the panel)
MAX_TWEETS_PER_ANALYST = 8

# Only consider tweets younger than this for the "freshness" check —
# we still show older ones, just clamp the panel to recent activity.
MAX_AGE_SECONDS = 7 * 24 * 3600  # 7 days

_UA = ("Mozilla/5.0 (compatible; OilDeskDashboard/1.0; "
       "+https://Ravish28-oil-trading-desk.hf.space)")


def _parse_pub_date(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    try:
        dt_obj = email.utils.parsedate_to_datetime(text.strip())
        return dt_obj.timestamp()
    except Exception:
        return None


def _clean_text(raw: str) -> str:
    """Normalize whitespace + decode HTML entities."""
    return html.unescape(re.sub(r"\s+", " ", raw).strip())


async def _fetch_from_instance(client: httpx.AsyncClient,
                               instance: str, handle: str) -> Optional[bytes]:
    """Pull RSS bytes from a single Nitter instance. Returns None on any
    error or non-RSS body so the caller can fall back."""
    url = f"https://{instance}/{handle}/rss"
    try:
        resp = await client.get(url, timeout=12.0,
                                follow_redirects=True,
                                headers={"User-Agent": _UA})
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    body = resp.content
    if b"<rss" not in body[:300] and b"<feed" not in body[:300]:
        return None
    return body


async def _fetch_tweets_for_handle(client: httpx.AsyncClient,
                                    handle: str) -> List[Dict]:
    """Try Nitter instances in order until one returns real RSS.
    Returns parsed tweets newest-first, or empty list if every instance
    fails."""
    raw: Optional[bytes] = None
    used_instance = None
    for inst in NITTER_INSTANCES:
        raw = await _fetch_from_instance(client, inst, handle)
        if raw:
            used_instance = inst
            break
    if raw is None:
        return []

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []

    out: List[Dict] = []
    cutoff = time.time() - MAX_AGE_SECONDS
    for item in root.iter("item"):
        title_el = item.find("title")
        link_el = item.find("link")
        date_el = item.find("pubDate")
        if title_el is None or not title_el.text:
            continue
        pub_ts = _parse_pub_date(date_el.text if date_el is not None else None)
        if pub_ts is None or pub_ts < cutoff:
            continue
        text = _clean_text(title_el.text)
        # Detect retweets — Nitter prefixes them with "RT by @user: ..."
        is_retweet = text.startswith("RT by")
        link = (link_el.text or "") if link_el is not None else ""
        sent = sentiment.classify(text)
        out.append({
            "headline": text,
            "source": f"@{handle}",     # render handle as source
            "sentiment": sent["label"],
            "sentiment_score": sent["compound"],
            "ts": pub_ts,
            "url": link,
            "is_retweet": is_retweet,
            "nitter_instance": used_instance,
        })
    out.sort(key=lambda t: t["ts"], reverse=True)
    return out[:MAX_TWEETS_PER_ANALYST]


async def fetch_analyst_tweets() -> Dict[str, Dict]:
    """Pull real tweets for the three tracked analysts via Nitter RSS.

    Returns the same `{analyst_name: {items: [...], summary: {...}}}`
    shape the dashboard's analyst panel already consumes — so the UI
    renderer needs no changes. Just swap the data source."""
    results: Dict[str, Dict] = {}
    async with httpx.AsyncClient() as client:
        # Stagger the per-analyst fetches slightly so we don't hammer
        # any single Nitter instance with 3 concurrent identical calls
        tasks = {}
        for analyst, handle in ANALYST_HANDLES.items():
            tasks[analyst] = asyncio.create_task(
                _fetch_tweets_for_handle(client, handle))
        for analyst, task in tasks.items():
            try:
                tweets = await task
            except Exception:
                tweets = []
            results[analyst] = {
                "items": tweets,
                "summary": sentiment.aggregate(tweets),
            }
    return results
