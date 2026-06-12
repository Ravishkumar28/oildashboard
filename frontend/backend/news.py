"""Live oil-news feed from real public RSS sources.

Aggregates four free RSS feeds — OilPrice.com, Google News (two queries),
and Hellenic Shipping News — into one rolling buffer of recent headlines.
Every item is region- and sentiment-tagged.

There is NO simulated/synthetic content. If all feeds fail temporarily the
panel just goes briefly empty rather than show invented headlines."""
from __future__ import annotations

import asyncio
import datetime as dt
import email.utils
import html
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from typing import Deque, Dict, List, Optional

import httpx

import sentiment

# Only surface items whose actual <pubDate> is within the last 6 hours.
# Was 24h, but Google News RSS reports its INDEXING time as <pubDate>,
# not the article's actual publication date — so old syndicated articles
# slip through with fresh timestamps. A 6h window cuts most of those.
MAX_AGE_SECONDS = 6 * 3600

# URL-date heuristic: most news URLs include /YYYY/MM/ or YYYY-MM-DD in
# their path. If we extract a date older than this window, drop the item
# even when RSS pubDate looks fresh (catches re-indexed old articles).
URL_DATE_MAX_AGE_DAYS = 14
_URL_DATE_RE = re.compile(r"/(20\d{2})[/\-_](\d{1,2})(?:[/\-_](\d{1,2}))?")
# Headline year heuristic: catches "...June 2025" or "in 2025" patterns
# in the title that contradict a fresh-looking RSS pubDate.
_HEADLINE_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _url_is_recent(url: str) -> bool:
    """Return False when the URL path encodes a date older than
    URL_DATE_MAX_AGE_DAYS. Returns True if no date pattern is found
    (cannot determine — pass through)."""
    if not url:
        return True
    m = _URL_DATE_RE.search(url)
    if not m:
        return True
    try:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3)) if m.group(3) else 15
        url_date = dt.date(year, month, max(1, min(28, day)))
    except (ValueError, TypeError):
        return True
    cutoff = dt.date.today() - dt.timedelta(days=URL_DATE_MAX_AGE_DAYS)
    return url_date >= cutoff


def _headline_year_is_current(headline: str) -> bool:
    """If the headline contains an explicit year, require it to be the
    current or immediately previous year. Catches headlines like
    'Crude oil prices June 2025' being shown on 2026-05-28."""
    if not headline:
        return True
    years = [int(y) for y in _HEADLINE_YEAR_RE.findall(headline)]
    if not years:
        return True
    current_year = dt.date.today().year
    return max(years) >= current_year - 1

# (display name, feed url) — display name is overridden for Google News
# items because each Google News item already names its underlying publisher
RSS_FEEDS = [
    ("OilPrice",
     "https://oilprice.com/rss/main"),
    ("Google News",
     "https://news.google.com/rss/search?q=crude+oil+price"
     "&hl=en-US&gl=US&ceid=US:en"),
    ("Google News",
     "https://news.google.com/rss/search?q=WTI+OPEC+brent"
     "&hl=en-US&gl=US&ceid=US:en"),
    ("Hellenic Shipping",
     "https://www.hellenicshippingnews.com/feed/"),
    # Financial Juice publishes a real-time public RSS at the URL below —
    # same data as the (paid) Apify scraper "akash9078/financialjuice-scraper".
    # 100 most-recent items returned; we filter to oil-relevant headlines
    # via FJ_OIL_KEYWORDS in _is_oil_relevant() below because FJ covers ALL
    # macro/finance not just oil.
    ("FinancialJuice",
     "https://www.financialjuice.com/feed.ashx?xy=rss"),
]

# Headlines from feeds that mix oil + general finance (like Financial Juice)
# are filtered to those containing one of these terms. Case-insensitive.
FJ_OIL_KEYWORDS = (
    "oil", "crude", "brent", "wti", "opec", "refin", "gasoline", "diesel",
    "petroleum", "pipeline", "shale", "tanker", "lng", "natgas", "natural gas",
    "rig count", "drilling", "cushing", "spr", "strategic petroleum reserve",
    "saudi", "russia", "iran", "iraq", "uae", "qatar", "kuwait",
    "venezuela", "nigeria", "libya", "angola",
    "energy", "fuel", "barrel", "bpd", "mbpd",
)


def _strip_fj_prefix(title: str) -> str:
    """Remove the 'FinancialJuice: ' prefix that FJ stamps on every item
    title — they put their brand name first which clutters the panel."""
    if title.startswith("FinancialJuice:"):
        return title[len("FinancialJuice:"):].strip()
    return title


def _is_oil_relevant(headline: str) -> bool:
    """Cheap keyword test to keep only oil-relevant items from feeds
    that aggregate broader finance (FinancialJuice). Returns True if
    ANY oil keyword appears in the headline."""
    lo = headline.lower()
    return any(kw in lo for kw in FJ_OIL_KEYWORDS)

# Twitter killed its free API in 2023 — paid only ($100/mo minimum) and
# scraping x.com is blocked too. The honest workaround is to search Google
# News for articles BY OR ABOUT these key oil analysts and run the same
# VADER sentiment classifier on those headlines.
# Each analyst has multiple search variants because Google News indexing
# is uneven — a single restrictive query (e.g. "Amena Bakr" AND "oil")
# returned 0-1 items, while broader OR queries surface dozens. Results
# from all variants are merged and deduped per-analyst.
ANALYST_FEEDS: Dict[str, List[str]] = {
    "Amena Bakr": [
        # Drop the +oil filter — Bakr's entire beat IS oil/OPEC so every
        # mention by her is relevant. Quoted phrase only.
        "https://news.google.com/rss/search?q=%22Amena+Bakr%22"
        "&hl=en-US&gl=US&ceid=US:en",
        # Argus Media (her employer) — articles by her or about her work
        "https://news.google.com/rss/search?q=%22Amena+Bakr%22+Argus"
        "&hl=en-US&gl=US&ceid=US:en",
        # AGBI publishes her oped column — direct RSS catches new ones
        "https://www.agbi.com/feed/",
    ],
    "Javier Blas": [
        # Same loosening — Blas is Bloomberg's commodities columnist;
        # every mention is energy/commodities related.
        "https://news.google.com/rss/search?q=%22Javier+Blas%22"
        "&hl=en-US&gl=US&ceid=US:en",
        # Bloomberg Opinion specifically — he posts there daily
        "https://news.google.com/rss/search?q=%22Javier+Blas%22+Bloomberg"
        "&hl=en-US&gl=US&ceid=US:en",
    ],
    # Key MUST match twitter_nitter.ANALYST_HANDLES exactly so per-analyst
    # Nitter→Google-News fallback merges into the same dict slot.
    "Trump": [
        # Trump search variants — capture his oil/energy statements
        "https://news.google.com/rss/search?q=Trump+OPEC+OR+crude+OR+drilling"
        "&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Trump+oil+price+OR+gasoline"
        "&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Trump+energy+policy"
        "&hl=en-US&gl=US&ceid=US:en",
    ],
}

REGIONS = {
    "Middle East": ["opec", "saudi", "iran", "iraq", "uae", "gulf", "qatar",
                    "kuwait", "yemen", "strait of hormuz", "israel"],
    "North America": ["wti", "cushing", "permian", "u.s.", "us ", "canada",
                      "shale", "texas", "gulf of mexico", "eia"],
    "Europe": ["brent", "north sea", "europe", "uk", "germany", "norway"],
    "Asia": ["china", "india", "japan", "asia", "singapore", "teapot"],
    "Russia/CIS": ["russia", "moscow", "urals", "kazakhstan", "sanction"],
    "Latin America": ["venezuela", "brazil", "mexico", "guyana", "argentina"],
    "Africa": ["nigeria", "libya", "angola", "algeria"],
}

def _classify_region(text: str) -> str:
    t = text.lower()
    best, score = "Global", 0
    for region, keys in REGIONS.items():
        hits = sum(1 for k in keys if k in t)
        if hits > score:
            best, score = region, hits
    return best


def _parse_pub_date(item) -> Optional[float]:
    """Extract RSS ``<pubDate>`` (RFC 822 format) and return a UTC unix
    timestamp. Returns None if the field is missing or unparseable —
    such items are dropped rather than back-dated to ``time.time()``."""
    pub_el = item.find("pubDate")
    if pub_el is None or not pub_el.text:
        # Atom feeds use <published>; try that as fallback
        pub_el = item.find("published")
    if pub_el is None or not pub_el.text:
        return None
    try:
        dt_obj = email.utils.parsedate_to_datetime(pub_el.text.strip())
        return dt_obj.timestamp()
    except Exception:
        return None


def _split_google_news_title(title: str) -> tuple:
    """Google News items use ``"Headline - Publisher"``. Split on the last
    ' - ' to recover the underlying publisher (BBC, Reuters, FT, etc.).
    Falls back to ('title', 'Google News') if the pattern isn't present."""
    if " - " in title:
        head, _, src = title.rpartition(" - ")
        # require src to be plausibly short (publishers are short)
        if 0 < len(src) <= 40:
            return head.strip(), src.strip()
    return title, "Google News"


class NewsFeed:
    """Rolling buffer of recent, region-tagged headlines from real RSS."""

    def __init__(self, capacity: int = 60) -> None:
        self.items: Deque[Dict[str, object]] = deque(maxlen=capacity)
        self._seen: set = set()
        self._next_id = 1
        self.source = "pending first fetch"

    # ----- internal helpers ------------------------------------------- #
    def _add(self, item: Optional[Dict[str, object]]) -> bool:
        if not item:
            return False
        key = str(item["headline"]).lower()[:90]
        if key in self._seen:
            return False
        self._seen.add(key)
        item["id"] = self._next_id
        self._next_id += 1
        self.items.appendleft(item)
        return True

    # ----- live RSS fetch --------------------------------------------- #
    async def _fetch_rss(self) -> List[Dict[str, object]]:
        """Fetch + parse all RSS feeds. Runs only VADER (sync, ~0.1ms each)
        on candidate items; FinBERT is deferred to ``refresh()``'s backfill
        phase so a slow CPU doesn't make each refresh take 30 s and starve
        the rotation of the news buffer."""
        out: List[Dict[str, object]] = []
        cutoff = time.time() - MAX_AGE_SECONDS    # 24-hour freshness window
        headers = {"User-Agent": "Mozilla/5.0 (compatible; OilDeskDashboard/1.0)"}
        # Bumped from 8s to 20s — HF Spaces free tier has flaky outbound
        # network; 8s was timing out on Hellenic + Google News mirrors,
        # which made the buffer stop rotating.
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True,
                                     headers=headers) as client:
            for name, url in RSS_FEEDS:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    root = ET.fromstring(resp.content)
                except Exception:
                    continue
                for item in root.iter("item"):
                    title_el = item.find("title")
                    link_el = item.find("link")
                    if title_el is None or not title_el.text:
                        continue
                    # use ACTUAL publish time, not the time we fetched the
                    # item — and drop anything older than MAX_AGE_SECONDS
                    pub_ts = _parse_pub_date(item)
                    if pub_ts is None or pub_ts < cutoff:
                        continue
                    raw_title = html.unescape(
                        re.sub(r"\s+", " ", title_el.text).strip())
                    if name == "Google News":
                        headline, source = _split_google_news_title(raw_title)
                    elif name == "FinancialJuice":
                        # FJ stamps "FinancialJuice:" on every headline title —
                        # strip it, and require the headline to be oil-relevant
                        # (FJ covers all macro, we only want crude/energy news).
                        headline = _strip_fj_prefix(raw_title)
                        source = "FinancialJuice"
                        if not _is_oil_relevant(headline):
                            continue
                    else:
                        headline, source = raw_title, name
                    # Drop items where the URL path encodes an old date
                    # (Google News sometimes re-indexes year-old articles
                    # with a fresh pubDate — URL still has /2025/06/...)
                    url_str = (link_el.text or "") if link_el is not None else ""
                    if not _url_is_recent(url_str):
                        continue
                    # Drop items where the headline mentions a non-current
                    # year (e.g. "Crude price update June 2025" today)
                    if not _headline_year_is_current(headline):
                        continue
                    # VADER (fast, sync, rule-based) — always available
                    sent = sentiment.classify(headline)
                    # FinBERT runs in refresh()'s backfill loop, NOT here —
                    # keeps each fetch <2s instead of 30s+.
                    out.append({
                        "headline": headline,
                        "source": source,
                        "region": _classify_region(headline),
                        "sentiment": sent["label"],
                        "sentiment_score": sent["compound"],
                        "finbert_label": None,
                        "finbert_score": None,
                        "ts": pub_ts,        # real publish time, not fetch time
                        "url": (link_el.text or "") if link_el is not None
                               else "",
                        "live": True,
                    })
        return out

    async def refresh(self) -> int:
        """Pull live RSS items from all configured feeds. Returns the count
        of new (deduplicated) items added to the buffer."""
        added = 0
        try:
            for item in await self._fetch_rss():
                if self._add(item):
                    added += 1
        except Exception:
            return 0
        # Backfill FinBERT on items that don't have a score yet. Bounded to
        # ~6 items per refresh so a slow CPU doesn't stall the next refresh.
        # New items get scored over the next few refresh cycles instead of
        # all at once.
        if sentiment.finbert_ready():
            loop = asyncio.get_running_loop()
            processed = 0
            for it in self.items:
                if processed >= 6:
                    break
                if it.get("finbert_score") is not None:
                    continue
                try:
                    fin = await loop.run_in_executor(
                        None, sentiment.classify_finbert,
                        str(it.get("headline", "")))
                    if fin:
                        it["finbert_score"] = fin["compound"]
                        it["finbert_label"] = fin["label"]
                    processed += 1
                except Exception:
                    pass
        if self.items:
            ts = int(time.time())
            self.source = (
                f"Live RSS (last 24h, last fetch +{added} new @ "
                f"{time.strftime('%H:%M:%SZ', time.gmtime(ts))})")
        return added

    def _is_fresh(self, item: Dict[str, object]) -> bool:
        ts = item.get("ts")
        return ts is not None and float(ts) >= time.time() - MAX_AGE_SECONDS

    def snapshot(self) -> List[Dict[str, object]]:
        """Only items published within the last 24h. Items in the buffer
        that have aged out are hidden from view even though they may still
        occupy the dedup memory."""
        return [it for it in self.items if self._is_fresh(it)]

    def by_region(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for it in self.items:
            if not self._is_fresh(it):
                continue
            r = str(it["region"])
            counts[r] = counts.get(r, 0) + 1
        return counts

    def sentiment_summary(self) -> Dict[str, object]:
        """Aggregate VADER sentiment across the last-24h news items.
        Used to render the Market Mood badge on the news panel."""
        return sentiment.aggregate([it for it in self.items
                                     if self._is_fresh(it)])

    async def fetch_analyst_news(self) -> Dict[str, object]:
        """Fetch news mentioning each oil analyst across multiple search
        variants per analyst (Twitter API is paid-only; this is the
        free proxy). Items from all variants are merged, deduped by
        headline, and sorted newest-first per analyst."""
        headers = {"User-Agent": "Mozilla/5.0 (compatible; OilDeskDashboard/1.0)"}
        results: Dict[str, object] = {}
        # Analyst commentary moves slower than breaking news — keep a
        # 7-day window. (3 * MAX_AGE_SECONDS used to mean 3 days but
        # since MAX_AGE was tightened from 24h to 6h, that became 18h.)
        cutoff = time.time() - 7 * 24 * 3600
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True,
                                     headers=headers) as client:
            for analyst, urls in ANALYST_FEEDS.items():
                # Allow either a list of URLs (new format) or a single
                # URL string (back-compat with older configs).
                url_list = urls if isinstance(urls, list) else [urls]
                items: List[Dict[str, object]] = []
                seen_headlines: set = set()
                for url in url_list:
                    try:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        root = ET.fromstring(resp.content)
                    except Exception:
                        continue
                    for item in list(root.iter("item"))[:12]:
                        title_el = item.find("title")
                        link_el = item.find("link")
                        if title_el is None or not title_el.text:
                            continue
                        pub_ts = _parse_pub_date(item)
                        if pub_ts is None or pub_ts < cutoff:
                            continue
                        raw_title = html.unescape(
                            re.sub(r"\s+", " ", title_el.text).strip())
                        headline, source = _split_google_news_title(raw_title)
                        # For AGBI (non-Google-News feed), the headline
                        # won't have a "- Publisher" suffix to split.
                        if "agbi.com" in url and source == "Google News":
                            source = "AGBI"
                        dedup_key = headline.lower()[:90]
                        if dedup_key in seen_headlines:
                            continue
                        url_str = (link_el.text or "") if link_el is not None else ""
                        if not _url_is_recent(url_str):
                            continue
                        if not _headline_year_is_current(headline):
                            continue
                        seen_headlines.add(dedup_key)
                        sent = sentiment.classify(headline)
                        items.append({
                            "headline": headline,
                            "source": source,
                            "sentiment": sent["label"],
                            "sentiment_score": sent["compound"],
                            "ts": pub_ts,
                            "url": url_str,
                        })
                items.sort(key=lambda x: x["ts"], reverse=True)
                results[analyst] = {
                    "items": items[:8],
                    "summary": sentiment.aggregate(items),
                }
        return results
