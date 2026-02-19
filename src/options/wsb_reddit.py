"""Reddit WallStreetBets data fetcher via archive APIs (no API key needed).

Uses Pullpush (Pushshift successor) and Arctic Shift as data sources.
These archive Reddit posts and serve them via free JSON APIs without
the IP blocking that Reddit's own endpoints enforce on cloud servers.
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class WSBPost:
    """A single WSB post relevant to a symbol."""
    title: str
    body: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: float
    permalink: str
    flair: Optional[str] = None


@dataclass
class WSBSymbolData:
    """Aggregated WSB data for one symbol."""
    symbol: str
    posts: list[WSBPost]
    fetched_at: datetime
    total_mentions: int
    avg_score: float
    avg_upvote_ratio: float
    total_comments: int


class WSBRedditFetcher:
    """
    Fetches WSB post data via Reddit archive APIs.

    Uses multiple sources in order of preference:
    1. Pullpush (Pushshift successor) - most reliable
    2. Arctic Shift - backup archive
    3. Reddit direct (old.reddit.com) - fallback for non-blocked IPs

    Caches results per symbol with configurable TTL.
    """

    # Archive API endpoints (no auth needed, no IP blocking)
    PULLPUSH_URL = "https://api.pullpush.io/reddit/search/submission/"
    ARCTIC_SHIFT_URL = "https://arctic-shift.photon-reddit.com/api/posts"
    REDDIT_URL = "https://old.reddit.com/r/{subreddit}/search.json"

    def __init__(
        self,
        cache_ttl_seconds: int = 600,
        max_posts_per_symbol: int = 25,
        subreddit_name: str = "wallstreetbets",
        request_delay: float = 2.0,
    ):
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.max_posts = max_posts_per_symbol
        self.subreddit_name = subreddit_name
        self.request_delay = request_delay

        self._cache: dict[str, WSBSymbolData] = {}
        self._last_request_time: float = 0

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })

    @property
    def is_configured(self) -> bool:
        """Always True - no API keys needed for archive APIs."""
        return True

    def _rate_limit(self):
        """Enforce delay between requests to avoid being blocked."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()

    def fetch_symbol_data(self, symbol: str) -> Optional[WSBSymbolData]:
        """
        Fetch WSB posts mentioning a symbol.
        Returns cached data if fresh enough. Returns None if no posts found.
        """
        # Check cache
        if symbol in self._cache:
            cached = self._cache[symbol]
            age = datetime.now() - cached.fetched_at
            if age < self.cache_ttl:
                logger.debug(f"WSB cache hit for {symbol} (age: {age.seconds}s)")
                return cached if cached.posts else None

        try:
            posts = []
            seen_ids = set()

            # Search with $ prefix first (more specific), then plain symbol
            search_terms = [f"${symbol}", symbol] if len(symbol) > 2 else [f"${symbol}"]

            for term in search_terms:
                fetched = self._search_posts(term)
                for post_data in fetched:
                    post_id = post_data.get("id", "")
                    if post_id in seen_ids:
                        continue
                    seen_ids.add(post_id)

                    title = post_data.get("title", "")
                    body = post_data.get("selftext", "") or ""
                    text = f"{title} {body}"

                    if not self._symbol_in_text(symbol, text):
                        continue

                    posts.append(WSBPost(
                        title=title,
                        body=body[:500],
                        score=post_data.get("score", 0) or 0,
                        upvote_ratio=post_data.get("upvote_ratio", 0.5) or 0.5,
                        num_comments=post_data.get("num_comments", 0) or 0,
                        created_utc=post_data.get("created_utc", 0) or 0,
                        permalink=post_data.get("permalink", ""),
                        flair=post_data.get("link_flair_text"),
                    ))

                    if len(posts) >= self.max_posts:
                        break

                if len(posts) >= self.max_posts:
                    break

            # Build aggregated data (cache even empty results)
            data = WSBSymbolData(
                symbol=symbol,
                posts=posts,
                fetched_at=datetime.now(),
                total_mentions=len(posts),
                avg_score=sum(p.score for p in posts) / len(posts) if posts else 0,
                avg_upvote_ratio=sum(p.upvote_ratio for p in posts) / len(posts) if posts else 0,
                total_comments=sum(p.num_comments for p in posts),
            )
            self._cache[symbol] = data

            if posts:
                logger.info(
                    f"WSB data for {symbol}: {len(posts)} posts, "
                    f"avg_score={data.avg_score:.0f}, "
                    f"total_comments={data.total_comments}"
                )
            else:
                logger.info(f"No WSB posts found for {symbol} (searched: {search_terms})")

            return data if posts else None

        except Exception as e:
            logger.error(f"Failed to fetch WSB data for {symbol}: {e}")
            return None

    def _search_posts(self, query: str) -> list[dict]:
        """Search WSB posts using archive APIs with fallback chain."""
        self._rate_limit()

        # Try each source in order
        sources = [
            ("Pullpush", self._search_pullpush),
            ("ArcticShift", self._search_arctic_shift),
            ("Reddit", self._search_reddit_direct),
        ]

        for source_name, search_fn in sources:
            try:
                results = search_fn(query)
                if results is not None:
                    logger.debug(f"{source_name}: {len(results)} results for '{query}'")
                    return results
                # None means source failed, try next
            except Exception as e:
                logger.debug(f"{source_name} failed for '{query}': {e}")
                continue

        logger.warning(f"All WSB sources failed for '{query}'")
        return []

    def _search_pullpush(self, query: str) -> Optional[list[dict]]:
        """Search via Pullpush (Pushshift successor)."""
        after_ts = int((datetime.now() - timedelta(days=7)).timestamp())

        params = {
            "subreddit": self.subreddit_name,
            "q": query,
            "after": after_ts,
            "size": self.max_posts,
            "sort": "score",
            "sort_type": "desc",
        }

        try:
            response = self._session.get(
                self.PULLPUSH_URL, params=params, timeout=15
            )

            if response.status_code != 200:
                logger.debug(f"Pullpush returned {response.status_code}")
                return None

            data = response.json()
            # Pullpush returns {"data": [...]} with posts directly
            posts = data.get("data", [])
            if isinstance(posts, list):
                return posts
            return None

        except (requests.RequestException, ValueError) as e:
            logger.debug(f"Pullpush request failed: {e}")
            return None

    def _search_arctic_shift(self, query: str) -> Optional[list[dict]]:
        """Search via Arctic Shift archive."""
        after_ts = int((datetime.now() - timedelta(days=7)).timestamp())

        params = {
            "subreddit": self.subreddit_name,
            "q": query,
            "after": after_ts,
            "limit": self.max_posts,
            "sort": "score",
            "order": "desc",
        }

        try:
            response = self._session.get(
                self.ARCTIC_SHIFT_URL, params=params, timeout=15
            )

            if response.status_code != 200:
                logger.debug(f"Arctic Shift returned {response.status_code}")
                return None

            data = response.json()
            # Arctic Shift returns {"data": [...]}
            posts = data.get("data", [])
            if isinstance(posts, list):
                return posts
            return None

        except (requests.RequestException, ValueError) as e:
            logger.debug(f"Arctic Shift request failed: {e}")
            return None

    def _search_reddit_direct(self, query: str) -> Optional[list[dict]]:
        """Fallback: search Reddit directly (may be blocked on cloud IPs)."""
        url = self.REDDIT_URL.format(subreddit=self.subreddit_name)
        params = {
            "q": query,
            "restrict_sr": "on",
            "sort": "relevance",
            "t": "week",
            "limit": self.max_posts,
            "type": "link",
        }

        try:
            response = self._session.get(url, params=params, timeout=10, allow_redirects=True)

            if response.status_code != 200:
                logger.debug(f"Reddit direct returned {response.status_code}")
                return None

            content_type = response.headers.get("Content-Type", "")
            if "json" not in content_type and "javascript" not in content_type:
                logger.debug(f"Reddit returned non-JSON: {content_type}")
                return None

            data = response.json()
            children = data.get("data", {}).get("children", [])
            return [child.get("data", {}) for child in children]

        except (requests.RequestException, ValueError) as e:
            logger.debug(f"Reddit direct failed: {e}")
            return None

    def _symbol_in_text(self, symbol: str, text: str) -> bool:
        """Check if a stock symbol genuinely appears in text."""
        upper_text = text.upper()
        if f"${symbol.upper()}" in upper_text:
            return True
        pattern = r'\b' + re.escape(symbol.upper()) + r'\b'
        return bool(re.search(pattern, upper_text))

    def clear_cache(self):
        """Clear the entire cache."""
        self._cache.clear()
