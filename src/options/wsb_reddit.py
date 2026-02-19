"""Reddit WallStreetBets data fetcher via web scraping (no API key needed)."""

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
    Fetches WSB post data by scraping Reddit's public JSON endpoints.

    No API key or registration required. Uses endpoints like:
    https://www.reddit.com/r/wallstreetbets/search.json?q=NVDA&sort=relevance&t=week

    Caches results per symbol with configurable TTL.
    """

    # Use old.reddit.com - more reliable for JSON endpoints (no consent walls)
    SEARCH_URL = "https://old.reddit.com/r/{subreddit}/search.json"

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
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
        })

    @property
    def is_configured(self) -> bool:
        """Always True - no API keys needed for web scraping."""
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
                    body = post_data.get("selftext", "")
                    text = f"{title} {body}"

                    if not self._symbol_in_text(symbol, text):
                        continue

                    posts.append(WSBPost(
                        title=title,
                        body=body[:500],
                        score=post_data.get("score", 0),
                        upvote_ratio=post_data.get("upvote_ratio", 0.5),
                        num_comments=post_data.get("num_comments", 0),
                        created_utc=post_data.get("created_utc", 0),
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
        """Search WSB for posts matching a query via public JSON endpoint."""
        self._rate_limit()

        url = self.SEARCH_URL.format(subreddit=self.subreddit_name)
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

            # Log the actual URL hit (catches redirects)
            logger.debug(f"Reddit search URL: {response.url} -> status {response.status_code}")

            if response.status_code == 429:
                logger.warning("Reddit rate limited - backing off")
                time.sleep(10)
                return []

            if response.status_code != 200:
                logger.warning(
                    f"Reddit search returned {response.status_code} for '{query}'. "
                    f"URL: {response.url}"
                )
                return []

            # Check content type - Reddit sometimes returns HTML instead of JSON
            content_type = response.headers.get("Content-Type", "")
            if "json" not in content_type and "javascript" not in content_type:
                logger.warning(
                    f"Reddit returned non-JSON content for '{query}': {content_type}. "
                    f"Body preview: {response.text[:200]}"
                )
                return []

            data = response.json()
            children = data.get("data", {}).get("children", [])
            result = [child.get("data", {}) for child in children]
            logger.debug(f"Reddit search for '{query}': {len(result)} raw results")
            return result

        except requests.RequestException as e:
            logger.error(f"Reddit request failed for '{query}': {e}")
            return []
        except ValueError as e:
            logger.error(f"Failed to parse Reddit JSON for '{query}': {e}")
            return []

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
