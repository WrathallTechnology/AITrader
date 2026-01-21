"""
News data fetching and sentiment analysis using Alpaca News API.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a news article."""
    id: str
    headline: str
    summary: str
    author: str
    source: str
    url: str
    symbols: list[str]
    created_at: datetime
    updated_at: datetime


class AlpacaNewsClient:
    """
    Client for fetching news data from Alpaca.

    Alpaca provides news data from Benzinga with articles dating back to 2015.
    No API keys required for news data access.
    """

    def __init__(self):
        """Initialize news client (no keys required)."""
        self._client = NewsClient()
        self._cache: dict[str, list[NewsArticle]] = {}
        self._cache_time: dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)  # Cache news for 5 minutes

    def get_news(
        self,
        symbols: list[str] | str,
        limit: int = 10,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> list[NewsArticle]:
        """
        Get news articles for symbols.

        Args:
            symbols: Single symbol or list of symbols
            limit: Maximum number of articles to return (default 10)
            start: Start datetime for news search
            end: End datetime for news search
            use_cache: Whether to use cached results

        Returns:
            List of NewsArticle objects
        """
        # Normalize symbols to list
        if isinstance(symbols, str):
            symbols = [symbols]

        symbols_key = ",".join(sorted(symbols))

        # Check cache
        if use_cache and symbols_key in self._cache:
            cache_age = datetime.now() - self._cache_time.get(symbols_key, datetime.min)
            if cache_age < self._cache_ttl:
                logger.debug(f"Using cached news for {symbols_key}")
                return self._cache[symbols_key][:limit]

        try:
            # Build request
            request_params = {
                "symbols": symbols if len(symbols) > 1 else symbols[0],
                "limit": limit,
            }

            if start:
                request_params["start"] = start
            if end:
                request_params["end"] = end

            request = NewsRequest(**request_params)
            response = self._client.get_news(request)

            # Parse articles
            articles = []
            for item in response.news:
                try:
                    article = NewsArticle(
                        id=str(item.id),
                        headline=item.headline or "",
                        summary=item.summary or "",
                        author=item.author or "Unknown",
                        source=item.source or "Unknown",
                        url=item.url or "",
                        symbols=list(item.symbols) if item.symbols else [],
                        created_at=item.created_at,
                        updated_at=item.updated_at,
                    )
                    articles.append(article)
                except Exception as e:
                    logger.debug(f"Failed to parse news article: {e}")
                    continue

            # Update cache
            self._cache[symbols_key] = articles
            self._cache_time[symbols_key] = datetime.now()

            logger.info(f"Fetched {len(articles)} news articles for {symbols_key}")
            return articles

        except Exception as e:
            logger.error(f"Failed to fetch news for {symbols}: {e}")
            return []

    def get_recent_headlines(
        self,
        symbols: list[str] | str,
        hours: int = 24,
        limit: int = 20,
    ) -> list[str]:
        """
        Get recent headlines for symbols.

        Args:
            symbols: Single symbol or list of symbols
            hours: Number of hours to look back
            limit: Maximum number of headlines

        Returns:
            List of headline strings
        """
        start = datetime.now() - timedelta(hours=hours)
        articles = self.get_news(symbols, limit=limit, start=start)
        return [a.headline for a in articles]

    def get_news_summary(self, symbols: list[str] | str) -> dict:
        """
        Get a summary of recent news for symbols.

        Returns:
            Dict with article count, sources, and time range
        """
        articles = self.get_news(symbols, limit=50)

        if not articles:
            return {
                "count": 0,
                "symbols": symbols if isinstance(symbols, list) else [symbols],
                "sources": [],
                "oldest": None,
                "newest": None,
            }

        sources = list(set(a.source for a in articles))

        return {
            "count": len(articles),
            "symbols": symbols if isinstance(symbols, list) else [symbols],
            "sources": sources,
            "oldest": min(a.created_at for a in articles),
            "newest": max(a.created_at for a in articles),
            "headlines": [a.headline for a in articles[:5]],
        }


class NewsProvider:
    """
    News provider that integrates with the sentiment strategy.

    Replaces MockNewsProvider with real Alpaca news data.
    """

    def __init__(self):
        self.client = AlpacaNewsClient()

    def get_headlines(self, symbol: str, count: int = 10, hours: int = 48) -> list[str]:
        """
        Get recent headlines for a symbol.

        Args:
            symbol: Stock symbol
            count: Number of headlines to return
            hours: How many hours back to look

        Returns:
            List of headline strings
        """
        return self.client.get_recent_headlines(symbol, hours=hours, limit=count)

    def get_articles(self, symbol: str, count: int = 10) -> list[NewsArticle]:
        """Get full article objects for a symbol."""
        return self.client.get_news(symbol, limit=count)
